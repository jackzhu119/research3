"""One lightweight login-node process: status collection and nonblocking GitHub sync."""
import argparse
import base64
import fcntl
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from .io import read, put, emit

class GitHub:
    def __init__(self, config):
        self.config = config
    def api(self, path, body=None, method=None):
        request = urllib.request.Request("https://api.github.com/" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": "Bearer " + self.config["token"], "Accept": "application/vnd.github+json",
                     "User-Agent": "paper3-followup-monitor", "Content-Type": "application/json"}, method=method)
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)
    def ensure_branch(self):
        repo, branch = self.config["repo"], self.config["branch"]
        try:
            self.api(f"repos/{repo}/git/ref/heads/{branch}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            default = self.api(f"repos/{repo}")["default_branch"]
            head = self.api(f"repos/{repo}/git/ref/heads/{default}")["object"]["sha"]
            try:
                self.api(f"repos/{repo}/git/refs", {"ref": "refs/heads/" + branch, "sha": head})
            except urllib.error.HTTPError as conflict:
                if conflict.code != 422:
                    raise
                self.api(f"repos/{repo}/git/ref/heads/{branch}")
    def publish(self, snapshot):
        repo, branch = self.config["repo"], self.config["branch"]
        path = f"repos/{repo}/contents/monitoring/paper3-followup/status.json"
        for attempt in range(2):
            try:
                existing = self.api(path + "?ref=" + branch)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                existing = None
            value = {"message": "Update Paper3 follow-up progress", "branch": branch,
                     "content": base64.b64encode(json.dumps(snapshot, ensure_ascii=False, indent=2).encode()).decode()}
            if existing:
                value["sha"] = existing["sha"]
            try:
                return self.api(path, value, "PUT")
            except urllib.error.HTTPError as e:
                if e.code not in (409, 422) or attempt:
                    raise

def collect(plan):
    tasks = []
    for task in plan["tasks"]:
        folder = Path(plan["output"]) / task["key"]
        row = {"task": task["key"], "status": "QUEUED_OR_NOT_STARTED"}
        try:
            if (folder / "progress.json").exists():
                raw = read(folder / "progress.json")
                row.update({k: raw[k] for k in ("status", "stage", "outer_update", "total_updates", "session", "sessions_total", "budget", "updated_utc", "objective", "gate_mean") if k in raw})
                row["progress_age_seconds"] = round(time.time() - (folder / "progress.json").stat().st_mtime)
            if (folder / "result.json").exists() and read(folder / "result.json").get("status") == "COMPLETE":
                row["status"] = "COMPLETE"
        except (OSError, ValueError):
            row["status"] = "TEMPORARILY_UNREADABLE"
        tasks.append(row)
    return {"experiment": "paper3-followup-20261023", "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "max_parallel_gpus": plan["gpus"], "cpus_per_gpu": plan["cpus"], "tasks": tasks,
            "complete_tasks": sum(t["status"] == "COMPLETE" for t in tasks), "total_tasks": len(tasks)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True); parser.add_argument("--credentials", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan); output = Path(plan["output"])
    lock = (output / "monitor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    github, ready = None, False
    while True:
        snapshot = collect(plan)
        put(output / "status.json", snapshot)
        # Reporting failure never controls or signals training processes.
        if snapshot["complete_tasks"] == snapshot["total_tasks"]:
            try:
                from .report import aggregate
                report = aggregate(plan)
                snapshot["comparisons"] = report["comparisons"]
                snapshot["budget_curves"] = report.get("budget_curves", [])
            except Exception as e:
                snapshot["report_error"] = str(e)
        try:
            if github is None:
                github = GitHub(read(args.credentials))
            if not ready:
                github.ensure_branch(); ready = True
            github.publish(snapshot)
            put(output / "github_sync.json", {"status": "SYNCED", "updated_utc": snapshot["updated_utc"]})
            emit("GITHUB_SYNCED", completed=snapshot["complete_tasks"])
        except Exception as e:
            # Never serialize request headers or credential-bearing config objects.
            put(output / "github_sync.json", {"status": "RETRY_PENDING", "error_type": type(e).__name__,
                                               "http_status": getattr(e, "code", None)})
            emit("GITHUB_RETRY_PENDING", error_type=type(e).__name__)
        if args.once:
            return
        # Continue after the final fit if upload failed; retry until the final snapshot is synced.
        if snapshot["complete_tasks"] == snapshot["total_tasks"] and read(output / "github_sync.json")["status"] == "SYNCED":
            return
        time.sleep(180)

if __name__ == "__main__":
    main()
