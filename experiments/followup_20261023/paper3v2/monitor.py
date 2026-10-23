"""One lightweight login-node process: status collection and nonblocking GitHub sync."""
import argparse
import base64
import fcntl
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import ssl
import sys
from .io import read, put, emit

class GitHub:
    def __init__(self, config):
        self.config = config
    def api(self, path, body=None, method=None):
        request = urllib.request.Request("https://api.github.com/" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": "Bearer " + self.config["token"], "Accept": "application/vnd.github+json",
                     "User-Agent": "paper3-followup-monitor", "Content-Type": "application/json"}, method=method)
        # The school login node previously needed this installed CA bundle for
        # GitHub TLS. Locate it without importing the heavy cloud SDK.
        context = getattr(self, "_ssl_context", None)
        if context is None:
            candidates = [Path(p) / "botocore/cacert.pem" for p in sys.path if p]
            candidates += [Path(p) / "certifi/cacert.pem" for p in sys.path if p]
            bundle = next((str(p) for p in candidates if p.is_file()), None)
            context = ssl.create_default_context()
            if bundle:
                context.load_verify_locations(cafile=bundle)
            self._ssl_context = context
        with urllib.request.urlopen(request, timeout=15, context=context) as response:
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
        status_path = self.config.get("status_path", "monitoring/paper3-followup/status.json")
        path = f"repos/{repo}/contents/{status_path}"
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
        except (OSError, ValueError, TypeError, AttributeError):
            row["status"] = "TEMPORARILY_UNREADABLE"
        tasks.append(row)
    return {"experiment": "paper3-followup-20261023", "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "max_parallel_gpus": plan["gpus"], "cpus_per_gpu": plan["cpus"], "tasks": tasks,
            "complete_tasks": sum(t["status"] == "COMPLETE" for t in tasks), "total_tasks": len(tasks)}

def best_effort_put(path, value):
    try:
        put(path, value)
        return True
    except (OSError, ValueError) as e:
        emit("MONITOR_LOCAL_WRITE_RETRY", file=Path(path).name, error_type=type(e).__name__)
        return False


REPORT_TABLES = ("comparisons", "budget_curves", "budget_comparisons", "reference_comparisons",
                 "sensitivities", "diagnostics", "diagnostic_comparisons")
SUMMARY_METRICS = ("ba", "bce", "f0_ba", "negative_flip_rate", "corrected_error_rate", "brier_balanced",
                   "nll_balanced", "ece_balanced_10bins", "f0_brier_balanced", "f0_nll_balanced",
                   "f0_ece_balanced_10bins", "run_ba_p10", "run_ba_min", "calibration_seconds",
                   "fallback_fraction", "temporal_eligible_fraction")
SUMMARY_FIELDS = set(SUMMARY_METRICS) | {
    "variant", "candidate", "baseline", "setting", "scenario", "arm", "budget", "seed",
    "n_participants", "sessions_per_seed", "completed_seed_count", "all_seeds_complete",
    "delta_ba_pp", "p", "p_method", "p_holm", "positive_evidence", "baseline_ba", "candidate_ba",
    "relative_error_reduction_pct", "participant_negative_transfer_fraction", "inference", "interpretation",
    "test_assumption", "synthetic_stress", "delta", "unit", "temporal_eligible_session_fraction",
    "fallback_session_fraction", "run_ba_p10_n_participants", "run_ba_p10_n_session_seed_rows",
    "run_ba_min_n_participants", "run_ba_min_n_session_seed_rows",
}


def summary_row(row):
    """Publish aggregate statistics only, never trial arrays or participant rows."""
    result = {k: (v[:1024] if isinstance(v, str) else v) for k, v in row.items()
              if k in SUMMARY_FIELDS and (v is None or isinstance(v, (str, int, float, bool)))}
    for key in ("ci95", "ci95_pp", "completed_seeds", "pending_seeds"):
        value = row.get(key)
        if value is None and key in row:
            result[key] = None
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
            result[key] = list(value[:64])
    if isinstance(row.get("per_seed"), list):
        result["per_seed"] = [summary_row(v) for v in row["per_seed"][:64] if isinstance(v, dict)]
    if isinstance(row.get("secondary_tradeoffs"), dict):
        result["secondary_tradeoffs"] = {k: summary_row(v) for k, v in row["secondary_tradeoffs"].items()
                                         if k in SUMMARY_METRICS and isinstance(v, dict)}
    if isinstance(row.get("change_from_own_clean"), dict):
        result["change_from_own_clean"] = summary_row(row["change_from_own_clean"])
    return result


def report_summary(report):
    result = {"report_status": report.get("status"), "report_complete_tasks": report.get("complete_tasks"),
              "report_total_tasks": report.get("total_tasks")}
    for key in ("core_complete", "planned_comparison_count", "interpretation", "confidence_interval_scope",
                "secondary_metrics_note"):
        if key in report:
            result[key] = report[key]
    result["report_table_rows"] = {}
    for key in REPORT_TABLES:
        rows = report.get(key, [])
        # The planned tables are far smaller; this also bounds future additions.
        selected = [summary_row(row) for row in rows[:200] if isinstance(row, dict)]
        result[key] = selected
        result["report_table_rows"][key] = {"total": len(rows), "published": len(selected)}
    return result


def cycle(plan, credentials, github=None, ready=False, report_cache=None):
    """One independent observation/export cycle; return whether all work is done."""
    if report_cache is None:
        report_cache = {}
    output = Path(plan["output"])
    snapshot = collect(plan)
    all_complete = snapshot["complete_tasks"] == snapshot["total_tasks"]
    count = snapshot["complete_tasks"]
    if (report_cache.get("complete_tasks") != count
            or (all_complete and report_cache.get("summary", {}).get("report_status") != "COMPLETE")):
        try:
            from .report import aggregate
            report = aggregate(plan)
            summary = report_summary(report)
            report_cache.update(complete_tasks=report.get("complete_tasks", count), summary=summary)
        except Exception as e:
            snapshot["report_error_type"] = type(e).__name__
            snapshot["report_error"] = str(e)[:1000]
            emit("REPORT_RETRY_PENDING", error_type=type(e).__name__)
    if "summary" in report_cache:
        snapshot.update(report_cache["summary"])
        snapshot["report_stale"] = report_cache.get("complete_tasks") != count
    report_ok = (report_cache.get("complete_tasks") == count
                 and report_cache.get("summary", {}).get("report_status") == "COMPLETE")
    best_effort_put(output / "status.json", snapshot)
    synced = False
    try:
        if github is None:
            github = GitHub(read(credentials))
        if not ready:
            github.ensure_branch(); ready = True
        github.publish(snapshot)
        synced = True
        best_effort_put(output / "github_sync.json", {"status": "SYNCED", "updated_utc": snapshot["updated_utc"]})
        emit("GITHUB_SYNCED", completed=snapshot["complete_tasks"])
    except Exception as e:
        # Never serialize request headers or credential-bearing config objects.
        best_effort_put(output / "github_sync.json", {"status": "RETRY_PENDING", "error_type": type(e).__name__,
                                                      "http_status": getattr(e, "code", None)})
        emit("GITHUB_RETRY_PENDING", error_type=type(e).__name__)
    # A successfully exported report error is not a successfully generated report.
    return github, ready, all_complete and report_ok and synced

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True); parser.add_argument("--credentials", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    lock = None
    while True:
        try:
            plan = read(args.plan); output = Path(plan["output"])
            lock = (output / "monitor.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close(); return
            break
        except (OSError, ValueError, KeyError, TypeError) as e:
            if lock is not None:
                lock.close(); lock = None
            emit("MONITOR_STARTUP_RETRY", error_type=type(e).__name__)
            if args.once:
                return
            time.sleep(30)
    github, ready, report_cache = None, False, {}
    while True:
        github, ready, finished = cycle(plan, args.credentials, github, ready, report_cache)
        if args.once:
            return
        # Continue after the final fit if upload failed; retry until the final snapshot is synced.
        if finished:
            return
        time.sleep(180)

if __name__ == "__main__":
    main()
