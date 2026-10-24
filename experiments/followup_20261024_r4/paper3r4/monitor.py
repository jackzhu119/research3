"""Independent local progress/reporting and best-effort GitHub synchronization."""
import argparse
import base64
import fcntl
import json
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.request

from .io import read, put, emit

REPOSITORY = "jackzhu119/research3"
BRANCH = "paper3-r4-progress"
STATUS_PATH = "monitoring/paper3-round4/status.json"
WEIBO_STATUS_PATH = "monitoring/paper3-round4/weibo-status.json"


def status_path_for_plan(plan):
    return (WEIBO_STATUS_PATH if "weibo" in str(plan.get("version", "")).lower()
            or plan.get("external_manifest") else STATUS_PATH)


class GitHub:
    def __init__(self, config, *, status_path=STATUS_PATH):
        if status_path not in (STATUS_PATH, WEIBO_STATUS_PATH):
            raise ValueError("Unsupported round-four monitoring status path")
        self.config = {**config, "repo": REPOSITORY, "branch": BRANCH, "status_path": status_path}
        self.context = ssl.create_default_context()
        for folder in sys.path:
            if not folder:
                continue
            for name in ("botocore/cacert.pem", "certifi/cacert.pem"):
                path = Path(folder) / name
                if path.is_file():
                    self.context.load_verify_locations(cafile=str(path))

    def api(self, path, body=None, method=None):
        request = urllib.request.Request("https://api.github.com/" + path,
            data=json.dumps(body, allow_nan=False).encode() if body is not None else None,
            headers={"Authorization": "Bearer " + self.config["token"],
                     "User-Agent": "paper3-round4-monitor", "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json"}, method=method)
        with urllib.request.urlopen(request, timeout=20, context=self.context) as response:
            return json.load(response)

    def ensure_branch(self):
        prefix = "repos/" + REPOSITORY
        try:
            self.api(prefix + "/git/ref/heads/" + BRANCH)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            default = self.api(prefix)["default_branch"]
            head = self.api(prefix + "/git/ref/heads/" + default)["object"]["sha"]
            try:
                self.api(prefix + "/git/refs", {"ref": "refs/heads/" + BRANCH, "sha": head})
            except urllib.error.HTTPError as conflict:
                if conflict.code != 422:
                    raise
                self.api(prefix + "/git/ref/heads/" + BRANCH)

    def publish(self, snapshot):
        path = "repos/" + REPOSITORY + "/contents/" + self.config["status_path"]
        for attempt in range(3):
            try:
                prior = self.api(path + "?ref=" + BRANCH)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                prior = None
            payload = {"message": "Update Paper3 round-four progress and aggregate results", "branch": BRANCH,
                       "content": base64.b64encode(json.dumps(snapshot, ensure_ascii=False, indent=2,
                                                               allow_nan=False).encode()).decode()}
            if prior:
                payload["sha"] = prior["sha"]
            try:
                return self.api(path, payload, "PUT")
            except urllib.error.HTTPError as error:
                if error.code not in (409, 422) or attempt == 2:
                    raise


def aggregate_only(value):
    """Keep aggregate tables; participant deltas and trial arrays stay local."""
    private = {"participant_deltas_ba", "participant_seed_counts", "participant_id", "session_id",
               "participants", "expected_sessions", "participant_ids", "session_ids",
               "event_uids", "labels", "predictions", "probabilities", "f0_predictions", "f0_probabilities"}
    if isinstance(value, dict):
        return {key: aggregate_only(child) for key, child in value.items() if key not in private}
    if isinstance(value, (tuple, list)):
        return [aggregate_only(child) for child in value]
    return value


def collect(plan):
    output, tasks = Path(plan["output"]), []
    state = read(output / "runtime/state.json") if (output / "runtime/state.json").is_file() else {}
    allocation_path = output / "runtime/allocation_state.json"
    allocation = read(allocation_path) if allocation_path.is_file() else {}
    public_allocation_path = output / "allocation_progress.json"
    public_allocation = read(public_allocation_path) if public_allocation_path.is_file() else {}
    operational = allocation.get("tasks", state.get("tasks", {}))
    attempts = state.get("attempts", [])
    latest = attempts[-1] if attempts else {}
    for index, task in enumerate(plan["tasks"]):
        folder = output / task["key"]
        operation = operational.get(str(index), {})
        row = {"task": task["key"], "family": task.get("family"), "seed": task.get("seed"),
               "status": operation.get("phase", "NOT_STARTED")}
        row.update({key: operation[key] for key in ("slot", "attempt", "error_type") if key in operation})
        if latest:
            row.update({key: latest[key] for key in ("job_id", "slurm_state", "queue_reason") if key in latest})
        path = folder / "progress.json"
        try:
            if path.is_file():
                current = read(path)
                row.update({key: current[key] for key in ("status", "stage", "outer_update", "total_updates",
                    "session", "sessions_total", "budget", "scenario", "updated_utc") if key in current})
                row["progress_age_seconds"] = round(time.time() - path.stat().st_mtime, 1)
            result_path = folder / "result.json"
            if result_path.is_file() and read(result_path).get("status") == "COMPLETE":
                row["status"] = "COMPLETE"
        except (OSError, ValueError, TypeError):
            row["status"] = "TEMPORARILY_UNREADABLE"
        if row["status"] != "COMPLETE" and latest.get("slurm_state") == "PENDING":
            row["last_worker_status"] = row["status"]
            row["status"] = "QUEUED_ALLOCATION"
        elif (row["status"] in ("NOT_STARTED", "NEW", "ACTIVE")
              and latest.get("slurm_state") in ("RUNNING", "CONFIGURING")):
            row["status"] = "WAITING_GPU_SLOT" if allocation else "STARTING_ALLOCATION"
        tasks.append(row)
    launch = read(output / "runtime/launch.json") if (output / "runtime/launch.json").is_file() else {}
    resources = {key: launch.get(key) for key in ("partition", "gres", "cpus", "memory", "slots")}
    resources.update(nodes=1, slurm_jobs=1, cpus_per_gpu=launch.get("cpus"), memory_per_gpu=launch.get("memory"))
    if launch.get("cpus") is not None:
        resources["cpus_per_job"] = int(launch["cpus"]) * int(launch.get("slots", 3))
    if launch.get("memory"):
        from .submit import allocation_memory
        resources["memory_per_job"] = allocation_memory(launch["memory"], launch.get("slots", 3))
    return {"experiment": "paper3-round4", "recipe_sha256": plan["recipe_sha256"],
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "complete_tasks": sum(task["status"] == "COMPLETE" for task in tasks), "total_tasks": len(tasks),
            "tasks": tasks, "requested_resources": resources,
            "allocation_mode": "SINGLE_NODE_THREE_GPUS",
            "allocation": {key: latest[key] for key in ("job_id", "slurm_state", "queue_reason") if key in latest},
            "gpu_slots": aggregate_only(public_allocation.get("slots", {})),
            "observed_allocation_resources": {key: public_allocation[key] for key in
                ("requested_slots", "requested_cpus_per_slot", "observed_cpus_per_slot", "allocation_status")
                if key in public_allocation},
            "resource_scope": "submission configuration, not live allocated GPU utilization",
            "controller_status": state.get("controller_status", state.get("status")),
            "scientific_completion_claimed": False}


def cycle(plan, credentials, client=None, branch_ready=False, cache=None):
    cache = cache if cache is not None else {}
    snapshot = collect(plan)
    if (cache.get("complete_tasks") != snapshot["complete_tasks"] or "report" not in cache
            or cache.get("report", {}).get("status") == "ERROR"):
        try:
            from .report import build_report
            report = build_report(plan)
            cache.update(report=report, complete_tasks=snapshot["complete_tasks"])
        except Exception as error:
            emit("REPORT_RETRY", error_type=type(error).__name__)
    report = cache.get("report", {})
    snapshot.update(report_status=report.get("status", "NOT_READY"),
        primary_family_complete=report.get("primary_family_complete", False),
        exploratory_internal_replication=True, fresh_confirmatory=False,
        interpretation=plan.get("interpretation"),
        report_table_rows={name: {"total": len(rows), "published": min(400, len(rows))}
                           for name, rows in report.get("tables", {}).items()},
        tables={name: aggregate_only(rows[:400]) for name, rows in report.get("tables", {}).items()})
    try:
        put(Path(plan["output"]) / "status.json", snapshot)
    except OSError as error:
        emit("LOCAL_STATUS_RETRY", error_type=type(error).__name__)
    synced = False
    try:
        if client is None:
            if not credentials or not Path(credentials).is_file():
                emit("GITHUB_CONFIG_NOT_AVAILABLE_LOCAL_PROGRESS_CONTINUES")
                return client, branch_ready, False
            client = GitHub(read(credentials), status_path=status_path_for_plan(plan))
        if not branch_ready:
            client.ensure_branch(); branch_ready = True
        client.publish(snapshot)
        synced = True
        emit("GITHUB_SYNCED", branch=BRANCH, path=status_path_for_plan(plan), complete_tasks=snapshot["complete_tasks"])
    except Exception as error:
        emit("GITHUB_SYNC_RETRY", error_type=type(error).__name__, training_controlled=False)
    finished = (snapshot["complete_tasks"] == snapshot["total_tasks"] and report.get("status") == "COMPLETE" and synced)
    return client, branch_ready, finished


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--credentials")
    parser.add_argument("--interval", type=float, default=180)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    plan = read(args.plan)
    output = Path(plan["output"]); output.mkdir(parents=True, exist_ok=True)
    with (output / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit("MONITOR_ALREADY_RUNNING"); return 0
        client, ready, cache = None, False, {}
        while True:
            try:
                client, ready, finished = cycle(plan, args.credentials, client, ready, cache)
            except Exception as error:
                emit("MONITOR_RETRY", error_type=type(error).__name__)
                finished = False
            if args.once or finished:
                return 0
            time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
