"""Durable, conservative rolling Slurm controller for the isolated Round 3 run.

Only this run's exact comment marker is owned. No cancellation or in-place
Slurm mutation is performed. Unknown submission outcomes reserve a slot until
queue/accounting evidence resolves them; they are never blindly resubmitted.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}
TRANSIENT_STATES = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REVOKED"}
DEFAULT_LAUNCH = {"python": sys.executable, "package": str(Path(__file__).resolve().parent.parent),
                  "account": "zhuziyuan", "partition": "hpc_gpu", "gres": "gpu:1", "cpus": 12,
                  "memory": "64G", "max_inflight": 3, "time_limit": "5-03:00:00", "poll_seconds": 30,
                  "max_startup_retries": 2, "max_resumes": 100, "submit_reject_limit": 0,
                  "accounting_grace_seconds": 300}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.%s" % os.getpid())
    with temp.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return default


def job_marker(plan):
    payload = json.dumps([plan.get("recipe_sha256"), str(Path(plan["output"]).resolve())], separators=(",", ":"))
    return "paper3r3-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _state(value):
    return value.split()[0].rstrip("+").upper() if value.strip() else "UNKNOWN"


def parse_queue(text, marker):
    rows = []
    for raw in text.splitlines():
        fields = raw.strip().split("|")
        if len(fields) < 3 or fields[1].strip() != marker:
            continue
        match = re.fullmatch(r"(\d+)_(\d+)", fields[0].strip())
        if not match:
            raise ValueError("Owned queue row is not an expanded single-element array")
        rows.append({"job_id": match[1] + "_" + match[2], "array_id": match[1],
                     "index": int(match[2]), "state": _state(fields[2]),
                     "name": fields[3].strip() if len(fields) > 3 else ""})
    return rows


def parse_accounting(text, marker, owned_names=()):
    rows = []
    for raw in text.splitlines():
        fields = raw.strip().split("|")
        if len(fields) < 5:
            continue
        comment, name = fields[3].strip(), fields[4].strip()
        # Some sites omit Comment from accounting unless job_comment storage
        # is enabled. A pre-existing durable intent with this unique run name
        # is sufficient ownership evidence, even after a lost acknowledgement.
        if comment != marker and not (not comment and name in owned_names):
            continue
        match = re.fullmatch(r"(\d+)_(\d+)", fields[0].strip())
        # .batch/.extern records and non-array parents are not task outcomes.
        if not match:
            continue
        try:
            exit_code, exit_signal = [int(v) for v in fields[2].split(":", 1)]
        except (ValueError, TypeError):
            exit_code, exit_signal = None, None
        rows.append({"job_id": match[1] + "_" + match[2], "array_id": match[1],
                     "index": int(match[2]), "state": _state(fields[1]),
                     "exit_code": exit_code, "exit_signal": exit_signal, "name": fields[4].strip()})
    return rows


@contextmanager
def exclusive_lock(path):
    handle = Path(path).open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle
    finally:
        handle.close()


class RollingController:
    def __init__(self, plan_path, launch=None, runner=None, clock=None):
        self.plan_path = Path(plan_path).resolve()
        self.plan = read_json(self.plan_path)
        if not isinstance(self.plan, dict) or not isinstance(self.plan.get("tasks"), list):
            raise ValueError("Invalid task plan")
        self.output = Path(self.plan["output"]).resolve()
        self.runtime = self.output / "runtime"
        (self.runtime / "receipts").mkdir(parents=True, exist_ok=True)
        (self.runtime / "outcomes").mkdir(exist_ok=True)
        self.launch = {**DEFAULT_LAUNCH, **(launch or {})}
        if not 1 <= int(self.launch["max_inflight"]) <= 3:
            raise ValueError("max_inflight must be between 1 and 3")
        self.runner = runner or subprocess.run
        self.clock = clock or time.time
        self.marker = job_marker(self.plan)
        self.state = read_json(self.runtime / "state.json", {})
        if self.state and self.state.get("marker") != self.marker:
            raise ValueError("Controller state belongs to a different immutable plan")
        self.state.setdefault("version", 1)
        self.state["marker"] = self.marker
        self.state.setdefault("tasks", {})
        self.state.setdefault("submit_rejections", 0)
        self.state.setdefault("next_submit_at", 0)
        for index, task in enumerate(self.plan["tasks"]):
            if not isinstance(task, dict) or not task.get("key"):
                raise ValueError("Each task requires a unique key")
            key = task["key"]
            if Path(key).name != key or key in (".", ".."):
                raise ValueError("Task key must be a single directory name")
            requires = task.get("requires", [])
            if not isinstance(requires, list) or any(not isinstance(key, str) or Path(key).name != key or key in (".", "..") for key in requires):
                raise ValueError("Task requires must be a list of task directory keys")
            self.state["tasks"].setdefault(str(index), {"phase": "NEW", "attempts": []})
        if len({t["key"] for t in self.plan["tasks"]}) != len(self.plan["tasks"]):
            raise ValueError("Task keys must be unique")
        # Recover after a process death between receipt fsync and state fsync.
        for path in sorted((self.runtime / "receipts").glob("*.json")):
            receipt = read_json(path)
            if not isinstance(receipt, dict) or receipt.get("marker") != self.marker:
                continue
            task = self.state["tasks"].get(str(receipt.get("index")))
            if task is None:
                continue
            existing = next((a for a in task["attempts"] if a["sequence"] == receipt["sequence"]), None)
            if existing is None:
                task["attempts"].append(receipt)
            else:
                existing.update(receipt)
            task["attempts"].sort(key=lambda a: a["sequence"])

    def _save(self):
        self.state["updated_at"] = self.clock()
        atomic_json(self.runtime / "state.json", self.state)

    def _receipt(self, attempt):
        atomic_json(self.runtime / "receipts" / ("i%05d-a%03d.json" % (attempt["index"], attempt["sequence"])), attempt)

    def _new_attempt(self, index, **extra):
        attempts = self.state["tasks"][str(index)]["attempts"]
        seq = max([a["sequence"] for a in attempts] or [0]) + 1
        value = {"index": index, "sequence": seq, "marker": self.marker, "intent_at": self.clock(),
                 "status": "INTENT", "job_id": None, "name": "%s-%s-a%s" % (self.marker, index, seq), **extra}
        attempts.append(value)
        self._receipt(value)
        self._save()
        return value

    def _run(self, command, timeout=30):
        return self.runner(command, capture_output=True, text=True, timeout=timeout)

    def _query(self):
        queue = self._run(["squeue", "--array", "--noheader", "--user", os.environ.get("USER", "zhuziyuan"),
                           "--format=%i|%k|%T|%j"])
        if queue.returncode:
            raise RuntimeError("SQUEUE_FAILED")
        # Query account history from plan preparation, not an assumed permanent job ID.
        created = self.plan.get("prepared_at", self.state.get("created_at", self.clock()))
        start = datetime.fromtimestamp(float(created) - 86400, timezone.utc).strftime("%Y-%m-%d")
        accounting = self._run(["sacct", "--noheader", "--parsable2", "--allocations", "--user",
                               os.environ.get("USER", "zhuziyuan"), "--starttime", start,
                               "--format=JobID%80,State%40,ExitCode,Comment%64,JobName%100"])
        if accounting.returncode:
            raise RuntimeError("SACCT_FAILED")
        owned_names = {attempt["name"] for task in self.state["tasks"].values() for attempt in task["attempts"]}
        return parse_queue(queue.stdout, self.marker), parse_accounting(accounting.stdout, self.marker, owned_names)

    def _complete(self, index):
        return self._key_complete(self.plan["tasks"][index]["key"])

    def _key_complete(self, key):
        value = read_json(self.output / key / "result.json", {})
        return isinstance(value, dict) and value.get("status") == "COMPLETE"

    def _stopped(self, index):
        return (self.runtime / ("task-%s.stop" % index)).exists()

    def _adopt(self, rows):
        for row in rows:
            task = self.state["tasks"].get(str(row["index"]))
            if task is None:
                raise ValueError("Owned job references an index outside this plan")
            attempt = next((a for a in task["attempts"] if a.get("job_id") == row["job_id"]), None)
            if attempt is None:
                candidates = [a for a in task["attempts"] if not a.get("job_id")
                              and a["status"] in ("INTENT", "UNCERTAIN")
                              and (not row["name"] or row["name"] == a["name"])]
                attempt = candidates[-1] if candidates else None
            if attempt is None:
                # A completed historical receipt may predate local state recovery.
                attempt = self._new_attempt(row["index"], origin="RECOVERED_FROM_SLURM", name=row["name"])
            attempt.update(job_id=row["job_id"], status="ACKNOWLEDGED", observed_at=self.clock())
            self._receipt(attempt)

    def _finish_attempt(self, index, attempt, row):
        state, code = row["state"], row.get("exit_code")
        local = read_json(self.runtime / "outcomes" / (attempt["job_id"] + ".json"), {})
        attempt.update(slurm_state=state, exit_code=code, exit_signal=row.get("exit_signal"), finished_at=self.clock())
        if state == "CANCELLED" or self._stopped(index) or local.get("stop_reason") == "terminated":
            outcome = "CANCELLED"
        elif self._complete(index):
            outcome = "COMPLETE"
        elif code == 75 or (local.get("exit_code") == 75 and not row.get("exit_signal")):
            outcome = "CHECKPOINT"
        elif state in TRANSIENT_STATES:
            outcome = "STARTUP_FAILURE"
        elif not (self.output / self.plan["tasks"][index]["key"] / "progress.json").exists():
            outcome = "STARTUP_FAILURE"
            attempt["classification_evidence"] = "No worker progress file; early failure is a bounded retry heuristic"
        else:
            outcome = "FAILED"
        attempt.update(status="FINISHED", outcome=outcome)
        self._receipt(attempt)

    def _phase(self, index):
        task = self.state["tasks"][str(index)]
        if self._complete(index):
            return "COMPLETE"
        if self._stopped(index):
            return "CANCELLED"
        attempts = task["attempts"]
        unresolved = [a for a in attempts if a["status"] not in ("FINISHED", "REJECTED")]
        if unresolved:
            return "UNCERTAIN" if any(a["status"] in ("INTENT", "UNCERTAIN") for a in unresolved) else "ACTIVE"
        outcomes = [a.get("outcome") for a in attempts if a["status"] == "FINISHED"]
        if "CANCELLED" in outcomes:
            return "CANCELLED"
        if "FAILED" in outcomes:
            return "FAILED"
        if outcomes.count("STARTUP_FAILURE") > int(self.launch["max_startup_retries"]):
            return "FAILED"
        if outcomes.count("CHECKPOINT") > int(self.launch["max_resumes"]):
            return "FAILED"
        missing = [key for key in self.plan["tasks"][index].get("requires", []) if not self._key_complete(key)]
        task["waiting_inputs"] = missing
        if missing:
            return "WAITING_INPUT"
        if attempts:
            return "RETRY"
        return "NEW"

    def _submit(self, index):
        from .submit import build_sbatch_command
        attempt = self._new_attempt(index)
        command = build_sbatch_command(self.plan, self.launch["python"], index,
                                       {**self.launch, "attempt": attempt["sequence"]})
        attempt["command"] = command
        self._receipt(attempt)
        try:
            result = self._run(command, timeout=45)
        except FileNotFoundError:
            attempt.update(status="REJECTED", error_type="SbatchExecutableMissing")
            self.state["submission_blocked"] = "sbatch executable is unavailable in this environment"
            self._receipt(attempt)
            return False
        except (subprocess.TimeoutExpired, OSError) as exc:
            # An interrupted client may already have delivered the request.
            attempt.update(status="UNCERTAIN", error_type=type(exc).__name__)
            self._receipt(attempt)
            return False
        if result.returncode == 0:
            match = re.fullmatch(r"(\d+)(?:;[^\s;]+)?", result.stdout.strip())
            if match:
                attempt.update(status="ACKNOWLEDGED", job_id=match[1] + "_" + str(index), acknowledged_at=self.clock())
                self.state["submit_rejections"] = 0
            else:
                attempt.update(status="UNCERTAIN", error_type="UnclearAcknowledgement")
        else:
            stderr = result.stderr or ""
            # Only explicit scheduler rejection is safe to retry. Other client
            # failures can mean that the acknowledgement was lost in transit.
            explicit = bool(re.search(r"QOSMaxSubmitJobPerUserLimit|Job violates accounting/QOS policy|Invalid (?:generic resource|partition|account|qos)|Requested (?:node configuration is not available|time limit is invalid)", stderr, re.I))
            if explicit:
                attempt.update(status="REJECTED", returncode=result.returncode,
                               error_type="QoSOrCapacityRejection" if re.search(r"QOS|limit|resources", stderr, re.I) else "SchedulerRejection")
                count = self.state["submit_rejections"] + 1
                self.state["submit_rejections"] = count
                self.state["next_submit_at"] = self.clock() + min(300, 30 * 2 ** min(count - 1, 4))
                limit = int(self.launch["submit_reject_limit"])
                if not re.search(r"QOS|MaxSubmitJob|accounting/QOS policy", stderr, re.I) or (limit > 0 and count >= limit):
                    self.state["submission_blocked"] = "Scheduler rejection requires operator review; automatic retry limit reached or configuration invalid"
            else:
                attempt.update(status="UNCERTAIN", returncode=result.returncode, error_type="UncertainSbatchFailure")
        self._receipt(attempt)
        return attempt["status"] == "ACKNOWLEDGED"

    def step(self):
        self.state.setdefault("created_at", self.clock())
        if (self.runtime / "STOP").exists() or (self.output / "STOP").exists():
            self.state["controller_status"] = "STOPPED"
            self._save()
            return self.summary()
        try:
            queue, accounting = self._query()
            self._adopt(accounting)
            self._adopt(queue)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            self.state.update(controller_status="OBSERVATION_BLOCKED", observation_error=type(exc).__name__ + ":" + str(exc)[:100])
            self._save()
            return self.summary()
        active = {row["job_id"] for row in queue}
        account = {row["job_id"]: row for row in accounting}
        for index in range(len(self.plan["tasks"])):
            task = self.state["tasks"][str(index)]
            for attempt in task["attempts"]:
                if attempt["status"] in ("FINISHED", "REJECTED"):
                    continue
                job_id = attempt.get("job_id")
                if job_id in active:
                    attempt["status"] = "ACKNOWLEDGED"
                elif self._complete(index):
                    attempt.update(status="FINISHED", outcome="COMPLETE", finished_at=self.clock())
                    self._receipt(attempt)
                elif job_id in account and account[job_id]["state"] not in ACTIVE_STATES:
                    self._finish_attempt(index, attempt, account[job_id])
                elif self.clock() - attempt["intent_at"] > int(self.launch["accounting_grace_seconds"]):
                    attempt.update(status="UNCERTAIN", error_type="AwaitingQueueOrAccountingEvidence")
                    self._receipt(attempt)
            task["phase"] = self._phase(index)
        # A COMPLETE result does not free a GPU until its job disappears from
        # squeue. Unknown requests also reserve capacity to prevent duplicates.
        unknown = sum(1 for task in self.state["tasks"].values() for attempt in task["attempts"]
                      if attempt["status"] not in ("FINISHED", "REJECTED") and attempt.get("job_id") not in active)
        in_flight = len(active) + unknown
        self.state["inflight_jobs"] = in_flight
        self.state["controller_status"] = "SUBMISSION_BLOCKED" if self.state.get("submission_blocked") else (
            "WAITING_FOR_CAPACITY" if self.clock() < self.state["next_submit_at"] else "RUNNING")
        if not self.state.get("submission_blocked") and self.clock() >= self.state["next_submit_at"]:
            for index in range(len(self.plan["tasks"])):
                if (self.runtime / "STOP").exists() or (self.output / "STOP").exists():
                    self.state["controller_status"] = "STOPPED"
                    break
                if in_flight >= int(self.launch["max_inflight"]):
                    break
                if self.state["tasks"][str(index)]["phase"] not in ("NEW", "RETRY"):
                    continue
                if self._stopped(index):
                    self.state["tasks"][str(index)]["phase"] = "CANCELLED"
                    continue
                accepted = self._submit(index)
                self.state["tasks"][str(index)]["phase"] = self._phase(index)
                self._save()
                if accepted:
                    in_flight += 1
                    continue
                if self.state["tasks"][str(index)]["phase"] == "UNCERTAIN":
                    in_flight += 1
                if self.clock() < self.state["next_submit_at"]:
                    self.state["controller_status"] = "WAITING_FOR_CAPACITY"
                # One policy rejection blocks the cycle; no rejection storm.
                break
        self.state["inflight_jobs"] = in_flight
        if self.state.get("submission_blocked") and self.state["controller_status"] != "STOPPED":
            self.state["controller_status"] = "SUBMISSION_BLOCKED"
        phases = [task["phase"] for task in self.state["tasks"].values()]
        if not in_flight and all(p in ("COMPLETE", "FAILED", "CANCELLED") for p in phases):
            self.state["controller_status"] = "FINISHED"
        elif not in_flight and all(p in ("COMPLETE", "FAILED", "CANCELLED", "WAITING_INPUT") for p in phases):
            # No runnable producer remains. Failed/missing inputs (including
            # dependency cycles) cannot be fixed by resubmitting consumers.
            self.state["controller_status"] = "NOT_COMPLETE"
        self._save()
        return self.summary()

    def summary(self):
        return {"controller_status": self.state.get("controller_status"), "marker": self.marker,
                "inflight_jobs": self.state.get("inflight_jobs", 0),
                "tasks": {i: task["phase"] for i, task in self.state["tasks"].items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config) if args.config else {}
    if not isinstance(config, dict):
        parser.error("Launch configuration is missing or invalid JSON")
    controller = RollingController(args.plan, config)
    stopping = {"requested": False}
    def stop(*_):
        stopping["requested"] = True
        try:
            (controller.runtime / "STOP").touch()
        except OSError:
            pass
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with exclusive_lock(controller.runtime / "controller.lock"):
            atomic_json(controller.runtime / "controller.pid.json", {"pid": os.getpid(), "started_at": time.time()})
            while True:
                if stopping["requested"]:
                    (controller.runtime / "STOP").touch()
                snapshot = controller.step()
                print(json.dumps(snapshot, sort_keys=True), flush=True)
                if args.once or snapshot["controller_status"] in ("STOPPED", "FINISHED", "NOT_COMPLETE", "SUBMISSION_BLOCKED"):
                    return 0
                # Short waits allow timely SIGTERM handling on the login node.
                deadline = time.monotonic() + float(config.get("poll_seconds", 30))
                while not stopping["requested"] and time.monotonic() < deadline:
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
    except BlockingIOError:
        print(json.dumps({"controller_status": "EXISTING_CONTROLLER_REUSED"}), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
