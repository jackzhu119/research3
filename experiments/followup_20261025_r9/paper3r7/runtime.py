"""One durable Slurm allocation; three GPU slots execute the task DAG inside it."""
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
TRANSIENT_STATES = {"BOOT_FAIL", "NODE_FAIL", "REVOKED"}
DEFAULT_LAUNCH = {"python": sys.executable, "package": str(Path(__file__).resolve().parent.parent),
                  "account": "zhuziyuan", "partition": "hpc_gpu", "gres": "gpu:3", "slots": 3,
                  "cpus": 12, "memory_total": "128G", "time_limit": "5-03:00:00", "poll_seconds": 30,
                  "max_startup_retries": 2, "max_resumes": 100, "submit_reject_limit": 0,
                  "accounting_grace_seconds": 300}


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.%s" % os.getpid())
    with temp.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
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
    return "paper3r7-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _state(value):
    return value.split()[0].rstrip("+").upper() if value.strip() else "UNKNOWN"


def parse_queue(text, marker):
    rows = []
    for raw in text.splitlines():
        fields = raw.strip().split("|")
        if len(fields) < 3 or fields[1].strip() != marker:
            continue
        job = fields[0].strip()
        if not re.fullmatch(r"\d+", job):
            raise ValueError("Round 7 owns a regular job ID, never an array")
        state = _state(fields[2])
        rows.append({"job_id": job, "state": state,
                     "name": fields[3].strip() if len(fields) > 3 else "",
                     "queue_reason": fields[4].strip() if len(fields) > 4 and state == "PENDING" else None})
    return rows


def parse_accounting(text, marker, owned_names=()):
    rows = []
    for raw in text.splitlines():
        fields = raw.strip().split("|")
        if len(fields) < 5:
            continue
        comment, name = fields[3].strip(), fields[4].strip()
        if comment != marker and not (not comment and name in owned_names):
            continue
        job = fields[0].strip()
        if not re.fullmatch(r"\d+", job):
            continue
        try:
            code, signum = map(int, fields[2].split(":", 1))
        except (ValueError, TypeError):
            code, signum = None, None
        rows.append({"job_id": job, "state": _state(fields[1]), "exit_code": code,
                     "exit_signal": signum, "name": name})
    return rows


@contextmanager
def exclusive_lock(path):
    handle = Path(path).open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle
    finally:
        handle.close()


class SingleAllocationController:
    def __init__(self, plan_path, launch=None, runner=None, clock=None):
        self.plan_path = Path(plan_path).resolve()
        self.plan = read_json(self.plan_path)
        if not isinstance(self.plan, dict) or not isinstance(self.plan.get("tasks"), list):
            raise ValueError("Invalid task plan")
        self.output = Path(self.plan["output"]).resolve(); self.runtime = self.output / "runtime"
        (self.runtime / "receipts").mkdir(parents=True, exist_ok=True)
        (self.runtime / "outcomes").mkdir(exist_ok=True)
        self.launch = {**DEFAULT_LAUNCH, **(launch or {})}
        if int(self.launch["slots"]) != 3:
            raise ValueError("This plan uses exactly three GPU slots in one allocation")
        self.runner, self.clock = runner or subprocess.run, clock or time.time
        self.marker = job_marker(self.plan)
        self.state = read_json(self.runtime / "state.json", {})
        if self.state and self.state.get("marker") != self.marker:
            raise ValueError("Controller state belongs to another numerical plan")
        self.state.setdefault("version", 2)
        self.state.setdefault("mode", "SINGLE_NODE_THREE_GPUS")
        self.state["marker"] = self.marker
        self.state.setdefault("attempts", [])
        self.state.setdefault("created_at", self.clock())
        self.state.setdefault("submit_rejections", 0)
        self.state.setdefault("next_submit_at", 0)
        keys = [task.get("key") for task in self.plan["tasks"]]
        if any(not isinstance(key, str) or Path(key).name != key or key in (".", "..") for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Every task requires a unique directory key")
        for path in sorted((self.runtime / "receipts").glob("*.json")):
            receipt = read_json(path)
            if not isinstance(receipt, dict) or receipt.get("marker") != self.marker:
                continue
            previous = next((a for a in self.state["attempts"] if a["sequence"] == receipt["sequence"]), None)
            if previous is None:
                self.state["attempts"].append(receipt)
            else:
                previous.update(receipt)
        self.state["attempts"].sort(key=lambda a: a["sequence"])

    def _save(self):
        self.state["updated_at"] = self.clock()
        atomic_json(self.runtime / "state.json", self.state)

    def _receipt(self, attempt):
        atomic_json(self.runtime / "receipts" / ("allocation-a%03d.json" % attempt["sequence"]), attempt)

    def _new_attempt(self, **extra):
        seq = max([a["sequence"] for a in self.state["attempts"]] or [0]) + 1
        attempt = {"sequence": seq, "marker": self.marker, "intent_at": self.clock(), "status": "INTENT",
                   "job_id": None, "name": "%s-a%s" % (self.marker, seq), **extra}
        self.state["attempts"].append(attempt)
        self._receipt(attempt); self._save()
        return attempt

    def _run(self, command, timeout=30):
        return self.runner(command, capture_output=True, text=True, timeout=timeout)

    def _query(self):
        queue = self._run(["squeue", "--noheader", "--user", os.environ.get("USER", "zhuziyuan"),
                           "--format=%i|%k|%T|%j|%R"])
        if queue.returncode:
            raise RuntimeError("SQUEUE_FAILED")
        created = self.plan.get("prepared_at", self.state["created_at"])
        start = datetime.fromtimestamp(float(created) - 86400, timezone.utc).strftime("%Y-%m-%d")
        accounting = self._run(["sacct", "--noheader", "--parsable2", "--allocations", "--user",
                               os.environ.get("USER", "zhuziyuan"), "--starttime", start,
                               "--format=JobID%80,State%40,ExitCode,Comment%64,JobName%100"])
        if accounting.returncode:
            raise RuntimeError("SACCT_FAILED")
        names = {a["name"] for a in self.state["attempts"]}
        return parse_queue(queue.stdout, self.marker), parse_accounting(accounting.stdout, self.marker, names)

    def _adopt(self, rows):
        for row in rows:
            attempt = next((a for a in self.state["attempts"] if a.get("job_id") == row["job_id"]), None)
            if attempt is None:
                attempt = next((a for a in reversed(self.state["attempts"])
                                if not a.get("job_id") and a["status"] in ("INTENT", "UNCERTAIN")
                                and row["name"] == a["name"]), None)
            if attempt is None:
                attempt = self._new_attempt(origin="RECOVERED_FROM_SLURM", name=row["name"])
            attempt.update(job_id=row["job_id"], slurm_state=row["state"], observed_at=self.clock())
            if "queue_reason" in row:
                attempt["queue_reason"] = row["queue_reason"]
            if attempt["status"] not in ("FINISHED", "REJECTED"):
                attempt["status"] = "ACKNOWLEDGED"
            self._receipt(attempt)

    def _complete(self):
        return all(read_json(self.output / task["key"] / "result.json", {}).get("status") == "COMPLETE"
                   for task in self.plan["tasks"])

    def _finish(self, attempt, row):
        local = read_json(self.runtime / "outcomes" / (attempt["job_id"] + ".json"), {})
        state, code = row["state"], row.get("exit_code")
        if state == "CANCELLED" or local.get("stop_reason") == "terminated":
            outcome = "CANCELLED"
        elif self._complete():
            outcome = "COMPLETE"
        elif code == 75 or state in ("TIMEOUT", "PREEMPTED") or local.get("exit_code") == 75:
            outcome = "RESUME"
        elif state in TRANSIENT_STATES:
            outcome = "STARTUP_RETRY"
        else:
            outcome = "FAILED"
        attempt.update(status="FINISHED", outcome=outcome, exit_code=code, slurm_state=state,
                       finished_at=self.clock())
        self._receipt(attempt)

    def _submit(self):
        from .submit import build_sbatch_command
        attempt = self._new_attempt()
        options = {**self.launch, "attempt": attempt["sequence"]}
        command = build_sbatch_command(self.plan, self.launch["python"], options)
        attempt["command"] = command; self._receipt(attempt)
        try:
            result = self._run(command)
        except (OSError, subprocess.SubprocessError) as exc:
            attempt.update(status="UNCERTAIN", error_type=type(exc).__name__)
        else:
            match = re.fullmatch(r"(\d+)(?:;[^\s]+)?", result.stdout.strip()) if result.returncode == 0 else None
            if match:
                attempt.update(status="ACKNOWLEDGED", job_id=match[1])
                self.state["submit_rejections"] = 0
            elif result.returncode == 0:
                attempt.update(status="UNCERTAIN", error_type="MissingParsableJobID")
            elif re.search(r"sbatch:\s*error:.*(?:Batch job submission failed|QOS|Invalid|Job violates)", result.stderr, re.I | re.S):
                attempt.update(status="REJECTED", returncode=result.returncode, stderr=result.stderr[-2000:])
                count = self.state["submit_rejections"] + 1
                self.state["submit_rejections"] = count
                self.state["next_submit_at"] = self.clock() + min(300, 30 * 2 ** min(count - 1, 4))
                limit = int(self.launch["submit_reject_limit"])
                if not re.search(r"QOS|MaxSubmitJob|accounting/QOS policy", result.stderr, re.I) or (limit and count >= limit):
                    self.state["submission_blocked"] = "Scheduler rejected the allocation configuration"
            else:
                attempt.update(status="UNCERTAIN", returncode=result.returncode, error_type="UncertainSbatchFailure")
        self._receipt(attempt)
        self._save()
        return attempt

    def step(self):
        if (self.runtime / "STOP").exists() or (self.output / "STOP").exists():
            self.state["controller_status"] = "STOPPED"
        elif self._complete():
            self.state["controller_status"] = "FINISHED"
        else:
            try:
                queue, accounting = self._query()
                self._adopt(accounting); self._adopt(queue)
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
                self.state.update(controller_status="OBSERVATION_BLOCKED", observation_error=type(exc).__name__)
                self._save(); return self.summary()
            active = {row["job_id"]: row for row in queue}
            account = {row["job_id"]: row for row in accounting}
            for attempt in self.state["attempts"]:
                if attempt["status"] in ("FINISHED", "REJECTED"):
                    continue
                job_id = attempt.get("job_id")
                if job_id in active:
                    attempt["status"] = "ACKNOWLEDGED"
                elif job_id in account and account[job_id]["state"] not in ACTIVE_STATES:
                    self._finish(attempt, account[job_id])
            unfinished = [a for a in self.state["attempts"] if a["status"] not in ("FINISHED", "REJECTED")]
            if unfinished:
                latest = unfinished[-1]; row = active.get(latest.get("job_id"))
                if row is None:
                    observed = account.get(latest.get("job_id"))
                    if observed and observed["state"] in ACTIVE_STATES:
                        row = observed
                if row:
                    self.state["controller_status"] = "QUEUED" if row["state"] == "PENDING" else "ALLOCATION_RUNNING"
                elif latest.get("job_id") and self.clock() - latest["intent_at"] <= int(self.launch["accounting_grace_seconds"]):
                    self.state["controller_status"] = "AWAITING_ACCOUNTING"
                else:
                    self.state["controller_status"] = "SUBMISSION_UNCERTAIN"
            else:
                outcomes = [a.get("outcome") for a in self.state["attempts"]]
                if "CANCELLED" in outcomes:
                    self.state["controller_status"] = "CANCELLED"
                elif outcomes and outcomes[-1] == "FAILED":
                    self.state["controller_status"] = "NOT_COMPLETE"
                elif outcomes.count("STARTUP_RETRY") > int(self.launch["max_startup_retries"]) or outcomes.count("RESUME") > int(self.launch["max_resumes"]):
                    self.state["controller_status"] = "NOT_COMPLETE"
                elif self.state.get("submission_blocked"):
                    self.state["controller_status"] = "SUBMISSION_BLOCKED"
                elif self.clock() < self.state["next_submit_at"]:
                    self.state["controller_status"] = "WAITING_FOR_CAPACITY"
                else:
                    attempt = self._submit()
                    self.state["controller_status"] = ("AWAITING_ACCOUNTING" if attempt["status"] == "ACKNOWLEDGED" else
                        "SUBMISSION_BLOCKED" if self.state.get("submission_blocked") else
                        "WAITING_FOR_CAPACITY" if attempt["status"] == "REJECTED" else "SUBMISSION_UNCERTAIN")
        self._save(); return self.summary()

    def summary(self):
        latest = self.state["attempts"][-1] if self.state["attempts"] else {}
        return {"controller_status": self.state.get("controller_status"), "marker": self.marker,
                "mode": "SINGLE_NODE_THREE_GPUS", "job_id": latest.get("job_id"),
                "slurm_state": latest.get("slurm_state"), "queue_reason": latest.get("queue_reason"),
                "allocation_attempts": len(self.state["attempts"]), "gpu_slots": 3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True); parser.add_argument("--config"); parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config, {}) if args.config else {}
    controller = SingleAllocationController(args.plan, config)
    stopping = {"requested": False}
    def stop(*_):
        stopping["requested"] = True
        try:
            (controller.runtime / "STOP").touch()
        except OSError:
            pass
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        with exclusive_lock(controller.runtime / "controller.lock"):
            atomic_json(controller.runtime / "controller.pid.json", {"pid": os.getpid(), "started_at": time.time()})
            while True:
                snapshot = controller.step(); print(json.dumps(snapshot, sort_keys=True), flush=True)
                if args.once or snapshot["controller_status"] in ("STOPPED", "FINISHED", "CANCELLED", "NOT_COMPLETE", "SUBMISSION_BLOCKED"):
                    return 0
                deadline = time.monotonic() + float(config.get("poll_seconds", 30))
                while not stopping["requested"] and time.monotonic() < deadline:
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
    except BlockingIOError:
        print(json.dumps({"controller_status": "EXISTING_CONTROLLER_REUSED"}), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
