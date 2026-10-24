"""Run an entire task DAG inside one single-node, three-GPU Slurm allocation.

This module is stdlib-only. Each child sees one allocated CUDA device and a
disjoint observed CPU affinity set. Scientific failures stay task-local;
cooperative checkpoint exits pause the allocation for the outer scheduler.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .io import emit, put, read

PACKAGE = Path(__file__).resolve().parent.parent


def optional_json(path, default=None):
    try:
        return read(path)
    except (OSError, ValueError, TypeError):
        return default


def allocated_devices(environment, slots):
    """Use only devices named by this allocation; never enumerate the node."""
    for name in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = environment.get(name, "").strip()
        if name == "CUDA_VISIBLE_DEVICES" and name in environment and (not value or value in ("-1", "NoDevFiles", "none", "void")):
            raise ValueError("CUDA_VISIBLE_DEVICES explicitly disables GPU access")
        if not value or value in ("-1", "NoDevFiles", "none", "void"):
            continue
        tokens = []
        for token in value.split(","):
            token = token.strip()
            match = re.fullmatch(r"(\d+)-(\d+)", token)
            if match and name != "CUDA_VISIBLE_DEVICES":
                first, last = map(int, match.groups())
                if last < first or last - first > 64:
                    raise ValueError("Invalid allocated GPU range")
                tokens.extend(str(index) for index in range(first, last + 1))
            elif token and (token.isdigit() or token.startswith(("GPU-", "MIG-"))):
                tokens.append(token)
            else:
                raise ValueError("Unsupported allocated GPU token in " + name)
        if len(tokens) != len(set(tokens)):
            raise ValueError("Allocated GPU mapping contains duplicates")
        if len(tokens) < slots:
            raise ValueError("Allocation exposes fewer GPUs than requested slots: " + name)
        return tokens[:slots], name
    raise ValueError("No allocation-specific GPU mapping; refusing to enumerate or use unallocated GPUs")


def cpu_groups(affinity, slots, cpus_per_slot):
    available = sorted(set(affinity))
    per_slot = min(cpus_per_slot, len(available) // slots)
    if per_slot < 1:
        raise ValueError("Observed CPU affinity cannot isolate all GPU workers")
    return [available[index * per_slot:(index + 1) * per_slot] for index in range(slots)]


def worker_command(python, plan_path, index, *, device="cuda:0", package=PACKAGE):
    return [str(python), "-I", "-S", "-B", "-u", str(Path(package) / "paper3r4/entry.py"),
            "worker", "--plan", str(plan_path), "--task", str(index), "--device", device]


class Allocation:
    def __init__(self, plan_path, *, python=None, slots=3, cpus_per_slot=16,
                 device="cuda:0", environment=None, affinity=None,
                 popen=None, clock=None, command_factory=None):
        self.plan_path = Path(plan_path).resolve()
        self.plan = read(self.plan_path)
        self.output = Path(self.plan["output"]).resolve()
        self.runtime = self.output / "runtime"
        self.runtime.mkdir(parents=True, exist_ok=True)
        (self.runtime / "errors").mkdir(exist_ok=True)
        (self.output / "logs").mkdir(parents=True, exist_ok=True)
        if not isinstance(self.plan.get("tasks"), list) or not self.plan["tasks"]:
            raise ValueError("Allocation requires a nonempty task plan")
        self.slots_count, self.cpus_per_slot = int(slots), int(cpus_per_slot)
        if not 1 <= self.slots_count <= 3 or self.cpus_per_slot < 1:
            raise ValueError("Require one to three GPU slots and positive CPUs per slot")
        if device not in ("cuda:0", "cpu"):
            raise ValueError("Only one visible CUDA device per worker is supported")
        self.device = device
        self.environment = dict(os.environ if environment is None else environment)
        if device == "cpu":
            self.devices, self.device_mapping_source = [""] * self.slots_count, "CPU_TEST_ONLY"
        else:
            self.devices, self.device_mapping_source = allocated_devices(self.environment, self.slots_count)
        available = os.sched_getaffinity(0) if affinity is None else affinity
        self.cpu_affinities = cpu_groups(available, self.slots_count, self.cpus_per_slot)
        self.python = str(python or sys.executable)
        self.popen = popen or subprocess.Popen
        self.clock = clock or time.time
        self.command_factory = command_factory or worker_command
        self.running = {}
        self.stop_reason = None
        self._lock = None
        self.state_path = self.runtime / "allocation_state.json"
        self.previous = optional_json(self.state_path, {})
        if self.previous and self.previous.get("recipe_sha256") != self.plan.get("recipe_sha256"):
            raise ValueError("Allocation state belongs to another numerical recipe")
        tasks = {}
        seen = set()
        for index, task in enumerate(self.plan["tasks"]):
            key = task.get("key")
            if not key or key in seen:
                raise ValueError("Task keys must be nonempty and unique")
            seen.add(key)
            previous = self.previous.get("tasks", {}).get(str(index), {})
            if previous and previous.get("key") != key:
                raise ValueError("Allocation task order changed")
            phase = previous.get("phase")
            # A terminated allocation's running/paused children resume in a
            # fresh process. Explicit scientific failures remain visible.
            if self._result_complete(index):
                phase = "COMPLETE"
            elif phase == "COMPLETE":
                phase = "FAILED"
            else:
                phase = phase if phase in ("FAILED", "CANCELLED") else "NEW"
            tasks[str(index)] = {"key": key, "phase": phase, "status": phase,
                                 "attempt": int(previous.get("attempt", 0))}
        known = {task["key"] for task in self.plan["tasks"]}
        for task in self.plan["tasks"]:
            if not set(task.get("requires", [])) <= known:
                raise ValueError("Task dependency does not exist in this plan")
        self.index_by_key = {task["key"]: index for index, task in enumerate(self.plan["tasks"])}
        self.state = {"version": 1, "recipe_sha256": self.plan.get("recipe_sha256"),
            "job_id": self.environment.get("SLURM_JOB_ID"), "allocation_status": "STARTING",
            "requested_slots": self.slots_count, "requested_cpus_per_slot": self.cpus_per_slot,
            "observed_cpus_per_slot": [len(group) for group in self.cpu_affinities],
            "gpu_mapping_source": self.device_mapping_source, "tasks": tasks,
            "slots": {str(slot): {"status": "IDLE", "task_index": None, "task_key": None,
                "pid": None, "cuda_visible_devices": self.devices[slot],
                "cpu_affinity": group} for slot, group in enumerate(self.cpu_affinities)}}
        self._last_heartbeat = -float("inf")
        self._last_persist = -float("inf")
        self._state_changed = True
        self._last_snapshot = None
        if self.environment.get("PAPER3R4_STOP_REQUESTED") == "1" or os.environ.get("PAPER3R4_STOP_REQUESTED") == "1":
            self.request_stop("early_signal")

    def acquire(self):
        self._lock = (self.runtime / "allocation.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise RuntimeError("Another allocation process already owns this run") from None
        return self

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _result_complete(self, index):
        task = self.plan["tasks"][index]
        result = optional_json(self.output / task["key"] / "result.json", {})
        if result.get("status") != "COMPLETE":
            return False
        if result.get("recipe_sha256", self.plan.get("recipe_sha256")) != self.plan.get("recipe_sha256"):
            return False
        return True

    def _set_phase(self, index, phase, **values):
        row = self.state["tasks"][str(index)]
        update = dict(phase=phase, status=phase, **values)
        self._state_changed |= any(row.get(key) != value for key, value in update.items())
        row.update(update)

    def request_stop(self, reason="checkpoint_requested"):
        if self.stop_reason is None:
            self.stop_reason = reason
            self._state_changed = True
            emit("ALLOCATION_CHECKPOINT_REQUESTED", reason=reason, active_workers=len(self.running))
        for active in self.running.values():
            if active.get("stop_sent"):
                continue
            try:
                active["process"].send_signal(signal.SIGUSR1)
                active["stop_sent"] = True
            except ProcessLookupError:
                pass

    def _error(self, index, reason, **values):
        try:
            put(self.runtime / "errors" / f"task-{index}.json",
                {"task_index": index, "task_key": self.plan["tasks"][index]["key"],
                 "reason": reason, "updated_at": self.clock(), **values})
        except OSError as error:
            emit("ALLOCATION_ERROR_RECORD_RETRY", task=index, reason=reason,
                 error_type=type(error).__name__)

    def _start(self, slot, index):
        state = self.state["tasks"][str(index)]
        attempt = state["attempt"] + 1
        out_path = self.output / "logs" / f"task-{index}-attempt{attempt}.out"
        err_path = self.output / "logs" / f"task-{index}-attempt{attempt}.err"
        stdout = stderr = None
        try:
            stdout, stderr = out_path.open("ab", buffering=0), err_path.open("ab", buffering=0)
        except OSError as error:
            if stdout is not None:
                stdout.close()
            self._set_phase(index, "FAILED", attempt=attempt, error_type=type(error).__name__)
            self._error(index, "worker_log_open_failed", error_type=type(error).__name__, message=str(error))
            return
        env = dict(self.environment)
        env.update(CUDA_VISIBLE_DEVICES=self.devices[slot], PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(min(4, len(self.cpu_affinities[slot]))),
            MKL_NUM_THREADS=str(min(4, len(self.cpu_affinities[slot]))), OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1", PAPER3R4_GPU_SLOT=str(slot),
            PAPER3R4_CPU_AFFINITY=",".join(map(str, self.cpu_affinities[slot])))
        env.pop("PAPER3R4_STOP_REQUESTED", None)
        for key in ("BASH_ENV", "ENV", "PYTHONHOME", "PYTHONPATH"):
            env.pop(key, None)
        affinity = set(self.cpu_affinities[slot])
        command = self.command_factory(self.python, self.plan_path, index, device=self.device)
        try:
            process = self.popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env=env, start_new_session=True, close_fds=True,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity))
        except OSError as error:
            stdout.close(); stderr.close()
            self._set_phase(index, "FAILED", attempt=attempt, error_type=type(error).__name__)
            self._error(index, "worker_spawn_failed", error_type=type(error).__name__, message=str(error))
            return
        self.running[slot] = {"process": process, "index": index, "stdout": stdout,
                              "stderr": stderr, "stop_sent": False}
        self._set_phase(index, "RUNNING", attempt=attempt, slot=slot, pid=process.pid,
                        stdout=str(out_path), stderr=str(err_path))
        self.state["slots"][str(slot)].update(status="RUNNING", task_index=index,
            task_key=self.plan["tasks"][index]["key"], pid=process.pid,
            stdout=str(out_path), stderr=str(err_path))
        emit("ALLOCATION_TASK_STARTED", slot=slot, task=index, pid=process.pid,
             cpu_count=len(affinity), stdout=str(out_path), stderr=str(err_path))

    def _reap(self):
        for slot, active in list(self.running.items()):
            code = active["process"].poll()
            if code is None:
                continue
            active["stdout"].close(); active["stderr"].close()
            index = active["index"]
            if self._result_complete(index):
                phase = "COMPLETE"
            elif code == 75 or self.stop_reason is not None:
                phase = "PAUSED"
            else:
                phase = "FAILED"
                self._error(index, "worker_returned_without_complete_result", exit_code=code)
            self._set_phase(index, phase, exit_code=code, finished_at=self.clock())
            self.state["slots"][str(slot)].update(status="IDLE", task_index=None,
                                                   task_key=None, pid=None)
            del self.running[slot]
            emit("ALLOCATION_TASK_FINISHED", slot=slot, task=index, exit_code=code, status=phase)
            if code == 75:
                self.request_stop("worker_checkpoint_exit")

    def _ready(self):
        ready = []
        for index, task in enumerate(self.plan["tasks"]):
            phase = self.state["tasks"][str(index)]["phase"]
            if phase in ("COMPLETE", "RUNNING", "FAILED", "CANCELLED", "PAUSED"):
                continue
            dependencies = [self.index_by_key[key] for key in task.get("requires", [])]
            if all(self.state["tasks"][str(dep)]["phase"] == "COMPLETE" for dep in dependencies):
                self._set_phase(index, "NEW")
                ready.append(index)
            else:
                self._set_phase(index, "WAITING_INPUT")
        return ready

    def snapshot(self):
        # Completion markers are inspected once at allocation startup and
        # exactly when a child exits, not rescanned on every polling tick.
        # Durable writes occur on transitions and every ten seconds; a shared
        # filesystem should not receive hundreds of redundant reads/fsyncs.
        if not self._state_changed and self.clock() - self._last_persist < 10:
            return self._last_snapshot
        self.state["updated_at"] = self.clock()
        self.state["complete_tasks"] = sum(row["phase"] == "COMPLETE" for row in self.state["tasks"].values())
        self.state["total_tasks"] = len(self.plan["tasks"])
        self.state["failed_tasks"] = sum(row["phase"] == "FAILED" for row in self.state["tasks"].values())
        self.state["waiting_tasks"] = sum(row["phase"] == "WAITING_INPUT" for row in self.state["tasks"].values())
        public_slots = {}
        for key, row in self.state["slots"].items():
            public = {name: row[name] for name in ("status", "task_index", "task_key", "pid")}
            public["gpu_slot"] = int(key)
            public["cpu_count"] = len(row["cpu_affinity"])
            if row["task_index"] is not None:
                path = self.output / row["task_key"] / "progress.json"
                progress = optional_json(path, {})
                public["worker_progress"] = {name: progress[name] for name in
                    ("status", "stage", "outer_update", "total_updates", "session", "sessions_total", "budget")
                    if name in progress}
                if path.is_file():
                    public["progress_age_seconds"] = round(max(0, self.clock() - path.stat().st_mtime), 1)
            public_slots[key] = public
        progress = {key: self.state[key] for key in ("allocation_status", "job_id", "recipe_sha256",
            "requested_slots", "requested_cpus_per_slot", "observed_cpus_per_slot", "updated_at",
            "complete_tasks", "total_tasks", "failed_tasks", "waiting_tasks")}
        progress.update(slots=public_slots, stop_reason=self.stop_reason)
        for path, value in ((self.state_path, self.state),
                            (self.output / "allocation_progress.json", progress)):
            try:
                put(path, value)
            except OSError as error:
                emit("ALLOCATION_OBSERVATION_WRITE_RETRY", filename=path.name,
                     error_type=type(error).__name__, training_controlled=False)
        self._last_snapshot = progress
        self._last_persist = self.clock()
        self._state_changed = False
        return progress

    def cycle(self):
        if (self.runtime / "STOP").exists() or (self.output / "STOP").exists():
            self.request_stop("explicit_stop")
        self._reap()
        ready = self._ready()
        if self.stop_reason is None:
            for slot in range(self.slots_count):
                if slot not in self.running and ready:
                    self._start(slot, ready.pop(0))
        if self.stop_reason is not None:
            status = "CHECKPOINTING" if self.running else "PAUSED_CHECKPOINT_SAVED"
        elif self.running:
            status = "RUNNING"
        elif all(row["phase"] == "COMPLETE" for row in self.state["tasks"].values()):
            status = "COMPLETE"
        else:
            status = "NOT_COMPLETE"
        self._state_changed |= self.state["allocation_status"] != status
        self.state["allocation_status"] = status
        snapshot = self.snapshot()
        if self.clock() - self._last_heartbeat >= 30 or not self.running:
            emit("ALLOCATION_PROGRESS", **snapshot)
            self._last_heartbeat = self.clock()
        return {"COMPLETE": 0, "NOT_COMPLETE": 2, "PAUSED_CHECKPOINT_SAVED": 75}.get(status)

    def run(self, poll_seconds=.5):
        self.acquire()
        try:
            if any(len(group) < self.cpus_per_slot for group in self.cpu_affinities):
                emit("ALLOCATION_CPU_AFFINITY_SHORTFALL", requested_per_slot=self.cpus_per_slot,
                    observed_per_slot=[len(group) for group in self.cpu_affinities],
                    allocation_request_changed=False)
            while True:
                result = self.cycle()
                if result is not None:
                    return result
                time.sleep(max(.01, poll_seconds))
        finally:
            self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--cpus-per-slot", type=int, default=16)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
    parser.add_argument("--poll-seconds", type=float, default=.5)
    args = parser.parse_args(argv)
    allocation = Allocation(args.plan, python=args.python, slots=args.slots,
                            cpus_per_slot=args.cpus_per_slot, device=args.device)
    old_handlers = {}
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda number, _frame: allocation.request_stop(signal.Signals(number).name))
    try:
        return allocation.run(args.poll_seconds)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
