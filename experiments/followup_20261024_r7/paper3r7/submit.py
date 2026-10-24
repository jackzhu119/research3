"""Round 7 preparation and detached login-node process startup (stdlib only)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .runtime import DEFAULT_LAUNCH, atomic_json, job_marker, read_json

PACKAGE = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = "/public/home/zhuziyuan/projects/paper3"
DEFAULT_PYTHON = "/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python"


def allocation_memory(memory_total, slots=None):
    """Normalize the authoritative whole-job memory request; never multiply it."""
    match = re.fullmatch(r"([1-9]\d*)([KMGT])", str(memory_total).strip().upper())
    if not match:
        raise ValueError("Total job memory must be a positive Slurm size such as 128G")
    return str(int(match[1])) + match[2]


def memory_budget_gib(memory_total, slots=3):
    """Display-only equal-share budget; Slurm enforces the shared total pool."""
    value = allocation_memory(memory_total)
    factors = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}
    return int(value[:-1]) * factors[value[-1]] / int(slots)


def build_sbatch_command(plan, python, launch=None):
    options = {**DEFAULT_LAUNCH, **(launch or {})}
    slots = int(options["slots"])
    if slots != 3:
        raise ValueError("Seventh round uses three GPU slots inside one node allocation")
    package = Path(options["package"]).resolve()
    output = Path(plan["output"]).resolve()
    name = "%s-a%s" % (job_marker(plan), options.get("attempt", 1))
    command = ["sbatch", "--parsable", "--job-name=" + name,
               "--partition=" + str(options["partition"]), "--nodes=1", "--ntasks=1",
               "--cpus-per-task=" + str(int(options["cpus"]) * slots), "--mem=" + allocation_memory(options["memory_total"]),
               "--gres=" + str(options["gres"]), "--time=" + str(options["time_limit"]),
               "--signal=B:USR1@300", "--no-requeue", "--chdir=/tmp", "--open-mode=append",
               "--export=ALL,BASH_ENV=,ENV=,PYTHONHOME=,PYTHONPATH=",
               "--output=" + str(output / "logs/train-%j.out"),
               "--error=" + str(output / "logs/train-%j.err"), "--comment=" + job_marker(plan)]
    if options.get("account"):
        command.append("--account=" + str(options["account"]))
    return command + [str(package / "train.sbatch"), str(python), str(package),
                      str(output / "plan.json"), str(output), str(slots), str(options["cpus"])]


def discover_credentials(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Credential file does not exist: " + str(path))
        return path
    env_path = os.environ.get("PAPER3_GITHUB_CREDENTIALS")
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path.resolve()
    candidates = [PACKAGE / "private/github.json", Path.home() / ".config/paper3/github.json"]
    # Reuse an already configured school credential by path. The credential
    # contents and previous rounds' run directories are never changed.
    for name in ("paper3-followup", "paper3-round3", "paper3-round4", "paper3-round5", "paper3-round6", "paper3-round7"):
        base = Path.home() / name
        if base.is_dir():
            candidates.extend(sorted(base.glob("*/private/github.json")))
            candidates.append(base / "private/github.json")
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _spawn(command, log):
    with Path(log).open("ab", buffering=0) as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    return process.pid


def _same_prepared_tasks(left, right):
    """Ignore input locations whose contents are already bound by the recipe.

    Retain the original prepared tasks so existing task/checkpoint locators
    remain unchanged. Every numerical field and dependency must still agree.
    """
    def numerical(tasks):
        if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
            return None
        return [{key: value for key, value in task.items()
                 if key not in ("external_manifest", "basis_output")} for task in tasks]
    a, b = numerical(left), numerical(right)
    return a is not None and b is not None and a == b


def _same_live_launch(left, right):
    # A documentation-only repackage may have a new installation directory.
    # Keep a running controller's original launch rather than disrupting it.
    return ({key: value for key, value in left.items() if key != "package"}
            == {key: value for key, value in right.items() if key != "package"})


def start_background(plan_path, launch, credentials=None):
    """Start observation before submission; QoS capacity cannot suppress it."""
    output = Path(read_json(plan_path)["output"])
    base = [str(launch["python"]), "-I", "-S", "-B", "-u", str(PACKAGE / "paper3r7/entry.py")]
    monitor_command = base + ["monitor", "--plan", str(plan_path)]
    if credentials:
        monitor_command += ["--credentials", str(credentials)]
    monitor_pid = _spawn(monitor_command, output / "monitor.log")
    controller_command = base + ["controller", "--plan", str(plan_path),
                                 "--config", str(output / "runtime/launch.json")]
    controller_pid = _spawn(controller_command, output / "controller.log")
    receipt = {"status": "DETACHED_PROCESSES_STARTED", "started_at": time.time(),
               "monitor_pid": monitor_pid, "controller_pid": controller_pid,
               "monitor_log": str(output / "monitor.log"), "controller_log": str(output / "controller.log"),
               "submission_receipts": str(output / "runtime/receipts"),
               "note": "PIDs acknowledge local process startup; Slurm receipts acknowledge actual submissions"}
    atomic_json(output / "runtime/startup.json", receipt)
    return receipt


def main(argv=None):
    # In earlier rounds --memory meant per-GPU. Never accept it as an
    # abbreviation of the new total-pool flag and silently underallocate RAM.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output")
    parser.add_argument("--external-manifest", help="Optional external-stage manifest; requires --basis-run")
    parser.add_argument("--basis-run", help="Read-only core Round 7 output containing source basis models")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--credentials", help="Private GitHub JSON file; contents are never copied into the package or plan")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--partition", default="hpc_gpu",
                        help="Requested A100-PCIe-40GB school partition; no automatic queue fallback")
    parser.add_argument("--account", default="zhuziyuan")
    parser.add_argument("--gres", default="gpu:3", help="Three GPUs in the selected partition; no queue fallback")
    parser.add_argument("--gpus", "--slots", dest="slots", type=int, default=3)
    parser.add_argument("--cpus", type=int, default=12, help="CPU cores per GPU slot (allocation total defaults to 36)")
    parser.add_argument("--memory-total", default="128G", help="Whole-job host memory shared by all three GPU slots; never multiplied")
    parser.add_argument("--time", dest="time_limit", default="5-03:00:00")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-startup-retries", type=int, default=2)
    parser.add_argument("--max-resumes", type=int, default=100)
    parser.add_argument("--submit-reject-limit", type=int, default=0,
                        help="0 waits for QoS capacity with capped backoff; positive values bound quota rejections")
    args = parser.parse_args(argv)
    if args.slots != 3 or args.cpus < 1 or args.poll_seconds < 1:
        parser.error("Require exactly three GPU slots in one job, positive CPUs and poll interval >=1 second")
    try:
        allocation_memory(args.memory_total)
    except ValueError as error:
        parser.error(str(error))
    if not re.fullmatch(r"gpu(?::[^:]+)?:3", args.gres):
        parser.error("GRES must request three GPUs, for example gpu:3")
    if min(args.max_startup_retries, args.max_resumes, args.submit_reject_limit) < 0:
        parser.error("Retry limits must be nonnegative")
    root, python = Path(args.root).expanduser().resolve(), Path(args.python).expanduser().absolute()
    if not root.is_dir():
        parser.error("Project root does not exist: " + str(root))
    if not python.is_file() or not os.access(python, os.X_OK):
        parser.error("Existing Python interpreter is not executable: " + str(python))
    if bool(args.external_manifest) != bool(args.basis_run):
        parser.error("--external-manifest and --basis-run must be provided together")
    external = bool(args.external_manifest and args.basis_run)
    external_manifest = Path(args.external_manifest).expanduser().resolve() if external else None
    basis_run = Path(args.basis_run).expanduser().resolve() if external else None
    if external and not external_manifest.is_file():
        parser.error("External manifest does not exist: " + str(external_manifest))
    default_name = "followup_20261024_r7_external" if external else "followup_20261024_r7"
    output = Path(args.output).expanduser().resolve() if args.output else root / "runs" / default_name
    protected = {(root / "runs" / name).resolve() for name in
                 ("source_training_v2", "meta", "training_campaign_v1", "followup_20261023_v2",
                  "followup_20261024_r3", "followup_20261024_r3_weibo",
                  "followup_20261024_r4", "followup_20261024_r4_weibo",
                  "followup_20261024_r5", "followup_20261024_r5_weibo",
                  "followup_20261024_r6", "followup_20261024_r6_external", "followup_20261024_r6_weibo")}
    if output in protected:
        parser.error("Output is a protected earlier run directory: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "runtime").mkdir(exist_ok=True)
    credentials = discover_credentials(args.credentials)
    with (output / "runtime/prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if external:
            from .plan import make_external_plan
            plan = make_external_plan(root, output, external_manifest, basis_run,
                                      gpus=args.slots, cpus=args.cpus)
        else:
            from .plan import make_plan
            plan = make_plan(root, output, gpus=args.slots, cpus=args.cpus)
        plan.update(root=str(root), output=str(output))
        path = output / "plan.json"
        previous = read_json(path)
        if previous:
            for field in ("recipe_sha256", "root", "output"):
                if previous.get(field) != plan.get(field):
                    raise SystemExit("Round 7 plan changed; preserve the existing run and use a separate project root/output for this recipe")
            if not _same_prepared_tasks(previous.get("tasks"), plan.get("tasks")):
                raise SystemExit("Round 7 numerical tasks changed; preserve the existing run and use a separate output")
            plan = previous
        else:
            plan["prepared_at"] = time.time()
            atomic_json(path, plan)
        launch = {**DEFAULT_LAUNCH, **{key: getattr(args, key) for key in
                  ("account", "partition", "gres", "cpus", "memory_total", "slots", "time_limit",
                   "poll_seconds", "max_startup_retries", "max_resumes", "submit_reject_limit")},
                  "python": str(python), "package": str(PACKAGE)}
        existing_launch = read_json(output / "runtime/launch.json")
        reused_live_controller = False
        if existing_launch and existing_launch != launch:
            # A live controller reads its launch configuration once. Avoid a
            # misleading on-disk resource update while it continues with old values.
            with (output / "runtime/controller.lock").open("a") as controller_lock:
                try:
                    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    if not _same_live_launch(existing_launch, launch):
                        raise SystemExit("Existing controller is running with a different launch configuration; stop that controller before changing its resources")
                    launch = existing_launch
                    reused_live_controller = True
        if not reused_live_controller:
            atomic_json(output / "runtime/launch.json", launch)
        prepared = {"status": "PREPARED", "plan": str(path), "tasks": len(plan["tasks"]),
                    "output": str(output), "partition": launch["partition"], "gres": launch["gres"],
                    "allocation_mode": "SINGLE_NODE_THREE_GPUS", "slurm_jobs": 1,
                    "gpu_slots": launch["slots"], "cpus_per_gpu": launch["cpus"],
                    "cpus_per_job": launch["slots"] * launch["cpus"],
                    "host_memory_per_gpu_budget_gib": memory_budget_gib(launch["memory_total"], launch["slots"]),
                    "host_memory_per_job": allocation_memory(launch["memory_total"]),
                    "host_memory_scope": "Shared total pool; per-GPU equal share is a display budget, not a process limit",
                    "credentials_discovered": credentials is not None,
                    "existing_controller_reused": reused_live_controller,
                    "gpu_resource": "A100-PCIe-40GB partition requested; live GPU allocation not yet observed",
                    "validation": "Existing root/interpreter and immutable plan checked; no GPU, allocation, or training claim"}
        print(json.dumps(prepared, sort_keys=True), flush=True)
        if args.submit:
            if (output / "runtime/STOP").exists() or (output / "STOP").exists():
                raise SystemExit("STOP is present. Respecting the recorded stop; no processes started")
            print(json.dumps(start_background(path, launch, credentials), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
