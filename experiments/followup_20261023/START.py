#!/usr/bin/env python3
"""Prepare or submit the independent 3-GPU follow-up. Stdlib only on login1."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import re

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from paper3v2.io import read, put, digest, sha

VARIANTS = ("P", "global", "scalar", "dual", "PCT", "softPCT", "temporal", "dual_temporal",
            "softPCT_mean", "runmean", "symonly", "skewonly")
REFERENCE_VARIANTS = ("head_only", "support_EA")
COMPARISONS = (("dual", "P"), ("dual", "scalar"), ("dual", "global"), ("dual", "symonly"),
    ("dual", "skewonly"), ("softPCT", "PCT"), ("softPCT", "softPCT_mean"),
    ("temporal", "P"), ("temporal", "runmean"), ("dual_temporal", "dual"))
SEEDS = (20261018, 20261019, 20261020)

def make_plan(root, output, gpus=3, cpus=24):
    scientific = {"version": "followup-20261023-v2", "variants": VARIANTS, "seeds": SEEDS,
        "rank": 4, "radius": .15, "updates": 1000, "inner_steps": 10, "episodes_per_update": 4,
        "validate_every": 50, "checkpoint_every": 10, "budgets": [20, 4, 8, 12, 16],
        "comparisons": COMPARISONS, "reference_variants": REFERENCE_VARIANTS,
        "sensitivity_settings": [(2, .15), (8, .15), (4, .05), (4, .35)],
        "diagnostics": ["label_noise10", "label_noise20", "gain10", "gain25", "rotation10", "rotation25"],
        "diagnostic_variants": ["P", "dual", "PCT", "softPCT", "softPCT_mean"]}
    # Operational fixes to logging, submission and checkpoints do not invalidate a fit.
    # Numerical / data-selection changes do create a distinct experiment.
    files = [PACKAGE / "paper3v2" / name for name in ("algorithm.py", "data.py", "diagnostics.py")]
    files += sorted((PACKAGE / "p3work").glob("*.py"))
    recipe = digest({"science": scientific, "code": {str(p.relative_to(PACKAGE)): sha(p) for p in files}})
    # Each wave contains different methods; matching seeds keep comparisons paired.
    tasks = [{"key": f"{variant}-seed{seed}", "seed": seed, "arm": "core", "setting": "r4-rho0p15",
              "config": {"variant": variant, "rank": 4, "radius": .15}}
             for seed in SEEDS for variant in VARIANTS]
    for rank, radius in scientific["sensitivity_settings"]:
        setting = f"r{rank}-rho{str(radius).replace('.', 'p')}"
        tasks += [{"key": f"{variant}-{setting}-seed{seed}", "seed": seed, "arm": "sensitivity",
                   "setting": setting, "budgets": [20],
                   "config": {"variant": variant, "rank": rank, "radius": radius}}
                  for seed in SEEDS for variant in ("P", "dual")]
    tasks += [{"key": f"{variant}-seed{seed}", "seed": seed, "arm": "reference", "setting": "no_meta_fit",
               "config": {"variant": variant, "rank": 4, "radius": .15}}
              for seed in SEEDS for variant in REFERENCE_VARIANTS]
    for task in tasks:
        task['diagnostics'] = list(scientific['diagnostics']) if task['arm'] == 'core' and task['config']['variant'] in scientific['diagnostic_variants'] else []
    return {**scientific, "root": str(root), "output": str(output), "gpus": gpus, "cpus": cpus,
            "memory_per_gpu": "128G", "recipe_sha256": recipe, "tasks": tasks,
            "seeds": list(SEEDS), "variants": list(VARIANTS)}

def job_marker(plan):
    return "paper3-followup-" + digest([plan["recipe_sha256"], plan["output"]])[:16]

def submit_command(plan, python, indices, *, after_arrays=()):
    command = ["sbatch", "--parsable", "--job-name=paper3-followup", "--account=zhuziyuan",
        "--partition=hpc_gpu1", "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(plan["cpus"]),
        "--mem=" + plan["memory_per_gpu"], "--gres=gpu:NVIDIAA100-SXM4-80GB:1", "--exclude=gpu3",
        "--time=5-03:00:00", "--signal=B:USR1@300", "--requeue", "--chdir=/tmp", "--open-mode=append",
        "--array=" + ",".join(map(str, indices)) + "%" + str(plan["gpus"]),
        "--output=" + str(Path(plan["output"]) / "logs/train-%A_%a.out"),
        "--error=" + str(Path(plan["output"]) / "logs/train-%A_%a.err"),
        "--comment=" + job_marker(plan)]
    if after_arrays:
        # A retry array may be queued while siblings run, but starts after the
        # old arrays finish so the experiment keeps its global three-GPU cap.
        command += ["--dependency=afterany:" + ":".join(sorted(set(after_arrays)))]
    return command + [str(PACKAGE / "train.sbatch"), plan["root"], str(python), str(PACKAGE),
        str(Path(plan["output"]) / "plan.json")]

def active_array_tasks(queue_text, marker):
    arrays, indices = set(), set()
    for row in queue_text.splitlines():
        job, separator, comment = row.partition("|")
        if not separator or comment.strip() != marker:
            continue
        match = re.fullmatch(r"(\d+)_(\d+)", job.strip())
        if match is None:
            raise ValueError("Unexpected expanded Slurm array ID: " + job)
        arrays.add(match[1]); indices.add(int(match[2]))
    return arrays, indices

def completed_task(path):
    try:
        value = read(path)
        return isinstance(value, dict) and value.get("status") == "COMPLETE"
    except (OSError, ValueError, TypeError):
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/public/home/zhuziyuan/projects/paper3")
    p.add_argument("--output")
    p.add_argument("--python", default="/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python")
    p.add_argument("--gpus", type=int, default=3); p.add_argument("--cpus", type=int, default=24)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    if args.gpus < 1 or args.cpus < 1:
        p.error("Positive resource counts required")
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else root / "runs/followup_20261023_v2"
    output.mkdir(parents=True, exist_ok=True); (output / "logs").mkdir(exist_ok=True)
    lock = (output / "submit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    plan = make_plan(root, output, args.gpus, args.cpus)
    if (output / "plan.json").exists() and read(output / "plan.json")["recipe_sha256"] != plan["recipe_sha256"]:
        raise SystemExit("Experiment recipe changed. Use --output with a new directory; previous results remain reusable.")
    put(output / "plan.json", plan)
    print(json.dumps({"status": "PREPARED", "tasks": len(plan["tasks"]), "gpus": args.gpus,
                      "cpus_per_gpu": args.cpus, "memory_per_gpu": "128G", "output": str(output)}, ensure_ascii=False), flush=True)
    if not args.submit:
        return
    queue = subprocess.run(["squeue", "--array", "-h", "-u", os.environ.get("USER", "zhuziyuan"), "-o", "%i|%k"],
                           capture_output=True, text=True, timeout=20, check=True)
    marker = job_marker(plan)
    running, active_indices = active_array_tasks(queue.stdout, marker)
    indices = [i for i, task in enumerate(plan["tasks"]) if i not in active_indices
               and not completed_task(output / task["key"] / "result.json")]
    if running:
        print("EXISTING_ARRAY_REUSED " + ",".join(sorted(set(running))), flush=True)
    if indices:
        # sbatch is invoked exactly once; a lost acknowledgement is reconciled on a later invocation.
        command = submit_command(plan, args.python, indices, after_arrays=running)
        put(output / "submission_intent.json", {"indices": indices, "utc": time.time(), "comment": marker,
                                               "after_arrays": sorted(running)})
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise SystemExit("sbatch failed: " + result.stderr.strip())
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise SystemExit("Submission acknowledgement unclear; run squeue before resubmitting")
        put(output / "submission.json", {"job": job, "indices": indices, "comment": marker,
                                        "after_arrays": sorted(running)})
        print("SUBMITTED_ARRAY " + job, flush=True)
    credentials = PACKAGE / "private/github.json"
    monitor_log = (output / "monitor.log").open("ab", buffering=0)
    subprocess.Popen([args.python, "-I", "-S", "-B", "-u", str(PACKAGE / "entry.py"), str(root), "monitor",
        "--plan", str(output / "plan.json"), "--credentials", str(credentials)], stdin=subprocess.DEVNULL,
        stdout=monitor_log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print("MONITOR_STARTED " + str(output / "monitor.log"), flush=True)

if __name__ == "__main__":
    main()
