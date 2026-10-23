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

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from paper3v2.io import read, put, digest, sha

VARIANTS = ("P", "global", "scalar", "dual", "PCT", "softPCT", "temporal", "dual_temporal")
SEEDS = (20261018, 20261019, 20261020)

def make_plan(root, output, gpus=3, cpus=24):
    scientific = {"version": "followup-20261023-v1", "variants": VARIANTS, "seeds": SEEDS,
        "rank": 4, "radius": .15, "updates": 1000, "inner_steps": 10, "episodes_per_update": 4,
        "validate_every": 50, "checkpoint_every": 10, "budgets": [20, 4, 8, 12, 16]}
    # Operational fixes to logging, submission and checkpoints do not invalidate a fit.
    # Numerical / data-selection changes do create a distinct experiment.
    files = [PACKAGE / "paper3v2" / name for name in ("algorithm.py", "data.py")]
    files += sorted((PACKAGE / "p3work").glob("*.py"))
    recipe = digest({"science": scientific, "code": {str(p.relative_to(PACKAGE)): sha(p) for p in files}})
    # Each wave contains different methods; matching seeds keep comparisons paired.
    tasks = [{"key": f"{variant}-seed{seed}", "seed": seed,
              "config": {"variant": variant, "rank": 4, "radius": .15}}
             for seed in SEEDS for variant in VARIANTS]
    return {**scientific, "root": str(root), "output": str(output), "gpus": gpus, "cpus": cpus,
            "memory_per_gpu": "128G", "recipe_sha256": recipe, "tasks": tasks,
            "seeds": list(SEEDS), "variants": list(VARIANTS)}

def job_marker(plan):
    return "paper3-followup-" + digest([plan["recipe_sha256"], plan["output"]])[:16]

def submit_command(plan, python, indices):
    return ["sbatch", "--parsable", "--job-name=paper3-followup", "--account=zhuziyuan",
        "--partition=hpc_gpu1", "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(plan["cpus"]),
        "--mem=" + plan["memory_per_gpu"], "--gres=gpu:NVIDIAA100-SXM4-80GB:1", "--exclude=gpu3",
        "--time=5-03:00:00", "--signal=B:USR1@300", "--requeue", "--chdir=/tmp", "--open-mode=append",
        "--array=" + ",".join(map(str, indices)) + "%" + str(plan["gpus"]),
        "--output=" + str(Path(plan["output"]) / "logs/train-%A_%a.out"),
        "--error=" + str(Path(plan["output"]) / "logs/train-%A_%a.err"),
        "--comment=" + job_marker(plan),
        str(PACKAGE / "train.sbatch"), plan["root"], str(python), str(PACKAGE),
        str(Path(plan["output"]) / "plan.json")]

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
    output = Path(args.output).resolve() if args.output else root / "runs/followup_20261023"
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
    queue = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "zhuziyuan"), "-o", "%A|%k"],
                           capture_output=True, text=True, timeout=20, check=True)
    marker = job_marker(plan)
    running = [r.split("|")[0] for r in queue.stdout.splitlines() if r.endswith("|" + marker)]
    indices = [i for i, task in enumerate(plan["tasks"]) if not ((output / task["key"] / "result.json").exists()
               and read(output / task["key"] / "result.json").get("status") == "COMPLETE")]
    if running:
        print("EXISTING_ARRAY_REUSED " + ",".join(sorted(set(running))), flush=True)
    elif indices:
        # sbatch is invoked exactly once; a lost acknowledgement is reconciled on a later invocation.
        command = submit_command(plan, args.python, indices)
        put(output / "submission_intent.json", {"indices": indices, "utc": time.time(), "comment": marker})
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise SystemExit("sbatch failed: " + result.stderr.strip())
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise SystemExit("Submission acknowledgement unclear; run squeue before resubmitting")
        put(output / "submission.json", {"job": job, "indices": indices, "comment": marker})
        print("SUBMITTED_ARRAY " + job, flush=True)
    credentials = PACKAGE / "private/github.json"
    monitor_log = (output / "monitor.log").open("ab", buffering=0)
    subprocess.Popen([args.python, "-I", "-B", "-u", str(PACKAGE / "entry.py"), str(root), "monitor",
        "--plan", str(output / "plan.json"), "--credentials", str(credentials)], stdin=subprocess.DEVNULL,
        stdout=monitor_log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print("MONITOR_STARTED " + str(output / "monitor.log"), flush=True)

if __name__ == "__main__":
    main()
