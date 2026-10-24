# Round 3 operational runtime

This package is independent of Round 2. Preparation and runtime state default to the new project directory `runs/followup_20261024_r3`; a different new output directory is supported. The launcher rejects the four known earlier result roots to avoid mixing scientific runs. No controller action edits source projects, cancels jobs, changes existing allocations, installs dependencies, or submits a bulk array.

`START.py` performs lightweight preparation with the standard library: existing project/interpreter paths, unique task keys, immutable scientific plan, and launch configuration. It does not claim an allocation or validate GPU training. `--submit` starts the detached monitor first, then the rolling controller, and immediately prints their PIDs, logs and receipt location. A local PID is a process-start receipt; only `runtime/receipts` identifies an actual scheduler submission. The monitor retains its own lock and continues to observe/report even while quota prevents submission.

```bash
# Prepare only. Substitute the actual root and existing interpreter.
python START.py --root /public/home/zhuziyuan/projects/paper3 \
  --python /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python

# Explicitly launch later, after reviewing the prepared package/plan.
python START.py --root /public/home/zhuziyuan/projects/paper3 \
  --python /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python \
  --credentials /private/path/github.json --submit
```

The defaults follow the configuration handoff: at most three independent submissions, partition `hpc_gpu`, one GPU, 12 CPU cores and 64G host memory per submission. Every `sbatch` uses one original task index (`--array=INDEX`), preserving the plan index and `train-%A_%a` logs. The default GRES is `gpu:1`: the retained school evidence does not establish the exact registered typed 40GB model string. `--gres` accepts a site-confirmed typed request; this package does not invent or silently substitute a model name. Host memory and GPU memory remain different quantities. No claim is made that every Round 3 method fits in 40GB.

The controller writes an fsynced submission intent before calling `sbatch`, then an acknowledgement or explicit rejection receipt. It queries expanded `squeue` and `sacct` records on every cycle and owns only the exact run marker. Job names also contain that marker and the attempt number. A site that does not retain accounting comments can be reconciled through an exact name already present in a durable intent. Accounting uses formatted `JobID`, preserving `PARENT_INDEX`, rather than raw database IDs. A missing or unclear acknowledgement reserves capacity until queue/accounting evidence resolves it. Absence alone never authorizes a duplicate submission. Observation errors suppress new submissions.

Successful `result.json` with `status=COMPLETE` is the scientific completion criterion, including producer/basis tasks with no evaluation rows. Slurm `COMPLETED` alone is insufficient. A complete result frees its slot after the job disappears from `squeue`, including when accounting lags. Both pending and running owned jobs count against the cap. Unresolved submissions also reserve a slot.

Each task may declare `requires: [TASK_KEY, ...]`. A consumer becomes runnable only after all required `result.json` files say `COMPLETE`. Ready siblings continue while consumers wait in `WAITING_INPUT`. If no job or runnable producer remains and all unfinished work waits on unavailable inputs, the controller exits `NOT_COMPLETE`; it never resubmits a failed scientific producer simply to unblock a consumer.

Explicit QoS rejection enters `WAITING_FOR_CAPACITY`, with backoff of 30, 60, 120, 240 and at most 300 seconds. By default this capacity wait can continue as existing school jobs finish. A positive `--submit-reject-limit` can explicitly bound these rejections. Invalid scheduler configuration blocks submission immediately. Socket errors or timeouts are uncertain outcomes, rather than evidence of a rejected request.

Early failures without a worker progress file are treated as an explicitly recorded heuristic for bounded startup retry. `BOOT_FAIL`, `NODE_FAIL` and related transient terminal states share that budget. The default allows two startup retries; failure after progress is a task failure and does not cancel siblings. Worker checkpoint exit 75 resumes the same unfinished task, with a separate default budget of 100 resumptions. No cumulative 12-hour limit is introduced. Slurm's per-allocation time limit remains configurable (`--time`, default `5-03:00:00`).

`train.sbatch` works with Bash 4.2, prints its identity immediately and forwards USR1/TERM to the worker as a checkpoint request. It writes a local exit receipt before returning the worker's status. It performs no automatic Slurm requeue; the controller reconciles the old attempt before submitting another. TERM additionally creates `runtime/task-INDEX.stop`. Explicit Slurm `CANCELLED` and per-task stop files prevent task revival. `runtime/STOP` or output `STOP` prevents further submissions; a controller SIGTERM also records that stop. These controls do not cancel any existing job. A subsequent launcher honors the recorded STOP.

The isolated worker command is:

```text
PYTHON -I -S -B -u PACKAGE/paper3r3/entry.py worker --plan PLAN --task INDEX --device cuda:0
```

The entry point adds only the package, the planned project root and existing interpreter site-package directories. It does not execute `.pth`, `sitecustomize` or `usercustomize`, and does not copy or rebuild the runtime. Before importing a worker it installs signal handlers that set `PAPER3R3_STOP_REQUESTED=1`. The worker remains responsible for checkpoint contents and atomic scientific outputs.

Credentials are supplied as a file path with `--credentials`, `PAPER3_GITHUB_CREDENTIALS`, or existing private-file discovery. No credential contents are embedded in the plan, command, logs or public package. The monitor owns the Round 3 branch/path override and aggregate export policy.

Operational artifacts are `plan.json`, `runtime/launch.json`, `runtime/state.json`, per-attempt `runtime/receipts`, local `runtime/outcomes`, `controller.log`, `monitor.log`, and Slurm logs. Prepared files and local mocked tests are evidence of implementation behavior only. They do not establish current school quota, GPU availability, successful remote submission, scientific results or speed.

Validation includes mocked queue/accounting/submission, receipt recovery, lost acknowledgements, dependency readiness and failed-input termination, indefinite capped QoS waiting, bounded task retries, monitor-before-controller startup, a fresh isolated Python venv, and actual Bash 4.2 signal forwarding. See `VALIDATION.json` for the final combined test count and pinned environment. No school jobs were launched.
