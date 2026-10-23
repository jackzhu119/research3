#!/bin/bash
set -uo pipefail
printf 'TRAIN_PROCESS_STARTED job=%s task=%s node=%s time=%s\n' "${SLURM_JOB_ID:-?}" "${SLURM_ARRAY_TASK_ID:-?}" "${HOSTNAME:-?}" "$(date -u +%FT%TZ)"
project=$1
python=$2
package=$3
plan=$4
unset BASH_ENV ENV PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
child_pid=
forward_stop() { if [ -n "$child_pid" ]; then kill -USR1 "$child_pid" 2>/dev/null || true; fi; }
trap forward_stop USR1 TERM INT
"$python" -I -B -u "$package/entry.py" "$project" worker --plan "$plan" --task "$SLURM_ARRAY_TASK_ID" &
child_pid=$!
while true; do
  wait "$child_pid"
  result=$?
  # A trapped signal interrupts wait. Re-wait even if the child just exited;
  # Bash retains its exit status, including checkpoint exit 75.
  if [ "$result" -lt 128 ]; then break; fi
  if ! kill -0 "$child_pid" 2>/dev/null; then
    wait "$child_pid"
    final_result=$?
    if [ "$final_result" -ne 127 ]; then result=$final_result; fi
    break
  fi
done
if [ "$result" -eq 75 ]; then
  printf 'CHECKPOINT_SAVED requesting Slurm requeue job=%s\n' "$SLURM_JOB_ID"
  if scontrol requeue "$SLURM_JOB_ID"; then exit 0; fi
  printf 'REQUEUE_UNAVAILABLE: rerun START.py --submit; checkpoint is preserved\n'
fi
exit "$result"
