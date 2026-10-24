#!/bin/bash
# Bash 4.2 compatible. Resource requests are supplied by the rolling controller.
set -uo pipefail
printf 'TRAIN_PROCESS_STARTED job=%s task=%s node=%s time=%s\n' "${SLURM_JOB_ID:-?}" "${SLURM_ARRAY_TASK_ID:-?}" "${HOSTNAME:-?}" "$(date -u +%FT%TZ)"
python=$1
package=$2
plan=$3
output=$4
task=${SLURM_ARRAY_TASK_ID:?Missing single-element array index}
unset BASH_ENV ENV PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
child_pid=
stop_reason=none
mkdir -p "$output/runtime/outcomes"
forward_stop() {
  stop_reason=$1
  if [ "$stop_reason" = terminated ]; then
    # A user cancellation must never be interpreted as an automatic resume.
    : > "$output/runtime/task-$task.stop"
  fi
  if [ -n "$child_pid" ]; then kill -USR1 "$child_pid" 2>/dev/null || true; fi
}
trap 'forward_stop time_limit' USR1
trap 'forward_stop terminated' TERM INT
"$python" -I -S -B -u "$package/paper3r3/entry.py" worker --plan "$plan" --task "$task" --device cuda:0 &
child_pid=$!
if [ "$stop_reason" != none ]; then kill -USR1 "$child_pid" 2>/dev/null || true; fi
while true; do
  wait "$child_pid"
  result=$?
  if [ "$result" -lt 128 ]; then break; fi
  if ! kill -0 "$child_pid" 2>/dev/null; then
    wait "$child_pid"
    final_result=$?
    if [ "$final_result" -ne 127 ]; then result=$final_result; fi
    break
  fi
done
receipt="$output/runtime/outcomes/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${task}.json"
printf '{"exit_code":%s,"stop_reason":"%s"}\n' "$result" "$stop_reason" > "$receipt.tmp.$$"
mv "$receipt.tmp.$$" "$receipt"
if [ "$result" -eq 75 ]; then
  printf 'CHECKPOINT_SAVED stop_reason=%s; rolling controller reconciles Slurm before resume\n' "$stop_reason"
fi
exit "$result"
