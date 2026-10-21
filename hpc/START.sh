#!/bin/bash
# Standalone bootstrap: retain a TTY for hidden input; no pip/root/SSH/training here.
set +x
set -euo pipefail
umask 077

project=/public/home/zhuziyuan/projects/paper3
python=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
branch=research/paper3-slurm-hpc-20261021
repository=https://github.com/jackzhu119/research3.git
forward=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --root) project="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        *) forward+=("$1"); shift ;;
    esac
done
if [ ! -d "$project" ] || [ ! -x "$python" ]; then
    echo 'BLOCKED: existing project and school Python environment are required.' >&2
    exit 2
fi
cd -- "$project"
if [ "$(git rev-parse --show-toplevel)" != "$(pwd -P)" ]; then
    echo 'BLOCKED: this must be the existing Paper3 repository root.' >&2
    exit 2
fi

# A default/plan invocation does not fetch, update or schedule anything.
execute=false
dry_run=false
for argument in "${forward[@]}"; do
    if [ "$argument" = --execute ]; then execute=true; fi
    if [ "$argument" = --dry-run ]; then dry_run=true; fi
done
if [ "$execute" != true ] || [ "$dry_run" = true ]; then
    echo 'PLAN: explicit branch fetch -> existing branch fast-forward -> data gates -> sbatch test -> prepare -> train.'
    echo 'Use --execute --network-host only on a school-permitted transfer/login host.'
    exit 0
fi

delay="${PAPER3_HPC_GIT_RETRY_DELAY:-2}"
case "$delay" in ''|*[!0-9]*) echo 'BLOCKED: retry delay must be an integer.' >&2; exit 2 ;; esac
if [ "$delay" -gt 10 ]; then echo 'BLOCKED: retry delay must be at most ten seconds.' >&2; exit 2; fi
fetched=false
for attempt in 1 2 3; do
    echo "GitHub: fetching the explicit research branch (attempt $attempt/3)."
    # No embedded credentials, force update, TLS bypass, new Git features or
    # reliance on the original clone's possibly single-branch fetch refspec.
    if GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 fetch --no-tags "$repository" \
        "refs/heads/$branch:refs/remotes/origin/$branch"; then
        fetched=true
        break
    fi
    if [ "$attempt" -lt 3 ]; then sleep "$delay"; fi
done
if [ "$fetched" != true ]; then
    echo 'BLOCKED: GitHub fetch failed three times; no data transfer or Slurm job was started.' >&2
    exit 2
fi
if git show-ref --verify --quiet "refs/heads/$branch"; then
    git checkout "$branch"
else
    git checkout --no-track -b "$branch" "refs/remotes/origin/$branch"
fi
git merge --ff-only "origin/$branch"
if [ ! -f hpc/start.py ]; then
    echo 'BLOCKED: the fetched version does not contain the new startup coordinator.' >&2
    exit 2
fi
exec "$python" -u hpc/start.py --root "$project" --python "$python" "${forward[@]}"
