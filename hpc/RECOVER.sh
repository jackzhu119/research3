#!/bin/bash
# Fixed deployment recovery only; scientific stages remain in Slurm.
set +x
set -euo pipefail
umask 077
project=/public/home/zhuziyuan/projects/paper3
python=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
branch=research/paper3-slurm-hpc-20261021
target=7371f0c2c5faa9e7e7ea78063cbe263f3bc7d308
base=9e421817f3ef1dda6c8086054eb8695379b19c15
release=https://raw.githubusercontent.com/jackzhu119/research3/distribution/paper3-hpc-offline-7371f0c
helper_sha=205db8d7a25fdb8dcc7133eefd71cad1536634c4023faba670d51a794fd67c00
bundle_sha=d2098accf4ec6309bf9d812f3887be00e9dd0b03e3984ed213f70a9882fa5b29
bundle_bytes=11516
execute=false
dry=false
failed_test=''
forward=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --root) project="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        --execute) execute=true; forward+=("$1"); shift ;;
        --dry-run) dry=true; forward+=("$1"); shift ;;
        --recover-failed-test) failed_test="$2"; shift 2 ;;
        *) forward+=("$1"); shift ;;
    esac
done
if [ "$execute" != true ] || [ "$dry" = true ]; then
    echo 'PLAN: verified small code bundle -> local Git fast-forward -> Slurm test/prepare/train; no action performed.'
    exit 0
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then echo 'BLOCKED: run recovery on the permitted submission host.' >&2; exit 2; fi
cd -- "$project"
if [ ! -x "$python" ] || [ "$(git rev-parse --show-toplevel)" != "$(pwd -P)" ]; then
    echo 'BLOCKED: existing Paper3 Git root and school Python are required.' >&2; exit 2
fi
if ! git cat-file -e "$base^{commit}"; then
    echo 'BLOCKED: exact existing HPC Git base is missing; retain all caches and request a full offline code bundle.' >&2; exit 2
fi
cache="$project/.hpc/code-updates/$target"
"$python" - "$cache" "$project" "$failed_test" <<'PY'
from pathlib import Path
import json, re, sys
root = Path(sys.argv[2])
sys.path.insert(0, str(root))
from hpc.start import active_jobs, submissions, records, read, owner, staged_owner
from hpc.run_stage import owner_lock
with owner(root), staged_owner(root), owner_lock(root):
    jobs = active_jobs(root, submissions(root))
    failed_test = sys.argv[3]
    allowed = set()
    if failed_test:
        if not re.fullmatch(r'[0-9]+', failed_test):
            raise SystemExit('BLOCKED: an exact failed CPU-test job ID is required')
        stage = read(root / '.paper3-hpc/jobs' / failed_test / 'stage-test.json')
        if (stage.get('job_id') != failed_test or stage.get('stage') != 'test'
                or type(stage.get('returncode')) is not int or stage['returncode'] == 0):
            raise SystemExit('BLOCKED: original failed CPU-test receipt is required')
        for _, attempt in records(root, '.paper3-hpc/start', r'attempt-[a-f0-9]{32}\.json'):
            submitted = attempt.get('submitted_jobs', [])
            if any(r.get('stage') == 'test' and r.get('job_id') == failed_test for r in submitted):
                allowed.update(r.get('job_id') for r in submitted if r.get('stage') in {'prepare', 'train'})
    if any(row['state'] != 'PENDING' or row['job_id'] not in allowed for row in jobs):
        print(json.dumps({'status': 'BLOCKED_BEFORE_CODE_UPDATE_PRESERVE_ACTIVE_JOBS', 'active_jobs': jobs}))
        raise SystemExit(2)
p = Path(sys.argv[1])
if any(q.is_symlink() for q in (p, *p.parents)):
    raise SystemExit('BLOCKED: code cache symlink forbidden')
p.mkdir(parents=True, exist_ok=True)
PY
helper="$cache/code_download.py"
if [ -f "$project/hpc/code_download.py" ] && printf '%s  %s\n' "$helper_sha" "$project/hpc/code_download.py" | sha256sum --check --status; then
    helper="$project/hpc/code_download.py"
fi
if ! [ -f "$helper" ] || ! printf '%s  %s\n' "$helper_sha" "$helper" | sha256sum --check --status; then
    ca=$("$python" -c 'from botocore.httpsession import get_cert_path; print(get_cert_path(True))')
    tmp=$(mktemp "$cache/.downloader.XXXXXX")
    echo 'Fetching small recovery downloader; no scientific data is downloaded.'
    curl --cacert "$ca" --fail --location --retry 4 --retry-delay 2 --connect-timeout 15 --max-time 120 \
        "$release/hpc/code_download.py" -o "$tmp"
    printf '%s  %s\n' "$helper_sha" "$tmp" | sha256sum --check
    mv -- "$tmp" "$helper"
fi
bundle="$cache/code.bundle"
"$python" -u "$helper" --url "$release/hpc/code-update.bundle" --output "$bundle" --bytes "$bundle_bytes" --sha256 "$bundle_sha"
git bundle verify "$bundle"
git fetch "$bundle" "refs/heads/$branch:refs/remotes/origin/$branch"
if git show-ref --verify --quiet "refs/heads/$branch"; then
    git checkout "$branch"
else
    git checkout --no-track -b "$branch" "origin/$branch"
fi
git merge --ff-only "origin/$branch"
if [ "$(git rev-parse HEAD)" != "$target" ]; then
    echo 'BLOCKED: local source differs from the fixed recovery commit; no job submitted.' >&2; exit 2
fi
if [ -n "$failed_test" ]; then
    if ! "$python" -u hpc/recover_failed_chain.py --root "$project" --failed-test-job "$failed_test" --execute; then
        echo 'Recovery blocked: collecting read-only scheduler details; no new job will be submitted.'
        "$python" -u hpc/recover_failed_chain.py --root "$project" --failed-test-job "$failed_test" --diagnose || true
        exit 2
    fi
fi
exec "$python" -u hpc/start.py --root "$project" --python "$python" --staged-data "${forward[@]}"
