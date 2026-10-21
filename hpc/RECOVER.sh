#!/bin/bash
# Fixed deployment recovery only; scientific stages remain in Slurm.
set +x
set -euo pipefail
umask 077
project=/public/home/zhuziyuan/projects/paper3
python=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
branch=research/paper3-slurm-hpc-20261021
target=57cf4a4d7b325808b20ec42158f727a4b7109d03
base=461d04bd86b18b95991077f17c3d58c2fd731fc5
release=https://raw.githubusercontent.com/jackzhu119/research3/distribution/paper3-hpc-offline-57cf4a4
helper_sha=205db8d7a25fdb8dcc7133eefd71cad1536634c4023faba670d51a794fd67c00
bundle_sha=c666cb6ef35d6262496fe1fa3a4f7280c5c5ffda408bc405d9fe3777787b75e8
bundle_bytes=24065
execute=false
dry=false
forward=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --root) project="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        --execute) execute=true; forward+=("$1"); shift ;;
        --dry-run) dry=true; forward+=("$1"); shift ;;
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
"$python" - "$cache" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
if any(q.is_symlink() for q in (p, *p.parents)):
    raise SystemExit('BLOCKED: code cache symlink forbidden')
p.mkdir(parents=True, exist_ok=True)
PY
helper="$cache/code_download.py"
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
exec "$python" -u hpc/start.py --root "$project" --python "$python" --staged-data "${forward[@]}"
