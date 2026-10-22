#!/bin/bash
# Fixed deployment update; the original claimed pilot continues only in Slurm.
set +x
set -euo pipefail
umask 077
project=/public/home/zhuziyuan/projects/paper3
python=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
branch=research/paper3-slurm-hpc-20261021
target=71cec2d1de75be6db2a0dfa8ca58cdcb3531efe2
base=7371f0c2c5faa9e7e7ea78063cbe263f3bc7d308
release=https://raw.githubusercontent.com/jackzhu119/research3/distribution/paper3-hpc-offline-71cec2d1de75
helper_sha=205db8d7a25fdb8dcc7133eefd71cad1536634c4023faba670d51a794fd67c00
bundle_sha=e25cbe772db5b9083406d16c1968fd39441fd924b52b0b3c7534f253d0e9eaac
bundle_bytes=51087
execute=false
dry=false
failed_train=''
forward=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --root) project="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        --execute) execute=true; shift ;;
        --dry-run) dry=true; shift ;;
        --failed-train-job) failed_train="$2"; shift 2 ;;
        --cpus|--mem-gib|--gpus|--time) forward+=("$1" "$2"); shift 2 ;;
        *) echo 'BLOCKED: unknown recovery argument.' >&2; exit 64 ;;
    esac
done
if [ "$execute" != true ] || [ "$dry" = true ]; then
    echo 'PLAN: verified small code bundle -> local Git fast-forward -> CPU retest then same-claim GPU pilot/train; no action performed.'
    exit 0
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then echo 'BLOCKED: use the permitted submission host.' >&2; exit 2; fi
case "$failed_train" in ''|*[!0-9]*) echo 'BLOCKED: --failed-train-job requires the original exact numeric job ID.' >&2; exit 2 ;; esac
cd -- "$project"
if [ ! -x "$python" ] || [ "$(git rev-parse --show-toplevel)" != "$(pwd -P)" ]; then
    echo 'BLOCKED: existing Paper3 Git root and school Python are required.' >&2; exit 2
fi
if ! git cat-file -e "$base^{commit}"; then
    echo 'BLOCKED: the exact existing HPC base is missing; preserve all caches and claims.' >&2; exit 2
fi
# A repeated command with this exact installed source delegates directly to the
# durable resume entry. It may return its two submitted IDs while a job runs.
if [ "$(git rev-parse HEAD)" != "$target" ]; then
    "$python" - "$project" "$python" "$target" "$base" "$branch" "$release" "$helper_sha" "$bundle_sha" "$bundle_bytes" "$failed_train" <<'PY'
from pathlib import Path
import hashlib, json, os, re, subprocess, sys, tempfile
root = Path(sys.argv[1])
python, target, base, branch, release, helper_sha, bundle_sha, bundle_bytes, failed = sys.argv[2:]
sys.path.insert(0, str(root))
from hpc.start import active_jobs, submissions, read, owner, staged_owner, safe
from hpc.run_stage import owner_lock
def run(command):
    subprocess.run(command, stdin=subprocess.DEVNULL, check=True)
def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()
try:
    root = safe(root)
    if (not re.fullmatch(r'[a-f0-9]{40}', target) or not re.fullmatch(r'[a-f0-9]{40}', base)
            or not re.fullmatch(r'[a-f0-9]{64}', helper_sha)
            or not re.fullmatch(r'[a-f0-9]{64}', bundle_sha)
            or not re.fullmatch(r'[0-9]{1,20}', failed) or not bundle_bytes.isdigit()):
        raise ValueError('fixed_recovery_release_identity_required')
    # Keep existing start/data/controller owners throughout code replacement.
    # Release all three BEFORE a new Slurm test can be submitted and run.
    with owner(root), staged_owner(root), owner_lock(root):
        jobs = active_jobs(root, submissions(root))
        if jobs:
            print(json.dumps({'status':'BLOCKED_BEFORE_CODE_UPDATE_PRESERVE_ACTIVE_JOBS','active_jobs':jobs}))
            raise SystemExit(2)
        old = read(root / '.paper3-hpc/jobs' / failed / 'stage-train.json')
        if (old.get('schema_version') != 'paper3-hpc-stage-receipt-v1'
                or old.get('job_id') != failed or old.get('stage') != 'train'
                or type(old.get('returncode')) is not int or old['returncode'] != 2
                or old.get('scientific_completion_claimed') is not False
                or old.get('external_query_accessed') is not False):
            raise ValueError('original_failed_GPU_stage_receipt_required')
        changed = subprocess.run(['git','status','--porcelain','--untracked-files=no'],
                                 stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
        if changed.stdout.strip():
            raise ValueError('tracked_source_edits_preserve_original_no_code_replacement')
        cache = safe(root / '.hpc/code-updates' / target); cache.mkdir(parents=True, exist_ok=True)
        helper = root / 'hpc/code_download.py'
        if not helper.is_file() or sha(helper) != helper_sha:
            helper = cache / 'code_download.py'
        if not helper.is_file() or sha(helper) != helper_sha:
            ca = subprocess.check_output([python,'-c','from botocore.httpsession import get_cert_path; print(get_cert_path(True))'], text=True).strip()
            fd, temporary = tempfile.mkstemp(prefix='.downloader-', dir=cache); os.close(fd)
            temporary = Path(temporary)
            try:
                run(['curl','--cacert',ca,'--fail','--location','--retry','4','--retry-delay','2',
                     '--connect-timeout','15','--max-time','120',release+'/hpc/code_download.py','-o',str(temporary)])
                if sha(temporary) != helper_sha:
                    raise ValueError('fixed_downloader_SHA256_mismatch')
                os.replace(temporary, helper)
            finally:
                temporary.unlink(missing_ok=True)
        bundle = safe(cache / 'code.bundle')
        run([python,'-u',str(helper),'--url',release+'/hpc/code-update.bundle',
             '--output',str(bundle),'--bytes',bundle_bytes,'--sha256',bundle_sha])
        run(['git','bundle','verify',str(bundle)])
        run(['git','fetch',str(bundle),'refs/heads/'+branch+':refs/remotes/origin/'+branch])
        if subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+branch], stdin=subprocess.DEVNULL).returncode == 0:
            run(['git','checkout',branch])
        else:
            run(['git','checkout','--no-track','-b',branch,'origin/'+branch])
        run(['git','merge','--ff-only','origin/'+branch])
        actual = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
        if actual != target:
            raise ValueError('fixed_recovery_commit_mismatch_no_job_submitted')
except (ValueError, OSError, subprocess.SubprocessError) as exc:
    # Fixed local codes only; do not echo network/SDK exceptions or credentials.
    reason = str(exc) if type(exc) is ValueError and re.fullmatch(r'[A-Za-z0-9_]+', str(exc)) else type(exc).__name__
    print(json.dumps({'status':'CODE_RECOVERY_BLOCKED_PRESERVE_ORIGINALS','reason':reason,
                      'new_jobs_submitted':False}), flush=True)
    raise SystemExit(2)
PY
fi
exec "$python" -u hpc/resume_prefit_train.py --root "$project" --python "$python" \
    --failed-train-job "$failed_train" --execute ${forward[@]+"${forward[@]}"}
