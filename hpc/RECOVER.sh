#!/bin/bash
# Fixed deployment update; completed pilot evidence is checked in Slurm without another pilot fit.
set +x
set -euo pipefail
umask 077
project=/public/home/zhuziyuan/projects/paper3
python=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
branch=research/paper3-slurm-hpc-20261021
target=4d43325fc960dddc94f273fc7d11ea17ed8eec01
base=71cec2d1de75be6db2a0dfa8ca58cdcb3531efe2
release=https://raw.githubusercontent.com/jackzhu119/research3/8ff2ec2ac1d1d86cd65bfd93df31f6656b539844
helper_sha=205db8d7a25fdb8dcc7133eefd71cad1536634c4023faba670d51a794fd67c00
bundle_sha=43099de6562288507c401cf61d74fa6836ae7a04cacf0f56b8aacfbceb855f3f
bundle_bytes=44473
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
    echo 'PLAN: verified small code bundle -> local Git fast-forward -> CPU retest then completed-pilot verification and GPU training; no action performed.'
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
import fcntl, hashlib, json, os, re, ssl, stat, subprocess, sys, tempfile
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
        bundle = safe(cache / 'code.bundle')
        partial = safe(str(bundle) + '.part')
        lock = safe(str(bundle) + '.lock')
        expected = int(bundle_bytes)
        if expected <= 0:
            raise ValueError('fixed_bundle_size_required')
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, 'a') as cache_owner:
            try:
                fcntl.flock(cache_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('curl_bundle_download_already_running') from None
            def regular(path):
                safe(path)
                if path.exists() and not stat.S_ISREG(path.stat().st_mode):
                    raise ValueError('curl_bundle_regular_cache_required')
            def complete(path):
                regular(path)
                return path.is_file() and path.stat().st_size == expected and sha(path) == bundle_sha
            def publish():
                if not complete(partial):
                    raise ValueError('curl_bundle_complete_size_or_SHA256_mismatch')
                if bundle.exists():
                    raise ValueError('curl_bundle_final_cache_appeared')
                os.link(partial, bundle)
                partial.unlink()
                directory = os.open(bundle.parent, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(directory)
                finally: os.close(directory)
                print('VERIFIED: complete curl code package GET/SHA256 passed', flush=True)
            regular(bundle); regular(partial)
            if bundle.exists():
                if not complete(bundle):
                    raise ValueError('curl_bundle_existing_final_cache_SHA256_mismatch')
                print('VERIFIED: existing code package SHA256 passed', flush=True)
            elif partial.exists() and partial.stat().st_size == expected:
                publish()
            else:
                if partial.exists() and partial.stat().st_size > expected:
                    raise ValueError('curl_bundle_partial_larger_than_pinned_object')
                try:
                    from botocore.httpsession import get_cert_path
                    ca = get_cert_path(True)
                except ImportError:
                    ca = ssl.get_default_verify_paths().cafile
                if not ca or not Path(ca).is_file():
                    raise ValueError('curl_bundle_existing_trusted_CA_file_required')
                verified = False
                for attempt in range(1, 7):
                    offset = partial.stat().st_size if partial.exists() else 0
                    print('CURL_DOWNLOADING: attempt %s/6; %s/%s bytes; %s' %
                          (attempt, offset, expected, 'IPv4; connect limit 15s, total limit 60s' if attempt % 2 else 'default address selection; connect limit 15s, total limit 60s'), flush=True)
                    body_fd, body_name = tempfile.mkstemp(prefix='.curl-body-', dir=cache); os.close(body_fd)
                    head_fd, head_name = tempfile.mkstemp(prefix='.curl-headers-', dir=cache); os.close(head_fd)
                    body_path, head_path = Path(body_name), Path(head_name)
                    try:
                        command = ['curl', '--cacert', str(ca), '--fail', '--location', '--silent', '--show-error',
                            '--proto', '=https', '--proto-redir', '=https', '--connect-timeout', '15',
                            '--max-time', '60', '--speed-limit', '1', '--speed-time', '20', '--retry', '0', '--max-filesize', str(expected),
                            '--header', 'Accept-Encoding: identity', '--dump-header', head_name,
                            '--output', body_name, '--write-out', '%{http_code}']
                        if attempt % 2:
                            command.append('-4')
                        if offset:
                            command.extend(['--header', 'Range: bytes=%s-' % offset])
                        command.append(release + '/hpc/code-update.bundle')
                        try:
                            response = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                                      text=True, check=False, timeout=75)
                            returncode = response.returncode
                        except subprocess.TimeoutExpired:
                            returncode = 28
                        headers_raw = head_path.read_bytes()
                        if len(headers_raw) > 65536:
                            raise ValueError('curl_bundle_response_headers_too_large')
                        headers, status = {}, None
                        for line in headers_raw.decode('iso-8859-1').splitlines():
                            match = re.fullmatch(r'HTTP/\S+\s+(\d{3})(?:\s+.*)?', line.strip())
                            if match:
                                status = int(match.group(1)); headers = {}
                            elif ':' in line:
                                name, value = line.split(':', 1); name = name.strip().lower()
                                if name in headers and name in {'content-length', 'content-range', 'content-encoding'}:
                                    raise ValueError('curl_bundle_duplicate_response_header')
                                headers[name] = value.strip()
                        size = body_path.stat().st_size
                        if returncode in (60, 77):
                            raise ValueError('curl_bundle_trusted_TLS_verification_failed')
                        if status is not None and status not in (200, 206, 408, 429, 500, 502, 503, 504):
                            raise ValueError('curl_bundle_unexpected_HTTP_status')
                        # A failed TLS/transport invocation cannot supply trusted
                        # bytes merely because a body happens to match the pin.
                        # Only success or interrupted reads retain valid ranges.
                        if returncode not in (0, 18, 28) or status not in (200, 206):
                            continue
                        if headers.get('content-encoding', 'identity').lower() != 'identity':
                            raise ValueError('curl_bundle_encoded_body_forbidden')
                        if status == 206:
                            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', headers.get('content-range', ''))
                            if (match is None or tuple(map(int, match.groups())) != (offset, expected - 1, expected)):
                                raise ValueError('curl_bundle_Content_Range_mismatch')
                            remaining, mode = expected - offset, os.O_APPEND
                        else:
                            remaining, mode = expected, os.O_TRUNC
                            if offset:
                                print('CURL_RESTARTING: server ignored Range; replacing this owned partial prefix', flush=True)
                        length = headers.get('content-length')
                        if length is not None and (not length.isdigit() or int(length) != remaining):
                            raise ValueError('curl_bundle_Content_Length_mismatch')
                        if size > remaining:
                            raise ValueError('curl_bundle_received_body_larger_than_pinned_object')
                        # Even curl exit 28/18 can leave a valid partial response.
                        # Retain it only after validating its pinned range/length.
                        if size:
                            descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | mode, 0o600)
                            with os.fdopen(descriptor, 'ab' if mode == os.O_APPEND else 'wb') as output:
                                with body_path.open('rb') as body:
                                    shutil_copy = body.read(65536)
                                    while shutil_copy:
                                        output.write(shutil_copy); shutil_copy = body.read(65536)
                                output.flush(); os.fsync(output.fileno())
                        elif status == 200 and offset and returncode == 0:
                            raise ValueError('curl_bundle_empty_complete_response')
                        regular(partial)
                        current = partial.stat().st_size if partial.exists() else 0
                        print('CURL_RETAINED: %s/%s verified-range bytes (full SHA pending)' % (current, expected), flush=True)
                        if current == expected:
                            publish(); verified = True; break
                    finally:
                        body_path.unlink(missing_ok=True); head_path.unlink(missing_ok=True)
                if not verified:
                    raise ValueError('curl_bundle_retry_limit_preserve_partial_prefix')
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
exec "$python" -u hpc/resume_completed_pilot.py --root "$project" --python "$python" \
    --failed-train-job "$failed_train" --execute ${forward[@]+"${forward[@]}"}
