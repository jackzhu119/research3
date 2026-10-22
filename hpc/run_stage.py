#!/usr/bin/env python3
"""School Slurm adapter around the original protocol-bound local producers.

No environment installer, remote server access, external query, or implicit
network operation is used by test/prepare/pilot/train. Backup is a separate job
or an explicit handoff flush on a network-permitted transfer host.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.environment import EnvironmentError, inspect_environment, verify_code


class StageError(ValueError):
    pass


class CleanupUnconfirmed(StageError):
    pass


def guard(root, stage, *, environ=None, hostname=None):
    env = os.environ if environ is None else environ
    host = hostname or socket.gethostname()
    node = env.get('SLURMD_NODENAME') or host
    if not re.fullmatch(r'\d+', env.get('SLURM_JOB_ID', '')):
        raise StageError('sbatch_compute_allocation_required')
    if any(name.lower().split('.')[0].startswith('login') for name in (host, node)):
        raise StageError('login_node_execution_forbidden')
    root = Path(os.path.abspath(root))
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise StageError('existing_real_project_directory_required')
    # Venv bin/python can legitimately be a symlink to its base interpreter.
    # Do not repeat the portable RunPod installer's stricter ownership rule.
    if env.get('PAPER3_HPC_ROOT') and Path(env['PAPER3_HPC_ROOT']) != root:
        raise StageError('allocation_project_identity_mismatch')
    counts = (env.get('SLURM_GPUS', ''), env.get('SLURM_GPUS_ON_NODE', ''))
    gpu_count = any(re.fullmatch(r'\d+', n) and int(n) > 0 for n in counts)
    # JOB_GPUS contains physical IDs, so "0" is one allocated GPU; the
    # *_ON_NODE counters instead use zero to mean no allocated devices.
    gpu_ids = re.fullmatch(r'(?:\d+(?:-\d+)?|GPU-[A-Za-z0-9-]+)(?:,(?:\d+(?:-\d+)?|GPU-[A-Za-z0-9-]+))*',
                           env.get('SLURM_JOB_GPUS', '')) is not None
    if stage in {'pilot', 'train'} and not (gpu_count or gpu_ids):
        raise StageError('Slurm_GPU_allocation_required')
    return root


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.stage-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def owner_lock(root):
    """Same deployment owner lock as the retained RunPod entry; no double fit."""
    paths = [root / '.paper3-hpc/controller.lock', root / '.paper3-deployment/owner.lock']
    opened = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise StageError('controller_lock_symlink_forbidden')
            stream = path.open('a')
            opened.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StageError('existing_controller_still_owns_project') from None
        yield
    finally:
        for stream in reversed(opened):
            stream.close()


@contextmanager
def stop_signals():
    event = threading.Event()
    previous = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, lambda *_: event.set())
        yield event
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def child(root, relative, args, output, stop, *, cooperative=False):
    if stop.is_set():
        return 3
    command = [sys.executable, '-u', str(root / relative), '--root', str(root), *args]
    print('Paper3 stage command: ' + relative, flush=True)
    with output.open('ab', buffering=0) as stream:
        proc = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, close_fds=True)
        forwarded = None
        heartbeat = started = time.monotonic()
        try:
            while proc.poll() is None:
                if stop.is_set() and forwarded is None:
                    # Controllers handle USR1 and signal only their own ACKed fit group.
                    proc.send_signal(signal.SIGUSR1 if cooperative else signal.SIGTERM)
                    forwarded = time.monotonic()
                if forwarded is not None and time.monotonic() - forwarded > 145:
                    proc.kill()
                    raise CleanupUnconfirmed('coordinator_cleanup_unconfirmed_preserve_claims_and_ledgers')
                if time.monotonic() - heartbeat >= 30:
                    observation = {'status': 'OWNED_COORDINATOR_RUNNING_NOT_COMPLETION',
                        'entry': relative, 'elapsed_seconds': round(time.monotonic() - started),
                        'log': str(output.relative_to(root)), 'log_bytes': output.stat().st_size}
                    if relative == 'hpc/startup_probe.py':
                        observation['startup_probe'] = startup_summary(output.parent / 'startup.json')
                    print(json.dumps(observation), flush=True)
                    heartbeat = time.monotonic()
                time.sleep(.2)
            if proc.returncode < 0 and cooperative:
                raise CleanupUnconfirmed('coordinator_killed_cleanup_unconfirmed_preserve_claims_and_ledgers')
            return proc.returncode
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()


def startup_summary(path):
    """Expose bounded diagnostic fields in the Slurm log, never raw output."""
    unavailable = {'status': 'STARTUP_REPORT_NOT_AVAILABLE'}
    try:
        if path.is_symlink() or path.stat().st_size > 128 * 1024:
            return unavailable
        value = json.loads(path.read_bytes())
        statuses = {'IMPORT_PROBES_RUNNING_NOT_COMPLETE', 'IMPORT_PROBES_INTERRUPTED_NOT_COMPLETE',
                    'IMPORT_PROBES_COMPLETE', 'IMPORT_PROBES_INCOMPLETE'}
        if (not isinstance(value, dict) or value.get('schema_version') != 'paper3-hpc-startup-import-probe-v1'
                or value.get('status') not in statuses):
            return unavailable
        from hpc.startup_probe import MODULES
        names = {name for name, _ in MODULES}
        observations = value.get('modules', [])
        if not isinstance(observations, list) or len(observations) > len(names):
            return unavailable
        rows = []
        for row in observations:
            if (not isinstance(row, dict) or row.get('module') not in names
                    or row.get('status') not in {'IMPORT_COMPLETE', 'IMPORT_FAILED',
                        'IMPORT_PROBE_TIMEOUT', 'IMPORT_PROBE_INVALID_RESULT'}):
                return unavailable
            safe = {'module': row['module'], 'status': row['status']}
            for key in ('process_wall_seconds', 'import_wall_seconds', 'user_seconds', 'system_seconds'):
                number = row.get(key)
                if type(number) in (int, float) and math.isfinite(number) and number >= 0:
                    safe[key] = round(number, 3)
            for key in ('error_type', 'missing_module'):
                text = row.get(key)
                if isinstance(text, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]{0,159}', text):
                    safe[key] = text
            rows.append(safe)
        active = value.get('active_module')
        return {'status': value['status'], 'modules_completed': len(rows), 'modules_total': len(names),
                'active_module': active if active in names else None, 'modules': rows}
    except (OSError, ValueError, TypeError, AttributeError):
        return unavailable


def audit_restore(root, output):
    from tools.hpc_restore_audit import restore
    archives = root / '.hpc/packages'
    if (archives / 'archives').is_dir():
        archives /= 'archives'
    result = restore(root, root / 'hpc/audit_inputs.json', archives, execute=True)
    write(output / 'historical_audit.json', result)
    if result['missing_files']:
        raise StageError('original_historical_audit_incomplete')
    return result


def tested(root, identity, environment):
    path = root / '.paper3-hpc/software_test_passed.json'
    if not path.is_file():
        raise StageError('successful_school_Slurm_software_test_required')
    value = json.loads(path.read_text())
    if (value.get('returncode') != 0 or value.get('software_identity') != identity
            or value.get('environment', {}).get('modules') != environment['modules']
            or value.get('environment', {}).get('python') != environment['python']):
        raise StageError('software_test_code_or_existing_environment_changed')


def verify_pilot(root):
    from tools.run_controlled_pilot import verify_inputs
    from p3work.source_training import verify_engineering_pilot
    cache = verify_inputs(root)
    return verify_engineering_pilot(cache, 'runs/controlled_source_pilot_seed20261018/engineering_receipt.json')


def coordinator_result(root, relative, output, code, stop):
    """Summarize the actual child exit without echoing arbitrary log contents.

    Coordinator logs retain their original bytes.  Only known status/type
    values and exact known control messages enter the public stage receipt.
    This is diagnostic metadata, never production completion authority.
    """
    result = {'entry': relative, 'returncode': code,
              'status': ('COORDINATOR_PAUSED_OR_STOPPED_NOT_COMPLETE' if code == 3 or stop.is_set() else
                         'COORDINATOR_RETURNED_ZERO_INSPECT_ORIGINAL_RECEIPTS' if code == 0 else
                         'COORDINATOR_RETURNED_NONZERO'),
              'log': str(output.relative_to(root)), 'stop_requested': stop.is_set(),
              'scientific_completion_claimed': False}
    statuses = {'BLOCKED_PRESERVE_EXISTING_JOBS', 'PENDING_DATA_GATES',
        'MAINLINE_THROUGH_ONE_INTERNAL_CHECK_COMPLETE', 'PAUSED_SCHEDULER_SIGNAL_SAME_JOB',
        'PAUSED_CAMPAIGN_WALL_LIMIT', 'PAUSED_SAME_JOB_RECOVERY_REQUIRED',
        'PAUSED_REQUESTED_ACTION_BOUND', 'INTERRUPTED_PRESERVE_SAME_JOB'}
    types = {'CampaignError', 'SourceTrainingError', 'SourceExtensionError', 'ValueError', 'OSError',
             'FileNotFoundError', 'RuntimeError', 'KeyError', 'TypeError', 'PermissionError', 'MemoryError'}
    pilot_error = 'Actual completed 200-update engineering pilot with exact resume is required'
    try:
        if output.is_symlink() or not output.is_file() or output.stat().st_size > 256 * 1024:
            return result
        text = output.read_text()
        positions = [0, *(m.end() - 1 for m in re.finditer(r'\n\{', text))][-16:]
        value = None
        for offset in reversed(positions):
            try:
                candidate, end = json.JSONDecoder().raw_decode(text, offset)
            except ValueError:
                continue
            if isinstance(candidate, dict) and not text[end:].strip():
                value = candidate
                break
        if not isinstance(value, dict):
            return result
        if value.get('status') in statuses:
            result['reported_status'] = value['status']
        if value.get('error_type') in types:
            result['error_type'] = value['error_type']
        if value.get('reason') in {pilot_error, 'stage_blocked:SourceTrainingError:' + pilot_error}:
            result['reason_code'] = 'engineering_pilot_evidence_invalid'
    except (OSError, ValueError, TypeError):
        pass
    return result


def run_training(root, stage, output, stop, *, runner=child):
    if os.environ.get('PAPER3_HPC_RUNTIME_PROFILE'):
        from hpc.runtime_profile import prepare_runtime_profile
        from hpc.pilot_prefit import prepare_prefit_resume
        profile = prepare_runtime_profile(root, os.environ['PAPER3_HPC_RUNTIME_PROFILE'])
        prefit = prepare_prefit_resume(root)
        write(output / 'runtime_profile.json', profile)
        write(output / 'pilot_prefit.json', prefit)
        print(json.dumps({'deployment_runtime_profile': profile, 'pilot_prefit': prefit}), flush=True)
    from p3work import training_campaign as campaign
    signal_flags = ['--handle-slurm-signals', '--stop-grace-seconds', '120']
    common = ['--execute', '--user-started-gpu', *signal_flags]
    observations = []

    def launch(entry, args, log):
        code = runner(root, entry, args, log, stop, cooperative=True)
        summary = coordinator_result(root, entry, log, code, stop)
        observations.append(summary)
        write(output / 'training_result.json', {'status': 'COORDINATOR_EXIT_RECORDED_NOT_STAGE_COMPLETION',
              'coordinators': observations, 'scientific_completion_claimed': False})
        print(json.dumps({'coordinator_result': summary}), flush=True)
        return code

    def finish(code):
        write(output / 'training_result.json', {
            'status': ('TRAINING_STAGE_RETURNED_ZERO_INSPECT_ORIGINAL_RECEIPTS' if code == 0 else
                       'TRAINING_STOPPED_OR_BLOCKED_NOT_COMPLETE'),
            'returncode': code, 'coordinators': observations, 'scientific_completion_claimed': False})
        return code

    if stage == 'pilot':
        action = campaign.next_action(root)
        if action is None or action['kind'] != 'pilot':
            verify_pilot(root)  # Real byte/checkpoint verification; never start the next fit.
            return finish(0)
        code = launch('tools/run_training_campaign.py', [*common, '--max-actions', '1'], output / 'campaign.log')
        if code not in (0, 3) or stop.is_set():
            return finish(code or 3)
        verify_pilot(root)
        return finish(0)
    code = launch('tools/run_training_campaign.py', common, output / 'campaign.log')
    if code or stop.is_set():
        return finish(code or 3)
    # Exactly the original launcher order: real mainline completion first, then
    # two independent source-development arms. No W7/external-test command.
    from hpc.stage_data import load_index
    index = load_index(root)
    for arm in ('BASE4_PLUS_YANG', 'BASE4_PLUS_DREYER'):
        for args in (
            ['--arm', arm, '--stage', 'initialize', '--primary-freeze',
             'runs/training_campaign_v1/pre_internal_freeze.json',
             '--extension-manifest', index['extension_manifests'][arm], *common],
            ['--arm', arm, '--stage', 'auto-development', *common],
        ):
            code = launch('tools/run_source_extension_downstream.py', args, output / (arm + '.log'))
            if code or stop.is_set():
                return finish(code or 3)
    return finish(0)


def run(root, stage):
    root = guard(root, stage)
    os.umask(0o077)
    output = root / '.paper3-hpc/jobs' / os.environ['SLURM_JOB_ID']
    output.mkdir(parents=True, exist_ok=True)
    code, result, snapshot_error, safe_snapshot = 2, None, None, True
    frozen_data_verification = None
    diagnostic_summary = None
    with stop_signals() as stop:
        try:
            with owner_lock(root):
                if stage == 'backup':
                    # flush owns the original producer locks itself; run outside
                    # owner_lock below to avoid double flock acquisition.
                    pass
                else:
                    audit_restore(root, output)
                    identity = verify_code(root)
                    environment = inspect_environment(gpu=stage in {'pilot', 'train'})
                    write(output / 'environment.json', environment)
                    if stage == 'test':
                        diagnostic = child(root, 'hpc/startup_probe.py',
                            ['--output', str(output / 'startup.json')], output / 'startup.log', stop)
                        diagnostic_summary = startup_summary(output / 'startup.json')
                        print(json.dumps({'startup_probe': diagnostic_summary}), flush=True)
                        if diagnostic:
                            raise StageError('deployment_startup_probe_failed_inspect_job_startup_log')
                        # Preserve all original tests, including real process-kill cases.
                        with (output / 'pytest.log').open('ab', buffering=0) as stream:
                            proc = subprocess.Popen([sys.executable, '-m', 'pytest', '-q', '-ra'],
                                cwd=root, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, close_fds=True)
                            heartbeat = time.monotonic()
                            while proc.poll() is None:
                                if stop.is_set():
                                    proc.terminate()
                                    break
                                if time.monotonic() - heartbeat >= 30:
                                    print('CPU software tests running; inspect ' + str(output / 'pytest.log'), flush=True)
                                    heartbeat = time.monotonic()
                                time.sleep(.2)
                            try:
                                code = proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.kill(); code = proc.wait()
                        result = {'status': 'ALL_SOFTWARE_TESTS_PASSED' if code == 0 else 'SOFTWARE_TESTS_FAILED_OR_STOPPED',
                                  'returncode': code, 'software_identity': identity, 'environment': environment,
                                  'job_id': os.environ['SLURM_JOB_ID'], 'startup_probe': diagnostic_summary}
                        write(output / 'software_test.json', result)
                        if code == 0:
                            write(root / '.paper3-hpc/software_test_passed.json', result)
                    else:
                        from hpc.stage_data import prepare_data
                        if stage == 'prepare':
                            result = prepare_data(root, mode='materialize',
                                progress=lambda event: print(json.dumps({'data_progress': event},
                                    ensure_ascii=False), flush=True))
                            code = 0 if result.get('status') == 'ready' else 2
                            write(output / 'data_prepared.json', result)
                        else:
                            tested(root, identity, environment)
                            result = prepare_data(root, mode='check',
                                progress=lambda event: print(json.dumps({'data_progress': event},
                                    ensure_ascii=False), flush=True))
                            frozen_data_verification = result
                            write(output / 'frozen_data_verified.json', result)
                            if result.get('status') != 'ready':
                                raise StageError('frozen_data_gate_blocked_inspect_frozen_data_verified_json')
                            code = run_training(root, stage, output, stop)
                            training_path = output / 'training_result.json'
                            result = (json.loads(training_path.read_text()) if training_path.is_file() else
                                      {'status': 'TRAINING_RESULT_UNAVAILABLE_INSPECT_COORDINATOR_LOGS',
                                       'returncode': code, 'scientific_completion_claimed': False})
            if stage == 'backup':
                from hpc.handoff import flush, publish_code
                result = flush(root)
                write(output / 'backup.json', result)
                # Automatic publication uses only already-committed allowlisted code.
                try:
                    publication = publish_code(root)
                except Exception as exc:
                    publication = {'status': 'GITHUB_PUBLICATION_PENDING', 'error_type': type(exc).__name__}
                write(output / 'github_publication.json', publication)
                r2_complete = result.get('uploaded') or result.get('status') == 'NO_PENDING_HANDOFFS'
                code = 0 if r2_complete and publication.get('status') == 'GITHUB_COMMITTED_CODE_REMOTE_SHA_VERIFIED' else 3
        except Exception as exc:
            # Do not serialize arbitrary dependency/SDK exception strings or credentials.
            safe_snapshot = not isinstance(exc, CleanupUnconfirmed)
            reason = str(exc) if isinstance(exc, (StageError, EnvironmentError)) else type(exc).__name__
            if (isinstance(exc, ValueError) and type(exc).__module__ in
                    {'hpc.runtime_profile', 'hpc.pilot_prefit'}
                    and re.fullmatch(r'[a-z][a-z0-9_]{1,159}', str(exc))):
                reason = str(exc)
            result = {'status': 'BLOCKED_PRESERVE_EXISTING_JOBS', 'reason': reason}
            code = 2
        finally:
            if stage in {'pilot', 'train'} and safe_snapshot:
                try:
                    from hpc.handoff import snapshot
                    handoff = snapshot(root, label=stage, job_exit_code=code)
                    write(output / 'handoff.json', handoff)
                except Exception as exc:
                    snapshot_error = type(exc).__name__
                    if code == 0:
                        code = 2
            elif stage in {'pilot', 'train'}:
                snapshot_error = 'SKIPPED_OWNED_WORKER_CLEANUP_UNCONFIRMED_ORIGINAL_CHECKPOINTS_RETAINED'
            receipt = {'schema_version': 'paper3-hpc-stage-receipt-v1', 'stage': stage,
                       'job_id': os.environ['SLURM_JOB_ID'], 'returncode': code,
                       'status': 'STAGE_RETURNED_ZERO_INSPECT_ORIGINAL_RECEIPTS' if code == 0 else 'STOPPED_OR_BLOCKED_NOT_COMPLETE',
                       'result': result, 'snapshot_error_type': snapshot_error,
                       'frozen_data_verification': frozen_data_verification,
                       'startup_probe': diagnostic_summary,
                       'scientific_completion_claimed': False, 'external_query_accessed': False,
                       'finished_utc': datetime.now(timezone.utc).isoformat()}
            write(output / ('stage-' + stage + '.json'), receipt)
            print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('test', 'prepare', 'pilot', 'train', 'backup'), required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.root, args.stage)
    except StageError as exc:
        print(json.dumps({'status': 'REFUSED_BEFORE_ANY_STAGE', 'reason': str(exc)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
