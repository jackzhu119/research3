"""R1 queue-only parallelism for existing workers; no scientific kernel rewrite.

The old supervisors reread queue.json each time they claim work. A replacement
CPU controller can therefore distribute independent work without restarting any
Slurm allocation or loaded numerical child. Native action identities and phase
barriers remain owned by the original kernel_bridge.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from atomic_io import atomic_bytes

REPAIR_ID = 'paper3-r1-existing-allocation-parallel-v1'
OPERATIONAL = 'runs/training_campaign_parallel_v1'
PRIMARY_PARALLEL_KINDS = frozenset(('calibration_profile', 'development'))
HARDWARE_SCHEMA = 'paper3-r1-parallel-hardware-observations-v1'


def emit(event, **fields):
    value = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    try:
        print(json.dumps(value, ensure_ascii=False), flush=True)
    except OSError:
        pass


def read_json(path, default=None, *, required=False):
    path = Path(path)
    try:
        with path.open('rb') as handle:
            size = os.fstat(handle.fileno()).st_size
            if not 0 <= size <= 32*1024*1024:
                raise ValueError('Operational JSON exceeds size ceiling')
            data = handle.read(size+1)
        value = json.loads(data)
        if required and not isinstance(value, dict):
            raise ValueError('Existing operational record must be a JSON object')
        return value
    except FileNotFoundError:
        if required:
            raise
        return default
    except ValueError:
        # An existing corrupted queue/state must never become an empty campaign.
        if required:
            raise
        return default


def read_state(path, default=None):
    path = Path(path)
    return read_json(path, default, required=path.exists())


def put(path, value):
    return atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         indent=2, allow_nan=False)+'\n').encode())


@contextmanager
def lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield handle


def _option(command, flag):
    if not isinstance(command, list) or command.count(flag) != 1:
        return None
    at = command.index(flag)
    return command[at+1] if at+1 < len(command) else None


def load_operations(root, package=None):
    root = Path(root).expanduser().resolve(); runtime = root/OPERATIONAL
    launch = read_json(runtime/'launch.json', required=True)
    package = Path(package or launch.get('package') or _option(launch.get('worker_command'), '--package') or '').expanduser().resolve()
    required = ('START.py', 'worker.py', 'scheduler.py', 'kernel_bridge.py')
    if not all((package/name).is_file() for name in required):
        raise ValueError('Original started R1 operational package is required')
    if Path(launch.get('root', root)).resolve() != root:
        raise ValueError('Original R1 launch root changed')
    state = read_json(runtime/'scheduler-state.json', required=True)
    wanted = 'paper3r1p-'+hashlib.sha256(json.dumps([str(runtime), launch.get('campaign_identity', 'original-campaign-v1')], separators=(',', ':')).encode()).hexdigest()[:16]
    if state.get('marker') != wanted or set(state.get('slots', {})) != {'0', '1', '2'}:
        raise ValueError('Original exact three-slot scheduler history required')
    for row in state['slots'].values():
        if not isinstance(row.get('attempts'), list) or not row['attempts']:
            raise ValueError('Original retained allocation intents required')
        if any(a.get('marker') != wanted for a in row['attempts']):
            raise ValueError('Original scheduler attempt marker changed')
    handoff = read_json(runtime/'HANDOFF_READY.json', required=True)
    if handoff.get('status') != 'READY':
        raise ValueError('Original R1 allocation handoff has not completed')
    for name in ('kernel_bridge', 'scheduler', 'handoff', 'retire_legacy', 'audited_runtime'):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != package:
            raise ValueError('Another operational package is already imported: '+name)
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location('_paper3_r1_existing_start', package/'START.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scheduler_module = sys.modules[module.ThreeJobScheduler.__module__]
    # Keep exact durable JSON encoding while replacing only fragile temp I/O.
    module.put = put
    module.lock = lock
    scheduler_module._atomic = put
    original_scheduler_read = scheduler_module._read
    def strict_state_read(path, default=None):
        return read_json(path, default, required=Path(path).name == 'scheduler-state.json')
    scheduler_module._read = strict_state_read
    return dict(root=root, runtime=runtime, package=package, launch=launch,
                original=module, bridge=sys.modules['kernel_bridge'], marker=wanted,
                original_scheduler_read=original_scheduler_read)


def _canonical_runtime(profile, classify):
    if not isinstance(profile, dict):
        raise ValueError('Complete runtime profile required')
    # The original classifier checks real UUID, full GPU memory and architecture.
    kind = classify(profile)
    value = copy.deepcopy(profile); value.pop('device_uuid', None)
    value['device_name'] = kind
    if isinstance(value.get('driver_versions'), list):
        value['driver_versions'] = sorted(set(value['driver_versions']))
    return value


def eligible_primary_slots(observations, reference_profile, current_jobs, *, classify=None):
    """Routing only: never authorize a native timing or runtime-policy mismatch.

    Eligibility requires the complete observed primary runtime to match the
    measured reference, apart from physical UUID and official model aliases.
    A receipt from an earlier allocation cannot authorize a replacement job.
    Native scientific runtime comparisons still run inside each action child.
    """
    if classify is None:
        from audited_runtime import hardware_type as classify
    if not isinstance(observations, dict) or observations.get('schema_version') != HARDWARE_SCHEMA:
        return [], {'all': 'MISSING_CURRENT_HARDWARE_OBSERVATIONS'}
    try:
        expected = _canonical_runtime(reference_profile, classify)
    except (ValueError, TypeError, KeyError):
        return [], {'all': 'REFERENCE_PRIMARY_RUNTIME_NOT_VERIFIED'}
    eligible = []; rejected = {}
    rows = observations.get('slots', {})
    for slot in range(3):
        current = current_jobs.get(str(slot), current_jobs.get(slot))
        row = rows.get(str(slot), {})
        if not current or str(row.get('job_id')) != str(current):
            rejected[str(slot)] = 'OBSERVATION_DOES_NOT_BIND_CURRENT_JOB'; continue
        if row.get('verified') is not True:
            rejected[str(slot)] = 'OBSERVATION_NOT_VERIFIED'; continue
        try:
            actual = _canonical_runtime(row.get('runtime_profile'), classify)
        except (ValueError, TypeError, KeyError):
            rejected[str(slot)] = 'INCOMPLETE_OR_UNSUPPORTED_PRIMARY_RUNTIME'; continue
        if actual != expected:
            rejected[str(slot)] = 'PRIMARY_RUNTIME_OR_MEASURED_HARDWARE_DIFFERS'; continue
        eligible.append(slot)
    return eligible, rejected


def parallel_frontier(frontier, eligible_slots):
    """Copy only operational routing fields; retain native DAG and identities."""
    eligible = sorted(set(eligible_slots))
    if any(type(slot) is not int or slot not in (0, 1, 2) for slot in eligible):
        raise ValueError('Only original three worker slots are allowed')
    value = dict(frontier); ready = []
    for action in frontier.get('ready_actions', []):
        if action.get('kind') not in PRIMARY_PARALLEL_KINDS or action.get('serial'):
            ready.append(dict(action)); continue
        if eligible == [0, 1, 2]:
            ready.append(dict(action, affinity_slot=None))
        else:
            # Existing old supervisors understand one affinity or None. Repeated
            # same-key routing rows remain deduplicated by original claims/leases.
            ready.extend(dict(action, affinity_slot=slot) for slot in eligible)
    value['ready'] = value['ready_actions'] = ready
    value['parallel_policy'] = dict(repair_id=REPAIR_ID, eligible_primary_slots=eligible,
        phase_barriers_preserved=True, native_action_identities_changed=False,
        primary_timing_hardware_pooled=False, serial_selections_preserved=True)
    value['device_policy'] = 'Independent primary actions share only equivalent observed full-runtime slots; original timing observations retained'
    return value


def reconcile_completed_claims(env):
    """Remove only completed, unlocked claims; never reap an unknown live child."""
    module, bridge, runtime, root = (env[k] for k in ('original', 'bridge', 'runtime', 'root'))
    removed = []
    with lock(runtime/'.queue.lock'):
        claims = read_state(runtime/'claims.json', {}) or {}
        state = read_state(runtime/'state.json', {}) or {}
        tasks = state.get('tasks', {})
        for key, claim in list(claims.items()):
            action = tasks.get(key, {})
            if not action.get('expected_receipt') or action.get('kind') not in bridge.STATUS:
                continue
            if not bridge.action_complete(root, action):
                continue
            try:
                with lock(runtime/'action-locks'/(hashlib.sha256(key.encode()).hexdigest()+'.lock'), blocking=False):
                    claims.pop(key); removed.append(key)
            except BlockingIOError:
                continue
        if removed:
            put(runtime/'claims.json', claims)
    return removed


def repair_progress_state(env, state):
    """Avoid phantom RUNNING rows inherited from earlier completed frontiers."""
    bridge, root, runtime = (env[k] for k in ('bridge', 'root', 'runtime'))
    claims = read_state(runtime/'claims.json', {}) or {}
    for key, row in state.get('tasks', {}).items():
        if row.get('expected_receipt') and row.get('kind') in bridge.STATUS and bridge.action_complete(root, row):
            row['status'] = 'COMPLETE'
            row.pop('error_type', None); row.pop('retry_at', None)
        elif row.get('status') == 'RUNNING' and key not in claims:
            row['status'] = 'WAITING_DEPENDENCY_OR_SLOT'
            row['previous_running_record_was_stale'] = True
    return state


def _current_jobs(snapshot):
    return {str(row['slot']): str(row['job_id']) for row in snapshot.get('jobs', [])
            if row.get('job_id') and row.get('status') not in ('FINISHED', 'REJECTED')}


def controller_loop(root, package, work, *, submit=False, runner=None, stop=None,
                    once=False, observations_path=None, scheduler_factory=None, clock=time.time,
                    launch_token=None):
    env = load_operations(root, package); module = env['original']
    runtime = env['runtime']; work = Path(work).expanduser().resolve(); work.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event(); first = True
    observations_path = Path(observations_path or work/'hardware-observations.json')
    launch = dict(env['launch'], submit_hold=False)
    with lock(runtime/'controller.lock', blocking=False):
        scheduler = (scheduler_factory or module.ThreeJobScheduler)(runtime, launch, runner=runner, clock=clock, emit=emit)
        state = read_state(runtime/'state.json', {}) or {}
        while not stop.is_set():
            try:
                # The initial observation can never submit. Later continuation
                # uses only the exact original marker and remaining three slots.
                frontier = env['bridge'].plan_ready(env['root'])
                desired = (False if first or not submit or frontier.get('all_complete') else
                           [1, 2] if frontier.get('mainline_complete') else True)
                snapshot = scheduler.step(keep_workers=desired)
                observed = snapshot.get('controller_status') not in ('OBSERVATION_RETRY', 'EXISTING_SCHEDULER_REUSED')
                if not observed:
                    emit('R1_FRESH_QUEUE_OBSERVATION_PENDING', status=snapshot.get('controller_status'))
                    if once:
                        return dict(status='QUEUE_OBSERVATION_PENDING', scheduler=snapshot)
                    stop.wait(2); continue
                module.reconcile_claims(runtime, snapshot)
                released_claims = reconcile_completed_claims(env)
                observations = read_json(observations_path, {}) or {}
                reference = observations.get('reference_runtime_profile')
                eligible, rejected = eligible_primary_slots(observations, reference, _current_jobs(snapshot))
                # Hardware discovery may publish a nonempty failed-probe
                # record. It is an observability failure, not a new startup
                # gate: retain the original slot0 route when no alternate
                # runtime has been verified. Native timing/runtime checks
                # inside its action child remain mandatory and unchanged.
                # load_operations has already required slot0's retained intent.
                fallback = not eligible
                routing = eligible if eligible else [0]
                routed = parallel_frontier(frontier, routing)
                routed['parallel_policy'].update(
                    eligible_primary_slots=eligible,
                    routing_primary_slots=routing,
                    legacy_primary_slot0_fallback=fallback,
                    hardware_equivalence_claimed_for_fallback=False)
                if fallback:
                    rejected['legacy_fallback'] = 'UNVERIFIED_OR_INCOMPATIBLE_PROBE_RETAIN_ORIGINAL_SLOT0_NATIVE_CHECKS'
                    routed['device_policy'] = 'Original slot0 routing retained; no alternate runtime equivalence claimed; native timing checks remain mandatory'
                routed['parallel_policy']['routing_rejections'] = rejected
                with lock(runtime/'.queue.lock'):
                    put(runtime/'queue.json', dict(status='COMPLETE' if frontier.get('all_complete') else 'RUNNING',
                        phase=frontier['phase'], ready_actions=routed['ready_actions'],
                        retired_slots=[0] if frontier.get('mainline_complete') else [],
                        updated_utc=clock(), parallel_policy=routed['parallel_policy']))
                repair_progress_state(env, state)
                progress = module.publish_progress(env['root'], runtime, routed, snapshot, state)
                progress['parallel_policy'] = routed['parallel_policy']
                put(runtime/'progress.json', progress)
                value = dict(status='RUNNING', repair_id=REPAIR_ID, scheduler=snapshot,
                    parallel_policy=routed['parallel_policy'], completed_claims_reconciled=released_claims,
                    phase=frontier['phase'], updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    original_scientific_sources_changed=False, live_allocations_signalled=0,
                    additional_gpu_jobs_beyond_original_slots=0)
                put(work/'controller-status.json', value)
                put(runtime/'parallel-repair-status.json', value)
                if first:
                    put(work/'controller-ready.json', dict(status='READY', pid=os.getpid(), repair_id=REPAIR_ID,
                        first_slurm_observation_completed=True, queue_observation_successful=True,
                        launch_token=launch_token, original_scheduler_marker=env['marker'],
                        original_worker_parents_reused=True, updated_utc=value['updated_utc']))
                    first = False
                if once:
                    return value
                if frontier.get('all_complete') and not snapshot.get('inflight_jobs'):
                    return dict(status='ALL_ORIGINAL_WORK_COMPLETE', repair_id=REPAIR_ID)
            except OSError as error:
                emit('R1_PARALLEL_OPERATIONAL_IO_RETRY', error_type=type(error).__name__,
                     scientific_results_changed=False, live_allocations_signalled=0)
                if once:
                    raise
            stop.wait(30)
    return dict(status='STOPPED', repair_id=REPAIR_ID)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True); parser.add_argument('--submit', action='store_true')
    parser.add_argument('--hardware-observations', type=Path); parser.add_argument('--launch-token')
    args = parser.parse_args(argv); stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    result = controller_loop(args.root, args.package, args.work, submit=args.submit, stop=stop,
                             observations_path=args.hardware_observations, launch_token=args.launch_token)
    emit('R1_PARALLEL_REPAIR_CONTROLLER_RETURNED', **result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
