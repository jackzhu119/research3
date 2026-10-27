from __future__ import annotations
import copy
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import types

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import parallel_runtime as parallel

SOURCE = HERE/'vendor'
ORIGINAL_BRIDGE = SOURCE/'kernel_bridge.py'
ORIGINAL_AUDIT = SOURCE/'audited_runtime.py'
_local_native = Path('/workspace/paper3-r11-repo')
NATIVE_ROOT = Path(os.environ.get('PAPER3_R1_NATIVE_ROOT', str(_local_native if _local_native.is_dir() else '/public/home/zhuziyuan/projects/paper3')))
NATIVE_GRID = NATIVE_ROOT/'p3work/method_registry.py'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


worker = load_file('_r1_original_worker_parallel_test', SOURCE/'worker.py')
audit = load_file('_r1_original_audited_runtime_parallel_test', ORIGINAL_AUDIT)
bridge = load_file('_r1_original_bridge_parallel_test', ORIGINAL_BRIDGE)


def profile(uuid='a05fbf1b-aa15-c226-c218-65faf52ae7b3'):
    return dict(python='3.12.3', torch='2.5.1+cu121', numpy='2.1.1', cuda='12.1', cudnn=90100,
        driver_versions=['535.104.12'], device_name='NVIDIA A100-PCIE-40GB', device_uuid=uuid,
        compute_capability=[8, 0], device_total_memory_bytes=42298834944,
        CPU_machine='x86_64', CPU_processor='', torch_CPU_threads=1,
        torch_interop_threads=1, CPU_BLAS_threadpools=[{'internal_api': 'openblas', 'num_threads': 1}],
        autocast=False, tf32=False, deterministic_algorithms=True, neural_device='cuda:0',
        cublas_workspace_config=':4096:8', timing_protocol='original', timing_scope='same GPU host')


def observations():
    return dict(schema_version=parallel.HARDWARE_SCHEMA,
        reference_runtime_profile=profile(), slots={str(i): dict(job_id=str(772906+i), verified=True,
            runtime_profile=profile('%08d-aa15-c226-c218-65faf52ae7b3'%i)) for i in range(3)})


def action(key, kind='development', serial=False):
    return dict(key=key, kind=kind, expected_receipt='runs/results/'+key+'/receipt.json',
                serial=serial, affinity_slot=0, method='B4', job={'configuration_index': 1, 'seed': 20261018})


def test_complete_actual_profiles_all_three_equivalent_eligible():
    value = observations()
    value['slots']['1']['runtime_profile']['device_name'] = 'A100 PCIe 40GB'
    value['slots']['2']['runtime_profile']['driver_versions'] *= 3
    eligible, reasons = parallel.eligible_primary_slots(value, value['reference_runtime_profile'],
        {str(i): str(772906+i) for i in range(3)}, classify=audit.hardware_type)
    assert eligible == [0, 1, 2] and reasons == {}


@pytest.mark.parametrize('change', [
    {'device_name': 'NVIDIA A100-SXM4-40GB'}, {'torch': 'different'},
    {'torch_CPU_threads': 12}, {'driver_versions': ['550.54.14']},
    {'CPU_BLAS_threadpools': [{'internal_api': 'mkl', 'num_threads': 1}]},
])
def test_routing_does_not_fake_hardware_software_thread_or_timing_equivalence(change):
    value = observations(); value['slots']['1']['runtime_profile'].update(change)
    before = copy.deepcopy(value)
    eligible, reasons = parallel.eligible_primary_slots(value, value['reference_runtime_profile'],
        {str(i): str(772906+i) for i in range(3)}, classify=audit.hardware_type)
    assert eligible == [0, 2]
    assert reasons['1'] == 'PRIMARY_RUNTIME_OR_MEASURED_HARDWARE_DIFFERS'
    assert value == before


def test_old_allocation_or_unverified_profile_cannot_authorize_replacement():
    value = observations(); value['slots']['1']['verified'] = False
    eligible, reasons = parallel.eligible_primary_slots(value, value['reference_runtime_profile'],
        {'0': 'different-new-job', '1': '772907', '2': '772908'}, classify=audit.hardware_type)
    assert eligible == [2]
    assert reasons['0'] == 'OBSERVATION_DOES_NOT_BIND_CURRENT_JOB'
    assert reasons['1'] == 'OBSERVATION_NOT_VERIFIED'


def test_parallel_queue_changes_only_operational_affinity_and_keeps_serial_barrier():
    primary = action('primary'); selection = action('selection', 'development_selection', True)
    frontier = dict(phase='DEVELOPMENT_SCREENING', ready_actions=[primary, selection],
        phase_actions=[primary, selection], phase_total=2, blocked=['exact existing dependency'])
    before = copy.deepcopy(frontier)
    routed = parallel.parallel_frontier(frontier, [0, 1, 2])
    assert routed['ready_actions'][0] == dict(primary, affinity_slot=None)
    assert routed['ready_actions'][1] == selection
    assert routed['phase_actions'] == frontier['phase_actions']
    assert frontier == before
    assert routed['blocked'] == before['blocked']
    assert [{k:v for k,v in a.items() if k not in {'serial', 'affinity_slot', 'configuration_receipt_path'}}
        for a in routed['ready_actions']] == [
        {k:v for k,v in a.items() if k not in {'serial', 'affinity_slot', 'configuration_receipt_path'}}
        for a in frontier['ready_actions']]


def test_compatible_subset_is_routed_without_teaching_old_worker_a_new_schema():
    item = action('primary')
    routed = parallel.parallel_frontier(dict(ready_actions=[item]), [0, 2])
    assert [x['affinity_slot'] for x in routed['ready_actions']] == [0, 2]
    assert {x['key'] for x in routed['ready_actions']} == {'primary'}
    assert all(set(x) == set(item) for x in routed['ready_actions'])


def test_existing_original_worker_parents_can_claim_three_distinct_actions_without_restart(tmp_path):
    runtime = tmp_path/'runtime'; runtime.mkdir()
    actions = [action('work-'+str(i)) for i in range(3)]
    routed = parallel.parallel_frontier(dict(ready_actions=actions), [0, 1, 2])
    parallel.put(runtime/'queue.json', dict(status='RUNNING', ready_actions=routed['ready_actions']))
    fake_bridge = types.SimpleNamespace(action_complete=lambda *_: False)
    leases = []
    try:
        for slot in range(3):
            claimed, reason = worker.claim_next(tmp_path, runtime, slot, str(9000+slot), fake_bridge)
            assert reason == 'CLAIMED'
            leases.append(claimed)
        assert len({x[0]['key'] for x in leases}) == 3
        assert {x[1]['slot'] for x in leases} == {0, 1, 2}
        none, reason = worker.claim_next(tmp_path, runtime, 0, '9000', fake_bridge)
        assert none is None and reason == 'WAITING_OWN_LEASE_RECONCILIATION'
    finally:
        for item, claim, lease in leases:
            worker.release_claim(runtime, item, claim); lease.close()


def test_subset_duplicate_routing_rows_still_execute_same_key_once(tmp_path):
    runtime = tmp_path/'runtime'; runtime.mkdir()
    routed = parallel.parallel_frontier(dict(ready_actions=[action('only')]), [0, 2])
    parallel.put(runtime/'queue.json', dict(status='RUNNING', ready_actions=routed['ready_actions']))
    fake_bridge = types.SimpleNamespace(action_complete=lambda *_: False)
    claim, _ = worker.claim_next(tmp_path, runtime, 0, '9000', fake_bridge)
    try:
        second, reason = worker.claim_next(tmp_path, runtime, 2, '9002', fake_bridge)
        assert second is None and reason == 'WAITING_READY_ACTION'
    finally:
        worker.release_claim(runtime, claim[0], claim[1]); claim[2].close()


def test_serial_selection_waits_until_all_original_claims_finish(tmp_path):
    runtime = tmp_path/'runtime'; runtime.mkdir()
    selection = action('selection', 'development_selection', True)
    parallel.put(runtime/'queue.json', dict(status='RUNNING', ready_actions=[selection]))
    parallel.put(runtime/'claims.json', {'other': dict(slot=2, job_id='9002', token='held', serial=False)})
    claimed, reason = worker.claim_next(tmp_path, runtime, 0, '9000', types.SimpleNamespace(action_complete=lambda *_:False))
    assert claimed is None and reason == 'WAITING_READY_ACTION'


def test_stale_progress_never_claims_old_meta_selection_is_running(tmp_path):
    runtime = tmp_path/'runtime'; runtime.mkdir()
    parallel.put(runtime/'claims.json', {'actual': {'slot': 0}})
    state = {'tasks': {'finished': dict(action('finished'), status='RUNNING', error_type='ExitCode-11'),
        'old': dict(action('old'), status='RUNNING'), 'actual': dict(action('actual'), status='RUNNING')}}
    fake_bridge = types.SimpleNamespace(STATUS={'development': 'COMPLETE'}, action_complete=lambda _, a:a['key']=='finished')
    parallel.repair_progress_state(dict(root=tmp_path, runtime=runtime, bridge=fake_bridge), state)
    assert state['tasks']['finished']['status'] == 'COMPLETE'
    assert 'error_type' not in state['tasks']['finished']
    assert state['tasks']['old']['status'] == 'WAITING_DEPENDENCY_OR_SLOT'
    assert state['tasks']['actual']['status'] == 'RUNNING'


def test_completed_claim_reconciliation_never_steals_active_original_lease(tmp_path):
    runtime = tmp_path/'runtime'; runtime.mkdir()
    item = action('finished')
    parallel.put(runtime/'claims.json', {'finished': dict(slot=0, job_id='9000', token='original')})
    parallel.put(runtime/'state.json', {'tasks': {'finished': item}})
    fake_bridge = types.SimpleNamespace(STATUS={'development':'COMPLETE'}, action_complete=lambda *_: True)
    env = dict(root=tmp_path, runtime=runtime, bridge=fake_bridge, original=None)
    folder = runtime/'action-locks'; folder.mkdir()
    lease = (folder/(hashlib.sha256(b'finished').hexdigest()+'.lock')).open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert parallel.reconcile_completed_claims(env) == []
    lease.close()
    assert parallel.reconcile_completed_claims(env) == ['finished']
    assert parallel.read_json(runtime/'claims.json') == {}


@pytest.fixture
def operational_package(tmp_path, monkeypatch):
    root = tmp_path/'project'; root.mkdir()
    runtime = root/parallel.OPERATIONAL; runtime.mkdir(parents=True)
    package = tmp_path/'package'; package.mkdir()
    for name in ('START.py', 'worker.py', 'scheduler.py', 'handoff.py', 'retire_legacy.py'):
        shutil.copyfile(SOURCE/name, package/name)
    shutil.copyfile(ORIGINAL_BRIDGE, package/'kernel_bridge.py')
    shutil.copyfile(ORIGINAL_AUDIT, package/'audited_runtime.py')
    for name in ('kernel_bridge', 'scheduler', 'handoff', 'retire_legacy', 'audited_runtime'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    launch = dict(root=str(root), package=str(package), python=sys.executable,
        worker_command=[sys.executable, str(package/'worker.py'), '--root', str(root)],
        campaign_identity='original-campaign-v1', partition='hpc_gpu', gres='gpu:1',
        cpus=12, memories=['43G', '43G', '42G'], submit_hold=True)
    marker = 'paper3r1p-'+hashlib.sha256(json.dumps([str(runtime), 'original-campaign-v1'], separators=(',', ':')).encode()).hexdigest()[:16]
    state = dict(marker=marker, slots={str(i): dict(attempts=[dict(slot=i, sequence=1,
        marker=marker, job_id=str(9000+i), name=marker+'-s%d-a1'%i, status='ACKNOWLEDGED')],
        next_submit_at=0, rejections=0) for i in range(3)}, created_at=1)
    parallel.put(runtime/'launch.json', launch); parallel.put(runtime/'scheduler-state.json', state)
    parallel.put(runtime/'HANDOFF_READY.json', dict(status='READY'))
    parallel.put(runtime/'state.json', dict(tasks={}))
    yield root, runtime, package, marker
    for name in ('kernel_bridge', 'scheduler', 'handoff', 'retire_legacy', 'audited_runtime'):
        sys.modules.pop(name, None)


def test_original_scheduler_adopts_existing_three_jobs_zero_duplicate_submit(operational_package, tmp_path):
    root, runtime, package, marker = operational_package
    env = parallel.load_operations(root, package)
    env['bridge'].plan_ready = lambda root: dict(phase='DEVELOPMENT_SCREENING',
        ready_actions=[action('a')], phase_actions=[action('a')], phase_total=1,
        phase_completed=0, blocked=[], all_complete=False, mainline_complete=False)
    calls = []
    def runner(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == 'squeue':
            return types.SimpleNamespace(returncode=0, stdout='\n'.join('%s|%s|RUNNING|%s-s%d-a1|gpu1|gpu1'%(9000+i,marker,marker,i) for i in range(3)), stderr='')
        if cmd[0] == 'sacct':
            return types.SimpleNamespace(returncode=0, stdout='', stderr='')
        pytest.fail('No submission/cancel expected: '+str(cmd))
    # Controller imports same recognized package; don't reload second module state.
    original_loader = parallel.load_operations
    parallel.load_operations = lambda *args: env
    try:
        before = (runtime/'launch.json').read_bytes()
        out = parallel.controller_loop(root, package, tmp_path/'work', submit=True, runner=runner, once=True)
        assert out['status'] == 'RUNNING'
        assert len(out['scheduler']['jobs']) == 3
        assert not [cmd for cmd in calls if cmd[0] in ('sbatch', 'scancel', 'scontrol')]
        assert (runtime/'launch.json').read_bytes() == before
    finally:
        parallel.load_operations = original_loader


def test_corrupted_existing_scheduler_history_never_resets(operational_package):
    root, runtime, package, marker = operational_package
    (runtime/'scheduler-state.json').write_text('{broken')
    with pytest.raises(ValueError):
        parallel.load_operations(root, package)


def test_native_full_frontier_phase_barriers_and_registered_scientific_action_bytes_are_preserved(tmp_path):
    root = tmp_path/'project'; (root/'p3work').mkdir(parents=True)
    shutil.copyfile(NATIVE_GRID, root/'p3work/method_registry.py')
    for path, status in [
        ('runs/controlled_source_pilot_seed20261018/engineering_receipt.json', 'ENGINEERING_PILOT_COMPLETE'),
        ('runs/source_training_v2/source_profile_development.json', 'SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING'),
        ('runs/source_classical_v2/receipt.json', 'SOURCE_CLASSICAL_CORE_COMPLETE'),
    ] + [(bridge.OUTPUT+'/references/seed-%s/receipt.json'%seed, 'SOURCE_REFERENCE_COMPLETE') for seed in bridge.SEEDS]:
        parallel.put(root/path, dict(status=status))
    def complete(item):
        value = dict(status=bridge.STATUS[item['kind']])
        if item['kind'] == 'meta_selection':
            setting = bridge._grid_function(root)(item['method'])[0]
            value.update(dimension=setting.get('dimension', setting.get('rank')), radius=setting['radius'])
        parallel.put(root/item['expected_receipt'], value)
    counts = []
    for expected in (5, 45, 5):
        frontier = bridge.plan_ready(root)
        assert frontier['phase'] == 'META_SCREENING'
        assert len(frontier['ready_actions']) == expected
        # Pure metadata scheduling cannot let target development cross the
        # native meta/selection catalog-freeze barrier.
        assert not [a for a in parallel.parallel_frontier(frontier, [0,1,2])['ready_actions']
                    if a['kind'] == 'development']
        for item in frontier['ready_actions']:
            complete(item)
        counts.append(expected)
    frontier = bridge.plan_ready(root)
    assert frontier['phase'] == 'DEVELOPMENT_SCREENING'
    assert len(frontier['ready_actions']) == 14
    routed = parallel.parallel_frontier(frontier, [0,1,2])
    assert all(x['affinity_slot'] is None for x in routed['ready_actions'])
    before = [{k:v for k,v in a.items() if k not in {'serial','affinity_slot','configuration_receipt_path'}}
              for a in frontier['ready_actions']]
    after = [{k:v for k,v in a.items() if k not in {'serial','affinity_slot','configuration_receipt_path'}}
             for a in routed['ready_actions']]
    assert before == after
    for item in frontier['ready_actions']:
        complete(item)
    frontier = bridge.plan_ready(root)
    assert len(frontier['ready_actions']) == 118
    assert all(x['kind'] == 'development' for x in frontier['ready_actions'])
    assert frontier['total_counts']['development'] == 118
    assert all(x['affinity_slot'] is None for x in parallel.parallel_frontier(frontier,[0,1,2])['ready_actions'])


@pytest.mark.parametrize('failed_observations', [
    {'schema_version': parallel.HARDWARE_SCHEMA, 'reference_runtime_profile': {},
     'slots': {'0': {'job_id': '9000', 'verified': False, 'error_type': 'ProbeTimeout'}}},
    {'schema_version': parallel.HARDWARE_SCHEMA, 'reference_runtime_profile': profile(),
     'slots': {str(i): {'job_id': str(9000+i), 'verified': True,
                       'runtime_profile': dict(profile(), device_name='NVIDIA A100-SXM4-40GB')}
               for i in range(3)}},
])
def test_failed_or_incompatible_nonempty_probes_preserve_original_slot0_route(
        operational_package, tmp_path, monkeypatch, failed_observations):
    root, runtime, package, marker = operational_package
    env = parallel.load_operations(root, package)
    item = action('existing-primary')
    env['bridge'].plan_ready = lambda root: dict(phase='DEVELOPMENT_SCREENING',
        ready_actions=[item], phase_actions=[item], phase_total=1,
        phase_completed=0, blocked=[], all_complete=False, mainline_complete=False)
    work = tmp_path/'repair-work'; work.mkdir()
    parallel.put(work/'hardware-observations.json', failed_observations)
    monkeypatch.setattr(parallel, 'load_operations', lambda *a: env)
    def runner(cmd, **kw):
        if cmd[0] == 'squeue':
            return types.SimpleNamespace(returncode=0, stdout='\n'.join(
                '%s|%s|RUNNING|%s-s%d-a1|gpu1|gpu1'%(9000+i,marker,marker,i)
                for i in range(3)), stderr='')
        if cmd[0] == 'sacct':
            return types.SimpleNamespace(returncode=0, stdout='', stderr='')
        pytest.fail('Probe failure must not trigger additional allocations: '+str(cmd))
    out = parallel.controller_loop(root, package, work, submit=True, runner=runner, once=True)
    queued = parallel.read_json(runtime/'queue.json')
    assert queued['ready_actions'] == [item]
    assert queued['ready_actions'][0]['affinity_slot'] == 0
    policy = queued['parallel_policy']
    assert policy['eligible_primary_slots'] == []
    assert policy['routing_primary_slots'] == [0]
    assert policy['legacy_primary_slot0_fallback'] is True
    assert policy['hardware_equivalence_claimed_for_fallback'] is False
    assert policy['primary_timing_hardware_pooled'] is False
    assert out['parallel_policy'] == policy
    # The unmodified old worker can still claim original work on slot0 while
    # failed hardware discovery has not fabricated eligibility for slot1/2.
    claim, reason = worker.claim_next(root, runtime, 0, '9000', env['bridge'])
    try:
        assert reason == 'CLAIMED'
        assert claim[0] == item
        other, reason = worker.claim_next(root, runtime, 1, '9001', env['bridge'])
        assert other is None and reason == 'WAITING_READY_ACTION'
    finally:
        worker.release_claim(runtime, claim[0], claim[1]); claim[2].close()
