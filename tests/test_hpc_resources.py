"""Resource observations use tiny owned commands or injected wait4 fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('paper3_hpc_resources', Path(__file__).resolve().parents[1] / 'hpc/resources.py')
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


def env(**extra):
    return {'SLURM_JOB_ID': '12345', 'SLURMD_NODENAME': 'gpu3', 'SLURM_JOB_PARTITION': 'hpc_gpu1',
            'SLURM_CPUS_PER_TASK': '24', 'SLURM_MEM_PER_NODE': str(128 * 1024),
            'SLURM_GPUS_ON_NODE': '1', **extra}


@pytest.mark.parametrize('value', ['', 'not-a-job', '123;hostname'])
def test_requires_real_slurm_job_shape(value):
    with pytest.raises(resources.ResourceError, match='SLURM_JOB_REQUIRED'):
        resources.allocation(env(SLURM_JOB_ID=value), hostname='gpu3')


@pytest.mark.parametrize('hostname,node', [('login1', 'gpu3'), ('gpu3', 'login1'), ('login1.school', 'gpu3')])
def test_refuses_login_even_when_slurm_fields_exist(hostname, node):
    with pytest.raises(resources.ResourceError, match='LOGIN_NODE_EXECUTION_REFUSED'):
        resources.allocation(env(SLURMD_NODENAME=node), hostname=hostname)


def test_only_allowlisted_allocation_metadata_is_returned():
    value = resources.allocation(env(GH_TOKEN='fixture-github-secret', AWS_SECRET_ACCESS_KEY='fixture-r2-secret'), hostname='gpu3')
    assert value['cpus'] == 24 and value['memory_gib'] == 128 and value['gpus'] == 1
    assert 'fixture-' not in json.dumps(value)
    assert not value['account_quota_verified']


def test_memory_per_cpu_conversion():
    variables = env(SLURM_MEM_PER_CPU='2048', SLURM_CPUS_PER_TASK='16')
    variables.pop('SLURM_MEM_PER_NODE')
    assert resources.allocation(variables, hostname='gpu4')['memory_gib'] == 32


def test_time_limit_is_observed_without_changing_any_budget():
    assigned = resources.allocation(env(SLURM_TIMELIMIT='1-00:00:00'), hostname='gpu3')
    assert assigned['time_limit_seconds'] == 86400
    row = observation(wall=80000, cpu=100)
    row['allocation']['time_limit_seconds'] = 86400
    row['actual_exit_code'] = 0
    result = resources.recommendations(row)
    assert 'without_changing_scientific_budget' in result['time']['action']
    assert not result['time']['automatic_apply']


def test_gpu_id_lists_are_counted_without_recording_ids():
    variables = env(SLURM_JOB_GPUS='2-3')
    variables.pop('SLURM_GPUS_ON_NODE')
    assigned = resources.allocation(variables, hostname='gpu3')
    assert assigned['gpus'] == 2 and '2-3' not in json.dumps(assigned)
    variables.pop('SLURM_JOB_GPUS')
    variables['CUDA_VISIBLE_DEVICES'] = 'GPU-private-identifier'
    assigned = resources.allocation(variables, hostname='gpu3')
    assert assigned['gpus'] == 1 and 'private-identifier' not in json.dumps(assigned)


def test_walltime_never_downsizes_paused_or_failed_stage():
    row = observation(wall=120, cpu=1)
    row['allocation']['requested_wall_seconds'] = 86400
    for code, forwarded in [(2, []), (0, [{'signal': 'SIGUSR1'}])]:
        row['actual_exit_code'], row['signals_forwarded'] = code, forwarded
        value = resources.recommendations(row)
        assert 'candidate_for_manual_walltime_seconds' not in value['time']
        assert 'incomplete_run' in value['time']['action']
    row['actual_exit_code'], row['signals_forwarded'] = 0, []
    assert resources.recommendations(row)['time']['candidate_for_manual_walltime_seconds'] == 600


def test_process_tree_reads_no_command_or_environment(tmp_path, monkeypatch):
    for pid, parent, rss in [(10, 1, 100), (11, 10, 200), (12, 11, 300), (30, 1, 999)]:
        root = tmp_path / str(pid); root.mkdir()
        (root / 'status').write_text(f'Name:\tpython3\nPPid:\t{parent}\nVmRSS:\t{rss} kB\n')
        (root / 'environ').write_text('fixture-secret')
    old = Path.read_text
    def checked(path, *args, **kwargs):
        assert path.name == 'status'
        return old(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', checked)
    pids, memory = resources.process_tree(10, tmp_path)
    assert pids == {10, 11, 12} and memory == 600 * 1024


def test_gpu_sampling_does_not_count_neighbor_jobs():
    def runner(command, **kwargs):
        text = ('GPU-owned,42,1200\nGPU-neighbor,90,75000\n' if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory' in command
                else 'GPU-owned,83,81920\nGPU-neighbor,100,81920\n')
        return SimpleNamespace(returncode=0, stdout=text)
    value = resources.query_gpu({42}, runner)
    assert value['owned_process_memory_mib'] == 1200
    assert value['owned_gpus'] == [{'gpu_uuid': 'GPU-owned', 'device_utilization_percent': 83,
                                   'total_memory_mib': 81920, 'owned_process_memory_mib': 1200}]


def test_absent_gpu_is_not_an_observed_training_failure():
    def unavailable(*args, **kwargs):
        raise FileNotFoundError
    value = resources.query_gpu({42}, unavailable)
    assert not value['available'] and value['owned_gpus'] == []


def observation(wall=120, cpu=2800, rss=120 * 1024**3, samples=10):
    return {'allocation': {'cpus': 24, 'memory_gib': 128, 'gpus': 1, 'partition': 'hpc_gpu1'},
            'measurements': {'wall_seconds': wall, 'cpu_seconds': cpu, 'peak_rss_bytes': rss, 'samples': samples}}


def test_unknown_quotas_never_create_higher_resource_request():
    value = resources.recommendations(observation())
    assert value['cpu']['suggested'] == 24 and value['memory']['suggested_gib'] == 128
    assert not value['quota_limits_supplied']
    assert value['gpu']['suggested'] == 1
    assert not value['scientific_thread_profile_changed']


def test_known_memory_limit_and_physical_node_bound_are_respected():
    value = resources.recommendations(observation(rss=126 * 1024**3),
                                      {'max_cpus': 32, 'max_mem_gib': 144, 'max_gpus': 8})
    assert value['memory']['suggested_gib'] == 144
    assert value['cpu']['candidate_for_manual_benchmark'] == 30
    assert not value['cpu']['automatic_apply'] and value['gpu']['suggested'] == 1


def test_short_fixture_does_not_justify_increase():
    value = resources.recommendations(observation(wall=3, cpu=72, samples=1), {'max_mem_gib': 256, 'max_cpus': 48})
    assert value['memory']['suggested_gib'] == 128
    assert 'candidate_for_manual_benchmark' not in value['cpu']


def test_wait4_metrics_preserve_child_exit_no_false_claims(tmp_path):
    seen = {}
    def popen(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(pid=999999, returncode=None)
    usage = SimpleNamespace(ru_maxrss=5000, ru_utime=2.5, ru_stime=.5)
    ticks = iter([100., 100., 110.])
    value = resources.run_stage(['python3', 'tiny_fixture.py'], tmp_path, 'test', tmp_path / 'reports',
        environ=env(GH_TOKEN='fixture-secret'), hostname='gpu3', popen=popen,
        waiter=lambda pid, flags: (pid, 7 << 8, usage), clock=lambda: next(ticks),
        gpu_query=lambda pids: {'available': False, 'owned_process_memory_mib': 0, 'owned_gpus': []},
        process_query=lambda pid: ({pid}, 6 * 1024 * 1024))
    assert value['actual_exit_code'] == 7 and value['measurements']['cpu_seconds'] == 3
    assert value['measurements']['peak_rss_bytes'] == 6 * 1024 * 1024
    assert seen['start_new_session'] is True and seen['stdin'] == subprocess.DEVNULL
    assert not value['checkpoint_saved_claimed'] and not value['upload_success_claimed']
    assert 'fixture-secret' not in (tmp_path / 'reports/resources.json').read_text()
    assert (tmp_path / 'reports/resource_recommendations.md').is_file()
    assert len(list((tmp_path / 'reports').glob('resources-*.json'))) == 1
    live = json.loads((tmp_path / 'reports/resources.live.json').read_text())
    assert live['actual_exit_code'] is None and 'NOT_A_COMPLETION_RECEIPT' in live['state']


def test_signal_is_forwarded_only_to_direct_owned_stage(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(resources.os, 'kill', lambda pid, sig: sent.append((pid, sig)))
    def waiter(pid, flags):
        signal.getsignal(signal.SIGUSR1)(signal.SIGUSR1, None)
        return pid, 0, SimpleNamespace(ru_maxrss=1, ru_utime=0, ru_stime=0)
    ticks = iter([100., 100., 110.])
    value = resources.run_stage(['python3'], tmp_path, 'pilot', tmp_path / 'reports',
        environ=env(), hostname='gpu3', popen=lambda *args, **kw: SimpleNamespace(pid=999999, returncode=None),
        waiter=waiter, clock=lambda: next(ticks), process_query=lambda pid: ({pid}, 0),
        gpu_query=lambda pids: {'available': False})
    assert sent == [(999999, signal.SIGUSR1)]
    assert value['signals_forwarded'] == [{'signal': 'SIGUSR1', 'direct_owned_child': True}]


def test_actual_tiny_owned_python_process(tmp_path):
    value = resources.run_stage([sys.executable, '-c', 'pass'], tmp_path, 'test', tmp_path / 'report',
        environ=env(), hostname='gpu3', gpu_query=lambda pids: {'available': False}, interval=2)
    assert value['actual_exit_code'] == 0 and value['measurements']['cpu_seconds'] >= 0
    assert value['measurements']['wall_seconds'] > 0


def test_sample_failure_and_interrupted_wait_never_abort_owned_child(tmp_path):
    calls = []
    def broken(*args):
        raise OSError('fixture measurement unavailable')
    def waiter(pid, flags):
        calls.append(pid)
        if len(calls) == 1:
            raise InterruptedError
        return pid, 0, SimpleNamespace(ru_maxrss=1, ru_utime=0, ru_stime=0)
    ticks = iter([100., 100., 100., 110.])
    value = resources.run_stage(['python3'], tmp_path, 'test', tmp_path / 'reports',
        environ=env(), hostname='gpu3', popen=lambda *args, **kw: SimpleNamespace(pid=999999, returncode=None),
        waiter=waiter, clock=lambda: next(ticks), process_query=broken, gpu_query=broken)
    assert value['actual_exit_code'] == 0 and len(calls) == 2
    assert value['measurements']['measurement_errors']['process_tree'] == 1
    assert value['measurements']['measurement_errors']['gpu_sample'] == 1


def test_sacct_rows_are_optional_and_do_not_claim_scientific_completion():
    def runner(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout='12345.batch|COMPLETED|0:0|60|00:00:50|16|1000K|64Gn|\n')
    value = resources.collect_sacct('12345', runner)
    assert value['status'] == 'SACCT_ROWS_OBSERVED' and value['rows'][0]['MaxRSS'] == '1000K'
    assert not value['scientific_completion_claimed']
    missing = resources.collect_sacct('12345', lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=''))
    assert missing['status'] == 'SACCT_NOT_YET_AVAILABLE'


def proc_stat(root, pid, *, counters=(100, 50, 20, 30), start=1, comm='worker name ) with spaces'):
    folder = root / str(pid); folder.mkdir(exist_ok=True)
    fields = ['S'] + ['0'] * 21
    for index, value in zip((11, 12, 13, 14), counters):
        fields[index] = str(value)
    fields[19] = str(start)
    (folder / 'stat').write_text(f'{pid} ({comm}) ' + ' '.join(fields))


def test_cpu_stat_uses_final_comm_parenthesis_and_includes_reaped_child_time(tmp_path, monkeypatch):
    proc_stat(tmp_path, 10, start=100)
    proc_stat(tmp_path, 11, counters=(40, 20, 0, 0), start=110)
    proc_stat(tmp_path, 99, counters=(99999, 0, 0, 0))
    original = Path.read_text
    def allowed(path, *args, **kwargs):
        assert path.name == 'stat' and path.parent.name in {'10', '11'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', allowed)
    sample = resources.process_cpu({10, 11}, tmp_path, clock_ticks=100)
    assert sample['available'] and sample['sampled_tree_cpu_seconds'] == 2.6
    assert sample['process_identities'] == {'10': 100, '11': 110}


@pytest.mark.parametrize('failure', ['missing', 'permission', 'malformed', 'negative'])
def test_partial_or_unreadable_cpu_stat_is_unavailable_never_zero(tmp_path, monkeypatch, failure):
    proc_stat(tmp_path, 10)
    if failure != 'missing':
        proc_stat(tmp_path, 11, counters=(-1, 0, 0, 0) if failure == 'negative' else (1, 0, 0, 0))
    if failure == 'permission':
        original = Path.read_text
        def denied(path, *args, **kwargs):
            if path.parent.name == '11':
                raise PermissionError('software proc fixture')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'read_text', denied)
    elif failure == 'malformed':
        (tmp_path / '11/stat').write_text('broken software fixture')
    sample = resources.process_cpu({10, 11}, tmp_path, clock_ticks=100)
    assert not sample['available'] and sample['sampled_tree_cpu_seconds'] is None
    assert sample['process_identities'] == {}


def cpu_sample(seconds, identities=None):
    return {'available': True, 'sampled_tree_cpu_seconds': seconds,
            'process_identities': {'10': 1} if identities is None else identities,
            'status': 'CPU_COUNTERS_OBSERVED'}


def test_cpu_interval_reports_used_cores_and_fraction_of_allocated_cpus():
    value = resources.cpu_interval(cpu_sample(6.6), cpu_sample(2.6), 4, 2, 24)
    assert value['used_cpu_cores'] == pytest.approx(2)
    assert value['allocated_CPU_util_percent'] == pytest.approx(100 / 12)
    assert value['cpu_sample_status'] == 'CPU_INTERVAL_OBSERVED'
    zero = resources.cpu_interval(cpu_sample(6.6), cpu_sample(6.6), 4, 2, 24)
    assert zero['used_cpu_cores'] == zero['allocated_CPU_util_percent'] == 0
    unknown_allocation = resources.cpu_interval(cpu_sample(6.6), cpu_sample(2.6), 4, 2, None)
    assert unknown_allocation['used_cpu_cores'] == pytest.approx(2)
    assert unknown_allocation['allocated_CPU_util_percent'] is None


@pytest.mark.parametrize('current,previous,now,before,status', [
    (cpu_sample(1), None, 1, None, 'FIRST_SAMPLE'),
    ({'available': False, 'status': 'PROCESS_STAT_UNAVAILABLE'}, cpu_sample(1), 2, 1, 'PROCESS_STAT'),
    (cpu_sample(2), {'available': False}, 2, 1, 'PREVIOUS_SAMPLE'),
    (cpu_sample(4), cpu_sample(2, {'10': 1, '11': 2}), 2, 1, 'PROCESS_SET_CHANGED'),
    (cpu_sample(4, {'10': 1, '11': 2}), cpu_sample(2), 2, 1, 'PROCESS_SET_CHANGED'),
    (cpu_sample(4, {'10': 2}), cpu_sample(2), 2, 1, 'PROCESS_IDENTITY_CHANGED'),
    (cpu_sample(1), cpu_sample(2), 2, 1, 'COUNTER_OR_INTERVAL_CHANGED'),
    (cpu_sample(2), cpu_sample(1), 1, 1, 'COUNTER_OR_INTERVAL_CHANGED')])
def test_uncertain_cpu_intervals_stay_null(current, previous, now, before, status):
    value = resources.cpu_interval(current, previous, now, before, 24)
    assert value['used_cpu_cores'] is value['allocated_CPU_util_percent'] is None
    assert value['cpu_sample_status'].startswith(status)


def test_live_current_values_are_separate_from_peaks_and_final_wait4(tmp_path):
    tick, samples = {'now': 0.}, {'index': -1}
    def process_query(pid):
        samples['index'] += 1
        return {pid}, [10, 20, 12][samples['index']] * 1024**2
    def gpu_query(pids):
        index = samples['index']; memory = [1, 10, 2][index]
        return {'available': True, 'owned_process_memory_mib': memory,
                'owned_gpus': [{'gpu_uuid': 'GPU-owned', 'device_utilization_percent': [20, 90, 25][index],
                                'total_memory_mib': 81920, 'owned_process_memory_mib': memory}]}
    usage = SimpleNamespace(ru_maxrss=5000, ru_utime=9, ru_stime=3)
    value = resources.run_stage(['software-metadata-child'], tmp_path, 'test', tmp_path / 'reports',
        environ=env(), hostname='gpu3', popen=lambda *args, **kwargs: SimpleNamespace(pid=10, returncode=None),
        waiter=lambda pid, flags: (pid, 0, usage) if tick['now'] >= 2 else (0, 0, None),
        process_query=process_query, cpu_query=lambda pids: cpu_sample([1, 3, 4][samples['index']]),
        gpu_query=gpu_query, clock=lambda: tick['now'], sleep=lambda seconds: tick.update(now=tick['now'] + seconds), interval=1)
    live = json.loads((tmp_path / 'reports/resources.live.json').read_text())
    assert live['sampled_tree_cpu_seconds'] == 4 and live['used_cpu_cores'] == 1
    assert live['allocated_CPU_util_percent'] == pytest.approx(100 / 24)
    assert live['sampled_tree_rss_bytes'] == 12 * 1024**2
    assert live['sampled_tree_peak_rss_bytes'] == 20 * 1024**2
    assert live['owned_process_gpu_memory_mib'] == 2 and live['owned_process_gpu_peak_memory_mib'] == 10
    assert live['owned_gpus'][0]['device_utilization_percent'] == 25
    assert 'neighbor_jobs_not_owned_process' in live['gpu_utilization_scope']
    assert value['measurements']['owned_gpu_peaks'][0]['device_utilization_percent'] == 90
    assert value['measurements']['cpu_seconds'] == 12  # Only the actual wait4 fact, not sampled CPU=4.
    assert live['cpu_seconds'] is live['actual_exit_code'] is None


def test_cpu_sampling_failure_never_stops_child_or_replaces_final_wait4_cpu(tmp_path):
    def unavailable(pids):
        raise OSError('software metadata unavailable')
    ticks = iter([100., 100., 110.])
    usage = SimpleNamespace(ru_maxrss=1, ru_utime=2, ru_stime=1)
    value = resources.run_stage(['software-metadata-child'], tmp_path, 'test', tmp_path / 'reports',
        environ=env(), hostname='gpu3', popen=lambda *args, **kwargs: SimpleNamespace(pid=10, returncode=None),
        waiter=lambda pid, flags: (pid, 0, usage), clock=lambda: next(ticks),
        process_query=lambda pid: ({pid}, 100), cpu_query=unavailable,
        gpu_query=lambda pids: {'available': False})
    live = json.loads((tmp_path / 'reports/resources.live.json').read_text())
    assert live['used_cpu_cores'] is live['allocated_CPU_util_percent'] is live['sampled_tree_cpu_seconds'] is None
    assert live['owned_process_gpu_memory_mib'] is None and not live['gpu_observation_available']
    assert value['actual_exit_code'] == 0 and value['measurements']['cpu_seconds'] == 3
    assert value['measurements']['measurement_errors']['cpu_sample'] == 1


def test_actual_owned_metadata_cpu_loop_has_live_rates_without_fit_or_gpu(tmp_path, monkeypatch):
    snapshots = []
    original = resources.write_json
    def record(path, value):
        if Path(path).name == 'resources.live.json':
            snapshots.append(dict(value))
        original(path, value)
    monkeypatch.setattr(resources, 'write_json', record)
    script = 'import time\nend=time.monotonic()+.3\nx=0\nwhile time.monotonic()<end: x+=1\n'
    value = resources.run_stage([sys.executable, '-B', '-c', script], tmp_path, 'test', tmp_path / 'report',
        environ=env(), hostname='gpu3', gpu_query=lambda pids: {'available': False}, interval=.04)
    assert value['actual_exit_code'] == 0 and value['measurements']['cpu_seconds'] > 0
    assert snapshots[0]['used_cpu_cores'] is None
    assert any(row['used_cpu_cores'] is not None and row['used_cpu_cores'] > 0 for row in snapshots)
    assert all(row['owned_process_gpu_memory_mib'] is None for row in snapshots)
    assert value['measurements']['cpu_seconds'] == (value['measurements']['cpu_user_seconds']
                                                   + value['measurements']['cpu_system_seconds'])
    assert not value['checkpoint_saved_claimed'] and not value['upload_success_claimed']
