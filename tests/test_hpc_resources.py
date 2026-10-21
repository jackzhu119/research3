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
