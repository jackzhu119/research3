"""Deployment-only probe fixtures; never a Slurm submission or scientific fit."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hpc import startup_probe as probe


def allocation():
    return {'SLURM_JOB_ID': '123456', 'SLURMD_NODENAME': 'cpu-fixture',
            'SLURM_JOB_PARTITION': 'cyhpc_1', 'SLURM_CPUS_PER_TASK': '16',
            'SLURM_MEM_PER_NODE': '65536'}


def observation(module):
    return {'status': 'IMPORT_COMPLETE', 'module': module, 'import_wall_seconds': 1.5,
            'user_seconds': .8, 'system_seconds': .3, 'peak_rss_bytes': 100 * 1024**2,
            'numeric_imports': {name: name in ('numpy', 'scipy') for name in probe.NUMERIC_LIBRARIES},
            'numerical_fit_performed': False, 'CUDA_operation_performed': False}


@pytest.mark.parametrize('env,host', [({}, 'cpu-fixture'),
    ({**allocation(), 'SLURMD_NODENAME': 'login1'}, 'cpu-fixture'),
    (allocation(), 'login1')])
def test_probe_refuses_login_or_missing_allocation_before_process_or_output(tmp_path, env, host):
    def forbidden(*args, **kwargs):
        pytest.fail('unguarded startup probe spawned an interpreter')
    with pytest.raises(ValueError):
        probe.run_probe(tmp_path, tmp_path / 'startup.json', environ=env, hostname=host, runner=forbidden)
    assert list(tmp_path.iterdir()) == []


def test_fixed_seven_imports_record_resources_without_adjusting_thread_or_test_budgets(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        assert command[:4] == [sys.executable, '-B', '-c', probe.WORKER]
        assert kwargs == {'cwd': tmp_path, 'stdin': subprocess.DEVNULL, 'stdout': subprocess.PIPE,
                          'stderr': subprocess.PIPE, 'text': True, 'timeout': 180, 'check': False}
        calls.append(command[4])
        return SimpleNamespace(returncode=0, stdout=json.dumps(observation(command[4])), stderr='')

    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    monkeypatch.setenv('R2_SECRET_ACCESS_KEY', 'SOFTWARE_SECRET_MUST_NOT_BE_PRINTED_OR_COPIED')
    result = probe.run_probe(tmp_path, '.paper3-hpc/jobs/123456/startup.json',
                             environ=allocation(), hostname='cpu-fixture', runner=runner)
    assert calls == [name for name, limit in probe.MODULES] and len(calls) == 7
    assert result['status'] == 'IMPORT_PROBES_COMPLETE'
    assert result['numerical_fit_performed'] is result['scientific_fit_budget_changed'] is False
    assert result['original_test_deadlines_changed'] is result['scientific_thread_environment_changed'] is False
    assert result['thread_environment']['OPENBLAS_NUM_THREADS'] == '1'
    assert 'SOFTWARE_SECRET' not in json.dumps(result)
    assert all(row['peak_rss_bytes'] == 100 * 1024**2 and row['user_seconds'] == .8 for row in result['modules'])
    assert json.loads((tmp_path / '.paper3-hpc/jobs/123456/startup.json').read_text()) == result


def test_probe_timeout_is_explicit_incomplete_and_preserves_original_fixture_timeout(tmp_path):
    def slow(command, **kwargs):
        assert kwargs['timeout'] == 180
        raise subprocess.TimeoutExpired(command, 180, output=b'UNEXPECTED_SECRET', stderr=b'UNEXPECTED_SECRET')

    result = probe.run_probe(tmp_path, 'startup.json', environ=allocation(), hostname='cpu-fixture', runner=slow)
    assert result['status'] == 'IMPORT_PROBES_INCOMPLETE'
    assert len(result['modules']) == 7 and all(row['status'] == 'IMPORT_PROBE_TIMEOUT' for row in result['modules'])
    assert [row['related_original_software_fixture_timeout_seconds'] for row in result['modules']] == [20, 5, 20, 20, 20, 20, 30]
    assert all(row['import_metrics_available'] is False for row in result['modules'])
    assert 'UNEXPECTED_SECRET' not in (tmp_path / 'startup.json').read_text()


@pytest.mark.parametrize('change', [{'numeric_imports': []}, {'user_seconds': float('nan')},
                                   {'status': 'UNKNOWN'}, {'numerical_fit_performed': True}])
def test_invalid_child_metrics_never_claim_completed_probe(tmp_path, change):
    def broken(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({**observation(command[4]), **change}), stderr='')

    result = probe.run_probe(tmp_path, 'startup.json', environ=allocation(), hostname='cpu-fixture', runner=broken)
    assert result['status'] == 'IMPORT_PROBES_INCOMPLETE'
    assert all(row['status'] == 'IMPORT_PROBE_INVALID_RESULT' for row in result['modules'])


def test_child_import_error_remains_failed_with_resources(tmp_path):
    def failed(command, **kwargs):
        value = {**observation(command[4]), 'status': 'IMPORT_FAILED',
                 'error_type': 'ModuleNotFoundError', 'missing_module': 'software_missing_dependency'}
        return SimpleNamespace(returncode=2, stdout=json.dumps(value), stderr='')

    result = probe.run_probe(tmp_path, 'startup.json', environ=allocation(), hostname='cpu-fixture', runner=failed)
    assert result['status'] == 'IMPORT_PROBES_INCOMPLETE'
    assert all(row['status'] == 'IMPORT_FAILED' and row['import_metrics_available']
               and row['missing_module'] == 'software_missing_dependency' for row in result['modules'])


def test_probe_report_cannot_escape_project_or_follow_symlinks(tmp_path):
    root = tmp_path / 'project'; root.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (root / 'linked').symlink_to(outside, target_is_directory=True)
    for output in (outside / 'startup.json', root / 'linked/startup.json'):
        with pytest.raises(probe.ProbeError, match='inside_real_project'):
            probe.run_probe(root, output, environ=allocation(), hostname='cpu-fixture',
                            runner=lambda *args, **kwargs: pytest.fail('unsafe output spawned child'))
    assert list(outside.iterdir()) == []


def test_actual_standard_library_child_reports_real_cpu_and_rss_without_numeric_imports(tmp_path):
    # Actual local CPU subprocess, not a forged Slurm allocation or a fit.
    row = probe._probe_module(tmp_path, 'json', 5)
    assert row['status'] == 'IMPORT_COMPLETE' and row['returncode'] == 0
    assert row['process_wall_seconds'] >= row['import_wall_seconds'] >= 0
    assert row['user_seconds'] >= 0 and row['system_seconds'] >= 0 and row['peak_rss_bytes'] > 0
    assert row['numeric_imports'] == dict.fromkeys(probe.NUMERIC_LIBRARIES, False)
    assert row['numerical_fit_performed'] is False and row['CUDA_operation_performed'] is False
    assert list(tmp_path.iterdir()) == []


def test_cli_does_not_probe_from_login_terminal(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    assert probe.main(['--root', str(tmp_path), '--output', str(tmp_path / 'startup.json')]) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'STARTUP_PROBE_BLOCKED'
    assert list(tmp_path.iterdir()) == []
