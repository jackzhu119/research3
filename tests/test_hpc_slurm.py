"""Slurm submission fixtures only; no school scheduler or scientific run."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper3_hpc_submit', ROOT / 'hpc/submit.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def args(stage='train', **changes):
    values = dict(stage=stage, root=None, python=None, account=None, partition=None,
                  cpus=None, mem_gib=None, gpus=None, time=None, dependency=None)
    values.update(changes)
    return SimpleNamespace(**values)


def config():
    return submit.load_config(ROOT / 'hpc/resources.json')


@pytest.mark.parametrize('stage,cpus,memory,gpus', [
    ('test', 16, 64, 0), ('prepare', 24, 128, 0), ('pilot', 24, 128, 1),
    ('train', 24, 128, 1), ('backup', 8, 32, 0)])
def test_stage_defaults_use_reported_school_resources_without_whole_node(stage, cpus, memory, gpus):
    request = submit.build_request(args(stage), config())
    assert request['resources']['cpus'] == cpus
    assert request['resources']['mem_gib'] == memory
    assert request['resources']['gpus'] == gpus
    assert '--nodes=1' in request['command'] and '--ntasks=1' in request['command']
    assert '--signal=B:USR1@180' in request['command']
    assert request['scheduler_preflight'][1] == '--test-only'
    assert not request['scientific_budget_changed'] and not request['scientific_threads_changed']
    assert not request['school_quota_known']


def test_manual_resource_overrides_are_explicit_argv_not_shell_text():
    root = Path('/public/home/zhuziyuan/my project')
    request = submit.build_request(args(root=root, cpus=16, mem_gib=96, gpus=2,
                                         time='1-02:30:00', dependency='afterok:123:456'), config())
    assert request['resources'] == {'account': 'zhuziyuan', 'partition': 'hpc_gpu1', 'nodes': 1,
                                    'tasks': 1, 'cpus': 16, 'mem_gib': 96, 'gpus': 2, 'time': '1-02:30:00'}
    assert '--gres=gpu:2' in request['command']
    assert '--dependency=afterok:123:456' in request['command']
    assert str(root / 'hpc/job.sbatch') in request['command']
    assert not request['multi_gpu_execution_validated']


@pytest.mark.parametrize('changes', [
    {'cpus': 77}, {'mem_gib': 1008}, {'gpus': 9}, {'gpus': 0}, {'cpus': 0},
    {'mem_gib': -1}, {'partition': 'unknown'}, {'partition': 'cyhpc_1'},
    {'time': '00:00:00'}, {'time': '01:99:00'}, {'time': '1-24:00:00'},
    {'account': 'name;command'}, {'dependency': 'afterok:123;command'}])
def test_invalid_or_unconfirmed_resources_fail_before_sbatch(changes):
    with pytest.raises(submit.SubmissionError):
        submit.build_request(args(**changes), config())


def test_confirmed_quota_is_enforced_without_guessing_unknown_limits():
    value = config()
    value['recommendation_limits']['max_cpus'] = 16
    with pytest.raises(submit.SubmissionError, match='confirmed_school_limit_max_cpus'):
        submit.build_request(args(cpus=24), value)
    assert submit.build_request(args(cpus=16), value)['resources']['cpus'] == 16


def project(tmp_path, monkeypatch):
    root = tmp_path / 'project'; (root / 'hpc').mkdir(parents=True)
    for name in ('job.sbatch', 'run_stage.py'):
        (root / 'hpc' / name).write_text('fixture only\n')
    python = root / 'python'; python.write_text('fixture interpreter\n'); python.chmod(0o700)
    monkeypatch.setattr(submit.shutil, 'which', lambda name: '/fixture/sbatch')
    return submit.build_request(args(root=root, python=python), config())


def test_scheduler_preflight_rejection_never_submits_or_claims_success(tmp_path, monkeypatch):
    request = project(tmp_path, monkeypatch)
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        assert kwargs['stdin'] == subprocess.DEVNULL and kwargs['timeout'] == 60
        return SimpleNamespace(returncode=1, stdout='', stderr='school permission rejected')
    with pytest.raises(submit.SubmissionError, match='test_only_rejected'):
        submit.submit(request, runner=runner)
    assert calls == [request['scheduler_preflight']]
    assert not list(Path(request['project']).glob('.paper3-hpc/submissions/*.json'))


def test_only_real_scheduler_id_creates_submission_receipt_not_training_claim(tmp_path, monkeypatch):
    request = project(tmp_path, monkeypatch)
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='123456;school\n' if '--parsable' in command else '', stderr='')
    receipt = submit.submit(request, runner=runner)
    assert calls == [request['scheduler_preflight'], request['command']]
    assert receipt['job_id'] == '123456'
    assert not receipt['scientific_completion_claimed'] and not receipt['upload_claimed']
    stored = list(Path(request['project']).glob('.paper3-hpc/submissions/*.json'))
    assert len(stored) == 1 and json.loads(stored[0].read_text())['job_id'] == '123456'


def test_unverified_scheduler_output_is_not_success(tmp_path, monkeypatch):
    request = project(tmp_path, monkeypatch)
    def runner(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout='uncertain job response', stderr='')
    with pytest.raises(submit.SubmissionError, match='id_not_verified'):
        submit.submit(request, runner=runner)


def test_dry_run_does_not_invoke_scheduler_or_require_school_paths(monkeypatch, capsys):
    monkeypatch.setattr(submit, 'submit', lambda *a, **kw: pytest.fail('scheduler invoked in dry run'))
    assert submit.main(['train', '--dry-run']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['status'] == 'DRY_RUN_NO_JOB_SUBMITTED'
    assert value['resources']['gpus'] == 1


def test_job_script_requires_slurm_and_denies_login_even_with_forged_job_id():
    script = ROOT / 'hpc/job.sbatch'
    base = {'PATH': '/usr/bin:/bin'}
    missing = subprocess.run(['bash', str(script)], env=base, capture_output=True, text=True)
    assert missing.returncode == 64 and 'requires sbatch' in missing.stderr
    login = subprocess.run(['bash', str(script), '--stage', 'train'],
                            env={**base, 'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'login1'},
                            capture_output=True, text=True)
    assert login.returncode == 64 and 'login node' in login.stderr


def test_job_script_existing_python_only_preserves_slurm_devices_and_thread_caps(tmp_path):
    root = tmp_path / 'project'; root.mkdir()
    python = tmp_path / 'venv/bin/python'; python.parent.mkdir(parents=True)
    # This stand-in prints argv and selected non-secret environment keys; no
    # scientific Python or Slurm command is called by this fixture.
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\nprintf "OMP=%s MKL=%s CVD=%s\\n" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$CUDA_VISIBLE_DEVICES"\n')
    python.chmod(0o700)
    env = {'PATH': '/usr/bin:/bin', 'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'gpu3',
           'CUDA_VISIBLE_DEVICES': '5', 'OMP_NUM_THREADS': '24', 'MKL_NUM_THREADS': '24',
           'PYTHONHOME': '/old/python', 'PYTHONPATH': '/old/package'}
    result = subprocess.run(['bash', str(ROOT / 'hpc/job.sbatch'), '--root', str(root),
                             '--python', str(python), '--stage', 'pilot'], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert str(root / 'hpc/resources.py') in result.stdout
    assert str(root / 'hpc/run_stage.py') in result.stdout
    assert 'OMP=1 MKL=1 CVD=5' in result.stdout
    assert 'pip' not in result.stdout and 'docker' not in result.stdout


def test_job_script_shell_syntax():
    result = subprocess.run(['bash', '-n', str(ROOT / 'hpc/job.sbatch')], capture_output=True)
    assert result.returncode == 0
