"""Metadata-only school prefit recovery; no CUDA, fitting or scheduler calls."""
import copy
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from hpc import pilot_prefit as prefit
from p3work.contracts import digest


def write(root, relative, value):
    path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / 'school project'; root.mkdir()
    receipt = {'status': 'ENGINEERING_PILOT_FAILED', 'scope': 'ENGINEERING_SOURCE_PILOT_ONLY',
        'training_started': False, 'exact_resume_verified': False, 'user_started_gpu_attested': True,
        'seed': 20261018, 'max_jobs': 1, 'actual_completed_updates': 0,
        'optimizer_updates_dispatched': 0, 'prediction_replay_optimizer_updates': 0,
        'checkpoint_files': {}, 'checkpoint_io_timings': [],
        'failure_type': 'FileNotFoundError',
        'failure': str(FileNotFoundError(2, 'No such file or directory',
                                       str(root / 'audit/gpu_resolved_dependencies.json'))),
        'total_monotonic_wall_seconds': .25, 'started_utc': '2026-10-21T15:45:00+00:00',
        'ended_utc': '2026-10-21T15:45:00.25+00:00'}
    claim = {'scope': 'ENGINEERING_SOURCE_PILOT_ONLY', 'max_jobs': 1,
             'output': prefit.PILOT, 'started_utc': '2026-10-21T15:45:00+00:00'}
    identity = {'scope': 'EXPLICIT_SOFTWARE_METADATA_TEST_DOUBLE', 'source': 'fixed'}
    sha = digest(identity)
    campaign_claim = {'schema_version': 'paper3-mainline-training-campaign-v1', 'identity': identity,
                      'identity_sha256': sha, 'fixed_output': prefit.CAMPAIGN,
                      'claim_must_not_be_deleted_or_renamed_to_retry': True}
    wall = {'schema_version': 'paper3-operational-wall-budget-v1', 'identity_sha256': sha,
            'scope': 'operational_only_no_scientific_identity_change',
            'initial_max_wall_seconds': 43200, 'max_wall_seconds': 43200,
            'durable_wall_seconds': 388.038, 'checkpoint_wall_high_water_seconds': 0,
            'interval': None, 'clock_block': None, 'amendments': [], 'recovery_events': [],
            'closed_intervals': 1, 'next_interval_sequence': 2}
    process = {'schema_version': 'paper3-campaign-owned-stage-group-v1',
               'campaign_identity_sha256': sha, 'action_key': 'pilot', 'pid': 12345, 'pgid': 12345,
               'owner_pid': 12344, 'pid_start_time_ticks': 1, 'boot_id': 'SOFTWARE_OLD_BOOT',
               'absolute_deadline_monotonic': 1800, 'cleanup_verified': True}
    state = {'schema_version': 'paper3-mainline-training-campaign-v1', 'identity_sha256': sha,
             'status': 'INTERRUPTED_PRESERVE_SAME_JOB', 'error_type': 'CampaignError',
             'active': copy.deepcopy(prefit.ACTION), 'active_process': process, 'actions': [],
             'wall_budget': {'identity_sha256': sha, 'wall_seconds': 388.038,
                             'minimum_charged_seconds': 388.038, 'blocked': False,
                             'clock_block': None, 'interval_open': False, 'exhausted': False}}
    for name, value in [('receipt', receipt), ('pilot_claim', claim), ('campaign_claim', campaign_claim),
                        ('campaign_progress', state), ('campaign_wall', wall)]:
        write(root, prefit.FILES[name], value)
    write(root, 'audit/hpc_runtime_profile.json', {'scope': 'SOFTWARE_PROFILE_TEST_DOUBLE'})
    monkeypatch.setenv('PAPER3_HPC_RUNTIME_PROFILE', 'school-cu121-v1')
    monkeypatch.setenv('PAPER3_HPC_ROOT', str(root)); monkeypatch.setenv('PAPER3_HPC_STAGE', 'train')
    monkeypatch.setenv('SLURM_JOB_ID', '812345'); monkeypatch.setenv('SLURM_JOB_GPUS', '0')
    monkeypatch.setenv('SLURMD_NODENAME', 'comput7')
    monkeypatch.setattr('socket.gethostname', lambda: 'comput7')
    return root


def original_bytes(root):
    return {path: (root / path).read_bytes() for path in prefit.FILES.values()}


def alter(root, name, **changes):
    value = json.loads((root / prefit.FILES[name]).read_bytes()); value.update(changes)
    write(root, prefit.FILES[name], value)


def test_inspection_is_pure_readonly_and_reports_existing_charged_seconds(project, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID'); monkeypatch.delenv('PAPER3_HPC_RUNTIME_PROFILE')
    before = {str(p.relative_to(project)): p.read_bytes() for p in project.rglob('*') if p.is_file()}
    result = prefit.inspect_failed_prefit(project)
    assert result['prior_wall_seconds'] == .25
    assert result['campaign_charged_wall_seconds'] == 388.038
    assert result['new_pilot_claim_authorized'] is False
    assert {str(p.relative_to(project)): p.read_bytes() for p in project.rglob('*') if p.is_file()} == before
    assert not (project / '.paper3-hpc').exists()


def test_same_claim_history_preserves_exact_original_bytes_and_budget_and_is_idempotent(project):
    before = original_bytes(project)
    result = prefit.prepare_prefit_resume(project)
    assert result['status'] == 'PREFIT_SAME_CLAIM_RESUME_PREPARED'
    assert prefit.prepare_prefit_resume(project) == result
    for name, descriptor in result['history'].items():
        assert (project / descriptor['path']).read_bytes() == before[prefit.FILES[name]]
    assert original_bytes(project) == before
    context = prefit.consume_prefit_resume(project, project / prefit.PILOT)
    assert context['prior_wall_seconds'] == .25
    assert context['original_started_utc'] == '2026-10-21T15:45:00+00:00'
    assert context['same_pilot_claim_preserved'] is True and context['max_jobs'] == 1
    assert original_bytes(project) == before


def test_consume_uses_bound_old_history_after_parent_dispatches_same_pilot_again(project):
    prefit.prepare_prefit_resume(project)
    alter(project, 'campaign_progress', status='RUNNING_LOCAL_USER_STARTED_GPU',
          active_process={'scope': 'NEW_PARENT_SOFTWARE_DISPATCH'})
    alter(project, 'campaign_wall', durable_wall_seconds=400,
          interval={'scope': 'NEW_ORIGINAL_CONTROLLER_INTERVAL'})
    context = prefit.consume_prefit_resume(project, project / prefit.PILOT)
    assert context['prior_wall_seconds'] == .25
    assert json.loads((project / prefit.FILES['campaign_wall']).read_text())['durable_wall_seconds'] == 400


@pytest.mark.parametrize('name,changes', [
    ('receipt', {'training_started': True}), ('receipt', {'actual_completed_updates': 1}),
    ('receipt', {'actual_completed_updates': False}), ('receipt', {'optimizer_updates_dispatched': 1}),
    ('receipt', {'prediction_replay_optimizer_updates': 1}), ('receipt', {'exact_resume_verified': True}),
    ('receipt', {'seed': 1}), ('receipt', {'max_jobs': 2}), ('receipt', {'failure_type': 'RuntimeError'}),
    ('receipt', {'failure': 'Different missing prerequisite'}),
    ('receipt', {'checkpoint_files': {'engineering_initial.pt': 'a' * 64}}),
    ('receipt', {'checkpoint_io_timings': [{'operation': 'checkpoint_write'}]}),
    ('receipt', {'timing_segments': [{'updates': 1}]}), ('receipt', {'identity_sha256': 'a' * 64}),
    ('receipt', {'total_monotonic_wall_seconds': 1800}),
    ('receipt', {'total_monotonic_wall_seconds': float('nan')}),
    ('pilot_claim', {'output': 'runs/another_pilot'}), ('pilot_claim', {'max_jobs': 2}),
    ('campaign_claim', {'identity_sha256': '0' * 64}),
    ('campaign_progress', {'actions': [{'kind': 'source'}]}),
    ('campaign_progress', {'status': 'RUNNING_LOCAL_USER_STARTED_GPU'}),
    ('campaign_progress', {'active_process': {'cleanup_verified': False}}),
    ('campaign_progress', {'active': {'kind': 'source'}}),
    ('campaign_wall', {'clock_block': {'scope': 'UNKNOWN_CLOCK'}}),
    ('campaign_wall', {'interval': {'scope': 'UNCLOSED_INTERVAL'}}),
    ('campaign_wall', {'durable_wall_seconds': 43200}),
    ('campaign_wall', {'durable_wall_seconds': 0}),
    ('campaign_wall', {'max_wall_seconds': 43201}),
])
def test_any_uncertain_numerical_ownership_or_wall_evidence_blocks_without_writes(project,name,changes):
    alter(project, name, **changes)
    before = original_bytes(project)
    with pytest.raises(prefit.PrefitError):
        prefit.prepare_prefit_resume(project)
    assert original_bytes(project) == before
    assert not (project / prefit.PILOT / 'prefit_history').exists()


@pytest.mark.parametrize('filename', ['identity.json', 'engineering_initial.pt', 'arbitrary.json',
                                     'engineering_predictions.npy'])
def test_any_existing_numerical_or_unknown_output_blocks_before_preparation(project,filename):
    (project / prefit.PILOT / filename).write_bytes(b'SOFTWARE_TEST_DOUBLE_NOT_A_REAL_MODEL')
    with pytest.raises(prefit.PrefitError, match='unknown_output'):
        prefit.prepare_prefit_resume(project)


def test_optional_failed_manifests_are_archived_with_their_original_bytes(project):
    for relative in prefit.OPTIONAL.values():
        write(project, relative, {'scope': 'SOFTWARE_EMPTY_PREFIT_MANIFEST', 'files': []})
    before = {name: (project / relative).read_bytes() for name, relative in prefit.OPTIONAL.items()}
    result = prefit.prepare_prefit_resume(project)
    for name, raw in before.items():
        assert (project / result['history'][name]['path']).read_bytes() == raw
    assert prefit.consume_prefit_resume(project, project / prefit.PILOT)['max_jobs'] == 1


@pytest.mark.parametrize('target', ['receipt', 'pilot_claim', 'campaign_claim', 'history', 'profile'])
def test_changed_evidence_after_proof_blocks_consume_without_repair(project,target):
    result = prefit.prepare_prefit_resume(project)
    if target == 'history':
        path = project / result['history']['receipt']['path']
    elif target == 'profile':
        path = project / 'audit/hpc_runtime_profile.json'
    else:
        path = project / prefit.FILES[target]
    path.write_bytes(path.read_bytes() + b'\n')
    before = path.read_bytes()
    with pytest.raises(prefit.PrefitError):
        prefit.consume_prefit_resume(project, project / prefit.PILOT)
    assert path.read_bytes() == before


def test_proof_cannot_be_consumed_in_another_job_or_for_another_output(project,monkeypatch):
    prefit.prepare_prefit_resume(project)
    with pytest.raises(prefit.PrefitError, match='same_output'):
        prefit.consume_prefit_resume(project, project / 'runs/other')
    monkeypatch.setenv('SLURM_JOB_ID', '812346')
    with pytest.raises(prefit.PrefitError, match='proof_file_unavailable'):
        prefit.consume_prefit_resume(project, project / prefit.PILOT)


@pytest.mark.parametrize('key,value', [
    ('PAPER3_HPC_RUNTIME_PROFILE', 'unregistered'), ('PAPER3_HPC_ROOT', '/elsewhere'),
    ('PAPER3_HPC_STAGE', 'test'), ('SLURM_JOB_ID', ''), ('SLURMD_NODENAME', 'login1')])
def test_non_GPU_or_unregistered_profile_cannot_prepare(project,monkeypatch,key,value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        prefit.prepare_prefit_resume(project)
    assert not (project / prefit.PILOT / 'prefit_history').exists()


def test_symlink_original_history_or_output_is_refused(project):
    receipt = project / prefit.FILES['receipt']
    copyfile = project / 'original-copy.json'; copyfile.write_bytes(receipt.read_bytes())
    receipt.unlink(); receipt.symlink_to(copyfile)
    with pytest.raises(prefit.PrefitError, match='symlink'):
        prefit.inspect_failed_prefit(project)


def test_a_numerical_file_cannot_be_hidden_under_prefit_history(project):
    history = project / prefit.PILOT / 'prefit_history' / ('a' * 64)
    history.mkdir(parents=True)
    (history / 'engineering_initial.pt').write_bytes(b'NOT_A_REAL_MODEL')
    with pytest.raises(prefit.PrefitError, match='history_numerical_or_unknown'):
        prefit.prepare_prefit_resume(project)


def test_unclosed_parent_history_cannot_be_disguised_as_empty_closed_journal(project):
    alter(project, 'campaign_wall', closed_intervals=0)
    with pytest.raises(prefit.PrefitError, match='closed_unchanged'):
        prefit.inspect_failed_prefit(project)


def test_directory_durability_failure_never_publishes_resumption_authority(project,monkeypatch):
    before = original_bytes(project)
    original_fsync = os.fsync
    def reject_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError('SOFTWARE_DIRECTORY_DURABILITY_FAILURE')
        return original_fsync(fd)
    monkeypatch.setattr(prefit.os, 'fsync', reject_directory_fsync)
    with pytest.raises(OSError, match='DURABILITY_FAILURE'):
        prefit.prepare_prefit_resume(project)
    assert original_bytes(project) == before
    assert not (project / '.paper3-hpc/jobs/812345/pilot_prefit_resume.json').exists()


def test_no_existing_pilot_and_completed_pilot_are_never_reset(project):
    receipt = project / prefit.FILES['receipt']; receipt.unlink(); receipt.parent.rmdir()
    assert prefit.prepare_prefit_resume(project)['status'] == 'PILOT_NOT_STARTED_NO_PREFIT_RESUME'
    write(project, prefit.FILES['receipt'], {'status': 'ENGINEERING_PILOT_COMPLETE',
                                         'scope': 'SOFTWARE_STATUS_ONLY_NOT_VERIFIED'})
    before = receipt.read_bytes()
    assert prefit.prepare_prefit_resume(project)['status'] == 'EXISTING_PILOT_REQUIRES_ORIGINAL_FULL_VERIFICATION'
    assert receipt.read_bytes() == before


def test_actual_pilot_prefix_consumes_real_proof_and_preserves_claim_and_prior_wall_before_any_fit(
        project, monkeypatch):
    from tools import run_controlled_pilot as pilot
    from p3work import runtime_clock
    original = original_bytes(project)
    proof = prefit.prepare_prefit_resume(project)
    monkeypatch.setattr(pilot.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(runtime_clock, 'collect_clock_preflight',
                        lambda: {'scope': 'EXPLICIT_SOFTWARE_CLOCK_TEST_DOUBLE'})
    def stop_before_fit(root):
        assert root == project
        raise RuntimeError('SOFTWARE_SENTINEL_BEFORE_ANY_NUMERICAL_INITIALIZATION')
    monkeypatch.setattr(pilot, '_resolved_dependencies', stop_before_fit)
    for name in ('_new_state', '_updates'):
        monkeypatch.setattr(pilot, name, lambda *a, **k: pytest.fail('No numerical work is authorized'))
    monkeypatch.setattr(pilot.torch, 'use_deterministic_algorithms',
                        lambda *a, **k: pytest.fail('Sentinel must run before numerical setup'))
    times = iter((100., 101.5))
    monkeypatch.setattr(pilot.time, 'monotonic', lambda: next(times))
    # Record this explicitly failed software run without invoking the artifact
    # producer, which needs real dependency/identity evidence we did not create.
    monkeypatch.setattr(pilot, '_finalize_result_manifests',
        lambda root, output, receipt: pilot._atomic_json(output / 'engineering_receipt.json', receipt))
    receipt = pilot.execute_pilot(SimpleNamespace(root=project), prefit.PILOT, user_started_gpu=True)
    assert receipt['status'] == 'ENGINEERING_PILOT_FAILED'
    assert receipt['failure'] == 'SOFTWARE_SENTINEL_BEFORE_ANY_NUMERICAL_INITIALIZATION'
    assert receipt['started_utc'] == '2026-10-21T15:45:00+00:00'
    assert receipt['actual_completed_updates'] == receipt['optimizer_updates_dispatched'] == 0
    assert receipt['total_monotonic_wall_seconds'] == 1.75  # 1.5 new + .25 original, never refunded.
    assert receipt['prefit_resume']['proof'] == proof['proof']
    assert (project / proof['history']['receipt']['path']).read_bytes() == original[prefit.FILES['receipt']]
    for relative in prefit.FILES.values():
        if relative != prefit.FILES['receipt']:
            assert (project / relative).read_bytes() == original[relative]


def test_actual_pilot_prefix_refuses_existing_output_without_job_bound_proof_before_writing(project,monkeypatch):
    from tools import run_controlled_pilot as pilot
    monkeypatch.setattr(pilot.torch.cuda, 'is_available', lambda: True)
    before = original_bytes(project)
    with pytest.raises(prefit.PrefitError, match='proof_file_unavailable'):
        pilot.execute_pilot(SimpleNamespace(root=project), prefit.PILOT, user_started_gpu=True)
    assert original_bytes(project) == before


def test_original_RunPod_duplicate_pilot_rejection_is_retained_without_school_selector(project,monkeypatch):
    from tools import run_controlled_pilot as pilot
    monkeypatch.setattr(pilot.torch.cuda, 'is_available', lambda: True)
    monkeypatch.delenv('PAPER3_HPC_RUNTIME_PROFILE')
    before = original_bytes(project)
    with pytest.raises(pilot.PilotGateError, match='Output exists'):
        pilot.execute_pilot(SimpleNamespace(root=project), prefit.PILOT, user_started_gpu=True)
    assert original_bytes(project) == before
