"""Deployment safety tests use metadata/doubles, never a real training fit."""
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
from types import SimpleNamespace

import pytest

from hpc import run_stage as stage
from hpc import environment


def test_login_shell_refused_before_any_environment_or_training_import(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr(stage, 'inspect_environment', lambda **kw: pytest.fail('must not import torch'))
    assert stage.main(['--root', str(tmp_path), '--stage', 'train']) == 2


@pytest.mark.parametrize('node', ['login1', 'login1.school', 'login2'])
def test_slurm_environment_does_not_authorize_login_node(tmp_path, node):
    with pytest.raises(stage.StageError, match='login_node'):
        stage.guard(tmp_path, 'test', environ={'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': node})


def test_gpu_stage_requires_allocation_not_just_sbatch_id(tmp_path):
    with pytest.raises(stage.StageError, match='GPU_allocation'):
        stage.guard(tmp_path, 'train', environ={'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'gpu3'})


def test_existing_shared_root_and_allocated_gpu_accepted_without_forcing_runpod(tmp_path):
    assert stage.guard(tmp_path, 'train', environ={'SLURM_JOB_ID': '123',
        'SLURMD_NODENAME': 'gpu3', 'SLURM_JOB_GPUS': '2'}) == tmp_path


def test_scheduler_node_label_cannot_override_actual_login_hostname(tmp_path):
    with pytest.raises(stage.StageError, match='login_node'):
        stage.guard(tmp_path, 'train', hostname='login1', environ={
            'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'gpu3', 'SLURM_JOB_GPUS': '0'})


@pytest.mark.parametrize('extra', [{'CUDA_VISIBLE_DEVICES': '0'}, {'SLURM_GPUS_ON_NODE': '0'}])
def test_visible_or_zero_devices_cannot_replace_slurm_gpu_allocation(tmp_path, extra):
    with pytest.raises(stage.StageError, match='GPU_allocation'):
        stage.guard(tmp_path, 'train', hostname='gpu3', environ={
            'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'gpu3', **extra})


def test_zero_physical_device_id_is_one_real_slurm_allocation(tmp_path):
    assert stage.guard(tmp_path, 'train', hostname='gpu3', environ={
        'SLURM_JOB_ID': '123', 'SLURM_JOB_GPUS': '0'}) == tmp_path


def test_owner_lock_blocks_second_launcher_and_releases(tmp_path):
    with stage.owner_lock(tmp_path):
        with pytest.raises(stage.StageError, match='existing_controller'):
            with stage.owner_lock(tmp_path):
                pytest.fail('duplicate owner entered')
    with stage.owner_lock(tmp_path):
        pass


def test_signal_is_local_stop_event_and_handlers_restored():
    before = signal.getsignal(signal.SIGUSR1)
    with stage.stop_signals() as stop:
        os.kill(os.getpid(), signal.SIGUSR1)
        assert stop.is_set()
    assert signal.getsignal(signal.SIGUSR1) == before


def test_completed_pilot_never_dispatches_source_fit(tmp_path, monkeypatch):
    campaign = importlib.import_module('p3work.training_campaign')
    monkeypatch.setattr(campaign, 'next_action', lambda root: {'kind': 'source'})
    verified = []
    monkeypatch.setattr(stage, 'verify_pilot', lambda root: verified.append(root))
    def forbidden(*args, **kwargs):
        pytest.fail('pilot command must not dispatch source after existing pilot')
    assert stage.run_training(tmp_path, 'pilot', tmp_path, threading.Event(), runner=forbidden) == 0
    assert verified == [tmp_path]


def test_first_pilot_uses_one_original_action_and_verifies_real_receipt(tmp_path, monkeypatch):
    campaign = importlib.import_module('p3work.training_campaign')
    monkeypatch.setattr(campaign, 'next_action', lambda root: {'kind': 'pilot'})
    seen, verified = [], []
    monkeypatch.setattr(stage, 'verify_pilot', lambda root: verified.append(root))
    def runner(root, tool, args, *rest, **kwargs):
        seen.append((tool, args, kwargs)); return 3
    assert stage.run_training(tmp_path, 'pilot', tmp_path, threading.Event(), runner=runner) == 0
    assert len(seen) == 1 and seen[0][1][-2:] == ['--max-actions', '1']
    assert '--max-wall-seconds' not in seen[0][1]
    assert '--handle-slurm-signals' in seen[0][1] and verified == [tmp_path]


def test_paused_mainline_cannot_initialize_extension_or_external_query(tmp_path):
    seen = []
    def runner(root, tool, args, *rest, **kwargs):
        seen.append(tool); return 3
    assert stage.run_training(tmp_path, 'train', tmp_path, threading.Event(), runner=runner) == 3
    assert seen == ['tools/run_training_campaign.py']


def test_completed_recovery_validates_existing_results_before_campaign_without_prefit(tmp_path, monkeypatch):
    from hpc import pilot_finalize, pilot_prefit, runtime_profile
    from tools import run_controlled_pilot, train_source
    monkeypatch.setenv('PAPER3_HPC_RUNTIME_PROFILE', runtime_profile.PROFILE)
    events = []
    cache, runtime, inspection = object(), {'device_uuid': 'ORIGINAL_DEVICE_TEST_DOUBLE'}, {'original': True}
    monkeypatch.setattr(runtime_profile, 'prepare_runtime_profile', lambda *a: {'profile': 'TEST_DOUBLE'})
    monkeypatch.setattr(run_controlled_pilot, 'verify_inputs', lambda root: cache)
    monkeypatch.setattr(train_source, 'cuda_runtime', lambda: runtime)
    monkeypatch.setattr(pilot_prefit, 'prepare_prefit_resume', lambda *a: pytest.fail('200 updates must not use zero-update recovery'))
    def finalize(root, **kw):
        assert kw == {'cache': cache, 'runtime': runtime, 'inspection': inspection}
        events.append('finalized_with_original_verifier')
        return {'status': 'SOFTWARE_TEST_DOUBLE_NO_FIT', 'optimizer_updates_performed': 0}
    monkeypatch.setattr(pilot_finalize, 'finalize_completed_pilot', finalize)
    def launch(root, entry, *args, **kw):
        assert events == ['finalized_with_original_verifier']
        events.append(entry)
        return 3
    assert stage.run_training(tmp_path, 'train', tmp_path, threading.Event(), runner=launch,
                              completed_resume={'inspection': inspection}) == 3
    assert events == ['finalized_with_original_verifier', 'tools/run_training_campaign.py']
    assert json.loads((tmp_path/'pilot_finalization.json').read_text())['optimizer_updates_performed'] == 0


def test_completed_recovery_rejected_finalization_cannot_dispatch_campaign(tmp_path, monkeypatch):
    from hpc import pilot_finalize, runtime_profile
    from tools import run_controlled_pilot, train_source
    monkeypatch.setenv('PAPER3_HPC_RUNTIME_PROFILE', runtime_profile.PROFILE)
    monkeypatch.setattr(runtime_profile, 'prepare_runtime_profile', lambda *a: {})
    monkeypatch.setattr(run_controlled_pilot, 'verify_inputs', lambda root: object())
    monkeypatch.setattr(train_source, 'cuda_runtime', lambda: {})
    def reject(*a, **kw):
        raise pilot_finalize.FinalizeError('pilot_finalize_original_deep_verifier_rejected_candidate')
    monkeypatch.setattr(pilot_finalize, 'finalize_completed_pilot', reject)
    with pytest.raises(pilot_finalize.FinalizeError, match='original_deep_verifier_rejected'):
        stage.run_training(tmp_path, 'train', tmp_path, threading.Event(),
                           runner=lambda *a, **kw: pytest.fail('invalid pilot must not fit'),
                           completed_resume={'inspection': {}})


def test_wrong_completed_pilot_GPU_blocks_before_frozen_data_reads(tmp_path, monkeypatch):
    from hpc import handoff, pilot_finalize, resume_completed_pilot, stage_data
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(stage, 'guard', lambda root, name: root)
    monkeypatch.setattr(stage, 'audit_restore', lambda *a: {})
    monkeypatch.setattr(stage, 'verify_code', lambda *a: {})
    monkeypatch.setattr(stage, 'inspect_environment', lambda **kw: {'runtime': {'device_uuid': 'OTHER_GPU'}})
    monkeypatch.setattr(stage, 'tested', lambda *a: None)
    monkeypatch.setattr(resume_completed_pilot, 'allocated_resume_proof', lambda *a, **kw: {'inspection': {}})
    def reject(*a):
        raise pilot_finalize.FinalizeError('pilot_finalize_original_runtime_device_mismatch')
    monkeypatch.setattr(pilot_finalize, 'require_runtime_identity', reject)
    monkeypatch.setattr(stage_data, 'prepare_data', lambda *a, **kw: pytest.fail('wrong GPU should block before expensive data reads'))
    monkeypatch.setattr(stage, 'run_training', lambda *a, **kw: pytest.fail('wrong GPU must not fit'))
    monkeypatch.setattr(handoff, 'snapshot', lambda *a, **kw: {})
    assert stage.run(tmp_path, 'train') == 2
    receipt = json.loads((tmp_path/'.paper3-hpc/jobs/123/stage-train.json').read_text())
    assert receipt['result']['reason'] == 'pilot_finalize_original_runtime_device_mismatch'
    assert receipt['frozen_data_verification'] is None


def test_completed_recovery_requires_explicit_registered_runtime_profile(tmp_path, monkeypatch):
    monkeypatch.delenv('PAPER3_HPC_RUNTIME_PROFILE', raising=False)
    with pytest.raises(stage.StageError, match='explicit_runtime_profile_required'):
        stage.run_training(tmp_path, 'train', tmp_path, threading.Event(),
                           runner=lambda *a, **kw: pytest.fail('must not dispatch'), completed_resume={})


def test_missing_school_software_test_gate_blocks_training(tmp_path):
    with pytest.raises(stage.StageError, match='software_test_required'):
        stage.tested(tmp_path, {}, {'modules': {}, 'python': '3.12.3'})


def test_startup_failure_summary_exposes_actual_timing_without_arbitrary_output(tmp_path):
    path = tmp_path / 'startup.json'
    value = {'schema_version': 'paper3-hpc-startup-import-probe-v1',
        'status': 'IMPORT_PROBES_INCOMPLETE', 'stdout': 'SECRET_MUST_NOT_LEAK',
        'modules': [{'module': 'p3work.classical_training', 'status': 'IMPORT_PROBE_TIMEOUT',
            'process_wall_seconds': 180.298, 'stdout': 'SECRET_MUST_NOT_LEAK'},
            {'module': 'p3work.training_state', 'status': 'IMPORT_COMPLETE',
             'import_wall_seconds': 120.974, 'user_seconds': 2.478, 'system_seconds': .907}]}
    path.write_text(json.dumps(value))
    summary = stage.startup_summary(path)
    assert summary['status'] == value['status'] and summary['modules_completed'] == 2
    assert summary['modules'][0]['process_wall_seconds'] == 180.298
    assert summary['modules'][1]['user_seconds'] == 2.478
    assert 'SECRET' not in json.dumps(summary)


@pytest.mark.parametrize('payload', ['not JSON', '[]', '{"schema_version":"wrong"}'])
def test_unusable_startup_report_never_claims_probe_passed(tmp_path, payload):
    path = tmp_path / 'startup.json'; path.write_text(payload)
    assert stage.startup_summary(path) == {'status': 'STARTUP_REPORT_NOT_AVAILABLE'}


@pytest.mark.parametrize('phase,expected_mode', [('prepare', 'materialize'), ('train', 'check')])
def test_staged_chain_preserves_failed_data_gate_and_never_fits(tmp_path, monkeypatch, phase, expected_mode):
    from hpc import stage_data, handoff
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(stage, 'guard', lambda root, name: root)
    monkeypatch.setattr(stage, 'audit_restore', lambda *a: {})
    monkeypatch.setattr(stage, 'verify_code', lambda *a: {})
    monkeypatch.setattr(stage, 'inspect_environment', lambda **kw: {})
    monkeypatch.setattr(stage, 'tested', lambda *a: None)
    def incomplete(root, *, mode, progress):
        assert mode == expected_mode and callable(progress)
        progress({'phase': 'frozen_file_SHA_verification', 'files_processed': 1, 'files_total': 2})
        return {'status': 'blocked', 'missing_cache_files': 1}
    monkeypatch.setattr(stage_data, 'prepare_data', incomplete)
    monkeypatch.setattr(stage, 'run_training', lambda *a: pytest.fail('unverified inputs must never fit'))
    monkeypatch.setattr(handoff, 'snapshot', lambda *a, **kw: {'status': 'NO_RESULT_TEST_DOUBLE'})
    assert stage.run(tmp_path, phase) == 2
    value = json.loads((tmp_path / f'.paper3-hpc/jobs/123/stage-{phase}.json').read_text())
    assert value['returncode'] == 2 and value['scientific_completion_claimed'] is False


def test_software_pass_is_bound_to_code_and_environment(tmp_path):
    path = tmp_path / '.paper3-hpc/software_test_passed.json'
    env = {'modules': {'torch': '2.5.1+cu121'}, 'python': '3.12.3'}
    stage.write(path, {'returncode': 0, 'software_identity': {'manifest_sha256': 'a'}, 'environment': env})
    stage.tested(tmp_path, {'manifest_sha256': 'a'}, env)
    with pytest.raises(stage.StageError, match='changed'):
        stage.tested(tmp_path, {'manifest_sha256': 'b'}, env)
    with pytest.raises(stage.StageError, match='changed'):
        stage.tested(tmp_path, {'manifest_sha256': 'a'}, {**env, 'modules': {'torch': '2.8'}})


def test_support_code_change_invalidates_frozen_deployment_identity(tmp_path):
    import hashlib
    (tmp_path / 'hpc').mkdir()
    support = tmp_path / 'hpc/run_stage.py'
    support.write_bytes(b'original software')
    value = {'schema_version': 'paper3-hpc-software-overlay-v1', 'baseline_files': [],
             'deployment_overrides': {}, 'deployment_support_files': [
                 {'path': 'hpc/run_stage.py', 'bytes': support.stat().st_size,
                  'sha256': hashlib.sha256(support.read_bytes()).hexdigest()}]}
    stage.write(tmp_path / 'hpc/scientific_code.json', value)
    assert environment.verify_code(tmp_path)['numerical_kernel_overrides'] == 0
    support.write_bytes(b'changed software')
    with pytest.raises(environment.EnvironmentError, match='SHA256'):
        environment.verify_code(tmp_path)


def test_numerical_kernel_override_is_never_accepted(tmp_path):
    stage.write(tmp_path / 'hpc/scientific_code.json', {
        'schema_version': 'paper3-hpc-software-overlay-v1', 'baseline_files': [],
        'deployment_overrides': {'p3work/adapt.py': {}}})
    with pytest.raises(environment.EnvironmentError, match='kernel_override_forbidden'):
        environment.verify_code(tmp_path)


def test_existing_environment_probe_does_not_install_or_fit(monkeypatch):
    for name in environment.THREADS:
        monkeypatch.setenv(name, '1')
    result = environment.inspect_environment()
    assert result['existing_environment_reused'] is True
    assert result['packages_installed'] == 0 and result['numerical_fit_performed'] is False
    assert result['CUDA_probe_performed'] is False


def test_thread_cap_violation_is_rejected_without_importing_torch(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.setattr(environment.importlib, 'import_module', lambda name: pytest.fail('must check before import'))
    with pytest.raises(environment.EnvironmentError, match='thread_caps'):
        environment.inspect_environment()


def test_real_owned_coordinator_receives_stop_and_writes_metadata_checkpoint(tmp_path):
    import time
    tool = tmp_path / 'coordinator.py'
    tool.write_text('''import signal, time\nfrom pathlib import Path\nroot = Path(__file__).parent\ndef stop(*args):\n (root / 'metadata_checkpoint').write_text('paused without any fit')\n raise SystemExit(3)\nsignal.signal(signal.SIGUSR1, stop)\n(root / 'started').write_text('metadata only')\nwhile True: time.sleep(.02)\n''')
    event = threading.Event()
    def request_stop():
        deadline = time.monotonic() + 5
        while not (tmp_path / 'started').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        event.set()
    thread = threading.Thread(target=request_stop)
    thread.start()
    try:
        assert stage.child(tmp_path, 'coordinator.py', [], tmp_path / 'child.log', event,
                           cooperative=True) == 3
    finally:
        thread.join(timeout=6)
    assert (tmp_path / 'metadata_checkpoint').read_text() == 'paused without any fit'


def test_unconfirmed_worker_cleanup_forbids_snapshot_publication(tmp_path, monkeypatch):
    from hpc import handoff, stage_data
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('SLURMD_NODENAME', 'gpu3')
    monkeypatch.setenv('SLURM_JOB_GPUS', '0')
    monkeypatch.delenv('PAPER3_HPC_ROOT', raising=False)
    monkeypatch.setattr(stage, 'audit_restore', lambda *args: {})
    monkeypatch.setattr(stage, 'verify_code', lambda *args: {})
    monkeypatch.setattr(stage, 'inspect_environment', lambda **kwargs: {})
    monkeypatch.setattr(stage, 'tested', lambda *args: None)
    monkeypatch.setattr(stage_data, 'prepare_data', lambda *args, **kwargs: {'status': 'ready'})
    def unsafe(*args):
        raise stage.CleanupUnconfirmed('coordinator_cleanup_unconfirmed')
    monkeypatch.setattr(stage, 'run_training', unsafe)
    monkeypatch.setattr(handoff, 'snapshot', lambda *args, **kwargs: pytest.fail('workers may still write'))
    assert stage.run(tmp_path, 'train') == 2
    receipt = json.loads((tmp_path / '.paper3-hpc/jobs/123/stage-train.json').read_text())
    assert receipt['scientific_completion_claimed'] is False
    assert receipt['snapshot_error_type'].startswith('SKIPPED_OWNED_WORKER_CLEANUP_UNCONFIRMED')


PILOT_ERROR = 'Actual completed 200-update engineering pilot with exact resume is required'


def test_coordinator_failure_retains_exit_log_and_safe_actual_pilot_gate(tmp_path, capsys):
    log = tmp_path / 'campaign.log'
    original = json.dumps({'status': 'BLOCKED_PRESERVE_EXISTING_JOBS', 'error_type': 'CampaignError',
        'reason': 'stage_blocked:SourceTrainingError:' + PILOT_ERROR})
    log.write_text(original)
    calls = []
    def blocked(root, entry, args, output, stop, **kwargs):
        calls.append((entry, args, kwargs)); assert output == log
        return 2
    assert stage.run_training(tmp_path, 'train', tmp_path, threading.Event(), runner=blocked) == 2
    result = json.loads((tmp_path / 'training_result.json').read_text())
    assert result['status'] == 'TRAINING_STOPPED_OR_BLOCKED_NOT_COMPLETE'
    row = result['coordinators'][0]
    assert row['entry'] == 'tools/run_training_campaign.py' and row['returncode'] == 2
    assert row['log'] == 'campaign.log' and row['error_type'] == 'CampaignError'
    assert row['reported_status'] == 'BLOCKED_PRESERVE_EXISTING_JOBS'
    assert row['reason_code'] == 'engineering_pilot_evidence_invalid'
    assert len(calls) == 1 and calls[0][2] == {'cooperative': True}
    assert '--max-wall-seconds' not in calls[0][1] and log.read_text() == original
    assert 'engineering_pilot_evidence_invalid' in capsys.readouterr().out


@pytest.mark.parametrize('code', [0, 2, 3])
def test_arbitrary_coordinator_metadata_is_not_echoed_or_mistaken_for_completion(tmp_path, code):
    log = tmp_path / 'campaign.log'
    log.write_text(json.dumps({'status': 'SECRET_METADATA_MUST_NOT_LEAK',
        'error_type': 'SECRET_METADATA_MUST_NOT_LEAK', 'reason': 'SECRET_METADATA_MUST_NOT_LEAK',
        'stdout': 'SECRET_METADATA_MUST_NOT_LEAK'}))
    summary = stage.coordinator_result(tmp_path, 'tools/run_training_campaign.py', log, code, threading.Event())
    assert summary['returncode'] == code and summary['scientific_completion_claimed'] is False
    assert 'SECRET_METADATA_MUST_NOT_LEAK' not in json.dumps(summary)
    assert 'reason_code' not in summary and 'reported_status' not in summary


def test_exact_known_reason_does_not_authorize_echoing_a_secret_suffix(tmp_path):
    log = tmp_path / 'campaign.log'
    log.write_text(json.dumps({'reason': PILOT_ERROR + ': SECRET_SUFFIX'}))
    summary = stage.coordinator_result(tmp_path, 'tools/run_training_campaign.py', log, 2, threading.Event())
    assert 'reason_code' not in summary and 'SECRET_SUFFIX' not in json.dumps(summary)


def test_last_bounded_coordinator_JSON_is_diagnostic_after_warnings_and_previous_records(tmp_path):
    log = tmp_path / 'extension.log'
    log.write_text('SECRET_WARNING_NOT_TO_ECHO\n' + json.dumps({'status': 'PENDING_DATA_GATES'}, indent=2)
        + '\n' + json.dumps({'status': 'BLOCKED_PRESERVE_EXISTING_JOBS', 'error_type': 'ValueError'}, indent=2))
    summary = stage.coordinator_result(tmp_path, 'tools/run_source_extension_downstream.py', log, 2, threading.Event())
    assert summary['reported_status'] == 'BLOCKED_PRESERVE_EXISTING_JOBS'
    assert summary['error_type'] == 'ValueError' and 'SECRET_WARNING' not in json.dumps(summary)


@pytest.mark.parametrize('kind', ['missing', 'oversized', 'symlink', 'invalid'])
def test_unusable_coordinator_log_still_reports_actual_nonzero_exit_without_payload(tmp_path, kind):
    log = tmp_path / 'campaign.log'
    if kind == 'oversized':
        log.write_text('SECRET_PAYLOAD' * 25000)
    elif kind == 'symlink':
        original = tmp_path / 'secret'; original.write_text('SECRET_PAYLOAD'); log.symlink_to(original)
    elif kind == 'invalid':
        log.write_text('SECRET_PAYLOAD')
    summary = stage.coordinator_result(tmp_path, 'tools/run_training_campaign.py', log, 2, threading.Event())
    assert summary['returncode'] == 2 and summary['status'] == 'COORDINATOR_RETURNED_NONZERO'
    assert 'SECRET_PAYLOAD' not in json.dumps(summary)


@pytest.mark.parametrize('failure_at', [None, 1, 2, 3, 4])
def test_success_and_each_extension_exit_keep_original_dispatch_order_and_result(
        tmp_path, monkeypatch, failure_at):
    from hpc import stage_data
    arms = ('BASE4_PLUS_YANG', 'BASE4_PLUS_DREYER')
    monkeypatch.setattr(stage_data, 'load_index', lambda root: {'extension_manifests': {a: a+'.json' for a in arms}})
    calls = []
    def runner(root, entry, args, output, stop, **kwargs):
        assert kwargs == {'cooperative': True}
        assert '--max-wall-seconds' not in args
        calls.append((entry, args))
        return 2 if len(calls)-1 == failure_at else 0
    code = stage.run_training(tmp_path, 'train', tmp_path, threading.Event(), runner=runner)
    assert code == (0 if failure_at is None else 2)
    expected = [('tools/run_training_campaign.py', None, None)]
    expected.extend(('tools/run_source_extension_downstream.py', arm, action)
                    for arm in arms for action in ('initialize', 'auto-development'))
    observed = [(entry, args[args.index('--arm')+1] if '--arm' in args else None,
                 args[args.index('--stage')+1] if '--stage' in args else None) for entry, args in calls]
    assert observed == expected[:5 if failure_at is None else failure_at+1]
    result = json.loads((tmp_path / 'training_result.json').read_text())
    assert result['returncode'] == code and len(result['coordinators']) == len(calls)
    assert result['coordinators'][-1]['returncode'] == code
    assert result['scientific_completion_claimed'] is False


@pytest.mark.parametrize('code, signal_stop', [(3, False), (0, True)])
def test_paused_or_signalled_mainline_is_recorded_and_never_dispatches_extension(tmp_path, code, signal_stop):
    stop = threading.Event(); calls = []
    def runner(root, entry, args, output, event, **kwargs):
        calls.append(entry)
        if signal_stop: event.set()
        return code
    assert stage.run_training(tmp_path, 'train', tmp_path, stop, runner=runner) == 3
    result = json.loads((tmp_path / 'training_result.json').read_text())
    assert result['returncode'] == 3 and calls == ['tools/run_training_campaign.py']
    assert result['coordinators'][0]['returncode'] == code
    assert result['coordinators'][0]['status'] == 'COORDINATOR_PAUSED_OR_STOPPED_NOT_COMPLETE'


def test_failed_training_stage_receipt_does_not_misreport_prior_data_ready_result(tmp_path, monkeypatch):
    from hpc import handoff, stage_data
    monkeypatch.setenv('SLURM_JOB_ID', '767530')
    monkeypatch.setattr(stage, 'guard', lambda root, name: root)
    monkeypatch.setattr(stage, 'audit_restore', lambda *args: {})
    monkeypatch.setattr(stage, 'verify_code', lambda *args: {})
    monkeypatch.setattr(stage, 'inspect_environment', lambda **kwargs: {})
    monkeypatch.setattr(stage, 'tested', lambda *args: None)
    data = {'status': 'ready', 'locally_SHA_verified': 21262, 'scientific_fits': 0}
    monkeypatch.setattr(stage_data, 'prepare_data', lambda *args, **kwargs: data)
    monkeypatch.setattr(handoff, 'snapshot', lambda *args, **kwargs: {'status': 'NO_RESULT_SOFTWARE_DOUBLE'})
    original_run_training = stage.run_training
    def blocked(root, entry, args, output, stop, **kwargs):
        output.write_text(json.dumps({'status': 'BLOCKED_PRESERVE_EXISTING_JOBS', 'error_type': 'CampaignError',
            'reason': 'stage_blocked:SourceTrainingError:' + PILOT_ERROR}))
        return 2
    monkeypatch.setattr(stage, 'run_training', lambda *args: original_run_training(*args, runner=blocked))
    assert stage.run(tmp_path, 'train') == 2
    result = json.loads((tmp_path / '.paper3-hpc/jobs/767530/stage-train.json').read_text())
    assert result['result']['status'] == 'TRAINING_STOPPED_OR_BLOCKED_NOT_COMPLETE'
    assert result['result']['coordinators'][0]['reason_code'] == 'engineering_pilot_evidence_invalid'
    assert result['frozen_data_verification'] == data
    assert result['scientific_completion_claimed'] is False


def _pilot_diagnostic_fixture(root, **changes):
    relative = 'runs/controlled_source_pilot_seed20261018/engineering_receipt.json'
    value = {'scope': 'ENGINEERING_SOURCE_PILOT_ONLY', 'status': 'ENGINEERING_PILOT_FAILED',
             'training_started': True, 'exact_resume_verified': True, 'actual_completed_updates': 200,
             'optimizer_updates_dispatched': 200, 'prediction_replay_optimizer_updates': 100,
             'seed': 20261018, 'max_jobs': 1, 'result_manifest_error_type': 'PilotGateError',
             'result_manifest_status': 'BLOCKED_NO_COMPLETE_BACKUP_MANIFEST',
             'checkpoint_files': {name: 'a'*64 for name in ('engineering_initial.pt',
                 'engineering_continuous100.pt', 'engineering_resume50.pt', 'engineering_resumed100.pt')},
             'checkpoint_io_timings': [{'payload': 'SECRET_DO_NOT_ECHO'}],
             'timing_segments': [{'payload': 'SECRET_DO_NOT_ECHO'}],
             'total_monotonic_wall_seconds': 210.5}
    value.update(changes); stage.write(root / relative, value)
    return root / relative


def test_failed_pilot_200_update_finalization_is_visible_without_claiming_completion(tmp_path):
    receipt = _pilot_diagnostic_fixture(tmp_path)
    before = receipt.read_bytes()
    summary = stage.pilot_receipt_summary(tmp_path)
    assert summary['reported_status'] == 'ENGINEERING_PILOT_FAILED'
    assert summary['actual_completed_updates'] == summary['optimizer_updates_dispatched'] == 200
    assert summary['exact_resume_verified'] is True
    assert summary['result_manifest_error_type'] == 'PilotGateError'
    assert summary['result_manifest_status'] == 'BLOCKED_NO_COMPLETE_BACKUP_MANIFEST'
    assert summary['failure_phase'] == 'result_manifest_publication'
    assert summary['checkpoint_files_count'] == 4
    assert summary['artifact_manifest_declared'] is summary['backup_manifest_declared'] is False
    assert summary['scientific_completion_claimed'] is False
    assert 'SECRET_DO_NOT_ECHO' not in json.dumps(summary) and receipt.read_bytes() == before


@pytest.mark.parametrize('failure,expected', [
    ('Controlled engineering pilot wall-time budget reached', 'engineering_pilot_wall_budget_exhausted'),
    ('CUBLAS workspace configuration is incompatible with deterministic execution', 'cublas_determinism_configuration_invalid'),
    ('UNKNOWN_SECRET_DO_NOT_ECHO', None),
    ('Controlled engineering pilot wall-time budget reached: SECRET_DO_NOT_ECHO', None)])
def test_only_exact_known_pilot_failure_text_becomes_a_constant_code(tmp_path, failure, expected):
    _pilot_diagnostic_fixture(tmp_path, failure_type='TimeoutError', failure=failure)
    summary = stage.pilot_receipt_summary(tmp_path)
    assert summary.get('reason_code') == expected and summary['failure_type'] == 'TimeoutError'
    assert 'SECRET_DO_NOT_ECHO' not in json.dumps(summary)


def test_known_missing_dependency_inventory_reports_code_without_absolute_path(tmp_path):
    missing = str(FileNotFoundError(2, 'No such file or directory', str(tmp_path / 'audit/gpu_resolved_dependencies.json')))
    _pilot_diagnostic_fixture(tmp_path, failure_type='FileNotFoundError', failure=missing,
                              actual_completed_updates=0, training_started=False, exact_resume_verified=False)
    summary = stage.pilot_receipt_summary(tmp_path)
    assert summary['reason_code'] == 'resolved_dependency_inventory_missing'
    assert str(tmp_path) not in json.dumps(summary)


def test_unknown_error_types_and_boolean_counts_are_not_echoed_as_pilot_facts(tmp_path):
    _pilot_diagnostic_fixture(tmp_path, failure_type='SECRET_DO_NOT_ECHO',
        result_manifest_error_type='SECRET_DO_NOT_ECHO', result_manifest_status='SECRET_DO_NOT_ECHO',
        actual_completed_updates=True, optimizer_updates_dispatched='SECRET_DO_NOT_ECHO')
    summary = stage.pilot_receipt_summary(tmp_path)
    assert 'failure_type' not in summary and 'result_manifest_error_type' not in summary
    assert 'actual_completed_updates' not in summary and 'optimizer_updates_dispatched' not in summary
    assert 'SECRET_DO_NOT_ECHO' not in json.dumps(summary)


@pytest.mark.parametrize('kind', ['missing', 'invalid', 'oversized', 'symlink'])
def test_unusable_pilot_receipt_cannot_claim_any_production_completion(tmp_path, kind):
    path = tmp_path / 'runs/controlled_source_pilot_seed20261018/engineering_receipt.json'
    path.parent.mkdir(parents=True)
    if kind == 'invalid': path.write_text('SECRET_DO_NOT_ECHO')
    elif kind == 'oversized': path.write_text('SECRET_DO_NOT_ECHO' * 20000)
    elif kind == 'symlink':
        target = tmp_path / 'secret'; target.write_text('SECRET_DO_NOT_ECHO'); path.symlink_to(target)
    summary = stage.pilot_receipt_summary(tmp_path)
    assert summary['status'] == 'PILOT_RECEIPT_DIAGNOSTIC_NOT_AVAILABLE'
    assert 'SECRET_DO_NOT_ECHO' not in json.dumps(summary)


def test_campaign_failure_summary_surfaces_current_pilot_receipt_and_change_without_refit(tmp_path):
    _pilot_diagnostic_fixture(tmp_path, actual_completed_updates=0)
    calls = []
    def runner(root, entry, args, output, stop, **kwargs):
        calls.append(entry)
        _pilot_diagnostic_fixture(root)
        output.write_text(json.dumps({'status': 'BLOCKED_PRESERVE_EXISTING_JOBS',
            'error_type': 'CampaignError', 'reason': 'stage_blocked:SourceTrainingError:' + PILOT_ERROR}))
        return 2
    assert stage.run_training(tmp_path, 'train', tmp_path, threading.Event(), runner=runner) == 2
    result = json.loads((tmp_path / 'training_result.json').read_bytes())
    summary = result['coordinators'][0]['pilot_receipt']
    assert summary['actual_completed_updates'] == 200 and summary['receipt_changed_during_coordinator'] is True
    assert summary['result_manifest_error_type'] == 'PilotGateError'
    assert summary['failure_phase'] == 'result_manifest_publication'
    assert result['returncode'] == 2 and calls == ['tools/run_training_campaign.py']
    assert result['scientific_completion_claimed'] is False


@pytest.mark.parametrize('kind,expected', [
    ('known', 'pilot_runtime_complete_dependency_metadata_required'), ('unknown', None), ('other_type', None)])
def test_handoff_diagnostic_accepts_only_actual_known_adapter_literal_codes(kind, expected):
    from hpc.handoff import HandoffError
    if kind == 'known': error = HandoffError(expected)
    elif kind == 'unknown': error = HandoffError('secret_lowercase_must_not_leak')
    else:
        error = RuntimeError('SECRET_DO_NOT_ECHO'); error.code = 'pilot_runtime_complete_dependency_metadata_required'
    assert stage.handoff_error_code(error) == expected


def test_train_stage_final_receipt_includes_safe_pilot_finalization_and_handoff_failure(tmp_path, monkeypatch):
    from hpc import handoff, stage_data
    monkeypatch.setenv('SLURM_JOB_ID', '767958')
    monkeypatch.setattr(stage, 'guard', lambda root, name: root)
    monkeypatch.setattr(stage, 'audit_restore', lambda *args: {})
    monkeypatch.setattr(stage, 'verify_code', lambda *args: {})
    monkeypatch.setattr(stage, 'inspect_environment', lambda **kwargs: {})
    monkeypatch.setattr(stage, 'tested', lambda *args: None)
    monkeypatch.setattr(stage_data, 'prepare_data', lambda *args, **kwargs: {'status': 'ready'})
    original_run = stage.run_training
    def runner(root, entry, args, output, stop, **kwargs):
        _pilot_diagnostic_fixture(root)
        output.write_text(json.dumps({'status': 'BLOCKED_PRESERVE_EXISTING_JOBS', 'error_type': 'CampaignError',
            'reason': 'stage_blocked:SourceTrainingError:' + PILOT_ERROR}))
        return 2
    monkeypatch.setattr(stage, 'run_training', lambda *args: original_run(*args, runner=runner))
    def snapshot(*args, **kwargs):
        raise handoff.HandoffError('pilot_runtime_complete_dependency_metadata_required')
    monkeypatch.setattr(handoff, 'snapshot', snapshot)
    assert stage.run(tmp_path, 'train') == 2
    result = json.loads((tmp_path / '.paper3-hpc/jobs/767958/stage-train.json').read_bytes())
    assert result['snapshot_error_type'] == 'HandoffError'
    assert result['snapshot_error_code'] == 'pilot_runtime_complete_dependency_metadata_required'
    pilot = result['result']['coordinators'][0]['pilot_receipt']
    assert pilot['actual_completed_updates'] == 200 and pilot['result_manifest_error_type'] == 'PilotGateError'
    assert result['returncode'] == 2 and result['scientific_completion_claimed'] is False
