"""One-command orchestration acceptance; all network/scheduler calls are doubles."""
import json
import errno
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hpc import start as launch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    root = tmp_path / 'school project'; (root / 'hpc').mkdir(parents=True)
    (root / 'hpc/resources.json').write_bytes((ROOT / 'hpc/resources.json').read_bytes())
    return root


def args(root, **changes):
    value = dict(root=root, config=None, python=None, cpus=None, mem_gib=None, gpus=None,
                 time=None, execute=True, dry_run=False, network_host=True)
    value.update(changes)
    return SimpleNamespace(**value)


def forbidden(*args, **kwargs):
    pytest.fail('forbidden network, scheduler, or identity operation')


def verifier(root):
    return {'manifest_sha256': 'a' * 64}


def runner(queue=''):
    def run(command, **kwargs):
        assert kwargs['stdin'] == subprocess.DEVNULL and kwargs['timeout'] == 30
        if command[0] == 'squeue':
            assert '--array' in command and '--format=%i|%j|%T|%Z' in command
            return SimpleNamespace(returncode=0, stdout=queue)
        assert command == ['git', 'rev-parse', '--verify', 'HEAD'] and 'cwd' in kwargs
        return SimpleNamespace(returncode=0, stdout='c' * 40 + '\n')
    return run


def ready(root, **kwargs):
    assert kwargs['mode'] == 'fetch' and kwargs['execute'] and kwargs['network_host']
    assert isinstance(kwargs['prompt'], bool) and callable(kwargs['progress'])
    return {'status': 'ready', 'attempt': '.paper3-hpc/data/software-fixture/receipt.json'}


def submitter(root, calls, *, durable=False):
    def submitted(request):
        attempt = root / '.paper3-hpc/start' / ('attempt-' + request['start_attempt_id'] + '.json')
        before = json.loads(attempt.read_text())
        intent = before['submission_intents'][-1]
        assert intent['state'] == 'SUBMISSION_INTENT_OUTCOME_UNKNOWN' and intent['request'] == request
        assert request['start_source_identity'] == {'source_commit': 'c' * 40, 'code_manifest_sha256': 'a' * 64}
        job_id = str(9000 + len(calls)); calls.append(request)
        result = {**request, 'status': 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE', 'job_id': job_id,
                  'scientific_completion_claimed': False}
        if durable:
            launch.write(root / '.paper3-hpc/submissions' / (job_id + '-aaaaaaaa.json'), result)
        return result
    return submitted


@pytest.mark.parametrize('changes', [{'execute': False}, {'dry_run': True}, {'execute': False, 'network_host': False}])
def test_default_or_explicit_plan_has_no_lock_queue_fetch_or_submit(project, changes):
    result = launch.start(args(project, **changes), runner=forbidden, verifier=forbidden,
                          preparer=forbidden, submitter=forbidden)
    assert result['status'] == 'PLAN_ONLY_NO_FETCH_OR_SUBMISSION'
    assert [row['stage'] for row in result['stages']] == ['test', 'prepare', 'train']
    assert result['stages'][1]['dependency_on_stage'] == 'test'
    assert not (project / '.paper3-hpc').exists()


def test_resource_overrides_affect_only_train_and_keep_original_defaults(project):
    result = launch.start(args(project, execute=False, cpus=20, mem_gib=96, gpus=2, time='16:00:00'), runner=forbidden)
    values = [row['request']['resources'] for row in result['stages']]
    assert [(row['cpus'], row['mem_gib'], row['gpus'], row['time']) for row in values] == [
        (16, 64, 0, '02:00:00'), (24, 128, 0, '12:00:00'), (20, 96, 2, '16:00:00')]
    default = launch.start(args(project, execute=False), runner=forbidden)['stages'][-1]['request']
    assert default['resources']['gpus'] == 1 and default['resources']['time'] == '24:00:00'
    assert not default['scientific_budget_changed']


def test_execute_requires_network_host_and_refuses_compute_job_before_any_mutation(project, monkeypatch):
    with pytest.raises(launch.StartError, match='network_host_permission'):
        launch.start(args(project, network_host=False), runner=forbidden)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    with pytest.raises(launch.StartError, match='not_compute_job'):
        launch.start(args(project), runner=forbidden)
    assert not (project / '.paper3-hpc').exists()


def test_existing_start_owner_prevents_duplicate_network_and_scheduler_calls(project):
    with launch.owner(project):
        with pytest.raises(launch.StartError, match='coordinator_is_still_running'):
            launch.start(args(project), runner=forbidden, verifier=forbidden, preparer=forbidden, submitter=forbidden)


def test_exact_project_active_job_returns_real_id_without_fetch_or_resubmission(project):
    result = launch.start(args(project), runner=runner(f'765432|paper3-train|PENDING|{project}\n'),
                          verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert result['job_ids'] == ['765432'] and result['active_jobs'][0]['binding'] == 'exact_project_workdir'
    assert result['status'] == 'EXISTING_ACTIVE_JOBS_NO_DUPLICATE_FETCH_OR_SUBMISSION'
    assert not list((project / '.paper3-hpc/start').glob('attempt-*.json'))


def test_other_project_same_name_does_not_block_this_project_and_dependencies_use_actual_ids(project):
    calls = []
    result = launch.start(args(project), runner=runner('765432|paper3-train|RUNNING|/other/project\n'),
                          verifier=verifier, preparer=ready, submitter=submitter(project, calls))
    assert result['job_ids'] == {'test': '9000', 'prepare': '9001', 'train': '9002'}
    assert not any(arg.startswith('--dependency=') for arg in calls[0]['command'])
    assert '--dependency=afterok:9000' in calls[1]['command']
    assert '--dependency=afterok:9001' in calls[2]['command']
    attempt = json.loads(next((project / '.paper3-hpc/start').glob('attempt-*.json')).read_text())
    assert len(attempt['submitted_jobs']) == 3 and all(row['state'] == 'SUBMISSION_RECEIPT_VERIFIED' for row in attempt['submission_intents'])
    assert 'monitor.py' in result['monitor_command'] and not result['scientific_completion_claimed']


@pytest.mark.parametrize('response', [SimpleNamespace(returncode=1, stdout=''), SimpleNamespace(returncode=0, stdout='unrecognized')])
def test_queue_failure_or_unrecognized_rows_block_before_fetch(project, response):
    with pytest.raises(launch.StartError, match='squeue'):
        launch.start(args(project), runner=lambda *args, **kwargs: response,
                     verifier=forbidden, preparer=forbidden, submitter=forbidden)


def test_data_not_ready_never_submits_and_records_minimal_gap(project):
    result = launch.start(args(project), runner=runner(), verifier=verifier,
        preparer=lambda *args, **kwargs: {'status': 'failed', 'error_code': 'R2_READ_FAILED',
            'failure_detail': {'phase': 'local_publish', 'errno': errno.ENOSPC}, 'minimal_gap': 'inspect data receipt'},
        submitter=forbidden)
    assert result['status'] == 'DATA_NOT_READY_NO_JOBS_SUBMITTED' and result['job_ids'] == {}
    assert result['error_code'] == 'R2_READ_FAILED'
    assert result['failure_detail'] == {'phase': 'local_publish', 'errno': errno.ENOSPC}
    assert json.loads((project / result['attempt']).read_text())['submission_intents'] == []


def test_failed_data_only_attempt_can_retry_resumable_original_data_stage(project):
    launch.start(args(project), runner=runner(), verifier=verifier,
                 preparer=lambda *args, **kwargs: {'status': 'blocked'}, submitter=forbidden)
    calls = []
    result = launch.start(args(project), runner=runner(), verifier=verifier, preparer=ready,
                          submitter=submitter(project, calls))
    assert len(calls) == 3 and result['new_jobs_submitted']


def test_successful_chain_is_never_replayed_when_it_has_left_queue(project):
    calls = []
    launch.start(args(project), runner=runner(), verifier=verifier, preparer=ready, submitter=submitter(project, calls))
    result = launch.start(args(project), runner=runner(), verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert result['status'] == 'EXISTING_CHAIN_NO_RESUBMISSION'
    assert result['job_ids'] == {'test': '9000', 'prepare': '9001', 'train': '9002'}
    assert len(calls) == 3 and not result['new_jobs_submitted']


@pytest.mark.parametrize('with_receipt', [False, True])
def test_unknown_sbatch_response_preserves_intent_and_never_retries_empty_queue(project, with_receipt):
    calls = []
    real = submitter(project, calls, durable=with_receipt)
    def uncertain(request):
        if with_receipt:
            real(request)
        raise subprocess.TimeoutExpired(['sbatch'], 60)
    with pytest.raises(subprocess.TimeoutExpired):
        launch.start(args(project), runner=runner(), verifier=verifier, preparer=ready, submitter=uncertain)
    result = launch.start(args(project), runner=runner(), verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert result['status'] == ('EXISTING_PARTIAL_CHAIN_REQUIRES_REVIEW' if with_receipt else 'SUBMISSION_OUTCOME_UNCERTAIN_NO_RESUBMISSION')
    assert result['job_ids'] == ({'test': '9000'} if with_receipt else {})
    assert result['missing_stages'] == (['prepare', 'train'] if with_receipt else ['test', 'prepare', 'train'])
    assert not result['new_jobs_submitted']
    saved = json.loads(next((project / '.paper3-hpc/start').glob('attempt-*.json')).read_text())
    assert saved['submission_intents'][0]['state'] == 'SUBMISSION_INTENT_OUTCOME_UNKNOWN'


def test_failed_original_stage_requires_review_without_clearing_claim_or_refitting(project):
    calls = []
    launch.start(args(project), runner=runner(), verifier=verifier, preparer=ready, submitter=submitter(project, calls))
    launch.write(project / '.paper3-hpc/jobs/9000/stage-test.json', {'schema_version': 'paper3-hpc-stage-receipt-v1',
        'stage': 'test', 'job_id': '9000', 'returncode': 2})
    claim = project / 'runs/original_claim.json'; claim.parent.mkdir(); claim.write_text('original')
    result = launch.start(args(project), runner=runner(), verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert result['status'] == 'PRIOR_STAGE_FAILED_MANUAL_REVIEW_REQUIRED'
    assert result['failed_stages'] == [{'stage': 'test', 'job_id': '9000', 'returncode': 2,
                                      'receipt': '.paper3-hpc/jobs/9000/stage-test.json'}]
    assert claim.read_text() == 'original'


def test_project_submission_receipt_can_bind_renamed_active_job_and_historical_train_blocks_restart(project):
    request = launch.requests(args(project), project, launch.submit.load_config(project / 'hpc/resources.json'))[-1]
    receipt = {**request, 'status': 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE', 'job_id': '777777'}
    launch.write(project / '.paper3-hpc/submissions/777777-aaaaaaaa.json', receipt)
    active = launch.start(args(project), runner=runner('777777|renamed|RUNNING|/different/workdir\n'),
                          verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert active['active_jobs'][0]['binding'] == 'root_submission_receipt'
    closed = launch.start(args(project), runner=runner(), verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert closed['status'] == 'EXISTING_TRAIN_SUBMISSION_REQUIRES_REVIEW' and closed['job_ids'] == ['777777']


def test_source_identity_change_during_data_fetch_blocks_all_submissions(project):
    identities = iter([{'manifest_sha256': 'a'*64}, {'manifest_sha256': 'b'*64}])
    with pytest.raises(launch.StartError, match='source_identity_changed'):
        launch.start(args(project), runner=runner(), verifier=lambda root: next(identities), preparer=ready, submitter=forbidden)


def test_active_work_appearing_while_fetching_prevents_second_chain(project):
    queues = iter(['', f'888888|paper3-train|PENDING|{project}\n'])
    base = runner()
    def query(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=next(queues)) if command[0] == 'squeue' else base(command, **kwargs)
    result = launch.start(args(project), runner=query, verifier=verifier, preparer=ready, submitter=forbidden)
    assert result['status'] == 'ACTIVE_JOBS_APPEARED_NO_NEW_SUBMISSION'
    assert result['active_jobs'][0]['job_id'] == '888888'


def test_code_change_between_true_submissions_preserves_first_job_and_blocks_remaining_stages(project):
    count = {'identity': 0}
    def changing(root):
        count['identity'] += 1
        return {'manifest_sha256': ('a' if count['identity'] <= 3 else 'b') * 64}
    calls = []
    with pytest.raises(launch.StartError, match='source_identity_changed_before_submission'):
        launch.start(args(project), runner=runner(), verifier=changing, preparer=ready, submitter=submitter(project, calls))
    assert len(calls) == 1 and calls[0]['stage'] == 'test'
    result = launch.start(args(project), runner=runner(), verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert result['status'] == 'EXISTING_PARTIAL_CHAIN_REQUIRES_REVIEW'
    assert result['job_ids'] == {'test': '9000'} and result['missing_stages'] == ['prepare', 'train']


def test_failed_chain_with_queued_afterok_dependents_reports_failed_original_stage(project):
    calls = []
    launch.start(args(project), runner=runner(), verifier=verifier, preparer=ready, submitter=submitter(project, calls))
    launch.write(project / '.paper3-hpc/jobs/9000/stage-test.json', {'schema_version': 'paper3-hpc-stage-receipt-v1',
        'stage': 'test', 'job_id': '9000', 'returncode': 2})
    result = launch.start(args(project), runner=runner(f'9001|paper3-prepare|PENDING|{project}\n9002|paper3-train|PENDING|{project}\n'),
        verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert result['status'] == 'EXISTING_CHAIN_HAS_FAILED_STAGE_REVIEW_REQUIRED'
    assert result['failed_stages'][0]['job_id'] == '9000'
    assert 'afterok' in result['minimal_gap'] and not result['new_jobs_submitted']


def test_real_prepare_data_api_accepts_progress_and_persists_its_own_attempt(project, monkeypatch, capsys):
    # Real deployment API/locks/persistence; data I/O and scheduler are explicit
    # software doubles, never approved data evidence or an actual submission.
    from hpc import stage_data
    def metadata_run(root, archive_root, release, *, progress, persist, **kwargs):
        assert kwargs['mode'] == 'fetch' and kwargs['execute'] and kwargs['network_host']
        progress({'phase': 'SOFTWARE_NO_TRANSFER', 'bytes_read': 0})
        result = {'status': 'ready', 'scientific_fits': 0}
        persist(result)
        return result
    monkeypatch.setattr(stage_data, '_run', metadata_run)
    monkeypatch.setattr(stage_data, 'load_release', lambda: {'SOFTWARE_TEST_DOUBLE': True})
    calls = []
    result = launch.start(args(project), runner=runner(), verifier=verifier,
        preparer=stage_data.prepare_data, submitter=submitter(project, calls))
    assert result['job_ids']['train'] == '9002' and len(calls) == 3
    assert 'SOFTWARE_NO_TRANSFER' in capsys.readouterr().out
    assert len(list((project / '.paper3-hpc/data').glob('attempt-*/receipt.json'))) == 1
    assert '--watch' not in result['monitor_command']


def test_main_reports_safe_errno_without_exception_text_or_discarding_intent(project, monkeypatch, capsys):
    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, 'SOFTWARE_SECRET_NEVER_PRINT')
    monkeypatch.setattr(launch, 'start', full_disk)
    assert launch.main(['--root', str(project), '--execute', '--network-host']) == 2
    output = capsys.readouterr().out
    value = json.loads(output)
    assert value['failure_detail'] == {'errno': errno.ENOSPC, 'errno_name': 'ENOSPC'}
    assert 'SOFTWARE_SECRET' not in output


def staged_packages(project, monkeypatch):
    from hpc import stage_data
    objects = {str(i): {'path': 'archives/fixed-' + str(i), 'bytes': i + 1} for i in range(5)}
    for row in objects.values():
        path = project / '.hpc/packages' / row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * row['bytes'])
    monkeypatch.setattr(stage_data, 'load_release', lambda: {'objects': objects})


def test_staged_plan_defers_data_gate_without_lock_network_or_submission(project):
    result = launch.start(args(project, staged_data=True, network_host=False, execute=False),
        runner=forbidden, verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert result['data_gate'] == 'SLURM_PREPARE_NOT_YET_VERIFIED'
    assert result['execute_requires'] == ['--execute', '--staged-data']
    assert not (project / '.paper3-hpc').exists()


def test_staged_mode_submits_same_gated_chain_without_foreground_data_processing(project, monkeypatch):
    staged_packages(project, monkeypatch)
    calls = []
    result = launch.start(args(project, staged_data=True, network_host=False),
        runner=runner(), verifier=verifier, preparer=forbidden, submitter=submitter(project, calls))
    assert result['job_ids'] == {'test': '9000', 'prepare': '9001', 'train': '9002'}
    assert '--dependency=afterok:9000' in calls[1]['command']
    assert '--dependency=afterok:9001' in calls[2]['command']
    proof = json.loads((project / result['attempt']).read_text())
    assert proof['data_receipt']['status'] == 'DEFERRED_TO_SLURM_PREPARE_NOT_YET_VERIFIED'
    assert proof['data_mode'] == 'STAGED_INPUTS_CPU_SLURM_VERIFICATION'
    assert not list((project / '.paper3-hpc/data').glob('attempt-*'))


def test_staged_retry_preserves_interrupted_empty_attempt_and_original_data_receipt(project, monkeypatch):
    staged_packages(project, monkeypatch)
    old = project / '.paper3-hpc/start/attempt-11111111111111111111111111111111.json'
    original = {'schema_version': launch.SCHEMA, 'project': str(project),
        'status': 'FETCHING_FROZEN_DATA', 'submission_intents': [], 'submitted_jobs': []}
    launch.write(old, original)
    data = project / '.paper3-hpc/data/attempt-old/receipt.json'
    launch.write(data, {'status': 'running', 'completed_groups': 1538})
    before = data.read_bytes()
    result = launch.start(args(project, staged_data=True, network_host=False), runner=runner(),
        verifier=verifier, preparer=forbidden, submitter=submitter(project, []))
    assert result['new_jobs_submitted'] is True
    assert json.loads(old.read_text()) == original and data.read_bytes() == before


def test_staged_existing_data_writer_blocks_before_queue_and_scheduling(project):
    with launch.staged_owner(project):
        with pytest.raises(launch.StartError, match='existing_data_coordinator'):
            launch.start(args(project, staged_data=True, network_host=False), runner=forbidden,
                verifier=forbidden, preparer=forbidden, submitter=forbidden)


def test_staged_mode_still_refuses_nested_compute_submission(project, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '456')
    with pytest.raises(launch.StartError, match='not_compute_job'):
        launch.start(args(project, staged_data=True, network_host=False), runner=forbidden)
    assert not (project / '.paper3-hpc').exists()


@pytest.mark.parametrize('mutation', ['missing', 'size', 'symlink'])
def test_staged_missing_or_changed_fixed_package_stops_before_any_sbatch(project, monkeypatch, mutation):
    staged_packages(project, monkeypatch)
    path = project / '.hpc/packages/archives/fixed-0'
    path.unlink()
    if mutation == 'size':
        path.write_bytes(b'wrong-length')
    elif mutation == 'symlink':
        target = project / 'another-file'
        target.write_bytes(b'x')
        path.symlink_to(target)
    with pytest.raises(launch.StartError):
        launch.start(args(project, staged_data=True, network_host=False), runner=runner(),
            verifier=verifier, preparer=forbidden, submitter=forbidden)
    attempt = json.loads(next((project / '.paper3-hpc/start').glob('attempt-*.json')).read_text())
    assert attempt['submission_intents'] == [] and attempt['error_type'] == 'StartError'
    assert 'error_code' in attempt


def test_staged_mode_keeps_unknown_previous_submission_blocked(project, monkeypatch):
    calls = []
    def uncertain(request):
        calls.append(request)
        raise subprocess.TimeoutExpired('sbatch', 60)
    with pytest.raises(subprocess.TimeoutExpired):
        launch.start(args(project), runner=runner(), verifier=verifier,
            preparer=ready, submitter=uncertain)
    result = launch.start(args(project, staged_data=True, network_host=False), runner=runner(),
        verifier=forbidden, preparer=forbidden, submitter=forbidden)
    assert result['status'] == 'SUBMISSION_OUTCOME_UNCERTAIN_NO_RESUBMISSION'
    assert len(calls) == 1


def test_staged_mode_preflight_failure_persists_actual_control_code(project, monkeypatch):
    staged_packages(project, monkeypatch)
    def reject(request):
        raise launch.submit.SubmissionError('Slurm_test_only_rejected_request_no_job_submitted')
    with pytest.raises(launch.submit.SubmissionError):
        launch.start(args(project, staged_data=True, network_host=False), runner=runner(),
            verifier=verifier, preparer=forbidden, submitter=reject)
    attempt = json.loads(next((project / '.paper3-hpc/start').glob('attempt-*.json')).read_text())
    assert attempt['submission_intents'][0]['state'] == 'NOT_SUBMITTED_BLOCKED'
    assert attempt['error_code'] == 'Slurm_test_only_rejected_request_no_job_submitted'


def test_arbitrary_external_exception_text_is_not_saved(project):
    def broken(*a, **kw):
        raise RuntimeError('SOFTWARE_SECRET_NEVER_PERSIST')
    with pytest.raises(RuntimeError):
        launch.start(args(project), runner=runner(), verifier=verifier, preparer=broken, submitter=forbidden)
    raw = next((project / '.paper3-hpc/start').glob('attempt-*.json')).read_text()
    assert 'SOFTWARE_SECRET' not in raw and json.loads(raw)['error_type'] == 'RuntimeError'
