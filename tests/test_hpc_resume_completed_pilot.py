"""Completed-pilot recovery tests use only local metadata and scheduler doubles."""
import importlib.util
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hpc import resume_completed_pilot as resume, resume_prefit_train as prefit, start, submit

ROOT = Path(__file__).resolve().parents[1]
USER = pwd.getpwuid(os.getuid()).pw_name
OLD = {'test': '767526', 'prepare': '767528', 'train': '767530'}
PREVIOUS = {'test': '767956', 'train': '767958'}


@pytest.fixture
def project(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('completed_pilot_original_chain_fixture',
        Path(__file__).with_name('test_hpc_resume_prefit_train.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = module.project.__wrapped__(tmp_path, monkeypatch)
    for name in resume.FROZEN_CODE:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes((ROOT / name).read_bytes())
    ancestor = prefit.original_binding(root, OLD['train'])
    source = {'source_commit': '1'*40, 'code_manifest_sha256': 'd'*64}
    config = submit.load_config(root / 'hpc/resources.json')
    intents = []
    for stage in resume.STAGES:
        requests = prefit.requests(options(root), root, config, account=ancestor['account'],
            dependency='afterok:' + PREVIOUS['test'] if stage == 'train' else None)
        request = next(r for r in requests if r['stage'] == stage)
        request.update(start_source_identity=source, prefit_resume_attempt_id='2'*32,
                       prefit_resume_failed_train_job=OLD['train'])
        row = {**request, 'status': 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE', 'job_id': PREVIOUS[stage]}
        start.write(root / '.paper3-hpc/submissions' / (PREVIOUS[stage] + '-cccccccc.json'), row)
        intents.append({'stage': stage, 'state': 'SUBMISSION_RECEIPT_VERIFIED', 'request': request, 'receipt': row})
        terminal = {'schema_version': 'paper3-hpc-stage-receipt-v1', 'stage': stage, 'job_id': PREVIOUS[stage],
                    'returncode': 2 if stage == 'train' else 0, 'scientific_completion_claimed': False,
                    'external_query_accessed': False}
        if stage == 'test':
            terminal['result'] = {'status': 'ALL_SOFTWARE_TESTS_PASSED', 'returncode': 0,
                                 'software_identity': {'manifest_sha256': source['code_manifest_sha256']}}
        else:
            terminal['frozen_data_verification'] = start.read(root / '.paper3-hpc/jobs/767528/stage-prepare.json')['result']
        start.write(root / '.paper3-hpc/jobs' / PREVIOUS[stage] / ('stage-' + stage + '.json'), terminal)
    start.write(root / '.paper3-hpc/prefit-train-resumes/train-767530.json', {
        'schema_version': prefit.SCHEMA, 'project': str(root), 'failed_train_job': OLD['train'],
        'attempt_id': '2'*32, 'uid': os.getuid(), 'user': USER, 'original_binding': ancestor,
        'source_identity': source, 'prefit_inspection': {
            'status': 'KNOWN_ZERO_UPDATE_PREFIT_FAILURE_SAME_CLAIM_ONLY',
            'scientific_updates_performed': 0, 'new_pilot_claim_authorized': False},
        'intents': intents, 'status': 'SLURM_PREFIT_RETEST_TRAIN_CHAIN_SUBMITTED_NOT_COMPLETE',
        'scientific_completion_claimed': False, 'scientific_budget_changed': False})
    start.write(root / '.paper3-hpc/jobs/767958/campaign.log', {
        'status': 'BLOCKED_PRESERVE_EXISTING_JOBS', 'error_type': 'CampaignError', 'reason': prefit.PILOT_REASON,
        'completion_claimed': False, 'external_query_accessed': False})
    _link_allocated_prefit_proof(root)
    return root


def _link_allocated_prefit_proof(root):
    from hpc import pilot_prefit
    previous = start.read(root / '.paper3-hpc/prefit-train-resumes/train-767530.json')
    proof_path = root / '.paper3-hpc/jobs/767958/pilot_prefit_resume.json'
    start.write(proof_path, {'schema_version': pilot_prefit.SCHEMA,
        'status': 'PREFIT_SAME_CLAIM_RESUME_PREPARED', 'project': str(root), 'job_id': '767958',
        'facts': previous['prefit_inspection']})
    receipt_path = root / pilot_prefit.PILOT / 'engineering_receipt.json'
    receipt = start.read(receipt_path) if receipt_path.exists() else {'scope': 'SOFTWARE_RECEIPT_DOUBLE_ONLY'}
    receipt['prefit_resume'] = {'proof': prefit.descriptor(root, proof_path)}
    start.write(receipt_path, receipt)


def options(root, **changes):
    result = dict(root=root, python=Path(sys.executable), failed_train_job='767958', execute=True,
                  dry_run=False, cpus=None, mem_gib=None, gpus=None, time=None)
    result.update(changes)
    return SimpleNamespace(**result)


def inspector(root):
    return {'status': 'COMPLETED_200_PILOT_METADATA_FINALIZATION_REQUIRED', 'project': str(root),
            'actual_completed_updates': 200, 'exact_resume_claimed': True,
            'numerical_evidence_not_yet_independently_verified': True,
            'code_files': resume.frozen_code(root), 'prior_pilot_wall_seconds': 950.5,
            'files': {'wall': prefit.descriptor(root, root / 'runs/original_wall.json')}}


def verifier(root):
    return {'manifest_sha256': '3'*64, 'numerical_kernel_overrides': 0}


class Scheduler:
    def __init__(self, root):
        self.root = root; self.calls = []; self.queue = ''; self.states = {}; self.owner = USER
        self.nodes = {'test': 'comput1', 'train': 'gpu3'}; self.extra = ''; self.reverse = False

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[0] == 'git':
            assert cmd == ['git', 'rev-parse', '--verify', 'HEAD']
            return SimpleNamespace(returncode=0, stdout='4'*40+'\n')
        if cmd[0] == 'sacct':
            assert '--allocations' in cmd and '--duplicates' in cmd
            assert '--jobs=767956,767958' in cmd and '--format=' + resume.FORMAT in cmd
            rows = []
            for index, (stage, job) in enumerate(PREVIOUS.items()):
                rows.append('|'.join([job, self.states.get(stage, 'FAILED' if stage == 'train' else 'COMPLETED'),
                    '2:0' if stage == 'train' else '0:0', '2026-10-22T01:00:01', '2026-10-22T02:00:00',
                    '3599', 'zhuziyuan', self.owner, 'paper3-' + stage, str(self.root), str(201+index),
                    '2026-10-22T01:00:00', self.nodes[stage]]))
            if self.reverse: rows.reverse()
            return SimpleNamespace(returncode=0, stdout='\n'.join(rows) + '\n' + self.extra)
        if cmd[0] == 'squeue':
            return SimpleNamespace(returncode=0, stdout=self.queue)
        pytest.fail('Unexpected scheduler action: ' + cmd[0])


class API:
    def __init__(self, root):
        self.root = root; self.calls = []; self.crash_before = False; self.crash_after = False

    def __call__(self, request):
        self.calls.append(request)
        if self.crash_before:
            self.crash_before = False
            raise subprocess.TimeoutExpired('SOFTWARE_DOUBLE', 1)
        job = str(820000 + len(self.calls))
        row = {**request, 'status': 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE', 'job_id': job}
        start.write(self.root / '.paper3-hpc/submissions' / (job + '-dddddddd.json'), row)
        if self.crash_after:
            self.crash_after = False
            raise subprocess.TimeoutExpired('SOFTWARE_DOUBLE', 1)
        return row


def invoke(root, scheduler=None, api=None, **changes):
    return resume.resume(options(root, **changes), runner=scheduler or Scheduler(root),
                         verifier=verifier, inspector=inspector, submitter=api or API(root))


def originals(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()
            and not p.name.endswith('.lock') and 'completed-pilot-resumes' not in p.parts
            and not ('submissions' in p.parts and p.name.startswith('820'))}


def test_recovery_derives_prior_CPU_and_submits_only_test_then_same_node_train(project):
    before = originals(project); api = API(project); scheduler = Scheduler(project)
    result = invoke(project, scheduler, api)
    assert result['job_ids'] == {'test': '820001', 'train': '820002'}
    assert result['pilot_optimizer_updates_requested'] == 0
    assert result['prepare_reused_full_SHA_rechecked_by_train'] is True
    assert [r['stage'] for r in api.calls] == ['test', 'train']
    assert '--dependency=afterok:820001' in api.calls[1]['command']
    assert '--nodelist=gpu3' in api.calls[1]['command']
    assert '--nodelist=gpu3' in api.calls[1]['scheduler_preflight']
    assert not any(c.startswith('--nodelist=') for c in api.calls[0]['command'])
    attempt = start.read(project / result['attempt'])
    assert attempt['predecessor_binding']['jobs'] == PREVIOUS
    assert attempt['predecessor_binding']['ancestor']['jobs'] == OLD
    assert attempt['completed_pilot_inspection']['actual_completed_updates'] == 200
    assert all(r['completed_pilot_resume_proof_sha256'] == attempt['proof_sha256'] for r in api.calls)
    assert all(r['scientific_budget_changed'] is False and r['scientific_threads_changed'] is False for r in api.calls)
    assert originals(project) == before


@pytest.mark.parametrize('execute,dry_run', [(False, False), (True, True)])
def test_plan_does_not_read_pilot_query_scheduler_create_locks_or_write(project, execute, dry_run):
    before = originals(project)
    def forbidden(*args, **kwargs): pytest.fail('Readonly plan called execution dependency')
    result = resume.resume(options(project, execute=execute, dry_run=dry_run), runner=forbidden,
                           verifier=forbidden, inspector=forbidden, submitter=forbidden)
    assert result['status'] == 'PLAN_COMPLETED_PILOT_RETEST_TRAIN_NO_ACTION'
    assert originals(project) == before and not list(project.rglob('*.lock'))


def test_repeat_returns_same_two_actual_IDs_while_new_controller_owns_project(project):
    from hpc.run_stage import owner_lock
    api = API(project); first = invoke(project, api=api)
    def forbidden(*args, **kwargs): pytest.fail('Already submitted chain inspected mutable pilot or resubmitted')
    with owner_lock(project):
        result = resume.resume(options(project), runner=forbidden, verifier=forbidden,
                               inspector=forbidden, submitter=forbidden)
    assert result['status'] == 'EXISTING_COMPLETED_PILOT_RESUME_NO_RESUBMISSION'
    assert result['job_ids'] == first['job_ids'] and len(api.calls) == 2


def test_unknown_sbatch_outcome_without_durable_receipt_cannot_repeat(project):
    api = API(project); api.crash_before = True
    with pytest.raises(subprocess.TimeoutExpired): invoke(project, api=api)
    with pytest.raises(resume.ResumeError, match='unknown_submission_outcome'): invoke(project, api=api)
    assert len(api.calls) == 1


def test_ACK_crash_recovers_actual_CPU_receipt_and_only_submits_missing_train(project):
    api = API(project); api.crash_after = True; scheduler = Scheduler(project)
    with pytest.raises(subprocess.TimeoutExpired): invoke(project, scheduler, api)
    scheduler.queue = '820001|paper3-test|RUNNING|' + str(project) + '\n'
    result = invoke(project, scheduler, api)
    assert result['job_ids'] == {'test': '820001', 'train': '820002'}
    assert [r['stage'] for r in api.calls] == ['test', 'train']


def test_predecessor_ACK_crash_is_bound_through_actual_receipts_without_rewriting_history(project):
    path = project / '.paper3-hpc/prefit-train-resumes/train-767530.json'
    previous = start.read(path); previous['status'] = 'SUBMITTING_TRAIN'
    previous['intents'][1].update(state='SUBMISSION_INTENT_OUTCOME_UNKNOWN')
    del previous['intents'][1]['receipt']; start.write(path, previous)
    before = path.read_bytes(); result = invoke(project)
    assert result['job_ids']['train'] == '820002' and path.read_bytes() == before


def test_immediately_scheduled_test_can_acquire_controller_lock(project):
    from hpc.run_stage import owner_lock
    api = API(project); acquired = []
    def immediate(request):
        with owner_lock(project): acquired.append(request['stage'])
        return api(request)
    invoke(project, api=immediate)
    assert acquired == ['test', 'train']


def test_active_other_project_job_blocks_without_new_attempt(project):
    scheduler = Scheduler(project); scheduler.queue = '999999|paper3-train|RUNNING|' + str(project) + '\n'
    api = API(project)
    with pytest.raises(resume.ResumeError, match='active_project'): invoke(project, scheduler, api)
    assert not api.calls and not (project / resume.DIRECTORY).exists()


@pytest.mark.parametrize('field,value', [('state', 'RUNNING'), ('owner', 'foreign'), ('node', 'gpu[1-2]')])
def test_scheduler_must_confirm_terminal_original_actor_and_single_GPU_node(project, field, value):
    scheduler = Scheduler(project); api = API(project)
    if field == 'state': scheduler.states['train'] = value
    elif field == 'owner': scheduler.owner = value
    else: scheduler.nodes['train'] = value
    with pytest.raises(resume.ResumeError): invoke(project, scheduler, api)
    assert not api.calls


def test_duplicate_scheduler_history_is_rejected(project):
    scheduler = Scheduler(project)
    output = scheduler(['sacct', '--allocations', '--duplicates', '--jobs=767956,767958', '--format=' + resume.FORMAT]).stdout
    scheduler.extra = output.splitlines()[0] + '\n'
    with pytest.raises(resume.ResumeError, match='history_ambiguous'): invoke(project, scheduler)


@pytest.mark.parametrize('target,field,value', [
    ('767956', 'returncode', 1), ('767958', 'returncode', 0),
    ('767958', 'external_query_accessed', True)])
def test_local_predecessor_stage_receipts_must_match_actual_failure_chain(project, target, field, value):
    stage = 'test' if target == PREVIOUS['test'] else 'train'
    path = project / '.paper3-hpc/jobs' / target / ('stage-' + stage + '.json')
    row = start.read(path); row[field] = value; start.write(path, row)
    with pytest.raises(resume.ResumeError, match='terminal_receipt'): invoke(project)


def test_changed_frozen_data_or_old_CPU_identity_cannot_authorize_recovery(project):
    path = project / '.paper3-hpc/jobs/767958/stage-train.json'
    row = start.read(path); row['frozen_data_verification']['closure_sha256'] = '0'*64; start.write(path, row)
    with pytest.raises(resume.ResumeError, match='frozen_data_changed'): invoke(project)


def test_unrelated_campaign_error_is_not_interpreted_as_finalization_failure(project):
    path = project / '.paper3-hpc/jobs/767958/campaign.log'
    row = start.read(path); row['reason'] = 'arbitrary error secret'; start.write(path, row)
    with pytest.raises(resume.ResumeError, match='exact_failed_campaign'): invoke(project)


@pytest.mark.parametrize('field,value', [('actual_completed_updates', 199), ('actual_completed_updates', True),
    ('exact_resume_claimed', False), ('numerical_evidence_not_yet_independently_verified', False)])
def test_non200_or_unverified_checkpoint_conditions_never_submit(project, field, value):
    api = API(project)
    def invalid(root):
        row = inspector(root); row[field] = value; return row
    with pytest.raises(resume.ResumeError, match='strict_finalization_proof'):
        resume.resume(options(project), runner=Scheduler(project), verifier=verifier, inspector=invalid, submitter=api)
    assert not api.calls


@pytest.mark.parametrize('name', list(resume.FROZEN_CODE))
def test_runner_and_source_training_checkpoint_identity_must_stay_byte_exact(project, name):
    (project / name).write_bytes(b'changed checkpoint code')
    with pytest.raises(resume.ResumeError, match='frozen_checkpoint_code_changed'): invoke(project)


def test_train_resource_overrides_leave_CPU_profile_and_same_node_binding(project):
    api = API(project); invoke(project, api=api, cpus=20, mem_gib=96, gpus=1, time='12:00:00')
    assert api.calls[0]['resources']['cpus'] == 16 and api.calls[0]['resources']['mem_gib'] == 64
    assert api.calls[1]['resources']['cpus'] == 20 and api.calls[1]['resources']['mem_gib'] == 96
    assert api.calls[1]['resources']['time'] == '12:00:00' and api.calls[1]['resources']['node_list'] == 'gpu3'


def test_code_change_between_submissions_keeps_actual_CPU_and_does_not_submit_train(project):
    calls = []; api = API(project)
    def changing(root):
        calls.append(root)
        return {'manifest_sha256': ('3' if len(calls) < 3 else '9')*64}
    with pytest.raises(resume.ResumeError, match='source_changed_before_submission'):
        resume.resume(options(project), runner=Scheduler(project), verifier=changing, inspector=inspector, submitter=api)
    assert len(api.calls) == 1 and api.calls[0]['stage'] == 'test'


def test_partial_resume_does_not_submit_train_after_new_CPU_failure(project):
    api = API(project); api.crash_after = True
    with pytest.raises(subprocess.TimeoutExpired): invoke(project, api=api)
    start.write(project / '.paper3-hpc/jobs/820001/stage-test.json', {'returncode': 1})
    with pytest.raises(resume.ResumeError, match='CPU_test_failed'): invoke(project, api=api)
    assert len(api.calls) == 1


def test_allocated_proof_requires_actual_current_train_and_new_CPU_pass(project, monkeypatch):
    result = invoke(project)
    monkeypatch.setenv('SLURM_JOB_ID', result['job_ids']['train'])
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'gpu3')
    path = project / '.paper3-hpc/jobs/820001/stage-test.json'
    start.write(path, {'schema_version': 'paper3-hpc-stage-receipt-v1', 'stage': 'test', 'job_id': '820001',
        'returncode': 0, 'result': {'status': 'ALL_SOFTWARE_TESTS_PASSED',
                                 'software_identity': {'manifest_sha256': '3'*64}}})
    proof = resume.allocated_resume_proof(project, inspector=inspector, verified_identity=verifier(project))
    assert proof['job_id'] == '820002' and proof['failed_train_job'] == '767958'
    assert proof['numerical_updates_authorized'] is False
    with pytest.raises(resume.ResumeError, match='allocated_job_binding'):
        resume.allocated_resume_proof(project, job_id='999999', inspector=inspector, verified_identity=verifier(project))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        resume.allocated_resume_proof(project, inspector=inspector, verified_identity=verifier(project))


def test_allocated_proof_rejects_changed_original_evidence(project, monkeypatch):
    invoke(project); monkeypatch.setenv('SLURM_JOB_ID', '820002')
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'gpu3')
    (project / 'runs/original_wall.json').write_text('{"charged":0}')
    with pytest.raises(resume.ResumeError, match='allocated_evidence_changed'):
        resume.allocated_resume_proof(project, inspector=inspector, verified_identity=verifier(project))


def test_queued_checkout_change_cannot_reuse_prior_CPU_pass_for_different_current_code(project, monkeypatch):
    invoke(project); monkeypatch.setenv('SLURM_JOB_ID', '820002')
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'gpu3')
    with pytest.raises(resume.ResumeError, match='current_code_identity_changed'):
        resume.allocated_resume_proof(project, inspector=lambda _: pytest.fail('Changed code must block first'),
                                     verified_identity={'manifest_sha256': '9'*64})


@pytest.mark.parametrize('change', ['job_id', 'facts', 'receipt_proof'])
def test_completed_pilot_is_explicitly_linked_to_selected_failed_job_prefit_proof(project, change):
    proof_path = project / '.paper3-hpc/jobs/767958/pilot_prefit_resume.json'
    proof = start.read(proof_path)
    if change == 'job_id':
        proof['job_id'] = '999999'; start.write(proof_path, proof)
    elif change == 'facts':
        proof['facts']['scientific_updates_performed'] = 1; start.write(proof_path, proof)
    else:
        from hpc import pilot_prefit
        path = project / pilot_prefit.PILOT / 'engineering_receipt.json'
        receipt = start.read(path); receipt['prefit_resume']['proof']['sha256'] = '0'*64
        start.write(path, receipt)
    with pytest.raises(resume.ResumeError, match='selected_failed_job_proof'): invoke(project)


def test_normal_untagged_allocated_job_returns_no_recovery_authority(project, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '767530')
    assert resume.allocated_resume_proof(project, inspector=lambda _: pytest.fail('Normal job inspected pilot')) is None


def test_tagged_train_missing_attempt_or_wrong_node_never_falls_back_to_normal_mode(project, monkeypatch):
    result = invoke(project); monkeypatch.setenv('SLURM_JOB_ID', '820002')
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'gpu99')
    with pytest.raises(resume.ResumeError, match='original_GPU_node'):
        resume.allocated_resume_proof(project, inspector=inspector, verified_identity=verifier(project))
    (project / result['attempt']).unlink()
    with pytest.raises(resume.ResumeError, match='allocated_durable_resume'):
        resume.allocated_resume_proof(project, inspector=inspector, verified_identity=verifier(project))


def test_unknown_new_intent_or_tampered_proof_cannot_be_reconciled(project):
    result = invoke(project); path = project / result['attempt']; value = start.read(path)
    value['completed_pilot_inspection']['actual_completed_updates'] = 1
    with pytest.raises(resume.ResumeError, match='stored_proof_changed'): resume.reconciled(project, value)


def test_old_zero_update_entry_remains_strictly_bound_to_original_three_job_chain(project):
    with pytest.raises(prefit.ResumeError, match='exact_original_chain'):
        prefit.original_binding(project, PREVIOUS['train'])


def test_cli_redacts_unknown_exception_string(project, monkeypatch, capsys):
    def fail(*args, **kwargs): raise ValueError('SDK password=secret absolute/path')
    monkeypatch.setattr(resume, 'resume', fail)
    assert resume.main(['--root', str(project), '--failed-train-job', '767958', '--execute']) == 2
    output = capsys.readouterr().out
    assert 'secret' not in output and json.loads(output)['reason'] == 'ValueError'


def test_cli_preserves_safe_finalizer_classification_code(project, monkeypatch, capsys):
    from hpc.pilot_finalize import FinalizeError
    def fail(*args, **kwargs): raise FinalizeError('pilot_finalize_original_code_identity_changed')
    monkeypatch.setattr(resume, 'resume', fail)
    assert resume.main(['--root', str(project), '--failed-train-job', '767958', '--execute']) == 2
    assert json.loads(capsys.readouterr().out)['reason'] == 'pilot_finalize_original_code_identity_changed'


def _actual_completed_fixture(project, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('completed_pilot_strict_inspection_fixture',
        Path(__file__).with_name('test_hpc_pilot_finalize.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    holder = tmp_path / 'strict completed pilot fixture'; holder.mkdir()
    metadata, _, _ = module.completed.__wrapped__(holder, monkeypatch)
    # The models are explicitly marked placeholders. This integration invokes
    # the real readonly classifier, never the independent deep promotion gate.
    for path in metadata.rglob('*'):
        if path.is_file():
            target = project / path.relative_to(metadata)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setenv('PAPER3_HPC_ROOT', str(project))
    _link_allocated_prefit_proof(project)
    return module.repair


def test_actual_completed_inspector_binds_200_counts_and_preserves_claims_checkpoint_bytes_and_wall(
        project, tmp_path, monkeypatch):
    _actual_completed_fixture(project, tmp_path, monkeypatch)
    before = originals(project); api = API(project)
    result = resume.resume(options(project), runner=Scheduler(project), verifier=verifier, submitter=api)
    attempt = start.read(project / result['attempt'])
    evidence = attempt['completed_pilot_inspection']
    assert evidence['actual_completed_updates'] == 200 and evidence['exact_resume_claimed'] is True
    assert evidence['numerical_evidence_not_yet_independently_verified'] is True
    assert evidence['campaign_charged_wall_seconds'] == 388.038
    assert evidence['files']['engineering_initial.pt']['sha256']
    assert originals(project) == before


def test_actual_completed_classifier_rejects_body_failure_before_any_new_jobs(project, tmp_path, monkeypatch):
    repair = _actual_completed_fixture(project, tmp_path, monkeypatch)
    path = project / repair.RECEIPT; value = start.read(path)
    value['failure_type'] = 'RuntimeError'; start.write(path, value)
    api = API(project)
    with pytest.raises(repair.FinalizeError):
        resume.resume(options(project), runner=Scheduler(project), verifier=verifier, submitter=api)
    assert not api.calls and not (project / resume.DIRECTORY).exists()
