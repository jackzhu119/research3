"""Failed prefit recovery uses metadata and scheduler doubles, never real jobs."""
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hpc import recover_failed_chain as recovery, start, submit

ROOT = Path(__file__).resolve().parents[1]
USER = pwd.getpwuid(os.getuid()).pw_name
IDS = {'test': '767434', 'prepare': '767436', 'train': '767438'}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    root = tmp_path / 'school project'; root.mkdir()
    config = submit.load_config(ROOT / 'hpc/resources.json')
    rows, intents = [], []
    source = {'source_commit': 'a'*40, 'code_manifest_sha256': 'b'*64}
    for index, stage in enumerate(start.STAGES):
        args = SimpleNamespace(stage=stage, root=root, python='/usr/bin/python', account=None,
                               partition=None, cpus=None, mem_gib=None, gpus=None, time=None,
                               dependency=None if index == 0 else 'afterok:' + rows[-1]['job_id'])
        request = {**submit.build_request(args, config), 'start_attempt_id': 'a'*32,
                   'start_source_identity': source}
        row = {**request, 'status': 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE',
               'job_id': IDS[stage], 'scientific_completion_claimed': False}
        rows.append(row)
        intents.append({'stage': stage, 'state': 'SUBMISSION_RECEIPT_VERIFIED', 'request': request})
        start.write(root / '.paper3-hpc/submissions' / (IDS[stage] + '-aaaaaaaa.json'), row)
    start.write(root / '.paper3-hpc/start' / ('attempt-' + 'a'*32 + '.json'),
                {'schema_version': start.SCHEMA, 'attempt_id': 'a'*32, 'project': str(root),
                 'status': 'SLURM_CHAIN_SUBMITTED_NOT_TRAINING_COMPLETE', 'source_identity': source,
                 'submitted_jobs': rows, 'submission_intents': intents})
    start.write(root / '.paper3-hpc/jobs/767434/stage-test.json',
                {'schema_version': 'paper3-hpc-stage-receipt-v1', 'stage': 'test',
                 'job_id': '767434', 'returncode': 2, 'status': 'STOPPED_OR_BLOCKED_NOT_COMPLETE',
                 'scientific_completion_claimed': False, 'external_query_accessed': False})
    return root


class Scheduler:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.states = {'767434': 'FAILED', '767436': 'PENDING', '767438': 'PENDING'}
        self.start = {'767434': '2026-10-21T17:00:00', '767436': 'Unknown', '767438': 'Unknown'}
        self.elapsed = {'767434': '1186', '767436': '0', '767438': '0'}
        self.end = {'767434': '2026-10-21T17:19:46', '767436': 'Unknown', '767438': 'Unknown'}
        self.normalize_cancelled_start = False
        self.owner = USER
        self.account = 'zhuziyuan'
        self.control_changes = {}
        self.cancel = True
        self.cancel_timeout_once = False
        self.queue = ''

    def __call__(self, command, **kwargs):
        assert kwargs == dict(stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              check=False, timeout=30)
        self.calls.append(command)
        if command[0] == 'sacct':
            lines = []
            for stage in start.STAGES:
                job = IDS[stage]
                lines.append('|'.join([job, self.states[job], '2:0' if stage == 'test' else '0:0',
                    self.start[job], self.elapsed[job], self.account, self.owner,
                    'paper3-' + stage, str(self.root),self.end[job]]))
            return SimpleNamespace(returncode=0, stdout='\n'.join(lines) + '\n')
        if command[0] == 'scontrol':
            job = command[-1]
            stage = next(stage for stage in start.STAGES if IDS[stage] == job)
            parent = IDS['test' if stage == 'prepare' else 'prepare']
            row = {'JobId': job, 'JobName': 'paper3-' + stage, 'UserId': f'{USER}({os.getuid()})',
                   'Account': 'zhuziyuan', 'JobState': self.states[job], 'RunTime': '00:00:00',
                   'Restarts': '0', 'Dependency': 'afterok:' + parent + '(unsatisfied)',
                   'WorkDir': str(self.root)}
            row.update(self.control_changes)
            return SimpleNamespace(returncode=0, stdout=' '.join(k+'='+v for k,v in row.items()))
        if command[0] == 'scancel':
            assert command[1:3] == ['--state=PENDING', '--user=' + str(os.getuid())]
            assert command[3] in {'767436', '767438'}
            if self.cancel and self.states[command[3]] == 'PENDING':
                self.states[command[3]] = 'CANCELLED'
                self.end[command[3]] = '2026-10-21T18:00:00'
                if self.normalize_cancelled_start:
                    self.start[command[3]] = self.end[command[3]]
            if self.cancel_timeout_once:
                self.cancel_timeout_once = False
                raise subprocess.TimeoutExpired(command, 30)
            return SimpleNamespace(returncode=0, stdout='')
        if command[0] == 'squeue':
            assert command == ['squeue','--noheader','--array','--user='+str(os.getuid()),'--format=%i|%T']
            return SimpleNamespace(returncode=0, stdout=self.queue)
        pytest.fail('unexpected scheduler command: ' + command[0])


def cancel_calls(scheduler):
    return [c for c in scheduler.calls if c[0] == 'scancel']


def originals(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*.json')
            if 'recoveries' not in p.parts}


def test_plan_observes_only_and_never_creates_lock_or_recovery_files(project):
    scheduler = Scheduler(project); before = originals(project)
    result = recovery.recover(project, '767434', runner=scheduler)
    assert result['status'] == 'PLAN_CANCEL_ONLY_NEVER_STARTED_DEPENDENTS'
    assert result['new_jobs_submitted'] is result['scientific_fit_performed'] is False
    assert originals(project) == before and not cancel_calls(scheduler)
    assert not (project / '.paper3-hpc/recoveries').exists()
    assert not (project / '.paper3-deployment').exists()
    assert not (project / '.paper3-hpc/start/owner.lock').exists()


def test_explicit_recovery_retires_only_pending_descendants_and_preserves_all_originals(project):
    marker = project / 'cache/original.npy'; marker.parent.mkdir(); marker.write_bytes(b'unchanged')
    ledger = project / 'runs/original-budget.json'; ledger.parent.mkdir(); ledger.write_text('{"charged": 19}')
    before = originals(project); scheduler = Scheduler(project)
    result = recovery.recover(project, '767434', execute=True, runner=scheduler, sleeper=lambda n: None)
    assert result['status'] == 'RECOVERY_VERIFIED_NO_FIT_REPLAY'
    assert [c[-1] for c in cancel_calls(scheduler)] == ['767438', '767436']
    assert result['new_jobs_submitted'] is result['scientific_budget_changed'] is False
    assert originals(project) == before and marker.read_bytes() == b'unchanged'
    excluded = recovery.superseded_attempts(project)
    assert list(excluded.values())[0]['job_ids'] == IDS
    assert result['binding']['attempt'] in excluded
    assert (project / '.paper3-hpc/recoveries/test-767434.intent.json').exists()


@pytest.mark.parametrize('state,start_time,elapsed', [('RUNNING','Unknown','0'),
    ('COMPLETED','Unknown','0'), ('PENDING','2026-10-21T17:00:00','0'), ('PENDING','Unknown','1')])
def test_running_started_or_completed_descendant_refuses_before_cancel(project, state, start_time, elapsed):
    scheduler = Scheduler(project)
    scheduler.states['767438'], scheduler.start['767438'], scheduler.elapsed['767438'] = state,start_time,elapsed
    with pytest.raises(recovery.RecoveryError, match='may_have_started'):
        recovery.recover(project, '767434', execute=True, runner=scheduler)
    assert not cancel_calls(scheduler)


@pytest.mark.parametrize('changes', [{'UserId':'foreign(9999)'}, {'WorkDir':'/other/project'},
    {'Dependency':'afterok:999999(unsatisfied)'}, {'Restarts':'1'}, {'Account':'foreign'},
    {'JobState':'RUNNING'}, {'RunTime':'00:00:01'}])
def test_live_identity_dependency_or_requeued_state_refuses_before_cancel(project, changes):
    scheduler=Scheduler(project); scheduler.control_changes=changes
    with pytest.raises(recovery.RecoveryError):
        recovery.recover(project, '767434', execute=True, runner=scheduler)
    assert not cancel_calls(scheduler)


def test_descendant_artifacts_refuse_before_any_scheduler_call(project):
    (project / '.paper3-hpc/jobs/767438').mkdir()
    scheduler=Scheduler(project)
    with pytest.raises(recovery.RecoveryError, match='execution_artifacts'):
        recovery.recover(project, '767434', execute=True, runner=scheduler)
    assert not scheduler.calls


@pytest.mark.parametrize('state,code', [('COMPLETED',0), ('RUNNING',2)])
def test_parent_must_be_scheduler_failed_as_well_as_original_receipt(project,state,code):
    scheduler=Scheduler(project); scheduler.states['767434']=state
    with pytest.raises(recovery.RecoveryError, match='failed_test_confirmation'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not cancel_calls(scheduler)


def test_original_failed_test_receipt_required_and_zero_not_accepted(project):
    path=project/'.paper3-hpc/jobs/767434/stage-test.json'
    value=json.loads(path.read_text()); value['returncode']=0; start.write(path,value)
    scheduler=Scheduler(project)
    with pytest.raises(recovery.RecoveryError,match='failed_CPU_test_receipt'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not scheduler.calls


def test_unknown_original_submission_intent_is_not_recoverable(project):
    path=next((project/'.paper3-hpc/start').glob('attempt-*.json'))
    value=json.loads(path.read_text()); value['submission_intents'][0]['state']='SUBMISSION_INTENT_OUTCOME_UNKNOWN'
    start.write(path,value); scheduler=Scheduler(project)
    with pytest.raises(recovery.RecoveryError,match='unknown_or_partial'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not scheduler.calls


def test_durable_submission_mismatch_not_recoverable(project):
    path=next((project/'.paper3-hpc/submissions').glob('767438-*.json'))
    value=json.loads(path.read_text()); value['python']='/other/python'; start.write(path,value)
    scheduler=Scheduler(project)
    with pytest.raises(recovery.RecoveryError,match='durable_submission'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not scheduler.calls


def test_sacct_identity_mismatch_stops_before_cancellation(project):
    scheduler=Scheduler(project); scheduler.owner='foreign'
    with pytest.raises(recovery.RecoveryError,match='job_identity_mismatch'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not cancel_calls(scheduler)


def test_unconfirmed_cancellation_retains_intent_but_never_final_proof(project):
    scheduler=Scheduler(project); scheduler.cancel=False
    with pytest.raises(recovery.RecoveryError,match='cancellation_unconfirmed'):
        recovery.recover(project,'767434',execute=True,runner=scheduler,sleeper=lambda n:None)
    parent=project/'.paper3-hpc/recoveries'
    assert (parent/'test-767434.intent.json').exists() and not (parent/'test-767434.json').exists()
    assert recovery.superseded_attempts(project)=={}


def test_cancel_timeout_reconciles_actual_never_started_terminal_state_on_retry(project):
    scheduler=Scheduler(project); scheduler.cancel_timeout_once=True
    with pytest.raises(recovery.RecoveryError,match='observation_unavailable'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert scheduler.states['767438']=='CANCELLED' and scheduler.states['767436']=='PENDING'
    result=recovery.recover(project,'767434',execute=True,runner=scheduler,sleeper=lambda n:None)
    assert result['status']=='RECOVERY_VERIFIED_NO_FIT_REPLAY'
    assert [c[-1] for c in cancel_calls(scheduler)]==['767438','767436']


def test_rerun_reuses_valid_proof_without_recancelling_or_submitting(project):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    count=len(cancel_calls(scheduler))
    result=recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert result['reused_existing_proof'] is True and len(cancel_calls(scheduler))==count


def test_still_queued_old_job_never_publishes_recovery_proof(project):
    scheduler=Scheduler(project); scheduler.queue='767438|PENDING\n'
    with pytest.raises(recovery.RecoveryError,match='still_queued'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not (project/'.paper3-hpc/recoveries/test-767434.json').exists()


def test_original_file_change_invalidates_previously_issued_exclusion(project):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    path=next((project/'.paper3-hpc/start').glob('attempt-*.json'))
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(recovery.RecoveryError,match='original_records_changed'):
        recovery.superseded_attempts(project)


def test_forged_final_claim_cannot_hide_a_descendant_that_started(project):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    path=project/'.paper3-hpc/recoveries/test-767434.json'
    value=json.loads(path.read_text()); value['accounting_after']['767438']['elapsed']='1'; start.write(path,value)
    with pytest.raises(recovery.RecoveryError,match='not_never_started'):
        recovery.superseded_attempts(project)


def test_compute_allocation_refused_before_scheduler_or_mutation(project,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','777')
    scheduler=Scheduler(project)
    with pytest.raises(recovery.RecoveryError,match='submission_host'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not scheduler.calls


def test_unrelated_notebook_in_same_account_queue_is_not_cancelled_or_blocked(project):
    scheduler=Scheduler(project); scheduler.queue='767288|RUNNING\n'
    value=recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert value['status']=='RECOVERY_VERIFIED_NO_FIT_REPLAY'
    assert all(c[-1] != '767288' for c in cancel_calls(scheduler))


def test_unrecognized_queue_line_never_publishes_recovery_proof(project):
    scheduler=Scheduler(project); scheduler.queue='unrecognized queue result\n'
    with pytest.raises(recovery.RecoveryError,match='queue_record_unrecognized'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not (project/'.paper3-hpc/recoveries/test-767434.json').exists()


def test_already_cancelled_never_started_dependents_need_no_cancel_replay(project):
    scheduler=Scheduler(project)
    scheduler.states['767436']=scheduler.states['767438']='CANCELLED'
    value=recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert value['status']=='RECOVERY_VERIFIED_NO_FIT_REPLAY' and not cancel_calls(scheduler)


def test_pending_to_running_race_is_not_killed_and_cannot_authorize_replacement(project):
    scheduler=Scheduler(project)
    def race(command,**kwargs):
        if command[0]=='scancel' and command[-1]=='767438':
            scheduler.states['767438']='RUNNING'
            scheduler.start['767438']='2026-10-21T18:00:00'
            scheduler.elapsed['767438']='1'
        return scheduler(command,**kwargs)
    with pytest.raises(recovery.RecoveryError,match='may_have_started'):
        recovery.recover(project,'767434',execute=True,runner=race)
    assert scheduler.states['767438']=='RUNNING'
    assert not (project/'.paper3-hpc/recoveries/test-767434.json').exists()


def test_tampered_final_parent_success_code_does_not_supersede_original(project):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    path=project/'.paper3-hpc/recoveries/test-767434.json'
    value=json.loads(path.read_text()); value['accounting_after']['767434']['exit_code']='0:0'; start.write(path,value)
    with pytest.raises(recovery.RecoveryError,match='test_not_failed'):
        recovery.superseded_attempts(project)


def test_missing_original_cancel_intent_invalidates_final_exclusion(project):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    (project/'.paper3-hpc/recoveries/test-767434.intent.json').unlink()
    with pytest.raises(recovery.RecoveryError,match='cancel_intent_changed'):
        recovery.superseded_attempts(project)


def test_changed_original_cancel_intent_invalidates_final_exclusion(project):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    path=project/'.paper3-hpc/recoveries/test-767434.intent.json'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(recovery.RecoveryError,match='cancel_intent_changed'):
        recovery.superseded_attempts(project)


def launch_args(root):
    return SimpleNamespace(root=root, config=ROOT/'hpc/resources.json', python=Path('/usr/bin/python'),
        cpus=None, mem_gib=None, gpus=None, time=None, execute=True, dry_run=False,
        staged_data=True, network_host=False)


def launch_runner(queue=''):
    def run(command,**kwargs):
        assert kwargs['stdin']==subprocess.DEVNULL and kwargs['timeout']==30
        if command[0]=='squeue':
            return SimpleNamespace(returncode=0,stdout=queue)
        assert command==['git','rev-parse','--verify','HEAD']
        return SimpleNamespace(returncode=0,stdout='d'*40+'\n')
    return run


def test_verified_recovery_allows_exactly_one_new_chain_without_changing_originals(project,monkeypatch):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    claim=project/'runs/retained-original-claim.json';claim.parent.mkdir();claim.write_text('{"charged": 19}')
    marker=project/'cache/retained-original.npy';marker.parent.mkdir();marker.write_bytes(b'preserved numeric cache')
    before=originals(project); calls=[]
    monkeypatch.setattr(start,'staged_inputs',lambda root:{'status':'DEFERRED_TO_SLURM_PREPARE_NOT_YET_VERIFIED'})
    def submitter(request):
        path=project/'.paper3-hpc/start'/('attempt-'+request['start_attempt_id']+'.json')
        intent=start.read(path)['submission_intents'][-1]
        assert intent['state']=='SUBMISSION_INTENT_OUTCOME_UNKNOWN' and intent['request']==request
        job=str(800001+len(calls));calls.append(request)
        value={**request,'status':'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE','job_id':job,
               'scientific_completion_claimed':False}
        start.write(project/'.paper3-hpc/submissions'/(job+'-bbbbbbbb.json'),value)
        return value
    value=start.start(launch_args(project),runner=launch_runner(),
                      verifier=lambda root:{'manifest_sha256':'e'*64},submitter=submitter)
    assert value['status']=='SLURM_CHAIN_SUBMITTED_NOT_TRAINING_COMPLETE'
    assert value['job_ids']=={'test':'800001','prepare':'800002','train':'800003'}
    assert [r['stage'] for r in calls]==['test','prepare','train']
    assert '--dependency=afterok:800001' in calls[1]['command']
    assert '--dependency=afterok:800002' in calls[2]['command']
    after=originals(project)
    assert all(after[key]==data for key,data in before.items())
    assert claim.read_text()=='{"charged": 19}' and marker.read_bytes()==b'preserved numeric cache'
    again=start.start(launch_args(project),runner=launch_runner(),
                      verifier=lambda root:pytest.fail('must not reverify or submit on retry'),
                      submitter=lambda request:pytest.fail('must not submit a duplicate chain'))
    assert again['status']=='EXISTING_CHAIN_NO_RESUBMISSION' and len(calls)==3


def test_requeued_retired_id_still_binds_all_original_receipts_and_blocks_duplicate(project,monkeypatch):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    def forbidden(*a,**kw):pytest.fail('requeued old job must block any new fit or submission')
    monkeypatch.setattr(start,'staged_inputs',forbidden)
    value=start.start(launch_args(project),runner=launch_runner('767438|renamed|PENDING|/different/workdir\n'),
                      verifier=forbidden,submitter=forbidden)
    assert value['status']=='EXISTING_ACTIVE_JOBS_NO_DUPLICATE_FETCH_OR_SUBMISSION'
    assert value['job_ids']==['767438']
    assert value['active_jobs'][0]['binding']=='root_submission_receipt'
    assert value['new_jobs_submitted'] is False


def test_corrupt_retirement_proof_stops_start_before_any_new_submission(project,monkeypatch):
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    intent=project/'.paper3-hpc/recoveries/test-767434.intent.json'
    intent.write_bytes(intent.read_bytes()+b'\n')
    def forbidden(*a,**kw):pytest.fail('corrupt proof must block all new submissions')
    monkeypatch.setattr(start,'staged_inputs',forbidden)
    with pytest.raises(recovery.RecoveryError,match='cancel_intent_changed'):
        start.start(launch_args(project),runner=launch_runner(),verifier=forbidden,submitter=forbidden)
    assert len(list((project/'.paper3-hpc/start').glob('attempt-*.json')))==1


@pytest.mark.parametrize('suffix',['(unfulfilled)','(fulfilled)','(unsatisfied)','(satisfied)',''])
def test_slurm20_dependency_suffix_forms_validate_live_and_immutable_proof(project,suffix):
    scheduler=Scheduler(project)
    def observed(command,**kwargs):
        row=scheduler(command,**kwargs)
        if command[0]=='scontrol':
            row.stdout=row.stdout.replace('(unsatisfied)',suffix)
        return row
    value=recovery.recover(project,'767434',execute=True,runner=observed)
    assert value['status']=='RECOVERY_VERIFIED_NO_FIT_REPLAY'
    assert value['binding']['attempt'] in recovery.superseded_attempts(project)


@pytest.mark.parametrize('dependency',['afterok:767434(unfulfilled):999999','afterok:767434(unknown)',
                                       'afterany:767434(unfulfilled)','afterok:7674340(unfulfilled)'])
def test_slurm20_dependency_rendering_never_allows_other_parent_or_semantics(project,dependency):
    scheduler=Scheduler(project);scheduler.control_changes={'Dependency':dependency}
    with pytest.raises(recovery.RecoveryError,match='dependency_mismatch'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert not cancel_calls(scheduler)


def test_slurm20_cancelled_start_normalization_requires_original_pending_witness(project):
    scheduler=Scheduler(project);scheduler.normalize_cancelled_start=True
    value=recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert value['status']=='RECOVERY_VERIFIED_NO_FIT_REPLAY'
    for job in ('767436','767438'):
        assert value['accounting_after'][job]['start']==value['accounting_after'][job]['end']
        assert value['accounting_after'][job]['elapsed']=='0'
    assert value['binding']['attempt'] in recovery.superseded_attempts(project)


def test_normalized_cancel_timeout_replay_uses_unchanged_legacy_pending_intent(project):
    scheduler=Scheduler(project);scheduler.normalize_cancelled_start=True;scheduler.cancel_timeout_once=True
    with pytest.raises(recovery.RecoveryError,match='observation_unavailable'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    intent=project/'.paper3-hpc/recoveries/test-767434.intent.json'
    value=json.loads(intent.read_text())
    for row in value['accounting_before'].values():row.pop('end',None)
    start.write(intent,value);before=intent.read_bytes()
    result=recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert result['status']=='RECOVERY_VERIFIED_NO_FIT_REPLAY'
    assert intent.read_bytes()==before
    assert [c[-1] for c in cancel_calls(scheduler)]==['767438','767436']
    assert recovery.superseded_attempts(project)


def test_first_observed_cancelled_equal_timestamps_are_not_own_never_started_proof(project):
    scheduler=Scheduler(project)
    scheduler.states['767438']='CANCELLED'
    scheduler.start['767438']=scheduler.end['767438']='2026-10-21T18:00:00'
    with pytest.raises(recovery.RecoveryError,match='may_have_started') as caught:
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert caught.value.diagnostic['cancel_intent_available_and_validated'] is False
    assert caught.value.diagnostic['children'][1]['actual']['end']=='2026-10-21T18:00:00'
    assert not cancel_calls(scheduler)


@pytest.mark.parametrize('state,start_time,end_time,elapsed',[
    ('PENDING','2026-10-21T18:00:00','2026-10-21T18:00:00','0'),
    ('CANCELLED','2026-10-21T17:59:59','2026-10-21T18:00:00','0'),
    ('CANCELLED','2026-10-21T18:00:00','2026-10-21T18:00:00','0.0'),
    ('CANCELLED','2026-99-21T18:00:00','2026-99-21T18:00:00','0'),
])
def test_pending_real_started_or_malformed_time_cannot_use_cancel_witness(project,state,start_time,end_time,elapsed):
    scheduler=Scheduler(project);scheduler.cancel_timeout_once=True
    with pytest.raises(recovery.RecoveryError):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    count=len(cancel_calls(scheduler))
    scheduler.states['767438']=state;scheduler.start['767438']=start_time
    scheduler.end['767438']=end_time;scheduler.elapsed['767438']=elapsed
    with pytest.raises(recovery.RecoveryError,match='may_have_started'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert len(cancel_calls(scheduler))==count


def test_malformed_existing_intent_is_blocked_before_any_additional_scheduler_mutation(project):
    scheduler=Scheduler(project);scheduler.cancel_timeout_once=True
    with pytest.raises(recovery.RecoveryError):recovery.recover(project,'767434',execute=True,runner=scheduler)
    path=project/'.paper3-hpc/recoveries/test-767434.intent.json'
    value=json.loads(path.read_text());value['live_pending_before']=['SECRET_MUST_NOT_LEAK'];start.write(path,value)
    count=len(cancel_calls(scheduler))
    with pytest.raises(recovery.RecoveryError,match='initial_live_pending_proof_invalid'):
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert len(cancel_calls(scheduler))==count
    diag=recovery.diagnose(project,'767434',runner=scheduler)
    assert diag['cancel_intent']=={'exists':True,'validated':False,'error_code':'recovery_initial_live_pending_proof_invalid'}
    assert 'SECRET_MUST_NOT_LEAK' not in json.dumps(diag)


def test_cancel_intent_change_during_observation_prevents_final_publication(project):
    scheduler=Scheduler(project)
    def changed(command,**kwargs):
        value=scheduler(command,**kwargs)
        if command[0]=='scancel':
            path=project/'.paper3-hpc/recoveries/test-767434.intent.json'
            path.write_bytes(path.read_bytes()+b'\n')
        return value
    with pytest.raises(recovery.RecoveryError,match='cancel_intent_changed'):
        recovery.recover(project,'767434',execute=True,runner=changed)
    assert not (project/'.paper3-hpc/recoveries/test-767434.json').exists()


def test_old_nine_field_unknown_start_recovery_proof_remains_valid(project):
    import hashlib
    scheduler=Scheduler(project)
    recovery.recover(project,'767434',execute=True,runner=scheduler)
    intent=project/'.paper3-hpc/recoveries/test-767434.intent.json'
    value=json.loads(intent.read_text())
    for row in value['accounting_before'].values():row.pop('end',None)
    start.write(intent,value)
    final=project/'.paper3-hpc/recoveries/test-767434.json';value=json.loads(final.read_text())
    for group in ('accounting_before','accounting_after'):
        for row in value[group].values():row.pop('end',None)
    value['intent_sha256']=hashlib.sha256(intent.read_bytes()).hexdigest();start.write(final,value)
    assert recovery.superseded_attempts(project)


def test_read_only_diagnose_includes_actual_failed_checks_without_creating_lock_or_files(project):
    scheduler=Scheduler(project);scheduler.start['767438']='2026-10-21T18:00:00'
    before={str(p.relative_to(project)):p.read_bytes() for p in project.rglob('*') if p.is_file()}
    diag=recovery.diagnose(project,'767434',runner=scheduler)
    assert diag['status']=='READ_ONLY_RECOVERY_DIAGNOSTIC'
    assert diag['diagnostic']['children'][1]['actual']['start']=='2026-10-21T18:00:00'
    assert diag['diagnostic']['children'][1]['checks']['start_is_unknown'] is False
    assert diag['cancel_intent']=={'exists':False,'validated':False,'error_code':None}
    assert len(diag['diagnostic']['accounting'])==3 and len(diag['live_controls'])==2
    assert not cancel_calls(scheduler)
    assert {str(p.relative_to(project)):p.read_bytes() for p in project.rglob('*') if p.is_file()}==before


def test_malformed_secret_scheduler_fields_are_redacted_in_failure_and_readonly_diagnosis(project):
    scheduler=Scheduler(project)
    scheduler.states['767438']=scheduler.start['767438']=scheduler.elapsed['767438']='SECRET_DO_NOT_COPY'
    scheduler.end['767438']='SECRET_DO_NOT_COPY'
    scheduler.control_changes={'StartTime':'SECRET_DO_NOT_COPY','Reason':'SECRET_DO_NOT_COPY',
        'Restarts':'SECRET_DO_NOT_COPY','RunTime':'SECRET_DO_NOT_COPY','Dependency':'SECRET_DO_NOT_COPY'}
    with pytest.raises(recovery.RecoveryError) as caught:
        recovery.recover(project,'767434',execute=True,runner=scheduler)
    assert 'SECRET_DO_NOT_COPY' not in json.dumps(caught.value.diagnostic)
    diag=recovery.diagnose(project,'767434',runner=scheduler)
    assert 'SECRET_DO_NOT_COPY' not in json.dumps(diag)
    assert diag['diagnostic']['children'][1]['actual']['state']=='UNRECOGNIZED_VALUE'
    assert diag['live_controls']['train']['runtime']=='UNRECOGNIZED_VALUE'
    assert not cancel_calls(scheduler)


def test_diagnose_owner_mismatch_returns_only_safe_error_per_child(project):
    scheduler=Scheduler(project);scheduler.control_changes={'UserId':'SECRET_DO_NOT_COPY(99999)'}
    diag=recovery.diagnose(project,'767434',runner=scheduler)
    assert all(not value['identity_validated'] for value in diag['live_controls'].values())
    assert 'SECRET_DO_NOT_COPY' not in json.dumps(diag)
    assert not cancel_calls(scheduler)


def test_cli_diagnose_even_with_execute_never_cancels_or_writes(project,monkeypatch,capsys):
    scheduler=Scheduler(project);scheduler.start['767438']='2026-10-21T18:00:00'
    actual=recovery.diagnose
    monkeypatch.setattr(recovery,'diagnose',lambda root,job:actual(root,job,runner=scheduler))
    assert recovery.main(['--root',str(project),'--failed-test-job','767434','--execute','--diagnose'])==0
    value=json.loads(capsys.readouterr().out)
    assert value['status']=='READ_ONLY_RECOVERY_DIAGNOSTIC' and not cancel_calls(scheduler)
    assert not (project/'.paper3-hpc/recoveries').exists()
    assert not (project/'.paper3-deployment').exists()


def _historical_cli_double(project, tmp_path, *, invalid_step=False, secret_step_metadata=False):
    """Run the real script; only scheduler executables are software doubles."""
    import importlib.util
    from hpc import never_allocated
    spec = importlib.util.spec_from_file_location('cli_history_software_fixture',
        Path(__file__).with_name('test_hpc_never_allocated.py'))
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    scheduler = fixture.Accounting(project)
    if invalid_step:
        scheduler.records[3]['db_index'] = '999999'
    if secret_step_metadata:
        for name in ('user','account','workdir','name','alloc_tres','node_list','submit'):
            scheduler.records[3][name] = 'SECRET_UNCHECKED_STEP_METADATA'
    basic = scheduler(['sacct', '--allocations'], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False, timeout=30).stdout
    full = '\n'.join('|'.join(row[key] for key in never_allocated.FIELDS)
                     for row in scheduler.records) + '\n'
    support = tmp_path / 'cli-support'; support.mkdir()
    (support / 'basic').write_text(basic); (support / 'full').write_text(full)
    calls = support / 'calls'
    binary = support / 'bin'; binary.mkdir()
    program = ('#!' + sys.executable + '\nimport json,pathlib,sys\n'
        'base=pathlib.Path(' + repr(str(support)) + ')\n'
        'kind=pathlib.Path(sys.argv[0]).name\n'
        'with (base/"calls").open("a") as out: out.write(json.dumps([kind,*sys.argv[1:]])+"\\n")\n'
        'if kind=="sacct": print((base/("full" if "--duplicates" in sys.argv else "basic")).read_text(),end="")\n'
        'elif kind=="squeue": print("")\n'
        'else:\n print("SECRET_SCHEDULER_STDERR",file=sys.stderr)\n raise SystemExit(1)\n')
    for name in ('sacct','squeue','scontrol','scancel','sbatch','srun'):
        path = binary / name; path.write_text(program); path.chmod(0o700)
    environment = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ.get('PATH',''))
    def run(*flags):
        return subprocess.run([sys.executable,'-B',str(ROOT/'hpc/recover_failed_chain.py'),
            '--root',str(project),'--failed-test-job','767434',*flags],
            env=environment,capture_output=True,text=True,timeout=20)
    return run, calls


def test_actual_script_preserves_adapter_exception_code_instead_of_class_name(project,tmp_path):
    run, calls = _historical_cli_double(project,tmp_path,invalid_step=True)
    before = originals(project)
    result = run('--execute')
    assert result.returncode == 2
    value = json.loads(result.stdout)
    assert value['reason'] == 'never_allocated_parent_step_identity_or_time_invalid'
    assert value['diagnostic']['job_id'] == '767434.batch'
    assert 'SECRET_SCHEDULER_STDERR' not in result.stdout + result.stderr
    assert all(json.loads(line)[0]=='sacct' for line in calls.read_text().splitlines())
    assert originals(project) == before and not (project/'.paper3-hpc/recoveries').exists()


@pytest.mark.parametrize('invalid_step,secret_step_metadata',[(False,False),(True,False),(False,True)])
def test_actual_script_diagnose_includes_complete_history_or_precise_safe_error_without_writes(
        project,tmp_path,invalid_step,secret_step_metadata):
    run, calls = _historical_cli_double(project,tmp_path,invalid_step=invalid_step,
                                       secret_step_metadata=secret_step_metadata)
    before = {str(p.relative_to(project)):p.read_bytes() for p in project.rglob('*') if p.is_file()}
    result = run('--diagnose')
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout); history = value['allocation_history']
    assert value['status']=='READ_ONLY_RECOVERY_DIAGNOSTIC'
    if invalid_step:
        assert history['status']=='HISTORY_NOT_VERIFIED'
        assert history['error_code']=='never_allocated_parent_step_identity_or_time_invalid'
    else:
        assert history['status']=='ACCOUNTING_SNAPSHOT_VERIFIED_NOT_A_RECOVERY_PROOF'
        assert len(history['facts']['rows'])==5
        assert history['single_observation_does_not_authorize_recovery'] is True
    assert 'SECRET_SCHEDULER_STDERR' not in result.stdout + result.stderr
    assert 'SECRET_UNCHECKED_STEP_METADATA' not in result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert {row[0] for row in commands} <= {'sacct','scontrol'}
    assert sum(row[0]=='sacct' and '--duplicates' in row for row in commands)==1
    assert {str(p.relative_to(project)):p.read_bytes() for p in project.rglob('*') if p.is_file()}==before
