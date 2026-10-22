"""Explicit prefit retry tests use metadata and scheduler doubles only."""
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hpc import resume_prefit_train as resume, start, submit

ROOT = Path(__file__).resolve().parents[1]
USER = pwd.getpwuid(os.getuid()).pw_name
IDS = {'test': '767526', 'prepare': '767528', 'train': '767530'}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    root = tmp_path / 'school project'; (root / 'hpc').mkdir(parents=True)
    (root / 'hpc/resources.json').write_bytes((ROOT / 'hpc/resources.json').read_bytes())
    config = submit.load_config(root / 'hpc/resources.json')
    source = {'source_commit': 'a'*40, 'code_manifest_sha256': 'b'*64}
    rows, intents = [], []
    for index, stage in enumerate(start.STAGES):
        options = SimpleNamespace(stage=stage, root=root, python=sys.executable, account=None,
            partition=None, cpus=None, mem_gib=None, gpus=None, time=None,
            dependency=None if index==0 else 'afterok:' + rows[-1]['job_id'])
        request = {**submit.build_request(options, config), 'start_attempt_id':'c'*32,
                   'start_source_identity':source}
        row = {**request, 'job_id':IDS[stage], 'status':'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE'}
        rows.append(row); intents.append({'stage':stage,'state':'SUBMISSION_RECEIPT_VERIFIED','request':request})
        start.write(root / '.paper3-hpc/submissions' / (IDS[stage]+'-aaaaaaaa.json'), row)
        data = {'status':'ready', 'frozen_data_files':21262, 'locally_SHA_verified':21262,
                'missing_files':[], 'missing_cache_files':0, 'closure_sha256':'f'*64, 'index_sha256':'e'*64}
        result = ({'status':'ALL_SOFTWARE_TESTS_PASSED','returncode':0,
                   'software_identity':{'manifest_sha256':source['code_manifest_sha256']}}
                  if stage=='test' else data)
        start.write(root / '.paper3-hpc/jobs' / IDS[stage] / ('stage-'+stage+'.json'), {
            'schema_version':'paper3-hpc-stage-receipt-v1','stage':stage,'job_id':IDS[stage],
            'returncode':2 if stage=='train' else 0,'result':result,
            'scientific_completion_claimed':False,'external_query_accessed':False})
    start.write(root / '.paper3-hpc/start' / ('attempt-'+'c'*32+'.json'), {
        'schema_version':start.SCHEMA,'attempt_id':'c'*32,'project':str(root),'source_identity':source,
        'status':'SLURM_CHAIN_SUBMITTED_NOT_TRAINING_COMPLETE','submitted_jobs':rows,'submission_intents':intents})
    start.write(root / '.paper3-hpc/jobs/767530/campaign.log', {
        'status':'BLOCKED_PRESERVE_EXISTING_JOBS','error_type':'CampaignError','reason':resume.PILOT_REASON,
        'completion_claimed':False,'external_query_accessed':False})
    start.write(root / 'runs/original_wall.json', {'charged':388,'must_not_reset':True})
    (root / 'cache').mkdir(); (root / 'cache/original.bytes').write_bytes(b'authentic fixture marker')
    return root


def options(root, **kw):
    result = dict(root=root, python=Path(sys.executable), failed_train_job='767530', execute=True,
                  dry_run=False, cpus=None, mem_gib=None, gpus=None, time=None)
    result.update(kw); return SimpleNamespace(**result)


def inspector(root):
    return {'status':'SOFTWARE_DOUBLE_ZERO_UPDATE_PREFIT_INSPECTION',
            'sources':[resume.descriptor(root,root/'runs/original_wall.json')]}


def verifier(root):
    return {'manifest_sha256':'d'*64, 'numerical_kernel_overrides':0}


class Scheduler:
    def __init__(self, root):
        self.root=root;self.calls=[];self.queue='';self.states={};self.owner=USER
    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[0]=='git':
            assert cmd==['git','rev-parse','--verify','HEAD']
            return SimpleNamespace(returncode=0,stdout='e'*40+'\n')
        if cmd[0]=='sacct':
            assert '--allocations' in cmd and '--duplicates' in cmd
            assert '--jobs=767526,767528,767530' in cmd and '--format='+resume.FORMAT in cmd
            rows=[]
            for index, (stage, job) in enumerate(IDS.items()):
                values=[job,self.states.get(stage,'FAILED' if stage=='train' else 'COMPLETED'),
                    '2:0' if stage=='train' else '0:0','2026-10-21T21:57:21',
                    '2026-10-21T23:52:32','100','zhuziyuan',self.owner,'paper3-'+stage,
                    str(self.root),str(100+index),'2026-10-21T21:57:20']
                rows.append('|'.join(values))
            return SimpleNamespace(returncode=0,stdout='\n'.join(rows)+'\n')
        if cmd[0]=='squeue':
            return SimpleNamespace(returncode=0,stdout=self.queue)
        pytest.fail('Unexpected real scheduler command: '+cmd[0])


class API:
    def __init__(self, root):self.root=root;self.calls=[];self.crash_before=False;self.crash_after=False
    def __call__(self, request):
        self.calls.append(request)
        if self.crash_before:
            self.crash_before=False;raise subprocess.TimeoutExpired('SOFTWARE_DOUBLE',1)
        job=str(810000+len(self.calls))
        receipt={**request,'status':'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE','job_id':job}
        start.write(self.root / '.paper3-hpc/submissions' / (job+'-bbbbbbbb.json'),receipt)
        if self.crash_after:
            self.crash_after=False;raise subprocess.TimeoutExpired('SOFTWARE_DOUBLE',1)
        return receipt


def invoke(root, scheduler, api, **kw):
    return resume.resume(options(root, **kw),runner=scheduler,verifier=verifier,inspector=inspector,submitter=api)


def original_files(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()
            and not p.name.endswith('.lock') and 'prefit-train-resumes' not in p.parts
            and not ('submissions' in p.parts and p.name.startswith('810'))}


def test_explicit_resume_queues_only_new_CPU_test_then_same_train_and_preserves_originals(project):
    scheduler=Scheduler(project);api=API(project);before=original_files(project)
    result=invoke(project,scheduler,api)
    assert result['job_ids']=={'test':'810001','train':'810002'}
    assert result['status']=='SLURM_PREFIT_RETEST_TRAIN_CHAIN_SUBMITTED_NOT_COMPLETE'
    assert result['prepare_reused_full_SHA_rechecked_by_train'] is True
    assert [r['stage'] for r in api.calls]==['test','train']
    assert '--dependency=afterok:810001' in api.calls[1]['command']
    assert '--max-wall-seconds' not in ' '.join(api.calls[1]['command'])
    assert all(r['scientific_budget_changed'] is False for r in api.calls)
    assert original_files(project)==before
    assert not any(c[0] in {'scancel','srun','ssh','sbatch'} for c in scheduler.calls)


def test_repeat_completed_submission_returns_old_IDs_even_if_new_train_has_advanced_pilot(project):
    scheduler=Scheduler(project);api=API(project);first=invoke(project,scheduler,api)
    def forbidden(*args):pytest.fail('Already submitted jobs must not trigger prefit check or new submission')
    before=len(scheduler.calls)
    result=resume.resume(options(project),runner=scheduler,verifier=forbidden,inspector=forbidden,submitter=forbidden)
    assert result['status']=='EXISTING_PREFIT_RESUME_NO_RESUBMISSION'
    assert result['job_ids']==first['job_ids'] and len(api.calls)==2 and len(scheduler.calls)==before


@pytest.mark.parametrize('execute,dry_run',[(False,False),(True,True)])
def test_plan_is_readonly_without_queries_locks_or_dependency_capture(project,execute,dry_run):
    before={str(p):p.read_bytes() for p in project.rglob('*') if p.is_file()}
    def forbidden(*args,**kwargs):pytest.fail('Plan must not inspect, submit or query')
    result=resume.resume(options(project,execute=execute,dry_run=dry_run),runner=forbidden,
                         verifier=forbidden,inspector=forbidden,submitter=forbidden)
    assert result['status']=='PLAN_PREFIT_RETEST_THEN_TRAIN_NO_ACTION'
    assert {str(p):p.read_bytes() for p in project.rglob('*') if p.is_file()}==before
    assert not list(project.rglob('*.lock'))


def test_unknown_submission_outcome_without_actual_receipt_blocks_duplicate(project):
    scheduler=Scheduler(project);api=API(project);api.crash_before=True
    with pytest.raises(subprocess.TimeoutExpired):invoke(project,scheduler,api)
    with pytest.raises(resume.ResumeError,match='unknown_submission_outcome'):invoke(project,scheduler,api)
    assert len(api.calls)==1


def test_actual_durable_receipt_recovers_ACK_crash_then_submits_only_missing_train(project):
    scheduler=Scheduler(project);api=API(project);api.crash_after=True
    with pytest.raises(subprocess.TimeoutExpired):invoke(project,scheduler,api)
    scheduler.queue='810001|paper3-test|RUNNING|'+str(project)+'\n'
    result=invoke(project,scheduler,api)
    assert result['job_ids']=={'test':'810001','train':'810002'}
    assert [r['stage'] for r in api.calls]==['test','train']


def test_other_active_project_work_blocks_before_any_submit(project):
    scheduler=Scheduler(project);api=API(project)
    scheduler.queue='999999|paper3-train|RUNNING|'+str(project)+'\n'
    with pytest.raises(resume.ResumeError,match='active_project'):invoke(project,scheduler,api)
    assert not api.calls and not (project/'.paper3-hpc/prefit-train-resumes').exists()


def test_original_running_job_or_other_owner_cannot_be_reinterpreted_as_prefit_failure(project):
    scheduler=Scheduler(project);api=API(project);scheduler.states['train']='RUNNING'
    with pytest.raises(resume.ResumeError,match='terminal_outcome'):invoke(project,scheduler,api)
    scheduler.states.clear();scheduler.owner='foreign'
    with pytest.raises(resume.ResumeError,match='identity_mismatch'):invoke(project,scheduler,api)
    assert not api.calls


@pytest.mark.parametrize('stage,field,value',[
    ('test','returncode',1),('prepare','returncode',2),('train','returncode',0),
    ('train','external_query_accessed',True)])
def test_wrong_original_terminal_receipt_cannot_authorize_resume(project,stage,field,value):
    path=project/'.paper3-hpc/jobs'/IDS[stage]/('stage-'+stage+'.json')
    record=start.read(path);record[field]=value;start.write(path,record)
    api=API(project)
    with pytest.raises(resume.ResumeError):invoke(project,Scheduler(project),api)
    assert not api.calls


def test_other_campaign_failure_reason_is_not_authorized(project):
    path=project/'.paper3-hpc/jobs/767530/campaign.log';value=start.read(path)
    value['reason']='OTHER_ERROR_OR_SECRET';start.write(path,value)
    with pytest.raises(resume.ResumeError,match='exact_failed_pilot_gate'):
        invoke(project,Scheduler(project),API(project))


def test_real_prefit_inspection_rejection_never_submits_or_creates_resume_attempt(project):
    def invalid(root):raise ValueError('not_a_zero_update_failed_pilot')
    api=API(project)
    with pytest.raises(ValueError,match='zero_update'):
        resume.resume(options(project),runner=Scheduler(project),verifier=verifier,inspector=invalid,submitter=api)
    assert not api.calls and not (project/'.paper3-hpc/prefit-train-resumes').exists()


def test_train_resource_overrides_leave_CPU_test_profile_and_scientific_parameters_unchanged(project):
    api=API(project)
    invoke(project,Scheduler(project),api,cpus=20,mem_gib=96,gpus=1,time='12:00:00')
    assert api.calls[0]['resources']['cpus']==16 and api.calls[0]['resources']['mem_gib']==64
    assert api.calls[1]['resources']['cpus']==20 and api.calls[1]['resources']['mem_gib']==96
    assert api.calls[1]['resources']['gpus']==1 and api.calls[1]['resources']['time']=='12:00:00'


def test_code_change_between_two_submissions_keeps_existing_test_and_does_not_submit_train(project):
    scheduler=Scheduler(project);api=API(project);checks=[]
    def changing(root):
        checks.append(root)
        return {'manifest_sha256':('d' if len(checks)<3 else 'f')*64}
    with pytest.raises(resume.ResumeError,match='source_changed_before_submission'):
        resume.resume(options(project),runner=scheduler,verifier=changing,inspector=inspector,submitter=api)
    assert len(api.calls)==1 and api.calls[0]['stage']=='test'


def test_partial_resume_with_failed_new_CPU_test_never_submits_train(project):
    scheduler=Scheduler(project);api=API(project);api.crash_after=True
    with pytest.raises(subprocess.TimeoutExpired):invoke(project,scheduler,api)
    start.write(project/'.paper3-hpc/jobs/810001/stage-test.json',{'returncode':1})
    with pytest.raises(resume.ResumeError,match='CPU_test_failed'):invoke(project,scheduler,api)
    assert len(api.calls)==1


def test_immediately_scheduled_test_can_acquire_controller_lock_during_submission(project):
    from hpc.run_stage import owner_lock
    api=API(project); acquired=[]
    def immediate_scheduler(request):
        # A separately opened flock conflicts even in this process if the
        # launcher retained its controller lock across the sbatch boundary.
        with owner_lock(project):
            acquired.append(request['stage'])
        return api(request)
    result=resume.resume(options(project),runner=Scheduler(project),verifier=verifier,
                         inspector=inspector,submitter=immediate_scheduler)
    assert acquired==['test','train'] and result['job_ids']['train']=='810002'


def test_completed_submission_can_be_observed_while_its_new_controller_owns_project(project):
    from hpc.run_stage import owner_lock
    scheduler=Scheduler(project);api=API(project);original=invoke(project,scheduler,api)
    with owner_lock(project):
        result=invoke(project,scheduler,api)
    assert result['status']=='EXISTING_PREFIT_RESUME_NO_RESUBMISSION'
    assert result['job_ids']==original['job_ids'] and len(api.calls)==2


def _strict_pilot_fixture(project, tmp_path, monkeypatch):
    import importlib.util
    from hpc import pilot_prefit
    spec=importlib.util.spec_from_file_location('strict_prefit_resume_software_fixture',
        Path(__file__).with_name('test_hpc_pilot_prefit.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    holder=tmp_path/'pilot metadata fixture';holder.mkdir()
    metadata=module.project.__wrapped__(holder,monkeypatch)
    for relative in pilot_prefit.FILES.values():
        value=json.loads((metadata/relative).read_text().replace(str(metadata),str(project)))
        start.write(project/relative,value)
    monkeypatch.delenv('SLURM_JOB_ID');monkeypatch.delenv('PAPER3_HPC_ROOT')
    return pilot_prefit


def test_actual_strict_prefit_inspector_integrates_without_GPU_capture_or_original_mutation(
        project,tmp_path,monkeypatch):
    _strict_pilot_fixture(project,tmp_path,monkeypatch)
    before=original_files(project);api=API(project)
    result=resume.resume(options(project),runner=Scheduler(project),verifier=verifier,submitter=api)
    attempt=start.read(project/result['attempt'])
    assert attempt['prefit_inspection']['scientific_updates_performed']==0
    assert attempt['prefit_inspection']['campaign_charged_wall_seconds']==388.038
    assert attempt['prefit_inspection']['new_pilot_claim_authorized'] is False
    assert original_files(project)==before


def test_actual_strict_inspector_rejects_one_scientific_update_without_submitting(
        project,tmp_path,monkeypatch):
    pilot_prefit=_strict_pilot_fixture(project,tmp_path,monkeypatch)
    path=project/pilot_prefit.FILES['receipt'];value=start.read(path)
    value['actual_completed_updates']=1;start.write(path,value)
    api=API(project)
    with pytest.raises(pilot_prefit.PrefitError):
        resume.resume(options(project),runner=Scheduler(project),verifier=verifier,submitter=api)
    assert not api.calls and not (project/'.paper3-hpc/prefit-train-resumes').exists()
