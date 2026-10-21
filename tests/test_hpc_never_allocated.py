"""No-allocation recovery uses scheduler doubles; never real Slurm or fitting.

The timestamp/TRES/NodeList/Reason observations match the user's real school
report. AllocNodes, DBIndex and Submit below are explicit software-fixture
assumptions; the deployed module must query and verify those actual fields.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hpc import never_allocated as proof, recover_failed_chain as recovery, start

spec=importlib.util.spec_from_file_location('paper3_original_recovery_software_fixtures',
    Path(__file__).with_name('test_hpc_failed_chain_recovery.py'))
software=importlib.util.module_from_spec(spec);spec.loader.exec_module(software)
project=software.project
USER=software.USER
IDS=software.IDS


class Accounting(software.Scheduler):
    def __init__(self,root):
        super().__init__(root)
        self.start['767434']='2026-10-21T17:30:43';self.end['767434']='2026-10-21T17:50:29'
        self.records=[]
        for index,stage in enumerate(('test','prepare','train')):
            job=IDS[stage]
            if stage!='test':
                self.states[job]='CANCELLED';self.start[job]=self.end[job]='2026-10-21T17:50:48'
            self.records.append(dict(zip(proof.FIELDS,[job,self.states[job],
                '2:0' if stage=='test' else '0:0',self.start[job],self.end[job],self.elapsed[job],
                'zhuziyuan',USER,'paper3-'+stage,str(root),'1' if stage=='test' else '0',
                'billing=16,cpu=16,mem=64G,node=1' if stage=='test' else '',
                'comput1' if stage=='test' else 'None assigned',
                'None' if stage=='test' else 'Dependency',str(100+index),'2026-10-21T17:30:00'])))
        for name,state,code in [('batch','FAILED','2:0'),('extern','COMPLETED','0:0')]:
            self.records.append({**self.records[0],'job_id':'767434.'+name,'state':state,'exit_code':code,
                'name':name,'account':'','user':'','workdir':'','alloc_tres':'cpu=16,mem=64G,node=1','reason':''})
        self.full_queries=0;self.change_second=None

    def __call__(self,command,**kwargs):
        if command[0]=='sacct' and '--duplicates' in command:
            assert '--allocations' not in command and '-X' not in command
            assert '--jobs=767434,767436,767438' in command
            assert '--format='+proof.FORMAT in command
            self.calls.append(command);self.full_queries+=1
            rows=[dict(r) for r in self.records]
            if self.full_queries>1 and self.change_second:self.change_second(rows)
            return SimpleNamespace(returncode=0,stdout='\n'.join('|'.join(r[k] for k in proof.FIELDS) for r in rows)+'\n')
        return super().__call__(command,**kwargs)


def binding(root):return recovery._bound(root,'767434')


def test_explicit_two_snapshot_proof_preserves_originals_and_never_creates_cancel_intent(project):
    scheduler=Accounting(project);original=software.originals(project)
    value=proof.recover(project,binding(project),execute=True,runner=scheduler,sleeper=lambda n:None)
    assert value['status']=='ACCOUNTING_CANCELLED_WITHOUT_ALLOCATION_VERIFIED'
    assert scheduler.full_queries==2 and not software.cancel_calls(scheduler)
    assert not any(c[0] in {'scancel','sbatch','scontrol','srun','ssh'} for c in scheduler.calls)
    assert not (project/'.paper3-hpc/recoveries/test-767434.intent.json').exists()
    assert all(software.originals(project)[k]==v for k,v in original.items())
    path=project/value['receipt'];assert proof.validate_proof(project,path,start.read(path))['job_ids']==IDS
    assert recovery.superseded_attempts(project)[value['binding']['attempt']]['job_ids']==IDS


def test_plan_is_readonly_without_any_file_or_lock_creation(project):
    scheduler=Accounting(project)
    before={str(p.relative_to(project)):p.read_bytes() for p in project.rglob('*') if p.is_file()}
    value=proof.recover(project,binding(project),runner=scheduler)
    assert value['status']=='PLAN_CANCELLED_WITHOUT_ALLOCATION_NO_WRITES'
    assert scheduler.full_queries==1 and not software.cancel_calls(scheduler)
    assert {str(p.relative_to(project)):p.read_bytes() for p in project.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('field,value',[('alloc_nodes','1'),('alloc_nodes',''),('alloc_tres','cpu=1,node=1'),
    ('alloc_tres','Unknown'),('node_list','gpu3'),('elapsed','1'),('elapsed','0.0'),('state','RUNNING'),
    ('exit_code','1:0'),('start','2026-10-21T17:50:47'),('submit','2026-10-21T18:50:00')])
def test_any_child_allocation_started_or_unknown_field_rejects_without_publish(project,field,value):
    scheduler=Accounting(project);scheduler.records[2][field]=value
    with pytest.raises(recovery.RecoveryError):
        proof.recover(project,binding(project),execute=True,runner=scheduler,sleeper=lambda n:None)
    assert not (project/'.paper3-hpc/recoveries/test-767434.json').exists()
    assert not software.cancel_calls(scheduler)


@pytest.mark.parametrize('field,value',[('alloc_nodes','0'),('alloc_tres',''),('alloc_tres','cpu=16'),
    ('node_list','None assigned'),('elapsed','0'),('exit_code','0:0'),('db_index',''),('state','COMPLETED')])
def test_positive_parent_accounting_control_is_required(project,field,value):
    scheduler=Accounting(project);scheduler.records[0][field]=value
    with pytest.raises(recovery.RecoveryError):proof.collect(binding(project),scheduler)


def test_parent_step_positive_control_is_required(project):
    scheduler=Accounting(project);scheduler.records=scheduler.records[:3]
    with pytest.raises(recovery.RecoveryError,match='step_positive_control'):proof.collect(binding(project),scheduler)


def test_zero_elapsed_parent_steps_are_not_positive_tracking_control(project):
    scheduler=Accounting(project)
    for row in scheduler.records[3:]:row['elapsed']='0'
    with pytest.raises(recovery.RecoveryError,match='step_positive_control'):proof.collect(binding(project),scheduler)


def test_any_descendant_step_prevents_retirement_even_when_parent_row_looks_unallocated(project):
    scheduler=Accounting(project)
    scheduler.records.append({**scheduler.records[3],'job_id':'767438.batch'})
    with pytest.raises(recovery.RecoveryError,match='descendant_step_found'):proof.collect(binding(project),scheduler)


def test_duplicates_or_requeued_allocation_records_are_not_silently_deduplicated(project):
    scheduler=Accounting(project);scheduler.records.append(dict(scheduler.records[2]))
    with pytest.raises(recovery.RecoveryError,match='duplicate_job_history'):proof.collect(binding(project),scheduler)


def test_changed_history_between_fresh_observations_is_not_published(project):
    scheduler=Accounting(project);scheduler.change_second=lambda rows:rows[2].update(reason='Priority')
    with pytest.raises(recovery.RecoveryError,match='changed_between_observations'):
        proof.recover(project,binding(project),execute=True,runner=scheduler,sleeper=lambda n:None)
    assert not (project/'.paper3-hpc/recoveries/test-767434.json').exists()


def test_requeued_old_job_in_live_queue_blocks_no_allocation_retirement(project):
    scheduler=Accounting(project);scheduler.queue='767438|PENDING\n'
    with pytest.raises(recovery.RecoveryError,match='original_job_is_queued'):
        proof.recover(project,binding(project),execute=True,runner=scheduler,sleeper=lambda n:None)
    assert not software.cancel_calls(scheduler)


def test_unrelated_notebook_is_preserved_and_does_not_block_terminal_proof(project):
    scheduler=Accounting(project);scheduler.queue='767288|RUNNING\n'
    value=proof.recover(project,binding(project),execute=True,runner=scheduler,sleeper=lambda n:None)
    assert value['status']=='ACCOUNTING_CANCELLED_WITHOUT_ALLOCATION_VERIFIED'
    assert not software.cancel_calls(scheduler)


def test_missing_accounting_capability_is_not_treated_as_zero_allocation(project):
    scheduler=Accounting(project)
    def incomplete(command,**kwargs):
        response=scheduler(command,**kwargs)
        response.stdout='\n'.join('|'.join(line.split('|')[:-1]) for line in response.stdout.splitlines())
        return response
    with pytest.raises(recovery.RecoveryError,match='capabilities_required'):proof.collect(binding(project),incomplete)


def test_unrecognized_secret_payload_never_leaks_in_error_diagnostic(project):
    scheduler=Accounting(project);scheduler.records[2]['alloc_tres']='SECRET_DO_NOT_COPY'
    with pytest.raises(recovery.RecoveryError) as caught:proof.collect(binding(project),scheduler)
    assert 'SECRET_DO_NOT_COPY' not in str(caught.value)+json.dumps(caught.value.diagnostic)


def test_forged_or_changed_final_snapshot_does_not_supersede_history(project):
    scheduler=Accounting(project)
    value=proof.recover(project,binding(project),execute=True,runner=scheduler,sleeper=lambda n:None)
    next(r for r in value['observations'][1]['rows'] if r['job_id']=='767438')['alloc_nodes']='1'
    with pytest.raises(recovery.RecoveryError):proof.validate_proof(project,project/value['receipt'],value)


def test_original_receipt_change_invalidates_new_proof(project):
    scheduler=Accounting(project)
    value=proof.recover(project,binding(project),execute=True,runner=scheduler,sleeper=lambda n:None)
    old=next((project/'.paper3-hpc/start').glob('attempt-*.json'));old.write_bytes(old.read_bytes()+b'\n')
    with pytest.raises(recovery.RecoveryError,match='binding_changed'):
        proof.validate_proof(project,project/value['receipt'],value)


def test_no_allocation_proof_does_not_allow_compute_host_recovery(project,monkeypatch):
    scheduler=Accounting(project);monkeypatch.setenv('SLURM_JOB_ID','777777')
    with pytest.raises(recovery.RecoveryError,match='submission_host'):
        proof.recover(project,binding(project),execute=True,runner=scheduler)
    assert not scheduler.calls


def test_actual_recovery_entry_falls_back_without_forging_intent_and_start_reads_new_proof(project,monkeypatch):
    scheduler=Accounting(project);before=software.originals(project)
    value=recovery.recover(project,'767434',execute=True,runner=scheduler,sleeper=lambda n:None)
    assert value['schema_version']==proof.SCHEMA
    assert not software.cancel_calls(scheduler)
    assert not (project/'.paper3-hpc/recoveries/test-767434.intent.json').exists()
    monkeypatch.setattr(start,'staged_inputs',lambda root:{'status':'DEFERRED_TO_SLURM_PREPARE_NOT_YET_VERIFIED'})
    calls=[]
    def submitter(request):
        job=str(810001+len(calls));calls.append(request)
        receipt={**request,'status':'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE','job_id':job}
        start.write(project/'.paper3-hpc/submissions'/(job+'-bbbbbbbb.json'),receipt)
        return receipt
    result=start.start(software.launch_args(project),runner=software.launch_runner(),
                       verifier=lambda root:{'manifest_sha256':'e'*64},submitter=submitter)
    assert result['job_ids']=={'test':'810001','prepare':'810002','train':'810003'}
    assert len(calls)==3 and all(software.originals(project)[k]==v for k,v in before.items())


@pytest.mark.parametrize('field,value',[('db_index','999999'),('start','2026-10-21T17:30:42'),
                                       ('end','2026-10-21T17:50:30')])
def test_parent_step_database_and_time_must_match_failed_allocation(project,field,value):
    scheduler=Accounting(project);scheduler.records[3][field]=value
    with pytest.raises(recovery.RecoveryError,match='parent_step_identity_or_time_invalid'):
        proof.collect(binding(project),scheduler)


@pytest.mark.parametrize('bad',[[],{'project':'wrong'},{'project':'placeholder','jobs':[]}])
def test_malformed_binding_is_a_safe_recovery_error(project,bad):
    if isinstance(bad,dict) and bad.get('project')=='placeholder':bad['project']=str(project)
    with pytest.raises(recovery.RecoveryError):proof.recover(project,bad,execute=True)
