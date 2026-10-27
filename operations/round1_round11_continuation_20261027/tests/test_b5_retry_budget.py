"""Observed B5 dispatch-cap recovery; no EEG fit or school operation."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,'/workspace/paper3-r11-repo')
import b5_retry_budget as repair
from p3work import development_evaluation as native


def plan():
    return native.build_calibration_profile_plan(('B5',))


def progress():
    p=plan()
    return dict(identity_sha256=repair.IDENTITY, fit_attempts_dispatched=9,
        initial_fit_attempt_limit=9,fit_attempt_limit=9,fit_limit_amendments=[],active=None,
        completed={r['job_sha256']:dict(job=r['job'],attempt=a,
            candidate_state={'path':'runs/preserved-model-%d/receipt.json'%i,'sha256':str(i)*64})
            for i,(r,a) in enumerate(zip(p['jobs'][:7],repair.KNOWN_ATTEMPTS))})


def budget():
    return dict(stage='development_calibration_profile',plan_sha256=plan()['plan_sha256'],
        max_jobs=1,max_download_bytes=0,max_fit_attempts=9,max_calibration_seconds=130,
        max_wall_seconds=500000,max_gpu_seconds=500000,
        operational_amendment_note='Existing cumulative wall-only continuation',
        original_budget={'path':'runs/original-budget.json','sha256':'f'*64})


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def descriptor(root,path):
    return dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_pure_amendment_preserves_every_counter_completed_model_and_limit():
    state,old=progress(),budget();before=copy.deepcopy(state)
    updated=repair.amended_budget(state,old,plan())
    assert state==before and old==budget()
    assert updated['max_fit_attempts']==11
    assert repair.NOTE in updated['operational_amendment_note']
    for key,value in old.items():
        if key not in ('max_fit_attempts','operational_amendment_note'):
            assert updated[key]==value
    assert updated['r1_b5_retry_repair']['lost_historical_dispatches']==[1,7]
    assert state['completed']==before['completed']


def test_actual_user_observed_job_keys_attempts_and_identity_match_native_fixture():
    # The user supplied text, not an uploaded complete file. Replay only the
    # observed identities/counters/attempt fields, not unseen parameter bytes.
    state=progress()
    assert list(state['completed'])==list(repair.KNOWN_JOB_IDS[:7])
    assert state['identity_sha256']=='3a22efc255ca65bf5e9abd4f4fcc0477ad3626d0538f19ad622108ef2b3516e5'
    assert [r['attempt'] for r in state['completed'].values()]==[2,3,4,5,6,8,9]
    assert plan()['plan_sha256']==repair.KNOWN_PLAN_SHA256
    assert repair.amended_budget(state,budget(),plan())['max_fit_attempts']==11


def test_actual_native_budget_reader_and_attempt_history_accept_audited_amendment(tmp_path):
    root=tmp_path/'project';root.mkdir()
    state=progress();before=copy.deepcopy(state)
    value=repair.amended_budget(state,budget(),plan())
    path=root/'runs/new-budget.json';write(path,value)
    limits=native._read_budget(root,descriptor(root,path),stage='development_calibration_profile',
                              expected_identity=plan()['plan_sha256'])
    native._attempt_limit(state,limits,initial_limit=9)
    assert state['fit_attempt_limit']==11 and state['initial_fit_attempt_limit']==9
    assert state['fit_attempts_dispatched']==9 and state['completed']==before['completed']
    assert state['active'] is None and len(state['fit_limit_amendments'])==1
    assert state['fit_limit_amendments'][0]['previous']==9
    assert state['fit_limit_amendments'][0]['new']==11
    native._attempt_limit(state,limits,initial_limit=9)
    assert len(state['fit_limit_amendments'])==1
    # A subsequent startup sees the original base budget again. It still
    # supplies cap 11, never sends an illegal downgrade to the native kernel.
    again=repair.amended_budget(state,budget(),plan())
    native._attempt_limit(state,again,initial_limit=9)
    assert state['fit_attempts_dispatched']==9 and len(state['fit_limit_amendments'])==1


def test_two_missing_native_jobs_use_dispatches_ten_and_eleven_without_refitting_seven():
    state=progress();before=copy.deepcopy(state['completed'])
    changed=repair.amended_budget(state,budget(),plan())
    native._attempt_limit(state,changed,initial_limit=9)
    fitted=[]
    for row in plan()['jobs']:
        if row['job_sha256'] in state['completed']:
            continue
        assert state['fit_attempts_dispatched']<state['fit_attempt_limit']
        state['fit_attempts_dispatched']+=1
        fitted.append(row['job']['configuration_index'])
        state['completed'][row['job_sha256']]=dict(job=row['job'],
            attempt=state['fit_attempts_dispatched'],candidate_state={'path':'new-model'})
    assert fitted==[7,8] and state['fit_attempts_dispatched']==11
    assert all(state['completed'][k]==v for k,v in before.items())
    assert repair.amended_budget(state,budget(),plan())['max_fit_attempts']==11


@pytest.mark.parametrize('change',[
    {'identity_sha256':'wrong'}, {'fit_attempts_dispatched':10},
    {'fit_attempt_limit':12}, {'initial_fit_attempt_limit':11},
    {'fit_limit_amendments':[{'previous':9,'new':11,'note':'unknown'}]},
    {'active':{'job_sha256':'unrecognized'}}])
def test_unknown_checkpoint_history_rejected_without_mutation(change):
    state=progress();state.update(change);before=copy.deepcopy(state)
    with pytest.raises(repair.B5RetryBudgetError):
        repair.amended_budget(state,budget(),plan())
    assert state==before


@pytest.mark.parametrize('change',[
    {'stage':'development_evaluation'}, {'plan_sha256':'wrong'},
    {'max_fit_attempts':12}, {'max_jobs':2}, {'max_download_bytes':1}])
def test_other_budget_identity_or_larger_retry_allowance_rejected(change):
    value=budget();value.update(change)
    with pytest.raises(repair.B5RetryBudgetError):
        repair.amended_budget(progress(),value,plan())


@pytest.mark.parametrize('change',['method','grid','order','plan-sha'])
def test_unknown_native_grid_or_method_rejected(change):
    value=plan()
    if change=='method':value['methods']=['B4']
    elif change=='grid':value['jobs'][0]['configuration']['shrinkage']=.99
    elif change=='order':value['jobs'][0],value['jobs'][1]=value['jobs'][1],value['jobs'][0]
    else:value['plan_sha256']='unexpected'
    with pytest.raises(repair.B5RetryBudgetError):
        repair.amended_budget(progress(),budget(),value)


def test_changed_completed_fit_record_is_not_silently_reused_or_replaced():
    state=progress();key=next(iter(state['completed']))
    state['completed'][key]['attempt']=1
    with pytest.raises(repair.B5RetryBudgetError):
        repair.amended_budget(state,budget(),plan())
    state=progress();state['completed'].pop(next(iter(state['completed'])))
    with pytest.raises(repair.B5RetryBudgetError):
        repair.amended_budget(state,budget(),plan())


def test_valid_remaining_active_attempt_is_preserved_for_native_resume():
    state=progress();changed=repair.amended_budget(state,budget(),plan())
    native._attempt_limit(state,changed,initial_limit=9)
    row=plan()['jobs'][7];state['fit_attempts_dispatched']=10
    state['active']=dict(job_sha256=row['job_sha256'],attempt=10,phase='FIT_DISPATCHED',
        directory=repair.OUTPUT+'/fits/'+row['job_sha256']+'/attempt-000010')
    before=copy.deepcopy(state)
    assert repair.amended_budget(state,budget(),plan())['max_fit_attempts']==11
    assert state==before


@pytest.fixture
def integration(tmp_path):
    root=tmp_path/'project';root.mkdir()
    write(root/repair.OUTPUT/'progress.json',progress())
    write(root/'runs/base-budget.json',budget())
    original_ref=descriptor(root,root/'runs/base-budget.json')
    calls=[];events=[]
    class Production:
        def __init__(self,root):self.root=root
        def budget(self,action_key,*args,**kwargs):
            calls.append((action_key,args,kwargs))
            return copy.deepcopy(original_ref)
    def immutable(root,relative,value):
        path=Path(root)/relative
        if path.exists():assert json.loads(path.read_text())==value
        else:write(path,value)
        return descriptor(Path(root),path)
    campaign=types.SimpleNamespace(Production=Production,OUTPUT='runs/training_campaign_v1',
        digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest(),
        immutable=immutable)
    @contextmanager
    def original_active(root,action,**kwargs):yield 'native-startup-value'
    startup=types.SimpleNamespace(active=original_active)
    action=dict(kind='calibration_profile',key='profile-B5',method='B5',
                expected_receipt=repair.OUTPUT+'/receipt.json')
    return types.SimpleNamespace(root=root,campaign=campaign,calls=calls,events=events,
        action=action,startup=startup,old_budget=Production.budget,
        emit=lambda event,**fields:events.append(dict(event=event,**fields)),
        original_ref=original_ref)


def inventory(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_future_native_budget_call_creates_only_new_immutable_budget(integration):
    e=integration;before=inventory(e.root)
    repair.install(e.startup,e.root,campaign=e.campaign,development=native,emit=e.emit)
    assert before==inventory(e.root)
    with e.startup.active(e.root,e.action) as result:
        assert result=='native-startup-value'
        ref=e.campaign.Production(e.root).budget('profile-B5',{'unchanged-body':True},evidence=['same'])
        again=e.campaign.Production(e.root).budget('profile-B5',{'unchanged-body':True},evidence=['same'])
        assert ref==again
    assert e.campaign.Production.budget is e.old_budget
    after=inventory(e.root)
    assert all(after[k]==v for k,v in before.items())
    assert len(set(after)-set(before))==1
    assert json.loads((e.root/ref['path']).read_text())['max_fit_attempts']==11
    assert e.calls==[('profile-B5',({'unchanged-body':True},),{'evidence':['same']})]*2


def test_complete_receipt_retains_original_budget_and_no_new_files(integration):
    e=integration;write(e.root/repair.OUTPUT/'receipt.json',{'status':'native-consumer-verifies'})
    before=inventory(e.root)
    with repair.active(e.root,e.action,campaign=e.campaign,development=native,emit=e.emit):
        assert e.campaign.Production(e.root).budget('profile-B5')==e.original_ref
    assert before==inventory(e.root)


def test_other_actions_or_budget_keys_unchanged(integration):
    e=integration
    with repair.active(e.root,dict(e.action,key='profile-B4'),campaign=e.campaign,development=native):
        assert e.campaign.Production.budget is e.old_budget
    before=inventory(e.root)
    with repair.active(e.root,e.action,campaign=e.campaign,development=native,emit=e.emit):
        assert e.campaign.Production(e.root).budget('other-key')==e.original_ref
    assert before==inventory(e.root)


def test_install_idempotent_and_has_no_eager_budget_publication(integration):
    e=integration;before=inventory(e.root)
    one=repair.install(e.startup,e.root,campaign=e.campaign,development=native)
    wrapped=e.startup.active
    two=repair.install(e.startup,e.root,campaign=e.campaign,development=native)
    assert one['operational_budget_published'] is False
    assert two['status']=='ALREADY_INSTALLED' and e.startup.active is wrapped
    assert before==inventory(e.root) and not e.calls


def test_native_budget_failure_preserved_and_wrapper_restored(integration):
    e=integration
    def failure(self,*args,**kwargs):raise RuntimeError('original clock/budget failure')
    e.campaign.Production.budget=failure
    before=inventory(e.root)
    with pytest.raises(RuntimeError,match='original clock/budget failure'):
        with repair.active(e.root,e.action,campaign=e.campaign,development=native):
            e.campaign.Production(e.root).budget('profile-B5')
    assert e.campaign.Production.budget is failure
    assert before==inventory(e.root)


def test_install_root_mismatch_rejected_before_scientific_work(integration,tmp_path):
    e=integration;other=tmp_path/'other';other.mkdir()
    repair.install(e.startup,e.root,campaign=e.campaign,development=native)
    with pytest.raises(repair.B5RetryBudgetError):
        with e.startup.active(other,e.action):
            raise AssertionError('must not reach action')
    assert e.campaign.Production.budget is e.old_budget
