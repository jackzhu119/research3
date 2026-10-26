import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from paper3r10 import cli,study
from paper3r10.algorithm import Config
from paper3r10.platform.common import atomic,digest,load_plan,sign_plan


@pytest.fixture
def prepared(tmp_path):
    root=tmp_path/'source';root.mkdir()
    package=tmp_path/'package';(package/'paper3r10/platform').mkdir(parents=True)
    for name in ('START.py','paper3r10/algorithm.py','paper3r10/platform/entry.py'):
        (package/name).write_text('# isolated software fixture\n')
    raw=root/'small.json';atomic(raw,{'scope':'software-test-no-EEG'})
    item=study.describe(raw,root)
    registry={'schema_version':'paper3-r10-study-registry-v1','root':str(root),
        'source_by_seed':{str(s):{'source_receipt':item,'reference_receipt':item} for s in study.SEEDS},
        'external_manifests':{'pan2023':item,'weibo2014':item},
        'baseline_artifacts':{str(s):{m:{'artifact':item} for m in study.NEEDS_ARTIFACTS} for s in study.SEEDS},
        'profile_reference_by_seed':{str(s):{'checkpoint':item,'result':item} for s in study.SEEDS},
        'profile_raw_reference_by_seed':{str(s):{'checkpoint':item,'result':item} for s in study.SEEDS},
        'previously_used_participants':['old-person']}
    path=root/'registry.json';atomic(path,registry)
    desc=study.describe(path,root)
    return root,package,registry,path,desc


def frozen(root,package,selected=None):
    selected=selected or study.default_configuration()
    value={'status':'FROZEN','scientific_scope':'REAL_DATA','selected_configuration':selected,
        'target_query_labels_used_for_selection':False,'development_query_labels_used':True,
        'selection_roles':['meta_train','meta_validation'],
        'study_registry_sha256':study.describe(root/'registry.json',root)['sha256'],
        'code_manifest_sha256':digest(study.production_manifest(package)),
        'comparison_family':study.PRIMARY_FAMILY,
        'secondary_ablation_family':study.SECONDARY_FAMILY,
        'reference_strategy':'reuse_r9_ea',
        'allowed_ablations':study.ablation_configurations(selected)}
    path=root/'freeze.json';atomic(path,value)
    return study.describe(path,root)


def test_defaults_equal_actual_algorithm_config():
    assert study.default_configuration()==asdict(Config())


def test_mechanism_is_fifteen_finite_development_only_tasks(prepared):
    root,package,_,_,desc=prepared
    plan=study.build_plan(root,desc,package=package)
    assert len(plan['tasks'])==15
    assert sum(t['family']=='feature_reference' for t in plan['tasks'])==1
    assert sum(t['method']=='profile' for t in plan['tasks'])==9
    assert {t['role'] for t in plan['tasks']}=={'meta_train','meta_validation'}
    cases={(t['config']['amplitude'],t['config']['evidence_fraction']) for t in plan['tasks'] if t['method']=='profile'}
    assert cases=={(a,c) for a in (.3,1.,3.) for c in (.01,.1,.5)}
    evaluations=[t for t in plan['tasks'] if t['family']=='evaluate']
    assert all(t['budgets']==[4,8,20] for t in evaluations)
    assert all(t['session_subset']['required_participants']==8 for t in evaluations)
    p=study.write_plan(plan,root/'plan.json');assert load_plan(p)[1]['recipe_sha256']==plan['recipe_sha256']


def test_preparation_is_idempotent_and_locked(prepared):
    root,package,_,_,desc=prepared
    a=study.build_plan(root,desc,package=package);b=study.build_plan(root,desc,package=package)
    assert a==b
    p=study.write_plan(a,root/'plan.json');mtime=p.stat().st_mtime_ns
    study.write_plan(b,p);assert p.stat().st_mtime_ns==mtime
    bad=copy.deepcopy(a);bad['tasks'][1]['config']['evidence_fraction']=.9
    with pytest.raises(ValueError):study.write_plan(bad,p)


def test_formal_requires_actual_freeze_and_all_strong_controls(prepared):
    root,package,registry,path,desc=prepared
    with pytest.raises(ValueError,match='freeze'):
        study.build_plan(root,desc,stage='formal_exploratory',package=package)
    freeze=frozen(root,package)
    plan=study.build_plan(root,desc,stage='formal_exploratory',package=package,freeze=freeze)
    assert len(plan['tasks'])==165
    assert {t['method'] for t in plan['tasks'] if t['family']=='evaluate'}==set(study.METHODS)
    assert all(t['budgets']==[4,8,12,16,20] for t in plan['tasks'] if t['family']=='evaluate')
    file=study.write_plan(plan,root/'formal.json');load_plan(file)
    registry['baseline_artifacts']['20261019'].pop('sgd100');atomic(path,registry)
    desc=study.describe(path,root);freeze=frozen(root,package)
    with pytest.raises(ValueError,match='sgd100'):
        study.build_plan(root,desc,stage='formal_exploratory',package=package,freeze=freeze)


def test_missing_external_cache_never_downgrades_dataset(prepared):
    root,package,registry,path,_=prepared
    registry['external_manifests'].pop('weibo2014');atomic(path,registry)
    desc=study.describe(path,root)
    with pytest.raises(ValueError,match='weibo2014'):
        study.build_plan(root,desc,stage='formal_exploratory',package=package,freeze=frozen(root,package))


def test_no_new_confirmation_data_is_a_real_block(prepared):
    root,package,_,_,desc=prepared
    freeze=frozen(root,package)
    with pytest.raises(ValueError,match='independent confirmation'):
        study.build_plan(root,desc,stage='formal_confirmatory',package=package,freeze=freeze)
    template=root/'confirmation.json';atomic(template,{'status':'TEMPLATE','cohort_kind':'independent_confirmation'})
    with pytest.raises(ValueError,match='chronology'):
        study.build_plan(root,desc,stage='formal_confirmatory',package=package,freeze=freeze,
                         confirmation=study.describe(template,root))


def test_reviewed_chronological_confirmation_has_fiftyseven_tasks(prepared):
    root,package,_,_,desc=prepared
    contract={'n_times':128,'synthetic_fixture':True}
    atomic(root/'contracts/input_contract.json',contract)
    signals=root/'new-signals.npy';np.save(signals,np.zeros((40,11,128),dtype=np.float64))
    rows=[{'participant_id':'new-person','session_id':'1','tasknumber':1,
        'event_uid':f'new-event-{i}','trial_uid':f'new-event-{i}','original_trial_index':i,
        'label':i%2,'valid':True,'cache_index':i,
        'physical_run_id':'prefix' if i<20 else 'future',
        'physical_run_complete_verified':True} for i in range(40)]
    events=root/'new-events.json';atomic(events,rows)
    dataset_identity={'id':'software-contract-fixture-only'}
    review=root/'independence-review.json'
    atomic(review,{'status':'REVIEWED','prior_design_history_reviewed':True,
        'previously_used_for_algorithm_design':False,'reviewer':'synthetic-software-fixture',
        'reviewed_utc':'2026-10-26T00:00:00Z','dataset_identity':dataset_identity,
        'participants':['new-person']})
    path=root/'confirmation.json'
    atomic(path,{'schema_version':'paper3-r10-independent-cache-v1','status':'COMPLETE',
        'cohort_kind':'independent_confirmation','dataset_identity':dataset_identity,
        'previously_used_for_algorithm_design':False,'acquisition_order_verified':True,
        'physical_run_id_available':True,'participants':['new-person'],
        'signal_contract_sha256':digest(contract),'independence_review':study.describe(review,root),
        'sessions':[{'participant_id':'new-person','session_id':'1','acquisition_order_verified':True,
                     'X':study.describe(signals,root),'events':study.describe(events,root)}]})
    plan=study.build_plan(root,desc,stage='formal_confirmatory',package=package,
                          freeze=frozen(root,package),confirmation=study.describe(path,root))
    assert len(plan['tasks'])==57
    study.write_plan(plan,root/'confirm-plan.json');load_plan(root/'confirm-plan.json')
    value=json.loads(path.read_text());value['participants']=['old-person'];atomic(path,value)
    with pytest.raises(ValueError,match='overlap'):
        study.build_plan(root,desc,stage='formal_confirmatory',package=package,
                         freeze=frozen(root,package),confirmation=study.describe(path,root))


def test_changed_code_or_source_registry_invalidates_freeze(prepared):
    root,package,registry,path,desc=prepared;freeze=frozen(root,package)
    (package/'paper3r10/algorithm.py').write_text('# changed implementation\n')
    with pytest.raises(ValueError,match='Code changed'):
        study.build_plan(root,desc,stage='formal_exploratory',package=package,freeze=freeze)
    registry['root_note']='new registry bytes';atomic(path,registry)
    with pytest.raises(ValueError,match='registry'):
        study.build_plan(root,study.describe(path,root),stage='formal_exploratory',package=package,freeze=freeze)


def test_submit_command_without_flag_only_prepares(monkeypatch,tmp_path,capsys):
    from paper3r10.platform import runtime
    calls=[]
    monkeypatch.setattr(runtime,'start',lambda path,**kw:calls.append(kw) or {'status':'PREPARED_NOT_SUBMITTED'})
    assert cli.main(['submit','--plan',str(tmp_path/'plan.json')])==0
    assert calls==[{'submit':False}]
    assert cli.main(['submit','--plan',str(tmp_path/'plan.json'),'--submit'])==0
    assert calls[-1]=={'submit':True}


def test_cli_error_is_bounded_and_does_not_submit(tmp_path,capsys):
    assert cli.main(['status','--plan',str(tmp_path/'absent.json')])==2
    assert 'R10_COMMAND_BLOCKED' in capsys.readouterr().out
