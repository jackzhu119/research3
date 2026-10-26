"""R11 plan/CLI contracts, no GPU or real Slurm/network operations."""
import copy,json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import pytest
from paper3r11 import cli,study
from paper3r11.algorithm import Config
from paper3r11.platform.common import atomic,load_plan

@pytest.fixture
def prepared(tmp_path):
 root=tmp_path/'source';root.mkdir();package=tmp_path/'package'
 (package/'paper3r11/platform').mkdir(parents=True);(package/'configs').mkdir()
 for name in ('START.py','paper3r11/algorithm.py','paper3r11/platform/entry.py'):
  (package/name).write_text('# isolated software fixture\n')
 for name in ('algorithm.default.json','profile_r10.default.json'):
  (package/'configs'/name).write_bytes((study.PACKAGE/'configs'/name).read_bytes())
 small=root/'small.json';atomic(small,{'scope':'SOFTWARE_TEST_ONLY'});item=study.describe(small,root)
 value={'schema_version':'paper3-r11-study-registry-v1','root':str(root),
  'source_by_seed':{str(s):{'source_receipt':item,'reference_receipt':item} for s in study.SEEDS},
  'external_manifests':{'pan2023':item,'weibo2014':item},
  'baseline_artifacts':{str(s):{m:{'artifact':item} for m in study.NEEDS_ARTIFACTS} for s in study.SEEDS},
  'profile_reference_by_seed':{str(s):{'checkpoint':item,'result':item} for s in study.SEEDS},
  'profile_raw_reference_by_seed':{str(s):{'checkpoint':item,'result':item} for s in study.SEEDS},
  'previously_used_participants':['old-person']}
 path=root/'registry.json';atomic(path,value)
 return root,package,value,path,study.describe(path,root)

def small_grid():return {'feature_ranks':[2],'noise_strengths':[1.],'ridges':[.01]}

def test_defaults_match_actual_pfst_and_retain_strong_solver_budget():
 assert study.default_configuration()==asdict(Config())
 assert Config().max_iter==5000
 assert len(study.METHODS)==25 and len(study.PRIMARY_FAMILY)==6 and len(study.SECONDARY_FAMILY)==18

def test_finite_preregistered_grid_has550_unique_configs():
 rows=study.development_candidates()
 assert Counter(x['method'] for x in rows)=={'operator':440,'operator_no_noise_tuned':55,'operator_erm_tuned':55}
 assert len({x['candidate_id'] for x in rows})==550
 assert all(x['config']['noise_strength']>0 for x in rows if x['method']=='operator')
 assert all(x['config']['noise_strength']==0 and x['config']['noise_type']=='none' for x in rows if x['method']!='operator')
 with pytest.raises(ValueError):study.development_candidates({'feature_ranks':[2,2],'noise_strengths':[1.],'ridges':[.01]})

def test_full_campaign10270_tasks_four64g_workers_and_phase_dag(prepared):
 root,package,_,_,descriptor=prepared
 plan=study.build_plan(root,descriptor,package=package)
 assert len(plan['tasks'])==10270
 assert plan['resources']=={'jobs':4,'gpus_per_job':1,'cpus_per_job':16,'memory_per_job':'64G','partition':'hpc_gpu','time_limit':'5-03:00:00'}
 assert Counter(t['family'] for t in plan['tasks'])=={'feature_reference':3,'operator_train':8385,'evaluate':1881,'selection':1}
 selection=next(t for t in plan['tasks'] if t['family']=='selection')
 assert len(selection['requires_terminal'])==1656
 assert selection['requires']==[]
 assert all(t['role']=='meta_train' for t in plan['tasks'] if t['family'] in ('feature_reference','operator_train'))
 assert all(selection['key'] in t['requires'] for t in plan['tasks'] if t['phase']=='exploratory')
 target=[t for t in plan['tasks'] if t['family']=='evaluate' and t['phase']=='exploratory']
 assert len(target)==225 and {t['method'] for t in target}==set(study.METHODS)
 assert all(t['budgets']==[4,8,12,16,20] for t in target)
 path=study.write_plan(plan,root/'full-plan.json')
 assert path.stat().st_size<64*1024*1024
 assert load_plan(path)[1]['recipe_sha256']==plan['recipe_sha256']

def test_required_data_missing_is_not_replaced_with_another_cohort(prepared):
 root,package,value,path,_=prepared
 value['external_manifests'].pop('weibo2014');atomic(path,value)
 with pytest.raises(ValueError,match='weibo2014'):
  study.build_plan(root,study.describe(path,root),package=package,grid=small_grid())

def test_optional_old_baseline_absence_does_not_block_campaign(prepared):
 root,package,value,path,_=prepared
 value['baseline_artifacts']['20261019'].pop('sgd100');atomic(path,value)
 plan=study.build_plan(root,study.describe(path,root),package=package,grid=small_grid())
 rows=[t for t in plan['tasks'] if t['method']=='sgd100' and t['seed']==20261019]
 assert len(rows)==3 and all(t['optional_baseline'] and not t['baseline_artifacts'] for t in rows)
 assert any(t['method']=='operator' and t['phase']=='exploratory' for t in plan['tasks'])

def test_plan_identity_idempotent_and_explicit_grid_extension(prepared):
 root,package,_,_,descriptor=prepared
 a=study.build_plan(root,descriptor,package=package,grid=small_grid(),seeds=[20261018],cohorts=['holdout'])
 b=study.build_plan(root,descriptor,package=package,grid=small_grid(),seeds=[20261018],cohorts=['holdout'])
 assert a==b and len(a['tasks'])==92
 path=study.write_plan(a,root/'small-plan.json');stamp=path.stat().st_mtime_ns
 study.write_plan(b,path);assert path.stat().st_mtime_ns==stamp
 bad=copy.deepcopy(a);bad['tasks'][0]['seed']=1
 with pytest.raises(ValueError):study.write_plan(bad,path)
 extended=study.build_plan(root,descriptor,package=package,
  grid={'feature_ranks':[2],'noise_strengths':[1.,3.],'ridges':[.01]},seeds=[20261018],cohorts=['holdout'])
 assert len(extended['tasks'])>len(a['tasks']) and extended['recipe_sha256']!=a['recipe_sha256']

def test_dev_only_9910_tasks_never_reads_target_roles(prepared):
 root,package,_,_,descriptor=prepared
 plan=study.build_plan(root,descriptor,package=package,stage='development')
 assert len(plan['tasks'])==9910 and {t['role'] for t in plan['tasks']}=={'meta_train','meta_validation'}
 assert plan['selection_rule']['performance_stop_gate'] is False

def test_no_fabricated_independent_confirmation(prepared):
 root,package,_,_,descriptor=prepared
 with pytest.raises(ValueError,match='actual new'):
  study.build_plan(root,descriptor,package=package,stage='confirmatory',grid=small_grid())

def test_launch_requires_explicit_submit_and_does_not_repeat_preparation(monkeypatch,tmp_path,capsys):
 from paper3r11.platform import runtime
 calls=[];path=tmp_path/'plan.json'
 monkeypatch.setattr(cli,'prepare_research',lambda args:(path,{'status':'PREPARED_NOT_SUBMITTED'}))
 monkeypatch.setattr(runtime,'start',lambda p,**kw:calls.append((p,kw)) or {'status':'DETACHED_CONTROLLER_STARTED'})
 assert cli.main(['launch','--root',str(tmp_path)])==0 and calls==[]
 assert cli.main(['launch','--root',str(tmp_path),'--submit'])==0
 assert calls==[(path,{'submit':True})]

def test_submit_without_flag_only_prepares(monkeypatch,tmp_path):
 from paper3r11.platform import runtime
 calls=[]
 monkeypatch.setattr(runtime,'start',lambda path,**kw:calls.append(kw) or {'status':'PREPARED_NOT_SUBMITTED'})
 assert cli.main(['submit','--plan',str(tmp_path/'plan.json')])==0 and calls==[{'submit':False}]
