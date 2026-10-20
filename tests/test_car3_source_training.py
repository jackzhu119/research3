"""Mathematical software fixtures only: CPU numbers never scientific EEG fits."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from p3work import car3_source_training as training
from p3work.contracts import SOURCE_IDS,digest
from p3work.f0_bridge import CompactEEGNet,load_verified_source
from tools.run_controlled_pilot import _same_tree

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('car3_source_cli_test',ROOT/'tools/train_car3_source.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


@pytest.fixture
def cache(tmp_path,monkeypatch):
    rng=np.random.default_rng(20261019)
    x=rng.normal(size=(16,3,128)).astype(np.float32);x-=x.mean(1,keepdims=True);x-=x.mean(2,keepdims=True)
    rows=[];labels=[];locations=[]
    for d,dataset in enumerate(sorted(SOURCE_IDS)):
        for j in range(4):
            i=d*4+j;label=j%2
            rows.append({'dataset':dataset,'subject_id':str(j//2+1),'event_uid':f'MATH_ONLY/{dataset}/{j}',
                'role':'source_sensitivity_preparation_only','processed_signal_sha256':training.preparation.signal_sha(x[i]),'label':label})
            labels.append(label);locations.append((x,i))
    monkeypatch.setattr(training,'_code_policy',lambda root:{'fixture_scope':'MATHEMATICAL_SOFTWARE_ONLY'})
    return training.VerifiedCar3Cache(tmp_path,rows,np.asarray(labels,dtype=np.int64),locations,
        {'fixture':'a'*64},'b'*64,manifest_path='fixture.json',manifest_sha256='c'*64)


def config():
    return {**training.SOURCE_CONFIG,'epochs':2,'steps_per_epoch':2,'batch_size':4,'threads':1,
            'checkpoint_every_updates':2}


def test_model_is_same_layers_with_independent_spatial_three_and_strict_weights():
    a=CompactEEGNet(11,128);b=training.CompactEEGNet3()
    assert a.features[2].weight.shape==(16,1,11,1)
    assert b.features[2].weight.shape==(16,1,3,1)
    assert b.classifier.weight.shape==a.classifier.weight.shape
    with pytest.raises(RuntimeError):b.load_state_dict(a.state_dict(),strict=True)
    with pytest.raises(training.SourceTrainingError):b(torch.zeros((2,11,128)))


def test_plan_source_only_lodo_seed_final_rule_and_immutable_config(cache):
    plan=training.build_training_plan(cache)
    assert plan['planned_neural_fits']==15 and plan['max_committed_updates']==76800
    assert training.build_training_plan(cache,profile='confirmation')['planned_neural_fits']==25
    assert plan['independent_3G_and_all14_methods_ready'] is False
    for job in plan['jobs']:
        assert set(job['train_dataset_ids'])<=SOURCE_IDS
        if job['phase']=='lodo':
            assert {cache.rows[i]['dataset'] for i in job['validation_indices']}=={job['heldout_dataset']}
            assert job['heldout_dataset'] not in job['train_dataset_ids']
        else:assert job['validation_indices']==[]
    assert training.final_epoch_from_lodo([2,3,4,80])==3
    original=copy.deepcopy(training.SOURCE_CONFIG);plan['configuration']['betas'][0]=0
    assert training.SOURCE_CONFIG==original
    with pytest.raises(training.SourceTrainingError):training.validate_training_plan(cache,plan)


def test_exact_CPU_fixture_resume_restores_optimizer_rng_sampler_and_predictions(cache):
    job=training.build_training_plan(cache)['jobs'][0];runtime={'device':'cpu','scope':'MATH_ONLY'}
    full=cache.root/'continuous';split=cache.root/'split'
    training.run_fit(cache,job,full,runtime=runtime,epochs=2,config=config(),synthetic_fixture=True)
    paused=training.run_fit(cache,job,split,runtime=runtime,epochs=2,config=config(),synthetic_fixture=True,max_new_updates=1)
    assert paused['complete'] is False
    training.run_fit(cache,job,split,runtime=runtime,epochs=2,config=config(),synthetic_fixture=True)
    left=training.verified_checkpoint(full/'last_state.pt');right=training.verified_checkpoint(split/'last_state.pt')
    assert left['completed_updates']==right['completed_updates']==4
    for key in ['model','optimizer','rng','best_model','history']:
        assert _same_tree(left[key],right[key]),key
    np.testing.assert_array_equal(np.load(full/'training_logits.npy'),np.load(split/'training_logits.npy'))
    assert json.loads((full/'status.json').read_text())['status']=='SYNTHETIC_FIXTURE_COMPLETE'
    with pytest.raises(training.SourceTrainingError):training.verify_complete_fit(cache,job,full)


def test_real_CPU_and_wrong_config_block_before_model_fit(cache):
    job=training.build_training_plan(cache)['jobs'][0]
    with pytest.raises(training.SourceTrainingError,match='CUDA'):
        training.run_fit(cache,job,cache.root/'not_created',runtime={'device':'cpu'},epochs=80)
    assert not (cache.root/'not_created').exists()


def test_integrity_torn_checkpoint_blocks_resume(cache):
    job=training.build_training_plan(cache)['jobs'][0];folder=cache.root/'tamper'
    training.run_fit(cache,job,folder,runtime={'device':'cpu'},epochs=2,config=config(),synthetic_fixture=True,max_new_updates=1)
    with (folder/'last_state.pt').open('ab') as stream:stream.write(b'TAMPER')
    with pytest.raises(training.SourceTrainingError,match='SHA256'):
        training.run_fit(cache,job,folder,runtime={'device':'cpu'},epochs=2,config=config(),synthetic_fixture=True)


def test_output_claim_is_one_output_per_seed_and_never_adopts_old(cache):
    rt={'device':'cuda:0','fixture_scope':'MATH_ONLY_NO_GPU'}
    a,_=training._claim(cache,'runs/three',rt,'CAR3_SOURCE_F0_ONLY',seeds=[20261018])
    training._claim(cache,'runs/three',rt,'CAR3_SOURCE_F0_ONLY',seeds=[20261018])
    with pytest.raises(training.SourceTrainingError):
        training._claim(cache,'runs/different',rt,'CAR3_SOURCE_F0_ONLY',seeds=[20261018])
    p=cache.root/'runs/unclaimed';p.mkdir(parents=True);(p/'old.pt').write_bytes(b'not weights')
    with pytest.raises(training.SourceTrainingError,match='unclaimed'):
        training._claim(cache,p,rt,'CAR3_SOURCE_ENGINEERING_ONLY')


def test_ledger_conservatively_charges_unclean_call_and_refuses_missing_history(cache):
    out=cache.root/'runs/ledger';folder=out/'one';folder.mkdir(parents=True)
    path,state,observe=training._ledger(out,[folder],limit=10)
    training.atomic_json(folder/'execution_progress.json',{'observed_optimizer_calls_total':3})
    assert observe()==7
    state['active_fit']='one';training.atomic_json(path,state)
    path,state,observe=training._ledger(out,[folder],limit=10)
    assert state['uncertain_optimizer_calls']==1 and observe()==6
    training.atomic_json(folder/'execution_progress.json',{'observed_optimizer_calls_total':2})
    with pytest.raises(training.SourceTrainingError,match='rolled_back'):observe()
    path.unlink()
    with pytest.raises(training.SourceTrainingError,match='not_reconstructed'):
        training._ledger(out,[folder],limit=10)


def test_immutable_array_and_epoch_refuse_changed_payload(tmp_path):
    p=tmp_path/'x.npy';training.immutable_npy(p,np.ones(3));before=p.read_bytes()
    training.immutable_npy(p,np.ones(3));assert p.read_bytes()==before
    with pytest.raises(training.SourceTrainingError):training.immutable_npy(p,np.zeros(3))
    q=tmp_path/'epoch.pt';training.immutable_checkpoint(q,{'x':torch.ones(1)})
    with pytest.raises(training.SourceTrainingError):training.immutable_checkpoint(q,{'x':torch.zeros(1)})


def test_old_main11_pilot_or_receipt_cannot_admit_car3(cache):
    p=cache.root/'oldpilot.json';p.write_text(json.dumps({'status':'ENGINEERING_SOURCE_PILOT_ONLY','exact_resume_verified':True}))
    with pytest.raises(training.SourceTrainingError,match='not_main11_proof'):
        training.verify_engineering_pilot(cache,p)
    with pytest.raises(training.SourceTrainingError):
        training.execute_plan(cache,training.build_training_plan(cache),'runs/forbidden',runtime={'device':'cuda:0'},pilot_path=p)
    assert not (cache.root/'runs/forbidden').exists()
    own=cache.root/'own.json';own.write_text(json.dumps({'status':'CAR3_SOURCE_F0_COMPLETE','architecture':training.ARCHITECTURE,'synthetic':False}))
    contract=json.loads((ROOT/'contracts/input_contract.json').read_text())
    with pytest.raises(ValueError):load_verified_source(cache.root,'own.json',contract)


def test_cli_execute_requires_two_flags_before_cache_or_CUDA(monkeypatch,capsys):
    monkeypatch.setattr(training,'verify_inputs',lambda *a:pytest.fail('cache read before flags'))
    monkeypatch.setattr(cli,'cuda_runtime',lambda:pytest.fail('GPU checked before flags'))
    assert cli.main(['--execute'])==2
    assert json.loads(capsys.readouterr().out)['execution_entered'] is False


def test_cli_dryrun_reads_no_GPU_and_claims_zero_fits(cache,monkeypatch,capsys):
    monkeypatch.setattr(training,'verify_inputs',lambda *a:cache)
    monkeypatch.setattr(cli,'cuda_runtime',lambda:pytest.fail('dryrun touched CUDA'))
    assert cli.main([])==0
    result=json.loads(capsys.readouterr().out)
    assert result['actual_fits_started']==0 and result['GPU_checked'] is False
    assert result['CAR3_GPU_resume_pilot_verified'] is False


def test_execution_contract_keeps_original_preparation_false_and_geometry():
    new=json.loads((ROOT/'contracts/car3_source_training_v1.json').read_text())
    old=json.loads((ROOT/'contracts/sensitivity_car3_v1.json').read_text())
    assert new==training.EXPECTED_CONTRACT
    assert old['training_ready'] is False and old['GPU_training_entry_implemented'] is False
    assert training.digest(old)==training.PREPARATION_SHA
    assert new['centered_dimension_d']==2 and new['Rref_shape']==[2,2]
    assert new['future_spatial_rank_choices']==new['future_LoRA_rank_choices']==[1,2]
    assert new['future_CSP_pairs']==1 and new['BCIIV2b_accepted'] is False
    assert new['Ma2022_in_source'] is False and new['all14_methods_implemented'] is False


def test_missing_wall_history_cannot_reset_consumed_source_work(cache):
    output=cache.root/'runs/missing_wall';output.mkdir(parents=True)
    training.atomic_json(output/'optimizer_call_ledger.json',{'observed_plus_uncertain_calls':3})
    with pytest.raises(training.SourceTrainingError,match='wall_ledger_not_reconstructed'):
        training._start_wall(output,60)
    assert not (output/'operational_budget.json').exists()


def test_journal_ahead_of_checkpoint_replays_with_RNG_and_accounts_extra_calls(cache):
    job=training.build_training_plan(cache)['jobs'][0];runtime={'device':'cpu','scope':'MATH_ONLY'}
    full=cache.root/'full_replay_reference';folder=cache.root/'replayed'
    training.run_fit(cache,job,full,runtime=runtime,epochs=2,config=config(),synthetic_fixture=True)
    training.run_fit(cache,job,folder,runtime=runtime,epochs=2,config=config(),synthetic_fixture=True,max_new_updates=1)
    saved=(folder/'last_state.pt').read_bytes();side=training.checkpoint_sidecar(folder/'last_state.pt').read_bytes()
    training.run_fit(cache,job,folder,runtime=runtime,epochs=2,config=config(),synthetic_fixture=True,max_new_updates=1)
    # Explicit interruption structure fixture: journal reached update2 while
    # latest durable checkpoint is update1. This is not a real crash claim.
    (folder/'last_state.pt').write_bytes(saved);training.checkpoint_sidecar(folder/'last_state.pt').write_bytes(side)
    for p in folder.glob('epoch_0001.pt*'):p.unlink()
    summary=training.run_fit(cache,job,folder,runtime=runtime,epochs=2,config=config(),synthetic_fixture=True)
    left=training.verified_checkpoint(full/'last_state.pt');right=training.verified_checkpoint(folder/'last_state.pt')
    for key in ('model','optimizer','rng','history'):assert _same_tree(left[key],right[key])
    assert summary['replayed_observed_updates_this_invocation']==1
    assert summary['committed_optimizer_updates']==4 and summary['observed_optimizer_calls_total']==5


def test_finalization_recovery_cannot_promote_synthetic_weights(cache):
    job=training.build_training_plan(cache)['jobs'][0];folder=cache.root/'synthetic_not_promotable'
    training.run_fit(cache,job,folder,runtime={'device':'cpu'},epochs=2,config=config(),synthetic_fixture=True)
    (folder/'status.json').unlink()
    with pytest.raises(training.SourceTrainingError):
        training.verify_complete_fit(cache,job,folder,recover_finalization=True)
    assert not (folder/'status.json').exists()


@pytest.mark.parametrize('cursor,reason',[(4,'Partial_CAR3_engineering_leg'),(8,'single_call_leg_proof_not_reconstructed')])
def test_interrupted_or_unproven_continuous_pilot_cannot_be_relabelled(cache,monkeypatch,cursor,reason):
    runtime={'device':'cuda:0','fixture_scope':'STRUCTURE_TEST_ONLY_NO_GPU'}
    output,_=training._claim(cache,'runs/engineering_partial',runtime,'CAR3_SOURCE_ENGINEERING_ONLY')
    folders=[output/'continuous',output/'resumed']
    training._ledger(output,folders,limit=32,max_invocations=8)
    _,_,close=training._start_wall(output,1800);close()
    training.atomic_checkpoint(folders[0]/'last_state.pt',{'completed_updates':cursor,'fixture_scope':'STRUCTURE_ONLY_NOT_MODEL'})
    training.atomic_json(folders[0]/'execution_progress.json',{'observed_optimizer_calls_total':cursor})
    monkeypatch.setattr(training,'_run_with_ledger',lambda *a,**k:pytest.fail('invalid continuous baseline was resumed'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('structure check touched GPU'))
    with pytest.raises(training.SourceTrainingError,match=reason):
        training.engineering_pilot(cache,output,runtime=runtime)
    assert not (output/'engineering_receipt.json').exists()


def test_final_wall_close_persists_late_serialization_overrun_before_block(cache,monkeypatch):
    output=cache.root/'runs/overrun';output.mkdir(parents=True)
    _,_,close=training._start_wall(output,60)
    def late_observation(path,state):
        state['consumed_active_seconds']=61.
        state['clock_state']['wall_high_water_seconds']=61.
        training.atomic_json(path,state)
    monkeypatch.setattr(training,'_observe_source_wall',late_observation)
    with pytest.raises(training.SourceTrainingError,match='overrun_no_completion'):close()
    wall=json.loads((output/'operational_budget.json').read_text())
    assert wall['consumed_active_seconds']==61 and wall['active_invocation'] is False
    assert wall['clock_state']['interval'] is None


def test_closed_resource_snapshot_is_immutable_across_later_live_work(cache):
    output=cache.root/'runs/snapshot';output.mkdir(parents=True)
    training._ledger(output,[],limit=32,max_invocations=8)
    _,_,close=training._start_wall(output,60);close()
    runtime={'device':'cpu','scope':'METADATA_MATH_FIXTURE_ONLY_NO_PILOT'}
    item=training._freeze_resource_snapshot(cache,output,20261018,runtime,scope='CAR3_SOURCE_ENGINEERING_ONLY',statuses=[])
    original=(cache.root/item['path']).read_bytes()
    wall=json.loads((output/'operational_budget.json').read_text())
    wall['consumed_active_seconds']+=1;wall['clock_state']['wall_high_water_seconds']=wall['consumed_active_seconds']
    training.atomic_json(output/'operational_budget.json',wall)
    training._verify_resource_snapshot(cache,item,output,20261018,runtime,scope='CAR3_SOURCE_ENGINEERING_ONLY')
    assert (cache.root/item['path']).read_bytes()==original
    broken=json.loads(original);broken['wall_snapshot']['consumed_active_seconds']=61
    broken['wall_snapshot']['clock_state']['wall_high_water_seconds']=61
    with pytest.raises(training.SourceTrainingError,match='not_closed_within_budget'):
        training._validate_closed_resource_values(broken['wall_snapshot'],broken['optimizer_snapshot'],scope='CAR3_SOURCE_ENGINEERING_ONLY')


def test_last_prediction_or_serialization_crossing_deadline_never_completes(cache,monkeypatch):
    job=next(j for j in training.build_training_plan(cache)['jobs'] if j['phase']=='final')
    folder=cache.root/'late_prediction';late={'value':False}
    predict=training.predict_indices
    def delayed_prediction(*args,**kwargs):
        value=predict(*args,**kwargs);late['value']=True;return value
    monkeypatch.setattr(training,'predict_indices',delayed_prediction)
    monkeypatch.setattr(training.time,'monotonic',lambda:10. if late['value'] else 0.)
    with pytest.raises(training.SourceTrainingError,match='serialization_crossed_wall_deadline'):
        training.run_fit(cache,job,folder,runtime={'device':'cpu'},epochs=2,config=config(),
                         synthetic_fixture=True,deadline=5.)
    status=json.loads((folder/'status.json').read_text())
    assert status['status'] not in {'SYNTHETIC_FIXTURE_COMPLETE','CAR3_SOURCE_FIT_COMPLETE'}
