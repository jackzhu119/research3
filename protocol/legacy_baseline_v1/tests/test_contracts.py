"""Synthetic software checks only. No EEG download or scientific result."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from p3.core import (array_hash, assert_subject_disjoint, chronological_support_query, chronological_session_split,
                     read_json, save_arrays, source_lodo, validate_rows)
from p3.models import fit_classical,predict_classical

def toy():
    rng=np.random.default_rng(431);X=rng.normal(0,5e-6,(48,11,128)).astype(np.float32)
    y=np.tile([0,1],24);rows=[]
    for i in range(len(y)):
        rows.append({"dataset":"A" if i<24 else "B","subject":str(i//8),
                     "session":str((i//4)%2),"run":str((i//2)%2),"event_uid":f"event-{i}",
                     "sample":i*512,"onset_s":float(i*4),"session_order":(i//4)%2,
                     "run_order":(i//2)%2,"label":int(y[i]),"signal_sha256":array_hash(X[i])})
    return X,y,rows

def test_subject_leakage_rejected_across_sessions():
    _,_,rows=toy()
    with pytest.raises(ValueError,match="Subject leakage"):
        assert_subject_disjoint(rows,np.array([0,1]),np.array([4,5]))

def test_lodo_is_subject_disjoint_and_preserves_entire_people():
    _,_,rows=toy();folds=source_lodo(rows)
    for name,(train,val) in folds.items():
        assert {rows[i]["dataset"] for i in val}=={name}
        assert {rows[i]["dataset"] for i in train}==({"A","B"}-{name})
        assert_subject_disjoint(rows,train,val)

def test_duplicate_event_uid_and_cross_provider_signal_copy_rejected():
    _,y,rows=toy();rows[1]["event_uid"]=rows[0]["event_uid"]
    with pytest.raises(ValueError,match="Duplicate event_uid"):validate_rows(rows,y)
    _,_,rows=toy();rows[24]["signal_sha256"]=rows[0]["signal_sha256"]
    with pytest.raises(ValueError,match="signal duplicate"):source_lodo(rows)

def test_chronological_split_uses_complete_run_not_random_trials():
    _,_,rows=toy();support,query=chronological_support_query(rows,("A","0"),"0")
    assert set(support)=={0,1};assert set(query)=={2,3}
    assert all(rows[i]["run_order"]<rows[j]["run_order"] for i in support for j in query)
    with pytest.raises(ValueError):chronological_support_query(rows,("A","0"),"0",2)
    early,later=chronological_session_split(rows,("A","0"))
    assert set(early)=={0,1,2,3} and set(later)=={4,5,6,7}
    assert all(rows[i]["session_order"]<rows[j]["session_order"] for i in early for j in later)
    with pytest.raises(ValueError):chronological_session_split(rows,("A","0"),2)

@pytest.mark.parametrize("kind",["csp_lda","ts_lr"])
def test_classical_npz_state_replays_probabilities_without_pickle(tmp_path,kind):
    X,y,rows=toy();state=fit_classical(kind,X[:24],y[:24],rows[:24])
    p,logits=predict_classical(kind,state,X[24:])
    path=tmp_path/"state.npz";save_arrays(path,**state)
    with np.load(path,allow_pickle=False) as z:reloaded=dict(z)
    p2,logits2=predict_classical(kind,reloaded,X[24:])
    np.testing.assert_array_equal(p,p2);np.testing.assert_array_equal(logits,logits2)
    np.testing.assert_allclose(p.sum(axis=1),1)
    assert np.isfinite(p).all()

def test_mne_raw_annotation_sample_and_volt_contract():
    mne=pytest.importorskip("mne")
    from p3.datasets import extract_run
    config=read_json(Path(__file__).parents[1]/"configs/default.json")
    channels=config["preprocessing"]["channels"];sfreq=128
    t=np.arange(10*sfreq)/sfreq
    signal=np.array([(i+1)*1e-6*np.sin(2*np.pi*10*t) for i in range(len(channels))])
    raw=mne.io.RawArray(signal,mne.create_info(channels,sfreq,"eeg"),first_samp=1280,verbose="ERROR")
    raw.set_annotations(mne.Annotations([2,6],[3,3],["left_hand","right_hand"]))
    ds=SimpleNamespace(event_id={"left_hand":1,"right_hand":2},interval=[0,3])
    X,y,rows,audit=extract_run(raw,"SYNTHETIC",1,"0","0",0,0,ds,config)
    assert y==[0,1] and len(X)==2 and X[0].shape==(11,256)
    assert rows[0]["sample"]==1536 and rows[0]["onset_s"]==2.0
    assert 1e-7<float(np.sqrt(np.mean(X[0]**2)))<1e-4
    assert audit["n_binary_events"]==audit["n_retained"]==2
    assert audit["full_eeg_channel_names"]==channels
    with pytest.raises(ValueError,match="missing channels"):
        extract_run(raw.copy().drop_channels([channels[0]]),"SYNTHETIC",1,"0","0",0,0,ds,config)

def test_epoch_boundary_resume_is_identical_to_uninterrupted_training(tmp_path):
    torch=pytest.importorskip("torch")
    from p3.neural import train_eegnet,predict
    X,y,rows=toy();train=np.arange(24);val=np.arange(24,48)
    config={"epochs":4,"patience":20,"steps_per_epoch":2,"batch_size":8,"dropout":0.25,
            "learning_rate":0.001,"min_delta":1e-6,"threads":1,"retain_epoch_checkpoints":False}
    identity={"test":"synthetic_same_protocol"}
    full,summary=train_eegnet(tmp_path/"uninterrupted",X,y,rows,train,val,config,71,identity)
    _,partial=train_eegnet(tmp_path/"resumed",X,y,rows,train,val,config,71,identity,max_new_epochs=2)
    assert not partial["complete"] and partial["epochs_completed"]==2
    resumed,again=train_eegnet(tmp_path/"resumed",X,y,rows,train,val,config,71,identity)
    assert summary==again
    for key,value in full.state_dict().items():assert torch.equal(value,resumed.state_dict()[key])
    np.testing.assert_array_equal(predict(full,X[val],"cpu")[0],predict(resumed,X[val],"cpu")[0])
    a=torch.load(tmp_path/"uninterrupted/last.pt",weights_only=True)
    b=torch.load(tmp_path/"resumed/last.pt",weights_only=True)
    for key in a["model"]:assert torch.equal(a["model"][key],b["model"][key])
    assert torch.equal(a["sampler_rng"],b["sampler_rng"])
    with pytest.raises(ValueError,match="identity changed"):
        train_eegnet(tmp_path/"resumed",X,y,rows,train,val,config,71,{"test":"changed"})

def test_physionet_known_feet_hands_runs_are_audited_and_unknown_labels_fail():
    mne=pytest.importorskip("mne")
    from p3.datasets import extract_run
    config=read_json(Path(__file__).parents[1]/"configs/default.json")
    channels=config["preprocessing"]["channels"]
    raw=mne.io.RawArray(np.zeros((len(channels),1280)),mne.create_info(channels,128,"eeg"),verbose="ERROR")
    raw.set_annotations(mne.Annotations([1,5],[3,3],["hands","feet"]))
    ds=SimpleNamespace(event_id={"left_hand":2,"right_hand":3,"hands":4,"feet":5,"rest":1},interval=[0,3])
    X,y,rows,audit=extract_run(raw,"PhysionetMI",1,"0","1",0,1,ds,config)
    assert X==y==rows==[] and audit["excluded_run"] and audit["n_binary_events"]==0
    raw.set_annotations(mne.Annotations([1],[3],["unknown"]))
    with pytest.raises(ValueError,match="No canonical"):
        extract_run(raw,"PhysionetMI",1,"0","1",0,1,ds,config)
