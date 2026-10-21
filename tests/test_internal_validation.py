"""CPU transaction and boundary tests. No verified source, fit, or GPU substitute."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from p3work import internal_validation as internal
from p3work.contracts import file_digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": path.name, "sha256": file_digest(path)}


def state():
    return {"fit_attempts_dispatched": 0, "dispatched_keys": [], "predictions": {},
        "scores": {}, "labels_revealed": False, "phase": "PREDICTIONS_ONLY"}


def test_dry_plan_never_opens_internal_data_or_CUDA(tmp_path, monkeypatch):
    put(tmp_path/"status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {"status": "COMPLETE"},
        "W3": {"status": "IN_PROGRESS_LOCAL_CPU"}})
    monkeypatch.setattr(internal, "load_episode_catalog", lambda *a, **k: pytest.fail("internal loader reached"))
    monkeypatch.setattr(internal.dev, "_production_runtime", lambda: pytest.fail("CUDA checked"))
    result = internal.plan_internal(tmp_path)
    assert result["minimum_artifact_gaps"] == ["W3_COMPLETE", "whole_package_real_three_seed_development_and_Bstar_freeze"]
    assert result["internal_payloads_accessed"] is False and result["W6_complete"] is False
    assert not (tmp_path/"runs").exists()


def test_execution_without_later_user_GPU_flag_stops_before_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(internal, "verify_freeze", lambda *a: pytest.fail("artifacts read before GPU authority"))
    monkeypatch.setattr(internal, "load_episode_catalog", lambda *a, **k: pytest.fail("internal data opened"))
    with pytest.raises(PermissionError, match="user_started_GPU"):
        internal.execute_internal(tmp_path, freeze=None, budget=None)
    assert list(tmp_path.iterdir()) == []


def test_actual_execute_missing_freeze_never_checks_CUDA_or_writes_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(internal.dev, "_production_runtime", lambda: pytest.fail("CUDA reached"))
    with pytest.raises(Exception):
        internal.execute_internal(tmp_path, freeze={"path": "missing.json", "sha256": "a"*64}, budget=None, user_started_gpu=True)
    assert not (tmp_path/internal.CLAIM).exists()


@pytest.mark.parametrize("change", ["method", "seed_count", "Bstar", "catalog"])
def test_whole_package_rejects_subset_selection(change):
    spec = {"schema_version": "paper3-internal-freeze-spec-v1",
        "configuration_receipts": {m: {} for m in internal.dev.METHODS},
        "confirmation_receipts": [{} for _ in range(42)], "strongest_control_receipt": {}, "artifact_catalog": {}}
    if change == "method": spec["configuration_receipts"].pop("B11")
    if change == "seed_count": spec["confirmation_receipts"].pop()
    if change == "Bstar": spec["strongest_control_receipt"] = None
    if change == "catalog": spec["artifact_catalog"] = None
    with pytest.raises(internal.InternalValidationError, match="all_fourteen"):
        internal._shape(spec)


def test_freeze_tamper_fails_before_rebuilding_any_development(tmp_path, monkeypatch):
    value = {"schema_version": internal.FREEZE_SCHEMA, "synthetic": False, "spec": {}, "freeze_sha256": "a"*64}
    ref = put(tmp_path/"freeze.json", value)
    monkeypatch.setattr(internal, "freeze_development", lambda *a: pytest.fail("unverified freeze rebuilt"))
    with pytest.raises(internal.InternalValidationError, match="immutable_real"):
        internal.verify_freeze(tmp_path, ref)


def test_complete_real_stage_gate_before_dev_or_internal_loading(tmp_path, monkeypatch):
    put(tmp_path/"status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {"status": "COMPLETE"}, "W3": {"status": "PARTIAL_CACHE"}})
    spec = {"schema_version": "paper3-internal-freeze-spec-v1", "configuration_receipts": {m: {} for m in internal.dev.METHODS},
        "confirmation_receipts": [{} for _ in range(42)], "strongest_control_receipt": {}, "artifact_catalog": {}}
    monkeypatch.setattr(internal.dev, "verify_configuration_receipt", lambda *a: pytest.fail("dev loader reached too early"))
    with pytest.raises(internal.InternalValidationError, match="W1_W2_W3"):
        internal.freeze_development(tmp_path, spec, output="runs/freeze.json", execute=True)
    assert not (tmp_path/"runs").exists()


def test_root_wide_claim_is_idempotent_and_changed_inputs_refused(tmp_path):
    identity = {"output": internal.OUTPUT, "freeze": "a"*64, "budget": "b"*64, "runtime": "gpu1"}
    first = internal._claim(tmp_path, identity)
    body = (tmp_path/internal.CLAIM).read_bytes()
    assert first == internal._claim(tmp_path, identity)
    assert (tmp_path/internal.CLAIM).read_bytes() == body
    for key in ("freeze", "budget", "runtime"):
        changed = dict(identity); changed[key] = "other"
        with pytest.raises(internal.InternalValidationError, match="already_bound"):
            internal._claim(tmp_path, changed)
        assert (tmp_path/internal.CLAIM).read_bytes() == body


def test_new_output_cannot_bypass_persistent_claim(tmp_path):
    with pytest.raises(internal.InternalValidationError, match="fixed_internal_output"):
        internal._claim(tmp_path, {"output": "runs/try_again"})
    assert not (tmp_path/internal.CLAIM).exists()


def test_claim_symlink_cannot_be_adopted(tmp_path):
    (tmp_path/"runs").mkdir(); target=tmp_path/"elsewhere.json";target.write_text("{}")
    (tmp_path/internal.CLAIM).symlink_to(target)
    with pytest.raises(Exception, match="symlink"):
        internal._claim(tmp_path, {"output": internal.OUTPUT})


@pytest.mark.parametrize("missing", ["output", "progress", "wall"])
def test_claimed_check_missing_ledger_cannot_restart_with_fresh_counters(tmp_path, missing):
    internal._claim(tmp_path,{"output":internal.OUTPUT})
    destination=tmp_path/internal.OUTPUT
    if missing != "output":
        destination.mkdir(parents=True)
        if missing != "progress":put(destination/"progress.json",state())
        if missing != "wall":put(destination/"wall_budget.json",{})
    with pytest.raises(internal.InternalValidationError,match="missing_ledger"):
        internal._existing_run_gate(tmp_path,destination)
    assert (tmp_path/internal.CLAIM).is_file()


def test_preexisting_unclaimed_output_preserved(tmp_path):
    destination=tmp_path/internal.OUTPUT;destination.mkdir(parents=True)
    (destination/"old.txt").write_text("preserve")
    with pytest.raises(internal.InternalValidationError,match="unowned_internal_output"):
        internal._existing_run_gate(tmp_path,destination)
    assert (destination/"old.txt").read_text() == "preserve"


def test_recorded_new_code_must_equal_actual_executed_package(tmp_path,monkeypatch):
    monkeypatch.setattr(internal.dev,"build_development_plan",lambda root:{"code":{},"policy":{}})
    executed=Path(internal.__file__).resolve().parents[1]
    for relative in internal.CODE:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((executed/relative).read_bytes())
    assert all(p in internal._software(tmp_path)["code"] for p in internal.CODE)
    (tmp_path/internal.CODE[0]).write_text("# different implementation")
    with pytest.raises(internal.InternalValidationError,match="executed_code"):
        internal._software(tmp_path)


def test_once_dispatch_charges_before_work_and_missing_attempt_cannot_refit():
    progress=state();internal._dispatch(progress, "job1", False, 2)
    assert progress["fit_attempts_dispatched"] == 1 and progress["dispatched_keys"] == ["job1"]
    internal._dispatch(progress, "job1", True, 2)
    assert progress["fit_attempts_dispatched"] == 1
    with pytest.raises(internal.InternalValidationError, match="prior_dispatched"):
        internal._dispatch(progress, "job1", False, 2)


def test_unknown_existing_candidate_and_budget_stop_preserve_ledger():
    progress=state();before=deepcopy(progress)
    with pytest.raises(internal.InternalValidationError, match="unowned"):
        internal._dispatch(progress, "job1", True, 2)
    assert progress == before
    internal._dispatch(progress, "job1", False, 1)
    before=deepcopy(progress)
    with pytest.raises(internal.InternalValidationError, match="budget_exhausted"):
        internal._dispatch(progress, "job2", False, 1)
    assert progress == before


def test_uncertain_partial_fit_is_preserved_without_refitting_or_capability(tmp_path):
    target=tmp_path/"runs/internal_validation_once/candidates/uncertain";target.mkdir(parents=True)
    marker=target/"dispatch.json";marker.write_text("{}")
    with pytest.raises(internal.InternalValidationError, match="uncertain_internal_attempt"):
        internal._fit_once(tmp_path, target.relative_to(tmp_path).as_posix(), None, None, None, None,
                           {}, None, None, None, 1)
    assert marker.read_text() == "{}" and not (target/"fitted").exists()


@pytest.mark.parametrize("tamper", ["counter", "duplicate", "foreign_prediction", "score_without_prediction", "labels_partial", "score_without_barrier"])
def test_corrupted_or_unsafe_phase_ledger_rejected(tamper):
    progress=state();internal._dispatch(progress,"job1",False,2)
    if tamper == "counter": progress["fit_attempts_dispatched"]=0
    if tamper == "duplicate": progress["dispatched_keys"].append("job1")
    if tamper == "foreign_prediction": progress["predictions"]["other"]={}
    if tamper == "score_without_prediction": progress["scores"]["job1"]={}
    if tamper == "labels_partial": progress["labels_revealed"]=True
    if tamper == "score_without_barrier": progress["predictions"]["job1"]={};progress["scores"]["job1"]={}
    with pytest.raises(internal.InternalValidationError):
        internal._phase_gate(progress,{"job1","job2"})


def test_no_fits_after_label_barrier_even_for_new_key():
    progress=state();progress["labels_revealed"]=True
    with pytest.raises(internal.InternalValidationError, match="no_more_fits"):
        internal._dispatch(progress,"job2",False,100)


def complete_predictions():
    progress=state()
    for key in ("job1","job2"):
        internal._dispatch(progress,key,False,2)
        progress["predictions"][key]={"path":key+".json","sha256":"a"*64}
    return progress


def test_partial_prediction_set_cannot_release_global_labels(tmp_path, monkeypatch):
    progress=complete_predictions();progress["predictions"].pop("job2")
    monkeypatch.setattr(internal,"verify_prediction_receipt",lambda *a: pytest.fail("partial set should fail first"))
    with pytest.raises(internal.InternalValidationError, match="every_frozen_prediction"):
        internal._release_barrier(tmp_path,tmp_path,progress,{"job1","job2"},{"freeze_sha256":"b"*64})
    assert progress["labels_revealed"] is False and not (tmp_path/"all_predictions_committed.json").exists()


def test_interrupted_prediction_verification_cannot_release_any_labels(tmp_path, monkeypatch):
    progress=complete_predictions();calls=[]
    def verify(root,ref):
        calls.append(ref["path"])
        if ref["path"] == "job2.json":raise RuntimeError("uncommitted second prediction")
    monkeypatch.setattr(internal,"verify_prediction_receipt",verify)
    with pytest.raises(RuntimeError):
        internal._release_barrier(tmp_path,tmp_path,progress,{"job1","job2"},{"freeze_sha256":"b"*64})
    assert calls == ["job1.json","job2.json"]
    assert progress["labels_revealed"] is False and not (tmp_path/"progress.json").exists()


def test_global_prediction_barrier_is_durable_before_label_release(tmp_path, monkeypatch):
    progress=complete_predictions();trace=[]
    monkeypatch.setattr(internal,"verify_prediction_receipt",lambda root,ref: trace.append(ref["path"]))
    internal._release_barrier(tmp_path,tmp_path,progress,{"job1","job2"},{"freeze_sha256":"b"*64})
    saved=json.loads((tmp_path/"progress.json").read_text())
    assert trace == ["job1.json","job2.json"]
    assert saved["labels_revealed"] is True and saved["phase"] == "LABELS_AFTER_ALL_PREDICTIONS"
    assert json.loads((tmp_path/"all_predictions_committed.json").read_text())["labels_revealed_before_commit"] is False
    internal._phase_gate(saved,{"job1","job2"})


def score_fixture(tmp_path):
    ref=put(tmp_path/"receipt.json",{})
    logits=np.tile([[2.,0.],[0.,2.]],(5,1));np.save(tmp_path/"logits.npy",logits,allow_pickle=False)
    episode=SimpleNamespace(participant_id="S18",session_id="1",temporal_audit_sha256="c"*64,
        query_event_uids=tuple("query"+str(i) for i in range(10)))
    receipt={"prediction_role":"internal_holdout","purpose":"internal_development_check",
        "selected_configuration_sha256":"b"*64,"episode_plan_sha256":"c"*64,
        "query_event_uids":list(episode.query_event_uids)}
    return ref,episode,receipt


def test_prediction_receipt_is_verified_before_label_capability_and_metrics(tmp_path, monkeypatch):
    ref,episode,receipt=score_fixture(tmp_path);trace=[]
    def verify(*a):trace.append("verified_prediction");return receipt
    class Labels:
        def labels(self,**kwargs):
            assert trace == ["verified_prediction"]
            assert kwargs == {"purpose":"internal_development_check","prediction_receipt":ref}
            trace.append("labels");return torch.tensor([0,1]*5)
    monkeypatch.setattr(internal,"verify_prediction_receipt",verify)
    result=internal._score_internal(tmp_path,episode,ref,SimpleNamespace(evaluation_labels=Labels()),"b"*64)
    assert trace == ["verified_prediction","labels"] and result["ba"] == 1.
    assert 0 < result["balanced_bce"] < .2


@pytest.mark.parametrize("field,value", [("purpose","model_selection"),("prediction_role","meta_validation"),
    ("selected_configuration_sha256","other"),("episode_plan_sha256","other"),("query_event_uids",["other"])])
def test_wrong_prediction_cannot_reveal_internal_labels(tmp_path, monkeypatch, field, value):
    ref,episode,receipt=score_fixture(tmp_path);receipt[field]=value
    monkeypatch.setattr(internal,"verify_prediction_receipt",lambda *a:receipt)
    labels=SimpleNamespace(labels=lambda **k:pytest.fail("internal labels leaked"))
    with pytest.raises(internal.InternalValidationError,match="before_labels"):
        internal._score_internal(tmp_path,episode,ref,SimpleNamespace(evaluation_labels=labels),"b"*64)


@pytest.mark.parametrize("key,value", [("max_jobs",2),("max_download_bytes",1),("max_fit_attempts",0),
    ("max_jobs",True),("max_download_bytes",False),("max_wall_seconds",0),("max_gpu_seconds",1),
    ("max_calibration_seconds",float("nan")),("freeze_sha256","other")])
def test_one_check_budget_cannot_expand_scope_or_hide_resource_limits(tmp_path,key,value):
    budget={"stage":"internal_development_check","freeze_sha256":"a"*64,"max_jobs":1,"max_download_bytes":0,
        "max_fit_attempts":100,"max_wall_seconds":60,"max_gpu_seconds":60,"max_calibration_seconds":10}
    budget[key]=value;ref=put(tmp_path/"budget.json",budget)
    with pytest.raises(internal.InternalValidationError,match="budget_required"):
        internal._budget(tmp_path,ref,"a"*64)
