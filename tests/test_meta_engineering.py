"""Explicit SOFTWARE_TEST_DOUBLE proofs: no real EEG, CUDA or scientific fit.

Flags and checkpoints are fabricated in isolated temporary directories solely
to verify this gate. The three real data loaders are explicitly replaced; the
untrained matrix and artificial timing ledger are never real engineering proof.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from p3work import meta_engineering as engineering
from p3work.adapt import FrozenSourceModel
from p3work.contracts import CHANNELS, digest, file_digest, validate_signal_contract
from p3work.meta_training import META_CODE_FILES, META_RUNTIME_KEYS, MetaJob
from p3work.selection import choose_checkpoint
from p3work.spatial_geometry import SpatialGeometry, initialize_dictionary, positive_qr
from p3work.training_state import capture_rng, save_checkpoint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


@pytest.fixture
def factory(tmp_path, monkeypatch):
    root = tmp_path
    repository = Path(__file__).resolve().parents[1]
    for relative in META_CODE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    split_sha = "b" * 64
    contract = {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
        "selected_profile": "short", "context_s": [0, 2], "crop_s": [.5, 1.5], "n_times": 128,
        "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": split_sha,
            "raw_metadata_manifest_sha256": "a" * 64,
            "long": {"participant_mean_coverage": .5, "participant_mean_abs_class_exclusion_gap": 0.},
            "short": {"participant_mean_coverage": 1., "participant_mean_abs_class_exclusion_gap": 0.}}}
    write(root / "contracts/input_contract.json", contract)
    signal_sha = validate_signal_contract(contract)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        source = FrozenSourceModel(nn.Sequential(nn.Flatten(), nn.Linear(11 * 6, 3)), nn.Linear(3, 2), feature_dim=3)
    reference = SimpleNamespace(matrix=torch.eye(10, dtype=torch.float64) / 10, events_sha256="c" * 64)
    geometry = SpatialGeometry.from_reference(reference.matrix)
    source_path, reference_path = root / "source-SOFTWARE_TEST_DOUBLE.json", root / "reference-SOFTWARE_TEST_DOUBLE.json"
    write(source_path, {"fixture": "SOFTWARE_TEST_DOUBLE_NOT_ACTUAL_SOURCE_FIT"})
    write(reference_path, {"fixture": "SOFTWARE_TEST_DOUBLE_NOT_ACTUAL_REFERENCE"})
    source_info = {"seed": 20261018, "pipeline_seed": 20261018, "architecture": "SOFTWARE_TEST_DOUBLE_ENCODER",
        "signal_contract_sha256": signal_sha, "source_cache_manifest_sha256": "d" * 64}
    reference_info = {"source_f0_receipt_sha256": file_digest(source_path), "source_cache_manifest": {"sha256": "d" * 64}}
    calls = []
    catalogs = {}
    for role in ("meta_train", "meta_validation"):
        data = {"fixture": "SOFTWARE_TEST_DOUBLE_NO_REAL_EPISODES", "role": role, "uids": [role + "/0"]}
        catalogs[role] = SimpleNamespace(plans=[SimpleNamespace(to_dict=lambda value=data: deepcopy(value))],
            manifest_sha256="e" * 64, input_contract_sha256=signal_sha, split_sha256=split_sha)
    def source_double(supplied_root, path, supplied_contract):
        calls.append(("SOFTWARE_TEST_DOUBLE_SOURCE_LOADER", path))
        assert supplied_root == root and path == source_path.name and supplied_contract == contract
        torch.rand(4)  # Real bridge allocation must not alter caller RNG.
        return source, source_info
    def reference_double(supplied_root, path, f0_path, supplied_contract):
        calls.append(("SOFTWARE_TEST_DOUBLE_REFERENCE_LOADER", path))
        assert path == reference_path.name and f0_path == source_path.name
        return reference, reference_info
    def catalog_double(supplied_root, cache, split, *, role, purpose):
        calls.append(("SOFTWARE_TEST_DOUBLE_CATALOG_LOADER", role, purpose))
        assert cache is split is None and role in catalogs
        assert purpose == ("meta_gradient" if role == "meta_train" else "model_selection")
        return catalogs[role]
    monkeypatch.setattr(engineering, "load_verified_source", source_double)
    monkeypatch.setattr(engineering, "load_verified_reference", reference_double)
    monkeypatch.setattr(engineering, "load_episode_catalog", catalog_double)

    def make(method="P", *, requested_dimension=2, engineering_dimension=None, confirmation=False):
        lowrank = method in {"B10", "B11"}
        dimension = engineering_dimension or (4 if lowrank else 8)
        engjob = MetaJob(method, dimension, .35, phase="engineering")
        job = MetaJob(method, requested_dimension, .15, seed=20261019 if confirmation else 20261018,
                      phase="confirmation" if confirmation else "screening")
        folder = root / "SOFTWARE_TEST_DOUBLE" / engjob.identity
        folder.mkdir(parents=True, exist_ok=True)
        inputs = {"signal_contract_sha256": signal_sha, "source_architecture": source_info["architecture"],
            "source_feature_dim": 3, "source_model_state_sha256": source.frozen_state_sha256,
            "source_training_event_uids_sha256": reference.events_sha256, "geometry_sha256": geometry.sha256(),
            "source_receipt_sha256": file_digest(source_path), "reference_receipt_sha256": file_digest(reference_path),
            "stieger_manifest_sha256": "e" * 64, "split_sha256": split_sha}
        episode_path = folder / "episode_plans.json"
        write(episode_path, {"schema_version": "paper3-frozen-meta-episode-plans-v1",
            "fixture_scope": "SOFTWARE_TEST_DOUBLE_NOT_REAL_DATA", "input_contract_sha256": signal_sha,
            "split_sha256": split_sha, "stieger_manifest_sha256": "e" * 64, "budget_acquired_trials": 20,
            "query_rule": "all_eligible_trials_in_future_complete_physical_runs_after_B20_boundary",
            "meta_train": [p.to_dict() for p in catalogs["meta_train"].plans],
            "meta_validation": [p.to_dict() for p in catalogs["meta_validation"].plans], "final_or_internal_query_included": False})
        identity = {**inputs, "job": asdict(engjob), "code": {p: file_digest(root / p) for p in META_CODE_FILES},
            "episode_plan_sha256": file_digest(episode_path), "python": "SOFTWARE_TEST_PYTHON", "torch": "SOFTWARE_TEST_TORCH",
            "cuda": "SOFTWARE_TEST_NO_REAL_CUDA", "device_name": "SOFTWARE_TEST_NO_GPU", "device_uuid": "SOFTWARE_TEST_UUID",
            "cudnn": 91002, "driver_versions": ["SOFTWARE_TEST_DRIVER"], "device_compute_capability": [8, 0],
            "device_total_memory_bytes": 16 * 1024**3,
            "float32_matmul_precision": "highest", "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8", "full_unroll": True, "autocast": False, "tf32": False}
        history = [{"outer_update": i, "participant_mean_ba": .5, "participant_mean_bce": .7} for i in (50, 100)]
        matrix = (positive_qr(torch.randn(10, dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(62)))
                  if lowrank else initialize_dictionary(dimension, 20261018))
        state = {"outer_cursor": 100, "parameter": matrix, "active_wall_seconds": 1., "validation": deepcopy(history),
            "fixture_scope": "SOFTWARE_TEST_DOUBLE_NO_ACTUAL_FIT_OR_GPU_TIMING",
            "update_receipts": [{"outer_update": i, "full_unroll": True, "inner_updates_actual": 40,
                "gpu_stream_elapsed_ms": 1.5, "end_to_end_wall_seconds": .002,
                "max_memory_allocated_bytes": 128 * 1024**2, "max_memory_reserved_bytes": 256 * 1024**2,
                "fixture_scope": "FABRICATED_SOFTWARE_TIMING_LEDGER"} for i in range(1, 101)]}
        checkpoint = folder / "outer-0100.pt"
        save_checkpoint(checkpoint, identity=identity, state=state, cuda=False)
        proof = {**inputs, "status": "META_ENGINEERING_COMPLETE", "synthetic": False, "training_started": True,
            "scientific_fit_completed": False, "source_state_unchanged": True, "full_unroll": True,
            "job": asdict(engjob), "outer_updates_completed": 100, "validation_checkpoints": 2,
            "validation": history, "selected_checkpoint": choose_checkpoint(history), "identity_sha256": digest(identity),
            "episode_plan": {"path": episode_path.relative_to(root).as_posix(), "sha256": file_digest(episode_path)},
            "source_receipt": {"path": source_path.name, "sha256": file_digest(source_path)},
            "reference_receipt": {"path": reference_path.name, "sha256": file_digest(reference_path)},
            "runtime_profile": {key: identity[key] for key in META_RUNTIME_KEYS},
            "checkpoint": {"path": checkpoint.name, "relative_to_job_output": True, "sha256": file_digest(checkpoint), "outer_cursor": 100},
            "inner_updates_actual": 4000, "active_wall_seconds": 1.1,
            "fixture_scope": "FABRICATED_SOFTWARE_TEST_DOUBLE_NOT_ENGINEERING_EVIDENCE"}
        path = folder / "receipt.json"; write(path, proof)
        return {"root": root, "job": job, "inputs": inputs, "proof": proof, "path": path, "state": state,
            "identity": identity, "checkpoint": checkpoint, "episode_path": episode_path, "source_info": source_info,
            "reference_info": reference_info, "calls": calls, "catalogs": catalogs}
    return make


def verify(f, **kwargs):
    return engineering.verify_meta_engineering(f["root"], {"path": f["path"].relative_to(f["root"]).as_posix(),
        "sha256": file_digest(f["path"])}, job=f["job"], inputs=f["inputs"], **kwargs)


def reseal(f):
    save_checkpoint(f["checkpoint"], identity=f["identity"], state=f["state"], cuda=False)
    f["proof"]["identity_sha256"] = digest(f["identity"])
    f["proof"]["checkpoint"]["sha256"] = file_digest(f["checkpoint"])
    write(f["path"], f["proof"])


@pytest.mark.parametrize("method", ["P", "P+PCT", "B8", "B10", "B11"])
def test_explicit_software_proof_verifies_without_cuda_initialization_or_rng_restore(factory, monkeypatch, method):
    f = factory(method)
    def cuda_forbidden(*args, **kwargs):
        raise AssertionError("The proof verifier initialized/queried CUDA or restored its RNG")
    monkeypatch.setattr(torch.cuda, "is_available", cuda_forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", cuda_forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", cuda_forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", cuda_forbidden)
    before = capture_rng(cuda=False)
    proof = verify(f, runtime_profile=f["proof"]["runtime_profile"])
    assert proof["fixture_scope"].startswith("FABRICATED_SOFTWARE")
    assert same(before, capture_rng(cuda=False))
    assert [row[0] for row in f["calls"]].count("SOFTWARE_TEST_DOUBLE_CATALOG_LOADER") == 2


def test_confirmation_may_use_different_source_weights_but_engineering_source_must_be_seed18(factory):
    f = factory(confirmation=True)
    f["inputs"] = {**f["inputs"], "source_model_state_sha256": "f" * 64,
        "source_receipt_sha256": "1" * 64, "reference_receipt_sha256": "2" * 64}
    assert verify(f)["job"]["seed"] == 20261018
    f["source_info"]["pipeline_seed"] = f["source_info"]["seed"] = 20261019
    with pytest.raises(ValueError, match="seed-18"):
        verify(f)


@pytest.mark.parametrize("violation", ["status", "synthetic", "scientific", "no_training", "source_changed", "full_unroll",
    "99_updates", "one_validation", "other_method", "input", "geometry", "catalog", "uid", "source_cache", "runtime"])
def test_engineering_identity_status_permission_and_actual_source_mismatches_reject(factory, violation):
    f = factory()
    if violation == "status": f["proof"]["status"] = "RUNNING"
    elif violation == "synthetic": f["proof"]["synthetic"] = True
    elif violation == "scientific": f["proof"]["scientific_fit_completed"] = True
    elif violation == "no_training": f["proof"]["training_started"] = False
    elif violation == "source_changed": f["proof"]["source_state_unchanged"] = False
    elif violation == "full_unroll": f["proof"]["full_unroll"] = False
    elif violation == "99_updates": f["proof"]["outer_updates_completed"] = 99
    elif violation == "one_validation": f["proof"]["validation_checkpoints"] = 1
    elif violation == "other_method": f["proof"]["job"]["method"] = "B8"
    elif violation == "input": f["inputs"]["signal_contract_sha256"] = "f" * 64
    elif violation == "geometry": f["inputs"]["geometry_sha256"] = "f" * 64
    elif violation == "catalog": f["catalogs"]["meta_train"].manifest_sha256 = "f" * 64
    elif violation == "uid": f["inputs"]["source_training_event_uids_sha256"] = "f" * 64
    elif violation == "source_cache": f["reference_info"]["source_cache_manifest"]["sha256"] = "f" * 64
    else: f["proof"]["runtime_profile"]["device_uuid"] = "OTHER_SOFTWARE_DEVICE"
    write(f["path"], f["proof"])
    with pytest.raises(ValueError): verify(f)


def test_engineering_dimension_must_cover_requested_configuration(factory):
    f = factory(requested_dimension=8, engineering_dimension=2)
    with pytest.raises(ValueError): verify(f)


@pytest.mark.parametrize("violation", ["missing_final", "final_bytes", "sidecar", "record_identity", "cursor", "ledger99",
    "ledger_order", "detached", "inner_count", "missing_gpu", "zero_gpu", "nan_gpu", "negative_wall", "matrix",
    "validation_history", "wrong_selection", "code_changed", "missing_code", "wrong_runtime"])
def test_actual_final_checkpoint_code_selection_and_positive_timing_ledger_required(factory, violation):
    f = factory()
    if violation == "missing_final": f["checkpoint"].unlink()
    elif violation == "final_bytes": f["checkpoint"].write_bytes(b"SOFTWARE_TEST_DOUBLE_NOT_TORCH")
    elif violation == "sidecar": f["checkpoint"].with_suffix(".pt.sha256.json").unlink()
    elif violation == "record_identity": f["identity"]["split_sha256"] = "f" * 64
    elif violation == "cursor": f["state"]["outer_cursor"] = 99
    elif violation == "ledger99": f["state"]["update_receipts"].pop()
    elif violation == "ledger_order": f["state"]["update_receipts"][-1]["outer_update"] = 1
    elif violation == "detached": f["state"]["update_receipts"][-1]["full_unroll"] = False
    elif violation == "inner_count": f["state"]["update_receipts"][-1]["inner_updates_actual"] = 41
    elif violation == "missing_gpu": f["state"]["update_receipts"][-1].pop("gpu_stream_elapsed_ms")
    elif violation == "zero_gpu": f["state"]["update_receipts"][-1]["gpu_stream_elapsed_ms"] = 0
    elif violation == "nan_gpu": f["state"]["update_receipts"][-1]["gpu_stream_elapsed_ms"] = float("nan")
    elif violation == "negative_wall": f["state"]["update_receipts"][-1]["end_to_end_wall_seconds"] = -1
    elif violation == "matrix": f["state"]["parameter"] *= 2
    elif violation == "validation_history": f["proof"]["validation"][-1]["participant_mean_bce"] = .8
    elif violation == "wrong_selection": f["proof"]["selected_checkpoint"] = f["proof"]["validation"][-1]
    elif violation == "code_changed": (f["root"] / "p3work/adapt.py").write_text("# SOFTWARE_TEST_DOUBLE_CHANGED_CODE")
    elif violation == "missing_code": f["identity"]["code"] = {}
    if violation not in {"missing_final", "final_bytes", "sidecar", "code_changed"}: reseal(f)
    else: write(f["path"], f["proof"])
    kwargs = {"runtime_profile": {**f["proof"]["runtime_profile"], "device_name": "OTHER_SOFTWARE_DEVICE"}} if violation == "wrong_runtime" else {}
    with pytest.raises((ValueError, FileNotFoundError)): verify(f, **kwargs)


@pytest.mark.parametrize("violation", ["delete", "change_bytes", "empty_plan", "internal_included", "split", "checkpoint_binding"])
def test_episode_plan_must_match_complete_live_authorized_catalogs(factory, violation):
    f = factory()
    if violation == "delete": f["episode_path"].unlink()
    elif violation == "change_bytes": f["episode_path"].write_text("SOFTWARE_TEST_DOUBLE_CHANGED_BYTES")
    elif violation == "checkpoint_binding": f["identity"]["episode_plan_sha256"] = "f" * 64; reseal(f)
    else:
        plan = json.loads(f["episode_path"].read_text())
        if violation == "empty_plan": plan["meta_train"] = []
        elif violation == "internal_included": plan["final_or_internal_query_included"] = True
        else: plan["split_sha256"] = "f" * 64
        write(f["episode_path"], plan)
        f["proof"]["episode_plan"]["sha256"] = file_digest(f["episode_path"])
        f["identity"]["episode_plan_sha256"] = file_digest(f["episode_path"])
        reseal(f)
    with pytest.raises((ValueError, FileNotFoundError)): verify(f)


@pytest.mark.parametrize("field,value", [("python", ""), ("cuda", None), ("cudnn", True), ("cudnn", "91002"),
    ("driver_versions", "SOFTWARE_TEST_DRIVER"), ("driver_versions", [42]), ("device_compute_capability", [8]),
    ("device_compute_capability", [True, 0]), ("device_compute_capability", [8, -1]),
    ("device_total_memory_bytes", 0), ("device_total_memory_bytes", True), ("device_total_memory_bytes", 1.5),
    ("autocast", True), ("tf32", True), ("float32_matmul_precision", "high"),
    ("deterministic_algorithms", False), ("cublas_workspace_config", ":bad")])
def test_writer_shaped_runtime_fields_have_strict_precision_driver_and_capability_contract(factory, field, value):
    f = factory()
    assert tuple(f["proof"]["runtime_profile"]) == META_RUNTIME_KEYS
    f["identity"][field] = value
    f["proof"]["runtime_profile"][field] = value
    reseal(f)
    with pytest.raises(ValueError): verify(f)


def test_writer_optional_driver_and_cudnn_none_are_preserved_without_cuda_queries(factory):
    f = factory()
    for key in ("cudnn", "driver_versions"):
        f["identity"][key] = f["proof"]["runtime_profile"][key] = None
    reseal(f)
    assert verify(f, runtime_profile=f["proof"]["runtime_profile"])["runtime_profile"]["cudnn"] is None


def test_stale_six_field_runtime_profile_cannot_mask_actual_writer_interface(factory):
    f = factory()
    legacy_keys = ("python", "torch", "cuda", "device_name", "device_uuid", "code")
    f["proof"]["runtime_profile"] = {k: f["identity"][k] for k in legacy_keys}
    write(f["path"], f["proof"])
    with pytest.raises(ValueError, match="runtime/device/code"):
        verify(f)


@pytest.mark.parametrize("field", ["cudnn", "driver_versions", "device_total_memory_bytes"])
def test_optional_runtime_none_must_still_be_explicitly_recorded(factory, field):
    f = factory()
    f["identity"].pop(field)
    f["proof"]["runtime_profile"][field] = None
    reseal(f)
    with pytest.raises(ValueError, match="writer-shaped"):
        verify(f)


@pytest.mark.parametrize("field,value", [("max_memory_allocated_bytes", None), ("max_memory_allocated_bytes", 0),
    ("max_memory_allocated_bytes", True), ("max_memory_allocated_bytes", 1.5),
    ("max_memory_reserved_bytes", None), ("max_memory_reserved_bytes", 0), ("max_memory_reserved_bytes", True),
    ("max_memory_reserved_bytes", 1.5), ("max_memory_reserved_bytes", 1),
    ("max_memory_reserved_bytes", 32 * 1024**3)])
def test_every_outer_update_requires_real_positive_peak_gpu_memory_within_device_capacity(factory, field, value):
    f = factory()
    row = f["state"]["update_receipts"][-1]
    if value is None:
        row.pop(field)
    else:
        row[field] = value
    reseal(f)
    with pytest.raises(ValueError, match="100 measured"):
        verify(f)
