"""Learned-artifact SOFTWARE fixture tests; no real data, GPU or meta fitting.

Completed-run JSON/checkpoints below are explicitly fabricated test doubles to
exercise verification paths. Their orthogonal matrices are untrained synthetic
arrays, NEVER evidence of a completed scientific fit or 1000 actual updates.
"""
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from p3work.adapt import CalibrationPrefix, FrozenSourceModel
from p3work.contracts import CHANNELS, digest, file_digest, validate_signal_contract
from p3work import meta_artifacts as artifacts
from p3work.meta_training import META_CODE_FILES, MetaJob, search_plan
from p3work.selection import choose_checkpoint
from p3work.spatial_geometry import SpatialGeometry, initialize_dictionary, positive_qr
from p3work.training_state import capture_rng, save_checkpoint

SPLIT_SHA = "b" * 64
CONTRACT = {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
    "selected_profile": "short", "context_s": [0, 2], "crop_s": [.5, 1.5], "n_times": 128,
    "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": SPLIT_SHA,
        "raw_metadata_manifest_sha256": "a" * 64,
        "long": {"participant_mean_coverage": .5, "participant_mean_abs_class_exclusion_gap": 0.},
        "short": {"participant_mean_coverage": 1., "participant_mean_abs_class_exclusion_gap": 0.}}}
SIGNAL_SHA = validate_signal_contract(CONTRACT)
CODE_FILES = META_CODE_FILES


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_fixture(seed=18):
    torch.manual_seed(seed)
    return FrozenSourceModel(nn.Sequential(nn.Flatten(), nn.Linear(11 * 6, 3), nn.Tanh()), nn.Linear(3, 2), feature_dim=3)


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


@pytest.fixture
def factory(tmp_path):
    root = tmp_path
    repository = Path(__file__).resolve().parents[1]
    for relative in CODE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    source = source_fixture()
    geometry = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64) / 10)
    source_receipt, reference_receipt = root / "source-SOFTWARE_TEST_DOUBLE.json", root / "reference-SOFTWARE_TEST_DOUBLE.json"
    write(source_receipt, {"fixture_scope": "SOFTWARE_TEST_DOUBLE_NOT_A_REAL_SOURCE_MODEL"})
    write(reference_receipt, {"fixture_scope": "SOFTWARE_TEST_DOUBLE_NOT_A_REAL_SOURCE_REFERENCE"})

    def make(method="P", *, seed=20261018, phase="screening", dimension=2, radius=.15):
        job = MetaJob(method, dimension, radius, seed, phase)
        folder = root / "fixtures" / job.identity
        folder.mkdir(parents=True, exist_ok=True)
        history = [{"outer_update": update, "participant_mean_ba": .8 if update in {250, 300} else .6,
                    "participant_mean_bce": .5 if update in {250, 300} else .7}
                   for update in range(50, 1001, 50)]
        selected = choose_checkpoint(history)
        episode_plan = folder / "episode_plans.json"
        write(episode_plan, {"schema_version": "paper3-frozen-meta-episode-plans-v1",
                             "fixture_scope": "SOFTWARE_TEST_DOUBLE_NO_REAL_EPISODES_OR_QUERY_LABELS",
                             "input_contract_sha256": SIGNAL_SHA, "split_sha256": SPLIT_SHA,
                             "meta_train": [], "meta_validation": [], "final_or_internal_query_included": False})
        identity = {"job": asdict(job), "signal_contract_sha256": SIGNAL_SHA, "split_sha256": SPLIT_SHA,
                    "source_model_state_sha256": source.frozen_state_sha256, "geometry_sha256": geometry.sha256(),
                    "source_receipt_sha256": file_digest(source_receipt), "reference_receipt_sha256": file_digest(reference_receipt),
                    "stieger_manifest_sha256": "e" * 64,
                    "episode_plan_sha256": file_digest(episode_plan),
                    "code": {p: file_digest(root / p) for p in CODE_FILES},
                    "fixture_scope": "UNTRAINED_SYNTHETIC_SOFTWARE_TEST_ONLY"}
        if method in {"B10", "B11"}:
            generator = torch.Generator().manual_seed(seed)
            parameter = positive_qr(torch.randn(10, dimension, dtype=torch.float64, generator=generator))
        else:
            parameter = initialize_dictionary(dimension, seed + (100 if method == "B8" else 0))
        checkpoint = folder / "selected_fixture.pt"
        state = {"parameter": parameter, "outer_cursor": selected["outer_update"],
                 "validation": [r for r in history if r["outer_update"] <= selected["outer_update"]],
                 "fixture_scope": "NOT_A_REAL_TRAINING_CHECKPOINT"}
        save_checkpoint(checkpoint, identity=identity, state=state, cuda=False)
        final_checkpoint = folder / "outer-1000.pt"
        final_state = {"parameter": parameter, "outer_cursor": 1000, "validation": history,
            "fixture_scope": "SOFTWARE_TEST_DOUBLE_NO_ACTUAL_UPDATES",
            "update_receipts": [{"outer_update": i, "full_unroll": True, "inner_updates_actual": 40,
                                 "fixture_scope": "FABRICATED_SOFTWARE_LEDGER"} for i in range(1, 1001)]}
        save_checkpoint(final_checkpoint, identity=identity, state=final_state, cuda=False)
        receipt = {"status": "META_JOB_COMPLETE", "synthetic": False, "scientific_fit_completed": True,
                   "full_unroll": True, "source_state_unchanged": True, "outer_updates_completed": 1000,
                   "validation_checkpoints": 20, "job": asdict(job), "validation": history,
                   "selected_checkpoint": selected, "identity_sha256": digest(identity),
                   **{k: identity[k] for k in ("signal_contract_sha256", "split_sha256",
                      "source_model_state_sha256", "geometry_sha256", "source_receipt_sha256", "reference_receipt_sha256", "stieger_manifest_sha256")},
                   "selected_parameter_checkpoint": {"path": checkpoint.relative_to(root).as_posix(), "sha256": file_digest(checkpoint)},
                   "episode_plan": {"path": episode_plan.relative_to(root).as_posix(), "sha256": file_digest(episode_plan)},
                   "checkpoint": {"path": final_checkpoint.name, "relative_to_job_output": True,
                                  "sha256": file_digest(final_checkpoint), "outer_cursor": 1000},
                   "fixture_scope": "FABRICATED_SOFTWARE_TEST_DOUBLE_NOT_AN_EEG_META_RESULT"}
        path = folder / "receipt.json"
        write(path, receipt)
        return {"root": root, "source": source, "geometry": geometry, "job": job, "parameter": parameter,
                "receipt": receipt, "receipt_path": path, "checkpoint": checkpoint, "identity": identity, "state": state,
                "final_checkpoint": final_checkpoint, "final_state": final_state,
                "source_descriptor": {"path": source_receipt.name, "sha256": file_digest(source_receipt)},
                "reference_descriptor": {"path": reference_receipt.name, "sha256": file_digest(reference_receipt)}}
    return make


def load(f, **kwargs):
    options = {"expected_method": f["job"].method, "expected_seed": f["job"].seed,
               "input_contract_sha256": SIGNAL_SHA, "split_sha256": SPLIT_SHA,
               "source": f["source"], "geometry": f["geometry"]}
    options.update(kwargs)
    return artifacts.load_learned_parameter(f["root"], f["receipt_path"].relative_to(f["root"]).as_posix(), **options)


def reseal(f):
    save_checkpoint(f["checkpoint"], identity=f["identity"], state=f["state"], cuda=False)
    f["receipt"]["identity_sha256"] = digest(f["identity"])
    f["receipt"]["selected_parameter_checkpoint"]["sha256"] = file_digest(f["checkpoint"])
    write(f["receipt_path"], f["receipt"])


@pytest.mark.parametrize("method", ["P", "B8", "B10", "B11"])
def test_software_fixture_full_sha_load_preserves_parameter_and_global_rng(factory, method):
    f = factory(method)
    random.random(); np.random.random(); torch.rand(4)
    before = capture_rng(cuda=False)
    learned = load(f)
    assert learned.selected_outer_update == 250  # BA/BCE tie resolves to earlier update.
    assert learned.job.method == method
    assert torch.equal(learned.matrix, f["parameter"])
    assert learned.matrix.requires_grad is False
    assert same(before, capture_rng(cuda=False))  # Inference must never restore training RNG.


@pytest.mark.parametrize("violation", ["engineering", "random", "synthetic", "999_updates", "19_checks", "incomplete_history", "wrong_selection"])
def test_engineering_random_or_incomplete_receipts_are_rejected(factory, violation):
    f = factory(phase="engineering" if violation == "engineering" else "screening")
    if violation == "random":
        f["receipt"]["status"] = "RANDOM_INITIALIZATION_ONLY"
    elif violation == "synthetic":
        f["receipt"]["synthetic"] = True
    elif violation == "999_updates":
        f["receipt"]["outer_updates_completed"] = 999
    elif violation == "19_checks":
        f["receipt"]["validation_checkpoints"] = 19
    elif violation == "incomplete_history":
        f["receipt"]["validation"].pop()
    elif violation == "wrong_selection":
        f["receipt"]["selected_checkpoint"] = f["receipt"]["validation"][-1]
    write(f["receipt_path"], f["receipt"])
    with pytest.raises(ValueError):
        load(f)


@pytest.mark.parametrize("violation", ["method", "seed", "source", "reference", "input", "split"])
def test_cross_method_seed_source_reference_input_or_split_is_rejected(factory, violation):
    f = factory()
    kwargs = {}
    if violation == "method":
        kwargs["expected_method"] = "B8"
    elif violation == "seed":
        kwargs["expected_seed"] = 20261019
    elif violation == "source":
        kwargs["source"] = source_fixture(19)
    elif violation == "reference":
        covariance = torch.eye(10, dtype=torch.float64) / 10
        covariance[0, 0] += .01
        covariance[1, 1] -= .01
        kwargs["geometry"] = SpatialGeometry.from_reference(covariance)
    elif violation == "input":
        kwargs["input_contract_sha256"] = "e" * 64
    else:
        kwargs["split_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        load(f, **kwargs)


@pytest.mark.parametrize("violation", ["checkpoint_bytes", "sidecar_sha", "sidecar_bytes", "identity", "code_changed", "missing_code"])
def test_selected_checkpoint_sidecar_identity_and_code_are_verified(factory, violation):
    f = factory()
    if violation == "checkpoint_bytes":
        with f["checkpoint"].open("ab") as stream:
            stream.write(b"SOFTWARE_TEST_CHANGED_BYTES")
    elif violation.startswith("sidecar"):
        path = f["checkpoint"].with_suffix(".pt.sha256.json")
        value = json.loads(path.read_text())
        value["sha256" if violation == "sidecar_sha" else "bytes"] = "0" * 64 if violation == "sidecar_sha" else -1
        write(path, value)
    elif violation == "identity":
        f["identity"]["split_sha256"] = "f" * 64
        reseal(f)
    elif violation == "code_changed":
        with (f["root"] / "p3work/adapt.py").open("ab") as stream:
            stream.write(b"\n# SOFTWARE_FIXTURE_CODE_CHANGE\n")
    else:
        f["identity"]["code"] = {}
        reseal(f)
    with pytest.raises(ValueError):
        load(f)


@pytest.mark.parametrize("violation", ["dtype", "shape", "orthogonality", "cursor"])
def test_selected_parameter_numeric_contract_is_checked_even_after_reseal(factory, violation):
    f = factory()
    if violation == "dtype":
        f["state"]["parameter"] = f["parameter"].float()
    elif violation == "shape":
        f["state"]["parameter"] = f["parameter"][:10]
    elif violation == "orthogonality":
        f["state"]["parameter"] = f["parameter"] * 2
    else:
        f["state"]["outer_cursor"] = 200
    reseal(f)
    with pytest.raises(ValueError):
        load(f)


@pytest.mark.parametrize("violation", ["delete", "change_bytes", "missing_descriptor", "checkpoint_identity"])
def test_pre_frozen_episode_plan_must_be_present_unchanged_and_checkpoint_bound(factory, violation):
    f = factory()
    descriptor = f["receipt"]["episode_plan"]
    path = f["root"] / descriptor["path"]
    if violation == "delete":
        path.unlink()
    elif violation == "change_bytes":
        with path.open("ab") as stream:
            stream.write(b"SOFTWARE_FIXTURE_EPISODE_PLAN_CHANGED")
    elif violation == "missing_descriptor":
        f["receipt"].pop("episode_plan")
        write(f["receipt_path"], f["receipt"])
    else:
        f["identity"]["episode_plan_sha256"] = "0" * 64
        reseal(f)
    with pytest.raises((ValueError, FileNotFoundError)):
        load(f)


def test_p_and_additive_require_independent_completed_method_artifacts(factory, monkeypatch):
    p, additive = factory("P"), factory("B8")
    with pytest.raises(ValueError):
        load(p, expected_method="B8")
    lp, la = load(p), load(additive)
    assert lp.receipt_path != la.receipt_path and not torch.equal(lp.matrix, la.matrix)
    torch.manual_seed(33)
    x = torch.randn(4, 11, 6)
    support = CalibrationPrefix(x, torch.tensor([0, 1, 0, 1]), tuple(f"s{i}" for i in range(4)),
                                tuple(f"s{i}" for i in range(20)), SIGNAL_SHA)
    calls = []
    def mathematical_kernel_spy(source, geometry, matrix, prefix, *, method, config):
        calls.append((method, matrix.clone(), prefix.role))
        return "SOFTWARE_DISPATCH_ONLY_NO_FIT"
    monkeypatch.setattr(artifacts, "fit_target", mathematical_kernel_spy)
    for learned in (lp, la):
        assert artifacts.fit_verified_meta_method(p["root"], learned, p["source"], p["geometry"], support) == "SOFTWARE_DISPATCH_ONLY_NO_FIT"
    assert [c[0] for c in calls] == ["P", "B8"]
    assert torch.equal(calls[0][1], p["parameter"]) and torch.equal(calls[1][1], additive["parameter"])


def test_raw_random_b10_tensor_cannot_enter_official_fit_wrapper(factory, monkeypatch):
    f = factory("B10")
    support = CalibrationPrefix(torch.ones(4, 11, 6), torch.tensor([0, 1, 0, 1]),
                                tuple(f"s{i}" for i in range(4)), tuple(f"s{i}" for i in range(20)), SIGNAL_SHA)
    def must_not_load(*args, **kwargs):
        pytest.fail("A raw random tensor reached official learned-artifact loading")
    monkeypatch.setattr(artifacts, "load_learned_parameter", must_not_load)
    with pytest.raises(TypeError):
        artifacts.fit_verified_meta_method(f["root"], f["parameter"], f["source"], f["geometry"], support)


@pytest.mark.parametrize("violation", ["missing_descriptor", "missing_file", "bytes", "sidecar", "later_metrics",
    "selected_metrics", "identity", "cursor", "ledger", "full_unroll", "matrix"])
def test_all_twenty_scores_and_thousand_updates_are_bound_to_final_checkpoint(factory, violation):
    f = factory()
    if violation == "missing_descriptor":
        f["receipt"].pop("checkpoint")
    elif violation == "missing_file":
        f["final_checkpoint"].unlink()
    elif violation == "bytes":
        with f["final_checkpoint"].open("ab") as stream:
            stream.write(b"CHANGED_SOFTWARE_TEST_DOUBLE")
    elif violation == "sidecar":
        f["final_checkpoint"].with_suffix(".pt.sha256.json").unlink()
    elif violation in {"later_metrics", "selected_metrics"}:
        index = -1 if violation == "later_metrics" else 4
        f["receipt"]["validation"][index]["participant_mean_bce"] += .01
    else:
        identity = dict(f["identity"])
        state = f["final_state"]
        if violation == "identity": identity["job"] = {**identity["job"], "radius": .35}
        elif violation == "cursor": state["outer_cursor"] = 999
        elif violation == "ledger": state["update_receipts"].pop()
        elif violation == "full_unroll": state["update_receipts"][-1]["full_unroll"] = False
        elif violation == "matrix": state["parameter"] = f["parameter"] * 2
        save_checkpoint(f["final_checkpoint"], identity=identity, state=state, cuda=False)
        f["receipt"]["checkpoint"]["sha256"] = file_digest(f["final_checkpoint"])
    write(f["receipt_path"], f["receipt"])
    with pytest.raises((ValueError, FileNotFoundError)):
        load(f)


def screening_fixture(factory, monkeypatch):
    """Nine fabricated completed runs + explicit doubles for real context loaders."""
    runs = [factory("P", dimension=row["dimension"], radius=row["radius"]) for row in search_plan("P")["jobs"]]
    f = runs[0]
    calls = []
    def source_double(root, path, contract):
        calls.append(("SOFTWARE_TEST_DOUBLE_SOURCE_CONTEXT", path))
        assert root == f["root"] and path == f["source_descriptor"]["path"] and contract == CONTRACT
        return f["source"], {"seed": 20261018, "pipeline_seed": 20261018}
    def reference_double(root, path, source_path, contract):
        calls.append(("SOFTWARE_TEST_DOUBLE_REFERENCE_CONTEXT", path))
        assert path == f["reference_descriptor"]["path"] and source_path == f["source_descriptor"]["path"]
        return SimpleNamespace(matrix=torch.eye(10, dtype=torch.float64) / 10), {}
    monkeypatch.setattr(artifacts, "load_verified_source", source_double)
    monkeypatch.setattr(artifacts, "load_verified_reference", reference_double)
    return runs, calls


def freeze(runs, **options):
    f = runs[0]
    kwargs = {"method": "P", "output": "selection-SOFTWARE_TEST_DOUBLE.json", "execute": False,
        "source_receipt": f["source_descriptor"], "reference_receipt": f["reference_descriptor"],
        "contract": CONTRACT, "split_sha256": SPLIT_SHA}
    kwargs.update(options)
    return artifacts.freeze_development_configuration(f["root"],
        [r["receipt_path"].relative_to(f["root"]).as_posix() for r in runs], **kwargs)


def test_nine_grid_freeze_reuses_full_learned_gate_and_records_screening_source_identity(factory, monkeypatch):
    runs, calls = screening_fixture(factory, monkeypatch)
    real_loader = artifacts.load_learned_parameter
    checked = []
    def spy(*args, **kwargs):
        checked.append(kwargs.copy())
        return real_loader(*args, **kwargs)
    monkeypatch.setattr(artifacts, "load_learned_parameter", spy)
    before = capture_rng(cuda=False)
    result = freeze(runs, execute=True)
    assert same(before, capture_rng(cuda=False))
    assert len(checked) == 9 and len(calls) == 2
    assert all(x["expected_seed"] == 20261018 and x["expected_method"] == "P" for x in checked)
    assert all(x["source"] is runs[0]["source"] for x in checked)
    assert result["screening_source_receipt"] == runs[0]["source_descriptor"]
    assert result["screening_reference_receipt"] == runs[0]["reference_descriptor"]
    assert result["dimension"] == 2 and result["radius"] == .05
    assert result["status"] == "META_CONFIGURATION_FROZEN"


@pytest.mark.parametrize("violation", ["arbitrary_pt", "missing_episode", "missing_code", "receipt_metrics",
    "wrong_source_descriptor", "wrong_reference_descriptor", "cross_source_receipt", "cross_reference_receipt"])
def test_nine_grid_status_and_arbitrary_file_sha_cannot_replace_actual_artifact_gates(factory, monkeypatch, violation):
    runs, _ = screening_fixture(factory, monkeypatch)
    f = runs[-1]
    options = {}
    if violation == "arbitrary_pt":
        f["checkpoint"].write_bytes(b"SOFTWARE_TEST_DOUBLE_NOT_A_TORCH_CHECKPOINT")
        f["receipt"]["selected_parameter_checkpoint"]["sha256"] = file_digest(f["checkpoint"])
    elif violation == "missing_episode":
        (f["root"] / f["receipt"]["episode_plan"]["path"]).unlink()
    elif violation == "missing_code":
        f["identity"]["code"] = {}
        reseal(f)
    elif violation == "receipt_metrics":
        f["receipt"]["validation"][-1]["participant_mean_bce"] += .01
    elif violation.startswith("wrong_"):
        key = "source_receipt" if violation == "wrong_source_descriptor" else "reference_receipt"
        options[key] = {**runs[0]["source_descriptor" if key == "source_receipt" else "reference_descriptor"], "sha256": "f" * 64}
    else:
        f["receipt"]["source_receipt_sha256" if violation == "cross_source_receipt" else "reference_receipt_sha256"] = "f" * 64
    write(f["receipt_path"], f["receipt"])
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze(runs, **options)


def test_screening_context_cannot_be_replaced_by_confirmation_seed_source(factory, monkeypatch):
    runs, _ = screening_fixture(factory, monkeypatch)
    monkeypatch.setattr(artifacts, "load_verified_source", lambda *args:
        (runs[0]["source"], {"seed": 20261019, "pipeline_seed": 20261019}))
    with pytest.raises(ValueError, match="screening-seed source"):
        freeze(runs)


def test_screening_selection_cannot_change_split_relative_to_window_audit(factory, monkeypatch):
    runs, calls = screening_fixture(factory, monkeypatch)
    with pytest.raises(ValueError, match="split differs"):
        freeze(runs, split_sha256="f" * 64)
    assert not calls
