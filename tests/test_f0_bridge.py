"""Bridge contract tests, using explicit SOFTWARE_TEST_DOUBLE metadata only.

Checkpoint tensors below are freshly initialized mathematical weights: there
is no optimizer step, EEG training, GPU execution or research completion claim.
"""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from p3work.f0_bridge import CompactEEGNet, load_verified_source
from p3work.adapt import FrozenSourceModel
from p3work.contracts import CHANNELS, SOURCE_IDS, digest, file_digest, validate_signal_contract
from p3work.source_training import SOURCE_CONFIG, build_training_plan, fit_identity
from p3work.spatial_geometry import joint_trial_rms
from torch import nn


def test_compact_baseline_bridge_keeps_original_double_logit_path():
    torch.manual_seed(4)
    model = CompactEEGNet(11, 128).eval()
    source = FrozenSourceModel(nn.Sequential(model.features, nn.Flatten(1)), model.classifier,
                               feature_dim=model.classifier.in_features, input_layout="N1CT")
    x = torch.randn(3, 11, 128)
    expected = model(joint_trial_rms(x).float())
    torch.testing.assert_close(source.source_logits(x), expected, rtol=0, atol=0)
    source.assert_unchanged()


def test_unfrozen_or_historical_synthetic_checkpoint_never_loads(tmp_path):
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps({"status": "synthetic_complete", "synthetic": True}))
    with pytest.raises(ValueError, match="not frozen"):
        load_verified_source(tmp_path, "receipt.json", {"status": "pending"})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


@pytest.fixture
def writer_shaped_software_double(tmp_path):
    """Use the actual writer's plan/identity shape, never its fit execution.

    Completion-looking fields are software test inputs beneath pytest's own
    temporary directory. Every metadata object declares the test-double scope.
    """
    scope = "SOFTWARE_TEST_DOUBLE_NOT_REAL_FIT_GPU_OR_RESEARCH_EVIDENCE"
    contract = {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
        "selected_profile": "short", "context_s": [0, 2], "crop_s": [.5, 1.5], "n_times": 128,
        "fixture_scope": scope, "window_audit_receipt": {"decision_pool_role": "meta_train",
        "split_sha256": "a" * 64, "raw_metadata_manifest_sha256": "b" * 64,
        "long": {"participant_mean_coverage": .7, "participant_mean_abs_class_exclusion_gap": .02},
        "short": {"participant_mean_coverage": .9, "participant_mean_abs_class_exclusion_gap": .02}}}
    signal_sha = validate_signal_contract(contract)
    cache_file = tmp_path / "software/source_cache.json"
    write(cache_file, {"fixture_scope": scope, "not_a_source_cache": True})
    cache_sha = file_digest(cache_file)
    rows = [{"dataset": dataset, "subject_id": "SOFTWARE_PERSON", "role": "source_train",
             "event_uid": f"SOFTWARE_TEST_DOUBLE/{dataset}/{label}",
             "processed_signal_sha256": digest([scope, dataset, label])}
            for dataset in sorted(SOURCE_IDS) for label in (0, 1)]
    project_root = Path(__file__).resolve().parents[1]
    cache = SimpleNamespace(root=project_root, rows=rows, labels=np.asarray([0, 1] * 4),
        input_contract_sha256=signal_sha, manifest_sha256=cache_sha,
        manifest_path=cache_file.relative_to(tmp_path).as_posix(), artifact_sha256={"software_cache": cache_sha})
    plan = build_training_plan(cache)
    seed = 20261018
    final_job = next(job for job in plan["jobs"] if job["phase"] == "final" and job["seed"] == seed)
    folds = [job for job in plan["jobs"] if job["phase"] == "lodo" and job["seed"] == seed]
    runtime = {"device": "cuda:0", "fixture_scope": scope}
    epochs = [3, 4, 7, 8]
    final_epochs = 5  # max(1, int(median([3,4,7,8])))
    identity = fit_identity(cache, final_job, {**copy.deepcopy(SOURCE_CONFIG), "effective_epochs": final_epochs}, runtime)
    identity["fixture_scope"] = scope
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(567)
        untrained = CompactEEGNet(11, 128).eval()
    checkpoint = {"identity": identity, "model": copy.deepcopy(untrained.state_dict()), "channels": 11,
        "n_times": 128, "config": copy.deepcopy(SOURCE_CONFIG), "selected_epoch": final_epochs,
        "synthetic": False, "fixture_scope": scope}
    final_folder = tmp_path / "software/source_jobs" / final_job["job_id"]
    checkpoint_file = final_folder / "selected.pt"
    uid_file = final_folder / "training_uids.json"
    uid = {"schema_version": "paper3-source-training-uids-v2", "role": "source_pretrain_train",
        "signal_contract_sha256": signal_sha, "event_uids": [r["event_uid"] for r in rows],
        "synthetic": False, "pipeline_seed": seed, "fixture_scope": scope}
    fold_data = []
    for job, selected in zip(folds, epochs):
        folder = tmp_path / "software/source_jobs" / job["job_id"]
        fold_identity = fit_identity(cache, job, {**copy.deepcopy(SOURCE_CONFIG), "effective_epochs": 80}, runtime)
        fold_identity["fixture_scope"] = scope
        completed = selected + SOURCE_CONFIG["patience"]
        summary = {"complete": True, "synthetic": False, "epochs_completed": completed,
            "selected_epoch": selected, "committed_optimizer_updates": completed * 64,
            "observed_optimizer_calls_total": completed * 64, "training_started": True,
            "stop_reason": "source_early_stopping", "fixture_scope": scope}
        status = {"status": "SOURCE_FIT_COMPLETE", "synthetic": False, "training_started": True,
                  "fixture_scope": scope}
        fold_data.append(SimpleNamespace(folder=folder, identity=fold_identity, summary=summary, status=status))
    lodo_file = tmp_path / "software/source_lodo_seed-20261018.json"
    lodo = {"status": "SOURCE_LODO_SELECTION_COMPLETE", "seed": seed, "pipeline_seed": seed,
        "rule": SOURCE_CONFIG["final_epochs"], "fold_selected_epochs": epochs, "final_epochs": final_epochs,
        "fold_receipts": [{"path": (f.folder / "status.json").relative_to(tmp_path).as_posix()} for f in fold_data],
        "validation_role": "source_development_selection_not_external_test", "input_contract_sha256": signal_sha,
        "cache_manifest": descriptor(tmp_path, cache_file), "fixture_scope": scope}
    receipt_file = tmp_path / "software/f0_receipt_seed-20261018.json"
    receipt = {"status": "SOURCE_F0_COMPLETE", "synthetic": False,
        "architecture": "legacy_compact_eegnet_v1_clinic11", "signal_contract_sha256": signal_sha,
        "source_cache_manifest_sha256": cache_sha, "cache_manifest": descriptor(tmp_path, cache_file),
        "training_dataset_ids": sorted(SOURCE_IDS), "source_lodo_selection_verified": True,
        "seed": seed, "pipeline_seed": seed, "configuration": copy.deepcopy(identity["configuration"]),
        "runtime": runtime, "fixture_scope": scope}

    def reseal():
        for fold in fold_data:
            sha = digest(fold.identity)
            fold.summary["identity_sha256"] = sha
            write(fold.folder / "identity.json", fold.identity)
            write(fold.folder / "fit_summary.json", fold.summary)
            manifest_file = fold.folder / "artifact_manifest.json"
            write(manifest_file, {"identity_sha256": sha, "fixture_scope": scope,
                "files": [descriptor(tmp_path, fold.folder / name) for name in ("identity.json", "fit_summary.json")]})
            fold.status.update(identity_sha256=sha, artifact_manifest=descriptor(tmp_path, manifest_file))
            write(fold.folder / "status.json", fold.status)
        for item in lodo["fold_receipts"]:
            item["sha256"] = file_digest(tmp_path / item["path"])
        write(lodo_file, lodo)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint["identity_sha256"] = digest(checkpoint.get("identity", {}))
        torch.save(checkpoint, checkpoint_file)
        write(uid_file, uid)
        receipt.update(checkpoint=descriptor(tmp_path, checkpoint_file), training_uid_manifest=descriptor(tmp_path, uid_file),
                       source_lodo_receipt=descriptor(tmp_path, lodo_file))
        write(receipt_file, receipt)
    reseal()
    return SimpleNamespace(root=tmp_path, contract=contract, checkpoint=checkpoint, identity=identity,
        lodo=lodo, folds=fold_data, uid=uid, receipt=receipt, untrained=untrained, reseal=reseal,
        receipt_path=receipt_file.relative_to(tmp_path).as_posix(), scope=scope)


def test_writer_shaped_math_checkpoint_loads_without_changing_rng_or_claiming_real_fit(writer_shaped_software_double):
    f = writer_shaped_software_double
    torch_before = torch.get_rng_state().clone()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    source, receipt = load_verified_source(f.root, f.receipt_path, f.contract)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert receipt["fixture_scope"] == f.scope
    x = torch.arange(11 * 128, dtype=torch.float32).reshape(1, 11, 128) / 1000
    with torch.no_grad():
        torch.testing.assert_close(source.source_logits(x), f.untrained(joint_trial_rms(x).float()), rtol=0, atol=0)
    source.assert_unchanged()
    assert all(not module.training for module in source.modules())
    assert all(not parameter.requires_grad for parameter in source.parameters())


@pytest.mark.parametrize("violation", ["checkpoint_synthetic", "identity_synthetic", "missing_identity", "seed", "cache", "final_epoch"])
def test_source_checkpoint_identity_violations_reject_even_resealed_bytes(writer_shaped_software_double, violation):
    f = writer_shaped_software_double
    if violation == "checkpoint_synthetic": f.checkpoint["synthetic"] = True
    elif violation == "identity_synthetic": f.identity["synthetic"] = True
    elif violation == "missing_identity": f.checkpoint["identity"] = {}
    elif violation == "seed": f.identity["job"]["seed"] += 1
    elif violation == "cache": f.identity["cache_manifest_sha256"] = "d" * 64
    else: f.checkpoint["selected_epoch"] += 1
    f.reseal()
    with pytest.raises(ValueError):
        load_verified_source(f.root, f.receipt_path, f.contract)


@pytest.mark.parametrize("violation", ["paused", "duplicate", "seed", "epoch", "cache", "source_namespace"])
def test_source_lodo_requires_four_matching_complete_source_folds(writer_shaped_software_double, violation):
    f = writer_shaped_software_double
    if violation == "paused": f.folds[0].status["status"] = "PAUSED_AT_COMMITTED_UPDATE"
    elif violation == "duplicate": f.lodo["fold_receipts"][1] = copy.deepcopy(f.lodo["fold_receipts"][0])
    elif violation == "seed": f.folds[0].identity["job"]["seed"] += 1
    elif violation == "epoch": f.lodo["fold_selected_epochs"][0] = 2  # Median stays 5: bind each fold summary too.
    elif violation == "cache": f.lodo["cache_manifest"]["sha256"] = "e" * 64
    else: f.folds[0].identity["job"]["heldout_dataset"] = "Stieger2021"
    f.reseal()
    with pytest.raises(ValueError):
        load_verified_source(f.root, f.receipt_path, f.contract)


@pytest.mark.parametrize("violation", ["synthetic", "role", "signal", "duplicate", "missing", "pipeline_seed"])
def test_training_uid_manifest_is_bound_to_actual_final_partition_shape(writer_shaped_software_double, violation):
    f = writer_shaped_software_double
    if violation == "synthetic": f.uid["synthetic"] = True
    elif violation == "role": f.uid["role"] = "source_validation"
    elif violation == "signal": f.uid["signal_contract_sha256"] = "f" * 64
    elif violation == "duplicate": f.uid["event_uids"][1] = f.uid["event_uids"][0]
    elif violation == "missing": f.uid["event_uids"].pop()
    else: f.uid["pipeline_seed"] += 1
    f.reseal()
    with pytest.raises(ValueError):
        load_verified_source(f.root, f.receipt_path, f.contract)
