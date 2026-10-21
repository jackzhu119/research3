"""Versioned source-only EEGNet training; no implicit execution or remote access.

This is the baseline f0, not the proposed spatial-update algorithm. The W4
source policy retains the immutable v1 configuration and its per-seed median
LODO epoch selection. Historical code/weights are never loaded as new fits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

from .contracts import SOURCE_IDS, digest, file_digest
from .f0_bridge import CompactEEGNet
from .recovery_budget import CLOCK_TOLERANCE_SECONDS, _boot_id
from .spatial_geometry import joint_trial_rms

ARCHITECTURE = "legacy_compact_eegnet_v1_clinic11"
SEED_PROFILES = {"development": (20261018, 20261019, 20261020),
                 "confirmation": (20261018, 20261019, 20261020, 20261021, 20261022)}
SOURCE_CONFIG = {"epochs": 80, "steps_per_epoch": 64, "batch_size": 64,
                 "optimizer": "Adam", "learning_rate": .001, "betas": [.9, .999],
                 "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False, "foreach": False,
                 "dropout": .25, "patience": 15, "min_delta": 1e-6,
                 "threads": 4, "checkpoint_every_updates": 16, "retain_epoch_checkpoints": True,
                 "validation": "dataset_then_participant_equal_mean_clipped_cross_entropy",
                 "final_epochs": "max(1,int(median(four_completed_same_seed_LODO_selected_epochs)))",
                 "sampling": "dataset_then_subject_then_observed_class_then_trial_with_replacement",
                 "precision": "float64_joint_RMS_then_float32_no_AMP_no_TF32"}
POLICY_FILES = ("protocol/EXPERIMENT_PROTOCOL.md", "protocol/METHOD_SPEC.md", "protocol/WORK_TASKS.md",
                "protocol/legacy_baseline_v1/configs/default.json", "protocol/legacy_baseline_v1/p3/neural.py",
                "protocol/legacy_baseline_v1/p3/cli.py")
CODE_FILES = ("p3work/source_training.py", "p3work/f0_bridge.py", "p3work/contracts.py",
              "p3work/window_amendment.py",
              "p3work/spatial_geometry.py", "tools/train_source.py", "tools/run_controlled_pilot.py")


class SourceTrainingError(ValueError):
    pass


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def immutable_json(path, value):
    """Publish once; retries preserve bytes and conflicting evidence is refused."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Atomic create without ever replacing an existing receipt.
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise SourceTrainingError("Immutable source receipt differs; preserve the original and use an explicit versioned output") from None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def checkpoint_sidecar(path):
    path = Path(path)
    return path.with_name(path.name + ".sha256.json")


def atomic_checkpoint(path, value):
    """Publish payload followed by its integrity record; a torn pair blocks resume."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        integrity = {"schema_version": "paper3-source-checkpoint-integrity-v1", "checkpoint_file": path.name,
                     "sha256": file_digest(temporary), "bytes": Path(temporary).stat().st_size}
        os.replace(temporary, path)
        atomic_json(checkpoint_sidecar(path), integrity)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verified_checkpoint(path, *, load=True):
    """Hash the complete saved payload before deserializing the same open file."""
    path = Path(path); sidecar = checkpoint_sidecar(path)
    try:
        if path.is_symlink() or sidecar.is_symlink():
            raise SourceTrainingError("Source checkpoint integrity paths must not be symlinks")
        record = json.loads(sidecar.read_text())
        if (set(record) != {"schema_version", "checkpoint_file", "sha256", "bytes"} or
                record["schema_version"] != "paper3-source-checkpoint-integrity-v1" or
                record["checkpoint_file"] != path.name or type(record["bytes"]) is not int or record["bytes"] <= 0 or
                not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or
                any(c not in "0123456789abcdef" for c in record["sha256"])):
            raise SourceTrainingError("Source checkpoint integrity sidecar is invalid")
        with path.open("rb") as stream:
            value = hashlib.sha256(); size = 0
            while chunk := stream.read(1024 * 1024):
                value.update(chunk); size += len(chunk)
            if size != record["bytes"] or value.hexdigest() != record["sha256"]:
                raise SourceTrainingError("Source checkpoint SHA256/bytes mismatch; torn pair cannot resume")
            if load:
                stream.seek(0)
                return torch.load(stream, map_location="cpu", weights_only=True)
            return record
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        raise SourceTrainingError("Source checkpoint integrity pair is missing or unreadable; it was not reconstructed") from None


def local(root, value, *, exists=True):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise SourceTrainingError("Explicit root-relative source artifact required")
    if "legacy_baseline_v1" in Path(value).parts or Path(value).suffix.lower() in {".npz", ".pkl", ".pickle"}:
        raise SourceTrainingError("Historical NPZ/pickle/code cannot be a new training artifact")
    path = (root / value).resolve(strict=exists)
    if not path.is_relative_to(root):
        raise SourceTrainingError("Source artifact escaped the independent project")
    return path


def artifact(root, path):
    path = Path(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def verify_artifact(root, item):
    if not isinstance(item, dict):
        raise SourceTrainingError("Missing source artifact descriptor")
    path = local(root, item.get("path"))
    if file_digest(path) != item.get("sha256"):
        raise SourceTrainingError("Source artifact SHA256 mismatch")
    return path


def _code_policy(root):
    # Reference text is hashed read-only; training imports only new p3work code.
    return {"policy": {name: file_digest(root / name) for name in POLICY_FILES},
            "code": {name: file_digest(root / name) for name in CODE_FILES}}


def _uids(cache, indices):
    return [cache.rows[int(i)]["event_uid"] for i in indices]


def hierarchical_weights(rows, labels):
    """Equal dataset, then participant, then observed class, then native trial."""
    if len(rows) != len(labels) or not len(rows):
        raise SourceTrainingError("Nonempty matching training rows and labels required")
    people, classes = defaultdict(set), defaultdict(set)
    counts = Counter()
    for row, label in zip(rows, labels):
        key = (row["dataset"], str(row["subject_id"]))
        people[key[0]].add(key[1]); classes[key].add(int(label)); counts[(*key, int(label))] += 1
    values = [1 / (len(people) * len(people[row["dataset"]]) *
                   len(classes[row["dataset"], str(row["subject_id"])]) *
                   counts[row["dataset"], str(row["subject_id"]), int(label)])
              for row, label in zip(rows, labels)]
    return torch.tensor(values, dtype=torch.float64)


def participant_validation_loss(rows, labels, probabilities):
    """Immutable-v1 source CE rule; deliberately not the meta-validation BA rule."""
    p = np.asarray(probabilities)
    labels = np.asarray(labels)
    if p.shape != (len(rows), 2) or not np.isfinite(p).all() or not len(rows):
        raise SourceTrainingError("Invalid source validation predictions")
    loss = -np.log(np.clip(p[np.arange(len(labels)), labels].astype(np.float64), 1e-12, 1))
    per_person = defaultdict(list)
    for row, value in zip(rows, loss):
        per_person[row["dataset"], str(row["subject_id"])].append(float(value))
    per_dataset = defaultdict(list)
    for (dataset, _), values in per_person.items():
        per_dataset[dataset].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in per_dataset.values()]))


def final_epoch_from_lodo(selected_epochs):
    if len(selected_epochs) != 4 or any(type(e) is not int or not 1 <= e <= 80 for e in selected_epochs):
        raise SourceTrainingError("Four actually selected source-LODO epochs in [1,80] required")
    return max(1, int(np.median(selected_epochs)))


def _pilot_local(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise SourceTrainingError("Explicit local engineering artifact path required")
    current = root
    for part in Path(value).parts:
        current = current / part
        if current.is_symlink():
            raise SourceTrainingError("Engineering artifacts must not use symlinks")
    try:
        return local(root, value)
    except OSError:
        raise SourceTrainingError("Engineering artifact is missing or unreadable") from None


def _pilot_replay_artifact(root, folder, item):
    """Require a full SHA/size and a local sibling, without GPU/RNG actions."""
    if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
            or type(item.get("bytes")) is not int or item["bytes"] <= 0):
        raise SourceTrainingError("Engineering prediction artifact needs an explicit full SHA256/bytes descriptor")
    path = _pilot_local(root, item["path"])
    if file_digest(path) != item.get("sha256"):
        raise SourceTrainingError("Engineering prediction artifact SHA256 differs")
    if (folder is not None and path.parent != folder) or path.stat().st_size != item["bytes"]:
        raise SourceTrainingError("Engineering prediction artifact size or pilot directory differs")
    return path


def _verify_engineering_prediction_replay(cache, receipt_file, receipt, identity, checkpoint_hashes):
    """Inspect actual saved outputs, not just the writer's exactness booleans.

    The same-device GPU logits are compared bit for bit across saved branches.
    This read-only verifier does not initialize CUDA, fit or restore any RNG.
    """
    from tools.run_controlled_pilot import engineering_prediction_selection, engineering_prediction_metrics, _same_tree
    if (len(cache.rows) != len(cache.labels) or {row.get("dataset") for row in cache.rows} != SOURCE_IDS
            or any(row.get("role") != "source_train" or not isinstance(row.get("event_uid"), str)
                   or not row["event_uid"] for row in cache.rows)
            or len({row["event_uid"] for row in cache.rows}) != len(cache.rows)
            or np.asarray(cache.labels).dtype.kind not in "iu" or not np.isin(cache.labels, [0, 1]).all()):
        raise SourceTrainingError("Engineering prediction replay requires actual unique legal four-source training UIDs and labels")
    try:
        _, event_uids, labels = engineering_prediction_selection(cache)
    except ValueError:
        raise SourceTrainingError("Engineering prediction source UID/label selection is invalid") from None
    source_batch_sha = digest({"event_uids": event_uids, "labels": labels})
    item = receipt.get("engineering_prediction_replay")
    proof_file = _pilot_replay_artifact(cache.root, receipt_file.parent, item)
    proof = json.loads(proof_file.read_text())
    if not isinstance(proof, dict):
        raise SourceTrainingError("Engineering prediction replay must be a structured proof")
    exact_keys = ("logits", "probabilities", "predictions", "labels", "metrics")
    selection_rule = "each sorted registered source: first min(16,n) event_uids sorted lexically; concatenate source order"
    common = {"scope": "ENGINEERING_SOURCE_PILOT_ONLY", "role": "source_train", "synthetic": False,
        "identity_sha256": digest(identity), "input_contract_sha256": cache.input_contract_sha256,
        "cache_manifest_sha256": cache.manifest_sha256, "source_batch_sha256": source_batch_sha,
        "event_uids": event_uids, "labels": labels}
    if (proof.get("schema_version") != "paper3-source-engineering-prediction-replay-v1"
            or any(not _same_tree(proof.get(key), value) for key, value in common.items())
            or proof.get("selection_rule") != selection_rule or proof.get("optimizer_updates_for_replay") != 0
            or type(proof.get("optimizer_updates_for_replay")) is not int
            or proof.get("all_exact") is not True
            or any(proof.get("comparisons", {}).get(key) is not True for key in exact_keys)):
        raise SourceTrainingError("Engineering prediction replay identity, legal fixed source batch or exactness proof is invalid")
    outputs = []
    prediction_descriptors = []
    for branch, checkpoint_name in (("continuous", "engineering_continuous100.pt"),
                                     ("resumed", "engineering_resumed100.pt")):
        saved_branch = proof.get(branch)
        if not isinstance(saved_branch, dict):
            raise SourceTrainingError("Engineering prediction replay requires both actual saved branches")
        checkpoint_item = saved_branch.get("checkpoint")
        checkpoint_path = _pilot_replay_artifact(cache.root, receipt_file.parent, checkpoint_item)
        if (checkpoint_path.name != checkpoint_name or checkpoint_item["sha256"] != checkpoint_hashes[checkpoint_name]):
            raise SourceTrainingError("Engineering prediction branch checkpoint differs from the actual 100-update state")
        prediction_item = saved_branch.get("predictions")
        prediction_file = _pilot_replay_artifact(cache.root, receipt_file.parent, prediction_item)
        output = json.loads(prediction_file.read_text())
        if not isinstance(output, dict):
            raise SourceTrainingError("Engineering predictions must be structured outputs")
        if (output.get("schema_version") != "paper3-source-engineering-predictions-v1"
                or any(not _same_tree(output.get(key), value) for key, value in common.items())
                or output.get("checkpoint") != checkpoint_item or output.get("model_update") != 100
                or type(output.get("model_update")) is not int or type(output.get("optimizer_updates")) is not int
                or output.get("prediction_rule") != "two_logits.argmax(-1); exact tie selects class 0"
                or output.get("eval_mode") is not True or output.get("optimizer_updates") != 0):
            raise SourceTrainingError("Engineering prediction output identity, source labels or no-update eval state is invalid")
        logits, probabilities = np.asarray(output.get("two_logits")), np.asarray(output.get("probabilities"))
        predictions = output.get("predictions")
        if (logits.shape != (len(labels), 2) or probabilities.shape != logits.shape
                or logits.dtype.kind not in "fiu" or probabilities.dtype.kind not in "fiu"
                or not np.isfinite(logits).all() or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not isinstance(predictions, list) or len(predictions) != len(labels)
                or any(type(value) is not int or value not in (0, 1) for value in predictions)):
            raise SourceTrainingError("Engineering prediction arrays are missing, nonfinite or malformed")
        stable = logits.astype(np.float64) - logits.astype(np.float64).max(axis=1, keepdims=True)
        expected_probabilities = np.exp(stable); expected_probabilities /= expected_probabilities.sum(axis=1, keepdims=True)
        # GPU float32 softmax need not equal CPU float64 softmax bitwise. This
        # only checks mathematical consistency; branch replay below is exact.
        tolerance = 4 * np.finfo(np.float32).eps
        if (not np.allclose(probabilities, expected_probabilities, rtol=tolerance, atol=tolerance)
                or predictions != logits.argmax(axis=1).tolist()):
            raise SourceTrainingError("Engineering logits, probabilities and predictions disagree")
        metrics = engineering_prediction_metrics(labels, probabilities, predictions=predictions)
        if not _same_tree(output.get("metrics"), metrics):
            raise SourceTrainingError("Engineering prediction metrics differ from the actual saved source outputs")
        outputs.append(output)
        prediction_descriptors.append(prediction_item)
    for key in ("two_logits", "probabilities", "predictions", "labels", "metrics"):
        if not _same_tree(outputs[0][key], outputs[1][key]):
            raise SourceTrainingError("Engineering continuous/resumed predictions or metrics actually differ")
    if prediction_descriptors[0]["path"] == prediction_descriptors[1]["path"]:
        raise SourceTrainingError("Engineering prediction branches must retain separate output artifacts")
    return {"engineering_prediction_replay": item, "prediction_artifacts": prediction_descriptors,
            "prediction_replay_exact": True, "metric_replay_exact": True,
            "fixed_source_batch_sha256": source_batch_sha, "fixed_source_prediction_trials": len(labels)}


def _verify_pilot_result_manifests(cache, receipt_file, receipt, identity, prediction_proof):
    """Recheck the finite backup-ready set; this is not an upload/GET receipt."""
    manifest_item = receipt.get("artifact_manifest")
    manifest_path = _pilot_replay_artifact(cache.root, receipt_file.parent, manifest_item)
    manifest = json.loads(manifest_path.read_text())
    paths = {receipt_file.with_name("identity.json").relative_to(cache.root).as_posix(),
             "runs/controlled_pilot_job_claim.json", "audit/gpu_resolved_dependencies.json",
             prediction_proof["engineering_prediction_replay"]["path"],
             *(item["path"] for item in prediction_proof["prediction_artifacts"]),
             *(receipt_file.with_name(name).relative_to(cache.root).as_posix() for name in receipt["checkpoint_files"])}
    files = manifest.get("files")
    if (manifest.get("schema_version") != "paper3-source-engineering-artifacts-v1"
            or manifest.get("scope") != "ENGINEERING_SOURCE_PILOT_ONLY" or manifest.get("scientific_result") is not False
            or manifest.get("pilot_status") != "ENGINEERING_PILOT_COMPLETE" or not isinstance(files, list)
            or any(not isinstance(row, dict) for row in files)
            or len(files) != len(paths) or {row.get("path") for row in files} != paths):
        raise SourceTrainingError("Engineering result artifact manifest must cover exactly every actual pilot output")
    for row in files:
        if row.get("role") != "result" or row.get("dataset") != "Paper3SourceEngineering":
            raise SourceTrainingError("Engineering artifact manifest role or dataset is invalid")
        _pilot_replay_artifact(cache.root, None, row)
    dependencies = identity.get("resolved_dependencies")
    dependency_path = _pilot_replay_artifact(cache.root, None, dependencies)
    if dependency_path != cache.root / "audit/gpu_resolved_dependencies.json":
        raise SourceTrainingError("Engineering pilot must bind its actual resolved dependency evidence")
    if receipt.get("resolved_dependencies") != dependencies:
        raise SourceTrainingError("Engineering pilot receipt and runtime dependency identities differ")
    resolved = json.loads(dependency_path.read_text())
    names = {}
    if not isinstance(resolved, list) or not resolved:
        raise SourceTrainingError("Engineering resolved dependency evidence is missing")
    for row in resolved:
        if (not isinstance(row, dict) or set(row) != {"name", "version"}
                or not all(isinstance(row[key], str) and row[key] for key in ("name", "version"))
                or re.fullmatch(r"[A-Za-z0-9_.-]+", row["name"]) is None
                or re.fullmatch(r"[A-Za-z0-9_.+!-]+", row["version"]) is None):
            raise SourceTrainingError("Engineering dependencies may contain only explicit package names and versions")
        name = re.sub(r"[-_.]+", "-", row["name"].lower())
        if name in names:
            raise SourceTrainingError("Engineering resolved dependencies contain duplicates")
        names[name] = row["version"]
    for line in (cache.root / "requirements-gpu.lock").read_text().splitlines():
        if not line.strip() or line.startswith(("#", "--")):
            continue
        name, version = line.strip().split("==")
        if names.get(re.sub(r"[-_.]+", "-", name.lower())) != version:
            raise SourceTrainingError("Engineering resolved dependencies omit or differ from the frozen GPU pins")
    if names.get("torch") != identity.get("torch") or names.get("numpy") != identity.get("numpy"):
        raise SourceTrainingError("Engineering dependency versions disagree with the pilot runtime")
    claim = json.loads((cache.root / "runs/controlled_pilot_job_claim.json").read_text())
    if (claim.get("scope") != "ENGINEERING_SOURCE_PILOT_ONLY" or claim.get("max_jobs") != 1
            or type(claim.get("max_jobs")) is not int
            or claim.get("output") != receipt_file.parent.relative_to(cache.root).as_posix()):
        raise SourceTrainingError("Engineering pilot one-job claim does not bind this output directory")
    backup_path = _pilot_local(cache.root, receipt.get("backup_manifest_path"))
    backup_item = {**artifact(cache.root, backup_path), "bytes": backup_path.stat().st_size}
    _pilot_replay_artifact(cache.root, receipt_file.parent, backup_item)
    backup = json.loads(backup_path.read_text())
    expected_files = files + [{**item, "role": "result", "dataset": "Paper3SourceEngineering"} for item in (
        manifest_item, {**artifact(cache.root, receipt_file), "bytes": receipt_file.stat().st_size})]
    backed = backup.get("files")
    if (backup.get("schema_version") != 1 or not isinstance(backed, list)
            or any(not isinstance(row, dict) for row in backed) or len(backed) != len(expected_files)
            or {row.get("path") for row in backed} != {row["path"] for row in expected_files}
            or {row["path"]: row for row in backed} != {row["path"]: row for row in expected_files}):
        raise SourceTrainingError("Engineering backup manifest is incomplete or differs from the actual outputs and final receipt")
    return {"artifact_manifest": manifest_item, "backup_manifest": backup_item, "backup_manifest_is_upload_receipt": False}


def verify_engineering_pilot(cache, receipt_path, *, runtime=None):
    """Verify the actual 200-update same-device engineering prerequisite."""
    receipt_file = _pilot_local(cache.root, receipt_path)
    receipt = json.loads(receipt_file.read_text())
    comparisons = ("model", "optimizer", "sampler_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "update", "identity_sha256")
    if (receipt.get("status") != "ENGINEERING_PILOT_COMPLETE" or receipt.get("scope") != "ENGINEERING_SOURCE_PILOT_ONLY" or
            receipt.get("training_started") is not True or receipt.get("actual_completed_updates") != 200 or
            receipt.get("optimizer_updates_dispatched") != 200 or receipt.get("seed") != 20261018 or receipt.get("max_jobs") != 1 or
            any(type(receipt.get(key)) is not int for key in ("actual_completed_updates", "optimizer_updates_dispatched", "seed", "max_jobs")) or
            receipt.get("exact_resume_verified") is not True or receipt.get("user_started_gpu_attested") is not True or
            any(receipt.get("resume_comparison", {}).get(key) is not True for key in comparisons)):
        raise SourceTrainingError("Actual completed 200-update engineering pilot with exact resume is required")
    identity_file = _pilot_local(cache.root, receipt_file.with_name("identity.json").relative_to(cache.root).as_posix())
    identity = json.loads(identity_file.read_text())
    required_code = {"runner_sha256": "tools/run_controlled_pilot.py", "bridge_sha256": "p3work/f0_bridge.py",
                     "geometry_sha256": "p3work/spatial_geometry.py", "contracts_sha256": "p3work/contracts.py"}
    if (identity.get("input_contract_sha256") != cache.input_contract_sha256 or
            identity.get("cache_manifest_sha256") != cache.manifest_sha256 or identity.get("device") != "cuda:0" or
            identity.get("tf32") is not False or identity.get("mixed_precision") is not False or
            identity.get("deterministic_algorithms") is not True or
            identity.get("seed") != 20261018 or identity.get("batch_size") != 64 or identity.get("optimizer") != "Adam" or
            identity.get("learning_rate") != .001 or identity.get("weight_decay") != 0 or identity.get("foreach") is not False or
            identity.get("dropout") != .25 or
            any(identity.get(key) != file_digest(cache.root / value) for key, value in required_code.items())):
        raise SourceTrainingError("Engineering pilot code/input/precision identity differs from formal training")
    prior_artifacts = identity.get("cache_artifacts")
    if (not isinstance(prior_artifacts, dict)
            or {k: v for k, v in prior_artifacts.items() if k != "status/tasks.json"}
                != {k: v for k, v in cache.artifact_sha256.items() if k != "status/tasks.json"}
            or identity.get("upstream_acceptance_sha256") != dict(getattr(cache, "upstream_acceptance_sha256", {}))):
        raise SourceTrainingError("Engineering pilot upstream metadata or acceptance identity differs from formal training")
    expected = {"engineering_initial.pt", "engineering_continuous100.pt", "engineering_resume50.pt", "engineering_resumed100.pt"}
    if set(receipt.get("checkpoint_files", {})) != expected:
        raise SourceTrainingError("Engineering pilot checkpoint evidence is incomplete")
    for name, expected_sha in receipt["checkpoint_files"].items():
        checkpoint_path = _pilot_local(cache.root, receipt_file.with_name(name).relative_to(cache.root).as_posix())
        if file_digest(checkpoint_path) != expected_sha:
            raise SourceTrainingError("Engineering pilot checkpoint SHA256 mismatch")
    from tools.run_controlled_pilot import _same_tree
    checkpoints = {name: torch.load(receipt_file.with_name(name), map_location="cpu", weights_only=True) for name in expected}
    for name, update in (("engineering_initial.pt", 0), ("engineering_continuous100.pt", 100),
                         ("engineering_resume50.pt", 50), ("engineering_resumed100.pt", 100)):
        saved = checkpoints[name]
        if (saved.get("scope") != "ENGINEERING_SOURCE_PILOT_ONLY" or saved.get("identity") != identity or
                saved.get("identity_sha256") != digest(identity) or saved.get("update") != update
                or not set(comparisons) <= set(saved)):
            raise SourceTrainingError("Engineering pilot checkpoint update/runtime identity is inconsistent")
    continuous, resumed = checkpoints["engineering_continuous100.pt"], checkpoints["engineering_resumed100.pt"]
    if any(not _same_tree(continuous.get(key), resumed.get(key)) for key in comparisons):
        raise SourceTrainingError("Engineering pilot continuous/resumed checkpoint states actually differ")
    prediction_proof = _verify_engineering_prediction_replay(cache, receipt_file, receipt, identity, receipt["checkpoint_files"])
    result_manifests = _verify_pilot_result_manifests(cache, receipt_file, receipt, identity, prediction_proof)
    if runtime is not None:
        mapping = {"device": "device", "device_name": "device_name", "device_uuid": "device_uuid",
                   "device_total_memory": "device_total_memory", "compute_capability": "compute_capability",
                   "torch": "torch", "numpy": "numpy", "python": "python",
                   "cuda_runtime": "cuda_runtime", "cudnn": "cudnn",
                   "driver_versions_local_nvidia_smi": "driver_versions"}
        if any(identity.get(key) != runtime.get(value) for key, value in mapping.items()):
            raise SourceTrainingError("Formal training runtime differs from the empirically resumed pilot device")
    return {"receipt": artifact(cache.root, receipt_file), "identity": artifact(cache.root, identity_file),
            "actual_completed_updates": 200, "exact_resume_verified": True, **prediction_proof, **result_manifests}


def build_training_plan(cache, *, profile="development"):
    if profile not in SEED_PROFILES:
        raise SourceTrainingError("Unknown registered seed profile")
    if {row.get("dataset") for row in cache.rows} != SOURCE_IDS or any(row.get("role") != "source_train" for row in cache.rows):
        raise SourceTrainingError("All and only four source_train datasets required")
    fingerprints = [row.get("processed_signal_sha256") for row in cache.rows]
    if any(not isinstance(value, str) or len(value) != 64 for value in fingerprints):
        raise SourceTrainingError("Native-trial processed signal hashes are required before source LODO")
    by_signal = defaultdict(set)
    for row, value in zip(cache.rows, fingerprints):
        by_signal[value].add(row["dataset"])
    if any(len(values) > 1 for values in by_signal.values()):
        raise SourceTrainingError("Identical original signals cross source LODO boundaries; resolve duplicate audit first")
    jobs = []
    for phase, heldout in [("lodo", source) for source in sorted(SOURCE_IDS)] + [("final", None)]:
        train = [i for i, row in enumerate(cache.rows) if heldout is None or row["dataset"] != heldout]
        val = [i for i, row in enumerate(cache.rows) if row["dataset"] == heldout] if heldout else []
        train_people = {(cache.rows[i]["dataset"], str(cache.rows[i]["subject_id"])) for i in train}
        val_people = {(cache.rows[i]["dataset"], str(cache.rows[i]["subject_id"])) for i in val}
        if train_people & val_people or set(np.asarray(cache.labels)[train]) != {0, 1} or (val and set(np.asarray(cache.labels)[val]) != {0, 1}):
            raise SourceTrainingError("Source subject namespace or explicit LR class separation failed")
        for seed in SEED_PROFILES[profile]:
            jobs.append({"job_id": f"{phase}/{heldout or 'ALL_SOURCE'}/eegnet/seed-{seed}",
                "phase": phase, "heldout_dataset": heldout, "seed": seed,
                "train_indices": train, "validation_indices": val,
                "train_uids_sha256": digest(_uids(cache, train)), "validation_uids_sha256": digest(_uids(cache, val)),
                "train_dataset_ids": sorted({cache.rows[i]["dataset"] for i in train}),
                "max_epochs": 80 if phase == "lodo" else None,
                "final_epochs_status": "PENDING_FOUR_SAME_SEED_LODO" if phase == "final" else "NOT_APPLICABLE"})
    classical = [{"job_id": f"{phase}/{source or 'ALL_SOURCE'}/{method}", "phase": phase,
                  "heldout_dataset": source, "method": method, "status": "PENDING_CLASSICAL_EXECUTOR",
                  "actual_fits_completed": 0}
                 for phase, source in [("lodo", d) for d in sorted(SOURCE_IDS)] + [("final", None)]
                 for method in ("source_csp_lda_fixed_v1", "source_ts_lr_fixed_v1")]
    identity = {"schema_version": "paper3-source-training-plan-v2", "profile": profile,
        "seeds": list(SEED_PROFILES[profile]), "input_contract_sha256": cache.input_contract_sha256,
        "cache_manifest": {"path": cache.manifest_path, "sha256": cache.manifest_sha256},
        "configuration": copy.deepcopy(SOURCE_CONFIG), "architecture": ARCHITECTURE,
        "source_policy_authority": "WORK_TASKS W4 retains immutable v1 baseline configuration/selection; explicit Adam defaults pinned to installed Torch",
        "policy_and_code": _code_policy(cache.root), "jobs": jobs, "classical_dependencies": classical,
        "planned_neural_fits": len(jobs), "planned_core_classical_fits": 10,
        "planned_core_total_fits": len(jobs) + 10, "max_neural_optimizer_updates": len(jobs) * 80 * 64,
        "additional_B4_B5_grid_source_fits": "separately counted; not included in core ten",
        "biological_cross_study_identity_independence": "not_verified; dataset/subject namespaces are disjoint",
        "status": "PLANNED_NOT_TRAINED", "training_started": False}
    return {**identity, "plan_sha256": digest(identity)}


def validate_training_plan(cache, plan):
    """Reject edited/subset plans before training or exporting any receipt."""
    if not isinstance(plan, dict) or plan != build_training_plan(cache, profile=plan.get("profile")):
        raise SourceTrainingError("The complete registered source plan/code/cache identity is required")


def classical_source_inputs(cache, job, signal_contract):
    """Dependency interface only; this function does not fit a classical model.

    Core v1 jobs use CSP m=2 / tangent shrink=.1. B4/B5 search-dependent source
    fits remain separate additional tasks rather than being hidden in ten fits.
    """
    from .methods.classical import SourceTrainingInputs
    heldout = job.get("heldout_dataset")
    if job.get("phase") not in {"lodo", "final"} or (job["phase"] == "lodo" and heldout not in SOURCE_IDS):
        raise SourceTrainingError("Registered classical source LODO/final job required")
    selected = np.asarray([i for i, row in enumerate(cache.rows) if heldout is None or row["dataset"] != heldout])
    return SourceTrainingInputs(cache.batch(selected), np.asarray(cache.labels)[selected].copy(),
        tuple(cache.rows[int(i)] for i in selected), signal_contract,
        cache.manifest_sha256, held_out_dataset=heldout, synthetic=False)


def capture_rng(sampler, *, device):
    n = np.random.get_state()
    return {"sampler": sampler.get_state(), "torch": torch.get_rng_state(),
            "python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []}


def restore_rng(state, sampler, *, device):
    sampler.set_state(state["sampler"]); torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _copy_model(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def predict_indices(model, cache, indices, *, device, batch_size=256):
    model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = np.asarray(indices[start:start + batch_size], dtype=np.int64)
            batch = joint_trial_rms(torch.from_numpy(cache.batch(selected))).float().to(device)
            values.append(model(batch).detach().cpu())
    logits = torch.cat(values) if values else torch.empty((0, 2))
    return logits.numpy(), torch.softmax(logits, dim=1).numpy()


def fit_identity(cache, job, config, runtime, *, synthetic_fixture=False):
    return {"schema_version": "paper3-source-fit-v2", "job": job, "configuration": config,
            "architecture": ARCHITECTURE, "input_contract_sha256": cache.input_contract_sha256,
            "cache_manifest_sha256": cache.manifest_sha256,
            "cache_artifacts": {k: v for k, v in cache.artifact_sha256.items() if k != "status/tasks.json"},
            "upstream_acceptance_sha256": dict(getattr(cache, "upstream_acceptance_sha256", {})),
            "policy_and_code": _code_policy(cache.root), "runtime": runtime,
            "synthetic": bool(synthetic_fixture), "device": runtime["device"]}


def run_fit(cache, job, folder, *, runtime, epochs, config=None,
            max_new_updates=None, synthetic_fixture=False, deadline=None, heartbeat=None):
    """Resumable kernel. CPU is restricted to explicit mathematical test fixtures.

    The public CLI never exposes synthetic_fixture/config overrides. A fixture
    status cannot be sealed as a scientific source fit or exported as f0.
    """
    config = copy.deepcopy(SOURCE_CONFIG if config is None else config)
    if not synthetic_fixture and (config != SOURCE_CONFIG or runtime.get("device") != "cuda:0"):
        raise SourceTrainingError("Real source fits require the fixed configuration and CUDA runtime")
    if not synthetic_fixture and job["phase"] == "lodo" and epochs != 80:
        raise SourceTrainingError("Source LODO uses the registered 80-epoch ceiling and source early stop")
    if not 1 <= epochs <= config["epochs"] or (max_new_updates is not None and max_new_updates < 1):
        raise SourceTrainingError("Invalid bounded source fit length")
    device = runtime["device"]
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    effective = {**config, "effective_epochs": epochs}
    identity = fit_identity(cache, job, effective, runtime, synthetic_fixture=synthetic_fixture)
    identity_sha = digest(identity)
    with (folder / ".fit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SourceTrainingError("The same source fit is already active") from None
        identity_file = folder / "identity.json"
        if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
            raise SourceTrainingError("Source checkpoint data/code/config/runtime identity changed")
        atomic_json(identity_file, identity)
        seed = job["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(config["threads"])
        model = CompactEEGNet(11, cache.n_times, config["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]),
            eps=config["eps"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"], foreach=False)
        sampler = torch.Generator(device="cpu").manual_seed(seed + 1000003)
        last = folder / "last_state.pt"
        progress_file = folder / "execution_progress.json"
        state = {"identity": identity, "identity_sha256": identity_sha, "completed_updates": 0,
                 "completed_epochs": 0, "update_in_epoch": 0, "epoch_loss_sum": 0.0,
                 "history": [], "best_loss": None, "best_epoch": 0, "bad_epochs": 0,
                 "best_model": None, "observed_optimizer_calls_total": 0}
        if last.exists() or checkpoint_sidecar(last).exists():
            saved = verified_checkpoint(last)
            if saved.get("identity_sha256") != identity_sha or saved.get("identity") != identity:
                raise SourceTrainingError("Checkpoint model/optimizer/RNG/cursor identity changed")
            # If SIGKILL occurs between the epoch commit and its archive, the
            # committed boundary checkpoint itself is the exact missing file.
            if config["retain_epoch_checkpoints"] and saved["completed_epochs"] > 0 and saved["update_in_epoch"] == 0:
                epoch_file = folder / f"epoch_{saved['completed_epochs']:04d}.pt"
                if not epoch_file.exists() and not checkpoint_sidecar(epoch_file).exists():
                    atomic_checkpoint(epoch_file, saved)
                else:
                    archived = verified_checkpoint(epoch_file)
                    from tools.run_controlled_pilot import _same_tree
                    if (archived.get("identity_sha256") != identity_sha or
                            archived.get("completed_updates") != saved["completed_updates"] or
                            archived.get("update_in_epoch") != 0 or
                            any(not _same_tree(archived.get(key), saved.get(key)) for key in ("model", "optimizer", "rng"))):
                        raise SourceTrainingError("Retained epoch checkpoint identity/cursor differs")
            model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"], sampler, device=device)
            state = {key: value for key, value in saved.items() if key not in {"model", "optimizer", "rng"}}
        else:
            if progress_file.exists() or (folder / "status.json").exists():
                raise SourceTrainingError("Initial source checkpoint is missing after observed work; it was not reconstructed")
            state.update(model=_copy_model(model), optimizer=optimizer.state_dict(), rng=capture_rng(sampler, device=device))
            atomic_checkpoint(last, state)
            state = {key: value for key, value in state.items() if key not in {"model", "optimizer", "rng"}}
        prior = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        if prior and prior.get("identity_sha256") != identity_sha:
            raise SourceTrainingError("Source optimizer-call journal belongs to a different fit identity")
        observed_total = max(state["observed_optimizer_calls_total"], prior.get("observed_optimizer_calls_total", 0))
        replayed_observed = max(0, prior.get("last_observed_trajectory_update", 0) - state["completed_updates"])
        state["observed_optimizer_calls_total"] = observed_total
        train = np.asarray(job["train_indices"], dtype=np.int64)
        val = np.asarray(job["validation_indices"], dtype=np.int64)
        weights = hierarchical_weights([cache.rows[i] for i in train], np.asarray(cache.labels)[train])
        target_updates = epochs * config["steps_per_epoch"]
        initial_update = state["completed_updates"]
        has_val = bool(len(val)); elapsed = time.monotonic(); new_updates = 0
        pause_reason = "explicit_update_pause"
        atomic_json(folder / "status.json", {"status": "RUNNING", "synthetic": synthetic_fixture,
            "identity_sha256": identity_sha, "training_started": state["completed_updates"] > 0})

        def commit():
            atomic_checkpoint(last, {**state, "model": _copy_model(model), "optimizer": optimizer.state_dict(),
                                      "rng": capture_rng(sampler, device=device)})

        while state["completed_updates"] < target_updates and not (has_val and state["bad_epochs"] >= config["patience"]):
            if deadline is not None and time.monotonic() >= deadline:
                pause_reason = "operational_wall_budget_pause"
                break
            if max_new_updates is not None and new_updates >= max_new_updates:
                break
            model.train()
            sampled = torch.multinomial(weights, config["batch_size"], replacement=True, generator=sampler).numpy()
            indices = train[sampled]
            batch = joint_trial_rms(torch.from_numpy(cache.batch(indices))).float().to(device)
            labels = torch.from_numpy(np.asarray(cache.labels)[indices].astype(np.int64, copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch), labels)
            if not bool(torch.isfinite(loss)):
                raise SourceTrainingError("Nonfinite source optimization loss")
            loss.backward(); optimizer.step()
            state["completed_updates"] += 1; state["update_in_epoch"] += 1
            state["epoch_loss_sum"] += float(loss.detach().cpu()); new_updates += 1
            state["observed_optimizer_calls_total"] += 1
            atomic_json(progress_file, {"identity_sha256": identity_sha,
                "last_observed_trajectory_update": state["completed_updates"],
                "observed_optimizer_calls_total": state["observed_optimizer_calls_total"],
                "replayed_observed_updates_this_invocation": replayed_observed,
                "unobserved_optimizer_call_before_hard_kill_remains_possible": True})
            if heartbeat is not None:
                heartbeat()
            if state["update_in_epoch"] == config["steps_per_epoch"]:
                epoch = state["completed_epochs"] + 1
                current = None
                if has_val:
                    _, probabilities = predict_indices(model, cache, val, device=device)
                    current = participant_validation_loss([cache.rows[i] for i in val], np.asarray(cache.labels)[val], probabilities)
                if not has_val or state["best_loss"] is None or current < state["best_loss"] - config["min_delta"]:
                    state.update(best_loss=current, best_epoch=epoch, bad_epochs=0, best_model=_copy_model(model))
                else:
                    state["bad_epochs"] += 1
                state["history"].append({"epoch": epoch, "optimizer_updates": state["completed_updates"],
                    "train_cross_entropy": state["epoch_loss_sum"] / config["steps_per_epoch"],
                    "source_validation_cross_entropy": current})
                state.update(completed_epochs=epoch, update_in_epoch=0, epoch_loss_sum=0.0)
                commit()
                if config["retain_epoch_checkpoints"]:
                    atomic_checkpoint(folder / f"epoch_{epoch:04d}.pt", verified_checkpoint(last))
            elif state["completed_updates"] % config["checkpoint_every_updates"] == 0:
                commit()
        commit()
        complete = state["completed_updates"] >= target_updates or (has_val and state["bad_epochs"] >= config["patience"])
        summary = {"complete": complete, "synthetic": synthetic_fixture, "identity_sha256": identity_sha,
            "epochs_completed": state["completed_epochs"], "selected_epoch": state["best_epoch"],
            "committed_optimizer_updates": state["completed_updates"], "new_updates_this_invocation": new_updates,
            "observed_optimizer_calls_total": state["observed_optimizer_calls_total"],
            "replayed_observed_updates_this_invocation": replayed_observed,
            "best_source_validation_cross_entropy": state["best_loss"], "history": state["history"],
            "wall_seconds_this_invocation": time.monotonic() - elapsed,
            "stop_reason": "source_early_stopping" if has_val and state["bad_epochs"] >= config["patience"] else "fixed_epoch_budget" if complete else pause_reason,
            "checkpoint_cursor": {"epoch": state["completed_epochs"], "update_in_epoch": state["update_in_epoch"]},
            "training_started": state["completed_updates"] > 0,
            "exact_continuation_scope": "same verified code/data/config/runtime identity only",
            "gpu_exact_resume_empirically_verified": False}
        atomic_json(folder / "fit_summary.json", summary)
        if not complete:
            atomic_json(folder / "status.json", {"status": "PAUSED_AT_COMMITTED_UPDATE", "synthetic": synthetic_fixture,
                                                 "identity_sha256": identity_sha, "training_started": summary["training_started"]})
            return summary
        if state["best_model"] is None:
            raise SourceTrainingError("No completed epoch exists for source checkpoint selection")
        selected = {"identity": identity, "identity_sha256": identity_sha, "model": state["best_model"],
                    "channels": 11, "n_times": cache.n_times, "config": config,
                    "selected_epoch": state["best_epoch"], "synthetic": synthetic_fixture}
        selected_path = folder / "selected.pt"
        if selected_path.exists() or checkpoint_sidecar(selected_path).exists():
            previous_selected = verified_checkpoint(selected_path)
            from tools.run_controlled_pilot import _same_tree
            if not _same_tree(previous_selected, selected):
                raise SourceTrainingError("Existing selected source checkpoint differs from the verified committed selection")
        else:
            atomic_checkpoint(selected_path, selected)
        model.load_state_dict(state["best_model"], strict=True)
        for role, indices in (("training", train), ("source_validation", val)):
            atomic_json(folder / f"{role}_uids.json", {
                "schema_version": "paper3-source-training-uids-v2" if role == "training" else "paper3-source-validation-uids-v2",
                "role": "source_pretrain_train" if role == "training" else "source_validation",
                "signal_contract_sha256": cache.input_contract_sha256, "event_uids": _uids(cache, indices),
                "synthetic": synthetic_fixture, "pipeline_seed": seed})
            if len(indices):
                logits, probabilities = predict_indices(model, cache, indices, device=device)
                np.save(folder / f"{role}_logits.npy", logits, allow_pickle=False)
                np.save(folder / f"{role}_probabilities.npy", probabilities, allow_pickle=False)
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix in {".json", ".pt", ".npy"}
                 and p.name not in {"status.json", "artifact_manifest.json"}]
        atomic_json(folder / "artifact_manifest.json", {"identity_sha256": identity_sha,
            "files": [artifact(cache.root, path) for path in sorted(files)]})
        atomic_json(folder / "status.json", {"status": "SYNTHETIC_FIXTURE_COMPLETE" if synthetic_fixture else "SOURCE_FIT_COMPLETE",
            "synthetic": synthetic_fixture, "identity_sha256": identity_sha, "training_started": True,
            "artifact_manifest": artifact(cache.root, folder / "artifact_manifest.json")})
        return summary


def verify_complete_fit(cache, job, folder):
    folder = Path(folder)
    status = json.loads((folder / "status.json").read_text())
    if status.get("status") != "SOURCE_FIT_COMPLETE" or status.get("synthetic") is not False:
        raise SourceTrainingError("A real completed source fit is required; fixtures/paused plans are pending")
    identity = json.loads((folder / "identity.json").read_text())
    if (identity.get("job") != job or identity.get("synthetic") is not False or
            identity.get("runtime", {}).get("device") != "cuda:0" or
            identity.get("input_contract_sha256") != cache.input_contract_sha256 or
            identity.get("cache_manifest_sha256") != cache.manifest_sha256 or
            identity.get("cache_artifacts") != {k: v for k, v in cache.artifact_sha256.items() if k != "status/tasks.json"} or
            identity.get("upstream_acceptance_sha256") != dict(getattr(cache, "upstream_acceptance_sha256", {})) or
            identity.get("policy_and_code") != _code_policy(cache.root) or
            status.get("identity_sha256") != digest(identity)):
        raise SourceTrainingError("Completed source fit identity is stale")
    manifest = json.loads(verify_artifact(cache.root, status["artifact_manifest"]).read_text())
    if manifest.get("identity_sha256") != digest(identity) or not isinstance(manifest.get("files"), list):
        raise SourceTrainingError("Source artifact manifest identity is incomplete")
    files = [verify_artifact(cache.root, item) for item in manifest["files"]]
    if len(set(files)) != len(files) or any(path.parent != folder.resolve() for path in files):
        raise SourceTrainingError("Source artifact manifest duplicates or crosses fit directories")
    summary = json.loads((folder / "fit_summary.json").read_text())
    if not summary.get("complete") or summary.get("synthetic") is not False or summary.get("selected_epoch", 0) < 1:
        raise SourceTrainingError("Actual source fit completion/selection evidence missing")
    config = identity["configuration"]
    if (set(config) != set(SOURCE_CONFIG) | {"effective_epochs"} or
            {key: config.get(key) for key in SOURCE_CONFIG} != SOURCE_CONFIG):
        raise SourceTrainingError("Completed source fit configuration differs from the frozen source policy")
    if job["phase"] == "lodo" and config.get("effective_epochs") != 80:
        raise SourceTrainingError("Completed source LODO did not use its registered epoch ceiling")
    last = verified_checkpoint(folder / "last_state.pt")
    selected = verified_checkpoint(folder / "selected.pt")
    from tools.run_controlled_pilot import _same_tree
    if (last.get("identity_sha256") != digest(identity) or selected.get("identity_sha256") != digest(identity) or
            last.get("identity") != identity or selected.get("identity") != identity or
            summary.get("identity_sha256") != digest(identity) or
            summary["committed_optimizer_updates"] != last.get("completed_updates") or
            last.get("completed_updates") != last.get("completed_epochs", 0) * config["steps_per_epoch"] or
            last.get("completed_epochs") != summary["epochs_completed"] or
            last.get("history") != summary.get("history") or last.get("update_in_epoch") != 0 or
            selected.get("selected_epoch") != summary["selected_epoch"] or
            last.get("best_epoch") != summary["selected_epoch"] or
            not _same_tree(selected.get("model"), last.get("best_model")) or
            selected.get("channels") != 11 or selected.get("n_times") != cache.n_times or
            selected.get("config") != SOURCE_CONFIG or selected.get("synthetic") is not False or
            not 1 <= summary["selected_epoch"] <= summary["epochs_completed"]):
        raise SourceTrainingError("Committed optimizer/cursor/selected-weight evidence is incomplete")
    terminal = (summary["epochs_completed"] == config["effective_epochs"] or
                (job["phase"] == "lodo" and last.get("bad_epochs", 0) >= config["patience"]))
    if not terminal or config.get("effective_epochs", 0) > 80:
        raise SourceTrainingError("Source fit did not reach its fixed budget or registered early stop")
    expected_files = {"identity.json", "last_state.pt", "selected.pt", "execution_progress.json",
                      "fit_summary.json", "training_uids.json", "source_validation_uids.json",
                      "training_logits.npy", "training_probabilities.npy"}
    expected_files.update(f"epoch_{epoch:04d}.pt" for epoch in range(1, summary["epochs_completed"] + 1))
    expected_files.update(name + ".sha256.json" for name in tuple(expected_files) if name.endswith(".pt"))
    if job["validation_indices"]:
        expected_files.update({"source_validation_logits.npy", "source_validation_probabilities.npy"})
    if {path.name for path in files} != expected_files:
        raise SourceTrainingError("Complete source fit artifact coverage is required")
    for path in files:
        if path.suffix == ".pt":
            verified_checkpoint(path, load=False)
    progress = json.loads((folder / "execution_progress.json").read_text())
    if (progress.get("identity_sha256") != digest(identity) or
            progress.get("last_observed_trajectory_update") != summary["committed_optimizer_updates"] or
            progress.get("observed_optimizer_calls_total") != summary.get("observed_optimizer_calls_total") or
            summary.get("observed_optimizer_calls_total", -1) < summary["committed_optimizer_updates"] or
            last.get("observed_optimizer_calls_total") != summary.get("observed_optimizer_calls_total")):
        raise SourceTrainingError("Actual source optimizer-call journal differs from its committed trajectory")
    uid = json.loads((folder / "training_uids.json").read_text())
    if (uid.get("schema_version") != "paper3-source-training-uids-v2" or
            uid.get("role") != "source_pretrain_train" or uid.get("synthetic") is not False or
            uid.get("signal_contract_sha256") != cache.input_contract_sha256 or
            uid.get("pipeline_seed") != job["seed"] or uid.get("event_uids") != _uids(cache, job["train_indices"]) or
            len(set(uid["event_uids"])) != len(uid["event_uids"])):
        raise SourceTrainingError("Actual f0 training UID evidence differs from its source partition")
    validation_uid = json.loads((folder / "source_validation_uids.json").read_text())
    if (validation_uid.get("schema_version") != "paper3-source-validation-uids-v2" or
            validation_uid.get("role") != "source_validation" or validation_uid.get("synthetic") is not False or
            validation_uid.get("signal_contract_sha256") != cache.input_contract_sha256 or
            validation_uid.get("pipeline_seed") != job["seed"] or
            validation_uid.get("event_uids") != _uids(cache, job["validation_indices"])):
        raise SourceTrainingError("Source validation UID evidence differs from the held-out dataset")
    for role, indices in (("training", job["train_indices"]), ("source_validation", job["validation_indices"])):
        if not indices:
            continue
        logits = np.load(folder / f"{role}_logits.npy", allow_pickle=False)
        probabilities = np.load(folder / f"{role}_probabilities.npy", allow_pickle=False)
        if (logits.shape != (len(indices), 2) or probabilities.shape != logits.shape or
                not np.isfinite(logits).all() or not np.isfinite(probabilities).all() or
                not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-6, atol=1e-6)):
            raise SourceTrainingError("Source fit prediction evidence has invalid coverage or values")
    return summary


def export_f0_receipts(cache, plan, output, *, pilot_evidence=None):
    """No receipt until every neural LODO and final fit in this profile is real."""
    output = Path(output)
    validate_training_plan(cache, plan)
    if not pilot_evidence or pilot_evidence.get("exact_resume_verified") is not True:
        raise SourceTrainingError("A real verified engineering pilot is required before f0 receipts")
    if verify_engineering_pilot(cache, pilot_evidence.get("receipt", {}).get("path")) != pilot_evidence:
        raise SourceTrainingError("Engineering pilot provenance changed before f0 export")
    summaries = {job["job_id"]: verify_complete_fit(cache, job, output / job["job_id"]) for job in plan["jobs"]}
    receipts = []
    for seed in plan["seeds"]:
        folds = [job for job in plan["jobs"] if job["phase"] == "lodo" and job["seed"] == seed]
        selected = [summaries[job["job_id"]]["selected_epoch"] for job in folds]
        epochs = final_epoch_from_lodo(selected)
        final_job = next(job for job in plan["jobs"] if job["phase"] == "final" and job["seed"] == seed)
        final = output / final_job["job_id"]
        final_identity = json.loads((final / "identity.json").read_text())
        if any(json.loads((output / job["job_id"] / "identity.json").read_text())["runtime"] != final_identity["runtime"]
               for job in folds):
            raise SourceTrainingError("Final source epoch selection cannot mix source-LODO runtime identities")
        if final_identity["configuration"]["effective_epochs"] != epochs:
            raise SourceTrainingError("Final source length did not follow completed same-seed LODO selection")
        lodo_path = output / f"source_lodo_seed-{seed}.json"
        immutable_json(lodo_path, {"status": "SOURCE_LODO_SELECTION_COMPLETE", "seed": seed, "pipeline_seed": seed,
            "rule": SOURCE_CONFIG["final_epochs"], "fold_selected_epochs": selected, "final_epochs": epochs,
            "fold_receipts": [artifact(cache.root, output / job["job_id"] / "status.json") for job in folds],
            "validation_role": "source_development_selection_not_external_test",
            "input_contract_sha256": cache.input_contract_sha256,
            "cache_manifest": {"path": cache.manifest_path, "sha256": cache.manifest_sha256}})
        receipt = {"status": "SOURCE_F0_COMPLETE", "synthetic": False,
            "architecture": ARCHITECTURE, "signal_contract_sha256": cache.input_contract_sha256,
            "source_cache_manifest_sha256": cache.manifest_sha256,
            "cache_manifest": {"path": cache.manifest_path, "sha256": cache.manifest_sha256},
            "training_dataset_ids": sorted(SOURCE_IDS), "source_lodo_selection_verified": True,
            "seed": seed, "pipeline_seed": seed, "checkpoint": artifact(cache.root, final / "selected.pt"),
            "training_uid_manifest": artifact(cache.root, final / "training_uids.json"),
            "source_lodo_receipt": artifact(cache.root, lodo_path), "configuration": final_identity["configuration"],
            "runtime": final_identity["runtime"], "precision": SOURCE_CONFIG["precision"],
            "classical_source_fits_status": "PENDING_SEPARATE_EXECUTOR",
            "overall_W4_complete": False, "engineering_pilot": pilot_evidence,
            "gpu_exact_resume_empirically_verified": bool(pilot_evidence and pilot_evidence.get("exact_resume_verified") is True),
            "gpu_exact_resume_verification_scope": "ENGINEERING_SOURCE_PILOT_ONLY",
            "formal_fit_exact_resume_comparison_performed": False}
        path = output / f"f0_receipt_seed-{seed}.json"
        immutable_json(path, receipt); receipts.append(artifact(cache.root, path))
    immutable_json(output / f"source_profile_{plan['profile']}.json", {
        "status": "SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING", "profile": plan["profile"],
        "plan_sha256": plan["plan_sha256"], "pipeline_seeds": plan["seeds"],
        "profile_neural_fits_verified_complete": len(summaries), "f0_receipts": receipts,
        "classical_core_fits_pending": 10, "overall_W4_complete": False})
    return receipts


def _wall_number(value, *, nonnegative=True):
    return (type(value) in {int, float} and math.isfinite(value) and (not nonnegative or value >= 0))


def _source_wall_clock():
    """Use the same boot identity and elapsed-clock rule as recovery_budget."""
    utc, mono, boot = time.time(), time.monotonic(), _boot_id()
    if not _wall_number(utc, nonnegative=False) or not _wall_number(mono) or not isinstance(boot, str) or not boot.strip():
        raise SourceTrainingError("Operational wall clock/boot identity is unavailable")
    return {"utc_seconds": utc, "monotonic_seconds": mono, "boot_id": boot}


def _observe_source_wall(path, state):
    """Publish a cumulative high-water mark before any clock block is raised."""
    clock_state = state["clock_state"]
    if clock_state["clock_block"] is not None:
        raise SourceTrainingError("Operational wall budget clock history is blocked; it was not reset")
    interval = clock_state["interval"]
    try:
        clock = _source_wall_clock()
    except Exception:
        clock_state["clock_block"] = {"reasons": ["clock_observation_unavailable"]}
        atomic_json(path, state)
        raise SourceTrainingError("Operational wall budget clock observation is unavailable; work is blocked") from None
    utc_delta = clock["utc_seconds"] - interval["start_utc_seconds"]
    mono_delta = clock["monotonic_seconds"] - interval["start_monotonic_seconds"]
    same_boot = clock["boot_id"] == interval["boot_id"]
    reasons = []
    if not same_boot:
        reasons.append("boot_changed_or_unknown")
    if utc_delta < 0 or clock["utc_seconds"] < interval["last_observed_utc_seconds"]:
        reasons.append("utc_clock_rolled_back")
    if same_boot and (mono_delta < 0 or clock["monotonic_seconds"] < interval["last_observed_monotonic_seconds"]):
        reasons.append("monotonic_clock_rolled_back")
    if same_boot and abs(utc_delta - mono_delta) > CLOCK_TOLERANCE_SECONDS:
        reasons.append("utc_monotonic_elapsed_disagreement_exceeds_5s")
    charged = max(state["consumed_active_seconds"], clock_state["wall_high_water_seconds"],
                  interval["base_wall_seconds"] + max(0.0, utc_delta, mono_delta if same_boot else 0.0))
    state["consumed_active_seconds"] = clock_state["wall_high_water_seconds"] = charged
    if reasons:
        clock_state["clock_block"] = {"reasons": reasons, "observed_clock": clock,
            "minimum_charged_seconds": charged, "resolution": "blocked_no_automatic_reset_or_clock_override"}
    else:
        interval["last_observed_utc_seconds"] = state["last_observed_unix"] = clock["utc_seconds"]
        interval["last_observed_monotonic_seconds"] = clock["monotonic_seconds"]
    atomic_json(path, state)
    if reasons:
        raise SourceTrainingError("Operational wall budget clock rollback/boot history is unknown; work is blocked")


def _validate_source_clock_state(path, state):
    active = state.get("active_invocation", False)
    if type(active) is not bool:
        raise SourceTrainingError("Operational wall budget active invocation flag is invalid")
    if "clock_state" not in state:
        state["clock_state"] = {"schema_version": "paper3-source-operational-clock-v1",
            "wall_high_water_seconds": state["consumed_active_seconds"], "interval": None,
            "clock_block": {"reasons": ["active_legacy_journal_missing_dual_clock_evidence"]} if active else None}
    clock = state["clock_state"]
    if (not isinstance(clock, dict) or clock.get("schema_version") != "paper3-source-operational-clock-v1" or
            not _wall_number(clock.get("wall_high_water_seconds")) or
            (clock.get("clock_block") is not None and not isinstance(clock["clock_block"], dict)) or
            "interval" not in clock or "clock_block" not in clock):
        raise SourceTrainingError("Operational wall budget dual-clock journal is invalid")
    state["consumed_active_seconds"] = clock["wall_high_water_seconds"] = max(
        state["consumed_active_seconds"], clock["wall_high_water_seconds"])
    if clock["clock_block"] is not None:
        atomic_json(path, state)
        raise SourceTrainingError("Operational wall budget clock history is blocked; missing evidence was not reconstructed")
    interval = clock["interval"]
    if active:
        if (not isinstance(interval, dict) or
                not _wall_number(interval.get("base_wall_seconds")) or interval["base_wall_seconds"] > state["consumed_active_seconds"] or
                not isinstance(interval.get("boot_id"), str) or not interval["boot_id"].strip() or
                any(not _wall_number(interval.get(key), nonnegative=False) for key in ("start_utc_seconds", "last_observed_utc_seconds")) or
                any(not _wall_number(interval.get(key)) for key in ("start_monotonic_seconds", "last_observed_monotonic_seconds")) or
                interval["last_observed_utc_seconds"] < interval["start_utc_seconds"] or
                interval["last_observed_monotonic_seconds"] < interval["start_monotonic_seconds"]):
            clock["clock_block"] = {"reasons": ["active_interval_missing_or_invalid_dual_clock_evidence"]}
            atomic_json(path, state)
            raise SourceTrainingError("Interrupted operational wall budget lacks valid dual-clock evidence; work is blocked")
    elif interval is not None:
        raise SourceTrainingError("Operational wall budget has an inconsistent open interval")
    state["active_invocation"] = active


def operational_budget(output, *, initial_seconds=86400, amend_seconds=None, amendment_note=None):
    """Operational limits never enter a scientific model/checkpoint identity."""
    path = Path(output) / "operational_budget.json"
    if type(initial_seconds) is not int or initial_seconds <= 0:
        raise SourceTrainingError("Positive integer initial wall budget required")
    try:
        state = json.loads(path.read_text()) if path.exists() else {"initial_seconds": initial_seconds,
            "authorized_total_seconds": initial_seconds, "consumed_active_seconds": 0.0, "revision": 1,
            "amendments": [], "scope": "operational_only_no_scientific_change"}
    except (OSError, json.JSONDecodeError):
        raise SourceTrainingError("Operational wall budget journal is unreadable; it was not reset") from None
    if (not isinstance(state, dict) or state.get("scope") != "operational_only_no_scientific_change" or
            type(state.get("initial_seconds")) is not int or state["initial_seconds"] <= 0 or
            type(state.get("authorized_total_seconds")) is not int or
            state["authorized_total_seconds"] < initial_seconds or
            not _wall_number(state.get("consumed_active_seconds")) or
            not isinstance(state.get("amendments"), list) or state.get("revision") != 1 + len(state["amendments"])):
        raise SourceTrainingError("Operational wall budget journal is invalid")
    authorized = initial_seconds
    for amendment in state["amendments"]:
        if (not isinstance(amendment, dict) or amendment.get("previous_seconds") != authorized or type(amendment.get("new_seconds")) is not int or
                amendment["new_seconds"] <= authorized or amendment.get("scientific_identity_changed") is not False or
                not isinstance(amendment.get("note"), str) or not amendment["note"].strip()):
            raise SourceTrainingError("Operational wall budget amendment history is invalid")
        authorized = amendment["new_seconds"]
    if authorized != state["authorized_total_seconds"]:
        raise SourceTrainingError("Operational wall budget total differs from its amendment history")
    if state["initial_seconds"] != initial_seconds:
        raise SourceTrainingError("Use an explicit operational budget amendment instead of changing the initial budget")
    if not isinstance(state.get("unclean_restart_charges", []), list):
        raise SourceTrainingError("Operational wall budget restart history is invalid")
    _validate_source_clock_state(path, state)
    if state.get("active_invocation") is True:
        before = state["consumed_active_seconds"]
        _observe_source_wall(path, state)
        state.setdefault("unclean_restart_charges", []).append({"charged_seconds": state["consumed_active_seconds"] - before,
            "policy": "max_same_boot_UTC_monotonic_elapsed_including_downtime"})
        state["active_invocation"] = False
        state["clock_state"]["interval"] = None
    # A rejected amendment must not discard a recovered high-water mark.
    atomic_json(path, state)
    if amend_seconds is not None:
        if (type(amend_seconds) is not int or amend_seconds <= state["authorized_total_seconds"] or
                not isinstance(amendment_note, str) or not amendment_note.strip()):
            raise SourceTrainingError("A larger operational-only budget needs a recorded amendment note")
        state["amendments"].append({"previous_seconds": state["authorized_total_seconds"],
            "new_seconds": amend_seconds, "note": amendment_note, "scientific_identity_changed": False})
        state["authorized_total_seconds"] = amend_seconds; state["revision"] += 1
    atomic_json(path, state)
    return path, state


def install_training_plan(cache, plan, output, *, runtime, extend_confirmation=False):
    """A registered 3-to-5 seed extension reuses only verified completed fits."""
    validate_training_plan(cache, plan)
    output = Path(output); path = output / "training_plan.json"
    if not path.exists():
        atomic_json(path, plan)
        return None
    previous = json.loads(path.read_text())
    if previous == plan:
        amendment = output / "confirmation_plan_extension.json"
        return artifact(cache.root, amendment) if amendment.exists() else None
    if (not extend_confirmation or previous.get("profile") != "development" or plan["profile"] != "confirmation"):
        raise SourceTrainingError("Different source plan requires the registered explicit confirmation extension")
    validate_training_plan(cache, previous)
    for job in previous["jobs"]:
        verify_complete_fit(cache, job, output / job["job_id"])
        old_identity = json.loads((output / job["job_id"] / "identity.json").read_text())
        if old_identity["runtime"] != runtime:
            raise SourceTrainingError("Confirmation extension cannot mix source runtime identities")
    new_by_id = {job["job_id"]: job for job in plan["jobs"]}
    if any(new_by_id.get(job["job_id"]) != job for job in previous["jobs"]):
        raise SourceTrainingError("Confirmation extension changed an existing source trajectory")
    archive = output / "training_plan_development_verified.json"
    atomic_json(archive, previous)
    amendment = output / "confirmation_plan_extension.json"
    atomic_json(amendment, {"status": "REGISTERED_CONFIRMATION_SEED_EXTENSION", "old_profile": "development",
        "new_profile": "confirmation", "previous_plan": artifact(cache.root, archive),
        "previous_plan_sha256": previous["plan_sha256"], "new_plan_sha256": plan["plan_sha256"],
        "verified_neural_fits_reused": len(previous["jobs"]),
        "additional_neural_fits_planned": len(plan["jobs"]) - len(previous["jobs"]),
        "existing_fit_policy_inputs_runtime_changed": False, "new_fits_started_by_this_operation": 0})
    atomic_json(path, plan)
    return artifact(cache.root, amendment)


def execute_plan(cache, plan, output, *, runtime, pilot_evidence,
                 max_new_updates=None, only_job=None, max_wall_seconds=86400,
                 amend_budget_seconds=None, operational_amendment_note=None, extend_confirmation=False):
    """Serial local CUDA fits. No remote boot, dataset fetch, or external query."""
    validate_training_plan(cache, plan)
    if max_new_updates is not None and (type(max_new_updates) is not int or max_new_updates < 1):
        raise SourceTrainingError("Positive integer bounded optimizer updates required")
    if runtime.get("device") != "cuda:0":
        raise SourceTrainingError("Formal source execution requires CUDA")
    if not pilot_evidence or pilot_evidence.get("exact_resume_verified") is not True:
        raise SourceTrainingError("Verified real engineering pilot prerequisite is pending")
    confirmed_pilot = verify_engineering_pilot(cache, pilot_evidence.get("receipt", {}).get("path"), runtime=runtime)
    if confirmed_pilot != pilot_evidence:
        raise SourceTrainingError("Engineering pilot provenance changed")
    output = Path(output)
    if not output.resolve().is_relative_to(cache.root):
        raise SourceTrainingError("Source training output must remain inside the independent project")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".source_training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SourceTrainingError("A source training job is already active") from None
        if only_job and only_job not in {j["job_id"] for j in plan["jobs"]}:
            raise SourceTrainingError("Requested job is outside the fixed source plan")
        extension = install_training_plan(cache, plan, output, runtime=runtime, extend_confirmation=extend_confirmation)
        plan_path = output / "training_plan.json"
        budget_path, wall = operational_budget(output, initial_seconds=max_wall_seconds,
            amend_seconds=amend_budget_seconds, amendment_note=operational_amendment_note)
        remaining = max(0.0, wall["authorized_total_seconds"] - wall["consumed_active_seconds"])
        try:
            clock = _source_wall_clock()
        except SourceTrainingError:
            wall["clock_state"]["clock_block"] = {"reasons": ["clock_identity_unavailable_before_work"]}
            atomic_json(budget_path, wall)
            raise
        deadline = clock["monotonic_seconds"] + remaining
        def heartbeat():
            _observe_source_wall(budget_path, wall)
        wall["clock_state"]["interval"] = {"base_wall_seconds": wall["consumed_active_seconds"],
            "start_utc_seconds": clock["utc_seconds"], "start_monotonic_seconds": clock["monotonic_seconds"],
            "last_observed_utc_seconds": clock["utc_seconds"], "last_observed_monotonic_seconds": clock["monotonic_seconds"],
            "boot_id": clock["boot_id"], "intent": "persisted_before_work"}
        wall["active_invocation"] = True
        wall["last_observed_unix"] = clock["utc_seconds"]
        atomic_json(budget_path, wall)
        completed, pending, budget = [], [], max_new_updates
        try:
            for job in plan["jobs"]:
                folder = output / job["job_id"]
                if (folder / "status.json").is_file() and json.loads((folder / "status.json").read_text()).get("status") == "SOURCE_FIT_COMPLETE":
                    verify_complete_fit(cache, job, folder)
                    if json.loads((folder / "identity.json").read_text())["runtime"] != runtime:
                        raise SourceTrainingError("Existing source fit belongs to a different runtime; use a recorded new branch")
                    completed.append(job["job_id"]); continue
                if (only_job and job["job_id"] != only_job) or budget == 0 or time.monotonic() >= deadline:
                    pending.append(job["job_id"]); continue
                epochs = SOURCE_CONFIG["epochs"]
                if job["phase"] == "final":
                    folds = [j for j in plan["jobs"] if j["phase"] == "lodo" and j["seed"] == job["seed"]]
                    try:
                        epochs = final_epoch_from_lodo([verify_complete_fit(cache, j, output / j["job_id"])["selected_epoch"] for j in folds])
                    except (SourceTrainingError, FileNotFoundError):
                        pending.append(job["job_id"]); continue
                try:
                    summary = run_fit(cache, job, folder, runtime=runtime, epochs=epochs, max_new_updates=budget,
                                      deadline=deadline, heartbeat=heartbeat)
                finally:
                    heartbeat()
                if budget is not None:
                    budget -= summary["new_updates_this_invocation"]
                (completed if summary["complete"] else pending).append(job["job_id"])
            result = {"status": "NEURAL_SOURCE_COMPLETE_CLASSICAL_PENDING" if not pending else "SOURCE_TRAINING_PENDING",
                      "plan_sha256": plan["plan_sha256"], "training_plan": artifact(cache.root, plan_path),
                      "confirmation_plan_extension": extension,
                      "actual_neural_fits_complete": len(completed), "pending_neural_jobs": pending,
                      "classical_core_fits_pending": 10, "overall_W4_complete": False,
                      "gpu_server_started": False, "external_data_accessed": False}
            if not pending:
                result["f0_receipts"] = export_f0_receipts(cache, plan, output, pilot_evidence=pilot_evidence)
                result["source_profile_receipt"] = artifact(cache.root, output / f"source_profile_{plan['profile']}.json")
        finally:
            # Ordinary errors/interrupts can close the operational timer. Only
            # a hard kill or unknown clock leaves the intent open and blocked.
            heartbeat()
            wall["active_invocation"] = False
            wall["clock_state"]["interval"] = None
            atomic_json(budget_path, wall)
        result["operational_budget"] = wall
        atomic_json(output / "source_training_progress.json", result)
        return result
