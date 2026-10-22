#!/usr/bin/env python3
"""Stage-gated, local-only SOURCE engineering pilot; default is a read-only dry run.

No API, SSH, RunPod, download, upload, validation-query, or external-data access.
Actual execution needs BOTH --execute and --user-started-gpu, completed W1/W2,
verified new cache artifacts, and CUDA. This produces no scientific f0 receipt.

Cache schema: W2.cache_manifest={path,sha256} binds a JSON manifest whose
schema_version is paper3-source-cache-v2, status SOURCE_CACHE_COMPLETE,
synthetic false, old_npz_reused false, role source_train, and input_contract_sha256
matches validate_signal_contract. Each shard has dataset, role, input_contract_sha256,
stage=pre_rms_car11_temporal_demeaned, X/y/events artifact {path,sha256}, and
adapter_code/processing_code artifact {path,sha256}. X/y must be .npy (no pickle),
events a JSON list, with event_uid, dataset, subject_id, source_record_id, original_trial_id, label,
role, input_contract_sha256, raw_sha256, adapter_sha256 and processing_sha256.
Raw checksums are provenance references, not a claim that raw files were reread
on the GPU host; upstream verified-native manifests remain required by W1/W2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.contracts import SOURCE_IDS, digest, file_digest, validate_signal_contract
from p3work.f0_bridge import CompactEEGNet
from p3work.spatial_geometry import joint_trial_rms

SEED = 20261018
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TOTAL_UPDATES = 200  # 100 continuous + 50 + restored 50, not 200 per branch.
MAX_WALL_SECONDS = 1800
WARMUP_UPDATES = 10  # Included within each branch's first 100; no extra updates.
PREDICTION_SELECTION_RULE = "each sorted registered source: first min(16,n) event_uids sorted lexically; concatenate source order"
PREDICTION_SCOPE = "ENGINEERING_SOURCE_PILOT_ONLY"
CHECKPOINT_NAMES = ("engineering_initial.pt", "engineering_continuous100.pt",
                    "engineering_resume50.pt", "engineering_resumed100.pt")
PREDICTION_NAMES = ("engineering_continuous100_predictions.json", "engineering_resumed100_predictions.json")


class PilotGateError(ValueError):
    pass


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PilotGateError("Concrete lowercase SHA256 required")
    return value


def _local(root, relative):
    if not isinstance(relative, str) or not relative:
        raise PilotGateError("Explicit root-relative artifact path required")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() in {".npz", ".pkl", ".pickle"}:
        raise PilotGateError("Legacy NPZ/pickle or escaping paths are forbidden")
    result = (root / path).resolve(strict=True)
    if not result.is_relative_to(root) or "legacy_baseline_v1" in result.parts:
        raise PilotGateError("Artifact escaped the new Paper3 cache root")
    return result


def _artifact(root, item, *, suffix=None):
    if not isinstance(item, dict):
        raise PilotGateError("Artifact path/SHA256 pair required")
    path = _local(root, item.get("path"))
    if suffix is not None and path.suffix.lower() != suffix:
        raise PilotGateError(f"Expected a {suffix} artifact")
    expected = _sha(item.get("sha256"))
    if not path.is_file() or file_digest(path) != expected:
        raise PilotGateError("Cache or code artifact SHA256 mismatch")
    return path


@dataclass
class VerifiedCache:
    root: Path
    n_times: int
    input_contract_sha256: str
    manifest_sha256: str
    manifest_path: str
    arrays: list[np.ndarray]
    labels: np.ndarray
    rows: list[dict]
    locations: list[tuple[int, int]]
    artifact_sha256: dict
    upstream_acceptance_sha256: dict = field(default_factory=dict)

    def batch(self, indices):
        # Gather ONLY authorized source training events, never whole target data.
        return np.stack([self.arrays[self.locations[i][0]][self.locations[i][1]]
                         for i in indices.tolist()])


def _verify_upstream_acceptance(cache):
    """Verify measured stage evidence; COMPLETE strings alone grant no access.

    These read-only verifiers use metadata and local cache bytes only. They do
    not restore raw EEG, contact R2, construct a model or initialize CUDA.
    """
    from p3work import (window_acceptance, window_amendment, source_cache_acceptance, contracts,
                        f0_bridge, spatial_geometry, source_training, runtime_clock)
    for module in (window_acceptance, window_amendment, source_cache_acceptance, contracts,
                   f0_bridge, spatial_geometry, source_training, runtime_clock):
        relative = "p3work/" + Path(module.__file__).name
        packaged = _local(cache.root, relative)
        if packaged.read_bytes() != Path(module.__file__).read_bytes():
            raise PilotGateError("Executed upstream verifier differs from packaged code")
        cache.artifact_sha256[relative] = file_digest(packaged)
    runner = _local(cache.root, "tools/run_controlled_pilot.py")
    if runner.read_bytes() != Path(__file__).read_bytes():
        raise PilotGateError("Executed upstream verifier differs from packaged code")
    cache.artifact_sha256["tools/run_controlled_pilot.py"] = file_digest(runner)
    window = window_acceptance.verify_frozen_window(cache.root)
    source = source_cache_acceptance.verify_source_cache_acceptance(cache)
    # Include the complete permission/provenance closure returned by the
    # measured window verifier. In v2.1 this also binds the one-candidate
    # approval, audit and retained original v2 failure evidence. Hash every
    # actual file instead of trusting a map or only the original descriptors.
    window_artifacts = window.get("artifact_files_sha256")
    if not isinstance(window_artifacts, dict) or not window_artifacts:
        raise PilotGateError("Complete exact-window artifact file map is required")
    for relative, expected in window_artifacts.items():
        path = _artifact(cache.root, {"path": relative, "sha256": expected})
        canonical = path.relative_to(cache.root).as_posix()
        if canonical != relative:
            raise PilotGateError("Exact-window artifact paths must be canonical")
        if canonical in cache.artifact_sha256 and cache.artifact_sha256[canonical] != expected:
            raise PilotGateError("Exact-window evidence changed during input verification")
        cache.artifact_sha256[canonical] = expected
    # The window API returns a proof digest; the source API returns a map of
    # actual file digests. Keep these distinct instead of treating a proof as
    # an artifact path or silently dropping its measured decision identity.
    for name in ("selection_plan", "split", "inventory", "window_input_session_manifest", "window_coverage"):
        descriptor = window[name]
        path = _artifact(cache.root, descriptor, suffix=".json")
        cache.artifact_sha256[path.relative_to(cache.root).as_posix()] = descriptor["sha256"]
    cache.artifact_sha256.update(source["artifact_sha256"])
    cache.upstream_acceptance_sha256.update(
        exact_window=_sha(window["artifact_sha256"]),
        full_source_cache=digest(source))


def verify_inputs(root, *, manifest_path=None) -> VerifiedCache:
    """Read-only gates and actual file hashes; no torch/CUDA fitting occurs here."""
    root = Path(root).resolve(strict=True)
    tasks = json.loads((root / "status/tasks.json").read_text())
    if tasks.get("W1", {}).get("status") != "COMPLETE":
        raise PilotGateError("W1 must be COMPLETE before a source pilot")
    if tasks.get("W2", {}).get("status") != "COMPLETE":
        raise PilotGateError("W2 new four-source caches must be COMPLETE")
    contract = json.loads((root / "contracts/input_contract.json").read_text())
    signal_sha = validate_signal_contract(contract)
    descriptor = tasks["W2"].get("cache_manifest")
    path = _artifact(root, descriptor, suffix=".json")
    if manifest_path is not None and _local(root, manifest_path) != path:
        raise PilotGateError("Requested cache manifest is not the W2 receipt-bound manifest")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != "paper3-source-cache-v2" or
            manifest.get("status") != "SOURCE_CACHE_COMPLETE" or
            manifest.get("synthetic") is not False or manifest.get("old_npz_reused") is not False or
            manifest.get("role") != "source_train"):
        raise PilotGateError("New, real, source_train cache provenance is required; no legacy NPZ")
    if manifest.get("input_contract_sha256") != signal_sha:
        raise PilotGateError("Cache input contract differs from the current frozen input")
    shards = manifest.get("shards", [])
    if not isinstance(shards, list) or not shards or {s.get("dataset") for s in shards} != SOURCE_IDS:
        raise PilotGateError("All and only four exact registered source dataset IDs are required")
    arrays, labels, rows, locations, seen, original_seen = [], [], [], [], set(), set()
    artifacts = {str(path.relative_to(root)): file_digest(path),
                 "status/tasks.json": file_digest(root / "status/tasks.json"),
                 "contracts/input_contract.json": file_digest(root / "contracts/input_contract.json")}
    for shard in shards:
        if (shard.get("role") != "source_train" or shard.get("input_contract_sha256") != signal_sha or
                shard.get("stage") != "pre_rms_car11_temporal_demeaned"):
            raise PilotGateError("Shard role, signal identity or pre-RMS stage mismatch")
        paths = {key: _artifact(root, shard.get(key), suffix=".npy" if key in {"X", "y"} else ".json" if key == "events" else ".py")
                 for key in ("X", "y", "events", "adapter_code", "processing_code")}
        if (paths["adapter_code"] != (root / "p3work/adapters/source_native.py").resolve() or
                paths["processing_code"] != (root / "p3work/preprocess.py").resolve()):
            raise PilotGateError("Cache must bind the actual new native adapter and preprocessing code")
        artifacts.update({str(p.relative_to(root)): file_digest(p) for p in paths.values()})
        x = np.load(paths["X"], mmap_mode="r", allow_pickle=False)
        y = np.load(paths["y"], mmap_mode="r", allow_pickle=False)
        events = json.loads(paths["events"].read_text())
        if x.ndim != 3 or x.shape[1:] != (11, contract["n_times"]) or x.dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise PilotGateError("X must contain new float32/64 N x 11 x frozen_time trials")
        if y.shape != (len(x),) or y.dtype.kind not in "iu" or not len(x) or not isinstance(events, list) or len(events) != len(x):
            raise PilotGateError("X/y/original-event count or label dtype mismatch")
        if not np.isin(y, [0, 1]).all():
            raise PilotGateError("Only explicit left/right 0/1 labels are allowed")
        for start in range(0, len(x), 64):
            chunk = np.asarray(x[start:start + 64], dtype=np.float64)
            if not np.isfinite(chunk).all():
                raise PilotGateError("Nonfinite source cache trial")
            rms = np.sqrt(np.mean(chunk ** 2, axis=(1, 2)))
            if not np.isfinite(rms).all() or np.any(rms == 0):
                raise PilotGateError("Zero/nonfinite source cache energy")
            if (np.any(np.abs(chunk.mean(1)).max(1) > rms * 5e-6) or
                    np.any(np.abs(chunk.mean(2)).max(1) > rms * 5e-6)):
                raise PilotGateError("Cache does not satisfy CAR11 and per-channel temporal demeaning")
        shard_index = len(arrays)
        for i, row in enumerate(events):
            if (not isinstance(row, dict) or row.get("role") != "source_train" or
                    row.get("dataset") != shard["dataset"] or row.get("input_contract_sha256") != signal_sha):
                raise PilotGateError("Only input-bound source_train original events can enter this runner")
            uid = row.get("event_uid")
            if not isinstance(uid, str) or not uid or uid in seen or not str(row.get("subject_id", "")) or not row.get("source_record_id") or row.get("original_trial_id") is None:
                raise PilotGateError("Missing/duplicated original-event or subject identity")
            for name in ("raw_sha256", "adapter_sha256", "processing_sha256"):
                _sha(row.get(name))
            if (row["adapter_sha256"] != shard["adapter_code"]["sha256"] or
                    row["processing_sha256"] != shard["processing_code"]["sha256"] or
                    type(row.get("label")) is not int or row["label"] != int(y[i])):
                raise PilotGateError("Original-event code identity or label disagrees with the cache")
            original = (row["dataset"], row["raw_sha256"], str(row["source_record_id"]), str(row["original_trial_id"]))
            if original in original_seen:
                raise PilotGateError("Repeated original trial under different event UIDs")
            seen.add(uid)
            original_seen.add(original)
            rows.append(row)
            locations.append((shard_index, i))
        arrays.append(x)
        labels.append(np.asarray(y, dtype=np.int64))
    cache = VerifiedCache(root, contract["n_times"], signal_sha, file_digest(path),
                          str(path.relative_to(root)), arrays, np.concatenate(labels), rows, locations, artifacts)
    _verify_upstream_acceptance(cache)
    return cache


def sampling_weights(cache):
    """Equal source dataset, then subject, then observed class, then trial."""
    subjects = defaultdict(set)
    classes = defaultdict(set)
    counts = Counter()
    for row, label in zip(cache.rows, cache.labels):
        dataset, subject, label = row["dataset"], str(row["subject_id"]), int(label)
        subjects[dataset].add(subject)
        classes[dataset, subject].add(label)
        counts[dataset, subject, label] += 1
    return torch.tensor([1 / (len(SOURCE_IDS) * len(subjects[row["dataset"]]) *
                                len(classes[row["dataset"], str(row["subject_id"])]) *
                                counts[row["dataset"], str(row["subject_id"]), int(label)])
                         for row, label in zip(cache.rows, cache.labels)], dtype=torch.float64)


def engineering_prediction_selection(cache):
    """Pure CPU selection from already verified source training events only."""
    y = np.asarray(cache.labels)
    if y.ndim != 1 or y.dtype.kind not in "iu" or len(cache.rows) != len(y):
        raise PilotGateError("Source engineering prediction event/label count mismatch")
    by_dataset, seen = defaultdict(list), set()
    for index, (row, label) in enumerate(zip(cache.rows, cache.labels)):
        uid = row.get("event_uid") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("dataset") not in SOURCE_IDS or row.get("role") != "source_train"
                or not isinstance(uid, str) or not uid or uid in seen
                or int(label) not in (0, 1) or type(row.get("label")) is not int or row.get("label") != int(label)):
            raise PilotGateError("Only distinct legal source_train prediction events are allowed")
        seen.add(uid)
        by_dataset[row["dataset"]].append((uid, index))
    if set(by_dataset) != SOURCE_IDS:
        raise PilotGateError("Fixed engineering predictions require all four sources")
    indices = np.asarray([index for dataset in sorted(SOURCE_IDS)
                          for _, index in sorted(by_dataset[dataset])[:16]], dtype=np.int64)
    return indices, [cache.rows[i]["event_uid"] for i in indices], [int(cache.labels[i]) for i in indices]


def engineering_prediction_metrics(labels, probabilities, *, predictions=None):
    """Pure engineering metrics; no fitting, query data, or CUDA initialization."""
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if (y.ndim != 1 or y.dtype.kind not in "iu" or not len(y) or not np.isin(y, [0, 1]).all()
            or p.shape != (len(y), 2) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)):
        raise PilotGateError("Invalid source engineering probability/label evidence")
    prediction = p.argmax(1) if predictions is None else np.asarray(predictions)
    if prediction.shape != y.shape or prediction.dtype.kind not in "iu" or not np.isin(prediction, [0, 1]).all():
        raise PilotGateError("Invalid engineering class predictions")
    counts = {str(c): int(np.sum(y == c)) for c in (0, 1)}
    recall = {str(c): float(np.mean(prediction[y == c] == c)) if counts[str(c)] else None for c in (0, 1)}
    floor = float(np.finfo(np.float64).tiny)
    return {"n_trials": len(y), "class_counts": counts, "class_recall": recall,
            "accuracy": float(np.mean(prediction == y)),
            "balanced_accuracy": float((recall["0"] + recall["1"]) / 2) if all(counts.values()) else None,
            "loss_cross_entropy": float(-np.log(np.maximum(p[np.arange(len(y)), y], floor)).mean()),
            "cross_entropy_probability_floor": floor}


def _atomic_json(path, record):
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("Controlled engineering pilot wall-time budget reached")


def _io_event(timings, operation, path, action, *, deadline=None):
    _check_deadline(deadline)
    start = time.monotonic()
    status = "complete"
    try:
        result = action()
        _check_deadline(deadline)
        return result
    except BaseException:
        status = "failed"
        raise
    finally:
        if timings is not None:
            timings.append({"operation": operation, "file": Path(path).name, "status": status,
                            "monotonic_seconds": max(0.0, time.monotonic() - start)})


def _atomic_checkpoint(path, record, *, io_timings=None, deadline=None):
    return _io_event(io_timings, "checkpoint_save_fsync_atomic_commit", path,
                     lambda: _write_checkpoint(path, record), deadline=deadline)


def _write_checkpoint(path, record):
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _reject_symlink_components(root, path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents) if p == root or root in p.parents):
        raise PilotGateError("Symlink components are forbidden in engineering evidence")


def _result_destination(root, output, *, require_directory=False):
    """Validate the original destination before any evidence file is written."""
    root, output = Path(root), Path(output)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise PilotGateError("Engineering result root must be canonical")
    try:
        relative = output.relative_to(root)
    except ValueError:
        raise PilotGateError("Engineering results must remain in the Paper3 runs directory") from None
    if len(relative.parts) < 2 or relative.parts[0] != "runs" or ".." in relative.parts:
        raise PilotGateError("Engineering results require a runs/ subdirectory")
    _reject_symlink_components(root, output)
    if output.resolve() != output or (require_directory and not output.is_dir()):
        raise PilotGateError("Engineering result directory must be a regular canonical directory")
    return output


def _descriptor(root, path):
    _reject_symlink_components(root, path)
    relative = Path(path).relative_to(root).as_posix()
    checked = _local(root, relative)
    if not checked.is_file() or checked.is_symlink():
        raise PilotGateError("Engineering artifact must be a regular local file")
    return {"path": relative, "sha256": file_digest(checked), "bytes": checked.stat().st_size}


def _load_checkpoint(path, *, expected_sha=None, io_timings=None, deadline=None):
    before = _io_event(io_timings, "checkpoint_sha256_before_load", path, lambda: file_digest(path), deadline=deadline)
    if expected_sha is not None and before != expected_sha:
        raise PilotGateError("Engineering checkpoint SHA256 differs from its committed artifact")
    saved = _io_event(io_timings, "checkpoint_load_weights_only", path,
                      lambda: torch.load(path, map_location="cpu", weights_only=True), deadline=deadline)
    after = _io_event(io_timings, "checkpoint_sha256_after_load", path, lambda: file_digest(path), deadline=deadline)
    if after != before:
        raise PilotGateError("Engineering checkpoint changed while loading")
    return saved


def _validate_checkpoint(saved, identity, update, *, path, io_timings=None, deadline=None):
    def validate():
        if (saved.get("scope") != PREDICTION_SCOPE or saved.get("identity") != identity
                or saved.get("identity_sha256") != digest(identity) or saved.get("update") != update):
            raise PilotGateError("Checkpoint update/code/data/config/device identity changed")
    _io_event(io_timings, "checkpoint_loaded_identity_validation", path, validate, deadline=deadline)


def _resolved_dependencies(root):
    """Validate the existing pip-list evidence without invoking pip or a shell."""
    if os.environ.get("PAPER3_HPC_RUNTIME_PROFILE") is not None:
        from hpc.runtime_profile import (RuntimeProfileError, selected_profile,
                                         validate_runtime_profile)
        try:
            profile = selected_profile(root)
            validate_runtime_profile(root, profile)
        except RuntimeProfileError as error:
            raise PilotGateError(str(error)) from None
        return _descriptor(root, root / "audit/gpu_resolved_dependencies.json")
    path = root / "audit/gpu_resolved_dependencies.json"
    before = _descriptor(root, path)  # Reject the original alias before opening it.
    value = json.loads(path.read_text())
    names = {}
    if not isinstance(value, list) or not value:
        raise PilotGateError("Actual complete resolved dependency list is required before the pilot")
    for row in value:
        if (not isinstance(row, dict) or set(row) != {"name", "version"}
                or not all(isinstance(row[k], str) and row[k] for k in ("name", "version"))
                or re.fullmatch(r"[A-Za-z0-9_.-]+", row["name"]) is None
                or re.fullmatch(r"[A-Za-z0-9_.+!-]+", row["version"]) is None):
            raise PilotGateError("Resolved dependency evidence must contain only package names and versions")
        name = re.sub(r"[-_.]+", "-", row["name"].lower())
        if name in names:
            raise PilotGateError("Duplicate resolved dependency package")
        names[name] = row["version"]
    if names.get("torch") != str(torch.__version__) or names.get("numpy") != np.__version__:
        raise PilotGateError("Resolved dependency evidence differs from the running Torch/NumPy")
    # A pip list that omits pinned packages is not complete deployment evidence.
    for line in (root / "requirements-gpu.lock").read_text().splitlines():
        if not line.strip() or line.startswith(("#", "--")):
            continue
        name, version = line.strip().split("==")
        if names.get(re.sub(r"[-_.]+", "-", name.lower())) != version:
            raise PilotGateError("Resolved dependencies do not satisfy every exact GPU profile pin")
    after = _descriptor(root, path)
    if before != after:
        raise PilotGateError("Resolved dependency evidence changed while being validated")
    return after


def _finalize_result_manifests(root, output, receipt):
    """Exact finite result allowlist; no discovery of data, env or configuration.

    artifact manifest excludes the receipt; the final receipt binds that
    manifest. The backup manifest then includes both, never itself.
    """
    output = _result_destination(root, output, require_directory=True)
    allowed = [root / "runs/controlled_pilot_job_claim.json",
               output / "identity.json", *(output / name for name in CHECKPOINT_NAMES),
               *(output / name for name in PREDICTION_NAMES),
               output / "engineering_prediction_replay.json"]
    if receipt.get("resolved_dependencies"):
        descriptor = _resolved_dependencies(root)
        if descriptor != receipt["resolved_dependencies"]:
            raise PilotGateError("Resolved dependencies changed during the engineering pilot")
        allowed.append(root / "audit/gpu_resolved_dependencies.json")
    if receipt.get("deployment_runtime_profile"):
        from hpc.runtime_profile import validate_runtime_profile
        if validate_runtime_profile(root, os.environ.get("PAPER3_HPC_RUNTIME_PROFILE")) != receipt["deployment_runtime_profile"]:
            raise PilotGateError("Engineering deployment runtime profile changed")
        allowed.append(root / "audit/hpc_runtime_profile.json")
    if receipt.get("status") == "ENGINEERING_PILOT_COMPLETE" and (
            not receipt.get("resolved_dependencies") or any(not path.is_file() for path in allowed)):
        raise PilotGateError("A completed engineering pilot requires every declared result artifact")
    files = []
    for path in allowed:
        if path.exists():
            item = _descriptor(root, path)
            files.append({**item, "role": "result", "dataset": "Paper3SourceEngineering"})
    artifact_path = output / "engineering_artifact_manifest.json"
    _atomic_json(artifact_path, {"schema_version": "paper3-source-engineering-artifacts-v1",
        "scope": PREDICTION_SCOPE, "scientific_result": False, "pilot_status": receipt["status"], "files": files})
    receipt["artifact_manifest"] = _descriptor(root, artifact_path)
    receipt["backup_manifest_path"] = (output / "engineering_backup_manifest.json").relative_to(root).as_posix()
    receipt["result_manifest_status"] = "LOCAL_MANIFESTS_COMPLETE_R2_NOT_ATTEMPTED"
    receipt["backup_performed"] = False
    receipt_path = output / "engineering_receipt.json"
    _atomic_json(receipt_path, receipt)
    backup_files = files + [{**_descriptor(root, path), "role": "result", "dataset": "Paper3SourceEngineering"}
                            for path in (artifact_path, receipt_path)]
    _atomic_json(output / "engineering_backup_manifest.json", {"schema_version": 1, "files": backup_files})


def _same_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_same_tree(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _checkpoint(model, optimizer, sampler, update, identity):
    numpy_rng = np.random.get_state()
    return {"scope": "ENGINEERING_SOURCE_PILOT_ONLY", "identity": identity,
            "identity_sha256": digest(identity), "update": update,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(0), "python_rng": random.getstate(),
            "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2], numpy_rng[3], numpy_rng[4]]}


def _restore(saved, model, optimizer, sampler, identity):
    if saved.get("scope") != "ENGINEERING_SOURCE_PILOT_ONLY" or saved.get("identity_sha256") != digest(identity) or saved.get("identity") != identity:
        raise PilotGateError("Checkpoint code/data/config/device identity changed")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    sampler.set_state(saved["sampler_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"], 0)
    random.setstate(saved["python_rng"])
    state = saved["numpy_rng"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))


def _new_state(n_times):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = CompactEEGNet(11, n_times, dropout=0.25).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0, foreach=False)
    sampler = torch.Generator(device="cpu").manual_seed(SEED + 1000003)
    return model, optimizer, sampler


def _timing_summary(steps):
    """Pure CPU summarizer; fixed updates 0..9 are warmup, 10..99 measured.

    CUDA events measure elapsed stream intervals, including possible host gaps;
    these are not a sum of kernel durations and are not GPU billing time.
    """
    measured = [row for row in steps if row["branch_update_zero_based"] >= WARMUP_UPDATES]
    def distribution(key):
        values = [row[key] for row in measured]
        return {"count": len(values), "p50_seconds": float(np.percentile(values, 50)) if values else None,
                "p90_seconds": float(np.percentile(values, 90)) if values else None,
                "total_seconds": float(sum(values))}
    return {"warmup_updates_in_this_segment": len(steps) - len(measured),
            "measured_updates_in_this_segment": len(measured),
            "warmup_rule": "branch update indices 0..9; included in the fixed optimizer-update budget",
            "cpu_sampling_normalization_total_seconds": sum(row["cpu_sampling_normalization_seconds"] for row in steps),
            "measured_cpu_sampling_normalization": distribution("cpu_sampling_normalization_seconds"),
            "measured_h2d": distribution("gpu_h2d_seconds"),
            "measured_forward_backward_step": distribution("gpu_forward_backward_step_seconds"),
            "percentile_method": "numpy.percentile default linear interpolation",
            "billing_time": False, "timing_scope": "CUDA event elapsed stream intervals; monotonic wall includes host preparation; neither is an account billing receipt",
            "per_update_timings": steps}


def _updates(cache, model, optimizer, sampler, begin, stop, *, deadline, progress, phase):
    wall_start = time.monotonic()
    weights = sampling_weights(cache)
    model.train()
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    markers = []
    event_start.record()
    for update in range(begin, stop):
        if time.monotonic() > deadline:
            raise TimeoutError("Controlled engineering pilot wall-time budget reached")
        cpu_start = time.monotonic()
        indices = torch.multinomial(weights, BATCH_SIZE, replacement=True, generator=sampler)
        host_batch = joint_trial_rms(torch.from_numpy(cache.batch(indices))).float()
        host_labels = torch.from_numpy(cache.labels[indices.numpy()].copy())
        cpu_prepare_seconds = time.monotonic() - cpu_start
        transfer_start, transfer_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        step_start, step_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        transfer_start.record()
        batch = host_batch.to("cuda:0")
        labels = host_labels.to("cuda:0")
        transfer_end.record()
        optimizer.zero_grad(set_to_none=True)
        step_start.record()
        loss = nn.functional.cross_entropy(model(batch), labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite source engineering loss")
        progress["training_started"] = True
        loss.backward()
        optimizer.step()
        step_end.record()
        progress["optimizer_updates_dispatched"] += 1
        markers.append((update, cpu_prepare_seconds, transfer_start, transfer_end, step_start, step_end))
    event_end.record()
    torch.cuda.synchronize(0)
    progress["actual_completed_updates"] += stop - begin
    elapsed_wall = time.monotonic() - wall_start
    steps = [{"branch_update_zero_based": update, "cpu_sampling_normalization_seconds": cpu,
              "gpu_h2d_seconds": transfer_start.elapsed_time(transfer_end) / 1000,
              "gpu_forward_backward_step_seconds": step_start.elapsed_time(step_end) / 1000}
             for update, cpu, transfer_start, transfer_end, step_start, step_end in markers]
    return {"phase": phase, "begin_update": begin, "end_update": stop, "actual_updates": stop - begin,
            "cuda_event_seconds": event_start.elapsed_time(event_end) / 1000,
            "monotonic_wall_seconds": elapsed_wall,
            "timing_summary": _timing_summary(steps),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def _prediction_replay(cache, output, identity, *, io_timings, deadline):
    """Reload both committed 100-update states and compare source-only evals."""
    indices, uids, labels = engineering_prediction_selection(cache)
    batch_sha = digest({"event_uids": uids, "labels": labels})
    outputs, descriptors = [], []
    for checkpoint_name, prediction_name in zip(CHECKPOINT_NAMES[1::2], PREDICTION_NAMES):
        _check_deadline(deadline)
        checkpoint_path = output / checkpoint_name
        checkpoint = _descriptor(cache.root, checkpoint_path)
        _check_deadline(deadline)
        saved = _load_checkpoint(checkpoint_path, expected_sha=checkpoint["sha256"], io_timings=io_timings, deadline=deadline)
        _validate_checkpoint(saved, identity, 100, path=checkpoint_path, io_timings=io_timings, deadline=deadline)
        model, optimizer, sampler = _new_state(cache.n_times)
        _io_event(io_timings, "checkpoint_restore_model_optimizer_rng", checkpoint_path,
                  lambda: _restore(saved, model, optimizer, sampler, identity), deadline=deadline)
        training = model.training
        torch.cuda.synchronize(0)
        _check_deadline(deadline)
        eval_start = time.monotonic()
        try:
            model.eval()
            with torch.no_grad():
                host = joint_trial_rms(torch.from_numpy(cache.batch(indices))).float()
                logits = model(host.to("cuda:0"))
                probabilities = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(-1)
                torch.cuda.synchronize(0)
                z, p = logits.cpu().numpy(), probabilities.cpu().numpy()
                predicted = predictions.cpu().numpy()
        finally:
            model.train(training)
        elapsed = time.monotonic() - eval_start
        _check_deadline(deadline)
        if (z.shape != (len(uids), 2) or p.shape != z.shape or not np.isfinite(z).all()
                or not np.isfinite(p).all() or not _same_tree(model.state_dict(), saved["model"])
                or not _same_tree(optimizer.state_dict(), saved["optimizer"])):
            raise PilotGateError("Source engineering eval changed saved state or produced invalid logits")
        record = {"schema_version": "paper3-source-engineering-predictions-v1",
            "scope": PREDICTION_SCOPE, "role": "source_train", "synthetic": False,
            "scientific_result": False, "identity_sha256": digest(identity),
            "input_contract_sha256": cache.input_contract_sha256, "cache_manifest_sha256": cache.manifest_sha256,
            "source_batch_sha256": batch_sha, "event_uids": uids, "labels": labels,
            "checkpoint": checkpoint, "model_update": 100, "optimizer_updates": 0, "eval_mode": True,
            "prediction_rule": "two_logits.argmax(-1); exact tie selects class 0",
            "two_logits": z.tolist(), "probabilities": p.tolist(), "predictions": predicted.tolist(),
            "metrics": engineering_prediction_metrics(labels, p, predictions=predicted),
            "eval_monotonic_seconds": elapsed}
        path = output / prediction_name
        _atomic_json(path, record)
        outputs.append(record)
        descriptors.append({"checkpoint": checkpoint, "predictions": _descriptor(cache.root, path)})
        del saved, model, optimizer, sampler, host, logits, probabilities, predictions
        torch.cuda.empty_cache()
        _check_deadline(deadline)
    comparisons = {key: outputs[0][key] == outputs[1][key]
                   for key in ("two_logits", "probabilities", "predictions", "labels", "metrics")}
    # The public proof uses the conventional name logits for the two-logit comparison.
    comparisons["logits"] = comparisons.pop("two_logits")
    proof = {"schema_version": "paper3-source-engineering-prediction-replay-v1",
        "scope": PREDICTION_SCOPE, "role": "source_train", "synthetic": False,
        "scientific_result": False, "identity_sha256": digest(identity),
        "input_contract_sha256": cache.input_contract_sha256, "cache_manifest_sha256": cache.manifest_sha256,
        "selection_rule": PREDICTION_SELECTION_RULE, "event_uids": uids, "labels": labels,
        "source_batch_sha256": batch_sha, "optimizer_updates_for_replay": 0,
        "continuous": descriptors[0], "resumed": descriptors[1],
        "comparisons": comparisons, "all_exact": all(comparisons.values())}
    proof_path = output / "engineering_prediction_replay.json"
    _atomic_json(proof_path, proof)
    _check_deadline(deadline)
    if not proof["all_exact"]:
        raise PilotGateError("Same-device continuous/resumed source predictions or metrics differ")
    return _descriptor(cache.root, proof_path)


def execute_pilot(cache, output_relative, *, user_started_gpu):
    """Only explicit CLI execution calls this; never invoked by fixture tests."""
    if not user_started_gpu:
        raise PilotGateError("Both --execute and --user-started-gpu are required")
    if not torch.cuda.is_available():
        raise PilotGateError("CUDA is unavailable; no CPU training fallback is permitted")
    root = cache.root
    relative = Path(output_relative)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("runs",):
        raise PilotGateError("Pilot outputs must use a new root-relative runs/ directory")
    output = _result_destination(root, root / relative)
    if not output.is_relative_to(root):
        raise PilotGateError("Output exists or escaped the execution root; no duplicate pilot")
    prefit_resume = None
    if output.exists():
        if os.environ.get("PAPER3_HPC_RUNTIME_PROFILE"):
            from hpc.pilot_prefit import consume_prefit_resume
            prefit_resume = consume_prefit_resume(root, output)
        else:
            raise PilotGateError("Output exists or escaped the execution root; no duplicate pilot")
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    claim = runs / "controlled_pilot_job_claim.json"
    # A persistent one-job claim prevents accidental reexecution with another output path.
    if prefit_resume is None:
        try:
            with claim.open("x") as stream:
                json.dump({"scope": "ENGINEERING_SOURCE_PILOT_ONLY", "max_jobs": 1,
                           "output": str(relative), "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
        except FileExistsError:
            raise PilotGateError("The one controlled pilot job was already claimed; inspect its receipt") from None
        output.mkdir()
    receipt = {"status": "RUNNING", "scope": "ENGINEERING_SOURCE_PILOT_ONLY",
               "scientific_f0": False, "source_lodo_selection_verified": False,
               "training_started": False,
               "remote_actions": False, "server_started_by_runner": False,
               "user_started_gpu_attested": True, "seed": SEED, "max_jobs": 1,
               "actual_completed_updates": 0, "exact_resume_verified": False,
               "optimizer_updates_dispatched": 0,
               "checkpoint_io_timings": [], "prediction_replay_optimizer_updates": 0,
               "completed_update_count_basis": "CUDA-synchronized complete segments; a lower bound on failure",
               "started_utc": datetime.now(timezone.utc).isoformat()}
    if prefit_resume is not None:
        receipt["started_utc"] = prefit_resume["original_started_utc"]
        receipt["prefit_resume"] = prefit_resume
    _atomic_json(output / "engineering_receipt.json", receipt)
    wall_start = time.monotonic() - (prefit_resume["prior_wall_seconds"] if prefit_resume else 0)
    try:
        from p3work.runtime_clock import collect_clock_preflight
        receipt["clock_preflight"] = collect_clock_preflight()
        _atomic_json(output / "engineering_receipt.json", receipt)
        dependencies = _resolved_dependencies(root)
        receipt["resolved_dependencies"] = dependencies
        if os.environ.get("PAPER3_HPC_RUNTIME_PROFILE"):
            from hpc.runtime_profile import validate_runtime_profile
            receipt["deployment_runtime_profile"] = validate_runtime_profile(
                root, os.environ["PAPER3_HPC_RUNTIME_PROFILE"])
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") not in {":4096:8", ":16:8"}:
            raise PilotGateError("CUBLAS workspace configuration is incompatible with deterministic execution")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        try:
            driver_query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=10, check=True)
            driver_version = driver_query.stdout.strip().splitlines()
        except (FileNotFoundError, subprocess.SubprocessError):
            driver_version = None
        identity = {"seed": SEED, "batch_size": BATCH_SIZE, "optimizer": "Adam",
                    "learning_rate": LEARNING_RATE, "weight_decay": 0, "foreach": False,
                    "dropout": 0.25, "torch": str(torch.__version__), "numpy": np.__version__,
                    "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "device": "cuda:0", "device_name": props.name,
                    "device_uuid": str(getattr(props, "uuid", "not_exposed")),
                    "device_total_memory": props.total_memory, "compute_capability": [props.major, props.minor],
                    "driver_versions_local_nvidia_smi": driver_version, "python": sys.version,
                    "tf32": False, "mixed_precision": False, "deterministic_algorithms": True,
                    "sampling": "equal dataset/subject/observed-class/trial with replacement",
                    "input_contract_sha256": cache.input_contract_sha256,
                    "cache_manifest_sha256": cache.manifest_sha256,
                    "cache_artifacts": cache.artifact_sha256,
                    "upstream_acceptance_sha256": cache.upstream_acceptance_sha256,
                    "runner_sha256": file_digest(__file__),
                    "bridge_sha256": file_digest(root / "p3work/f0_bridge.py"),
                    "geometry_sha256": file_digest(root / "p3work/spatial_geometry.py"),
                    "contracts_sha256": file_digest(root / "p3work/contracts.py")}
        identity["resolved_dependencies"] = dependencies
        if receipt.get("deployment_runtime_profile"):
            identity["deployment_runtime_profile"] = receipt["deployment_runtime_profile"]
        _atomic_json(output / "identity.json", identity)
        deadline = wall_start + MAX_WALL_SECONDS
        io_timings = receipt["checkpoint_io_timings"]
        model, optimizer, sampler = _new_state(cache.n_times)
        initial = _checkpoint(model, optimizer, sampler, 0, identity)
        _atomic_checkpoint(output / "engineering_initial.pt", initial, io_timings=io_timings, deadline=deadline)
        timings = [_updates(cache, model, optimizer, sampler, 0, 100, deadline=deadline, progress=receipt, phase="continuous100_primary_timing")]
        _atomic_checkpoint(output / "engineering_continuous100.pt", _checkpoint(model, optimizer, sampler, 100, identity), io_timings=io_timings, deadline=deadline)
        receipt.update(actual_completed_updates=100, timing_segments=timings)
        _atomic_json(output / "engineering_receipt.json", receipt)
        # Destroy the first branch before constructing the replay branch on this GPU.
        del initial, model, optimizer, sampler
        torch.cuda.empty_cache()
        model, optimizer, sampler = _new_state(cache.n_times)
        fresh = _checkpoint(model, optimizer, sampler, 0, identity)
        original = _load_checkpoint(output / "engineering_initial.pt", io_timings=io_timings, deadline=deadline)
        _validate_checkpoint(original, identity, 0, path=output / "engineering_initial.pt", io_timings=io_timings, deadline=deadline)
        initial_equal = _io_event(io_timings, "checkpoint_initial_state_exact_comparison", output / "engineering_initial.pt",
            lambda: all(_same_tree(fresh[key], original[key]) for key in ("model", "optimizer", "sampler_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng")), deadline=deadline)
        if not initial_equal:
            raise RuntimeError("Continuous and resume branches did not share the exact initial state")
        del fresh, original
        timings.append(_updates(cache, model, optimizer, sampler, 0, 50, deadline=deadline, progress=receipt, phase="resume_before_checkpoint"))
        _atomic_checkpoint(output / "engineering_resume50.pt", _checkpoint(model, optimizer, sampler, 50, identity), io_timings=io_timings, deadline=deadline)
        receipt.update(actual_completed_updates=150, timing_segments=timings)
        _atomic_json(output / "engineering_receipt.json", receipt)
        del model, optimizer, sampler
        torch.cuda.empty_cache()
        model, optimizer, sampler = _new_state(cache.n_times)
        checkpoint_path = output / "engineering_resume50.pt"
        saved = _load_checkpoint(checkpoint_path, io_timings=io_timings, deadline=deadline)
        _validate_checkpoint(saved, identity, 50, path=checkpoint_path, io_timings=io_timings, deadline=deadline)
        _io_event(io_timings, "checkpoint_restore_model_optimizer_rng", checkpoint_path,
                  lambda: _restore(saved, model, optimizer, sampler, identity), deadline=deadline)
        del saved
        timings.append(_updates(cache, model, optimizer, sampler, 50, 100, deadline=deadline, progress=receipt, phase="resume_after_checkpoint_reload"))
        _atomic_checkpoint(output / "engineering_resumed100.pt", _checkpoint(model, optimizer, sampler, 100, identity), io_timings=io_timings, deadline=deadline)
        receipt.update(actual_completed_updates=TOTAL_UPDATES, timing_segments=timings)
        continuous = _load_checkpoint(output / "engineering_continuous100.pt", io_timings=io_timings, deadline=deadline)
        resumed = _load_checkpoint(output / "engineering_resumed100.pt", io_timings=io_timings, deadline=deadline)
        for name, saved_state in (("engineering_continuous100.pt", continuous), ("engineering_resumed100.pt", resumed)):
            _validate_checkpoint(saved_state, identity, 100, path=output / name, io_timings=io_timings, deadline=deadline)
        compared = ("model", "optimizer", "sampler_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "update", "identity_sha256")
        same = _io_event(io_timings, "checkpoint_final_state_exact_comparison", output / "engineering_resumed100.pt",
                         lambda: {key: _same_tree(continuous[key], resumed[key]) for key in compared}, deadline=deadline)
        receipt.update(resume_comparison=same, exact_resume_verified=all(same.values()))
        if not all(same.values()):
            raise RuntimeError("Same-device 100 versus 50+50 checkpoint continuation differs")
        del continuous, resumed, model, optimizer, sampler
        torch.cuda.empty_cache()
        receipt["engineering_prediction_replay"] = _prediction_replay(cache, output, identity,
            io_timings=io_timings, deadline=deadline)
        receipt["status"] = "ENGINEERING_PILOT_COMPLETE"
    except BaseException as error:
        receipt.update(status="ENGINEERING_PILOT_FAILED", failure_type=type(error).__name__,
                       failure=str(error), exact_resume_verified=bool(receipt.get("exact_resume_verified", False)))
    finally:
        receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                       total_monotonic_wall_seconds=time.monotonic() - wall_start)
        try:
            receipt["checkpoint_files"] = {}
            for name in CHECKPOINT_NAMES:
                path = output / name
                if path.exists() or path.is_symlink():
                    receipt["checkpoint_files"][name] = _descriptor(root, path)["sha256"]
            _finalize_result_manifests(root, output, receipt)
        except BaseException as error:
            receipt.update(status="ENGINEERING_PILOT_FAILED", result_manifest_error_type=type(error).__name__,
                           result_manifest_status="BLOCKED_NO_COMPLETE_BACKUP_MANIFEST")
            _atomic_json(output / "engineering_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-manifest", default=None)
    parser.add_argument("--output", default="runs/controlled_source_pilot_seed20261018")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-started-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute and not args.user_started_gpu:
            raise PilotGateError("Both --execute and --user-started-gpu are required; the runner never boots a server")
        cache = verify_inputs(args.root, manifest_path=args.cache_manifest)
        if not args.execute:
            result = {"status": "DRY_RUN_GATES_PASSED", "training_started": False,
                      "gpu_checked": False, "remote_actions": False,
                      "input_contract_sha256": cache.input_contract_sha256,
                      "cache_manifest_sha256": cache.manifest_sha256,
                      "source_datasets": sorted({r["dataset"] for r in cache.rows}),
                      "original_training_trials": len(cache.rows), "planned_actual_updates": TOTAL_UPDATES,
                      "resume_design": "100 continuous versus independently initialized 50+checkpoint reload+50 on same device",
                      "actual_gpu_resume_verified": False}
        else:
            result = execute_pilot(cache, args.output, user_started_gpu=args.user_started_gpu)
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") != "ENGINEERING_PILOT_FAILED" else 3
    except (PilotGateError, ValueError, FileNotFoundError, KeyError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error),
                          "training_started": False, "remote_actions": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
