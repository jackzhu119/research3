"""CPU-only, receipt-verified Rref production/loading from f0 training events.

No source fit, GPU, external dataset, or download is performed. Actual production
requires completed W1/W2, verified new source caches and a real SOURCE_F0_COMPLETE
receipt. In-memory matrices/provenance claims alone do not pass these gates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch

from .contracts import SOURCE_IDS, file_digest, validate_signal_contract
from .f0_bridge import load_verified_source
from .spatial_geometry import (ReferenceCovariance, SourceCovarianceTrial, helmert_rows,
                               source_reference, spd_roots, tensor_sha256)

SCHEMA = "paper3-source-reference-v2"
UID_SCHEMA = "paper3-source-training-uids-v2"
WEIGHTING = "dataset_equal/subject_equal/trial_equal"


def _local(root, relative, *, exists=True):
    from tools.r2_restore import no_symlink_components
    if not isinstance(relative, str) or not relative:
        raise ValueError("Explicit root-relative source artifact path required")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Source reference artifact path escaped the root")
    result = root / path
    no_symlink_components(result)
    resolved = result.resolve(strict=exists)
    if not resolved.is_relative_to(root) or "legacy_baseline_v1" in path.parts:
        raise ValueError("Source reference cannot use legacy or escaping artifacts")
    return resolved


def _artifact(root, item):
    if not isinstance(item, dict) or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None:
        raise ValueError("Source reference artifact requires an actual SHA256 descriptor")
    path = _local(root, item.get("path"))
    if not path.is_file() or file_digest(path) != item["sha256"]:
        raise ValueError("Source reference artifact integrity mismatch")
    if "bytes" in item and (type(item["bytes"]) is not int or item["bytes"] != path.stat().st_size):
        raise ValueError("Source reference artifact byte count mismatch")
    return path


def _uid_set_sha(uids):
    return hashlib.sha256("\n".join(sorted(uids)).encode()).hexdigest()


def _context(root, source_receipt_path, contract):
    """The production and loading gates are identical and run before output writes."""
    from tools.run_controlled_pilot import verify_inputs
    root = Path(root).resolve(strict=True)
    signal_sha = validate_signal_contract(contract)
    cache = verify_inputs(root)
    if cache.input_contract_sha256 != signal_sha:
        raise ValueError("Reference contract differs from the frozen source cache")
    receipt_path = _local(root, source_receipt_path)
    # Preserve caller RNG state. The bridge builds CPU modules before loading
    # their real state, and those initial allocations otherwise consume RNG.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        source, source_receipt = load_verified_source(root, source_receipt_path, contract)
    source.assert_unchanged()
    if any(p.device.type != "cpu" for p in source.parameters()):
        raise ValueError("Source reference generation/loading is CPU only")
    del source
    training_manifest_path = _artifact(root, source_receipt["training_uid_manifest"])
    training = json.loads(training_manifest_path.read_text())
    uids = training.get("event_uids")
    if (training.get("schema_version") != UID_SCHEMA or training.get("role") != "source_pretrain_train"
            or training.get("synthetic") is not False or training.get("signal_contract_sha256") != signal_sha
            or not isinstance(uids, list) or not uids or any(not isinstance(uid, str) or not uid for uid in uids)
            or len(set(uids)) != len(uids)):
        raise ValueError("A unique real source_pretrain_train UID allowlist bound to the input is required")
    uid_set = set(uids)
    cache_index = {}
    for row, location in zip(cache.rows, cache.locations):
        uid = row["event_uid"]
        if uid in cache_index:
            raise ValueError("Duplicated source-cache original event UID")
        cache_index[uid] = (row, location)
    if not uid_set <= cache_index.keys():
        raise PermissionError("f0 training UID allowlist contains events outside the verified source caches")
    selected_datasets = {cache_index[uid][0]["dataset"] for uid in uids}
    if selected_datasets != SOURCE_IDS:
        raise PermissionError("Rref needs only and all four registered source training datasets")
    for uid in uids:
        if cache_index[uid][0]["role"] != "source_train":
            raise PermissionError("Rref cannot consume source validation/development/query events")
    if (source_receipt.get("source_cache_manifest_sha256") is not None
            and source_receipt["source_cache_manifest_sha256"] != cache.manifest_sha256):
        raise ValueError("f0 and Rref source cache manifests differ")
    return root, cache, receipt_path, source_receipt, training_manifest_path, tuple(sorted(uid_set)), cache_index


def _trials(cache, uids, index):
    for uid in uids:
        row, (shard, trial) = index[uid]
        # Only selected training rows are ever used for covariance estimation.
        x = torch.from_numpy(np.asarray(cache.arrays[shard][trial]).copy()).to(dtype=torch.float64, device="cpu")
        yield SourceCovarianceTrial(row["dataset"], str(row["subject_id"]), uid, x,
                                    role="source_pretrain_train")


def _write_json(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _describe(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}


def prepare_source_reference(root, source_receipt_path, contract, *, execute=False,
                             output="runs/source_reference_v2"):
    """Default read-only plan; --execute computes actual Rref on authorized CPU data."""
    root, cache, source_path, source_receipt, uid_path, uids, index = _context(root, source_receipt_path, contract)
    result = {"schema_version": SCHEMA, "status": "PLANNED_SOURCE_REFERENCE", "execute": execute,
              "training_started": False, "gpu_execution": False, "external_query_accessed": False,
              "signal_contract_sha256": cache.input_contract_sha256,
              "source_f0_receipt_sha256": file_digest(source_path),
              "source_f0_receipt": _describe(root, source_path),
              "source_training_uid_manifest": _describe(root, uid_path),
              "source_training_uid_manifest_sha256": file_digest(uid_path),
              "training_uid_set_sha256": _uid_set_sha(uids), "training_events": len(uids),
              "training_dataset_ids": sorted(SOURCE_IDS), "role": "source_pretrain_train",
              "source_cache_manifest": {"path": cache.manifest_path, "sha256": cache.manifest_sha256},
              "weighting": WEIGHTING, "shrinkage": 0.05,
              "implementation_sha256": file_digest(__file__),
              "geometry_implementation_sha256": file_digest(Path(__file__).with_name("spatial_geometry.py"))}
    if not execute:
        return result
    folder = _local(root, output, exists=False)
    if folder.exists():
        raise ValueError("Reference output already exists; do not overwrite a frozen reference")
    reference = source_reference(_trials(cache, uids, index), q=helmert_rows(device="cpu"))
    if reference.events_sha256 != result["training_uid_set_sha256"]:
        raise ValueError("Estimated reference events differ from the f0 training UID allowlist")
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {"Q": helmert_rows(device="cpu"), "reference": reference.matrix,
              "reference_sqrt": reference.sqrt, "reference_invsqrt": reference.invsqrt}
    descriptors = {}
    for name, value in arrays.items():
        path = folder / (name + ".npy")
        np.save(path, value.detach().cpu().numpy(), allow_pickle=False)
        descriptors[name] = {**_describe(root, path), "array_sha256": tensor_sha256(value)}
    event_path = folder / "training_event_uids.json"
    _write_json(event_path, {"role": "source_pretrain_train", "event_uids": list(reference.event_uids),
                             "training_uid_set_sha256": reference.events_sha256})
    result.update(status="SOURCE_REFERENCE_COMPLETE", synthetic=False,
                  recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  arrays=descriptors, training_event_uids=_describe(root, event_path))
    _write_json(folder / "receipt.json", result)
    return result


def load_verified_reference(root, receipt_path, source_receipt_path, contract):
    """Recheck stage/f0/cache/UID/file/array/SPD bindings before returning Rref."""
    root, cache, source_path, source_receipt, uid_path, uids, index = _context(root, source_receipt_path, contract)
    path = _local(root, receipt_path)
    receipt = json.loads(path.read_text())
    if (receipt.get("schema_version") != SCHEMA or receipt.get("status") != "SOURCE_REFERENCE_COMPLETE"
            or receipt.get("synthetic") is not False or receipt.get("execute") is not True
            or receipt.get("role") != "source_pretrain_train" or receipt.get("training_started") is not False
            or receipt.get("gpu_execution") is not False or receipt.get("external_query_accessed") is not False
            or receipt.get("signal_contract_sha256") != cache.input_contract_sha256
            or receipt.get("source_f0_receipt_sha256") != file_digest(source_path)
            or receipt.get("source_training_uid_manifest_sha256") != file_digest(uid_path)
            or receipt.get("training_uid_set_sha256") != _uid_set_sha(uids)
            or receipt.get("training_events") != len(uids)
            or receipt.get("training_dataset_ids") != sorted(SOURCE_IDS)
            or receipt.get("weighting") != WEIGHTING or receipt.get("shrinkage") != 0.05):
        raise ValueError("A real, source-training-only input/f0/UID-bound Rref receipt is required")
    if (_artifact(root, receipt.get("source_f0_receipt")) != source_path
            or _artifact(root, receipt.get("source_training_uid_manifest")) != uid_path
            or _artifact(root, receipt.get("source_cache_manifest")) != root / cache.manifest_path):
        raise ValueError("Reference source artifact identities changed")
    if (receipt.get("implementation_sha256") != file_digest(__file__)
            or receipt.get("geometry_implementation_sha256") != file_digest(Path(__file__).with_name("spatial_geometry.py"))):
        raise ValueError("Reference implementation changed; produce an explicitly versioned reference")
    event_path = _artifact(root, receipt.get("training_event_uids"))
    events = json.loads(event_path.read_text())
    if (events.get("role") != "source_pretrain_train" or events.get("event_uids") != list(uids)
            or events.get("training_uid_set_sha256") != _uid_set_sha(uids)):
        raise ValueError("Saved Rref event set differs from the exact f0 training allowlist")
    expected_shapes = {"Q": (10, 11), "reference": (10, 10), "reference_sqrt": (10, 10), "reference_invsqrt": (10, 10)}
    values = {}
    for name, shape in expected_shapes.items():
        item = receipt.get("arrays", {}).get(name)
        artifact_path = _artifact(root, item)
        if artifact_path.suffix != ".npy":
            raise ValueError("Reference arrays must be pickle-free NPY files")
        array = np.load(artifact_path, allow_pickle=False)
        if array.shape != shape or array.dtype != np.dtype("float64") or not np.isfinite(array).all():
            raise ValueError("Reference array dtype/shape/finiteness mismatch")
        value = torch.from_numpy(array.copy())
        if tensor_sha256(value) != item.get("array_sha256"):
            raise ValueError("Reference numeric-array digest mismatch")
        values[name] = value
    if not torch.equal(values["Q"], helmert_rows(device="cpu")):
        raise ValueError("Saved Q differs from the fixed clinic11 Helmert array")
    reference = values["reference"]
    root_matrix, inverse = spd_roots(reference)
    if (not torch.allclose(torch.trace(reference), reference.new_tensor(1.), atol=1e-10, rtol=0)
            or float(torch.linalg.eigvalsh(reference).min()) < 0.005 - 1e-12
            or not torch.allclose(values["reference_sqrt"], root_matrix, atol=1e-10, rtol=1e-10)
            or not torch.allclose(values["reference_invsqrt"], inverse, atol=1e-10, rtol=1e-10)):
        raise ValueError("Reference unit-trace/source-shrinkage/SPD-root checks failed")
    return ReferenceCovariance(reference, root_matrix, inverse, uids, _uid_set_sha(uids)), receipt
