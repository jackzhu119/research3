"""Verify measured development engineering evidence without running a fit.

Only real seed-18 source/reference and development catalogs can authorize this
proof. Loading checkpoints never restores RNG and never initializes CUDA.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .contracts import digest, file_digest, validate_signal_contract
from .episodes import load_episode_catalog
from .f0_bridge import load_verified_source
from .meta_training import META_CODE_FILES, META_RUNTIME_KEYS, MetaJob
from .selection import choose_checkpoint
from .source_reference_io import load_verified_reference
from .spatial_geometry import SpatialGeometry
from .training_state import local_path


_RUNTIME_KEYS = META_RUNTIME_KEYS
_COMMON_INPUTS = ("signal_contract_sha256", "source_architecture", "source_feature_dim",
    "source_training_event_uids_sha256", "geometry_sha256", "stieger_manifest_sha256", "split_sha256")
_CHECKPOINT_INPUTS = _COMMON_INPUTS + ("source_model_state_sha256", "source_receipt_sha256", "reference_receipt_sha256")


def _artifact(root, descriptor):
    if type(descriptor) is not dict or not {"path", "sha256"} <= set(descriptor):
        raise ValueError("An independently SHA-bound engineering artifact required")
    path = local_path(root, descriptor["path"])
    if (not path.is_file() or file_digest(path) != descriptor["sha256"]
            or ("bytes" in descriptor and descriptor["bytes"] != path.stat().st_size)):
        raise ValueError("Engineering artifact full SHA/bytes mismatch")
    return path


def _positive(value):
    return type(value) in {int, float} and math.isfinite(value) and value > 0


def _runtime_valid(profile):
    strings = ("python", "torch", "cuda", "device_name", "device_uuid")
    driver, capability, cudnn = profile.get("driver_versions"), profile.get("device_compute_capability"), profile.get("cudnn")
    return (all(isinstance(profile.get(key), str) and profile[key].strip() for key in strings)
        and (cudnn is None or type(cudnn) is int and cudnn > 0)
        and (driver is None or type(driver) is list and all(isinstance(v, str) and v.strip() for v in driver))
        and type(profile.get("device_total_memory_bytes")) is int and profile["device_total_memory_bytes"] > 0
        and type(capability) is list and len(capability) == 2 and all(type(v) is int and v >= 0 for v in capability)
        and profile.get("autocast") is False and profile.get("tf32") is False
        and profile.get("float32_matmul_precision") == "highest"
        and profile.get("deterministic_algorithms") is True
        and isinstance(profile.get("cublas_workspace_config"), str)
        and profile["cublas_workspace_config"] in {":4096:8", ":16:8"})


def verify_meta_engineering(root, descriptor, *, job, inputs, runtime_profile=None):
    """Read-only proof gate; another pipeline seed may have different f0 weights.

    ``inputs`` comes from the currently verified scientific job. The engineering
    source weights are checked against their own real seed-18 receipts, while
    architecture, reference, UIDs, input and catalog must cover the current job.
    The optional runtime profile compares recorded CPU/CUDA/device/code metadata
    directly; this function never queries a CUDA device.
    """
    root = Path(root).resolve(strict=True)
    if type(job) is not MetaJob or job.phase == "engineering":
        raise ValueError("Engineering evidence gates a registered scientific meta job")
    path = _artifact(root, descriptor)
    proof = json.loads(path.read_text())
    engineering = MetaJob(**proof["job"])
    if (proof.get("status") != "META_ENGINEERING_COMPLETE" or proof.get("synthetic") is not False
            or proof.get("training_started") is not True or proof.get("scientific_fit_completed") is not False
            or proof.get("source_state_unchanged") is not True or proof.get("full_unroll") is not True
            or engineering.phase != "engineering" or engineering.seed != 20261018
            or engineering.method != job.method or engineering.dimension < job.dimension
            or proof.get("outer_updates_completed") != 100 or proof.get("validation_checkpoints") != 2
            or any(proof.get(key) != inputs.get(key) for key in _COMMON_INPUTS)):
        raise ValueError("A complete real same-input same-method 100-outer engineering proof required")
    contract = json.loads(local_path(root, "contracts/input_contract.json").read_text())
    signal_sha = validate_signal_contract(contract)
    if (signal_sha != proof.get("signal_contract_sha256")
            or contract["window_audit_receipt"]["split_sha256"] != proof.get("split_sha256")):
        raise ValueError("Engineering input/split differs from the frozen window contract")
    source_descriptor = proof.get("source_receipt")
    reference_descriptor = proof.get("reference_receipt")
    source_path, reference_path = _artifact(root, source_descriptor), _artifact(root, reference_descriptor)
    if (file_digest(source_path) != proof.get("source_receipt_sha256")
            or file_digest(reference_path) != proof.get("reference_receipt_sha256")):
        raise ValueError("Engineering source/reference receipts differ from the recorded fit identity")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        source, source_info = load_verified_source(root, source_path.relative_to(root).as_posix(), contract)
        reference, reference_info = load_verified_reference(root, reference_path.relative_to(root).as_posix(),
            source_path.relative_to(root).as_posix(), contract)
    if (source_info.get("pipeline_seed") != 20261018 or source_info.get("seed") != 20261018
            or source_info.get("signal_contract_sha256") != signal_sha
            or source_info.get("architecture") != proof.get("source_architecture")
            or source.feature_dim != proof.get("source_feature_dim")
            or source.frozen_state_sha256 != proof.get("source_model_state_sha256")
            or reference.events_sha256 != proof.get("source_training_event_uids_sha256")
            or reference_info.get("source_f0_receipt_sha256") != file_digest(source_path)
            or reference_info.get("source_cache_manifest", {}).get("sha256") != source_info.get("source_cache_manifest_sha256")
            or not source_info.get("source_cache_manifest_sha256")
            or SpatialGeometry.from_reference(reference.matrix).sha256() != proof.get("geometry_sha256")):
        raise ValueError("Engineering needs its actual same-input seed-18 source/reference artifacts")
    source.assert_unchanged()
    train = load_episode_catalog(root, None, None, role="meta_train", purpose="meta_gradient")
    validation = load_episode_catalog(root, None, None, role="meta_validation", purpose="model_selection")
    if any(catalog.manifest_sha256 != proof.get("stieger_manifest_sha256")
           or catalog.input_contract_sha256 != signal_sha or catalog.split_sha256 != proof.get("split_sha256")
           for catalog in (train, validation)):
        raise ValueError("Engineering catalog/split/signal identity changed")
    episode_path = _artifact(root, proof.get("episode_plan"))
    episode = json.loads(episode_path.read_text())
    if (episode.get("schema_version") != "paper3-frozen-meta-episode-plans-v1"
            or episode.get("input_contract_sha256") != signal_sha
            or episode.get("split_sha256") != proof.get("split_sha256")
            or episode.get("stieger_manifest_sha256") != proof.get("stieger_manifest_sha256")
            or episode.get("budget_acquired_trials") != 20
            or episode.get("query_rule") != "all_eligible_trials_in_future_complete_physical_runs_after_B20_boundary"
            or episode.get("final_or_internal_query_included") is not False
            or episode.get("meta_train") != [plan.to_dict() for plan in train.plans]
            or episode.get("meta_validation") != [plan.to_dict() for plan in validation.plans]):
        raise ValueError("Engineering episode plan differs from the complete authorized development catalogs")
    history = proof.get("validation", [])
    if ([row.get("outer_update") for row in history] != [50, 100]
            or proof.get("selected_checkpoint") != choose_checkpoint(history)):
        raise ValueError("Both measured engineering validation scores and their registered selection required")
    final = proof.get("checkpoint", {})
    if (final.get("path") != "outer-0100.pt" or final.get("relative_to_job_output") is not True
            or final.get("outer_cursor") != 100):
        raise ValueError("The actual final engineering checkpoint is required")
    final_path = _artifact(root, {"path": (path.parent.relative_to(root) / final["path"]).as_posix(), "sha256": final.get("sha256")})
    sidecar_path = local_path(root, final_path.with_suffix(final_path.suffix + ".sha256.json").relative_to(root).as_posix())
    sidecar = json.loads(sidecar_path.read_text())
    if (sidecar.get("sha256") != final["sha256"] or sidecar.get("bytes") != final_path.stat().st_size
            or sidecar.get("identity_sha256") != proof.get("identity_sha256")):
        raise ValueError("Engineering final checkpoint has a torn/mismatched SHA sidecar")
    record = torch.load(final_path, map_location="cpu", weights_only=True)
    identity, state = record.get("identity", {}), record.get("state", {})
    if (record.get("schema_version") != "paper3-trusted-training-state-v1"
            or record.get("identity_sha256") != digest(identity) or proof.get("identity_sha256") != digest(identity)
            or identity.get("job") != proof["job"]
            or any(identity.get(key) != proof.get(key) for key in _CHECKPOINT_INPUTS)
            or identity.get("episode_plan_sha256") != file_digest(episode_path)
            or identity.get("full_unroll") is not True or identity.get("autocast") is not False or identity.get("tf32") is not False):
        raise ValueError("Engineering checkpoint/episode/source/fit identities disagree")
    if set(identity.get("code", {})) != set(META_CODE_FILES):
        raise ValueError("The complete engineering implementation code manifest is required")
    for relative, expected_sha in identity["code"].items():
        if file_digest(local_path(root, relative)) != expected_sha:
            raise ValueError("Engineering implementation changed after the measured fit")
    if not set(META_RUNTIME_KEYS) <= set(identity):
        raise ValueError("The complete writer-shaped engineering runtime record is required")
    recorded_runtime = {key: identity[key] for key in META_RUNTIME_KEYS}
    if (not _runtime_valid(recorded_runtime)
            or proof.get("runtime_profile") != recorded_runtime
            or runtime_profile is not None and runtime_profile != recorded_runtime):
        raise ValueError("Engineering and current runtime/device/code profiles differ")
    updates = state.get("update_receipts", [])
    matrix = state.get("parameter")
    dimension_rows = 10 if engineering.method in {"B10", "B11"} else 99
    if (state.get("outer_cursor") != 100 or state.get("validation") != history or len(updates) != 100
            or [row.get("outer_update") for row in updates] != list(range(1, 101))
            or any(row.get("full_unroll") is not True or type(row.get("inner_updates_actual")) is not int
                   or not 0 <= row["inner_updates_actual"] <= 40
                   or not _positive(row.get("gpu_stream_elapsed_ms"))
                   or not _positive(row.get("end_to_end_wall_seconds"))
                   or type(row.get("max_memory_allocated_bytes")) is not int or row["max_memory_allocated_bytes"] <= 0
                   or type(row.get("max_memory_reserved_bytes")) is not int
                   or not row["max_memory_allocated_bytes"] <= row["max_memory_reserved_bytes"] <= recorded_runtime["device_total_memory_bytes"]
                   for row in updates)
            or proof.get("inner_updates_actual") != sum(row["inner_updates_actual"] for row in updates)
            or not _positive(state.get("active_wall_seconds")) or not _positive(proof.get("active_wall_seconds"))
            or proof["active_wall_seconds"] < state["active_wall_seconds"]
            or not isinstance(matrix, torch.Tensor) or matrix.dtype != torch.float64
            or matrix.shape != (dimension_rows, engineering.dimension) or not torch.isfinite(matrix).all()
            or not torch.allclose(matrix.T @ matrix, torch.eye(engineering.dimension, dtype=torch.float64), atol=1e-10, rtol=1e-10)):
        raise ValueError("Engineering final state needs 100 measured full-unroll updates, validation and a valid parameter")
    return proof
