"""Verified learned-adapter artifacts, separate from raw mathematical kernels.

An arbitrary orthogonal tensor is useful for a software test, but is never a
completed P/B8/B10/B11 model. This module binds real training, selected snapshot,
source model, reference geometry, split, input, method, config and pipeline seed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch

from .adapt import CalibrationPrefix, InnerConfig, fit_target
from .contracts import digest, file_digest, validate_signal_contract
from .f0_bridge import load_verified_source
from .meta_training import META_CODE_FILES, MetaJob
from .selection import choose_checkpoint
from .spatial_geometry import helmert_rows
from .source_reference_io import load_verified_reference
from .training_state import atomic_json, local_path


@dataclass(frozen=True)
class LearnedMetaParameter:
    matrix: torch.Tensor
    job: MetaJob
    input_contract_sha256: str
    split_sha256: str
    source_model_state_sha256: str
    geometry_sha256: str
    receipt_path: str
    receipt_sha256: str
    selected_outer_update: int


def load_learned_parameter(root, receipt_path, *, expected_method, expected_seed,
                           input_contract_sha256, split_sha256, source, geometry):
    root = Path(root).resolve(strict=True)
    path = local_path(root, receipt_path)
    value = json.loads(path.read_text())
    job = MetaJob(**value["job"])
    if (value.get("status") != "META_JOB_COMPLETE" or value.get("synthetic") is not False
            or value.get("scientific_fit_completed") is not True or value.get("full_unroll") is not True
            or value.get("source_state_unchanged") is not True or value.get("outer_updates_completed") != 1000
            or value.get("validation_checkpoints") != 20 or job.phase == "engineering"
            or job.method != expected_method or job.seed != expected_seed):
        raise ValueError("A completed independent same-method/same-seed meta fit is required")
    bindings = ("signal_contract_sha256", "split_sha256", "source_model_state_sha256", "geometry_sha256",
                "source_receipt_sha256", "reference_receipt_sha256", "stieger_manifest_sha256")
    if (any(re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, ""))) is None for key in bindings)
            or value.get("signal_contract_sha256") != input_contract_sha256 or value.get("split_sha256") != split_sha256
            or value.get("source_model_state_sha256") != source.frozen_state_sha256
            or value.get("geometry_sha256") != geometry.sha256()):
        raise ValueError("Learned adapter source/reference/split/input identity mismatch")
    if not torch.allclose(geometry.q, helmert_rows(device=geometry.q.device), atol=1e-14, rtol=0):
        raise ValueError("The registered learned adapter uses the fixed Helmert Q")
    history = value.get("validation", [])
    if [row["outer_update"] for row in history] != list(range(50, 1001, 50)):
        raise ValueError("All twenty predeclared validation checkpoints must be retained")
    selected = choose_checkpoint(history)
    if value.get("selected_checkpoint") != selected:
        raise ValueError("Selected snapshot does not follow BA/BCE/earlier-checkpoint rule")
    artifact = value.get("selected_parameter_checkpoint", {})
    checkpoint = local_path(root, artifact.get("path", ""))
    if file_digest(checkpoint) != artifact.get("sha256"):
        raise ValueError("Learned parameter checkpoint SHA256 mismatch")
    sidecar_path = local_path(root, checkpoint.with_suffix(checkpoint.suffix + ".sha256.json").relative_to(root).as_posix())
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("sha256") != artifact["sha256"] or sidecar.get("bytes") != checkpoint.stat().st_size:
        raise ValueError("Selected meta checkpoint has a torn/mismatched full-SHA sidecar")
    # Loading for inference must not restore RNG, unlike a training resume.
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = record.get("identity", {})
    if (record.get("schema_version") != "paper3-trusted-training-state-v1"
            or record.get("identity_sha256") != digest(identity)
            or value.get("identity_sha256") != digest(identity) or sidecar.get("identity_sha256") != digest(identity)
            or identity.get("job") != value["job"]
            or any(identity.get(k) != value.get(k) for k in bindings)):
        raise ValueError("Meta checkpoint and completed run identities disagree")
    episode_descriptor = value.get("episode_plan", {})
    episode_path = local_path(root, episode_descriptor.get("path", ""))
    if (file_digest(episode_path) != episode_descriptor.get("sha256")
            or identity.get("episode_plan_sha256") != episode_descriptor["sha256"]):
        raise ValueError("Complete pre-frozen real episode plan artifact is missing or changed")
    if set(identity.get("code", {})) != set(META_CODE_FILES):
        raise ValueError("Complete versioned meta-method dependency code manifest required")
    for relative, expected in identity["code"].items():
        if file_digest(local_path(root, relative)) != expected:
            raise ValueError("Learned method implementation changed after its fit")
    state = record["state"]
    matrix = state.get("parameter")
    rows = 10 if job.method in {"B10", "B11"} else 99
    if (state.get("outer_cursor") != selected["outer_update"] or not isinstance(matrix, torch.Tensor)
            or matrix.dtype != torch.float64 or matrix.shape != (rows, job.dimension)
            or not torch.isfinite(matrix).all()
            or not torch.allclose(matrix.T @ matrix, torch.eye(job.dimension, dtype=torch.float64), atol=1e-10, rtol=1e-10)):
        raise ValueError("Invalid selected learned Stiefel parameter")
    if state.get("validation") != [row for row in history if row["outer_update"] <= selected["outer_update"]]:
        raise ValueError("Selected checkpoint validation history differs from its completed run")
    # The selected snapshot can precede most validation observations. Bind all
    # twenty observations to the final full-SHA state before trusting selection.
    final = value.get("checkpoint", {})
    if (final.get("relative_to_job_output") is not True or final.get("path") != "outer-1000.pt"
            or final.get("outer_cursor") != 1000):
        raise ValueError("The completed final 1000-update checkpoint is required")
    final_path = local_path(root, (path.parent.relative_to(root) / final["path"]).as_posix())
    final_sidecar_path = local_path(root, final_path.with_suffix(final_path.suffix + ".sha256.json").relative_to(root).as_posix())
    final_sidecar = json.loads(final_sidecar_path.read_text())
    if (file_digest(final_path) != final.get("sha256") or final_sidecar.get("sha256") != final["sha256"]
            or final_sidecar.get("bytes") != final_path.stat().st_size
            or final_sidecar.get("identity_sha256") != digest(identity)):
        raise ValueError("Final meta checkpoint full SHA/bytes/identity mismatch")
    final_record = torch.load(final_path, map_location="cpu", weights_only=True)
    final_state = final_record.get("state", {})
    final_updates = final_state.get("update_receipts", [])
    final_matrix = final_state.get("parameter")
    if (final_record.get("schema_version") != "paper3-trusted-training-state-v1"
            or final_record.get("identity") != identity or final_record.get("identity_sha256") != digest(identity)
            or final_state.get("outer_cursor") != 1000 or final_state.get("validation") != history
            or len(final_updates) != 1000 or [row.get("outer_update") for row in final_updates] != list(range(1, 1001))
            or any(row.get("full_unroll") is not True or type(row.get("inner_updates_actual")) is not int
                   or not 0 <= row["inner_updates_actual"] <= 40 for row in final_updates)
            or not isinstance(final_matrix, torch.Tensor) or final_matrix.dtype != torch.float64
            or final_matrix.shape != (rows, job.dimension) or not torch.isfinite(final_matrix).all()
            or not torch.allclose(final_matrix.T @ final_matrix, torch.eye(job.dimension, dtype=torch.float64), atol=1e-10, rtol=1e-10)):
        raise ValueError("Final meta checkpoint trajectory/validation/Stiefel evidence differs from its completed receipt")
    source.assert_unchanged()
    return LearnedMetaParameter(matrix.detach(), job, input_contract_sha256, split_sha256,
        source.frozen_state_sha256, geometry.sha256(), path.relative_to(root).as_posix(), file_digest(path), selected["outer_update"])


def fit_verified_meta_method(root, learned, source, geometry, support):
    """Official target path; raw random U0/B tensors cannot enter this wrapper."""
    if type(learned) is not LearnedMetaParameter or type(support) is not CalibrationPrefix:
        raise TypeError("Verified learned artifact and calibration-prefix capabilities required")
    current = load_learned_parameter(root, learned.receipt_path, expected_method=learned.job.method,
        expected_seed=learned.job.seed, input_contract_sha256=learned.input_contract_sha256,
        split_sha256=learned.split_sha256, source=source, geometry=geometry)
    if current.receipt_sha256 != learned.receipt_sha256 or support.protocol_freeze_sha != learned.input_contract_sha256:
        raise ValueError("Calibration input or learned receipt changed")
    if support.role != "calibration_prefix":
        raise PermissionError("Production fitting only accepts its acquired calibration prefix")
    device = next(source.parameters()).device
    matrix = current.matrix.to(device)
    job = current.job
    if job.method in {"B10", "B11"}:
        from .methods.lowrank import LowRankConfig, fit_lowrank_target
        return fit_lowrank_target(source, geometry, support, method=job.method,
            config=LowRankConfig(rank=job.dimension, radius=job.radius, pct_weight=.1 if job.method == "B11" else 0),
            seed=job.seed, u0=matrix)
    return fit_target(source, geometry, matrix, support, method=job.method,
        config=InnerConfig(radius=job.radius, pct_weight=.1 if job.method == "P+PCT" else 0))


def freeze_development_configuration(root, receipt_paths, *, method, output, source_receipt,
                                     reference_receipt, contract, split_sha256, execute=False):
    """Choose only from nine completed screening fits; never consumes test data."""
    root = Path(root).resolve(strict=True)
    signal_sha = validate_signal_contract(contract)
    if split_sha256 != contract["window_audit_receipt"]["split_sha256"]:
        raise ValueError("Screening selection split differs from the frozen input window audit")
    if len(receipt_paths) != 9 or len(set(receipt_paths)) != 9:
        raise ValueError("All nine distinct screening receipts are required")
    descriptors = []
    for item in (source_receipt, reference_receipt):
        if type(item) is not dict or not {"path", "sha256"} <= set(item):
            raise ValueError("Screening source/reference need independently SHA-bound artifact descriptors")
        path = local_path(root, item["path"])
        if file_digest(path) != item["sha256"]:
            raise ValueError("Screening source/reference receipt SHA mismatch")
        descriptors.append({"path": path.relative_to(root).as_posix(), "sha256": item["sha256"]})
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        source, source_info = load_verified_source(root, descriptors[0]["path"], contract)
        reference, _ = load_verified_reference(root, descriptors[1]["path"], descriptors[0]["path"], contract)
    if source_info.get("seed") != 20261018 or source_info.get("pipeline_seed") != 20261018:
        raise ValueError("The nine-grid search must use its actual screening-seed source f0")
    from .spatial_geometry import SpatialGeometry
    geometry = SpatialGeometry.from_reference(reference.matrix)
    records, seen, identities = [], set(), set()
    for relative in receipt_paths:
        path = local_path(root, relative)
        value = json.loads(path.read_text())
        job = MetaJob(**value["job"])
        if (job.method != method or job.phase != "screening" or job.seed != 20261018
                or value.get("status") != "META_JOB_COMPLETE" or value.get("synthetic") is not False
                or value.get("scientific_fit_completed") is not True or value.get("outer_updates_completed") != 1000
                or value.get("validation_checkpoints") != 20):
            raise ValueError("Every screening fit must actually be complete on the development seed")
        load_learned_parameter(root, relative, expected_method=method, expected_seed=20261018,
            input_contract_sha256=signal_sha, split_sha256=split_sha256, source=source, geometry=geometry)
        if (value.get("source_receipt_sha256") != descriptors[0]["sha256"]
                or value.get("reference_receipt_sha256") != descriptors[1]["sha256"]):
            raise ValueError("Every screening fit must bind the verified screening source/reference receipts")
        if [r["outer_update"] for r in value.get("validation", [])] != list(range(50, 1001, 50)):
            raise ValueError("Incomplete registered validation schedule")
        best = choose_checkpoint(value["validation"])
        if value.get("selected_checkpoint") != best:
            raise ValueError("Screening fit selected the wrong validation checkpoint")
        checkpoint = value.get("selected_parameter_checkpoint", {})
        selected_path = local_path(root, checkpoint.get("path", ""))
        if file_digest(selected_path) != checkpoint.get("sha256"):
            raise ValueError("Screening selected weights are missing or changed")
        seen.add(job.identity)
        identities.add(tuple(value.get(k) for k in ("signal_contract_sha256", "split_sha256",
            "source_receipt_sha256", "reference_receipt_sha256", "stieger_manifest_sha256")))
        records.append((job, best, {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}))
    from .meta_training import search_plan
    if seen != {j["job_sha256"] for j in search_plan(method)["jobs"]} or len(identities) != 1:
        raise ValueError("Frozen search must use the exact same nine source/data/split configurations")
    # Checkpoint tie rules first; remaining exact configuration ties choose
    # smaller registered dimension and then radius, an explicit stable ordering.
    common = next(iter(identities))
    selected_job, selected_metrics, _ = min(records, key=lambda x: (-x[1]["participant_mean_ba"],
        x[1]["participant_mean_bce"], x[1]["outer_update"], x[0].dimension, x[0].radius))
    result = {"status": "META_CONFIGURATION_FROZEN" if execute else "PLANNED_SELECTION_NOT_WRITTEN",
              "method": method, "dimension": selected_job.dimension, "radius": selected_job.radius,
              "selection_role": "meta_validation", "screening_seed": 20261018,
              "signal_contract_sha256": common[0], "split_sha256": common[1], "selected_validation": selected_metrics,
              "screening_source_receipt": descriptors[0], "screening_reference_receipt": descriptors[1],
              "tie_order": "participant BA descending, BCE ascending, earlier checkpoint, dimension, radius",
              "screening_receipts": [r[2] for r in records], "internal_or_external_access": False}
    if execute:
        target = local_path(root, output)
        if target.exists() and json.loads(target.read_text()) != result:
            raise ValueError("Existing frozen development selection cannot be overwritten")
        atomic_json(target, result)
    return result
