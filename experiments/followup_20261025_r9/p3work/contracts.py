"""Explicit data permissions and content identities; no training or downloads."""
from hashlib import sha256
import json
import re
from pathlib import Path
from .window_amendment import is_v21_contract, validate_v21_contract

SOURCE_IDS = frozenset(("BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI"))
DEVELOPMENT_ID = "Stieger2021"
EXTERNAL_IDS = frozenset(("HSStroke2026_v6", "Wang2026"))
CHANNELS = ("FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def permit_preparation(dataset, role):
    """Current user authorization excludes external payloads and internal evaluation."""
    allowed = ((dataset in SOURCE_IDS and role in {"source_schema", "source_train"}) or
               (dataset == DEVELOPMENT_ID and role in {"development_schema", "meta_train_metadata"}))
    if not allowed:
        raise PermissionError("Dataset/partition is outside the current preparation authorization")


def validate_signal_contract(contract):
    if contract.get("status") != "window_frozen":
        raise ValueError("W1 window is not frozen; numerical caches and source fitting are blocked")
    if contract.get("channels") != list(CHANNELS) or contract.get("sfreq_hz") != 128:
        raise ValueError("Signal montage/sampling contract mismatch")
    if is_v21_contract(contract):
        # Structure only: the separate read-only window gate checks the pinned
        # approval/audit bytes and every original meta-train metadata object.
        validate_v21_contract(contract)
        return digest(contract)
    profile = contract.get("selected_profile")
    windows = {"long": ([0, 3], [0.5, 2.5], 256), "short": ([0, 2], [0.5, 1.5], 128)}
    if profile not in windows:
        raise ValueError("Unresolved window")
    context, crop, samples = windows[profile]
    if (contract.get("context_s"), contract.get("crop_s"), contract.get("n_times")) != (context, crop, samples):
        raise ValueError("Window fields disagree")
    receipt = contract.get("window_audit_receipt", {})
    if receipt.get("decision_pool_role") != "meta_train" or not receipt.get("split_sha256"):
        raise ValueError("Window needs split-bound meta-train-only coverage evidence")
    if not receipt.get("raw_metadata_manifest_sha256"):
        raise ValueError("Window needs raw metadata provenance")
    if any(re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(key, ""))) is None
           for key in ("split_sha256", "raw_metadata_manifest_sha256")):
        raise ValueError("Window provenance must contain concrete SHA256 references")
    long = receipt.get("long", {})
    short = receipt.get("short", {})
    for row in (long, short):
        for key in ("participant_mean_coverage", "participant_mean_abs_class_exclusion_gap"):
            if type(row.get(key)) not in (float, int) or not 0 <= row[key] <= 1:
                raise ValueError("Both candidate windows need measured coverage and class-gap summaries")
    def passes(row):
        return (isinstance(row.get("participant_mean_coverage"), (float, int)) and
                isinstance(row.get("participant_mean_abs_class_exclusion_gap"), (float, int)) and
                0.80 <= row["participant_mean_coverage"] <= 1 and
                0 <= row["participant_mean_abs_class_exclusion_gap"] <= 0.10)
    expected = "long" if passes(long) else "short" if passes(short) else None
    if profile != expected:
        raise ValueError("Window does not follow the predeclared coverage rule")
    return digest(contract)
