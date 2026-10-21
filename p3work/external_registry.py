"""Offline qualification of public metadata; never an external-data loader.

The fixed pilot archive and original protocol are identified, not opened. Only
small, explicitly listed public metadata evidence files are read. Passing this
validator does not authorize raw data, query access, fitting, or evaluation.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import urlsplit


SCHEMA = "paper3-external-expansion-registry-v1"
INPUT_SHA256 = "5c8ea3f01634c060546af29b9330d314c4f03c265677a74b875ac7c302e31872"
PROTOCOL_NAME = "Paper3_Work_Handoff_20261018.zip"
PROTOCOL_SHA256 = "0398664faf34682a07fb38de4e46df9829f0e5ad050f657f905d0373dbfaf175"
PROTOCOL_BYTES = 529763
BASE_NAME = "paper3-controlled-pilot-ready-b5fb2b424951eb16.tar.gz"
BASE_SHA256 = "b5fb2b424951eb16e7ccf7fa807b642384d48148665db388f39a7194ea6003c1"
BASE_BYTES = 32795394
EVIDENCE_ROOT = "audit/external_expansion_20261019/"
MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_EVIDENCE_BYTES = 64 * 1024 * 1024
ROLES = frozenset({"candidate_locked_secondary", "montage_sensitivity_reserve",
    "exposed_exploratory", "access_blocked", "incompatible",
    "duplicate_of_existing", "unresolved_schema"})
PRIOR_EXPOSURE = frozenset({"unexposed", "prior_project_exposed", "unknown"})
CHANNEL_STATES = frozenset({"nominal_only", "metadata_compatible", "incompatible", "unresolved"})
CHANNELS = ("FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4")
KNOWN_PROJECT_EXPOSURE = frozenset({"Dreyer2023", "Kumar2024", "Yang2025_2C"})
METADATA_SUFFIXES = frozenset({".json", ".txt", ".md", ".html", ".xml", ".csv", ".tsv", ".pdf"})
FALSE_FLAGS = frozenset({"training_started", "external_payload_downloaded", "external_query_opened",
    "selection_use_allowed", "payload_access_allowed", "actual_training_or_evaluation_claims",
    "evaluation_ready", "scientific_training_ready", "raw_schema_verified", "external_fit_completed",
    "query_opened", "payload_downloaded", "fit_completed", "training_completed", "evaluation_completed",
    "biological_identity_independence_verified"})
NULL_COUNTS = frozenset({"eligible_subjects", "actual_trials", "actual_bytes",
    "actual_download_bytes", "actual_eligible_participants", "actual_evaluation_participants"})
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")


def safe_relative(value: object) -> str:
    """Return a canonical POSIX relative path, rejecting aliases and traversal."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("path must be a nonempty POSIX relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(value).is_absolute():
        raise ValueError("absolute, traversal, or noncanonical path")
    return value


def read_metadata(root: Path, relative: str, expected_bytes: int | None = None) -> bytes:
    """Read a bounded regular metadata file without following any symlink."""
    relative = safe_relative(relative)
    if Path(relative).suffix.lower() not in METADATA_SUFFIXES:
        raise ValueError("not an allowed public metadata suffix")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    directory_fd = os.open(root, flags | os.O_DIRECTORY)
    fd = None
    try:
        parts = relative.split("/")
        for component in parts[:-1]:
            next_fd = os.open(component, flags | os.O_DIRECTORY, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=directory_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_EVIDENCE_BYTES:
            raise ValueError("metadata is not a bounded regular file")
        if expected_bytes is not None and info.st_size != expected_bytes:
            raise ValueError("metadata byte count mismatch")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, MAX_EVIDENCE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_EVIDENCE_BYTES:
                raise ValueError("metadata exceeds byte limit")
        if total != info.st_size or (expected_bytes is not None and total != expected_bytes):
            raise ValueError("metadata changed while being read")
        return b"".join(chunks)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory_fd)


def _public_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            return False
        host = url.hostname.lower().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return "." in host
    except ValueError:
        return False


def validate_registry(registry: object, root: Path | str) -> dict:
    """Return a fail-closed JSON-safe report, without writes or network calls."""
    errors = []
    verified = []
    selected = []
    root = Path(root).resolve()

    def require(condition, where, reason):
        if not condition:
            errors.append({"field": where, "reason": reason})

    def exact(mapping, key, value, where="registry"):
        observed = mapping.get(key)
        require(key in mapping and type(observed) is type(value) and observed == value,
                f"{where}.{key}", "required immutable value differs or is absent")

    def inspect_claims(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in FALSE_FLAGS:
                    require(item is False, f"{where}.{key}", "metadata-only permission or claim must be false")
                if key in NULL_COUNTS:
                    require(item is None, f"{where}.{key}", "actual count is unknown before raw qualification")
                inspect_claims(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect_claims(item, f"{where}[{index}]")

    def frozen_descriptor(value, name, sha256, byte_count, where):
        if not isinstance(value, dict):
            require(False, where, "immutable artifact descriptor is required")
            return
        exact(value, "sha256", sha256, where)
        exact(value, "bytes", byte_count, where)
        path = value.get("path")
        require(isinstance(path, str) and Path(path).name == name and ".." not in PurePosixPath(path).parts
                and "\\" not in path and "\x00" not in path,
                f"{where}.path", "immutable artifact filename or path is unsafe")

    if not isinstance(registry, dict):
        registry = {}
        require(False, "registry", "JSON object required")
    exact(registry, "schema_version", SCHEMA)
    exact(registry, "scope", "METADATA_ONLY_QUALIFICATION")
    exact(registry, "input_contract_sha256", INPUT_SHA256)
    exact(registry, "W7_protocol_freeze_sha256", None)
    for key in ("training_started", "external_payload_downloaded", "external_query_opened",
                "actual_training_or_evaluation_claims"):
        exact(registry, key, False)
    frozen_descriptor(registry.get("original_protocol_archive"), PROTOCOL_NAME, PROTOCOL_SHA256,
                      PROTOCOL_BYTES, "registry.original_protocol_archive")
    frozen_descriptor(registry.get("base_delivery"), BASE_NAME, BASE_SHA256, BASE_BYTES,
                      "registry.base_delivery")
    hypothesis = registry.get("primary_hypothesis")
    if not isinstance(hypothesis, dict):
        hypothesis = {}
    for key, value in {"dataset_id": "HSStroke2026_v6", "contrast": "P_vs_Bstar", "budget": 20}.items():
        exact(hypothesis, key, value, "registry.primary_hypothesis")
    signal = registry.get("signal_contract")
    if not isinstance(signal, dict):
        signal = {}
    for key, value in {"channels_required": 11, "sampling_hz": 128, "context_seconds": 1.5,
                       "crop_seconds": [0.25, 1.25]}.items():
        exact(signal, key, value, "registry.signal_contract")
    inspect_claims(registry, "registry")
    prefix = registry.get("evidence_prefix")
    valid_prefix = isinstance(prefix, str) and prefix.startswith(EVIDENCE_ROOT) and prefix.endswith("/")
    require(valid_prefix,
            "registry.evidence_prefix", "public evidence must remain in the external expansion audit directory")
    if isinstance(prefix, str):
        try:
            safe_relative(prefix.rstrip("/"))
        except ValueError:
            valid_prefix = False
            require(False, "registry.evidence_prefix", "unsafe evidence directory")
    evidence = registry.get("all_evidence")
    require(isinstance(evidence, list) and bool(evidence) and len(evidence) <= 1000,
            "registry.all_evidence", "one to 1000 explicit public evidence descriptors are required")
    evidence = evidence if isinstance(evidence, list) else []
    descriptors = {}
    total_bytes = 0
    for index, descriptor in enumerate(evidence):
        where = f"registry.all_evidence[{index}]"
        if not isinstance(descriptor, dict):
            require(False, where, "evidence descriptor object required")
            continue
        path = descriptor.get("path")
        sha256 = descriptor.get("sha256")
        byte_count = descriptor.get("bytes")
        valid_path = False
        try:
            safe_relative(path)
            valid_path = valid_prefix and path.startswith(prefix)
        except ValueError:
            pass
        require(valid_path, f"{where}.path", "evidence path must be safe and under the declared audit prefix")
        require(isinstance(sha256, str) and SHA_RE.fullmatch(sha256) is not None,
                f"{where}.sha256", "lowercase SHA256 required")
        valid_bytes = type(byte_count) is int and 0 <= byte_count <= MAX_EVIDENCE_BYTES
        require(valid_bytes, f"{where}.bytes", "bounded metadata byte count required")
        require(_public_url(descriptor.get("url")), f"{where}.url", "public HTTP(S) provenance URL required")
        if isinstance(path, str):
            require(path not in descriptors, f"{where}.path", "duplicate evidence path")
            descriptors[path] = descriptor
        if valid_bytes:
            total_bytes += byte_count
        if valid_path and valid_bytes and total_bytes <= MAX_TOTAL_EVIDENCE_BYTES:
            try:
                content = read_metadata(root, path, byte_count)
                require(hashlib.sha256(content).hexdigest() == sha256, f"{where}.sha256", "evidence SHA256 mismatch")
                if hashlib.sha256(content).hexdigest() == sha256:
                    verified.append(path)
            except (ValueError, OSError):
                require(False, where, "evidence is missing, unsafe, nonregular, oversized, or has different bytes")
    require(total_bytes <= MAX_TOTAL_EVIDENCE_BYTES, "registry.all_evidence", "total metadata evidence exceeds limit")
    datasets = registry.get("datasets")
    require(isinstance(datasets, list) and bool(datasets), "registry.datasets", "nonempty candidate list required")
    datasets = datasets if isinstance(datasets, list) else []
    seen = set()
    for index, dataset in enumerate(datasets):
        where = f"registry.datasets[{index}]"
        if not isinstance(dataset, dict):
            require(False, where, "dataset object required")
            continue
        dataset_id = dataset.get("dataset_id")
        valid_id = isinstance(dataset_id, str) and ID_RE.fullmatch(dataset_id) is not None
        require(valid_id, f"{where}.dataset_id", "canonical dataset identifier required")
        if valid_id:
            require(dataset_id not in seen, f"{where}.dataset_id", "duplicate dataset identifier")
            seen.add(dataset_id)
        role = dataset.get("role")
        require(isinstance(role, str) and role in ROLES, f"{where}.role", "controlled scientific role required")
        require(not valid_id or dataset_id not in {"HSStroke2026_v6", "Wang2026", "Stieger2021", "BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI", "Yang2025_2C"}
                or role == "duplicate_of_existing", f"{where}.role", "existing primary, replication, development, and source cohorts cannot be relabeled as new secondary data")
        if role == "candidate_locked_secondary" and valid_id:
            selected.append(dataset_id)
        cohort = dataset.get("cohort_reported")
        require(cohort is None or (type(cohort) is int and cohort > 0), f"{where}.cohort_reported", "reported cohort is a positive nominal count or null")
        exact(dataset, "eligible_subjects", None, where)
        exact(dataset, "selection_use_allowed", False, where)
        exact(dataset, "payload_access_allowed", False, where)
        prior = dataset.get("prior_exposure")
        if not isinstance(prior, dict):
            prior = {}
        prior_status = prior.get("status")
        require(isinstance(prior_status, str) and prior_status in PRIOR_EXPOSURE,
                f"{where}.prior_exposure.status", "explicit project exposure history required")
        require(isinstance(prior.get("details"), str) and bool(prior["details"].strip()),
                f"{where}.prior_exposure.details", "exposure history explanation required")
        if role == "candidate_locked_secondary":
            require(prior_status in {"unexposed", "unknown"} if isinstance(prior_status, str) else False,
                    f"{where}.prior_exposure.status", "prior project exposed cohorts cannot be registered as secondary replication candidates")
            if prior_status == "unknown":
                gates = dataset.get("remaining_gates")
                require(isinstance(gates, list) and "prior_project_exposure_audit" in gates,
                        f"{where}.prior_exposure.status", "unknown exposure requires the explicit prior_project_exposure_audit gate; candidate status is not qualification")
        if valid_id and dataset_id in KNOWN_PROJECT_EXPOSURE:
            require(prior_status == "prior_project_exposed", f"{where}.prior_exposure.status", "recorded Paper1/Paper2 exposure must be preserved")
        if role == "exposed_exploratory":
            require(prior_status == "prior_project_exposed", f"{where}.prior_exposure.status", "exploratory exposure role requires recorded prior exposure")
        own_evidence = dataset.get("evidence")
        require(isinstance(own_evidence, list) and bool(own_evidence), f"{where}.evidence", "dataset-specific evidence required")
        own_paths = set()
        for number, item in enumerate(own_evidence if isinstance(own_evidence, list) else []):
            matched = isinstance(item, dict) and isinstance(item.get("path"), str) and descriptors.get(item["path"]) == item
            require(matched, f"{where}.evidence[{number}]", "evidence descriptor must match the complete verified evidence registry")
            if matched:
                own_paths.add(item["path"])
        channel = dataset.get("channel_qualification")
        if not isinstance(channel, dict):
            channel = {}
        exact(channel, "required", 11, f"{where}.channel_qualification")
        state = channel.get("status")
        matched = channel.get("matched_count")
        matched_channels = channel.get("matched_channels")
        require(isinstance(state, str) and state in CHANNEL_STATES, f"{where}.channel_qualification.status", "controlled montage qualification status required")
        require(matched is None or (type(matched) is int and 0 <= matched <= 11),
                f"{where}.channel_qualification.matched_count", "matched canonical channel count must be null or 0..11")
        if state in {"nominal_only", "unresolved"}:
            require(matched is None, f"{where}.channel_qualification.matched_count", "nominal amplifier counts do not establish canonical channel matches")
            exact(channel, "matched_channels", None, f"{where}.channel_qualification")
        if matched is not None:
            valid_names = isinstance(matched_channels, list) and all(isinstance(name, str) and name in CHANNELS for name in matched_channels)
            require(valid_names and len(matched_channels) == matched
                    and matched_channels == [name for name in CHANNELS if name in matched_channels],
                    f"{where}.channel_qualification.matched_channels", "explicit canonical ordered unique channel matches are required; neighbours are not substitutes")
        if state == "metadata_compatible":
            require(type(matched) is int and matched == 11, f"{where}.channel_qualification.matched_count", "all 11 canonical channels require metadata evidence")
        if state == "incompatible":
            require(type(matched) is int and 0 <= matched < 11, f"{where}.channel_qualification.matched_count", "incompatible montage requires a documented missing-channel count")
            require(role != "candidate_locked_secondary", f"{where}.role", "incompatible montage belongs to a separate sensitivity or incompatible role")
        provenance = channel.get("provenance_evidence_paths")
        require(isinstance(provenance, list) and all(isinstance(p, str) and p in own_paths for p in provenance),
                f"{where}.channel_qualification.provenance_evidence_paths", "channel provenance must reference this dataset's verified evidence")
        if matched is not None:
            require(isinstance(provenance, list) and bool(provenance), f"{where}.channel_qualification.provenance_evidence_paths", "channel matching requires provenance")
        gates = dataset.get("remaining_gates")
        require(isinstance(gates, list) and bool(gates) and all(isinstance(g, str) and bool(g.strip()) for g in gates),
                f"{where}.remaining_gates", "unresolved raw/schema/W7/evaluation gates must be explicit")
    family = registry.get("secondary_analysis")
    if not isinstance(family, dict):
        family = {}
    for key, value in {"comparison": "P_vs_Bstar", "budget": 20, "unit": "participant", "multiplicity": "Holm"}.items():
        exact(family, key, value, "registry.secondary_analysis")
    declared = family.get("pvalue_family_datasets")
    require(isinstance(declared, list) and all(isinstance(item, str) for item in declared)
            and declared == sorted(set(selected)), "registry.secondary_analysis.pvalue_family_datasets",
            "one predeclared sorted complete family must include every locked secondary cohort exactly once")
    return {"schema_version": "paper3-external-registry-check-v1", "registry_valid": not errors,
            "status": "PREPARED_CANDIDATE_METADATA_ONLY" if not errors else "INVALID_METADATA_REGISTRY",
            "evaluation_ready": False, "evaluation_blockers": ["W7 protocol freeze absent",
                "raw external adapter/evaluator transaction not implemented or qualified",
                "external payload and query access remain locked"],
            "dataset_records": len(datasets), "secondary_family": sorted(set(selected)),
            "verified_metadata_files": len(set(verified)), "declared_metadata_bytes": total_bytes,
            "eligible_external_subjects": None, "training_started": False,
            "external_payload_downloaded": False, "external_query_opened": False,
            "network_actions": False, "evidence_payload_scope": "public metadata only; immutable archive descriptors were not opened",
            "errors": errors}
