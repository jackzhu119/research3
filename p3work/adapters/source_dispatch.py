"""Explicit source-reader revisions; the frozen source_native stays unchanged."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from p3work.contracts import digest, file_digest
from . import source_native as legacy

REVISION_ID = "physionet_original_tal_inward_v1"
REVISION_SCHEMA = "paper3-source-native-reader-revision-v1"
FROZEN_BASE_SHA256 = "886568fd24782856c42a3c5f0038d03486a1132292bf88a2eeef4d474a084a9e"


def current_revision():
    """Actual component file digests; no descriptor is embedded in its source."""
    folder = Path(__file__).resolve().parent
    def component(filename):
        path = folder / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("Reader revision requires actual regular code files")
        return {"path": "p3work/adapters/" + filename, "sha256": file_digest(path), "bytes": path.stat().st_size}
    base = component("source_native.py")
    if base["sha256"] != FROZEN_BASE_SHA256:
        raise ValueError("Frozen source_native bytes differ from the original source plan")
    result = {"schema_version": REVISION_SCHEMA, "revision_id": REVISION_ID, "dataset": "PhysionetMI",
        "base_adapter": base, "variant_reader": component("physionet_tal_v1.py"),
        "dispatcher": component("source_dispatch.py"),
        "sampling_policy": "exact_decimal_TAL; ceil_onset; floor_min_annotation_end_next_onset_recording_end",
        "original_annotation_identity_preserved": True, "waveform_padding_permitted": False}
    result["revision_sha256"] = digest(result)
    return result


def validate_revision(revision):
    if type(revision) is not dict or digest(revision) != digest(current_revision()):
        raise ValueError("Explicit current source-reader revision with exact component SHAs required")
    return deepcopy(revision)


def audit_native_source(dataset, path, *, provider_subject, session_id, source_version,
                        expected_sha256, revision=None):
    """No revision selects the original reader, including historical Physio."""
    if revision is not None:
        revision = validate_revision(revision)
    if dataset != "PhysionetMI" or revision is None:
        return legacy.audit_native_source(dataset, path, provider_subject=provider_subject,
            session_id=session_id, source_version=source_version, expected_sha256=expected_sha256)
    from .physionet_tal_v1 import audit_native_physionet
    return audit_native_physionet(path, provider_subject=provider_subject, session_id=session_id,
        source_version=source_version, expected_sha256=expected_sha256, revision=revision)


def iter_native_contexts(dataset, path, *, rows, contract, expected_sha256, revision=None):
    """Recompute metadata with its actual reader; never upgrade old rows."""
    row_revisions = [r.get("native_reader_revision") for r in rows]
    if any(r is not None for r in row_revisions):
        selected = validate_revision(row_revisions[0])
        if dataset != "PhysionetMI" or any(digest(r) != digest(selected) for r in row_revisions):
            raise ValueError("All native rows must carry the same registered Physio revision")
        if revision is not None and digest(validate_revision(revision)) != digest(selected):
            raise ValueError("Requested reader revision differs from original native rows")
        from .physionet_tal_v1 import iter_physionet_contexts
        yield from iter_physionet_contexts(path, rows=rows, contract=contract,
            expected_sha256=expected_sha256, revision=selected)
    else:
        if revision is not None and dataset == "PhysionetMI":
            raise ValueError("Legacy Physio rows cannot be relabelled as a new reader revision")
        if revision is not None:
            validate_revision(revision)
        yield from legacy.iter_native_contexts(dataset, path, rows=rows, contract=contract,
            expected_sha256=expected_sha256)
