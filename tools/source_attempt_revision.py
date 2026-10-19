"""Explicit retries with immutable prior attempts and new audit/R2 paths.

The caller holds the existing worker lock and uses the original resource ledger.
This module neither downloads nor resets budgets. A registered retry starts with
fresh provider validators; quarantined old partials are never resumed as mirror
partials. Only the real process_file result is promoted to the canonical index.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from tools import source_preparation_batch as base


SCHEMA = "paper3-source-attempt-revision-v1"
IMPLEMENTATION_FILES = frozenset({
    "tools/source_attempt_revision.py",
    "tools/source_preparation_small_files.py",
    "tools/physionet_official_transport.py",
    "tools/source_preparation_batch.py",
    "p3work/adapters/source_native.py",
    "p3work/adapters/source_dispatch.py",
    "p3work/adapters/physionet_tal_v1.py",
})
_RUNTIME_ROOT = Path(__file__).resolve().parents[1]
_STATUSES = {"PENDING", "HEAD_VERIFIED", "RAW_VERIFIED", "METADATA_EXTRACTED",
             "BACKUP_VERIFIED", "FAILED", "SCHEMA_BLOCKED", "COMPLETE"}


def _checked(root, descriptor):
    if not isinstance(descriptor, dict):
        raise base.BatchError("retry_revision_descriptor_required")
    path = base.local_path(root, descriptor.get("path", ""))
    if not path.is_file() or base.file_digest(path) != descriptor.get("sha256"):
        raise base.BatchError("retry_revision_evidence_integrity_failed")
    return path


def _read(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        raise base.BatchError("retry_revision_state_or_registration_JSON_invalid") from None
    if not isinstance(value, dict):
        raise base.BatchError("retry_revision_state_or_registration_JSON_invalid")
    return value


class RevisionProcessor:
    """Process registered Physio failures in append-only revision directories.

    Other exact original candidates retain the original worker behavior. A
    revision indexes failed results as well as success, including hashes of its
    live staging files, so Range recovery cannot adopt an old or changed partial.
    The complete original state and quarantined partial remain immutable.
    """
    def __init__(self, registration_path, *, root, plan, contract_sha256):
        base.no_symlink_components(Path(root).absolute())
        self.root = Path(root).resolve(strict=True)
        base.validate_plan(plan)
        self.plan, self.contract_sha = deepcopy(plan), contract_sha256
        path = base.local_path(self.root, str(registration_path))
        self.registration = _read(path)
        self.descriptor = base.descriptor(self.root, path)
        self.batch = self.root / "audit/source_preparation/batches" / plan["plan_sha256"]
        self.revision_batch = self.batch / "attempt_revisions" / self.registration.get("revision_sha256", "")
        base.no_symlink_components(self.revision_batch)
        self.candidates = {entry["object_id"]: entry for entry in self.plan["files"]}
        self.retries = {}
        self._validate_registration()
        objects = self.registration.get("objects")
        if not isinstance(objects, list) or not 1 <= len(objects) <= 327:
            raise base.BatchError("retry_revision_nonempty_registered_failures_required")
        for item in objects:
            uid = item.get("object_id") if isinstance(item, dict) else None
            if uid not in self.candidates or uid in self.retries:
                raise base.BatchError("retry_revision_exact_unique_candidate_required")
            if self.candidates[uid]["dataset"] != "PhysionetMI":
                raise base.BatchError("retry_revision_PhysionetMI_failures_only")
            self.retries[uid] = item
            self._validate_prior(item)
            self._canonical(self.candidates[uid], item)

    def _validate_registration(self):
        record = _read(_checked(self.root, self.descriptor))
        unsigned = {k: v for k, v in record.items() if k != "revision_sha256"}
        if (record != self.registration or record.get("schema_version") != SCHEMA
                or record.get("revision_sha256") != base.digest(unsigned)
                or record.get("plan_sha256") != self.plan["plan_sha256"]
                or record.get("input_contract_sha256") != self.contract_sha
                or record.get("fresh_provider_validators_required") is not True
                or record.get("budgets_reset") is not False):
            raise base.BatchError("retry_revision_identity_or_policy_mismatch")
        descriptors = record.get("implementation_files")
        if (not isinstance(descriptors, list) or any(not isinstance(d, dict) for d in descriptors)
                or len(descriptors) != len(IMPLEMENTATION_FILES)
                or {d.get("path") for d in descriptors} != IMPLEMENTATION_FILES):
            raise base.BatchError("retry_revision_actual_implementation_binding_required")
        for descriptor in descriptors:
            _checked(self.root, descriptor)
            actual = base.local_path(_RUNTIME_ROOT, descriptor["path"])
            if not actual.is_file() or base.file_digest(actual) != descriptor["sha256"]:
                raise base.BatchError("retry_revision_actual_implementation_binding_required")

    def _validate_identity(self, state, entry):
        if (state.get("object_id") != entry["object_id"]
                or state.get("identity") != list(base.identity(entry))
                or state.get("plan_sha256") != self.plan["plan_sha256"]
                or state.get("input_contract_sha256") != self.contract_sha
                or state.get("status") not in _STATUSES):
            raise base.BatchError("retry_revision_real_result_identity_mismatch")

    def _validate_prior(self, item):
        entry = self.candidates[item["object_id"]]
        prior_path = _checked(self.root, item.get("prior_state"))
        if (prior_path == self.batch / "objects" / entry["object_id"] / "state.json"
                or prior_path.is_relative_to(self.revision_batch)):
            raise base.BatchError("retry_revision_prior_must_be_preserved_outside_active_state")
        prior = _read(prior_path)
        self._validate_identity(prior, entry)
        if prior.get("status") not in {"FAILED", "SCHEMA_BLOCKED"}:
            raise base.BatchError("retry_revision_only_exact_failed_prior_attempts_allowed")
        partial = item.get("quarantined_partial")
        if partial is not None:
            path = _checked(self.root, partial)
            raw = base.raw_staging_path(self.root, self.plan, entry)
            if path in {raw, raw.with_name(raw.name + ".partial")} or path.is_relative_to(self.revision_batch):
                raise base.BatchError("retry_revision_partial_must_be_preserved_outside_active_staging")

    def _staging(self, entry):
        raw = base.raw_staging_path(self.root, self.plan, entry)
        result = {}
        for name, path in (("raw", raw), ("partial", raw.with_name(raw.name + ".partial"))):
            base.no_symlink_components(path)
            if path.exists():
                if not path.is_file():
                    raise base.BatchError("retry_revision_staging_must_be_regular_file")
                result[name] = base.descriptor(self.root, path)
        return result

    def _index(self, result, entry, item, revised_path):
        return dict(result, attempt_revision=self.descriptor,
            prior_attempt_state=item["prior_state"],
            attempt_revision_state=base.descriptor(self.root, revised_path),
            attempt_revision_staging=self._staging(entry))

    def _canonical(self, entry, item):
        path = self.batch / "objects" / entry["object_id"] / "state.json"
        base.no_symlink_components(path)
        before = path.read_bytes()
        state = _read(path)
        revised_path = self.revision_batch / "objects" / entry["object_id"] / "state.json"
        base.no_symlink_components(revised_path)
        if base.file_digest(path) == item["prior_state"]["sha256"]:
            if revised_path.exists():
                raise base.BatchError("retry_revision_unindexed_state_requires_review")
            return before, state
        if (state.get("attempt_revision") != self.descriptor
                or state.get("prior_attempt_state") != item["prior_state"]
                or state.get("attempt_revision_state", {}).get("path") != revised_path.relative_to(self.root).as_posix()):
            raise base.BatchError("retry_revision_canonical_state_changed_without_provenance")
        _checked(self.root, state["attempt_revision_state"])
        result = _read(revised_path)
        self._validate_identity(result, entry)
        expected = self._index(result, entry, item, revised_path)
        if state != expected:
            # The cache worker durably records ownership before restoring a
            # previously archived raw. A kill after its atomic commit can make
            # that raw reappear without changing either source state. Validate
            # that exact exception read-only; the original source worker below
            # performs recovery and writes its actual revision state.
            if (result.get("status") != "COMPLETE"
                    or result.get("temporary_raw_owned") is not True
                    or result.get("raw_removed_after_verified_backup") is not True
                    or set(expected["attempt_revision_staging"]) != {"raw"}
                    or state != dict(expected, attempt_revision_staging={})):
                raise base.BatchError("retry_revision_canonical_or_staging_integrity_failed")
            try:
                from tools.prepare_source_cache import _verify_active_cache_restore
                _verify_active_cache_restore(self.root, dict(entry,
                    object_uid=entry["object_id"], path=result["raw_path"],
                    sha256=result["raw_integrity"]["sha256"], bytes=result["raw_integrity"]["bytes"],
                    backup_receipt=result["backup_receipt"], temporary_raw_owned=True,
                    raw_storage="verified_r2_only"))
            except Exception:
                raise base.BatchError("retry_revision_reappeared_raw_requires_active_cache_restore") from None
        return before, state

    def __call__(self, entry, **kwargs):
        if (not isinstance(entry, dict) or entry.get("object_id") not in self.candidates
                or entry != self.candidates[entry["object_id"]]):
            raise base.BatchError("retry_revision_exact_plan_entry_required")
        base.validate_entry(entry)
        if (Path(kwargs.get("root", "")).resolve() != self.root or kwargs.get("plan") != self.plan
                or kwargs.get("contract_sha256") != self.contract_sha
                or Path(kwargs.get("batch_dir", "")) != self.batch):
            raise base.BatchError("retry_revision_process_identity_mismatch")
        self._validate_registration()
        # Revalidate every immutable registration dependency before any worker
        # action; a changed quarantined partial never becomes a live partial.
        for item in self.retries.values():
            self._validate_prior(item)
        item = self.retries.get(entry["object_id"])
        if item is None:
            return base.process_file(entry, **kwargs)
        before, state = self._canonical(entry, item)
        canonical = self.batch / "objects" / entry["object_id"] / "state.json"
        revised_path = self.revision_batch / "objects" / entry["object_id"] / "state.json"
        if not revised_path.exists() and self._staging(entry):
            raise base.BatchError("retry_revision_requires_prior_raw_and_partial_preserved_elsewhere")
        options = dict(kwargs, batch_dir=self.revision_batch)
        try:
            # COMPLETE replay also uses the revision state: the original worker
            # verifies existing receipts without HEAD/GET/audit and can record a
            # legitimate cache-restore recovery before the index is promoted.
            return base.process_file(entry, **options)
        finally:
            if revised_path.is_file():
                self._validate_registration()
                for registered in self.retries.values():
                    self._validate_prior(registered)
                base.no_symlink_components(canonical)
                if canonical.read_bytes() != before:
                    raise base.BatchError("retry_revision_concurrent_canonical_mutation_preserved")
                result = _read(revised_path)
                self._validate_identity(result, entry)
                base.atomic_json(canonical, self._index(result, entry, item, revised_path))
