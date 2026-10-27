"""Verified Stieger B20 episodes and reproducible subject/session sampling.

This module performs no fitting, downloading, status writes or GPU scheduling.
It reads only the requested frozen partition. Invalid original trials remain in
the 20-acquired-trial prefix; the remainder of its boundary physical run is
excluded in full. Meta-gradient and evaluation label capabilities are distinct.
The historical CalibrationPrefix.protocol_freeze_sha field carries the actual
signal/input-contract identity here, never a claim of final W7 external freeze.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
import torch

from .adapt import CalibrationPrefix, PredictionInputs
from .contracts import CHANNELS, digest, file_digest, validate_signal_contract
from .methods.meta_train import MetaTrainEpisode
from .splits import _hash, build_prefix_episode


PARTITIONS = ("meta_train", "meta_validation", "internal_holdout")
PURPOSES = {"meta_train": "meta_gradient", "meta_validation": "model_selection",
            "internal_holdout": "internal_development_check"}
STAGE = "pre_rms_car11_temporal_demeaned"
_CATALOG_ISSUER = object()
_CATALOG_SEALS = WeakKeyDictionary()


class EpisodeGateError(ValueError):
    """A fixed-code data, identity or stage-gate failure."""


def _sha(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EpisodeGateError("concrete_sha256_required")
    return value


def _local(root: Path, relative: str, *, must_exist=True) -> Path:
    if (not isinstance(relative, str) or not relative or relative.startswith("/")
            or "\\" in relative or any(part in {"", ".", ".."} for part in relative.split("/"))
            or any(ord(char) < 32 for char in relative)
            or any(part.lower().startswith(".env") or part.lower() in {".aws", ".ssh", "credentials", "secrets", "legacy_baseline_v1"}
                   for part in relative.split("/"))
            or Path(relative).suffix.lower() in {".npz", ".pkl", ".pickle", ".key", ".pem"}):
        raise EpisodeGateError("unsafe_or_legacy_artifact_path")
    cursor = root
    for part in relative.split("/"):
        cursor = cursor / part
        if cursor.is_symlink():
            raise EpisodeGateError("symlink_artifact_forbidden")
    resolved = cursor.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise EpisodeGateError("artifact_outside_paper3_root")
    return resolved


def _artifact(root: Path, descriptor: Mapping, *, suffix=None) -> Path:
    if (not isinstance(descriptor, Mapping)
            or set(descriptor) not in ({"path", "sha256"}, {"path", "sha256", "bytes"})):
        raise EpisodeGateError("artifact_path_and_sha256_required")
    if "bytes" in descriptor and (type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0):
        raise EpisodeGateError("artifact_nonnegative_integer_bytes_required")
    expected = _sha(descriptor["sha256"])
    path = _local(root, descriptor["path"])
    if (not path.is_file() or (suffix is not None and path.suffix != suffix)
            or ("bytes" in descriptor and path.stat().st_size != descriptor["bytes"])
            or file_digest(path) != expected):
        raise EpisodeGateError("artifact_sha256_or_type_mismatch")
    return path


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _split(path: Path) -> dict:
    value = _read_json(path)
    body = dict(value)
    sha = body.pop("split_sha256", None)
    partitions = value.get("partitions", {})
    if (value.get("dataset") != "Stieger2021" or value.get("status") != "FROZEN_METADATA_ONLY"
            or value.get("no_post_window_reassignment") is not True
            or set(partitions) != set(PARTITIONS) or _hash(body) != _sha(sha)):
        raise EpisodeGateError("frozen_participant_split_required")
    flat = [person for role in PARTITIONS for person in partitions[role]]
    if (len(flat) != len(set(flat)) or set(flat) != {f"S{i}" for i in range(1, 63)}
            or {role: len(partitions[role]) for role in PARTITIONS}
            != {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11}):
        raise EpisodeGateError("exact_62_person_disjoint_split_required")
    return value


def _official_sessions(path: Path) -> dict[str, dict]:
    inventory = _read_json(path)
    if (inventory.get("article_id") != 13123148 or inventory.get("version") != 1
            or inventory.get("license", {}).get("name") != "CC BY 4.0"):
        raise EpisodeGateError("official_stieger_v1_inventory_required")
    result, file_ids = {}, set()
    for row in inventory.get("files", []):
        if not row.get("name", "").endswith(".mat"):
            continue
        match = re.fullmatch(r"(S[1-9][0-9]*)_Session_([1-9][0-9]*)\.mat", row["name"])
        md5 = row.get("computed_md5") or row.get("supplied_md5")
        if (not match or row.get("provider_subject") != match[1]
                or row.get("session_ordinal") != int(match[2]) or type(row.get("size")) is not int
                or row["size"] <= 0 or row["name"] in result or row.get("id") in file_ids
                or row.get("download_url") != f"https://ndownloader.figshare.com/files/{row.get('id')}"
                or not isinstance(md5, str) or re.fullmatch(r"[0-9a-f]{32}", md5) is None
                or (row.get("computed_md5") and row.get("supplied_md5")
                    and row["computed_md5"] != row["supplied_md5"])):
            raise EpisodeGateError("invalid_official_session_identity_or_integrity")
        result[row["name"]] = {**row, "official_md5": md5}
        file_ids.add(row["id"])
    if (len(result) != 598
            or {row["provider_subject"] for row in result.values()} != {f"S{i}" for i in range(1, 63)}):
        raise EpisodeGateError("complete_598_session_official_inventory_required")
    return result


def _internal_access(root: Path, tasks: dict, descriptor: dict | None,
                     split_sha: str, contract_sha: str) -> tuple[dict, str]:
    # Refuse before opening any internal events or signal array. Materializing
    # CPU caches does not itself authorize internal evaluation or label access.
    expected = tasks.get("W3", {}).get("internal_access_receipt")
    if descriptor is None or descriptor != expected:
        raise PermissionError("internal_holdout_requires_explicit_post_selection_access_receipt")
    path = _artifact(root, descriptor, suffix=".json")
    access = _read_json(path)
    if (access.get("schema_version") != "paper3-internal-check-access-v1"
            or access.get("status") != "AUTHORIZED_ONCE_DEVELOPMENT_CHECK"
            or access.get("role") != "internal_holdout"
            or access.get("split_sha256") != split_sha
            or access.get("input_contract_sha256") != contract_sha
            or access.get("labels_after_predictions_only") is not True
            or access.get("training_and_selection_allowed") is not False):
        raise PermissionError("invalid_internal_development_check_permission")
    _sha(access.get("selected_configuration_sha256"))
    return access, descriptor["sha256"]


@dataclass(frozen=True, slots=True)
class EpisodePlan:
    participant_id: str
    session_id: str
    role: str
    acquired_event_uids: tuple[str, ...]
    support_event_uids: tuple[str, ...]
    query_event_uids: tuple[str, ...]
    unused_boundary_run_uids: tuple[str, ...]
    incomplete_future_run_uids: tuple[str, ...]
    query_excluded_uids: tuple[str, ...]
    query_eligible: bool
    fallback_reason: str | None
    query_rule: str
    manifest_sha256: str
    split_sha256: str
    input_contract_sha256: str
    temporal_audit_sha256: str
    temporal_audit_json: str = field(default="", repr=False)

    @property
    def acquired_calibration_trials(self) -> int:
        return len(self.acquired_event_uids)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["temporal_audit"] = json.loads(value.pop("temporal_audit_json"))
        return value


@dataclass(frozen=True)
class EvaluationLabels:
    """Explicit evaluation capability, never accepted by the outer optimizer."""
    event_uids: tuple[str, ...]
    role: str
    plan_sha256: str
    _values: torch.Tensor = field(repr=False)
    _root: Path = field(repr=False)
    _internal_configuration_sha256: str | None = field(default=None, repr=False)

    def labels(self, *, purpose="model_selection", prediction_receipt=None) -> torch.Tensor:
        if self.role == "meta_validation":
            if purpose not in {"model_selection", "development_evaluation"}:
                raise PermissionError("validation_query_labels_cannot_supply_meta_gradients")
        elif self.role == "internal_holdout":
            if purpose != "internal_development_check" or prediction_receipt is None:
                raise PermissionError("internal_labels_require_committed_predictions")
            # prediction_store imports EpisodePlan; import lazily after this
            # module is initialized. A self-hashed JSON claim is insufficient:
            # every committed payload, reference, UID and same-logit output
            # must still verify before any internal label can be returned.
            from .prediction_store import verify_prediction_receipt
            try:
                prediction = verify_prediction_receipt(self._root, prediction_receipt)
            except Exception:
                raise PermissionError("internal_prediction_transaction_verification_failed") from None
            if (prediction.get("status") != "PREDICTIONS_COMMITTED"
                    or prediction.get("episode_plan_sha256") != self.plan_sha256
                    or tuple(prediction.get("query_event_uids", [])) != self.event_uids
                    or prediction.get("labels_accessed_before_predictions") is not False
                    or prediction.get("selected_configuration_sha256") != self._internal_configuration_sha256):
                raise PermissionError("internal_prediction_before_label_receipt_required")
        else:
            raise PermissionError("evaluation_label_capability_role_invalid")
        return self._values.detach().clone()

    def __call__(self, **kwargs) -> torch.Tensor:
        return self.labels(**kwargs)


@dataclass(frozen=True)
class EvaluationEpisode:
    support: CalibrationPrefix
    prediction_inputs: PredictionInputs
    evaluation_labels: EvaluationLabels
    plan: EpisodePlan


def _immutable(value):
    """Copy all nested scientific bindings into immutable containers once."""
    if isinstance(value, Mapping):
        return MappingProxyType({_immutable(key): _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    if value is None or type(value) in {str, int, float, bool} or isinstance(value, (Path, EpisodePlan)):
        return value
    raise EpisodeGateError("catalog_binding_contains_unfreezable_value")


def _event_tensors(x, rows: Mapping, uids: tuple[str, ...], *, n_times: int, device):
    selected = [rows[uid] for uid in uids]
    indices = [row["cache_index"] for row in selected]
    values = np.asarray(x[indices], dtype=np.float64).copy() if indices else np.empty((0, 11, n_times), dtype=np.float64)
    signal = torch.from_numpy(values).to(device=device)
    labels = torch.tensor([row["label"] for row in selected], dtype=torch.long, device=device)
    return signal, labels


def _plan_binding_sha(plan: EpisodePlan):
    return digest({name: getattr(plan, name) for name in EpisodePlan.__dataclass_fields__})


def _task_scientific_bindings(tasks, contract_sha, split_sha):
    if (tasks.get("W1", {}).get("input_contract_sha256") != contract_sha
            or tasks.get("W3", {}).get("input_contract_sha256") != contract_sha
            or tasks.get("W3", {}).get("split_sha256") != split_sha):
        raise EpisodeGateError("current_W1_W3_scientific_binding_mismatch")


class VerifiedEpisodeCatalog:
    """Construct with load_episode_catalog; all plans bind actual local artifacts."""

    __slots__ = ("root", "role", "purpose", "manifest_sha256", "split_sha256", "input_contract_sha256",
                 "final_protocol_freeze_sha256", "n_times", "plans", "partition_people", "_sessions", "_plans",
                 "_internal", "identity_sha256", "__weakref__")

    def __init__(self, *, root: Path, role: str, purpose: str, manifest_sha256: str,
                 split_sha256: str, input_contract_sha256: str, n_times: int,
                 sessions: dict, plans: tuple[EpisodePlan, ...], partition_people: tuple[str, ...],
                 internal_access: dict | None = None, access_sha256: str | None = None, _issuer=None):
        if _issuer is not _CATALOG_ISSUER:
            raise PermissionError("episode_catalog_must_be_issued_by_verified_loader")
        attributes = {"root": root, "role": role, "purpose": purpose, "manifest_sha256": manifest_sha256,
            "split_sha256": split_sha256, "input_contract_sha256": input_contract_sha256,
            "final_protocol_freeze_sha256": None, "n_times": n_times, "plans": tuple(plans),
            "partition_people": tuple(partition_people), "_sessions": _immutable(sessions),
            "_plans": _immutable({(plan.participant_id, plan.session_id): plan for plan in plans}),
            "_internal": _immutable(internal_access)}
        attributes["identity_sha256"] = digest({"manifest_sha256": manifest_sha256, "split_sha256": split_sha256,
            "input_contract_sha256": input_contract_sha256, "role": role, "purpose": purpose,
            "internal_access_sha256": access_sha256})
        for name, value in attributes.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("issued_episode_catalog_bindings_are_immutable")

    def __delattr__(self, name):
        raise AttributeError("issued_episode_catalog_bindings_are_immutable")

    def _assert_current_authority(self):
        # The issuance record is independent of mutable object attributes. It
        # contains immutable references, never a freshly reconstructed catalog.
        seal = _CATALOG_SEALS.get(self)
        if seal is None:
            raise PermissionError("episode_catalog_has_no_verified_issuance_seal")
        for name, value in seal["bindings"].items():
            actual = getattr(self, name, None)
            if (actual is not value if name in {"_sessions", "_plans", "plans", "partition_people", "_internal"} else actual != value):
                raise PermissionError("issued_episode_catalog_binding_changed")
        tasks = _read_json(_local(seal["root"], "status/tasks.json"))
        if tasks.get("W1", {}).get("status") != "COMPLETE" or tasks.get("W3", {}).get("status") != "COMPLETE":
            raise EpisodeGateError("W1_and_full_real_W3_must_be_COMPLETE")
        if tasks["W3"].get("cache_manifest") != seal["cache_manifest"]:
            raise EpisodeGateError("catalog_current_W3_authority_changed")
        _task_scientific_bindings(tasks, seal["input_contract_sha256"], seal["split_sha256"])
        for descriptor in seal["authority_artifacts"]:
            _artifact(seal["root"], descriptor)
        if seal["role"] == "internal_holdout":
            access, access_sha = _internal_access(seal["root"], tasks, dict(seal["internal_access_receipt"]),
                                                  seal["split_sha256"], seal["input_contract_sha256"])
            if _immutable(access) != self._internal or access_sha != seal["access_sha256"]:
                raise PermissionError("internal_development_check_permission_changed")
        return seal

    def _assert_issued_plan(self, plan, seal):
        if (type(plan) is not EpisodePlan or self._plans.get((plan.participant_id, plan.session_id)) is not plan
                or plan.role != self.role or plan.manifest_sha256 != self.manifest_sha256):
            raise PermissionError("episode_plan_not_issued_by_this_catalog")
        if _plan_binding_sha(plan) != seal["plan_binding_sha256"][(plan.participant_id, plan.session_id)]:
            raise PermissionError("issued_episode_plan_binding_changed")

    def _assert_plan(self, plan):
        seal = self._assert_current_authority()
        self._assert_issued_plan(plan, seal)
        if self.purpose != PURPOSES.get(self.role):
            raise PermissionError("partition_purpose_permission_mismatch")
        if not plan.query_eligible or len(plan.acquired_event_uids) != 20:
            raise EpisodeGateError("episode_not_evaluable_under_predeclared_B20_query_rules")

    def plan(self, participant_id: str, session_id: str) -> EpisodePlan:
        seal = self._assert_current_authority()
        try:
            plan = self._plans[participant_id, str(session_id)]
        except KeyError:
            raise PermissionError("session_not_in_authorized_partition") from None
        self._assert_issued_plan(plan, seal)
        return plan

    def materialize_support(self, plan: EpisodePlan, *, device="cpu") -> CalibrationPrefix:
        """Issue only the original B20 support, without querying future labels.

        Complete file hashes are rechecked, but the numerical view indexes only
        valid acquired-prefix events. Invalid acquired originals still cost 20;
        neither absent classes nor excluded originals receive replacements.
        """
        self._assert_plan(plan)
        session = self._sessions[plan.participant_id, plan.session_id]
        array_path = _artifact(self.root, session["X"], suffix=".npy")
        _artifact(self.root, session["events"], suffix=".json")
        x = np.load(array_path, mmap_mode="r", allow_pickle=False)
        signal, labels = _event_tensors(x, session["rows"], plan.support_event_uids, n_times=self.n_times, device=device)
        role = {"meta_train": "meta_train_support", "meta_validation": "meta_validation_support",
                "internal_holdout": "calibration_prefix"}[self.role]
        return CalibrationPrefix(signal, labels, plan.support_event_uids, plan.acquired_event_uids,
                                 self.input_contract_sha256, role=role)

    def materialize(self, plan: EpisodePlan, *, device="cpu") -> MetaTrainEpisode | EvaluationEpisode:
        self._assert_plan(plan)
        session = self._sessions[plan.participant_id, plan.session_id]
        # Revalidate selected bytes at materialization, so a mutated file cannot
        # inherit an earlier successful catalog load.
        array_path = _artifact(self.root, session["X"], suffix=".npy")
        _artifact(self.root, session["events"], suffix=".json")
        x = np.load(array_path, mmap_mode="r", allow_pickle=False)
        support_x, support_y = _event_tensors(x, session["rows"], plan.support_event_uids, n_times=self.n_times, device=device)
        support_role = {"meta_train": "meta_train_support", "meta_validation": "meta_validation_support",
                        "internal_holdout": "calibration_prefix"}[self.role]
        support = CalibrationPrefix(support_x, support_y, plan.support_event_uids,
                                    plan.acquired_event_uids, self.input_contract_sha256, role=support_role)
        query_x, query_y = _event_tensors(x, session["rows"], plan.query_event_uids, n_times=self.n_times, device=device)
        if self.role == "meta_train":
            if self.purpose != "meta_gradient":
                raise PermissionError("meta_gradient_capability_required")
            return MetaTrainEpisode(support, query_x, query_y, plan.query_event_uids,
                                    self.split_sha256, plan.temporal_audit_sha256)
        capability = EvaluationLabels(
            plan.query_event_uids, self.role, plan.temporal_audit_sha256, query_y,
            self.root, self._internal.get("selected_configuration_sha256") if self._internal else None,
        )
        return EvaluationEpisode(support, PredictionInputs(query_x, plan.query_event_uids,
                                  self.input_contract_sha256), capability, plan)


def _validate_rows(rows: list, session: dict, *, contract: dict, contract_sha: str,
                   adapter_sha: str, processing_sha: str, seen: set[str]) -> tuple[list[dict], int]:
    if not isinstance(rows, list) or not rows or session.get("acquisition_order_verified") is not True:
        raise EpisodeGateError("all_original_LR_events_and_verified_acquisition_order_required")
    output, order, valid_indices = [], [], []
    sfreq = session.get("native_sfreq_hz")
    if type(sfreq) not in {int, float} or not math.isfinite(sfreq) or sfreq != 1000:
        raise EpisodeGateError("documented_stieger_v1_native_clock_required")
    ncontext = round(contract["context_s"][1] * sfreq)
    for original in rows:
        row = dict(original)
        uid = row.get("event_uid")
        index = row.get("original_trial_index")
        expected_uid = (f"Stieger2021:v1:{session['participant_id']}:session{session['session_id']}:"
                        f"{session['source_file_sha256']}:trial{index}")
        if (not isinstance(uid, str) or not uid or uid != row.get("trial_uid") or uid in seen
                or uid != expected_uid
                or row.get("dataset") != "Stieger2021" or type(row.get("dataset_version")) is not int
                or row.get("dataset_version") != 1
                or row.get("participant_id") != session["participant_id"] or row.get("session_id") != session["session_id"]
                or row.get("partition") != session["partition"]
                or row.get("source_record_id") != session["source_record_id"]
                or row.get("source_file_sha256") != session["source_file_sha256"]
                or type(index) is not int or index < 0 or type(row.get("tasknumber")) is not int
                or row.get("tasknumber") != 1
                or type(row.get("label")) is not int or row["label"] not in {0, 1}
                or row.get("acquisition_order_verified") is not True
                or row.get("native_sfreq_hz") != sfreq
                or row.get("input_contract_sha256") != contract_sha
                or row.get("adapter_sha256") != adapter_sha or row.get("processing_sha256") != processing_sha
                or type(row.get("valid")) is not bool):
            raise EpisodeGateError("event_identity_task_order_or_code_binding_failed")
        if row.get("canonical_label") != {0: "left_hand", 1: "right_hand"}[row["label"]]:
            raise EpisodeGateError("original_LR_label_mapping_failed")
        seen.add(uid)
        order.append(index)
        integers = ("analysis_anchor_sample", "feedback_stop_exclusive_sample",
                    "native_trial_samples_including_padding", "context_start_sample", "context_stop_sample")
        if any(type(row.get(key)) is not int for key in integers):
            raise EpisodeGateError("integer_original_sample_bounds_required")
        anchor, end, native = (row[key] for key in integers[:3])
        native_lengths = session.get("source_n_times")
        if native_lengths is not None and (not isinstance(native_lengths, list) or index >= len(native_lengths)
                                          or native_lengths[index] != native):
            raise EpisodeGateError("row_source_bounds_disagree_with_original_trial_index")
        if (anchor < 0 or not anchor < native or type(row.get("feedback_end_verified")) is not bool
                or (row["feedback_end_verified"] and not anchor < end <= native)
                or row["context_start_sample"] != anchor or row["context_stop_sample"] != anchor + ncontext):
            raise EpisodeGateError("original_feedback_or_requested_context_bounds_failed")
        if row["valid"]:
            qc = row.get("preprocessing_qc", {})
            if (type(row.get("cache_index")) is not int or row["cache_index"] < 0
                    or row["feedback_end_verified"] is not True
                    or row["context_stop_sample"] > end or row["context_stop_sample"] > native
                    or row.get("exclusion_reason") is not None
                    or qc.get("signal_contract_sha256") != contract_sha
                    or qc.get("normalization_applied") is not False
                    or qc.get("data_after_context_accessed") is not False
                    or qc.get("filter_context_ends_at_native_sample") != row["context_stop_sample"]):
                raise EpisodeGateError("valid_trial_QC_or_real_source_context_bound_failed")
            _sha(row.get("signal_sha256"))
            _sha(row.get("raw_context_sha256"))
            valid_indices.append(row["cache_index"])
        elif row.get("cache_index") is not None or not isinstance(row.get("exclusion_reason"), str) or not row["exclusion_reason"]:
            raise EpisodeGateError("invalid_original_trial_must_keep_cost_and_exclusion_reason")
        output.append(row)
    if order != sorted(set(order)) or valid_indices != list(range(len(valid_indices))):
        raise EpisodeGateError("original_acquisition_order_or_valid_cache_index_sequence_failed")
    runs = [row.get("physical_run_id") for row in output]
    if any(run is None for run in runs) and not all(run is None for run in runs):
        raise EpisodeGateError("partial_physical_run_identity_forbidden")
    if all(run is None for run in runs):
        if session.get("continuous_original_trial_block_verified") is not True:
            raise EpisodeGateError("missing_physical_run_needs_verified_continuous_original_trial_block")
    else:
        groups, closed, previous = {}, set(), None
        for row in output:
            run = row["physical_run_id"]
            if type(run) not in {int, str} or (isinstance(run, str) and not run):
                raise EpisodeGateError("real_physical_run_identity_required")
            if run != previous:
                if run in closed:
                    raise EpisodeGateError("discontiguous_physical_run_forbidden")
                closed.add(run)
                previous = run
            groups.setdefault(run, []).append(row)
        for rr in groups.values():
            numbers = [row.get("provider_trialnumber") for row in rr]
            flags = [row.get("physical_run_complete_verified") for row in rr]
            if (any(type(number) is not int or number < 1 for number in numbers)
                    or any(type(flag) is not bool for flag in flags) or len(set(flags)) != 1):
                raise EpisodeGateError("whole_run_completion_evidence_required")
            # Native trialnumber can be session-global. Completeness concerns
            # the actual 25 records within this run, not an invented 1..25 or
            # session-grid origin. Preserve the provider numbers unchanged.
            if flags[0] and (len(numbers) != 25 or any(b != a + 1 for a, b in zip(numbers, numbers[1:]))):
                raise EpisodeGateError("complete_stieger_run_needs_all_25_original_trials")
    if session.get("original_lr_trials") != len(output) or session.get("valid_trials") != len(valid_indices):
        raise EpisodeGateError("session_original_or_valid_trial_count_failed")
    return output, len(valid_indices)


def _array_sha(value: np.ndarray) -> str:
    result = hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode())
    result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def _check_array(path: Path, nvalid: int, n_times: int, rows: list[dict]) -> None:
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (nvalid, 11, n_times) or x.dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise EpisodeGateError("real_pre_RMS_N11T_cache_required")
    for start in range(0, len(x), 64):
        chunk = np.asarray(x[start:start + 64], dtype=np.float64)
        rms = np.sqrt(np.mean(chunk ** 2, axis=(1, 2)))
        if (not np.isfinite(chunk).all() or not np.isfinite(rms).all() or np.any(rms == 0)
                or np.any(np.abs(chunk.mean(1)).max(1) > rms * 5e-6)
                or np.any(np.abs(chunk.mean(2)).max(1) > rms * 5e-6)):
            raise EpisodeGateError("cache_nonfinite_zero_energy_or_CAR_temporal_demeaning_failed")
    for row in rows:
        if row["valid"] and _array_sha(x[row["cache_index"]]) != row["signal_sha256"]:
            raise EpisodeGateError("original_event_to_actual_signal_bytes_binding_failed")


def load_episode_catalog(root, cache_manifest=None, split_path=None, *,
                         input_contract_path="contracts/input_contract.json", role="meta_train",
                         purpose=None, internal_access_receipt=None) -> VerifiedEpisodeCatalog:
    """Read a real, stage-bound W3 manifest and ONLY the requested role payloads.

    cache_manifest is an optional path/SHA descriptor; when provided it must
    equal status/tasks.json W3.cache_manifest. It cannot override the stage gate.
    All 62 people and 598 official sessions must remain registered in that cache.
    """
    if role not in PARTITIONS:
        raise PermissionError("only_frozen_Stieger_development_partitions_are_supported")
    purpose = purpose or PURPOSES[role]
    if purpose != PURPOSES[role]:
        raise PermissionError("partition_purpose_permission_mismatch")
    root = Path(root).resolve(strict=True)
    tasks = _read_json(_local(root, "status/tasks.json"))
    if tasks.get("W1", {}).get("status") != "COMPLETE" or tasks.get("W3", {}).get("status") != "COMPLETE":
        raise EpisodeGateError("W1_and_full_real_W3_must_be_COMPLETE")
    contract_path = _local(root, input_contract_path)
    if contract_path != _local(root, "contracts/input_contract.json"):
        raise EpisodeGateError("canonical_frozen_input_contract_required")
    contract = _read_json(contract_path)
    contract_sha = validate_signal_contract(contract)
    authority = tasks["W3"].get("cache_manifest")
    if cache_manifest is not None and cache_manifest != authority:
        raise EpisodeGateError("cache_manifest_not_bound_to_W3_stage_receipt")
    manifest_path = _artifact(root, authority, suffix=".json")
    manifest_sha = authority["sha256"]
    manifest = _read_json(manifest_path)
    if (manifest.get("schema_version") != "paper3-stieger-cache-v2"
            or manifest.get("status") != "STIEGER_CACHE_COMPLETE" or manifest.get("dataset") != "Stieger2021"
            or manifest.get("synthetic") is not False or manifest.get("old_npz_reused") is not False
            or manifest.get("stage") != STAGE or manifest.get("normalization_applied") is not False
            or set(manifest.get("partition_scope", [])) != set(PARTITIONS)
            or manifest.get("input_contract_sha256") != contract_sha):
        raise EpisodeGateError("complete_real_frozen_Stieger_cache_identity_required")
    if _artifact(root, manifest.get("input_contract"), suffix=".json") != contract_path:
        raise EpisodeGateError("cache_input_contract_file_identity_mismatch")
    split_file = _artifact(root, manifest.get("participant_split"), suffix=".json")
    if split_path is not None:
        requested = Path(split_path)
        relative = requested.relative_to(root).as_posix() if requested.is_absolute() else str(requested)
        if _local(root, relative) != split_file:
            raise EpisodeGateError("cache_participant_split_file_identity_mismatch")
    split = _split(split_file)
    split_sha = split["split_sha256"]
    if (manifest.get("split_sha256") != split_sha
            or contract["window_audit_receipt"]["split_sha256"] != split_sha):
        raise EpisodeGateError("input_cache_and_person_split_identities_disagree")
    _task_scientific_bindings(tasks, contract_sha, split_sha)
    internal, access_sha = None, None
    if role == "internal_holdout":
        internal, access_sha = _internal_access(root, tasks, internal_access_receipt, split_sha, contract_sha)
    adapter = _artifact(root, manifest.get("adapter_code"), suffix=".py")
    processing = _artifact(root, manifest.get("processing_code"), suffix=".py")
    if (adapter != _local(root, "p3work/adapters/stieger_native.py")
            or processing != _local(root, "p3work/preprocess.py")):
        raise EpisodeGateError("new_native_adapter_and_bounded_preprocessing_code_required")
    official = _official_sessions(_artifact(root, manifest.get("official_inventory"), suffix=".json"))
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 598:
        raise EpisodeGateError("all_598_official_cache_sessions_required")
    registry = {}
    people_to_role = {person: partition for partition, people in split["partitions"].items() for person in people}
    for session in sessions:
        name = session.get("source_record_id")
        original = official.get(name)
        if (original is None or name in registry
                or session.get("participant_id") != original["provider_subject"]
                or session.get("session_id") != str(original["session_ordinal"])
                or session.get("partition") != people_to_role.get(session.get("participant_id"))
                or session.get("source_file_bytes") != original["size"]
                or session.get("raw_integrity", {}).get("bytes") != original["size"]
                or session.get("raw_integrity", {}).get("md5") != original["official_md5"]
                or session.get("raw_integrity", {}).get("sha256") != session.get("source_file_sha256")):
            raise EpisodeGateError("cache_session_not_exact_official_split_registered_object")
        _sha(session.get("source_file_sha256"))
        registry[name] = session
    if set(registry) != set(official):
        raise EpisodeGateError("cache_session_registry_not_exact_official_598_objects")
    selected, plans, seen = {}, [], set()
    for session in sessions:
        if session["partition"] != role:
            continue
        events = _artifact(root, session.get("events"), suffix=".json")
        rows, nvalid = _validate_rows(_read_json(events), session, contract=contract,
            contract_sha=contract_sha, adapter_sha=manifest["adapter_code"]["sha256"],
            processing_sha=manifest["processing_code"]["sha256"], seen=seen)
        _check_array(_artifact(root, session.get("X"), suffix=".npy"), nvalid, contract["n_times"], rows)
        audit = build_prefix_episode(rows, acquisition_order_verified=True, budget=20)
        fallback = audit["fallback_reason"]
        if len(audit["acquired_prefix_uids"]) == 20 and len({row["label"] for row in rows if row["event_uid"] in audit["support_uids"]}) < 2:
            fallback = "insufficient_class_support"
        audit.update(dataset="Stieger2021", participant_id=session["participant_id"], session_id=session["session_id"],
                     role=role, purpose=purpose, manifest_sha256=manifest_sha, split_sha256=split_sha,
                     input_contract_sha256=contract_sha, original_lr_trials=len(rows),
                     final_protocol_freeze_sha256=None, source_record_id=session["source_record_id"],
                     source_file_bytes=session["source_file_bytes"],
                     source_file_sha256=session["source_file_sha256"], events_sha256=session["events"]["sha256"],
                     support_fallback_reason=fallback)
        # Persist the actual temporal/source coordinates for every original,
        # including excluded support/query trials. Per-trial labels deliberately
        # stay behind the role-specific materialization capability.
        audit["original_event_audit"] = [{key: row.get(key) for key in (
            "event_uid", "original_trial_index", "provider_trialnumber", "physical_run_id",
            "physical_run_complete_verified", "valid", "cache_index", "exclusion_reason",
            "analysis_anchor_sample", "feedback_stop_exclusive_sample", "feedback_end_verified",
            "native_trial_samples_including_padding", "context_start_sample", "context_stop_sample",
            "signal_sha256", "raw_context_sha256",
        )} for row in rows]
        audit_sha = digest(audit)
        plans.append(EpisodePlan(
            session["participant_id"], session["session_id"], role, tuple(audit["acquired_prefix_uids"]),
            tuple(audit["support_uids"]), tuple(audit["query_uids"]), tuple(audit["unused_boundary_run_uids"]),
            tuple(audit["incomplete_or_unverified_future_run_uids"]), tuple(audit["query_excluded_uids"]),
            audit["query_eligible"], fallback, audit["query_rule"], manifest_sha, split_sha, contract_sha, audit_sha,
            json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False),
        ))
        selected[session["participant_id"], session["session_id"]] = {
            "X": dict(session["X"]), "events": dict(session["events"]),
            "rows": MappingProxyType({row["event_uid"]: MappingProxyType(row) for row in rows}),
        }
    plans.sort(key=lambda plan: (int(plan.participant_id[1:]), int(plan.session_id)))
    catalog = VerifiedEpisodeCatalog(root=root, role=role, purpose=purpose, manifest_sha256=manifest_sha,
        split_sha256=split_sha, input_contract_sha256=contract_sha, n_times=contract["n_times"],
        sessions=selected, plans=tuple(plans), partition_people=tuple(split["partitions"][role]),
        internal_access=internal, access_sha256=access_sha, _issuer=_CATALOG_ISSUER)
    _CATALOG_SEALS[catalog] = MappingProxyType({
        "root": root, "role": role, "split_sha256": split_sha, "input_contract_sha256": contract_sha,
        "cache_manifest": _immutable(authority), "internal_access_receipt": _immutable(internal_access_receipt),
        "access_sha256": access_sha,
        "plan_binding_sha256": _immutable({(plan.participant_id, plan.session_id): _plan_binding_sha(plan)
                                           for plan in catalog.plans}),
        "authority_artifacts": tuple(_immutable(item) for item in (
            authority, manifest["input_contract"], manifest["participant_split"], manifest["adapter_code"],
            manifest["processing_code"], manifest["official_inventory"])),
        "bindings": MappingProxyType({name: getattr(catalog, name) for name in catalog.__slots__ if name != "__weakref__"}),
    })
    catalog._assert_current_authority()
    return catalog


class DeterministicEpisodeSampler:
    """Stateless SHA draws: uniform subject, then its eligible sessions.

    Cursor and identity are sufficient for exact restart. Sessions with missing
    support classes remain in the pool and fall back to f0. An entire frozen
    person with no evaluable future query is a reported gate failure; the sampler
    cannot silently replace or remove them. An explicit participant_subset can
    select a predeclared nested-identity experiment within the same frozen split.
    """

    __slots__ = ("catalog", "seed", "cursor", "device", "people", "_eligible", "selection_sha256")

    def __setattr__(self, name, value):
        if name != "cursor":
            raise AttributeError("episode_sampler_metadata_is_immutable")
        if type(value) is not int or value < 0:
            raise EpisodeGateError("integer_seed_and_nonnegative_sampler_cursor_required")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError("episode_sampler_metadata_is_immutable")

    def __init__(self, catalog: VerifiedEpisodeCatalog, seed: int, *, cursor=0,
                 device="cpu", participant_subset=None):
        if type(catalog) is not VerifiedEpisodeCatalog:
            raise PermissionError("training_sampler_requires_meta_train_catalog")
        catalog._assert_current_authority()
        if catalog.role != "meta_train" or catalog.purpose != "meta_gradient":
            raise PermissionError("training_sampler_requires_meta_train_catalog")
        if type(seed) is not int or type(cursor) is not int or cursor < 0:
            raise EpisodeGateError("integer_seed_and_nonnegative_sampler_cursor_required")
        people = tuple(sorted(participant_subset if participant_subset is not None else catalog.partition_people,
                              key=lambda person: int(person[1:])))
        if (not people or len(people) != len(set(people))
                or not set(people) <= set(catalog.partition_people)):
            raise PermissionError("sampler_subset_must_stay_within_frozen_meta_train_people")
        eligible = {person: tuple(plan for plan in catalog.plans if plan.participant_id == person
                                  and plan.query_eligible and len(plan.acquired_event_uids) == 20)
                    for person in people}
        if any(not plans for plans in eligible.values()):
            raise EpisodeGateError("frozen_meta_train_person_has_no_evaluable_future_query_no_silent_reassignment")
        attributes = {"catalog": catalog, "seed": seed, "cursor": cursor, "device": device,
            "people": people, "_eligible": _immutable(eligible),
            "selection_sha256": digest({person: [plan.temporal_audit_sha256 for plan in plans]
                                         for person, plans in eligible.items()})}
        for name, value in attributes.items():
            object.__setattr__(self, name, value)

    def _draw(self, size: int, stage: str) -> int:
        # Rejection avoids modulo bias and never consumes global Python/NumPy/
        # torch RNG state used by the scientific model or another configuration.
        limit = 2**256 - (2**256 % size)
        attempt = 0
        while True:
            value = int(digest(["paper3-episode-sampler-v1", self.seed, self.cursor, stage, attempt]), 16)
            if value < limit:
                return value % size
            attempt += 1

    def next_plan(self) -> EpisodePlan:
        seal = self.catalog._assert_current_authority()
        person = self.people[self._draw(len(self.people), "subject")]
        sessions = self._eligible[person]
        plan = sessions[self._draw(len(sessions), "session:" + person)]
        self.catalog._assert_issued_plan(plan, seal)
        self.cursor += 1
        return plan

    def next(self) -> MetaTrainEpisode:
        cursor = self.cursor
        plan = self.next_plan()
        try:
            return self.catalog.materialize(plan, device=self.device)
        except BaseException:
            self.cursor = cursor
            raise

    def __next__(self) -> MetaTrainEpisode:
        return self.next()

    def __iter__(self):
        return self

    def state_dict(self) -> dict:
        self.catalog._assert_current_authority()
        return {"schema_version": "paper3-episode-sampler-v1", "seed": self.seed,
                "cursor": self.cursor, "catalog_sha256": self.catalog.identity_sha256,
                "manifest_sha256": self.catalog.manifest_sha256, "split_sha256": self.catalog.split_sha256,
                "input_contract_sha256": self.catalog.input_contract_sha256, "role": self.catalog.role,
                "selection_sha256": self.selection_sha256, "participant_subset": list(self.people)}

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise EpisodeGateError("sampler_resume_schema_mismatch")
        for key, value in expected.items():
            if key != "cursor" and state[key] != value:
                raise EpisodeGateError("sampler_resume_data_split_role_or_seed_mismatch")
        if type(state["cursor"]) is not int or state["cursor"] < 0:
            raise EpisodeGateError("sampler_resume_cursor_invalid")
        self.cursor = state["cursor"]
