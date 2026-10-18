"""Read native Stieger2021 BCI MAT records without MOABB's task/run collapse.

Only development records are supported. This reader never downloads anything;
callers must provide an original file plus an independently checked digest.
Native volts are obtained from documented microvolts using exactly 1e-6.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


CLINIC11 = ("FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4")
TARGET_TO_LABEL = {1: 1, 2: 0}  # Canonical 0=left, 1=right.
FEEDBACK_ONSET_MS = 2000.0


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _integer(value, field: str) -> int:
    x = float(value)
    if not np.isfinite(x) or x != int(x):
        raise ValueError(f"Invalid integer {field}: {value!r}")
    return int(x)


@dataclass
class StiegerSession:
    path: Path
    source_sha256: str
    participant_id: str
    session_id: str
    native_sfreq: float
    native_channels: tuple[str, ...]
    selected_indices: tuple[int, ...]
    native_bad_channels: tuple[str, ...]
    metadata: dict
    rows: list[dict]
    _data: object
    _time: object

    def pure_lr_metadata(self) -> list[dict]:
        """Return duration/identity rows without EEG and without padding time."""
        return [dict(row) for row in self.rows if row["tasknumber"] == 1 and row["label"] is not None]

    def context_volts(self, original_trial_index: int, seconds: float) -> np.ndarray:
        """Return verified feedback [0, seconds) on real clinic11 channels.

        This performs no filter/CAR/RMS; those belong to the versioned signal
        pipeline. Results/posttrial samples are never used to extend context.
        """
        row = self.rows[original_trial_index]
        if row["tasknumber"] != 1 or row["label"] is None:
            raise ValueError("Only tasknumber==1 explicit left/right trials are eligible")
        if not row["feedback_end_verified"]:
            raise ValueError("Feedback endpoint is unverified")
        if seconds <= 0 or abs(seconds * self.native_sfreq - round(seconds * self.native_sfreq)) > 1e-7:
            raise ValueError("Context length must be positive and match native sample grid")
        start = row["analysis_anchor_sample"]
        stop = start + round(seconds * self.native_sfreq)
        if seconds > row["available_feedback_seconds"] + 1e-9 or stop > row["feedback_stop_exclusive_sample"]:
            raise ValueError("Insufficient real feedback; post-result padding is forbidden")
        x = np.asarray(self._data[original_trial_index])[list(self.selected_indices), start:stop].astype(np.float64) * 1e-6
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite signal")
        if not np.any(x):
            raise ValueError("Zero-energy signal")
        return x

    def schema_audit(self) -> dict:
        lr = self.pure_lr_metadata()
        labels = {str(c): sum(r["label"] == c for r in lr) for c in (0, 1)}
        return {
            "dataset": "Stieger2021", "source_version": 1, "role": "development",
            "path": str(self.path), "source_file_sha256": self.source_sha256,
            "source_file_bytes": self.path.stat().st_size,
            "participant_id": self.participant_id, "session_id": self.session_id,
            "native_sfreq": self.native_sfreq, "native_unit_string": "uV", "scale_to_volts": 1e-6,
            "unit_evidence": "official Figshare v1 README, EEG data organization; independently SHA-pinned",
            "native_channels": list(self.native_channels), "clinic11_order": list(CLINIC11),
            "selected_indices_zero_based": list(self.selected_indices),
            "native_bad_channels": list(self.native_bad_channels),
            "selected_native_bad_channels": [x for x in self.native_bad_channels if x.upper() in {c.upper() for c in CLINIC11}],
            "interpolated_channels": [], "all_original_trials": len(self.rows),
            "pure_lr_trials": len(lr), "pure_lr_label_counts": labels,
            "physical_runs": sorted({r["physical_run_id"] for r in lr}),
            "date_raw": self.metadata.get("date"), "day_raw": self.metadata.get("day"), "time_raw": self.metadata.get("time"),
            "mbsr_subject_raw": self.metadata.get("MBSRsubject"),
            "absolute_timezone": None,
            "acquisition_order_evidence": "official README defines native cell/trial/run order; actual trialnumber/runnumber checked within file",
            "feedback_onset_cue_relative_ms": FEEDBACK_ONSET_MS,
            "endpoint_errors": sum(not r["feedback_end_verified"] for r in lr),
            "provider_artifact_flagged_lr": sum(bool(r["provider_artifact"]) for r in lr),
            "provider_artifact_handling": "recorded separately; not silently rejected or interpolated",
            "new_algorithm_training_completed": False,
        }


def load_session(path: str | Path, *, expected_sha256: str) -> StiegerSession:
    path = Path(path)
    match = re.fullmatch(r"(S[1-9][0-9]*)_Session_([1-9][0-9]*)\.mat", path.name)
    if not match:
        raise ValueError("Original provider subject/session filename required")
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError("Original MAT SHA256 mismatch")
    bci = loadmat(path, simplify_cells=True).get("BCI")
    if not isinstance(bci, dict):
        raise ValueError("Native BCI MATLAB struct is required")
    required = {"data", "time", "SRATE", "TrialData", "metadata", "chaninfo"}
    if not required <= bci.keys():
        raise ValueError(f"Missing BCI fields: {required - bci.keys()}")
    sfreq = float(bci["SRATE"])
    if not np.isfinite(sfreq) or sfreq != 1000:
        raise ValueError("Stieger v1 documented native sampling rate must be 1000 Hz")
    channel_names = tuple(str(c).strip() for c in np.atleast_1d(bci["chaninfo"]["label"]))
    canonical = [c.upper() for c in channel_names]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate native channel label")
    if not {c.upper() for c in CLINIC11} <= set(canonical):
        raise ValueError("Missing measured clinic11 channels; no interpolation permitted")
    indices = tuple(canonical.index(c.upper()) for c in CLINIC11)
    bad = []
    for i in np.atleast_1d(bci["chaninfo"].get("noisechan", [])):
        idx = _integer(i, "noisechan") - 1
        if not 0 <= idx < len(channel_names):
            raise ValueError("Native noisechan index outside measured channels")
        bad.append(channel_names[idx])
    td = bci["TrialData"]
    if isinstance(td, dict):
        td = [td]
    data, times = bci["data"], bci["time"]
    if len(td) != len(data) or len(td) != len(times):
        raise ValueError("Native trial/data/time length mismatch")
    person, session = match[1], match[2]
    rows, sequence = [], []
    for i, trial in enumerate(td):
        task = _integer(trial["tasknumber"], "tasknumber")
        target = _integer(trial["targetnumber"], "targetnumber")
        run = _integer(trial["runnumber"], "runnumber")
        original = _integer(trial["trialnumber"], "trialnumber")
        x, t = np.asarray(data[i]), np.asarray(times[i], dtype=float)
        if x.ndim != 2 or x.shape[0] != len(channel_names) or x.shape[1] != t.size:
            raise ValueError(f"Native trial {i} matrix/channel/time mismatch")
        if not np.isfinite(t).all() or t.ndim != 1 or not np.allclose(np.diff(t), 1000 / sfreq, atol=1e-6, rtol=0):
            raise ValueError(f"Native trial {i} irregular time coordinate")
        anchors = np.flatnonzero(np.isclose(t, FEEDBACK_ONSET_MS, atol=1e-6))
        if anchors.size != 1:
            raise ValueError(f"Native trial {i} feedback anchor missing/nonunique")
        anchor = int(anchors[0])
        resultind = _integer(trial["resultind"], "resultind")
        length = float(trial["triallength"])
        endpoint_ok = anchor < resultind <= t.size and np.isfinite(length) and length >= 0
        # MATLAB resultind is inclusive and 1-based; the Python exclusive stop
        # is numerically equal. Keep reported feedback duration as a conservative
        # bound; no use of the extra 1s post-result interval in BCI.data.
        available = min(length, (resultind - anchor) / sfreq) if endpoint_ok else None
        endpoint_ok = bool(endpoint_ok and abs((t[resultind - 1] - FEEDBACK_ONSET_MS) / 1000 - length) <= 1.5 / sfreq)
        sequence.append((run, original))
        row = {
            "dataset": "Stieger2021", "dataset_version": 1, "participant_id": person,
            "session_id": session, "source_record_id": path.name,
            "source_file_sha256": digest, "original_trial_index": i,
            "provider_trialnumber": original, "physical_run_id": run,
            "tasknumber": task, "label_code": target,
            "label": TARGET_TO_LABEL.get(target) if task == 1 else None,
            "canonical_label": "right_hand" if task == 1 and target == 1 else "left_hand" if task == 1 and target == 2 else None,
            "trial_uid": f"Stieger2021:v1:{person}:session{session}:{digest}:trial{i}",
            "cue_time_ms": 0.0, "task_time_ms": FEEDBACK_ONSET_MS,
            "analysis_anchor_sample": anchor, "resultind_matlab_one_based_inclusive": resultind,
            "feedback_stop_exclusive_sample": resultind, "triallength_reported_seconds": length,
            "feedback_end_verified": endpoint_ok, "available_feedback_seconds": available if endpoint_ok else None,
            "native_trial_samples_including_padding": t.size,
            "provider_artifact": int(trial["artifact"]),
            "task_eligible": task == 1 and target in TARGET_TO_LABEL,
            "exclusion_reason": None if task == 1 and target in TARGET_TO_LABEL else "not_pure_horizontal_lr",
        }
        rows.append(row)
    if len(set(sequence)) != len(sequence) or sequence != sorted(sequence):
        raise ValueError("Physical run/trial chronology inconsistent with original cell order")
    for run in {row["physical_run_id"] for row in rows}:
        rr = [row for row in rows if row["physical_run_id"] == run]
        numbers = [row["provider_trialnumber"] for row in rr]
        # The provider may number trials across the entire session. A complete
        # physical run has 25 actual records with consecutive positive native
        # numbers; its first number need not be 1 or follow a synthetic grid.
        # Keep every original number/index unchanged, including incomplete runs.
        complete = (len(rr) == 25 and all(number > 0 for number in numbers)
                    and all(right == left + 1 for left, right in zip(numbers, numbers[1:])))
        one_task = len({row["tasknumber"] for row in rr}) == 1
        for row in rr:
            row["physical_run_complete_verified"] = complete and one_task
    return StiegerSession(path, digest, person, session, sfreq, channel_names, indices, tuple(bad), bci["metadata"], rows, data, times)
