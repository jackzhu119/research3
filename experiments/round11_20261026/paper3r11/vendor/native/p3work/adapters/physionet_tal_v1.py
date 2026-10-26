"""Original EDF TAL timing with inward sample bounds; no resampling or fitting.

TAL decimal tokens are authoritative. MNE annotation durations may have already
been clipped at the recording boundary, and are retained separately. Shared
identity/montage helpers remain in the byte-frozen source_native component.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
from pathlib import Path
import re

import numpy as np

from p3work.contracts import CHANNELS, digest, file_digest, permit_preparation, validate_signal_contract
from . import source_native as legacy


def _number(token):
    try:
        value = Decimal(token)
        if not value.is_finite():
            raise ValueError("Nonfinite EDF decimal timing")
        return Fraction(value)
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise ValueError("Finite original EDF decimal token required") from error


def _ratio(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def read_original_tal(path):
    """Read header and annotation signal bytes, never EEG signal arrays."""
    path = Path(path)
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256 or fixed[:8].strip() != b"0":
            raise ValueError("Registered 16-bit EDF header required")
        ns, records, header_bytes = int(fixed[252:256]), int(fixed[236:244]), int(fixed[184:192])
        record_duration_token = fixed[244:252].decode("ascii").strip()
        record_duration = _number(record_duration_token)
        if (not 1 <= ns <= 4096 or records <= 0 or header_bytes != 256 * (ns + 1)
                or record_duration <= 0 or not fixed[192:236].startswith(b"EDF+C")):
            raise ValueError("Complete continuous EDF+C record clock required")
        header = stream.read(header_bytes - 256)
        if len(header) != header_bytes - 256:
            raise ValueError("Incomplete EDF signal header")
        labels = [header[16*i:16*(i+1)].decode("ascii").strip() for i in range(ns)]
        samples = [int(header[216*ns+8*i:216*ns+8*(i+1)]) for i in range(ns)]
        indices = [i for i, label in enumerate(labels) if label == "EDF Annotations"]
        if len(indices) != 1 or any(value <= 0 for value in samples):
            raise ValueError("One original EDF annotation signal required")
        ai = indices[0]
        stride, offset, count = 2*sum(samples), 2*sum(samples[:ai]), 2*samples[ai]
        if path.stat().st_size != header_bytes + records*stride:
            raise ValueError("EDF byte count differs from complete native records")
        annotations = []
        for record in range(records):
            stream.seek(header_bytes + record*stride + offset)
            block = stream.read(count)
            clocks = []
            for chunk in block.split(b"\x00"):
                if not chunk:
                    continue
                fields = chunk.decode("utf-8").split("\x14")
                if len(fields) < 2 or fields[-1] != "":
                    raise ValueError("Complete original TAL terminator required")
                clock = fields[0].split("\x15")
                if len(clock) not in (1, 2):
                    raise ValueError("Ambiguous EDF TAL onset/duration")
                onset = _number(clock[0])
                descriptions = [value for value in fields[1:] if value]
                if not descriptions:
                    if len(clock) != 1:
                        raise ValueError("Timekeeping TAL cannot contain task duration")
                    clocks.append(onset)
                    continue
                if len(descriptions) != 1 or len(clock) != 2 or descriptions[0] not in {"T0", "T1", "T2"}:
                    raise ValueError("Only original T0/T1/T2 task TALs are registered")
                duration = _number(clock[1])
                if onset < 0 or duration <= 0:
                    raise ValueError("Positive original task duration at nonnegative onset required")
                annotations.append({"data_record": record, "onset_token": clock[0], "duration_token": clock[1],
                    "description": descriptions[0], "onset": onset, "duration": duration})
            if clocks != [record*record_duration]:
                raise ValueError("EDF+C original data-record clock is discontinuous")
    if (not annotations or any(b["onset"] <= a["onset"] for a, b in zip(annotations, annotations[1:]))):
        raise ValueError("Original task TAL acquisition order must be strictly increasing")
    return {"record_count": records, "record_duration_token": record_duration_token,
        "recording_end": records*record_duration, "signal_labels": labels,
        "samples_per_record": samples, "annotation_signal_index": ai, "annotations": annotations}


def _revision_fields(revision):
    return {"native_reader_revision": revision,
        "physionet_tal_reader_sha256": revision["variant_reader"]["sha256"],
        "source_dispatch_sha256": revision["dispatcher"]["sha256"]}


def _audit(path, base, revision, raw):
    match = re.fullmatch(r"S(\d{3})R(\d{2})\.edf", path.name)
    if (not match or int(match.group(1)) != int(base["provider_subject"])
            or int(match.group(2)) not in {4, 8, 12}):
        raise PermissionError("Only registered original Physio LR imagery runs are allowed")
    run = int(match.group(2))
    sfreq = float(raw.info["sfreq"])
    original_names = list(raw.ch_names)
    names = [name.strip(".") for name in original_names]
    selected = legacy._selection(names)
    units = dict(raw._orig_units)
    factors = {"µV": 1e-6, "uV": 1e-6, "mV": 1e-3, "V": 1.0}
    if (sfreq not in {128.0, 160.0} or len(names) != 64 or len(set(units.values())) != 1
            or next(iter(units.values()), None) not in factors or raw.first_samp != 0):
        raise ValueError("Registered native Physio header, SI units and zero run origin required")
    source = read_original_tal(path)
    clock = Fraction(str(sfreq))
    recording_end = Fraction(int(raw.n_times), 1)/clock
    if source["recording_end"] != recording_end:
        raise ValueError("EDF header and MNE recording endpoints disagree")
    eeg_samples = [n for i,n in enumerate(source["samples_per_record"]) if i != source["annotation_signal_index"]]
    if len(eeg_samples) != 64 or any(Fraction(n, 1)/_number(source["record_duration_token"]) != clock for n in eeg_samples):
        raise ValueError("All measured EDF EEG signals must use the registered common native clock")
    events = source["annotations"]
    if len(events) != len(raw.annotations):
        raise ValueError("MNE must retain every original task TAL without adding/reordering events")
    rows = []
    for index, (event, mne_event) in enumerate(zip(events, raw.annotations), 1):
        onset, duration = event["onset"], event["duration"]
        mne_onset, mne_duration = float(mne_event["onset"]), float(mne_event["duration"])
        clipped_duration = min(onset+duration, recording_end)-onset
        if (str(mne_event["description"]) != event["description"] or clipped_duration <= 0
                or not math.isfinite(mne_onset) or not math.isfinite(mne_duration)
                or abs(Fraction(str(mne_onset))-onset) > Fraction(1, 10**9)
                or abs(Fraction(str(mne_duration))-clipped_duration) > Fraction(1, 10**9)):
            raise ValueError("MNE annotation differs from original TAL beyond documented record clipping")
        code = event["description"]
        if code == "T0":
            continue
        next_onset = events[index]["onset"] if index < len(events) else None
        bound = min(onset+duration, recording_end, next_onset if next_onset is not None else recording_end)
        start, stop = math.ceil(onset*clock), math.floor(bound*clock)
        if not 0 <= start < stop <= raw.n_times:
            raise ValueError("Original Physio task has no complete inward native sample interval")
        delay, loss = Fraction(start, 1)/clock-onset, bound-Fraction(stop, 1)/clock
        if not 0 <= delay < 1/clock or not 0 <= loss < 1/clock:
            raise ValueError("Inward task quantization exceeds one original sample")
        rows.append(legacy._row(base, source_record_id=path.name, physical_run_id=str(run),
            original_trial_index=index, raw_label_code=code,
            canonical_label="left_hand" if code == "T1" else "right_hand",
            session_order=None, run_order=run, run_order_evidence="provider filename original R04/R08/R12",
            event_sample_or_null=start,
            event_time_coordinate="first native sample at/after original decimal EDF TAL onset; inward task interval",
            cue_time=float(onset), task_time=float(onset), analysis_anchor_sample=start, mi_onset_verified=True,
            task_anchor_evidence="original EDF TAL T1/T2 onset; explicit inward sampling, not exact-grid onset claim",
            native_sfreq=sfreq, native_channel_order=original_names, selected_indices=selected,
            canonicalized_channel_names=names,
            channel_order_evidence="actual EDF labels; periods stripped and case matched only",
            native_unit_string=next(iter(units.values())), physical_scale_evidence="actual EDF physical_dimension and MNE SI conversion",
            unit_verified_from_native_header=True, physical_scale_to_si_volts=factors[next(iter(units.values()))],
            data_returned_by_mne_unit="V", real_trial_start_end=[start, stop], task_end_sample=stop,
            available_task_seconds=float(Fraction(stop-start, 1)/clock), acquisition_order_verified=True,
            annotation_onset_token=event["onset_token"], annotation_duration_token=event["duration_token"],
            annotation_onset_seconds=float(onset), annotation_duration_seconds=float(duration),
            annotation_onset_fraction=_ratio(onset), annotation_duration_fraction=_ratio(duration),
            annotation_end_fraction=_ratio(onset+duration),
            mne_annotation_onset_seconds=mne_onset, mne_annotation_duration_seconds=mne_duration,
            mne_onset_float_error_seconds=float(Fraction(str(mne_onset))-onset),
            mne_duration_minus_original_seconds=float(Fraction(str(mne_duration))-duration),
            annotation_next_onset_token=events[index]["onset_token"] if next_onset is not None else None,
            annotation_next_onset_fraction=_ratio(next_onset) if next_onset is not None else None,
            recording_end_fraction=_ratio(recording_end), bounded_task_end_fraction=_ratio(bound),
            annotation_end_after_recording=onset+duration > recording_end,
            annotation_end_after_next_onset=next_onset is not None and onset+duration > next_onset,
            record_clipping_seconds=float(max(Fraction(0), onset+duration-recording_end)),
            next_annotation_clipping_seconds=float(max(Fraction(0), onset+duration-next_onset)) if next_onset is not None else 0.0,
            onset_was_exact_native_sample=(onset*clock).denominator == 1,
            inward_onset_delay_seconds=float(delay), inward_stop_rounding_loss_seconds=float(loss),
            sampling_policy="ceil_original_TAL_onset_floor_bounded_end",
            **_revision_fields(revision)))
    if not rows:
        raise ValueError("Original Physio LR task annotations required")
    summary = legacy._summary(base, rows, native_n_times=int(raw.n_times), native_sfreq=sfreq,
        native_sampling_variant="known_128Hz_provider_variant" if sfreq == 128 else "reported_160Hz_variant",
        native_channel_order=original_names, native_channel_units=units,
        annotation_counts=dict(Counter(e["description"] for e in events)),
        original_TAL_annotation_count=len(events), original_recording_end_fraction=_ratio(recording_end),
        adapter_role="frozen_shared_metadata_helpers_component",
        active_variant_file_sha256=revision["variant_reader"]["sha256"],
        exact_grid_onsets=sum(r["onset_was_exact_native_sample"] for r in rows),
        inward_quantized_onsets=sum(not r["onset_was_exact_native_sample"] for r in rows),
        remaining_gates=["complete source inventory, new code-bound numerical caches and full-GET receipts"],
        **_revision_fields(revision))
    return {"summary": summary, "trials": rows}


def audit_native_physionet(path, *, provider_subject, session_id, source_version, expected_sha256, revision):
    import mne
    from .source_dispatch import validate_revision
    revision = validate_revision(revision)
    permit_preparation("PhysionetMI", "source_schema")
    path = Path(path)
    if file_digest(path) != expected_sha256:
        raise ValueError("Native Physio file differs from its full SHA256 receipt")
    base = {"dataset": "PhysionetMI", "role": "source", "provider_subject": str(provider_subject),
        "source_version": source_version, "source_file_sha256": expected_sha256,
        "source_file_bytes": path.stat().st_size, "session_id": str(session_id)}
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        return _audit(path, base, revision, raw)
    finally:
        raw.close()


def iter_physionet_contexts(path, *, rows, contract, expected_sha256, revision):
    import mne
    from .source_dispatch import validate_revision
    revision = validate_revision(revision)
    validate_signal_contract(contract)
    permit_preparation("PhysionetMI", "source_train")
    path = Path(path)
    if not rows or file_digest(path) != expected_sha256:
        raise ValueError("Full original Physio metadata and matching raw SHA256 required")
    first = rows[0]
    base = {k: first[k] for k in ("dataset", "role", "provider_subject", "source_version",
        "source_file_sha256", "source_file_bytes", "session_id")}
    if (base["dataset"] != "PhysionetMI" or base["role"] != "source"
            or base["source_file_sha256"] != expected_sha256 or base["source_file_bytes"] != path.stat().st_size):
        raise ValueError("Native Physio recording identity disagrees with metadata")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        expected = _audit(path, base, revision, raw)["trials"]
        if len(expected) != len(rows) or any(digest(a) != digest(b) for a,b in zip(expected, rows)):
            raise ValueError("Native Physio rows do not match the same verified reader revision")
        for row in rows:
            anchor, end = row["analysis_anchor_sample"], row["task_end_sample"]
            stop = anchor + round(contract["context_s"][1]*row["native_sfreq"])
            if stop > end:
                yield {"row": row, "excluded": "insufficient_original_task_context"}
                continue
            signal = raw.get_data(picks=row["selected_indices"], start=anchor, stop=stop)
            if signal.shape != (len(CHANNELS), stop-anchor):
                raise ValueError("Bounded Physio context changed shape")
            yield {"row": row, "signal": signal, "channel_names": list(CHANNELS), "sfreq_hz": row["native_sfreq"],
                "task_sample": 0, "task_end_sample": end-anchor, "task_end_verified": True,
                "native_unit": "V", "unit_evidence_kind": "native_EDF_header_MNE_SI",
                "original_context_samples": [anchor, stop], **_revision_fields(revision)}
    finally:
        raw.close()
