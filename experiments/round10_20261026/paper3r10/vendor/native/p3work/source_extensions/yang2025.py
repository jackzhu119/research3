"""Independent native Yang v5 source extension; no training or external access.

Only the frozen 2C native identities and explicit 1/2 labels are admitted.
Returned arrays are pre-RMS CAR11 calibrated numeric, with unresolved SI unit.
The source model must apply joint trial RMS before use. This is not P1 NPZ
reuse, a new blind cohort, a physical MI-onset claim, or a meta episode loader.
"""
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from ..contracts import CHANNELS, digest, validate_signal_contract
from ..preprocess import prepare_trial


SCHEMA = "paper3-yang-source-profile-v1"
INPUT_SHA = "5c8ea3f01634c060546af29b9330d314c4f03c265677a74b875ac7c302e31872"
CLOCK_SHA = "aa0c90e91e6a1dd876673db7319a067646d8604536d879b58b09b739d344f89f"
LABEL_SHA = "5291d362185a8db9bf45ddc95d9cb49d40de57893d2fd4da8681a802b5ab36f9"
_POLICY_ROOT = "audit/data_plan_20261019/yang/evidence/"
_REPO = Path(__file__).resolve().parents[2]


class YangQualificationError(ValueError):
    """A native identity, clock, calibration or source-profile gate failed."""


def _require(condition, reason):
    if not condition:
        raise YangQualificationError(reason)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _profile(profile):
    _require(isinstance(profile, dict), "explicit_source_profile_required")
    fixed = {"schema_version": SCHEMA, "dataset_id": "Yang2025_2C", "source_only": True,
             "context_from_marker_s": [2.25, 3.75], "crop_from_marker_s": [2.5, 3.5],
             "analysis_anchor_from_marker_s": 2.25, "MI_onset_verified": False,
             "prior_analysis_intersection_s": [2, 4], "sfreq_out_hz": 128, "n_times": 128,
             "channels": list(CHANNELS), "input_contract_sha256": INPUT_SHA,
             "native_unit_policy": "common_gain_numeric_unresolved_no_absolute_QC",
             "stored_stage": "pre_rms_car11_temporal_demeaned"}
    for key, value in fixed.items():
        _require(type(profile.get(key)) is type(value) and profile[key] == value,
                 "source_profile_changed_" + key)
    _require(validate_signal_contract(profile.get("input_contract")) == INPUT_SHA,
             "source_input_contract_changed")
    return profile["input_contract"]


def _policy(profile, key, filename, expected):
    ref = profile.get(key)
    _require(ref == {"path": _POLICY_ROOT + filename, "sha256": expected},
             "frozen_policy_reference_changed")
    path = _REPO / ref["path"]
    _require(not path.is_symlink() and path.is_file() and _sha(path) == expected,
             "frozen_policy_bytes_changed")
    return json.loads(path.read_bytes())


def _parse_header(path):
    """Strict continuous BDF parsing; no MNE unit inference or event shifting."""
    path = Path(path)
    with path.open("rb") as stream:
        fixed = stream.read(256)
        _require(len(fixed) == 256 and fixed[:8] == b"\xffBIOSEMI", "native_BDF_required")
        try:
            ns, header_bytes = int(fixed[252:256]), int(fixed[184:192])
            records, duration = int(fixed[236:244]), Fraction(fixed[244:252].decode().strip())
        except (ValueError, ZeroDivisionError):
            raise YangQualificationError("invalid_BDF_numeric_header") from None
        _require(0 < ns <= 256 and header_bytes == 256 * (ns + 1), "BDF_header_length")
        body = stream.read(header_bytes - 256)
    _require(len(body) == header_bytes - 256 and records > 0 and duration > 0,
             "truncated_or_unknown_BDF_records")
    _require(b"+D" not in fixed[192:236], "discontinuous_BDF_not_supported")
    definitions = (("label", 16), ("transducer", 80), ("unit", 8), ("physical_min", 8),
                   ("physical_max", 8), ("digital_min", 8), ("digital_max", 8),
                   ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32))
    fields, pos = {}, 0
    for name, width in definitions:
        fields[name] = [body[pos + i*width:pos + (i+1)*width].decode("latin1").strip()
                        for i in range(ns)]
        pos += ns * width
    channels = []
    for i in range(ns):
        row = {key: value[i] for key, value in fields.items()}
        try:
            for name in ("physical_min", "physical_max", "digital_min", "digital_max"):
                row[name] = Fraction(row[name])
            row["samples_per_record"] = int(row["samples_per_record"])
        except (ValueError, ZeroDivisionError):
            raise YangQualificationError("invalid_BDF_channel_calibration") from None
        _require(row["samples_per_record"] > 0, "nonpositive_BDF_sample_count")
        row["sfreq"] = Fraction(row["samples_per_record"], 1) / duration
        channels.append(row)
    record_bytes = 3 * sum(c["samples_per_record"] for c in channels)
    _require(header_bytes + records * record_bytes == path.stat().st_size,
             "BDF_record_file_size_mismatch")
    return {"channels": channels, "header_bytes": header_bytes, "records": records,
            "duration": duration, "record_bytes": record_bytes,
            "start_date_time": fixed[168:184].decode("latin1"),
            "header_sha256": hashlib.sha256(fixed + body).hexdigest()}


def _common_gain(header):
    labels = [r["label"] for r in header["channels"]]
    _require(len(labels) == len(set(labels)) and all(c in labels for c in CHANNELS),
             "missing_or_duplicate_measured_clinic11")
    chosen = [header["channels"][labels.index(c)] for c in CHANNELS]
    required = {"unit": "?V", "physical_min": Fraction(-375000),
                "physical_max": Fraction(375000), "digital_min": Fraction(-8388608),
                "digital_max": Fraction(8388607), "sfreq": Fraction(1000)}
    for row in chosen:
        _require(all(row[k] == value for k, value in required.items()),
                 "unreviewed_native_relative_calibration")
    scales = [(c["physical_max"]-c["physical_min"])/(c["digital_max"]-c["digital_min"])
              for c in chosen]
    offsets = [c["physical_min"]-c["digital_min"]*scale for c, scale in zip(chosen, scales)]
    _require(len(set(scales)) == len(set(offsets)) == 1 and scales[0] > 0,
             "noncommon_channel_gain_or_offset")
    return {"declared_native_unit": "?V", "provider_channels_tsv_unit": "uV",
            "SI_unit_verified": False, "stored_unit": "unresolved",
            "common_positive_gain_verified": True, "common_scale_rational": str(scales[0]),
            "common_offset_rational": str(offsets[0]), "absolute_amplitude_QC_allowed": False,
            "selected_indices": [labels.index(c) for c in CHANNELS]}


def _read_annotations(path, header):
    indices = [i for i, c in enumerate(header["channels"])
               if c["label"] in {"BDF Annotations", "EDF Annotations"}]
    _require(len(indices) == 1, "one_annotation_channel_required")
    index = indices[0]
    offsets = np.r_[0, np.cumsum([c["samples_per_record"]*3 for c in header["channels"]])]
    events = []
    with Path(path).open("rb") as stream:
        for record in range(header["records"]):
            stream.seek(header["header_bytes"] + record*header["record_bytes"] + int(offsets[index]))
            block = stream.read(int(offsets[index+1]-offsets[index]))
            for tal_index, tal in enumerate(block.split(b"\0")):
                if not tal:
                    continue
                _require(b"\x14" in tal, "malformed_native_TAL")
                fields = tal.decode("utf8", errors="strict").split("\x14")
                timing = fields[0].split("\x15")
                _require(len(timing) <= 2, "malformed_TAL_time")
                try:
                    onset = Fraction(timing[0])
                    duration = Fraction(timing[1]) if len(timing) == 2 and timing[1] else Fraction(0)
                except (ValueError, ZeroDivisionError):
                    raise YangQualificationError("invalid_TAL_time") from None
                _require(onset >= 0 and duration >= 0, "negative_TAL_time")
                for description in fields[1:]:
                    if description:
                        events.append({"original_event_index": len(events), "native_record_index": record,
                                       "native_TAL_index": tal_index, "onset_token": timing[0],
                                       "onset_fraction": onset, "duration_fraction": duration,
                                       "description": description})
    _require(all(a["onset_fraction"] <= b["onset_fraction"] for a, b in zip(events, events[1:])),
             "native_events_not_chronological")
    return events


def _sample(onset, sfreq=1000):
    """Nonnegative nearest sample; preserve exact TAL token and quantization."""
    value = Fraction(onset) * sfreq
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _context(path, header, indices, start, stop):
    nper = header["channels"][indices[0]]["samples_per_record"]
    _require(all(header["channels"][i]["samples_per_record"] == nper for i in indices),
             "selected_native_rates_differ")
    _require(0 <= start < stop <= header["records"]*nper, "context_out_of_record")
    offsets = np.r_[0, np.cumsum([c["samples_per_record"]*3 for c in header["channels"]])]
    data = np.memmap(path, mode="r", dtype=np.uint8, offset=header["header_bytes"],
                     shape=(header["records"], header["record_bytes"]))
    result = np.empty((len(indices), stop-start), dtype=np.float64)
    try:
        for target, ci in enumerate(indices):
            c = header["channels"][ci]
            gain = float((c["physical_max"]-c["physical_min"])/(c["digital_max"]-c["digital_min"]))
            offset = float(c["physical_min"])-float(c["digital_min"])*gain
            write = 0
            for rec in range(start//nper, (stop-1)//nper + 1):
                lo, hi = max(start-rec*nper, 0), min(stop-rec*nper, nper)
                b = np.asarray(data[rec, int(offsets[ci])+lo*3:int(offsets[ci])+hi*3]).reshape(-1, 3)
                unsigned = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
                signed = (unsigned ^ 0x800000) - 0x800000
                result[target, write:write+hi-lo] = signed.astype(np.float64)*gain + offset
                write += hi-lo
    finally:
        del data
    return result


def _native_facts(pair, headers, observed, gain):
    """Native facts collected before waveform slicing/QC, not copied cache rows.

    This is provenance from the native parser, not an independently implemented
    second parser. A separate verifier can join these facts to output event UIDs
    and check labels/bounds without trusting filtered event rows.
    """
    rows, segment = [], 0
    for event in observed:
        code, onset = event["description"], event["onset_fraction"]
        marker = _sample(onset)
        lr = code in {"1", "2"}
        rows.append({"event_uid": f"Yang2025_2C:v5:sub-{pair['subject']:03d}:ses-{pair['session']:02d}:event-{event['original_event_index']}",
                     "original_event_index": event["original_event_index"],
                     "native_record_index": event["native_record_index"], "native_TAL_index": event["native_TAL_index"],
                     "onset_token": event["onset_token"], "onset_s": float(onset),
                     "onset_fraction_seconds": str(onset), "duration_fraction_seconds": str(event["duration_fraction"]),
                     "label_code": code, "label": int(code)-1 if lr else None,
                     "original_marker_sample": marker,
                     "marker_quantization_error_samples": str(Fraction(marker)-1000*onset),
                     "operational_separator_segment": segment, "physical_run_id": None,
                     "context_start_sample": marker+2250 if lr else None,
                     "context_stop_sample_exclusive": marker+3750 if lr else None,
                     "crop_start_sample": marker+2500 if lr else None,
                     "crop_stop_sample_exclusive": marker+3500 if lr else None})
        if code == "9":
            segment += 1
    return {"schema_version": "paper3-yang-native-inventory-record-v1", "dataset_id": "Yang2025_2C",
            "dataset_version": 5, "participant_id": f"Yang2025_2C:sub-{pair['subject']:03d}",
            "subject": pair["subject"], "session": pair["session"],
            "source_record_id": pair["signal"]["path"],
            "raw_files": {kind: pair[kind].copy() for kind in ("signal", "event")},
            "native_header_sha256": {k: v["header_sha256"] for k, v in headers.items()},
            "native_header_clocks": {k: v["start_date_time"] for k, v in headers.items()},
            "native_sfreq_hz": 1000, "common_gain_evidence": gain.copy(),
            "applied_event_offset_seconds": 0, "clock_registry_sha256": CLOCK_SHA,
            "eligibility_manifest_sha256": LABEL_SHA,
            "derived_before_waveform_preprocessing": True, "independent_second_parser_claim": False,
            "native_original_event_count": len(rows), "events": rows}


def materialize(raw_data_path, raw_evt_path, *, expected_sha256, profile):
    """Return (pre-RMS float32[N,11,128], int64 labels, events, metadata).

    expected_sha256 must have exactly data.bdf and evt.bdf keys; their values
    must match one frozen P1 2C pair and be rehashed on these current files.
    Profile is the separately frozen Yang source-only center-window artifact.
    """
    contract = _profile(profile)
    _require(isinstance(expected_sha256, dict) and set(expected_sha256) == {"data.bdf", "evt.bdf"}
             and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in expected_sha256.values()),
             "explicit_two_native_SHA256_values_required")
    paths = {"data.bdf": Path(raw_data_path), "evt.bdf": Path(raw_evt_path)}
    for name, path in paths.items():
        _require(path.suffix.lower() == ".bdf" and not path.is_symlink() and path.is_file(),
                 "native_regular_BDF_required_no_old_cache")
        _require(_sha(path) == expected_sha256[name], "current_full_native_SHA_mismatch")
    clock = _policy(profile, "clock_registry", "p1_yang_clock_registry_v5.json", CLOCK_SHA)
    labels = _policy(profile, "eligibility_manifest", "p1_eligibility_manifest.json", LABEL_SHA)
    pairs = [p for p in clock["pairs"] if p["signal"]["sha256"] == expected_sha256["data.bdf"]
             and p["event"]["sha256"] == expected_sha256["evt.bdf"]]
    _require(len(pairs) == 1, "native_pair_not_in_frozen_2C_registry")
    pair = pairs[0]
    headers = {name: _parse_header(path) for name, path in paths.items()}
    for name, kind in (("data.bdf", "signal"), ("evt.bdf", "event")):
        expected = pair[kind]
        _require(paths[name].stat().st_size == expected["bytes"] and
                 headers[name]["header_sha256"] == expected["header_sha256"] and
                 headers[name]["start_date_time"] == expected["start_date_time"],
                 "native_pair_header_or_clock_identity_changed")
    gain = _common_gain(headers["data.bdf"])
    observed = _read_annotations(paths["evt.bdf"], headers["evt.bdf"])
    eligible = [s for s in labels["sessions"] if (s["subject"], s["session"]) == (pair["subject"], pair["session"])]
    _require(len(eligible) == 1 and eligible[0]["sha256"] == expected_sha256["evt.bdf"],
             "explicit_label_registry_mismatch")
    expected = eligible[0]
    _require(dict(Counter(x["description"] for x in observed)) == expected["all_marker_counts"],
             "native_marker_counts_changed")
    if "all_events" in expected:
        projection = [{"onset": float(x["onset_fraction"]), "duration": float(x["duration_fraction"]),
                       "description": x["description"]} for x in observed]
        _require(projection == expected["all_events"], "reviewed_anomalous_event_sequence_changed")
    native_metadata = _native_facts(pair, headers, observed, gain)
    events, xs, ys, segment, retained = [], [], [], 0, 0
    n_native = headers["data.bdf"]["records"] * headers["data.bdf"]["channels"][gain["selected_indices"][0]]["samples_per_record"]
    for i, event in enumerate(observed):
        code, onset = event["description"], event["onset_fraction"]
        marker = _sample(onset)
        row = {k: v for k, v in event.items() if k not in {"onset_fraction", "duration_fraction"}}
        row.update(onset_s=float(onset), duration_s=float(event["duration_fraction"]),
                   original_marker_sample=marker, marker_quantization_error_samples=str(Fraction(marker)-onset*1000),
                   operational_separator_segment=segment, physical_run_id=None,
                   original_trial_id=f"native_event_{event['original_event_index']}",
                   event_uid=f"Yang2025_2C:v5:sub-{pair['subject']:03d}:ses-{pair['session']:02d}:event-{event['original_event_index']}",
                   label_code=code, valid=False, exclusion_reason="not_explicit_LR_label", cache_row=None)
        if code == "9":
            segment += 1
        if code in {"1", "2"}:
            row["label"] = int(code)-1
            row["retained_LR_ordinal"] = retained
            retained += 1
            start, stop = marker + 2250, marker + 3750
            next_marker = _sample(observed[i+1]["onset_fraction"]) if i+1 < len(observed) else n_native
            row.update(context_start_sample=start, context_stop_sample_exclusive=stop,
                       crop_start_sample=marker+2500, crop_stop_sample_exclusive=marker+3500,
                       analysis_anchor_sample=start, MI_onset_verified=False)
            if stop > min(n_native, marker+4000, next_marker):
                row["exclusion_reason"] = "insufficient_original_source_context"
            else:
                raw = _context(paths["data.bdf"], headers["data.bdf"], gain["selected_indices"], start, stop)
                try:
                    x, qc = prepare_trial(raw, channel_names=list(CHANNELS), sfreq_hz=1000,
                                          task_sample=0, task_end_sample=1500, task_end_verified=True,
                                          contract=contract, native_unit="common_gain_numeric_unresolved",
                                          common_positive_gain_verified=True)
                    converted = np.ascontiguousarray(x, dtype=np.float32)
                    _require(np.isfinite(converted).all() and np.any(converted != 0), "float32_cache_overflow_or_zero")
                except ValueError as exc:
                    # Shape/contract errors are programming/qualification failures, not convenient QC exclusions.
                    if str(exc) not in {"Nonfinite context", "Zero/nonfinite trial energy", "float32_cache_overflow_or_zero"}:
                        raise
                    row["exclusion_reason"] = "nonfinite_or_zero_trial_energy"
                else:
                    row.update(valid=True, exclusion_reason=None, cache_row=len(xs),
                               pre_rms_joint_energy_rms=qc["joint_rms"],
                               end_bound_scope="frozen_source_analysis_intersection_and_next_native_event",
                               physical_MI_end_verified=False)
                    xs.append(converted); ys.append(int(code)-1)
        events.append(row)
    x = np.stack(xs) if xs else np.empty((0, 11, 128), dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    signal_clock = datetime.strptime(headers["data.bdf"]["start_date_time"], "%d.%m.%y%H.%M.%S")
    event_clock = datetime.strptime(headers["evt.bdf"]["start_date_time"], "%d.%m.%y%H.%M.%S")
    metadata = {"schema_version": "paper3-yang-native-materialization-v1", "dataset_id": "Yang2025_2C",
                "dataset_version": 5, "role": "enhanced_source_only", "subject": pair["subject"], "session": pair["session"],
                "participant_id": native_metadata["participant_id"], "source_record_id": pair["signal"]["path"],
                "native_metadata": native_metadata,
                "profile_sha256": digest(profile), "signal_contract_sha256": INPUT_SHA,
                "native_files_sha256": expected_sha256.copy(), "native_header_sha256": {k: v["header_sha256"] for k, v in headers.items()},
                "clock_registry_sha256": CLOCK_SHA, "eligibility_manifest_sha256": LABEL_SHA,
                "native_header_clock_difference_seconds": (signal_clock-event_clock).total_seconds(),
                "applied_event_offset_seconds": 0, "clock_scope": "exact_frozen_native_pair_author_single_signal_TAL_convention_not_physical_sync_proof",
                "calibration": gain, "stored_unit": "unresolved", "stored_stage": "pre_rms_car11_temporal_demeaned",
                "normalization_applied": False, "joint_trial_RMS_required_before_model_use": True,
                "MI_onset_verified": False, "context_from_marker_s": [2.25, 3.75], "crop_from_marker_s": [2.5, 3.5],
                "shape": list(x.shape), "dtype": str(x.dtype), "native_marker_counts": dict(Counter(z["description"] for z in observed)),
                "explicit_LR_before_waveform_QC": retained, "valid_LR_trials": len(y),
                "excluded_explicit_LR_trials": retained-len(y), "unknown_255_retained_in_audit": sum(z["description"] == "255" for z in observed),
                "physical_run_identity_verified": False, "old_NPZ_reused": False, "fit_completed": False,
                "arrays_sha256": {"X_C_order_float32": hashlib.sha256(x.tobytes(order="C")).hexdigest(),
                                  "y_C_order_int64": hashlib.sha256(y.tobytes(order="C")).hexdigest()}}
    return x, y, events, metadata
