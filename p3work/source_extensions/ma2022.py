"""Ma2022 v1 native EDF -> separate, preparation-only CAR3 development cache.

This is not a main11 adapter, raw acquisition chronology, a B20 prefix, a fit,
final external evaluation, or an authorization to download any additional object.
The observed original event table conflicts with its generic JSON sidecar:
this interface accepts only a verified 1-based, 250 Hz sample clock and complete
retained 4 s windows. It never reconstructs rejected acquisition trials.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

CHANNELS = ("C3", "Cz", "C4")
CLINIC11 = ("FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4")
FIELDS = (("labels", 16), ("transducer", 80), ("units", 8),
          ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
          ("digital_max", 8), ("prefiltering", 80), ("samples_per_record", 8),
          ("reserved", 32))
CLOCK = "observed_1_based_native_samples_not_generic_sidecar_milliseconds"
STAGE = "Ma2022-independent-development-CAR3-preRMS-v1"


class Ma2022Error(ValueError):
    pass


def descriptor(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _verify(path, expected):
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise Ma2022Error("explicit_SHA256_required")
    item = descriptor(path)
    if item["sha256"] != expected:
        raise Ma2022Error("native_file_SHA256_mismatch")
    return item


def read_header(path):
    """Read only EDF metadata; reject malformed structure and selected calibration."""
    path = Path(path)
    with path.open("rb") as handle:
        common = handle.read(256)
        if len(common) != 256 or common[:8].strip() != b"0":
            raise Ma2022Error("ordinary_EDF_required")
        try:
            nchan = int(common[252:256])
            header_bytes = int(common[184:192])
            nrecords = int(common[236:244])
            seconds = float(common[244:252])
        except ValueError as error:
            raise Ma2022Error("malformed_EDF_header") from error
        if nchan != 32 or header_bytes != 256 * (nchan + 1) or nrecords <= 0 or seconds != 1.:
            raise Ma2022Error("Ma2022_v1_32channel_1second_records_required")
        block = handle.read(header_bytes - 256)
    if len(block) != header_bytes - 256:
        raise Ma2022Error("truncated_EDF_header")
    values, offset = {}, 0
    try:
        for field, width in FIELDS:
            values[field] = [block[offset + i*width:offset + (i+1)*width].decode("ascii").strip()
                             for i in range(nchan)]
            offset += nchan * width
        samples = [int(v) for v in values["samples_per_record"]]
    except (ValueError, UnicodeError) as error:
        raise Ma2022Error("malformed_EDF_signal_header") from error
    labels = values["labels"]
    folded = [name.casefold() for name in labels]
    if len(set(folded)) != nchan or any(c.casefold() not in folded for c in CHANNELS):
        raise Ma2022Error("unique_measured_C3_Cz_C4_required")
    if samples != [250]*nchan or nrecords % 4 or path.stat().st_size != header_bytes + nrecords*nchan*250*2:
        raise Ma2022Error("complete_250Hz_retained_4second_windows_required")
    select = [folded.index(c.casefold()) for c in CHANNELS]
    selected = []
    for ch in select:
        try:
            pmin, pmax = float(values["physical_min"][ch]), float(values["physical_max"][ch])
            dmin, dmax = int(values["digital_min"][ch]), int(values["digital_max"][ch])
        except ValueError as error:
            raise Ma2022Error("selected_channel_calibration_missing") from error
        if values["units"][ch] not in ("uV", "µV"):
            raise Ma2022Error("native_microvolt_header_required")
        if (not np.isfinite([pmin,pmax]).all() or pmax <= pmin or dmax <= dmin
                or dmin < -32768 or dmax > 32767):
            raise Ma2022Error("selected_channel_calibration_invalid")
        selected.append({"name": CHANNELS[len(selected)], "edf_index": ch,
                         "unit": values["units"][ch], "physical_min": pmin,
                         "physical_max": pmax, "digital_min": dmin, "digital_max": dmax})
    return {"channels": labels, "n_channels": nchan, "sfreq_hz": 250,
            "n_records": nrecords, "record_seconds": seconds, "header_bytes": header_bytes,
            "n_samples": nrecords*250, "retained_trials": nrecords//4,
            "stored_trial_samples": 1000, "selected_calibration": selected,
            "clinic11_present": [c for c in CLINIC11 if c.casefold() in folded],
            "clinic11_missing": [c for c in CLINIC11 if c.casefold() not in folded],
            "prefiltering_header_uninformative": all("NaN" in x for x in values["prefiltering"]),
            "recording_is_concatenated_retained_windows": True,
            "original_acquisition_ordinals_available": False}


def _read_event_table(path, header):
    """Validate declared columns or the observed headerless six-column layout.

    Some official v1 TSVs omit the header. The first numeric row is a genuine
    retained trial and is checked by exactly the same full extent/clock/label
    rules as every other row. No missing row, sample or label is invented.
    """
    try:
        physical = [[v.strip() for v in row] for row in
                    csv.reader(io.StringIO(Path(path).read_text()), delimiter="\t")]
        required = {"onset", "duration", "trial_type", "sample", "value"}
        if not physical:
            raise Ma2022Error("empty_native_event_table")
        first = physical[0]
        declared = (len(first) == len(set(first)) and required <= set(first)
                    and set(first) <= required | {"response_time"})
        if declared:
            columns, data, first_data_line = first, physical[1:], 2
        else:
            # The only undeclared layout actually observed in original v1.
            columns = ["onset", "duration", "trial_type", "response_time", "sample", "value"]
            data, first_data_line = physical, 1
            if len(first) != 6:
                raise Ma2022Error("headerless_native_six_column_layout_required")
        if any(len(row) != len(columns) for row in data):
            raise Ma2022Error("native_event_column_count_mismatch")
        rows = [dict(zip(columns, row)) for row in data]
        if len(rows) != header["retained_trials"]:
            raise Ma2022Error("event_count_EDF_extent_mismatch")
        output = []
        for index, row in enumerate(rows):
            onset, sample, duration, value = [float(row[k]) for k in ("onset", "sample", "duration", "value")]
            expected = 1 + index*1000
            if onset != expected or sample != expected or duration != 1000:
                raise Ma2022Error("observed_1based_sample_clock_required_no_ms_guess")
            if value not in (1.,2.) or row["trial_type"].strip() != {1.:"left",2.:"right"}[value]:
                raise Ma2022Error("native_hand_label_inconsistent")
            if "response_time" in row and row["response_time"] != "n/a":
                raise Ma2022Error("native_no_response_time_column_required")
            output.append({"retained_trial_index": index, "analysis_anchor_sample": index*1000,
                           "task_end_sample": (index+1)*1000, "label": int(value)-1,
                           "canonical_label": {1.:"left_hand",2.:"right_hand"}[value],
                           "raw_label_code": int(value), "original_acquisition_ordinal": None,
                           "physical_run_id": None, "original_paid_B20_prefix_recoverable": False,
                           "native_event_data_line_1based": first_data_line+index,
                           "native_event_header_present": declared})
        return output, {"parser_version":"Ma2022-native-event-layout-v2",
                        "header_present":declared,"columns":columns,
                        "first_data_line_1based":first_data_line,"physical_lines":len(physical),
                        "data_rows":len(rows),"missing_trials_inserted":0,"rows_discarded":0}
    except (KeyError, UnicodeError, ValueError) as error:
        if isinstance(error, Ma2022Error):
            raise
        raise Ma2022Error("malformed_native_event_table") from error


def read_events(path, header):
    """Return every real native event, preserving a headerless first trial."""
    return _read_event_table(path,header)[0]


def audit_pair(edf, events, *, edf_sha256, events_sha256,
               reference_evidence="official_v1_sidecar_unipolar_M1"):
    if reference_evidence != "official_v1_sidecar_unipolar_M1":
        raise Ma2022Error("verified_common_monopolar_reference_required")
    identity = {"edf": _verify(edf, edf_sha256), "events": _verify(events, events_sha256)}
    header = read_header(edf)
    rows, layout = _read_event_table(events, header)
    return {"schema_version": "paper3-Ma2022-native-development-audit-v1",
            "dataset": "Ma2022", "provider_version": 1, "role": "independent_development",
            "identity": identity, "header": header, "events": rows,
            "native_event_table_layout": layout,
            "event_clock": CLOCK, "event_clock_generic_sidecar_conflict": True,
            "reference_evidence": reference_evidence,
            "reference_sidecar_sha256": "2a609a657f6741b0433af7565c20fd4536919948c57303ef661e30f98a05ed3a",
            "label_definition_sidecar_sha256": "2180f87d60d2eb40448033fde743ae2a61ea11265cff53a8945c750b1d39998b",
            "integrity_hash_reads_full_payload_bytes": True,
            "primary_11channel_compatible": False,
            "training_ready": False, "scientific_pool_assignment_frozen": False,
            "actual_fits": 0, "external_evaluation": False,
            "prior_P1_P2_exposure": "unknown", "raw_acquisition_prefix_eligible": False}


def _context(memmap, header, anchor):
    times = anchor + np.arange(375)
    output = []
    for calibration in header["selected_calibration"]:
        digital = np.asarray(memmap[times//250, calibration["edf_index"], times%250], dtype=np.float64)
        if (digital.min() < calibration["digital_min"] or digital.max() > calibration["digital_max"]):
            raise Ma2022Error("digital_samples_exceed_header_calibration")
        gain = (calibration["physical_max"]-calibration["physical_min"])/(calibration["digital_max"]-calibration["digital_min"])
        output.append(((digital-calibration["digital_min"])*gain+calibration["physical_min"])*1e-6)
    return np.stack(output)


def prepare_context(context_volts):
    """Per-window 1.5 s -> CAR3/8-30 Hz/128 Hz/[.25,1.25), no RMS division."""
    selected = np.asarray(context_volts, dtype=np.float64).copy()
    if selected.shape != (3,375) or not np.isfinite(selected).all():
        raise Ma2022Error("finite_3_by_375_context_required")
    selected -= selected.mean(axis=0, keepdims=True)
    sos = butter(4, (8,30), btype="bandpass", fs=250, output="sos")
    filtered = sosfiltfilt(sos, selected, axis=-1)
    result = resample_poly(filtered,64,125,axis=-1,padtype="line")[:,32:160].copy()
    result -= result.mean(axis=-1,keepdims=True)
    energy = float(np.sqrt(np.mean(result**2)))
    if result.shape != (3,128) or not np.isfinite(result).all() or energy <= 0:
        raise Ma2022Error("nonfinite_or_zero_energy_CAR3_crop")
    return result.astype("float32"), energy


def create_development_cache(edf, events, output, *, edf_sha256, events_sha256):
    """Write one SHA-bound, explicitly unassigned development shard, never fit."""
    audit = audit_pair(edf,events,edf_sha256=edf_sha256,events_sha256=events_sha256)
    output = Path(output)
    if output.exists():
        raise Ma2022Error("output_must_be_new_no_overwrite")
    header = audit["header"]
    memmap = np.memmap(edf,dtype="<i2",mode="r",offset=header["header_bytes"],
                       shape=(header["n_records"],32,250))
    x, y, rows = [], [], []
    for row in audit["events"]:
        prepared, energy = prepare_context(_context(memmap,header,row["analysis_anchor_sample"]))
        uid_data = {"edf_sha256":edf_sha256,"events_sha256":events_sha256,
                    "retained_trial_index":row["retained_trial_index"],"stage":STAGE}
        uid = hashlib.sha256(json.dumps(uid_data,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        x.append(prepared); y.append(row["label"])
        rows.append({**row,"event_uid":uid,"trial_uid":uid,"dataset":"Ma2022",
                     "role":"independent_development_unassigned", "stage":STAGE,
                     "native_sfreq_hz":250,"original_context_samples":[row["analysis_anchor_sample"],row["analysis_anchor_sample"]+375],
                     "native_unit":"uV","stored_unit":"V","reference":"CAR_exact_C3_Cz_C4",
                     "data_after_context_accessed":False,"normalization_applied":False,
                     "pre_RMS_joint_rms_volts":energy,"event_clock":CLOCK,
                     "processed_signal_sha256":hashlib.sha256(prepared.astype("<f4").tobytes()).hexdigest()})
    del memmap
    output.mkdir(parents=True,exist_ok=False)
    np.save(output/"X.npy",np.stack(x),allow_pickle=False)
    np.save(output/"y.npy",np.asarray(y,dtype="int64"),allow_pickle=False)
    (output/"events.json").write_text(json.dumps(rows,indent=2)+"\n")
    manifest = {k:v for k,v in audit.items() if k != "events"}
    manifest.update({"status":"SINGLE_NATIVE_OBJECT_CPU_PREPARATION_ONLY_NOT_COMPLETE_DEVELOPMENT",
        "stage":STAGE,"channels":list(CHANNELS),"sfreq_hz":128,"context_s":[0,1.5],
        "crop_s":[.25,1.25],"n_times":128,"band_hz":[8,30],"reference":"CAR_exact_C3_Cz_C4",
        "normalization_applied":False,"source_acquisition_order_recovered":False,
        "preprocessing_before_release":["provider_bad_trial_removal","baseline_correction","FIR_0.5_40Hz"],
        "past_to_future_session_evaluation_requires_frozen_assignments_and_more_native_sessions":True,
        "retained_trials":len(rows),"observed_people":1,"observed_sessions":1,
        "independent_external_people_added":0,"adapter_code":descriptor(__file__),
        "artifacts":{name:descriptor(output/name) for name in ("X.npy","y.npy","events.json")}})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest
