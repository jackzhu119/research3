"""Bounded Dreyer OSF-EDF source extension, separate from the frozen source reader.

Only the declared, previously exposed source branch is materialized. This module
does not download data, estimate a model/reference, grant query access, or fit.
Native EDF physical units are converted to SI by MNE exactly once. EDF trigger
durations are not MI durations: original code 800 bounds each 769/770 cue.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
from pathlib import Path
import re

import numpy as np

from p3work.contracts import CHANNELS, digest, file_digest, validate_signal_contract
from p3work.preprocess import prepare_trial


PROFILE_ID = "dreyer2023-osf-edf-original-cue-clinic11-v1"
CONTRACT_SHA256 = "5c8ea3f01634c060546af29b9330d314c4f03c265677a74b875ac7c302e31872"
SOURCE_VERSION = "NEMAR-nm000250-v1.0.5-sourcedata-OSF-EDF"
LABELS = {"769": 0, "770": 1}
# The original R2 EDF also contains the documented OpenViBE EndOfFile marker.
# It is retained in the native index and accepted only as a terminal zero-width
# bookkeeping event after ExperimentStop, never as an MI cue or trial boundary.
TERMINAL_EOF = "33284"
ONLINE_POST_TRIAL_CONTROLS = ("1010", "33281")  # GDF EndOfSession, OpenViBE Train
ALLOWED_MARKERS = frozenset({"32769", "32770", "768", "786", "33282", "769", "770", "781", "800", TERMINAL_EOF,
                             *ONLINE_POST_TRIAL_CONTROLS})


def _number(token):
    try:
        value = Decimal(token)
        if not value.is_finite():
            raise ValueError("Nonfinite native EDF clock")
        return Fraction(value)
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise ValueError("Finite original EDF decimal token required") from error


def _fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _validate_profile(profile):
    required = {
        "schema_version": "paper3-dreyer2023-native-source-profile-v1",
        "profile_id": PROFILE_ID,
        "role": "exposed_source_extension",
        "dataset_id": "Dreyer2023",
        "source_version": SOURCE_VERSION,
        "native_anchor": "original_769_770_cue",
        "native_end": "paired_800",
        "sampling_policy": "ceil_onset_floor_end",
        "native_sfreq_hz": 512,
        "native_units": ["uV", "µV"],
        "raw_label_to_y": LABELS,
        "raw_label_to_hand": {"769": "left_hand", "770": "right_hand"},
        "signal_contract_semantic_sha256": CONTRACT_SHA256,
        "external_query_access": False,
        "training_started": False,
    }
    if (type(profile) is not dict or any(profile.get(k) != v for k, v in required.items())
            or profile.get("external_query_access") is not False or profile.get("training_started") is not False):
        raise ValueError("Exact registered exposed-source native profile required")
    contract = profile.get("signal_contract")
    if type(contract) is not dict or validate_signal_contract(contract) != CONTRACT_SHA256:
        raise ValueError("Dreyer extension requires the unchanged 5c8e signal contract")
    return contract


def _identity(path):
    match = re.fullmatch(r"sub-(\d{2})_task-R([1-6])(acquisition|online)_eeg\.edf", path.name)
    if not match:
        raise ValueError("Original Dreyer subject/run EDF filename required")
    subject, run = int(match[1]), int(match[2])
    phase = "acquisition" if run <= 2 else "online"
    if (not 1 <= subject <= 87 or match[3] != phase
            or path.parent.name != "eeg" or path.parent.parent.name != f"sub-{subject:02d}"):
        raise ValueError("Native subject, run phase and original parent path disagree")
    if subject == 59 and run >= 5:
        raise ValueError("Original release has no subject59 R5/R6; missing runs cannot be invented")
    return subject, run, f"R{run}{phase}"


def _read_native_metadata(path):
    """Read fixed headers and TAL bytes only; never read EEG arrays."""
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256 or fixed[:8].strip() != b"0":
            raise ValueError("Original 16-bit EDF header required")
        ns = int(fixed[252:256])
        records = int(fixed[236:244])
        header_bytes = int(fixed[184:192])
        record_token = fixed[244:252].decode("ascii").strip()
        record_duration = _number(record_token)
        if (ns != 33 or records <= 0 or record_duration != 1
                or header_bytes != 256 * (ns + 1) or not fixed[192:236].startswith(b"EDF+C")):
            raise ValueError("Registered 32-signal continuous EDF+C 1-second record clock required")
        header = stream.read(header_bytes - 256)
        if len(header) != header_bytes - 256:
            raise ValueError("Incomplete native EDF signal header")
        labels = [header[16*i:16*(i+1)].decode("ascii").strip() for i in range(ns)]
        units = [header[96*ns+8*i:96*ns+8*(i+1)].decode("latin1").strip() for i in range(ns)]
        physical_min = [_number(header[104*ns+8*i:104*ns+8*(i+1)].decode("ascii").strip()) for i in range(ns)]
        physical_max = [_number(header[112*ns+8*i:112*ns+8*(i+1)].decode("ascii").strip()) for i in range(ns)]
        digital_min = [int(header[120*ns+8*i:120*ns+8*(i+1)]) for i in range(ns)]
        digital_max = [int(header[128*ns+8*i:128*ns+8*(i+1)]) for i in range(ns)]
        samples = [int(header[216*ns+8*i:216*ns+8*(i+1)]) for i in range(ns)]
        if len(set(labels)) != ns or labels.count("EDF Annotations") != 1 or any(n <= 0 for n in samples):
            raise ValueError("Unique native signals and exactly one TAL channel required")
        ai = labels.index("EDF Annotations")
        signal_indices = [i for i in range(ns) if i != ai]
        if any(samples[i] != 512 for i in signal_indices):
            raise ValueError("Every original signal must share the registered 512Hz native clock")
        missing = set(CHANNELS) - set(labels)
        if missing:
            raise ValueError("Missing measured clinic11 electrodes")
        selected = [labels.index(c) for c in CHANNELS]
        if any(units[i] not in {"uV", "µV"} or physical_max[i] <= physical_min[i]
               or digital_max[i] <= digital_min[i] for i in selected):
            raise ValueError("Unresolved original EDF physical units or ADC scale")
        stride, offset, count = 2*sum(samples), 2*sum(samples[:ai]), 2*samples[ai]
        if path.stat().st_size != header_bytes + records * stride:
            raise ValueError("Complete native EDF byte count differs from declared records")
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
                timing = fields[0].split("\x15")
                if len(timing) not in {1, 2}:
                    raise ValueError("Ambiguous native TAL clock")
                onset = _number(timing[0])
                descriptions = [value for value in fields[1:] if value]
                if not descriptions:
                    if len(timing) != 1:
                        raise ValueError("Timekeeping TAL cannot be an event")
                    clocks.append(onset)
                    continue
                if (len(descriptions) != 1 or len(timing) != 2
                        or descriptions[0] not in ALLOWED_MARKERS):
                    raise ValueError("Unknown original Dreyer trigger label or TAL structure")
                duration = _number(timing[1])
                if onset < 0 or duration < 0 or duration > Fraction(2, 512):
                    raise ValueError("Registered nonnegative impulse width required, not a task duration")
                annotations.append({"original_annotation_index": len(annotations),
                    "data_record": record, "onset_token": timing[0], "duration_token": timing[1],
                    "description": descriptions[0], "onset": onset, "duration": duration})
            if clocks != [record * record_duration]:
                raise ValueError("Native EDF data-record timekeeping is discontinuous")
        if (not annotations or any(b["onset"] < a["onset"] for a, b in zip(annotations, annotations[1:]))):
            raise ValueError("Original marker acquisition order is not monotonic")
    return {"n_times": records*512, "record_count": records, "record_duration_token": record_token,
        "labels": labels, "signal_labels": [labels[i] for i in signal_indices],
        "signal_units": {labels[i]: units[i] for i in signal_indices},
        "selected_indices": selected, "annotations": annotations}


def _trials(native, *, run_phase=None):
    annotations = native["annotations"]
    counts = Counter(e["description"] for e in annotations)
    stop_index = len(annotations)-1
    if counts[TERMINAL_EOF]:
        eof = annotations[-1]
        if (counts[TERMINAL_EOF] != 1 or eof["description"] != TERMINAL_EOF
                or len(annotations) < 2 or annotations[-2]["description"] != "32770"
                or eof["duration"] != 0
                or not annotations[-2]["onset"] <= eof["onset"] < Fraction(native["n_times"],512)):
            raise ValueError("Only one terminal zero-width OpenViBE EndOfFile after ExperimentStop is allowed")
        stop_index -= 1
    if (counts["32769"] != 1 or counts["32770"] != 1
            or annotations[0]["description"] != "32769" or annotations[stop_index]["description"] != "32770"
            or counts["768"] != 40 or counts["769"] != 20 or counts["770"] != 20):
        raise ValueError("Original run markers and forty balanced native MI trials required")
    trial_region_end = stop_index
    if any(counts[code] for code in ONLINE_POST_TRIAL_CONTROLS):
        if (run_phase == "acquisition" or any(counts[code] != 1 for code in ONLINE_POST_TRIAL_CONTROLS)
                or [e["description"] for e in annotations[stop_index-2:stop_index]] != list(ONLINE_POST_TRIAL_CONTROLS)
                or annotations[stop_index-3]["description"] != "800"
                or not annotations[stop_index-3]["onset"] < annotations[stop_index-2]["onset"]
                or not annotations[stop_index-2]["onset"] < annotations[stop_index-1]["onset"]
                    < annotations[stop_index]["onset"]):
            raise ValueError("Only ordered online post-trial EndOfSession/Train before ExperimentStop is allowed")
        trial_region_end -= 2
    starts = [i for i, e in enumerate(annotations) if e["description"] == "768"]
    rows = []
    tolerance = Fraction(2, 512)
    for trial, begin in enumerate(starts, 1):
        finish = starts[trial] if trial < len(starts) else trial_region_end
        block = annotations[begin:finish]
        if (len(block) != 6 or [e["description"] for e in block[:3]] != ["768", "786", "33282"]
                or block[3]["description"] not in LABELS
                or [e["description"] for e in block[4:]] != ["781", "800"]):
            raise ValueError("Original start/fixation/audio/MI cue/feedback/end trial sequence required")
        start, fixation, audio, cue, feedback, end = block
        if (start["onset"] != fixation["onset"] or not start["onset"] < audio["onset"]
                < cue["onset"] < feedback["onset"] < end["onset"]):
            raise ValueError("Trial marker order or MI end is invalid")
        for observed, expected in [(audio["onset"]-start["onset"], 2),
                                   (cue["onset"]-start["onset"], 3),
                                   (feedback["onset"]-cue["onset"], Fraction(5,4)),
                                   (end["onset"]-cue["onset"], 5)]:
            if abs(observed-expected) > tolerance:
                raise ValueError("Native cue/feedback/end timing differs from the registered protocol")
        bound = min(end["onset"], annotations[finish]["onset"], Fraction(native["n_times"],512))
        sample, stop = math.ceil(cue["onset"]*512), math.floor(bound*512)
        if not 0 <= sample < sample+768 <= stop <= native["n_times"]:
            raise ValueError("Insufficient genuine 1.5-second MI context; no padding permitted")
        rows.append({"original_trial_index": trial,
            "original_annotation_index": cue["original_annotation_index"],
            "raw_label_code": cue["description"], "canonical_label": "left_hand" if cue["description"] == "769" else "right_hand",
            "label": LABELS[cue["description"]], "analysis_anchor_sample": sample,
            "task_end_sample": stop, "original_context_samples": [sample, sample+768],
            "native_cue_onset_token": cue["onset_token"], "native_cue_impulse_duration_token": cue["duration_token"],
            "native_feedback_onset_token": feedback["onset_token"], "native_end_onset_token": end["onset_token"],
            "native_trial_start_onset_token": start["onset_token"],
            "native_cue_onset_fraction": _fraction(cue["onset"]), "native_bounded_end_fraction": _fraction(bound),
            "native_cue_was_exact_sample": (cue["onset"]*512).denominator == 1,
            "inward_start_delay_seconds": float(Fraction(sample,512)-cue["onset"]),
            "inward_end_loss_seconds": float(bound-Fraction(stop,512)),
            "task_anchor_evidence": "Original EDF TAL code769/770 cue, paired original800 end; trigger width is not MI duration",
            "acquisition_order_verified": True, "mi_onset_verified": True})
    return rows


def audit_native(raw_path, *, expected_sha256, profile):
    """Issue native header/TAL trial evidence without reading EEG sample arrays.

    Controllers persist this result separately before numerical preprocessing.
    Native trial membership/label/bounds can then be checked independently of
    the cache events, including for trials later excluded by waveform QC.
    """
    _validate_profile(profile)
    path = Path(raw_path)
    subject, run, run_id = _identity(path)
    if (not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or file_digest(path) != expected_sha256):
        raise ValueError("Complete native EDF SHA256 differs from its declared object")
    native = _read_native_metadata(path)
    # Every original marker must lie in the physically recorded file clock,
    # including the non-MI controls that follow the final complete trial.
    if any(e["onset"]+e["duration"] > Fraction(native["n_times"],512) for e in native["annotations"]):
        raise ValueError("Original annotation lies outside the physical EDF clock")
    adapter_sha = file_digest(__file__)
    rows = []
    for original in _trials(native,run_phase="acquisition" if run <= 2 else "online"):
        row = dict(original, dataset="Dreyer2023", role="exposed_source_extension",
            provider_subject=str(subject), provider_subject_id=str(subject), session_id="provider_single_day",
            source_version=SOURCE_VERSION, source_file_sha256=expected_sha256, source_file_bytes=path.stat().st_size,
            source_record_id=path.name, physical_run_id=run_id, run_order=run,
            original_native_unit="µV", native_sfreq_hz=512, selected_channel_order=list(CHANNELS),
            signal_contract_sha256=CONTRACT_SHA256, native_profile_sha256=digest(profile), adapter_sha256=adapter_sha)
        row["trial_uid"] = digest({k: row[k] for k in ("dataset", "provider_subject", "physical_run_id",
            "source_file_sha256", "original_annotation_index", "native_cue_onset_token", "raw_label_code")})
        rows.append(row)
    if file_digest(path) != expected_sha256 or file_digest(__file__) != adapter_sha:
        raise ValueError("Native source or adapter changed during independent metadata audit")
    return {"schema_version": "paper3-dreyer-independent-native-metadata-v1",
        "status": "NATIVE_HEADER_TAL_VERIFIED_NO_EEG_ARRAY_READ", "dataset": "Dreyer2023",
        "role": "exposed_source_extension", "provider_subject_id": str(subject),
        "session_id": "provider_single_day", "physical_run_id": run_id,
        "source_version": SOURCE_VERSION, "source_file_sha256": expected_sha256,
        "source_file_bytes": path.stat().st_size, "native_n_times": native["n_times"],
        "native_record_count": native["record_count"], "native_sfreq_hz": 512,
        "native_signal_order": native["signal_labels"], "native_signal_units": native["signal_units"],
        "selected_channel_order": list(CHANNELS), "original_trials": len(rows),
        "native_marker_counts": dict(Counter(e["description"] for e in native["annotations"])),
        "non_task_marker_policy": "optional_ordered_online_EndOfSession_Train_after_last_trial_before_ExperimentStop_then_optional_terminal_zero_width_EndOfFile",
        "post_trial_control_annotation_indices": [e["original_annotation_index"] for e in native["annotations"]
            if e["description"] in ONLINE_POST_TRIAL_CONTROLS],
        "terminal_eof_annotation_indices": [e["original_annotation_index"] for e in native["annotations"]
            if e["description"] == TERMINAL_EOF],
        "native_annotations": [{k: e[k] for k in ("original_annotation_index", "data_record", "onset_token",
            "duration_token", "description")} for e in native["annotations"]],
        "signal_contract_sha256": CONTRACT_SHA256, "native_profile_sha256": digest(profile),
        "adapter_sha256": adapter_sha, "trials": rows, "waveform_QC_applied": False,
        "EEG_array_samples_read": 0, "actual_fits": 0, "network_operations": 0, "R2_operations": 0}


def materialize(raw_path, *, expected_sha256, profile):
    """Return (float32 N×11×128 X, int64 y, original-event rows, metadata).

    Native structure errors block the file. Waveform QC failures are recorded as
    explicit excluded original trials; they never cause padding or fake trials.
    This API has no model/optimizer or network operation.
    """
    contract = _validate_profile(profile)
    path = Path(raw_path)
    subject, run, run_id = _identity(path)
    independent = audit_native(path, expected_sha256=expected_sha256, profile=profile)
    native = _read_native_metadata(path)
    rows = independent["trials"]
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    arrays, labels, events, excluded = [], [], [], []
    adapter_sha = file_digest(__file__)
    try:
        if (raw.first_samp != 0 or raw.info["sfreq"] != 512 or int(raw.n_times) != native["n_times"]
                or list(raw.ch_names) != native["signal_labels"] or len(raw.annotations) != len(native["annotations"])):
            raise ValueError("MNE/native header, signal order or annotation count disagree")
        for channel in CHANNELS:
            if raw._orig_units.get(channel) not in {"uV", "µV"} or int(raw.info["chs"][raw.ch_names.index(channel)]["unit"]) != 107:
                raise ValueError("MNE SI voltage decoding differs from the original physical-unit evidence")
        for original, decoded in zip(native["annotations"], raw.annotations):
            if (str(decoded["description"]) != original["description"]
                    or abs(_number(str(float(decoded["onset"]))) - original["onset"]) > Fraction(1,10**9)
                    or abs(_number(str(float(decoded["duration"]))) - original["duration"]) > Fraction(1,10**9)):
                raise ValueError("MNE annotation differs from the original TAL clock/order")
        for row in rows:
            row = dict(row, data_returned_by_mne_unit="V", SI_conversion_already_applied_by_MNE=True, state="valid")
            start, stop = row["original_context_samples"]
            signal = raw.get_data(picks=list(CHANNELS), start=start, stop=stop)
            if signal.shape != (11,768):
                raise ValueError("Native bounded signal shape differs from 1.5-second context")
            try:
                x, qc = prepare_trial(signal, channel_names=list(CHANNELS), sfreq_hz=512,
                    task_sample=0, task_end_sample=768, task_end_verified=True, contract=contract, native_unit="V")
            except ValueError as error:
                row.update(state="excluded", exclusion_reason=str(error))
                excluded.append(row)
                continue
            x = np.asarray(x, dtype=np.float32)
            if x.shape != (11,128) or not np.isfinite(x).all():
                raise ValueError("Invalid final source trial shape/finite values")
            row.update(cache_row=len(events), preprocessing=qc)
            arrays.append(x)
            labels.append(row["label"])
            events.append(row)
    finally:
        raw.close()
    if file_digest(path) != expected_sha256 or file_digest(__file__) != adapter_sha:
        raise ValueError("Native source file or adapter changed during materialization")
    X = np.stack(arrays) if arrays else np.empty((0,11,128), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    metadata = {"schema_version": "paper3-dreyer-source-materialization-v1",
        "status": "REAL_SOURCE_CACHE_MATERIALIZED_NOT_FITTED", "dataset": "Dreyer2023",
        "role": "exposed_source_extension", "source_version": SOURCE_VERSION,
        "prior_paper1_external_exposure": True, "new_unseen_external_claim": False,
        "source_file_sha256": expected_sha256, "source_file_bytes": path.stat().st_size,
        "provider_subject_id": str(subject), "session_id": "provider_single_day", "physical_run_id": run_id,
        "original_trials": len(rows), "valid_trials": len(events), "excluded_trials": len(excluded), "exclusions": excluded,
        "original_class_counts": dict(Counter(row["canonical_label"] for row in rows)),
        "valid_class_counts": dict(Counter(row["canonical_label"] for row in events)),
        "native_marker_counts": dict(Counter(e["description"] for e in native["annotations"])),
        "native_sfreq_hz": 512, "native_n_times": native["n_times"], "native_record_count": native["record_count"],
        "native_signal_order": native["signal_labels"], "native_signal_units": native["signal_units"],
        "selected_channels": list(CHANNELS), "MNE_version": mne.__version__,
        "SI_conversion": "Actual EDF microvolt dimensions decoded by MNE into volts exactly once; prepare_trial native_unit=V",
        "signal_contract_sha256": CONTRACT_SHA256, "native_profile_sha256": digest(profile),
        "adapter_sha256": adapter_sha, "preprocess_file_sha256": file_digest(Path(__file__).parents[1]/"preprocess.py"),
        "independent_native_metadata_semantic_sha256": digest(independent),
        "stage": "pre_rms_car11_temporal_demeaned", "normalization_applied": False,
        "synthetic": False, "old_NPZ_reused": False, "fit_started": False, "actual_fits": 0,
        "GPU_actions": 0, "network_operations": 0, "R2_operations": 0}
    return X, y, events, metadata
