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


# This registry comes only from the all-520 native TAL qualification, exact
# 522-object full-GET receipt and closed conserved body ledger. These are
# previously exposed source records, never new unseen external evaluation.
QUALIFICATION_PLAN_SHA256 = "f39a24a3d26c6eb3de6e286e2efb8afc032397b91626f84fd647ae2c182b5efd"
QUALIFICATION_REVIEW_SHA256 = "2a769a20353f1f478a57fd06333c6b1745938e99b820d88e417df96891d5cc37"
MISSING_STOP_RECORDS = {('081bff4ccbae4e0cc436d8c54b33947f91d53cd2e3c8fdc90db2025bd1be4180', 'sub-85_task-R6online_eeg.edf'): ('1010',
                                                                                                        '33281',
                                                                                                        '33284'),
 ('19d41e3d2d8f9bc6152bfdf77176547f018c4d3455da51f6968ed682324b9947', 'sub-85_task-R4online_eeg.edf'): ('1010',
                                                                                                        '33281',
                                                                                                        '33284'),
 ('1f06d2a046ca15e6f22dcd33ee67682120e33106b0c27d41c4a7b7b69275f66b', 'sub-31_task-R6online_eeg.edf'): ('1010',
                                                                                                        '33281'),
 ('2034bc4029a0a601694e0ae0e8b28b8e9f12ccf40af8ec26f6d3b2f8dd6ba69e', 'sub-43_task-R3online_eeg.edf'): ('1010',
                                                                                                        '33281',
                                                                                                        '33284'),
 ('3265baded70e0817b71b1b5cab57327b7c9a2d36b447c514395e9ec738e7c004', 'sub-22_task-R3online_eeg.edf'): ('1010',
                                                                                                        '33281'),
 ('352342d2a9c645330fe3329c604d1fc4586474ab39293376482f3e05b45ff3d3', 'sub-43_task-R2acquisition_eeg.edf'): ('1010',
                                                                                                             '33281'),
 ('35f01dd89d93f958ce2035a3d68d5c20686cf41b0c5c7ebc51bd415941147454', 'sub-48_task-R1acquisition_eeg.edf'): ('1010',
                                                                                                             '33281'),
 ('3cffe5bd3a7518d815b0688a6a5fd25c36e0247feb2cff55ad20d5a1612623ef', 'sub-33_task-R2acquisition_eeg.edf'): ('1010',
                                                                                                             '33281'),
 ('420ba3d3142d2e96da72b8284b35fa7dfb9fb77d0a7324450d49ba5ada50d153', 'sub-45_task-R5online_eeg.edf'): ('1010',
                                                                                                        '33281',
                                                                                                        '33284'),
 ('430e00b8e2aa9bb606433ec3469c36b14fbb8a420fcd8abae2221cd0e9736d3c', 'sub-48_task-R6online_eeg.edf'): ('1010',
                                                                                                        '33281',
                                                                                                        '33284'),
 ('56e36b6a38477c6cde54825ebb0d14ec4a3cd7e226649f6d1c749f8888d57891', 'sub-25_task-R2acquisition_eeg.edf'): ('1010',
                                                                                                             '33281'),
 ('5877201a8b119119137061be7aaebcfdea55c1acb381b87f0225fad065f8f3b3', 'sub-31_task-R5online_eeg.edf'): ('1010',
                                                                                                        '33281'),
 ('625da7061cad74793cb3a35e51569cde96a1c94a5814017c4b2eb1e6fc948944', 'sub-22_task-R4online_eeg.edf'): ('1010',),
 ('662f950163c5339ad044f1c94669999a2988973b84ff32fb9ecbb84fb562a43c', 'sub-27_task-R1acquisition_eeg.edf'): ('1010',
                                                                                                             '33281'),
 ('783a36d13235e3e50566c8ac76119632b580b6432d57d46e1fb12ccc773b468e', 'sub-43_task-R4online_eeg.edf'): ('1010',
                                                                                                        '33281',
                                                                                                        '33284'),
 ('8352114acc4c59f3c4128039fb2b237cf279405bf125f28f5e5327bbd18c1efa', 'sub-27_task-R3online_eeg.edf'): ('1010',
                                                                                                        '33281'),
 ('b4e1ff3656bd04016f0d27149465850f115b36028f5dc55345d73fbda754f1bd', 'sub-27_task-R5online_eeg.edf'): ('1010',
                                                                                                        '33281'),
 ('bf936fa35701c71c593278f65703ae63bd5bb808ba549992ba6a98bcd3393cfd', 'sub-33_task-R1acquisition_eeg.edf'): ('1010',
                                                                                                             '33281'),
 ('c899d7baa090cc4a08aeb0917deab17a1bf2db5f0261d4da8ab093a0c92e267d', 'sub-48_task-R3online_eeg.edf'): ('1010',
                                                                                                        '33281',
                                                                                                        '33284'),
 ('dd3626c31d6b305a0cf2dcd6dbb20b44eb6ab4720a7a95acadb506a02edf9b1a', 'sub-59_task-R2acquisition_eeg.edf'): ('1010',
                                                                                                             '33281'),
 ('ed30d90d15dacdedc9fdfd2a99e7c9a26f09918cdc56b5b7564286ce5ae6591d', 'sub-29_task-R5online_eeg.edf'): ('1010',
                                                                                                        '33281'),
 ('f85500929531dabe5b0dbd018fc92617ad83a70e5037d047146ec6f1c7c9b102', 'sub-48_task-R5online_eeg.edf'): ('1010',
                                                                                                        '33281',
                                                                                                        '33284')}
PARTIAL_SOURCE_SHA256 = "204dc887c651676b4bfd9f0c9c664de78004637c30d9657d33655690502d7b21"
PARTIAL_SOURCE_RECORD_ID = "sub-40_task-R3online_eeg.edf"
PARTIAL_SOURCE_BYTES = 11648932
# Only these two qualified raw objects contain the zero-width EndOfFile
# exactly at the physical recording end. It remains bookkeeping, never data.
MISSING_STOP_EOF_AT_FILE_END_RECORDS = frozenset({
    ("19d41e3d2d8f9bc6152bfdf77176547f018c4d3455da51f6968ed682324b9947", "sub-85_task-R4online_eeg.edf"),
    ("081bff4ccbae4e0cc436d8c54b33947f91d53cd2e3c8fdc90db2025bd1be4180", "sub-85_task-R6online_eeg.edf"),
})


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


def _legacy_trial_layout(native, *, run_phase=None):
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
        # These documented session/training controls also occur in original
        # acquisition files (for example provider02 R1). Their position and
        # order, rather than the filename's phase, establish that they lie
        # outside all forty complete MI trials.
        if (any(counts[code] != 1 for code in ONLINE_POST_TRIAL_CONTROLS)
                or [e["description"] for e in annotations[stop_index-2:stop_index]] != list(ONLINE_POST_TRIAL_CONTROLS)
                or annotations[stop_index-3]["description"] != "800"
                or not annotations[stop_index-3]["onset"] < annotations[stop_index-2]["onset"]
                or not annotations[stop_index-2]["onset"] < annotations[stop_index-1]["onset"]
                    < annotations[stop_index]["onset"]):
            raise ValueError("Only ordered post-trial EndOfSession/Train before ExperimentStop is allowed")
        trial_region_end -= 2
    starts = [i for i, e in enumerate(annotations) if e["description"] == "768"]
    return starts, trial_region_end


def _qualified_annotations(native):
    annotations = native["annotations"]
    if (not annotations or type(native.get("n_times")) is not int or native["n_times"] <= 0
            or [e["original_annotation_index"] for e in annotations] != list(range(len(annotations)))
            or any(e["description"] not in ALLOWED_MARKERS or e["onset"] < 0
                or e["duration"] < 0 or e["duration"] > Fraction(2,512)
                or e["onset"]+e["duration"] > Fraction(native["n_times"],512) for e in annotations)
            or any(b["onset"] < a["onset"] for a,b in zip(annotations,annotations[1:]))):
        raise ValueError("Qualified source requires original ordered, bounded native annotations")
    return annotations


def _missing_stop_layout(native, *, source_sha256, source_record_id):
    expected = MISSING_STOP_RECORDS.get((source_sha256,source_record_id))
    if expected is None:
        raise ValueError("Missing ExperimentStop requires an exact qualified raw SHA256 and original record identity")
    annotations = _qualified_annotations(native)
    codes = [e["description"] for e in annotations]
    counts = Counter(codes)
    starts = [i for i,e in enumerate(annotations) if e["description"] == "768"]
    trial_region_end = 241
    if (len(annotations) != trial_region_end+len(expected) or codes[:1] != ["32769"]
            or counts["32769"] != 1 or counts["32770"] != 0
            or starts != list(range(1,241,6)) or counts["769"] != 20 or counts["770"] != 20
            or tuple(codes[trial_region_end:]) != expected
            or annotations[trial_region_end-1]["description"] != "800"):
        raise ValueError("Qualified missing-stop record must retain forty complete balanced MI blocks and its exact observed tail")
    suffix = annotations[trial_region_end:]
    if (not annotations[trial_region_end-1]["onset"] < suffix[0]["onset"]
            or any(b["onset"] <= a["onset"] for a,b in zip(suffix,suffix[1:]))):
        raise ValueError("Qualified post-trial bookkeeping tail must be strictly ordered after the last paired800")
    if expected[-1] == TERMINAL_EOF:
        file_end = Fraction(native["n_times"],512)
        exact_end_allowed = (source_sha256,source_record_id) in MISSING_STOP_EOF_AT_FILE_END_RECORDS
        if (suffix[-1]["duration"] != 0 or suffix[-1]["onset"] > file_end
                or (suffix[-1]["onset"] == file_end and not exact_end_allowed)):
            raise ValueError("Only the qualified terminal zero-width EndOfFile clock is allowed")
    return starts, trial_region_end


def _partial_trial_layout(native, *, source_sha256, source_record_id):
    if (source_sha256,source_record_id) != (PARTIAL_SOURCE_SHA256,PARTIAL_SOURCE_RECORD_ID):
        raise ValueError("Partial source eligibility requires the exact reviewed raw SHA256 and original record identity")
    annotations = _qualified_annotations(native)
    counts = Counter(e["description"] for e in annotations)
    starts = [i for i,e in enumerate(annotations) if e["description"] == "768"]
    prefix = annotations[193:]
    prefix_fields = [(e["original_annotation_index"],e["onset_token"],e["duration_token"],e["description"])
                     for e in prefix]
    if (native["n_times"] != 181248 or native.get("record_count") != 354
            or len(annotations) != 195 or annotations[0]["description"] != "32769"
            or counts != Counter({"32769":1,"768":33,"786":33,"33282":32,
                                  "769":14,"770":18,"781":32,"800":32})
            or starts != list(range(1,194,6))
            or prefix_fields != [(193,"+352","0.0019","768"),(194,"+352","0.0019","786")]
            or any(e["onset"] != Fraction(352) or e["duration"] != Fraction(19,10000) for e in prefix)
            or annotations[192]["description"] != "800" or not annotations[192]["onset"] < Fraction(352)):
        raise ValueError("Reviewed sub40R3 requires exactly32 genuine MI blocks and its original terminal no-cue768/786 prefix")
    # The terminal prefix is never passed into the MI trial loop, labelled,
    # assigned an MI UID, padded, balanced, or mixed with waveform QC exclusions.
    return starts[:-1], 193


def _trial_layout(native, *, run_phase=None, source_sha256=None, source_record_id=None):
    special_sha = {key[0] for key in MISSING_STOP_RECORDS}
    special_record = {key[1] for key in MISSING_STOP_RECORDS}
    if source_sha256 == PARTIAL_SOURCE_SHA256 or source_record_id == PARTIAL_SOURCE_RECORD_ID:
        return _partial_trial_layout(native,source_sha256=source_sha256,source_record_id=source_record_id)
    if (source_sha256 in special_sha or source_record_id in special_record
            or not any(e["description"] == "32770" for e in native["annotations"])):
        return _missing_stop_layout(native,source_sha256=source_sha256,source_record_id=source_record_id)
    return _legacy_trial_layout(native,run_phase=run_phase)


def _trials(native, *, run_phase=None, source_sha256=None, source_record_id=None):
    annotations = native["annotations"]
    starts, trial_region_end = _trial_layout(native,run_phase=run_phase,
        source_sha256=source_sha256,source_record_id=source_record_id)
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



def _native_eligibility(native, rows, *, source_sha256, source_record_id):
    partial = (source_sha256,source_record_id) == (PARTIAL_SOURCE_SHA256,PARTIAL_SOURCE_RECORD_ID)
    is_registered_missing_stop = (source_sha256,source_record_id) in MISSING_STOP_RECORDS
    terminal_prefixes = []
    if partial:
        terminal_prefixes.append({"reason":"terminal_start_fixation_prefix_without_native_MI_cue_or_paired800",
            "original_annotations":[{k:e[k] for k in ("original_annotation_index","data_record","onset_token",
                "duration_token","description")} for e in native["annotations"][193:]],
            "cue_present":False,"paired800_present":False,"canonical_label_available":False,
            "MI_trial_UID_available":False,"included_in_MI_trials":False,"included_in_waveform_QC_exclusions":False})
    return {"schema_version":"paper3-dreyer-original-native-MI-eligibility-v1",
        "policy":"exact_source_identity_terminal_prefix_review" if partial else
            "exact_source_identity_observed_missing_stop_tail" if is_registered_missing_stop else "unchanged_legacy_complete_MI_grammar",
        "qualification_plan_sha256":QUALIFICATION_PLAN_SHA256,"qualification_review_sha256":QUALIFICATION_REVIEW_SHA256,
        "source_file_sha256":source_sha256,"source_record_id":source_record_id,
        "native_original_trial_start_markers":sum(e["description"] == "768" for e in native["annotations"]),
        "eligible_original_MI_trials":len(rows),"eligible_original_class_counts":dict(Counter(r["canonical_label"] for r in rows)),
        "experiment_stop_observed":any(e["description"] == "32770" for e in native["annotations"]),
        "post_trial_bookkeeping_suffix":[{k:e[k] for k in ("original_annotation_index","onset_token","duration_token","description")}
            for e in native["annotations"][241:]] if not partial else [],
        "terminal_incomplete_prefixes":terminal_prefixes,"incomplete_prefixes_are_not_MI_trials":True,
        "labels_or_trials_synthesized":False,"padding_applied":False,"class_balancing_applied":False,
        "waveform_QC_applied_to_eligibility":False}


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
    if expected_sha256 == PARTIAL_SOURCE_SHA256 and (subject,run,run_id,path.stat().st_size) != (40,3,"R3online",PARTIAL_SOURCE_BYTES):
        raise ValueError("Reviewed partial source requires its original subject/run identity and full EDF byte count")
    native = _read_native_metadata(path)
    # Every original marker must lie in the physically recorded file clock,
    # including the non-MI controls that follow the final complete trial.
    if any(e["onset"]+e["duration"] > Fraction(native["n_times"],512) for e in native["annotations"]):
        raise ValueError("Original annotation lies outside the physical EDF clock")
    adapter_sha = file_digest(__file__)
    rows = []
    for original in _trials(native,run_phase="acquisition" if run <= 2 else "online",
            source_sha256=expected_sha256,source_record_id=path.name):
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
        "non_task_marker_policy": "unchanged_legacy_stop_tail_or_exact_SHA_record_bound_observed_missing_stop_tail_or_reviewed_terminal_no_cue_prefix",
        "native_eligibility": _native_eligibility(native,rows,source_sha256=expected_sha256,source_record_id=path.name),
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
        "native_eligibility": independent["native_eligibility"],
        "stage": "pre_rms_car11_temporal_demeaned", "normalization_applied": False,
        "synthetic": False, "old_NPZ_reused": False, "fit_started": False, "actual_fits": 0,
        "GPU_actions": 0, "network_operations": 0, "R2_operations": 0}
    return X, y, events, metadata
