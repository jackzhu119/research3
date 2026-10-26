"""Native source-schema adapters for W1; no synthetic Raw, filtering, or training.

Only the four registered sources are accepted. The methods retain provider
coordinates and explicitly distinguish native headers from external mappings.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat

from p3work.contracts import CHANNELS, digest, file_digest, permit_preparation, validate_signal_contract


BNCI_CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    "EOG1", "EOG2", "EOG3",
)
CHO_EEG_CHANNELS = (
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
)


def _selection(names):
    folded = [name.casefold() for name in names]
    if len(set(folded)) != len(folded):
        raise ValueError("Ambiguous native channel names")
    missing = [name for name in CHANNELS if name.casefold() not in folded]
    if missing:
        raise ValueError("Missing measured clinic11 channels: " + ", ".join(missing))
    return [folded.index(name.casefold()) for name in CHANNELS]


def _row(base, *, source_record_id, physical_run_id, original_trial_index,
         raw_label_code, canonical_label, **fields):
    row = dict(base, source_record_id=source_record_id, physical_run_id=physical_run_id,
               original_trial_index=int(original_trial_index), raw_label_code=raw_label_code,
               canonical_label=canonical_label, task="left_right_hand_motor_imagery",
               cohort="source", feedback_condition=None, pair_or_cycle_id_or_null=None,
               pipeline_hash=None, split_manifest_hash=None,
               eligibility_reason="native_schema_only_window_not_frozen", **fields)
    row["trial_uid"] = digest({key: row[key] for key in
                              ("dataset", "source_version", "provider_subject", "session_id",
                               "source_file_sha256", "source_record_id", "physical_run_id",
                               "original_trial_index", "raw_label_code")})
    row["physical_run_id_or_null"] = physical_run_id
    row["task_end_verified"] = True
    return row


def _summary(base, rows, **fields):
    return dict(base, schema_audit_status="actual_native_schema_read",
                adapter_file_sha256=file_digest(__file__),
                scope="one_pre_registered_native_schema_specimen_not_full_cohort",
                numerical_cache_created=False, training_started=False,
                window_frozen=False, source_train_ready=False,
                biological_identity_independence_verified=False,
                n_native_lr_trials=len(rows),
                label_counts=dict(Counter(row["canonical_label"] for row in rows)),
                selected_channel_order=list(CHANNELS), **fields)


def _bnci(path, base, container=None):
    runs = np.atleast_1d(container if container is not None else
                        loadmat(path, variable_names=["data"], squeeze_me=True,
                                struct_as_record=False)["data"])
    selected = _selection(BNCI_CHANNELS)
    rows, records = [], []
    mi_run_order = 0
    for record_index, run in enumerate(runs):
        x = np.asarray(run.X)
        labels, trials = np.atleast_1d(run.y), np.atleast_1d(run.trial)
        if x.ndim != 2 or x.shape[1] != len(BNCI_CHANNELS) or int(run.fs) != 250:
            raise ValueError("Unregistered BNCI native channel/rate variant")
        classes = [str(value) for value in np.atleast_1d(run.classes)]
        if classes != ["left hand", "right hand", "feet", "tongue"]:
            raise ValueError("BNCI native class meanings changed")
        if len(labels) != len(trials):
            raise ValueError("BNCI trial/label mismatch")
        records.append({"provider_record_index_zero_based": record_index,
                        "native_shape": list(x.shape), "native_trial_count": len(trials),
                        "role": "MI_physical_run" if len(trials) else "EOG_calibration_no_MI_events"})
        if not len(trials):
            continue
        mi_run_order += 1
        artifacts = np.atleast_1d(run.artifacts)
        if len(artifacts) != len(trials) or not np.all(np.diff(trials) > 0):
            raise ValueError("BNCI chronology/artifact identity mismatch")
        for trial_index, (trial, label, artifact) in enumerate(zip(trials, labels, artifacts), 1):
            code = int(label)
            if code not in (1, 2):
                continue
            start = int(trial) - 1
            cue = start + 2 * int(run.fs)
            stop = start + 6 * int(run.fs)
            if start < 0 or stop > x.shape[0]:
                raise ValueError("BNCI original MI trial exceeds physical run")
            rows.append(_row(base, source_record_id=f"{path.name}::data[{record_index}]",
                physical_run_id=f"MI{mi_run_order}", original_trial_index=trial_index,
                raw_label_code=code, canonical_label="left_hand" if code == 1 else "right_hand",
                session_order=None, run_order=mi_run_order,
                run_order_evidence="native MAT record order; provider describes six MI runs after three EOG blocks",
                event_sample_or_null=int(trial), event_time_coordinate="MATLAB 1-based physical-run trial-start sample",
                trial_start_sample_zero_based=start, cue_time=cue / float(run.fs),
                task_time=cue / float(run.fs), analysis_anchor_sample=cue, mi_onset_verified=True,
                task_anchor_evidence="provider desc_2a: cue at trial+2s prompts MI until trial+6s",
                native_sfreq=float(run.fs), native_channel_order=list(BNCI_CHANNELS),
                channel_order_evidence="MAT lacks chan field; provider 22EEG+3EOG montage with fixed MOABB v1.7.2 bnci_2014.py column mapping",
                selected_indices=selected, native_unit_string=None,
                numeric_unit_declared_by_fixed_adapter="uV", physical_scale_evidence="MOABB v1.7.2 bnci/base.py explicitly converts run.X from uV; not native-header evidence",
                unit_verified_from_native_header=False, absolute_amplitude_qc_authorized=False,
                real_trial_start_end=[start, stop], task_end_sample=stop,
                available_task_seconds=4.0, provider_artifact_flag=int(artifact),
                acquisition_order_verified=True))
    return {"summary": _summary(base, rows, native_records=records,
              native_channel_header_present=False, channel_mapping_contains_all_clinic11=True,
              explicit_secondary_unit_evidence=True,
              remaining_gates=["freeze Stieger meta-train window", "full source roster/file inventory", "native MAT unit-string absent: retain fixed-adapter conversion evidence and do not claim header units"]),
            "trials": rows}


def _cho(path, base, container=None):
    eeg = container if container is not None else loadmat(path, variable_names=["eeg"], squeeze_me=True, struct_as_record=False)["eeg"]
    sfreq = float(eeg.srate)
    names = list(CHO_EEG_CHANNELS) + ["EMG1", "EMG2", "EMG3", "EMG4"]
    selected = _selection(names)
    events = np.flatnonzero(np.asarray(eeg.imagery_event))
    codes = np.unique(eeg.imagery_event)
    frame = np.atleast_1d(eeg.frame).astype(int).tolist()
    n = int(eeg.n_imagery_trials)
    if sfreq != 512 or frame != [-2000, 5000] or len(events) != n or set(codes) != {0, 1}:
        raise ValueError("Unregistered Cho source event/frame/rate variant")
    rows = []
    for label, field in (("left_hand", "imagery_left"), ("right_hand", "imagery_right")):
        signal = np.asarray(getattr(eeg, field))
        if signal.ndim != 2 or signal.shape[0] != 68 or signal.shape[1] % n:
            raise ValueError("Cho native class stream shape mismatch")
        epoch_samples = signal.shape[1] // n
        for index, event in enumerate(events):
            start, stop = index * epoch_samples, (index + 1) * epoch_samples
            if not start <= event < stop:
                raise ValueError("Cho onset outside corresponding provider epoch")
            rows.append(_row(base, source_record_id=f"{path.name}::eeg.{field}", physical_run_id=None,
                original_trial_index=index + 1, raw_label_code=field, canonical_label=label,
                session_order=None, run_order=None,
                run_order_evidence="not recoverable from class-separated MAT streams; never synthesize chronological episodes",
                event_sample_or_null=int(event), event_time_coordinate="zero-based sample within original class-specific array",
                cue_time=float(event / sfreq), task_time=float(event / sfreq),
                analysis_anchor_sample=int(event), mi_onset_verified=True,
                task_anchor_evidence="Cho paper Data format: imagery_event value 1 represents MI onset; preserve observed pulse without recalculating it from frame",
                native_sfreq=sfreq, native_channel_order=names, selected_indices=selected,
                channel_order_evidence="MAT has sensor coordinates but no channel-name strings; Cho paper Fig1 + fixed MOABB v1.7.2 gigadb.py Biosemi64 mapping",
                native_unit_string=None, numeric_unit_declared_by_fixed_adapter="uV",
                physical_scale_evidence="MOABB v1.7.2 gigadb.py multiplies native imagery arrays by 1e-6; MAT has no units/calibration fields",
                unit_verified_from_native_header=False, absolute_amplitude_qc_authorized=False,
                real_trial_start_end=[start, stop], task_end_sample=int(event + 3 * sfreq),
                available_task_seconds=3.0, acquisition_order_verified=False,
                provider_epoch_samples=epoch_samples, provider_frame_ms=frame,
                pulse_offset_in_epoch=int(event - start),
                pulse_minus_frame_zero_sample=int(event - start - round(-frame[0] * sfreq / 1000))))
    return {"summary": _summary(base, rows, source_fields=list(eeg._fieldnames),
              class_stream_shapes={name: list(np.shape(getattr(eeg, name))) for name in ("imagery_left", "imagery_right")},
              provider_frame_ms=frame, event_count_per_class=n,
              chronology_usable_for_prefix_query=False, physical_run_recoverable=False,
              raw_event_anchor_offset_from_frame_zero_samples=sorted({r["pulse_minus_frame_zero_sample"] for r in rows}),
              provider_bad_trial_flags_retained_separately=True,
              bad_trial_flag_fields=list(eeg.bad_trial_indices._fieldnames),
              provider_bad_trial_indices_native={name: [np.asarray(value).ravel().tolist()
                  for value in np.atleast_1d(getattr(eeg.bad_trial_indices, name))]
                  for name in eeg.bad_trial_indices._fieldnames},
              remaining_gates=["freeze Stieger meta-train window", "full source roster/file inventory", "native MAT unit-string absent: retain fixed-adapter conversion evidence and do not claim header units"]),
            "trials": rows}


def _lee(path, base, container=None):
    # Provider container also includes EEG_MI_test. It is deliberately not loaded.
    eeg = container if container is not None else loadmat(path, variable_names=["EEG_MI_train"], squeeze_me=True,
                  struct_as_record=False)["EEG_MI_train"]
    sfreq = float(eeg.fs)
    names = [str(value) for value in np.atleast_1d(eeg.chan)]
    selected = _selection(names)
    native_mapping = {int(code): str(label) for code, label in np.asarray(getattr(eeg, "class"))}
    if native_mapping != {1: "right", 2: "left"} or sfreq != 1000 or len(names) != 62:
        raise ValueError("Unregistered OpenBMI class/channel/rate variant")
    x, smt = np.asarray(eeg.x), np.asarray(eeg.smt)
    events, labels = np.atleast_1d(eeg.t), np.atleast_1d(eeg.y_dec)
    if (len(events) != len(labels) or smt.shape != (4000, len(events), len(names)) or
            not np.all(np.diff(events) > 0) or not np.array_equal(eeg.ival, np.arange(1, 4001))):
        raise ValueError("OpenBMI native event/epoch/time consistency failed")
    # Resolve sample indexing by exact native x/smt identity, not MATLAB convention guessing.
    offsets = []
    for offset in (-1, 0, 1):
        if all(0 <= int(event) + offset and int(event) + offset + 4000 <= x.shape[0] and
               np.array_equal(x[int(event) + offset:int(event) + offset + 4000], smt[:, i, :])
               for i, event in enumerate(events)):
            offsets.append(offset)
    if offsets != [0]:
        raise ValueError("OpenBMI native t coordinate is ambiguous; review schema rather than guess")
    rows = []
    for index, (event, code) in enumerate(zip(events, labels), 1):
        code, start = int(code), int(event)
        label = {1: "right_hand", 2: "left_hand"}.get(code)
        if label is None or str(eeg.y_class[index - 1]) != native_mapping[code]:
            raise ValueError("OpenBMI native label representations disagree")
        rows.append(_row(base, source_record_id=f"{path.name}::EEG_MI_train", physical_run_id=None,
            original_trial_index=index, raw_label_code=code, canonical_label=label,
            session_order=int(base["session_id"]), run_order=None,
            run_order_evidence="one offline train block; no finer physical-run field in this MAT structure",
            event_sample_or_null=start, event_time_coordinate="native t; x[t:t+4000] exactly equals provider smt for every trial",
            cue_time=start / sfreq, task_time=start / sfreq, analysis_anchor_sample=start,
            mi_onset_verified=True, task_anchor_evidence="OpenBMI paper: t is stimulus onset, four-second hand MI; native x/smt/t identity verified",
            native_sfreq=sfreq, native_channel_order=names, selected_indices=selected,
            channel_order_evidence="actual EEG_MI_train.chan strings, not montage interpolation",
            native_unit_string=None, numeric_unit_declared_by_fixed_adapter="uV",
            physical_scale_evidence="MOABB v1.7.2 Lee2019._scalings eeg=1e-6; native MAT has no unit/calibration field",
            unit_verified_from_native_header=False, absolute_amplitude_qc_authorized=False,
            real_trial_start_end=[start, start + smt.shape[0]], task_end_sample=start + smt.shape[0],
            available_task_seconds=4.0, acquisition_order_verified=True,
            source_phase="offline_train_only", calendar_session_date=None))
    return {"summary": _summary(base, rows, loaded_variables=["EEG_MI_train"],
              excluded_container_variables=["EEG_MI_test"], native_train_fields=list(eeg._fieldnames),
              native_channel_order=names, native_continuous_shape=list(x.shape),
              native_epoch_shape=list(smt.shape), native_t_to_x_verified_offset=0,
              x_to_smt_exactly_matched_trials=len(events),
              remaining_gates=["freeze Stieger meta-train window", "full source roster/file inventory", "native MAT unit-string absent: retain fixed-adapter conversion evidence and do not claim header units"]),
            "trials": rows}


def _physionet(path, base, container=None):
    import mne
    match = re.fullmatch(r"S(\d{3})R(\d{2})\.edf", path.name)
    if not match or int(match.group(2)) not in (4, 8, 12) or int(match.group(1)) != int(base["provider_subject"]):
        raise PermissionError("Only native PhysioNet left/right imagery runs 4, 8, 12 are allowed")
    run = int(match.group(2))
    raw = container if container is not None else mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    original_names = list(raw.ch_names)
    names = [name.strip(".") for name in original_names]
    selected = _selection(names)
    sfreq = float(raw.info["sfreq"])
    units = dict(raw._orig_units)
    si_factors = {"µV": 1e-6, "uV": 1e-6, "mV": 1e-3, "V": 1.0}
    if (sfreq not in {128.0, 160.0} or len(names) != 64 or len(set(units.values())) != 1 or
            next(iter(units.values()), None) not in si_factors):
        raise ValueError("PhysioNet native rate/channel/unit variant requires review")
    rows = []
    for index, annotation in enumerate(raw.annotations, 1):
        code = str(annotation["description"])
        if code == "T0":
            continue
        if code not in {"T1", "T2"}:
            raise ValueError("Unexpected annotation in native LR imagery run")
        onset, duration = float(annotation["onset"]), float(annotation["duration"])
        start, end = round(onset * sfreq), round((onset + duration) * sfreq)
        if abs(start / sfreq - onset) > 1e-6 or not 0 <= start < end <= raw.n_times:
            raise ValueError("PhysioNet annotation clock does not map to complete native samples")
        rows.append(_row(base, source_record_id=path.name, physical_run_id=str(run),
            original_trial_index=index, raw_label_code=code,
            canonical_label="left_hand" if code == "T1" else "right_hand",
            session_order=None, run_order=run,
            run_order_evidence="provider filename original R04/R08/R12",
            event_sample_or_null=start, event_time_coordinate="EDF+ annotation onset at native run time zero",
            cue_time=onset, task_time=onset, analysis_anchor_sample=start, mi_onset_verified=True,
            task_anchor_evidence="provider run4/8/12 instructions and T1=left/T2=right task-onset annotations",
            native_sfreq=sfreq, native_channel_order=original_names, selected_indices=selected,
            canonicalized_channel_names=names, channel_order_evidence="actual EDF signal labels; periods stripped and case matched only",
            native_unit_string=next(iter(units.values())), physical_scale_evidence="actual EDF physical_dimension and MNE SI conversion",
            unit_verified_from_native_header=True, physical_scale_to_si_volts=si_factors[next(iter(units.values()))],
            data_returned_by_mne_unit="V", real_trial_start_end=[start, end],
            task_end_sample=end, available_task_seconds=duration, acquisition_order_verified=True))
    summary = _summary(base, rows, native_n_times=int(raw.n_times), native_sfreq=sfreq,
                       native_sampling_variant="known_128Hz_provider_variant" if sfreq == 128 else "reported_160Hz_variant",
                       native_channel_order=original_names, native_channel_units=units,
                       annotation_counts=dict(Counter(raw.annotations.description)),
                       remaining_gates=["freeze Stieger meta-train window", "full source roster/file inventory"])
    if container is None:
        raw.close()
    return {"summary": summary, "trials": rows}


def audit_native_source(dataset, path, *, provider_subject, session_id,
                        source_version, expected_sha256):
    """Read a verified source native file and return metadata plus immutable trial IDs."""
    permit_preparation(dataset, "source_schema")
    path = Path(path)
    actual_sha = file_digest(path)
    if actual_sha != expected_sha256:
        raise ValueError("Native source file differs from the download manifest SHA256")
    base = {"dataset": dataset, "role": "source", "provider_subject": str(provider_subject),
            "source_version": source_version, "source_file_sha256": actual_sha,
            "source_file_bytes": path.stat().st_size, "session_id": str(session_id)}
    return {"BNCI2014_001": _bnci, "Cho2017": _cho,
            "Lee2019_MI": _lee, "PhysionetMI": _physionet}[dataset](path, base)


def iter_native_contexts(dataset, path, *, rows, contract, expected_sha256):
    """Yield only frozen bounded measured-clinic11 contexts from one native file.

    This performs no filtering and writes no cache. Native MAT decoding loads a
    provider container once; only the verified trial context is selected from
    its EEG arrays. OpenBMI's test struct is never decoded. Metadata must exactly
    match a new native schema audit, including original task end and trial UID.
    Returned effective numeric units are distinguished from native header units.
    """
    validate_signal_contract(contract)
    permit_preparation(dataset, "source_train")
    path = Path(path)
    if not rows:
        raise ValueError("A native recording needs its full original LR metadata")
    if file_digest(path) != expected_sha256:
        raise ValueError("Native source file differs from its complete SHA256 receipt")
    first = rows[0]
    base = {key: first[key] for key in ("dataset", "role", "provider_subject", "source_version",
                                       "source_file_sha256", "source_file_bytes", "session_id")}
    if (base["dataset"] != dataset or base["role"] != "source" or
            base["source_file_sha256"] != expected_sha256 or base["source_file_bytes"] != path.stat().st_size):
        raise ValueError("Native recording identity disagrees with source metadata")
    if dataset == "PhysionetMI":
        import mne
        container = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    else:
        variable = {"BNCI2014_001": "data", "Cho2017": "eeg", "Lee2019_MI": "EEG_MI_train"}[dataset]
        container = loadmat(path, variable_names=[variable], squeeze_me=True, struct_as_record=False)[variable]
    try:
        expected = {"BNCI2014_001": _bnci, "Cho2017": _cho,
                    "Lee2019_MI": _lee, "PhysionetMI": _physionet}[dataset](path, base, container)["trials"]
        if len(expected) != len(rows) or any(digest(a) != digest(b) for a, b in zip(expected, rows)):
            raise ValueError("Original trial metadata differs from the verified native source")
        for row in rows:
            anchor, end = row["analysis_anchor_sample"], row["task_end_sample"]
            if row.get("task_end_verified") is not True or type(anchor) is not int or type(end) is not int:
                raise ValueError("Explicit verified native task start/end required")
            stop = anchor + round(contract["context_s"][1] * row["native_sfreq"])
            if stop > end or anchor < row["real_trial_start_end"][0] or stop > row["real_trial_start_end"][1]:
                # Preserve this original trial identity as a QC exclusion, never pad it.
                yield {"row": row, "excluded": "insufficient_original_task_context"}
                continue
            indices = row["selected_indices"]
            if dataset == "BNCI2014_001":
                match = re.fullmatch(re.escape(path.name) + r"::data\[(\d+)\]", row["source_record_id"])
                if match is None:
                    raise ValueError("Invalid native BNCI record coordinate")
                array = np.atleast_1d(container)[int(match.group(1))].X
                context = np.asarray(array[anchor:stop, indices]).T.copy()
            elif dataset == "Cho2017":
                field = {"left_hand": "imagery_left", "right_hand": "imagery_right"}[row["canonical_label"]]
                context = np.asarray(getattr(container, field)[indices, anchor:stop]).copy()
            elif dataset == "Lee2019_MI":
                context = np.asarray(container.x[anchor:stop, indices]).T.copy()
            else:
                context = container.get_data(picks=indices, start=anchor, stop=stop)
            if context.shape != (len(CHANNELS), stop - anchor):
                raise ValueError("Native context selection changed shape")
            unit = "V" if dataset == "PhysionetMI" else row.get("numeric_unit_declared_by_fixed_adapter")
            if unit not in {"V", "uV"}:
                raise ValueError("Missing retained evidence for native numeric scale")
            yield {"row": row, "signal": context, "channel_names": list(CHANNELS),
                   "sfreq_hz": row["native_sfreq"], "task_sample": 0,
                   "task_end_sample": end - anchor, "task_end_verified": True,
                   "native_unit": unit,
                   "unit_evidence_kind": "native_EDF_header_MNE_SI" if dataset == "PhysionetMI" else "fixed_adapter_numeric_unit_declaration_not_native_header",
                   "original_context_samples": [anchor, stop]}
    finally:
        if dataset == "PhysionetMI":
            container.close()
