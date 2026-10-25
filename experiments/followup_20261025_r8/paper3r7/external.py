"""Optional Weibo source preparation and read-only published-order evaluation.

Workers use ExternalCatalog only. Downloads are explicit preparation actions;
this module never downloads while opening or materializing a catalog.
"""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import os
import time
import urllib.request

import numpy as np
from .io import atomic, digest, emit, put, read, sha

CHANNELS = ["FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4"]
NATIVE_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
    "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
    "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2",
    "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
    "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2", "VEO", "HEO",
]
INDICES = [NATIVE_CHANNELS.index(c) for c in CHANNELS]
ORDER_SCOPE = "published_MAT_order_LR_subtask_not_verified_physical_chronology"
LOADER_URL = "https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/Weibo2014.py"


def validate_contract(contract):
    expected = {"channels": CHANNELS, "context_s": [0, 1.5], "crop_s": [.25, 1.25],
                "n_times": 128, "sfreq_hz": 128, "band_hz": [8, 30],
                "reference": "CAR_exact_clinic11"}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Weibo adapter requires the exact frozen 11-channel V2.1 signal contract")
    return digest(contract)


def native_schema(data, label):
    """Validate original MAT layout, never guess axes or remap class IDs."""
    a, y = np.asarray(data), np.asarray(label).reshape(-1)
    if (a.ndim != 3 or a.shape[:2] != (64, 1600) or a.shape[2] != len(y)
            or a.dtype.kind != "f" or len(y) < 21):
        raise ValueError("Expected original floating 64 x 1600 x N data and N labels")
    if not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all() or not np.isin(y, range(1, 8)).all():
        raise ValueError("Original Weibo labels must be finite integer codes 1..7")
    y = y.astype(np.int64)
    ids = np.flatnonzero(np.isin(y, [1, 2]))
    if len(ids) <= 20:
        raise ValueError("No LR query after the paid B20 prefix")
    values, counts = np.unique(y, return_counts=True)
    rest = np.flatnonzero(y == 7)
    return {"data_shape": list(a.shape), "data_dtype": str(a.dtype),
            "label_shape": list(np.asarray(label).shape), "label_dtype": str(np.asarray(label).dtype),
            "class_counts": {str(int(k)): int(v) for k, v in zip(values, counts)},
            "lr_trials": int(len(ids)), "first20_native_labels": y[:20].tolist(),
            "first20_lr_original_trial_indices_zero_based": ids[:20].tolist(),
            "first20_lr_class_counts": {str(k): int(np.sum(y[ids[:20]] == k)) for k in [1, 2]},
            "rest_is_terminal_contiguous_block": bool(len(rest) and np.array_equal(rest, np.arange(rest[0], len(y)))),
            "order_scope": ORDER_SCOPE, "acquisition_order_verified": False,
            "physical_run_id_available": False, "sampling_rate_hz": 200,
            "mi_onset_trial_sample": 600, "mi_end_trial_sample": 1400,
            "channel_names_basis": "MOABB original MAT parser order; MAT has no embedded channel-name variable",
            "native_channel_names": NATIVE_CHANNELS, "selected_indices_zero_based": INDICES,
            "unit": "uV", "unit_evidence": "Original MOABB parser explicitly applies 1e-6; MAT has no unit field",
            "unit_independently_calibrated_from_MAT": False, "unit_conversion_to_V": 1e-6,
            "unit_reference_url": LOADER_URL}


def download_source(entry, raw_dir, *, deadline_s=900, timeout_s=45):
    """Preparation only. A complete pinned source is accepted after byte/SHA checks."""
    destination = Path(raw_dir) / Path(entry["path"]).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_sha = int(entry["size"]), entry["checksum"]
    if entry.get("checksum_algorithm") != "sha256":
        raise ValueError("Pinned SHA-256 required")
    if destination.exists():
        if destination.stat().st_size == expected_size and sha(destination) == expected_sha:
            return destination
        raise ValueError("Existing raw source differs from pinned file; refusing overwrite")
    url = entry["url"]
    if not url.startswith("https://data.nemar.org/nm000146/v1.0.3/sourcedata/subject_"):
        raise ValueError("Unexpected public source endpoint")
    temporary = destination.with_suffix(".mat.part")
    started, count, last = time.monotonic(), 0, 0
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response, temporary.open("wb") as target:
            if response.status != 200:
                raise ValueError("Full source download must return HTTP 200")
            while True:
                if time.monotonic() - started > deadline_s:
                    raise TimeoutError("Bounded Weibo source download deadline exceeded")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > expected_size:
                    raise ValueError("Source exceeded the pinned byte budget")
                target.write(chunk); h.update(chunk)
                if time.monotonic() - last >= 20:
                    emit("OPTIONAL_SOURCE_DOWNLOAD", subject=destination.stem, bytes=count, expected_bytes=expected_size)
                    last = time.monotonic()
            target.flush(); os.fsync(target.fileno())
        if count != expected_size or h.hexdigest() != expected_sha:
            raise ValueError("Source download failed pinned byte/SHA-256 verification")
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def preprocess_weibo(mat_path, contract, subject, expected_sha):
    """Original-MAT LR order -> pre-RMS cache; invalid paid positions stay present."""
    from scipy.io import loadmat, whosmat
    signal_sha = validate_contract(contract)
    mat_path = Path(mat_path)
    actual_sha = sha(mat_path)
    if actual_sha != expected_sha:
        raise ValueError("Raw Weibo SHA-256 differs from the pinned source")
    variables = whosmat(mat_path)
    if {v[0] for v in variables} != {"data", "label"}:
        raise ValueError("Unexpected original MAT variables; inspect before adapting")
    value = loadmat(mat_path, variable_names=["data", "label"], squeeze_me=False,
                    struct_as_record=False, verify_compressed_data_integrity=True)
    data = value["data"]
    schema = native_schema(data, value["label"])
    labels = value["label"].reshape(-1).astype(np.int64)
    schema.update(source_sha256=actual_sha, source_bytes=mat_path.stat().st_size,
                  original_variables=[{"name": n, "shape": list(s), "class": c} for n, s, c in variables],
                  signal_contract_sha256=signal_sha,
                  bids_event_order_reused=False,
                  chronology_limitation="Terminal rest block and missing original physical-run/time metadata; MAT publication order only",
                  preprocessing="Exact local numeric copy of p3work.preprocess.prepare_trial; CAR11,SOS4,polyphase16/25(line),crop,time-demean; no RMS")
    indices = np.flatnonzero(np.isin(labels, [1, 2]))
    x = np.zeros((len(indices), 11, 128), dtype=np.float32)
    rows, rms_uv = [], []
    for cache_index, original_index in enumerate(indices):
        uid = digest(["Weibo2014", actual_sha, int(original_index), "original_MAT_LR"])
        row = {"event_uid": uid, "trial_uid": uid, "dataset": "Weibo2014",
               "participant_id": f"W{int(subject):02d}", "session_id": "0",
               "cache_index": cache_index, "original_trial_index": int(original_index),
               "original_trial_index_one_based": int(original_index) + 1,
               "lr_subtask_index": cache_index, "raw_label_code": int(labels[original_index]),
               "label": int(labels[original_index]) - 1, "physical_run_id": None,
               "order_scope": ORDER_SCOPE, "acquisition_order_verified": False,
               "mi_onset_trial_sample": 600, "mi_end_trial_sample": 1400,
               "context_start_sample": 600, "context_stop_sample": 900,
               "native_sfreq": 200, "native_unit": "uV", "stored_unit": "V",
               "source_sha256": actual_sha, "valid": False}
        try:
            result, qc = prepare_context(data[:, :, original_index], contract)
            x[cache_index] = result.astype(np.float32)
            row.update(valid=True, qc=qc)
            rms_uv.append(qc["joint_rms"] * 1e6)
        except ValueError as error:
            row["invalid_reason"] = str(error)
        rows.append(row)
    schema["valid_lr_trials"] = sum(r["valid"] for r in rows)
    schema["invalid_lr_trials"] = len(rows) - schema["valid_lr_trials"]
    schema["pre_rms_joint_rms_uV_quantiles"] = np.quantile(rms_uv, [0, .25, .5, .75, 1]).tolist() if rms_uv else []
    return x, rows, schema


def prepare_context(trial, contract):
    """Exact numeric operations of p3work.preprocess.prepare_trial for this source.

    SciPy is imported only by explicit preparation; catalogs need NumPy/Torch.
    Original native indices600:900 alone are read for signal/QC. Unit gain comes
    from the original MOABB MAT parser, not an inferred BIDS unit field.
    """
    from scipy.signal import butter, sosfiltfilt, resample_poly
    signal_sha = validate_contract(contract)
    if not isinstance(trial, np.ndarray) or trial.shape != (64, 1600):
        raise ValueError("Expected one original 64 x 1600 trial")
    selected = np.asarray(trial[INDICES, 600:900], dtype=np.float64) * 1e-6
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite context")
    selected -= selected.mean(axis=0, keepdims=True)
    sos = butter(4, (8, 30), btype="bandpass", fs=200, output="sos")
    filtered = sosfiltfilt(sos, selected, axis=-1)
    resampled = resample_poly(filtered, 16, 25, axis=-1, padtype="line")
    result = resampled[:, 32:160].copy()
    if result.shape != (11, 128):
        raise ValueError("Crop shape mismatch")
    result -= result.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(result), dtype=np.float64))
    if not np.isfinite(rms) or rms == 0:
        raise ValueError("Zero/nonfinite trial energy")
    return result, {"signal_contract_sha256": signal_sha, "native_unit": "uV", "stored_unit": "V",
                    "si_conversion_factor": 1e-6, "reference": "CAR_exact_clinic11", "joint_rms": float(rms),
                    "normalization_applied": False, "filter_context_ends_at_native_sample": 900,
                    "data_after_context_accessed": False}


def _relative(root, path):
    return {"path": str(Path(path).resolve().relative_to(Path(root).resolve())), "sha256": sha(path),
            "bytes": Path(path).stat().st_size}


def prepare(root, output, source_manifest, subjects, raw_dir, *, download=False,
            delete_raw_after_prepare=False):
    root, output, raw_dir = Path(root), Path(output), Path(raw_dir)
    if any(output.resolve().is_relative_to((root / p).resolve()) for p in ("cache", "runs", "contracts")) or output.resolve() == root.resolve():
        raise ValueError("Optional preparation must not write into the existing project/cache/run/contract trees")
    contract = read(root / "contracts/input_contract.json")
    signal_sha = validate_contract(contract)
    source = read(source_manifest)
    if source.get("dataset_id") != "Weibo2014" or source.get("mirror_version") != "v1.0.3":
        raise ValueError("Expected pinned Weibo original-source manifest")
    output.mkdir(parents=True, exist_ok=True)
    sessions, failures = [], []
    requested = set(subjects)
    manifest_path = output / "manifest.json"
    source_sha = sha(source_manifest)
    if manifest_path.exists():
        previous = read(manifest_path)
        if (previous.get("signal_contract_sha256") != signal_sha
                or previous.get("source_manifest_sha256") != source_sha
                or previous.get("order_scope") != ORDER_SCOPE):
            raise ValueError("Existing optional cache belongs to a different source/contract/order")
        for session in previous.get("sessions", []):
            for key in ("X", "events", "schema"):
                path = (output / session[key]["path"]).resolve()
                if not path.is_relative_to(output.resolve()) or sha(path) != session[key]["sha256"]:
                    raise ValueError("Existing optional cache artifact failed reuse verification")
            session.setdefault("prepared_by_code_sha256", previous.get("preparer_code_sha256"))
            sessions.append(session)
        requested.update(previous.get("requested_subjects", []))
        failures = [f for f in previous.get("failures", []) if f["subject"] not in subjects]

    def persist():
        ordered_sessions = sorted(sessions, key=lambda s: s["participant_id"])
        prepared = {int(s["participant_id"][1:]) for s in sessions}
        manifest = {"schema_version": "paper3r7-weibo-cache-v1", "dataset_id": "Weibo2014",
            "status": "WEIBO_EXTERNAL_CACHE_COMPLETE" if sessions else "SKIPPED_OPTIONAL",
            "requested_subjects": sorted(requested), "prepared_subjects": [s["participant_id"] for s in ordered_sessions],
            "all_requested_subjects_prepared": requested <= prepared, "failures": failures,
            "signal_contract_sha256": signal_sha, "source_manifest_sha256": source_sha,
            "preparer_code_sha256": sha(__file__),
            "license": "CC0-1.0", "data_doi": "10.7910/DVN/27306", "mirror_version": "v1.0.3",
            "order_scope": ORDER_SCOPE, "external_claim": "published-order LR-subtask robustness; not strict physical-time confirmation",
            "stage": "pre_rms_car11_temporal_demeaned", "normalization_applied": False,
            "channels": CHANNELS, "shape_per_trial": [11, 128], "stored_dtype": "float32",
            "budget_scope": "first20_LR_positions_in_original_MAT_release_order_invalid_positions_paid",
            "query_policy": "fixed_valid_LR_positions_after_B20_for_all_shorter_budgets",
            "temporal_risk_eligible": False, "acquisition_order_verified": False,
            "physical_run_id_available": False, "sessions": ordered_sessions}
        put(manifest_path, manifest)
        return manifest

    for subject in sorted(set(subjects)):
        if any(s["participant_id"] == f"W{subject:02d}" for s in sessions):
            emit("OPTIONAL_SUBJECT_REUSED", subject=subject, raw_download_required=False)
            continue
        entry = next(e for e in source["files"] if Path(e["path"]).name == f"subject_{subject}.mat")
        raw = raw_dir / f"subject_{subject}.mat"
        raw_preexisted = raw.exists()
        phase = "local_input"
        try:
            if download:
                phase = "download"
                raw = download_source(entry, raw_dir)
            phase = "preprocessing"
            if not raw.is_file():
                raise FileNotFoundError("Pinned raw file is absent; use explicit --download during preparation")
            if raw.stat().st_size != entry["size"]:
                raise ValueError("Raw byte count differs from published source")
            x, rows, schema = preprocess_weibo(raw, contract, subject, entry["checksum"])
            query_labels = [r["label"] for r in rows[20:] if r["valid"]]
            if len(query_labels) < 20 or any(query_labels.count(label) < 5 for label in (0, 1)):
                raise ValueError("Optional subject has insufficient two-class valid query after B20")
            person_dir = output / f"W{subject:02d}"
            person_dir.mkdir(parents=True, exist_ok=True)
            xp, ep, sp = person_dir / "X.npy", person_dir / "events.json", person_dir / "schema.json"
            def write_array(path):
                with open(path, "wb") as target:
                    np.save(target, x, allow_pickle=False)
            atomic(xp, write_array); put(ep, rows); put(sp, schema)
            sessions.append({"participant_id": f"W{subject:02d}", "session_id": "0",
                "partition": "external_robustness", "X": _relative(output, xp),
                "events": _relative(output, ep), "schema": _relative(output, sp),
                "source_sha256": entry["checksum"], "source_bytes": entry["size"],
                "prepared_by_code_sha256": sha(__file__), "raw_removed_after_prepare": False,
                "acquisition_order_verified": False, "physical_run_id_available": False})
            persist()  # A later network failure/interruption never loses completed people.
            if delete_raw_after_prepare and download and not raw_preexisted:
                raw.unlink()
                sessions[-1]["raw_removed_after_prepare"] = True
                persist()
                emit("NEW_OPTIONAL_RAW_REMOVED", subject=subject, source_sha256=entry["checksum"])
            emit("OPTIONAL_SUBJECT_PREPARED", subject=subject, shape=list(x.shape), valid_trials=schema["valid_lr_trials"])
        except Exception as error:
            failures.append({"subject": subject, "status": "SKIPPED_OPTIONAL", "reason": type(error).__name__ + ": " + str(error)})
            emit("SKIPPED_OPTIONAL", subject=subject, reason=failures[-1]["reason"])
            persist()
            if phase == "download":
                break  # Stop only this optional preparation; completed cache stays usable.
    manifest = persist()
    emit("OPTIONAL_PREPARATION_FINISHED", status=manifest["status"], manifest=str(output / "manifest.json"),
         subjects=len(sessions), skipped_subjects=len(failures))
    return manifest


class ExternalCatalog:
    """Read existing local prepared data; published order is explicitly qualified."""
    def __init__(self, root, manifest_path, *, cache_gb=8):
        from .data import Plan
        self.root, self.role = Path(root), "external_robustness"
        manifest_path = Path(manifest_path).resolve()
        self.cache_root = manifest_path.parent
        manifest = read(manifest_path)
        if (manifest.get("status") != "WEIBO_EXTERNAL_CACHE_COMPLETE"
                or manifest.get("order_scope") != ORDER_SCOPE or manifest.get("temporal_risk_eligible") is not False):
            raise ValueError("Optional Weibo cache missing/incompatible")
        self.signal_sha = validate_contract(read(self.root / "contracts/input_contract.json"))
        if manifest.get("signal_contract_sha256") != self.signal_sha:
            raise ValueError("Optional cache and frozen source signal contracts differ")
        self.identity = {"manifest_sha256": sha(manifest_path), "signal_contract_sha256": self.signal_sha,
                         "dataset_id": "Weibo2014", "order_scope": ORDER_SCOPE}
        self.people, self.plans, self.excluded = [], [], []
        for session in manifest["sessions"]:
            person = session["participant_id"]
            if person in self.people:
                raise ValueError("Duplicate external participant/session identity")
            rows = read(self._artifact(session["events"]))
            xp = self._artifact(session["X"])
            if (len(rows) <= 20 or any(r.get("label") not in (0, 1) or type(r.get("valid")) is not bool or r.get("cache_index") != i
                    or r.get("physical_run_id") is not None for i, r in enumerate(rows))
                    or any(rows[i]["original_trial_index"] >= rows[i + 1]["original_trial_index"] for i in range(len(rows)-1))):
                raise ValueError("External rows violate original MAT LR order or unknown-run policy")
            uids = [r["event_uid"] for r in rows]
            if len(set(uids)) != len(uids):
                raise ValueError("Duplicate optional trial identity")
            query = [r for r in rows[20:] if r["valid"]]
            if len(query) < 20 or any(sum(r["label"] == label for r in query) < 5 for label in (0, 1)):
                self.excluded.append({"participant": person, "reason": "no_two_class_query_after_B20"})
                continue
            audit = {"acquired_prefix_uids": uids[:20], "query_uids": [r["event_uid"] for r in query],
                     "query_eligible": True, "order_scope": ORDER_SCOPE, "acquisition_order_verified": False,
                     "temporal_risk_eligible": False, "physical_run_id_available": False,
                     "budget": 20, "query_policy": manifest["query_policy"]}
            self.plans.append(Plan(person, str(session["session_id"]), xp,
                                   {r["event_uid"]: r for r in rows}, audit, self.role))
            self.people.append(person)
        self.people.sort(); self.plans.sort(key=lambda p: p.participant_id)
        self.by_person = {p: [s for s in self.plans if s.participant_id == p] for p in self.people}
        if not self.plans:
            raise ValueError("No usable optional external sessions")
        self.cache, self.cache_bytes, self.cache_limit = OrderedDict(), 0, int(cache_gb * 1024**3)
        emit("EXTERNAL_CATALOG_READY", people=len(self.people), sessions=len(self.plans),
             order_scope=ORDER_SCOPE, temporal_risk_eligible=False)

    def _artifact(self, value):
        path = (self.cache_root / value["path"]).resolve()
        if not path.is_relative_to(self.cache_root) or sha(path) != value["sha256"]:
            raise ValueError("Optional cache path/hash mismatch")
        return path

    def array(self, plan):
        from .data import Catalog
        value = Catalog.array(self, plan)
        if value.shape != (len(plan.rows), 11, 128) or value.dtype != np.dtype("float32"):
            raise ValueError("Optional pre-RMS cache shape differs from rows/contract")
        valid = value[[r["cache_index"] for r in plan.rows.values() if r["valid"]]]
        if not np.isfinite(valid).all() or np.any(np.square(valid.astype(np.float64)).sum(axis=(1, 2)) == 0):
            raise ValueError("Valid optional cache rows contain nonfinite or zero-energy signal")
        return value

    def materialize(self, plan, device, *, budget=20):
        import torch
        from p3work.adapt import CalibrationPrefix
        if budget not in (4, 8, 12, 16, 20):
            raise ValueError("Unsupported optional calibration budget")
        acquired = tuple(plan.audit["acquired_prefix_uids"][:budget])
        support_ids = tuple(uid for uid in acquired if plan.rows[uid]["valid"])
        query_ids = tuple(plan.audit["query_uids"])
        value = self.array(plan)
        def tensors(ids):
            rows = [plan.rows[u] for u in ids]
            x = np.array(value[[r["cache_index"] for r in rows]], dtype=np.float64)
            return torch.from_numpy(x).to(device), torch.tensor([r["label"] for r in rows], device=device, dtype=torch.long)
        sx, sy = tensors(support_ids); qx, qy = tensors(query_ids)
        support = CalibrationPrefix(sx, sy, support_ids, acquired, self.signal_sha, "calibration_prefix")
        return SimpleNamespace(support=support, query_x=qx, query_y=qy, query_ids=query_ids,
            run_ids=tuple("unknown" for _ in query_ids), order_scope=ORDER_SCOPE,
            temporal_risk_eligible=False)

    def draw(self, seed, cursor):
        from .data import Catalog
        return Catalog.draw(self, seed, cursor)
