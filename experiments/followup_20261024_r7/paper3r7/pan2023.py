"""Pan2023 original-MAT preparation and read-only external multi-session catalog.

Only explicit preparation imports scipy/h5py or downloads. GPU workers read the
small pre-RMS cache; they never fetch raw data or reconstruct physical blocks.
"""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import os
import urllib.request

import numpy as np
from .external import CHANNELS, validate_contract
from .io import atomic, digest, emit, put, read, sha

NATIVE_CHANNELS = (
    "FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 "
    "CP5 CP3 CP1 CPz CP2 CP4 CP6 P5 P3 P1 Pz P2 P4 P6"
).split()
INDICES = [NATIVE_CHANNELS.index(c) for c in CHANNELS]
ORDER_SCOPE = "original_Pan2023_session_MAT_trial_index_not_verified_physical_block_chronology"
NEMAR_BASE = "https://data.nemar.org/nm000291/v1.0.0/sourcedata/"
LOADER_URL = "https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/pan2023.py"


def prepare_context(trial, contract):
    """Exact existing signal contract on original MI samples750:1125 only."""
    from scipy.signal import butter, sosfiltfilt, resample_poly
    signal_sha = validate_contract(contract)
    if not isinstance(trial, np.ndarray) or trial.shape != (28, 1750):
        raise ValueError("Expected original Pan2023 28 x 1750 trial")
    selected = np.asarray(trial[INDICES, 750:1125], dtype=np.float64) * 1e-6
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite context")
    selected -= selected.mean(axis=0, keepdims=True)
    filtered = sosfiltfilt(butter(4, (8, 30), btype="bandpass", fs=250,
                                output="sos"), selected, axis=-1)
    result = resample_poly(filtered, 64, 125, axis=-1, padtype="line")[:, 32:160].copy()
    if result.shape != (11, 128):
        raise ValueError("Crop shape mismatch")
    result -= result.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(result), dtype=np.float64))
    if not np.isfinite(rms) or rms == 0:
        raise ValueError("Zero/nonfinite trial energy")
    return result, {"signal_contract_sha256": signal_sha, "native_unit": "uV",
                    "stored_unit": "V", "si_conversion_factor": 1e-6,
                    "reference": "CAR_exact_clinic11", "joint_rms": float(rms),
                    "normalization_applied": False, "filter_context_ends_at_native_sample": 1125,
                    "data_after_context_accessed": False}


def native_schema(data, labels, sfreq):
    """Original h5py axis order is120 x1750 x28; no axis/label guessing."""
    a, y = np.asarray(data), np.asarray(labels)
    if a.shape != (120, 1750, 28) or a.dtype != np.dtype("float64"):
        raise ValueError("Expected original float64 HDF5 data shape (120,1750,28)")
    if y.shape != (1, 120) or not np.isfinite(y).all() or not np.isin(y, [1, 2]).all():
        raise ValueError("Expected original label shape (1,120), left=1/right=2")
    if float(np.asarray(sfreq).reshape(-1)[0]) != 250 or np.asarray(sfreq).size != 1:
        raise ValueError("Expected native sampling rate250Hz")
    y = y.reshape(-1).astype(np.int64)
    return {"data_shape": list(a.shape), "data_dtype": str(a.dtype),
            "label_shape": list(np.asarray(labels).shape), "label_dtype": str(np.asarray(labels).dtype),
            "class_counts": {str(k): int(np.sum(y == k)) for k in (1, 2)},
            "trials": len(y), "first20_native_labels": y[:20].tolist(),
            "first20_class_counts": {str(k): int(np.sum(y[:20] == k)) for k in (1, 2)},
            "order_scope": ORDER_SCOPE, "acquisition_order_verified": False,
            "physical_run_id_available": False, "session_identity": "original_author_D1_or_D2_file",
            "sampling_rate_hz": 250, "mi_onset_trial_sample": 750, "mi_end_trial_sample": 1750,
            "native_channel_names": NATIVE_CHANNELS, "selected_indices_zero_based": INDICES,
            "channel_names_basis": "author/MOABB original loader; MAT contains no channel names",
            "unit": "uV", "unit_conversion_to_V": 1e-6,
            "unit_reference_url": LOADER_URL, "unit_independently_calibrated_from_MAT": False}


def preprocess_mat(path, contract, subject, session, expected_sha):
    import h5py
    path = Path(path)
    if sha(path) != expected_sha:
        raise ValueError("Pan2023 original source SHA256 mismatch")
    with h5py.File(path, "r") as f:
        if set(f.keys()) != {"data", "label", "fs"}:
            raise ValueError("Unexpected original Pan2023 MAT variables")
        data, labels, fs = np.asarray(f["data"]), np.asarray(f["label"]), np.asarray(f["fs"])
    schema = native_schema(data, labels, fs)
    signal_sha = validate_contract(contract)
    schema.update(source_sha256=expected_sha, source_bytes=path.stat().st_size,
                  signal_contract_sha256=signal_sha, subject=int(subject), session=int(session),
                  preprocessing="CAR11,SOS4 8-30Hz,polyphase64/125(line),crop32:160,time-demean; pre-RMS",
                  bids_event_order_reused=False)
    x = np.zeros((120, 11, 128), dtype=np.float32)
    rows, rms = [], []
    for i, label in enumerate(labels.reshape(-1).astype(np.int64)):
        uid = digest(["Pan2023", expected_sha, i, "original_session_MAT_trial"])
        row = {"event_uid": uid, "trial_uid": uid, "dataset": "Pan2023",
               "participant_id": f"PAN{int(subject):02d}", "session_id": str(session),
               "cache_index": i, "original_trial_index": i,
               "original_trial_index_one_based": i + 1, "raw_label_code": int(label),
               "label": int(label) - 1, "physical_run_id": None, "order_scope": ORDER_SCOPE,
               "acquisition_order_verified": False, "mi_onset_trial_sample": 750,
               "mi_end_trial_sample": 1750, "context_start_sample": 750,
               "context_stop_sample": 1125, "native_sfreq": 250,
               "native_unit": "uV", "stored_unit": "V", "source_sha256": expected_sha,
               "valid": False}
        try:
            value, qc = prepare_context(data[i].T, contract)
            x[i] = value.astype(np.float32)
            row.update(valid=True, qc=qc); rms.append(qc["joint_rms"] * 1e6)
        except ValueError as error:
            row["invalid_reason"] = str(error)
        rows.append(row)
    schema.update(valid_trials=sum(r["valid"] for r in rows),
                  invalid_trials=sum(not r["valid"] for r in rows),
                  pre_rms_joint_rms_uV_quantiles=np.quantile(rms, [0, .25, .5, .75, 1]).tolist() if rms else [])
    return x, rows, schema


def download_source(entry, raw_dir):
    """Preparation-only pinned28-file source, no credentials/network in workers."""
    name = str(entry["file_id"])
    if not name.isdigit() or entry["url"] != NEMAR_BASE + name or entry.get("checksum_algorithm") != "sha256":
        raise ValueError("Expected pinned public Pan2023 source endpoint and SHA256")
    p = Path(raw_dir) / name; p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file():
        if p.stat().st_size == entry["size"] and sha(p) == entry["checksum"]:
            return p
        raise ValueError("Existing raw Pan2023 file differs; refusing overwrite")
    temporary = p.with_suffix(".part")
    try:
        h, count = hashlib.sha256(), 0
        with urllib.request.urlopen(entry["url"], timeout=60) as source, temporary.open("wb") as target:
            if source.status != 200:
                raise ValueError("Full Pan2023 source request must return200")
            for block in iter(lambda: source.read(1024 * 1024), b""):
                count += len(block)
                if count > entry["size"]:
                    raise ValueError("Pan2023 source exceeded pinned size")
                h.update(block); target.write(block)
            target.flush(); os.fsync(target.fileno())
        if count != entry["size"] or h.hexdigest() != entry["checksum"]:
            raise ValueError("Pan2023 source size/SHA256 mismatch")
        os.replace(temporary, p)
        return p
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(root, path):
    return {"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}


def prepare(root, output, source_manifest, raw_dir, *, download=False):
    root, output, raw_dir = Path(root), Path(output), Path(raw_dir)
    contract = read(root / "contracts/input_contract.json")
    contract_sha = validate_contract(contract)
    pinned = read(source_manifest)
    entries = pinned["files"]
    if pinned.get("dataset_id") != "Pan2023" or len(entries) != 28:
        raise ValueError("Expected pinned28-file Pan2023 source manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    sessions = []
    if manifest_path.is_file():
        old = read(manifest_path)
        if old.get("signal_contract_sha256") != contract_sha or old.get("source_manifest_sha256") != sha(source_manifest):
            raise ValueError("Existing Pan2023 cache belongs to a different source/signal recipe")
        sessions = list(old["sessions"])
    done = {(s["participant_id"], s["session_id"]) for s in sessions}
    def persist():
        v = {"dataset_id": "Pan2023", "status": "PAN2023_EXTERNAL_CACHE_COMPLETE" if len(sessions) == 28 else "PAN2023_EXTERNAL_CACHE_PARTIAL",
             "signal_contract_sha256": contract_sha, "source_manifest_sha256": sha(source_manifest),
             "source_doi": "10.7910/DVN/251NOW", "source_mirror": "nm000291/v1.0.0",
             "license": "CC0-1.0", "order_scope": ORDER_SCOPE,
             "acquisition_order_verified": False, "physical_run_id_available": False,
             "temporal_risk_eligible": False,
             "query_policy": "fixed_valid_original_session_trial_positions_after_B20_for_all_budgets",
             "sessions": sorted(sessions, key=lambda s: (s["participant_id"], s["session_id"]))}
        put(manifest_path, v); return v
    for e in entries:
        subject, session = int(e["subject"]), str(e["session"])
        person = f"PAN{subject:02d}"
        if (person, session) in done:
            continue
        p = download_source(e, raw_dir) if download else raw_dir / str(e["file_id"])
        if not p.is_file() or p.stat().st_size != e["size"]:
            raise FileNotFoundError("Pinned Pan2023 raw file absent; prepare with --download")
        x, rows, schema = preprocess_mat(p, contract, subject, session, e["checksum"])
        d = output / person / ("D" + str(int(session) + 1)); d.mkdir(parents=True, exist_ok=True)
        xp, ep, sp = d / "X.npy", d / "events.json", d / "schema.json"
        def save_array(path):
            with open(path, "wb") as f: np.save(f, x, allow_pickle=False)
        atomic(xp, save_array); put(ep, rows); put(sp, schema)
        sessions.append({"participant_id": person, "session_id": session, "partition": "external_robustness",
                         "X": _artifact(output, xp), "events": _artifact(output, ep), "schema": _artifact(output, sp),
                         "source_sha256": e["checksum"], "source_bytes": e["size"],
                         "acquisition_order_verified": False, "physical_run_id_available": False})
        persist(); emit("PAN2023_SESSION_PREPARED", participant=person, session=session, shape=list(x.shape), valid=sum(r["valid"] for r in rows))
    return persist()


class PanCatalog:
    """Read only existing cache; preserve two real sessions per participant."""
    def __init__(self, root, manifest_path, *, cache_gb=8):
        from .data import Plan
        self.root, self.role = Path(root), "external_robustness"
        manifest_path = Path(manifest_path).resolve(); self.cache_root = manifest_path.parent
        manifest = read(manifest_path)
        if (manifest.get("dataset_id") != "Pan2023" or manifest.get("status") != "PAN2023_EXTERNAL_CACHE_COMPLETE"
                or manifest.get("order_scope") != ORDER_SCOPE or manifest.get("temporal_risk_eligible") is not False):
            raise ValueError("Missing/incompatible Pan2023 external cache")
        self.signal_sha = validate_contract(read(self.root / "contracts/input_contract.json"))
        if manifest.get("signal_contract_sha256") != self.signal_sha:
            raise ValueError("Pan2023/source signal contracts differ")
        self.identity = {"manifest_sha256": sha(manifest_path), "signal_contract_sha256": self.signal_sha,
                         "dataset_id": "Pan2023", "order_scope": ORDER_SCOPE}
        self.plans, self.excluded, identities = [], [], set()
        for s in manifest["sessions"]:
            person, session = str(s["participant_id"]), str(s["session_id"])
            if (person, session) in identities:
                raise ValueError("Duplicate Pan2023 participant/session identity")
            identities.add((person, session))
            rows = read(self._artifact(s["events"])); xp = self._artifact(s["X"], verify=False)
            if (len(rows) != 120 or any(r.get("label") not in (0, 1) or type(r.get("valid")) is not bool
                    or r.get("cache_index") != i or r.get("original_trial_index") != i
                    or r.get("physical_run_id") is not None for i, r in enumerate(rows))):
                raise ValueError("Pan2023 rows violate original trial-index/unknown-block contract")
            uids = [r["event_uid"] for r in rows]
            if len(set(uids)) != 120:
                raise ValueError("Duplicate Pan2023 trial identity")
            query = [r for r in rows[20:] if r["valid"]]
            if len(query) < 20 or any(sum(r["label"] == k for r in query) < 5 for k in (0, 1)):
                self.excluded.append({"participant": person, "session": session, "reason": "no_two_class_query_after_B20"})
                continue
            audit = {"acquired_prefix_uids": uids[:20], "query_uids": [r["event_uid"] for r in query],
                     "query_eligible": True, "order_scope": ORDER_SCOPE, "acquisition_order_verified": False,
                     "temporal_risk_eligible": False, "physical_run_id_available": False,
                     "budget": 20, "query_policy": manifest["query_policy"]}
            self.plans.append(Plan(person, session, xp, {r["event_uid"]: r for r in rows}, audit, self.role))
        self.plans.sort(key=lambda p: (p.participant_id, p.session_id))
        self.people = sorted({p.participant_id for p in self.plans})
        self.by_person = {p: [s for s in self.plans if s.participant_id == p] for p in self.people}
        if not self.plans:
            raise ValueError("No usable Pan2023 external sessions")
        self.cache, self.cache_bytes, self.cache_limit = OrderedDict(), 0, int(cache_gb * 1024**3)
        emit("PAN2023_CATALOG_READY", people=len(self.people), sessions=len(self.plans), order_scope=ORDER_SCOPE)

    def _artifact(self, value, *, verify=True):
        path = (self.cache_root / value["path"]).resolve()
        if (not path.is_relative_to(self.cache_root) or not path.is_file()
                or ("bytes" in value and path.stat().st_size != value["bytes"])
                or (verify and sha(path) != value["sha256"])):
            raise ValueError("Pan2023 cache path/hash mismatch")
        return path

    def array(self, plan):
        from .data import Catalog
        key = str(plan.array)
        if key in self.cache:
            return Catalog.array(self, plan)
        x = Catalog.array(self, plan)
        try:
            if x.shape != (120, 11, 128) or x.dtype != np.dtype("float32"):
                raise ValueError("Pan2023 pre-RMS cache shape/type mismatch")
            valid = x[[r["cache_index"] for r in plan.rows.values() if r["valid"]]]
            if not np.isfinite(valid).all() or np.any(np.square(valid.astype(np.float64)).sum(axis=(1, 2)) == 0):
                raise ValueError("Pan2023 valid rows contain nonfinite/zero signal")
        except Exception:
            bad = self.cache.pop(key, None)
            if bad is not None:self.cache_bytes -= bad.nbytes
            raise
        return x

    def materialize(self, plan, device, *, budget=20):
        import torch
        from p3work.adapt import CalibrationPrefix
        if budget not in (4, 8, 12, 16, 20):
            raise ValueError("Unsupported Pan2023 calibration budget")
        acquired = tuple(plan.audit["acquired_prefix_uids"][:budget])
        support_ids = tuple(u for u in acquired if plan.rows[u]["valid"])
        query_ids = tuple(plan.audit["query_uids"]); value = self.array(plan)
        def tensors(ids):
            rows = [plan.rows[u] for u in ids]
            x = np.array(value[[r["cache_index"] for r in rows]], dtype=np.float64)
            return torch.from_numpy(x).to(device), torch.tensor([r["label"] for r in rows], device=device, dtype=torch.long)
        sx, sy = tensors(support_ids); qx, qy = tensors(query_ids)
        return SimpleNamespace(support=CalibrationPrefix(sx, sy, support_ids, acquired, self.signal_sha, "calibration_prefix"),
                               query_x=qx, query_y=qy, query_ids=query_ids,
                               run_ids=tuple("unknown" for _ in query_ids), order_scope=ORDER_SCOPE,
                               temporal_risk_eligible=False)

    def draw(self, seed, cursor):
        from .data import Catalog
        return Catalog.draw(self, seed, cursor)
