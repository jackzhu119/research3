"""File contracts, non-pickle arrays, leakage checks and reproducible splits."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from pathlib import Path
import numpy as np

SCHEMA = "paper3-mi-shards-v1"
LABELS = {"left_hand": 0, "right_hand": 1}

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")

def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def array_hash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(canonical({"dtype": x.dtype.str, "shape": list(x.shape)}))
    h.update(x.tobytes())
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, obj):
    atomic_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True,
                                allow_nan=False).encode("utf-8") + b"\n")

def atomic_bytes(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def save_arrays(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def environment():
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "mne", "moabb", "torch"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}

def code_identity():
    here = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(here.glob("*.py"))}

@contextlib.contextmanager
def exclusive_lock(path):
    """Kernel lock releases on process exit; stale files are not stale locks."""
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError(f"Active job holds {path}") from e
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)

def natural_key(text):
    return tuple((1, int(x)) if x.isdigit() else (0, x.lower())
                 for x in re.split(r"(\d+)", str(text)))

def validate_rows(rows, y):
    if len(rows) != len(y) or not rows: raise ValueError("Empty or mismatched trial metadata")
    seen = set()
    for row, label in zip(rows, y):
        required = {"dataset", "subject", "session", "run", "event_uid", "onset_s",
                    "session_order", "run_order", "sample", "label", "signal_sha256"}
        if not required <= row.keys(): raise ValueError("Missing trial identity fields")
        if row["event_uid"] in seen: raise ValueError("Duplicate event_uid: " + row["event_uid"])
        seen.add(row["event_uid"])
        if label not in (0, 1) or row["label"] != int(label): raise ValueError("Invalid label")
        if not np.isfinite(row["onset_s"]): raise ValueError("Non-finite onset")

def person_key(row):
    # Provider IDs are namespaced, not proof of biological independence.
    return (row["dataset"], str(row["subject"]))

def assert_subject_disjoint(rows, train, validation):
    if not len(train) or not len(validation): raise ValueError("Empty train/validation")
    if set(train) & set(validation): raise ValueError("Trial index leakage")
    people_train = {person_key(rows[i]) for i in train}
    people_val = {person_key(rows[i]) for i in validation}
    if people_train & people_val: raise ValueError("Subject leakage")
    uid_train = {rows[i]["event_uid"] for i in train}
    if uid_train & {rows[i]["event_uid"] for i in validation}: raise ValueError("Event leakage")
    hashes_train = {rows[i]["signal_sha256"] for i in train}
    if hashes_train & {rows[i]["signal_sha256"] for i in validation}:
        raise ValueError("Exact processed-signal duplicate crosses split")

def source_lodo(rows):
    datasets = sorted({r["dataset"] for r in rows})
    if len(datasets) < 2: raise ValueError("LODO requires at least two source datasets")
    folds = {}
    for dataset in datasets:
        train = np.array([i for i, r in enumerate(rows) if r["dataset"] != dataset], dtype=int)
        val = np.array([i for i, r in enumerate(rows) if r["dataset"] == dataset], dtype=int)
        assert_subject_disjoint(rows, train, val)
        folds[dataset] = (train, val)
    return folds

def chronological_support_query(rows, subject, session, support_run_count=1):
    """Future adaptation helper: whole early runs support, later runs query.

    This helper does not execute an adaptation experiment. Session order is a
    provider/adapter order; it is not independently verified acquisition time.
    """
    if support_run_count < 1: raise ValueError("Support needs at least one whole early run")
    selected = [i for i, r in enumerate(rows)
                if person_key(r) == tuple(subject) and str(r["session"]) == str(session)]
    runs = sorted({rows[i]["run_order"] for i in selected})
    if len(runs) <= support_run_count: raise ValueError("Need later whole query runs")
    support_runs = set(runs[:support_run_count])
    order = lambda i: (rows[i]["run_order"], rows[i]["sample"], rows[i]["event_uid"])
    support = sorted((i for i in selected if rows[i]["run_order"] in support_runs), key=order)
    query = sorted((i for i in selected if rows[i]["run_order"] not in support_runs), key=order)
    return np.array(support), np.array(query)

def chronological_session_split(rows, subject, early_session_count=1):
    """Unused future adaptation helper: complete early sessions, then later ones.

    This is a within-person temporal split, not source train/validation, which
    always uses subject-disjoint source LODO. Adapter order needs source audit
    before any longitudinal scientific claim.
    """
    if early_session_count < 1: raise ValueError("Need at least one whole early session")
    selected=[i for i,r in enumerate(rows) if person_key(r)==tuple(subject)]
    sessions=sorted({rows[i]["session_order"] for i in selected})
    if len(sessions)<=early_session_count: raise ValueError("Need later whole query sessions")
    early=set(sessions[:early_session_count])
    order=lambda i:(rows[i]["session_order"],rows[i]["run_order"],rows[i]["sample"],rows[i]["event_uid"])
    before=sorted((i for i in selected if rows[i]["session_order"] in early),key=order)
    after=sorted((i for i in selected if rows[i]["session_order"] not in early),key=order)
    return np.array(before),np.array(after)

def save_shard(folder, subject, X, y, rows, header):
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int64)
    validate_rows(rows, y)
    if X.ndim != 3 or X.shape[0] != len(y) or not np.isfinite(X).all():
        raise ValueError("Invalid EEG array")
    if header.get("units") != "V": raise ValueError("Unified interface requires volts")
    for row, trial in zip(rows, X):
        if row["signal_sha256"] != array_hash(trial): raise ValueError("Trial hash mismatch")
    base = Path(folder) / ("subject-" + str(subject))
    npz, meta = base.with_suffix(".npz"), base.with_suffix(".json")
    if npz.exists() or meta.exists(): raise FileExistsError("Preserve existing shard: " + str(base))
    save_arrays(npz, X=X, y=y)
    write_json(meta, {"schema": SCHEMA, **header, "npz_sha256": file_hash(npz), "rows": rows})
    return {"subject": str(subject), "npz": npz.name, "json": meta.name,
            "npz_sha256": file_hash(npz), "json_sha256": file_hash(meta), "n_trials": len(y)}

def load_collection(folder, allowed_roles=("source",)):
    root = Path(folder); arrays=[]; labels=[]; rows=[]; contract=None; identities=[]
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = read_json(manifest_path)
        if manifest["role"] not in allowed_roles: continue
        for shard in manifest["shards"]:
            npz = manifest_path.parent / shard["npz"]
            meta_path = manifest_path.parent / shard["json"]
            for path, key in ((npz, "npz_sha256"), (meta_path, "json_sha256")):
                if file_hash(path) != shard[key]: raise ValueError("Shard SHA mismatch: " + str(path))
            meta = read_json(meta_path)
            if meta["schema"] != SCHEMA or meta["units"] != "V": raise ValueError("Unsupported interface")
            current = {k: meta[k] for k in ("channels", "sfreq", "n_times", "units", "preprocessing", "processing_implementation")}
            if contract is not None and current != contract: raise ValueError("Different preprocessing/montage contracts")
            contract = current
            with np.load(npz, allow_pickle=False) as z: X=z["X"]; y=z["y"]
            if X.shape != (len(y), len(meta["channels"]), meta["n_times"]): raise ValueError("Shape mismatch")
            if not np.isfinite(X).all(): raise ValueError("Nonfinite EEG")
            validate_rows(meta["rows"], y)
            for row, trial in zip(meta["rows"], X):
                if row["signal_sha256"] != array_hash(trial): raise ValueError("Signal hash mismatch")
            arrays.append(X); labels.append(y); rows.extend(meta["rows"])
            identities.append({"manifest": str(manifest_path), "sha256": file_hash(manifest_path), **shard})
    if not arrays: raise ValueError("No prepared source shards. Run explicit prep first.")
    y = np.concatenate(labels); validate_rows(rows, y)
    return np.concatenate(arrays), y, rows, contract, identities

def standardized_trials(X):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(X * X, axis=(-2, -1), keepdims=True))
    if (rms <= 0).any() or not np.isfinite(rms).all(): raise ValueError("Invalid trial RMS")
    return (X / rms).astype(np.float32)

def participant_metrics(rows, y, probabilities):
    predictions = probabilities.argmax(axis=1)
    metrics=[]
    for person in sorted({person_key(r) for r in rows}):
        idx=np.array([i for i,r in enumerate(rows) if person_key(r)==person])
        recall=[float(np.mean(predictions[idx[y[idx]==c]]==c)) if np.any(y[idx]==c) else None
                for c in (0,1)]
        metrics.append({"dataset": person[0], "subject": person[1], "n_trials": len(idx),
                        "recall_left": recall[0], "recall_right": recall[1],
                        "balanced_accuracy": float(np.mean(recall)) if None not in recall else None})
    defined=[r["balanced_accuracy"] for r in metrics if r["balanced_accuracy"] is not None]
    return {"participants": metrics, "mean_participant_ba": float(np.mean(defined)) if defined else None,
            "n_defined": len(defined), "n_participants": len(metrics)}
