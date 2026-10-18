#!/usr/bin/env python3
"""Stage-gated, local-only SOURCE engineering pilot; default is a read-only dry run.

No API, SSH, RunPod, download, upload, validation-query, or external-data access.
Actual execution needs BOTH --execute and --user-started-gpu, completed W1/W2,
verified new cache artifacts, and CUDA. This produces no scientific f0 receipt.

Cache schema: W2.cache_manifest={path,sha256} binds a JSON manifest whose
schema_version is paper3-source-cache-v2, status SOURCE_CACHE_COMPLETE,
synthetic false, old_npz_reused false, role source_train, and input_contract_sha256
matches validate_signal_contract. Each shard has dataset, role, input_contract_sha256,
stage=pre_rms_car11_temporal_demeaned, X/y/events artifact {path,sha256}, and
adapter_code/processing_code artifact {path,sha256}. X/y must be .npy (no pickle),
events a JSON list, with event_uid, dataset, subject_id, source_record_id, original_trial_id, label,
role, input_contract_sha256, raw_sha256, adapter_sha256 and processing_sha256.
Raw checksums are provenance references, not a claim that raw files were reread
on the GPU host; upstream verified-native manifests remain required by W1/W2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.contracts import SOURCE_IDS, digest, file_digest, validate_signal_contract
from p3work.f0_bridge import CompactEEGNet
from p3work.spatial_geometry import joint_trial_rms

SEED = 20261018
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TOTAL_UPDATES = 200  # 100 continuous + 50 + restored 50, not 200 per branch.
MAX_WALL_SECONDS = 1800
WARMUP_UPDATES = 10  # Included within each branch's first 100; no extra updates.


class PilotGateError(ValueError):
    pass


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PilotGateError("Concrete lowercase SHA256 required")
    return value


def _local(root, relative):
    if not isinstance(relative, str) or not relative:
        raise PilotGateError("Explicit root-relative artifact path required")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() in {".npz", ".pkl", ".pickle"}:
        raise PilotGateError("Legacy NPZ/pickle or escaping paths are forbidden")
    result = (root / path).resolve(strict=True)
    if not result.is_relative_to(root) or "legacy_baseline_v1" in result.parts:
        raise PilotGateError("Artifact escaped the new Paper3 cache root")
    return result


def _artifact(root, item, *, suffix=None):
    if not isinstance(item, dict):
        raise PilotGateError("Artifact path/SHA256 pair required")
    path = _local(root, item.get("path"))
    if suffix is not None and path.suffix.lower() != suffix:
        raise PilotGateError(f"Expected a {suffix} artifact")
    expected = _sha(item.get("sha256"))
    if not path.is_file() or file_digest(path) != expected:
        raise PilotGateError("Cache or code artifact SHA256 mismatch")
    return path


@dataclass
class VerifiedCache:
    root: Path
    n_times: int
    input_contract_sha256: str
    manifest_sha256: str
    manifest_path: str
    arrays: list[np.ndarray]
    labels: np.ndarray
    rows: list[dict]
    locations: list[tuple[int, int]]
    artifact_sha256: dict
    upstream_acceptance_sha256: dict = field(default_factory=dict)

    def batch(self, indices):
        # Gather ONLY authorized source training events, never whole target data.
        return np.stack([self.arrays[self.locations[i][0]][self.locations[i][1]]
                         for i in indices.tolist()])


def _verify_upstream_acceptance(cache):
    """Verify measured stage evidence; COMPLETE strings alone grant no access.

    These read-only verifiers use metadata and local cache bytes only. They do
    not restore raw EEG, contact R2, construct a model or initialize CUDA.
    """
    from p3work import (window_acceptance, source_cache_acceptance, contracts,
                        f0_bridge, spatial_geometry, source_training, runtime_clock)
    for module in (window_acceptance, source_cache_acceptance, contracts,
                   f0_bridge, spatial_geometry, source_training, runtime_clock):
        relative = "p3work/" + Path(module.__file__).name
        packaged = _local(cache.root, relative)
        if packaged.read_bytes() != Path(module.__file__).read_bytes():
            raise PilotGateError("Executed upstream verifier differs from packaged code")
        cache.artifact_sha256[relative] = file_digest(packaged)
    runner = _local(cache.root, "tools/run_controlled_pilot.py")
    if runner.read_bytes() != Path(__file__).read_bytes():
        raise PilotGateError("Executed upstream verifier differs from packaged code")
    cache.artifact_sha256["tools/run_controlled_pilot.py"] = file_digest(runner)
    window = window_acceptance.verify_frozen_window(cache.root)
    source = source_cache_acceptance.verify_source_cache_acceptance(cache)
    # The window API returns a proof digest; the source API returns a map of
    # actual file digests. Keep these distinct instead of treating a proof as
    # an artifact path or silently dropping its measured decision identity.
    for name in ("selection_plan", "split", "inventory", "window_input_session_manifest", "window_coverage"):
        descriptor = window[name]
        path = _artifact(cache.root, descriptor, suffix=".json")
        cache.artifact_sha256[path.relative_to(cache.root).as_posix()] = descriptor["sha256"]
    cache.artifact_sha256.update(source["artifact_sha256"])
    cache.upstream_acceptance_sha256.update(
        exact_window=_sha(window["artifact_sha256"]),
        full_source_cache=digest(source))


def verify_inputs(root, *, manifest_path=None) -> VerifiedCache:
    """Read-only gates and actual file hashes; no torch/CUDA fitting occurs here."""
    root = Path(root).resolve(strict=True)
    tasks = json.loads((root / "status/tasks.json").read_text())
    if tasks.get("W1", {}).get("status") != "COMPLETE":
        raise PilotGateError("W1 must be COMPLETE before a source pilot")
    if tasks.get("W2", {}).get("status") != "COMPLETE":
        raise PilotGateError("W2 new four-source caches must be COMPLETE")
    contract = json.loads((root / "contracts/input_contract.json").read_text())
    signal_sha = validate_signal_contract(contract)
    descriptor = tasks["W2"].get("cache_manifest")
    path = _artifact(root, descriptor, suffix=".json")
    if manifest_path is not None and _local(root, manifest_path) != path:
        raise PilotGateError("Requested cache manifest is not the W2 receipt-bound manifest")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != "paper3-source-cache-v2" or
            manifest.get("status") != "SOURCE_CACHE_COMPLETE" or
            manifest.get("synthetic") is not False or manifest.get("old_npz_reused") is not False or
            manifest.get("role") != "source_train"):
        raise PilotGateError("New, real, source_train cache provenance is required; no legacy NPZ")
    if manifest.get("input_contract_sha256") != signal_sha:
        raise PilotGateError("Cache input contract differs from the current frozen input")
    shards = manifest.get("shards", [])
    if not isinstance(shards, list) or not shards or {s.get("dataset") for s in shards} != SOURCE_IDS:
        raise PilotGateError("All and only four exact registered source dataset IDs are required")
    arrays, labels, rows, locations, seen, original_seen = [], [], [], [], set(), set()
    artifacts = {str(path.relative_to(root)): file_digest(path),
                 "status/tasks.json": file_digest(root / "status/tasks.json"),
                 "contracts/input_contract.json": file_digest(root / "contracts/input_contract.json")}
    for shard in shards:
        if (shard.get("role") != "source_train" or shard.get("input_contract_sha256") != signal_sha or
                shard.get("stage") != "pre_rms_car11_temporal_demeaned"):
            raise PilotGateError("Shard role, signal identity or pre-RMS stage mismatch")
        paths = {key: _artifact(root, shard.get(key), suffix=".npy" if key in {"X", "y"} else ".json" if key == "events" else ".py")
                 for key in ("X", "y", "events", "adapter_code", "processing_code")}
        if (paths["adapter_code"] != (root / "p3work/adapters/source_native.py").resolve() or
                paths["processing_code"] != (root / "p3work/preprocess.py").resolve()):
            raise PilotGateError("Cache must bind the actual new native adapter and preprocessing code")
        artifacts.update({str(p.relative_to(root)): file_digest(p) for p in paths.values()})
        x = np.load(paths["X"], mmap_mode="r", allow_pickle=False)
        y = np.load(paths["y"], mmap_mode="r", allow_pickle=False)
        events = json.loads(paths["events"].read_text())
        if x.ndim != 3 or x.shape[1:] != (11, contract["n_times"]) or x.dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise PilotGateError("X must contain new float32/64 N x 11 x frozen_time trials")
        if y.shape != (len(x),) or y.dtype.kind not in "iu" or not len(x) or not isinstance(events, list) or len(events) != len(x):
            raise PilotGateError("X/y/original-event count or label dtype mismatch")
        if not np.isin(y, [0, 1]).all():
            raise PilotGateError("Only explicit left/right 0/1 labels are allowed")
        for start in range(0, len(x), 64):
            chunk = np.asarray(x[start:start + 64], dtype=np.float64)
            if not np.isfinite(chunk).all():
                raise PilotGateError("Nonfinite source cache trial")
            rms = np.sqrt(np.mean(chunk ** 2, axis=(1, 2)))
            if not np.isfinite(rms).all() or np.any(rms == 0):
                raise PilotGateError("Zero/nonfinite source cache energy")
            if (np.any(np.abs(chunk.mean(1)).max(1) > rms * 5e-6) or
                    np.any(np.abs(chunk.mean(2)).max(1) > rms * 5e-6)):
                raise PilotGateError("Cache does not satisfy CAR11 and per-channel temporal demeaning")
        shard_index = len(arrays)
        for i, row in enumerate(events):
            if (not isinstance(row, dict) or row.get("role") != "source_train" or
                    row.get("dataset") != shard["dataset"] or row.get("input_contract_sha256") != signal_sha):
                raise PilotGateError("Only input-bound source_train original events can enter this runner")
            uid = row.get("event_uid")
            if not isinstance(uid, str) or not uid or uid in seen or not str(row.get("subject_id", "")) or not row.get("source_record_id") or row.get("original_trial_id") is None:
                raise PilotGateError("Missing/duplicated original-event or subject identity")
            for name in ("raw_sha256", "adapter_sha256", "processing_sha256"):
                _sha(row.get(name))
            if (row["adapter_sha256"] != shard["adapter_code"]["sha256"] or
                    row["processing_sha256"] != shard["processing_code"]["sha256"] or
                    type(row.get("label")) is not int or row["label"] != int(y[i])):
                raise PilotGateError("Original-event code identity or label disagrees with the cache")
            original = (row["dataset"], row["raw_sha256"], str(row["source_record_id"]), str(row["original_trial_id"]))
            if original in original_seen:
                raise PilotGateError("Repeated original trial under different event UIDs")
            seen.add(uid)
            original_seen.add(original)
            rows.append(row)
            locations.append((shard_index, i))
        arrays.append(x)
        labels.append(np.asarray(y, dtype=np.int64))
    cache = VerifiedCache(root, contract["n_times"], signal_sha, file_digest(path),
                          str(path.relative_to(root)), arrays, np.concatenate(labels), rows, locations, artifacts)
    _verify_upstream_acceptance(cache)
    return cache


def sampling_weights(cache):
    """Equal source dataset, then subject, then observed class, then trial."""
    subjects = defaultdict(set)
    classes = defaultdict(set)
    counts = Counter()
    for row, label in zip(cache.rows, cache.labels):
        dataset, subject, label = row["dataset"], str(row["subject_id"]), int(label)
        subjects[dataset].add(subject)
        classes[dataset, subject].add(label)
        counts[dataset, subject, label] += 1
    return torch.tensor([1 / (len(SOURCE_IDS) * len(subjects[row["dataset"]]) *
                                len(classes[row["dataset"], str(row["subject_id"])]) *
                                counts[row["dataset"], str(row["subject_id"]), int(label)])
                         for row, label in zip(cache.rows, cache.labels)], dtype=torch.float64)


def _atomic_json(path, record):
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_checkpoint(path, record):
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _same_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_same_tree(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _checkpoint(model, optimizer, sampler, update, identity):
    numpy_rng = np.random.get_state()
    return {"scope": "ENGINEERING_SOURCE_PILOT_ONLY", "identity": identity,
            "identity_sha256": digest(identity), "update": update,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(0), "python_rng": random.getstate(),
            "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2], numpy_rng[3], numpy_rng[4]]}


def _restore(saved, model, optimizer, sampler, identity):
    if saved.get("scope") != "ENGINEERING_SOURCE_PILOT_ONLY" or saved.get("identity_sha256") != digest(identity) or saved.get("identity") != identity:
        raise PilotGateError("Checkpoint code/data/config/device identity changed")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    sampler.set_state(saved["sampler_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"], 0)
    random.setstate(saved["python_rng"])
    state = saved["numpy_rng"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))


def _new_state(n_times):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = CompactEEGNet(11, n_times, dropout=0.25).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0, foreach=False)
    sampler = torch.Generator(device="cpu").manual_seed(SEED + 1000003)
    return model, optimizer, sampler


def _timing_summary(steps):
    """Pure CPU summarizer; fixed updates 0..9 are warmup, 10..99 measured.

    CUDA events measure elapsed stream intervals, including possible host gaps;
    these are not a sum of kernel durations and are not GPU billing time.
    """
    measured = [row for row in steps if row["branch_update_zero_based"] >= WARMUP_UPDATES]
    def distribution(key):
        values = [row[key] for row in measured]
        return {"count": len(values), "p50_seconds": float(np.percentile(values, 50)) if values else None,
                "p90_seconds": float(np.percentile(values, 90)) if values else None,
                "total_seconds": float(sum(values))}
    return {"warmup_updates_in_this_segment": len(steps) - len(measured),
            "measured_updates_in_this_segment": len(measured),
            "warmup_rule": "branch update indices 0..9; included in the fixed optimizer-update budget",
            "cpu_sampling_normalization_total_seconds": sum(row["cpu_sampling_normalization_seconds"] for row in steps),
            "measured_cpu_sampling_normalization": distribution("cpu_sampling_normalization_seconds"),
            "measured_h2d": distribution("gpu_h2d_seconds"),
            "measured_forward_backward_step": distribution("gpu_forward_backward_step_seconds"),
            "percentile_method": "numpy.percentile default linear interpolation",
            "billing_time": False, "timing_scope": "CUDA event elapsed stream intervals; monotonic wall includes host preparation; neither is an account billing receipt",
            "per_update_timings": steps}


def _updates(cache, model, optimizer, sampler, begin, stop, *, deadline, progress, phase):
    wall_start = time.monotonic()
    weights = sampling_weights(cache)
    model.train()
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    markers = []
    event_start.record()
    for update in range(begin, stop):
        if time.monotonic() > deadline:
            raise TimeoutError("Controlled engineering pilot wall-time budget reached")
        cpu_start = time.monotonic()
        indices = torch.multinomial(weights, BATCH_SIZE, replacement=True, generator=sampler)
        host_batch = joint_trial_rms(torch.from_numpy(cache.batch(indices))).float()
        host_labels = torch.from_numpy(cache.labels[indices.numpy()].copy())
        cpu_prepare_seconds = time.monotonic() - cpu_start
        transfer_start, transfer_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        step_start, step_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        transfer_start.record()
        batch = host_batch.to("cuda:0")
        labels = host_labels.to("cuda:0")
        transfer_end.record()
        optimizer.zero_grad(set_to_none=True)
        step_start.record()
        loss = nn.functional.cross_entropy(model(batch), labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite source engineering loss")
        progress["training_started"] = True
        loss.backward()
        optimizer.step()
        step_end.record()
        progress["optimizer_updates_dispatched"] += 1
        markers.append((update, cpu_prepare_seconds, transfer_start, transfer_end, step_start, step_end))
    event_end.record()
    torch.cuda.synchronize(0)
    progress["actual_completed_updates"] += stop - begin
    elapsed_wall = time.monotonic() - wall_start
    steps = [{"branch_update_zero_based": update, "cpu_sampling_normalization_seconds": cpu,
              "gpu_h2d_seconds": transfer_start.elapsed_time(transfer_end) / 1000,
              "gpu_forward_backward_step_seconds": step_start.elapsed_time(step_end) / 1000}
             for update, cpu, transfer_start, transfer_end, step_start, step_end in markers]
    return {"phase": phase, "begin_update": begin, "end_update": stop, "actual_updates": stop - begin,
            "cuda_event_seconds": event_start.elapsed_time(event_end) / 1000,
            "monotonic_wall_seconds": elapsed_wall,
            "timing_summary": _timing_summary(steps),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def execute_pilot(cache, output_relative, *, user_started_gpu):
    """Only explicit CLI execution calls this; never invoked by fixture tests."""
    if not user_started_gpu:
        raise PilotGateError("Both --execute and --user-started-gpu are required")
    if not torch.cuda.is_available():
        raise PilotGateError("CUDA is unavailable; no CPU training fallback is permitted")
    root = cache.root
    relative = Path(output_relative)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("runs",):
        raise PilotGateError("Pilot outputs must use a new root-relative runs/ directory")
    output = (root / relative).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise PilotGateError("Output exists or escaped the execution root; no duplicate pilot")
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    claim = runs / "controlled_pilot_job_claim.json"
    # A persistent one-job claim prevents accidental reexecution with another output path.
    try:
        with claim.open("x") as stream:
            json.dump({"scope": "ENGINEERING_SOURCE_PILOT_ONLY", "max_jobs": 1,
                       "output": str(relative), "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
    except FileExistsError:
        raise PilotGateError("The one controlled pilot job was already claimed; inspect its receipt") from None
    output.mkdir()
    receipt = {"status": "RUNNING", "scope": "ENGINEERING_SOURCE_PILOT_ONLY",
               "scientific_f0": False, "source_lodo_selection_verified": False,
               "training_started": False,
               "remote_actions": False, "server_started_by_runner": False,
               "user_started_gpu_attested": True, "seed": SEED, "max_jobs": 1,
               "actual_completed_updates": 0, "exact_resume_verified": False,
               "optimizer_updates_dispatched": 0,
               "completed_update_count_basis": "CUDA-synchronized complete segments; a lower bound on failure",
               "started_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(output / "engineering_receipt.json", receipt)
    wall_start = time.monotonic()
    try:
        from p3work.runtime_clock import collect_clock_preflight
        receipt["clock_preflight"] = collect_clock_preflight()
        _atomic_json(output / "engineering_receipt.json", receipt)
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") not in {":4096:8", ":16:8"}:
            raise PilotGateError("CUBLAS workspace configuration is incompatible with deterministic execution")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        try:
            driver_query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=10, check=True)
            driver_version = driver_query.stdout.strip().splitlines()
        except (FileNotFoundError, subprocess.SubprocessError):
            driver_version = None
        identity = {"seed": SEED, "batch_size": BATCH_SIZE, "optimizer": "Adam",
                    "learning_rate": LEARNING_RATE, "weight_decay": 0, "foreach": False,
                    "dropout": 0.25, "torch": str(torch.__version__), "numpy": np.__version__,
                    "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "device": "cuda:0", "device_name": props.name,
                    "device_uuid": str(getattr(props, "uuid", "not_exposed")),
                    "device_total_memory": props.total_memory, "compute_capability": [props.major, props.minor],
                    "driver_versions_local_nvidia_smi": driver_version, "python": sys.version,
                    "tf32": False, "mixed_precision": False, "deterministic_algorithms": True,
                    "sampling": "equal dataset/subject/observed-class/trial with replacement",
                    "input_contract_sha256": cache.input_contract_sha256,
                    "cache_manifest_sha256": cache.manifest_sha256,
                    "cache_artifacts": cache.artifact_sha256,
                    "upstream_acceptance_sha256": cache.upstream_acceptance_sha256,
                    "runner_sha256": file_digest(__file__),
                    "bridge_sha256": file_digest(root / "p3work/f0_bridge.py"),
                    "geometry_sha256": file_digest(root / "p3work/spatial_geometry.py"),
                    "contracts_sha256": file_digest(root / "p3work/contracts.py")}
        _atomic_json(output / "identity.json", identity)
        deadline = wall_start + MAX_WALL_SECONDS
        model, optimizer, sampler = _new_state(cache.n_times)
        initial = _checkpoint(model, optimizer, sampler, 0, identity)
        _atomic_checkpoint(output / "engineering_initial.pt", initial)
        timings = [_updates(cache, model, optimizer, sampler, 0, 100, deadline=deadline, progress=receipt, phase="continuous100_primary_timing")]
        _atomic_checkpoint(output / "engineering_continuous100.pt", _checkpoint(model, optimizer, sampler, 100, identity))
        receipt.update(actual_completed_updates=100, timing_segments=timings)
        _atomic_json(output / "engineering_receipt.json", receipt)
        # Destroy the first branch before constructing the replay branch on this GPU.
        del initial, model, optimizer, sampler
        torch.cuda.empty_cache()
        model, optimizer, sampler = _new_state(cache.n_times)
        fresh = _checkpoint(model, optimizer, sampler, 0, identity)
        original = torch.load(output / "engineering_initial.pt", map_location="cpu", weights_only=True)
        if not all(_same_tree(fresh[key], original[key]) for key in ("model", "optimizer", "sampler_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng")):
            raise RuntimeError("Continuous and resume branches did not share the exact initial state")
        del fresh, original
        timings.append(_updates(cache, model, optimizer, sampler, 0, 50, deadline=deadline, progress=receipt, phase="resume_before_checkpoint"))
        _atomic_checkpoint(output / "engineering_resume50.pt", _checkpoint(model, optimizer, sampler, 50, identity))
        receipt.update(actual_completed_updates=150, timing_segments=timings)
        _atomic_json(output / "engineering_receipt.json", receipt)
        del model, optimizer, sampler
        torch.cuda.empty_cache()
        model, optimizer, sampler = _new_state(cache.n_times)
        checkpoint_path = output / "engineering_resume50.pt"
        checkpoint_sha = file_digest(checkpoint_path)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _restore(saved, model, optimizer, sampler, identity)
        if saved["update"] != 50 or file_digest(checkpoint_path) != checkpoint_sha:
            raise PilotGateError("Resume checkpoint update/integrity changed")
        del saved
        timings.append(_updates(cache, model, optimizer, sampler, 50, 100, deadline=deadline, progress=receipt, phase="resume_after_checkpoint_reload"))
        _atomic_checkpoint(output / "engineering_resumed100.pt", _checkpoint(model, optimizer, sampler, 100, identity))
        receipt.update(actual_completed_updates=TOTAL_UPDATES, timing_segments=timings)
        continuous = torch.load(output / "engineering_continuous100.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(output / "engineering_resumed100.pt", map_location="cpu", weights_only=True)
        compared = ("model", "optimizer", "sampler_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "update", "identity_sha256")
        same = {key: _same_tree(continuous[key], resumed[key]) for key in compared}
        receipt.update(resume_comparison=same, exact_resume_verified=all(same.values()))
        if not all(same.values()):
            raise RuntimeError("Same-device 100 versus 50+50 checkpoint continuation differs")
        receipt["status"] = "ENGINEERING_PILOT_COMPLETE"
    except BaseException as error:
        receipt.update(status="ENGINEERING_PILOT_FAILED", failure_type=type(error).__name__,
                       failure=str(error), exact_resume_verified=bool(receipt.get("exact_resume_verified", False)))
    finally:
        receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                       total_monotonic_wall_seconds=time.monotonic() - wall_start,
                       checkpoint_files={p.name: file_digest(p) for p in output.glob("*.pt")})
        _atomic_json(output / "engineering_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-manifest", default=None)
    parser.add_argument("--output", default="runs/controlled_source_pilot_seed20261018")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-started-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute and not args.user_started_gpu:
            raise PilotGateError("Both --execute and --user-started-gpu are required; the runner never boots a server")
        cache = verify_inputs(args.root, manifest_path=args.cache_manifest)
        if not args.execute:
            result = {"status": "DRY_RUN_GATES_PASSED", "training_started": False,
                      "gpu_checked": False, "remote_actions": False,
                      "input_contract_sha256": cache.input_contract_sha256,
                      "cache_manifest_sha256": cache.manifest_sha256,
                      "source_datasets": sorted({r["dataset"] for r in cache.rows}),
                      "original_training_trials": len(cache.rows), "planned_actual_updates": TOTAL_UPDATES,
                      "resume_design": "100 continuous versus independently initialized 50+checkpoint reload+50 on same device",
                      "actual_gpu_resume_verified": False}
        else:
            result = execute_pilot(cache, args.output, user_started_gpu=args.user_started_gpu)
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") != "ENGINEERING_PILOT_FAILED" else 3
    except (PilotGateError, ValueError, FileNotFoundError, KeyError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error),
                          "training_started": False, "remote_actions": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
