"""Atomic trusted local checkpoints with explicit code/data identity and RNG.

These helpers do not train, download, inspect credentials or initialize CUDA.
Only the explicit GPU runner opts in to reading its device RNG state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .contracts import digest, file_digest


def local_path(root, relative):
    root = Path(root).resolve(strict=True)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Explicit root-relative training artifact required")
    result = root / path
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Training artifact symlinks are forbidden")
    if not result.resolve().is_relative_to(root):
        raise ValueError("Training artifact escaped the research root")
    return result


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capture_rng(*, cuda=False):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch_cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if cuda else None}


def restore_rng(state, *, cuda=False):
    if (state.get("cuda") is not None) != cuda:
        raise ValueError("Checkpoint CUDA RNG profile differs from the current run")
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if cuda:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, *, identity, state, cuda=False):
    """Write bytes first, then a SHA sidecar; a torn pair fails closed on load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": "paper3-trusted-training-state-v1", "identity": identity,
              "identity_sha256": digest(identity), "state": state, "rng": capture_rng(cuda=cuda)}
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    sha = file_digest(path)
    atomic_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": sha, "bytes": path.stat().st_size,
                                                              "identity_sha256": digest(identity)})
    return sha


def load_checkpoint(path, *, identity, cuda=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Checkpoint symlink forbidden")
    sidecar = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if (sidecar.get("sha256") != file_digest(path) or sidecar.get("bytes") != path.stat().st_size
            or sidecar.get("identity_sha256") != digest(identity)):
        raise ValueError("Checkpoint full SHA/bytes/run identity mismatch")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (record.get("schema_version") != "paper3-trusted-training-state-v1"
            or record.get("identity_sha256") != digest(identity) or record.get("identity") != identity):
        raise ValueError("Checkpoint code/data/config identity mismatch")
    restore_rng(record["rng"], cuda=cuda)
    return record["state"]


def sha_uids(uids):
    values = tuple(sorted(uids))
    if not values or len(set(values)) != len(values) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Unique original training UID list required")
    return hashlib.sha256("\n".join(values).encode()).hexdigest()
