"""Read the existing frozen cache; no downloads, source retraining or whole-tree scan."""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from p3work.adapt import CalibrationPrefix
from p3work.f0_bridge import CompactEEGNet
from p3work.spatial_geometry import SpatialGeometry, joint_trial_rms
from p3work.splits import build_prefix_episode
from .io import read, artifact, sha, digest, emit

class FrozenSource(nn.Module):
    """Same encoder/head; freeze once, without CPU hashing on every inner forward."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.source_head = model.classifier
        self.feature_dim = model.classifier.in_features
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def features_and_logits(self, x):
        template = next(self.model.parameters())
        features = self.model.features(x.to(template)[:, None]).flatten(1)
        return features, self.model.classifier(features)

    def source_logits(self, x):
        return self.features_and_logits(joint_trial_rms(x))[1]

def load_source(root, seed, device):
    root = Path(root)
    p = root / f"runs/source_training_v2/f0_receipt_seed-{seed}.json"
    source_info = read(p)
    if source_info.get("status") != "SOURCE_F0_COMPLETE" or source_info.get("pipeline_seed") != seed:
        raise ValueError(f"Source fit not complete for seed {seed}: {p}")
    checkpoint = torch.load(artifact(root, source_info["checkpoint"]), map_location="cpu", weights_only=True)
    contract = read(root / "contracts/input_contract.json")
    if digest(contract) != source_info.get("signal_contract_sha256"):
        raise ValueError("Source checkpoint and current signal window differ")
    model = CompactEEGNet(11, contract["n_times"], checkpoint["config"]["dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    reference_path = root / f"runs/training_campaign_v1/references/seed-{seed}/receipt.json"
    reference = read(reference_path)
    if (reference.get("status") != "SOURCE_REFERENCE_COMPLETE"
            or reference.get("source_f0_receipt_sha256") != sha(p)
            or reference.get("signal_contract_sha256") != source_info.get("signal_contract_sha256")):
        raise ValueError("Reference/source input mismatch")
    matrix = torch.from_numpy(np.load(artifact(root, reference["arrays"]["reference"]), allow_pickle=False)).to(device)
    geometry = SpatialGeometry.from_reference(matrix)
    geometry.validate()
    return FrozenSource(model).to(device), geometry, {
        "source_receipt_sha256": sha(p), "reference_receipt_sha256": sha(reference_path),
        "source_checkpoint_sha256": source_info["checkpoint"]["sha256"],
        "signal_contract_sha256": source_info["signal_contract_sha256"]}

@dataclass
class Plan:
    participant_id: str
    session_id: str
    array: Path
    rows: dict
    audit: dict
    role: str

class Catalog:
    def __init__(self, root, role, *, cache_gb=8):
        self.root, self.role = Path(root), role
        if role not in ("meta_train", "meta_validation", "internal_holdout"):
            raise ValueError("Unknown partition")
        tasks = read(self.root / "status/tasks.json")
        manifest_path = artifact(self.root, tasks["W3"]["cache_manifest"])
        manifest = read(manifest_path)
        if manifest.get("status") != "STIEGER_CACHE_COMPLETE":
            raise ValueError("Existing Stieger cache is incomplete")
        split = read(artifact(self.root, manifest["participant_split"]))
        self.people = sorted(split["partitions"][role], key=lambda p: int(p[1:]))
        flat = [p for values in split["partitions"].values() for p in values]
        if len(flat) != len(set(flat)):
            raise ValueError("Participant split overlaps")
        self.signal_sha = manifest["input_contract_sha256"]
        self.identity = {"manifest_sha256": sha(manifest_path), "split_sha256": split["split_sha256"],
                         "signal_contract_sha256": self.signal_sha}
        self.plans, self.excluded = [], []
        for session in manifest["sessions"]:
            if session["partition"] != role:
                continue
            if session["participant_id"] not in self.people:
                raise ValueError("Session assigned to a different participant partition")
            rows = read(artifact(self.root, session["events"]))
            audit = build_prefix_episode(rows, acquisition_order_verified=session["acquisition_order_verified"], budget=20)
            if not audit["query_eligible"] or len(audit["acquired_prefix_uids"]) != 20:
                self.excluded.append({"participant": session["participant_id"], "session": session["session_id"],
                                      "reason": "original_B20_future_query_ineligible"})
                continue
            # Previous verified large arrays are read on demand, not rehashed in every episode.
            path = artifact(self.root, session["X"], verify=False)
            self.plans.append(Plan(session["participant_id"], str(session["session_id"]), path,
                {r["event_uid"]: r for r in rows}, audit, role))
        self.plans.sort(key=lambda p: (int(p.participant_id[1:]), int(p.session_id)))
        self.by_person = {p: [plan for plan in self.plans if plan.participant_id == p] for p in self.people}
        if any(not plans for plans in self.by_person.values()):
            raise ValueError("A registered participant has no eligible session; report rather than silently drop")
        self.cache, self.cache_bytes, self.cache_limit = OrderedDict(), 0, int(cache_gb * 1024**3)
        emit("CATALOG_READY", role=role, people=len(self.people), sessions=len(self.plans),
             excluded_sessions=len(self.excluded), full_data_rehash=False)

    def array(self, plan):
        key = str(plan.array)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        value = np.load(plan.array, mmap_mode="r", allow_pickle=False)
        if value.ndim != 3 or value.shape[1] != 11 or value.dtype.kind != "f":
            raise ValueError("Expected pre-RMS N x 11 x T signal")
        if value.nbytes <= self.cache_limit:
            value = np.array(value)
            while self.cache and self.cache_bytes + value.nbytes > self.cache_limit:
                _, old = self.cache.popitem(last=False); self.cache_bytes -= old.nbytes
            self.cache[key] = value; self.cache_bytes += value.nbytes
        return value

    def materialize(self, plan, device, *, budget=20):
        # The future query remains the same B20 future complete runs at ALL budgets.
        acquired = tuple(plan.audit["acquired_prefix_uids"][:budget])
        support_ids = tuple(u for u in acquired if plan.rows[u]["valid"])
        query_ids = tuple(plan.audit["query_uids"])
        x = self.array(plan)
        def tensors(ids):
            ids_rows = [plan.rows[u] for u in ids]
            a = np.array(x[[r["cache_index"] for r in ids_rows]], dtype=np.float64)
            return torch.from_numpy(a).to(device), torch.tensor([r["label"] for r in ids_rows], device=device, dtype=torch.long)
        sx, sy = tensors(support_ids)
        qx, qy = tensors(query_ids)
        support_role = {"meta_train": "meta_train_support", "meta_validation": "meta_validation_support",
                        "internal_holdout": "calibration_prefix"}[self.role]
        support = CalibrationPrefix(sx, sy, support_ids, acquired, self.signal_sha, support_role)
        return SimpleNamespace(support=support, query_x=qx, query_y=qy, query_ids=query_ids,
            run_ids=tuple(str(plan.rows[u].get("physical_run_id", "unknown")) for u in query_ids))

    def draw(self, seed, cursor):
        def index(n, stage):
            limit = 2**256 - (2**256 % n)
            attempt = 0
            while True:
                v = int(digest(["paper3-episode-sampler-v1", seed, cursor, stage, attempt]), 16)
                if v < limit:
                    return v % n
                attempt += 1
        person = self.people[index(len(self.people), "subject")]
        sessions = self.by_person[person]
        return sessions[index(len(sessions), "session:" + person)]
