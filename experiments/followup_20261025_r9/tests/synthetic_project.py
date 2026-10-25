from pathlib import Path
import numpy as np
import torch
from paper3r7.io import put,read,sha,digest
from p3work.f0_bridge import CompactEEGNet
SEEDS=(20261018,20261019,20261020)
def write_project(root):
    """Three seed-specific frozen source receipts and disjoint participant roles."""
    contract = {"n_times": 128, "fixture": "SYNTHETIC_SOFTWARE_TEST_ONLY"}
    put(root / "contracts/input_contract.json", contract)
    contract_sha = digest(contract)
    def desc(path):
        return {"path": str(path.relative_to(root)), "sha256": sha(path)}
    np.save(root / "reference.npy", np.eye(10) / 10)
    for seed in SEEDS:
        torch.manual_seed(seed)
        path = root / f"source-{seed}.pt"
        torch.save({"model": CompactEEGNet(11, 128).state_dict(), "config": {"dropout": .25}}, path)
        receipt = root / f"runs/source_training_v2/f0_receipt_seed-{seed}.json"
        put(receipt, {"status": "SOURCE_F0_COMPLETE", "pipeline_seed": seed,
            "checkpoint": desc(path), "signal_contract_sha256": contract_sha})
        put(root / f"runs/training_campaign_v1/references/seed-{seed}/receipt.json", {
            "status": "SOURCE_REFERENCE_COMPLETE", "source_f0_receipt_sha256": sha(receipt),
            "signal_contract_sha256": contract_sha, "arrays": {"reference": desc(root / "reference.npy")}})
    roles = dict(zip(("meta_train", "meta_validation", "internal_holdout"), ("S1", "S2", "S3")))
    put(root / "split.json", {"partitions": {key: [person] for key, person in roles.items()}, "split_sha256": "b" * 64})
    sessions = []
    rng = np.random.default_rng(18)
    for role, person in roles.items():
        x = rng.normal(size=(100, 11, 128)).astype(np.float32)
        x -= x.mean(1, keepdims=True)
        x -= x.mean(2, keepdims=True)
        np.save(root / f"{person}.npy", x)
        rows = [{"participant_id": person, "session_id": "1", "tasknumber": 1, "label": i % 2,
            "trial_uid": f"{person}-{i}", "event_uid": f"{person}-{i}", "original_trial_index": i,
            "physical_run_id": str(i // 20), "physical_run_complete_verified": True,
            "valid": i != 0, "cache_index": i if i else None} for i in range(100)]
        put(root / f"{person}.json", rows)
        sessions.append({"participant_id": person, "session_id": "1", "partition": role,
            "acquisition_order_verified": True, "X": desc(root / f"{person}.npy"),
            "events": desc(root / f"{person}.json")})
    put(root / "cache.json", {"status": "STIEGER_CACHE_COMPLETE", "participant_split": desc(root / "split.json"),
        "input_contract_sha256": contract_sha, "sessions": sessions})
    put(root / "status/tasks.json", {"W3": {"cache_manifest": desc(root / "cache.json")}})
