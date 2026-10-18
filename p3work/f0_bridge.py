"""Receipt-checked baseline EEGNet bridge; this architecture is not the new method.

The compact architecture mirrors the immutable legacy neural.py class. A real
checkpoint is accepted only with a new-protocol source-fit receipt. Historical
synthetic weights or a matching tensor shape alone are insufficient.
"""
from pathlib import Path
import json
import statistics
import torch
from torch import nn
from .contracts import SOURCE_IDS, digest, file_digest, validate_signal_contract
from .adapt import FrozenSourceModel


class CompactEEGNet(nn.Module):
    def __init__(self, channels, times, dropout=0.25):
        super().__init__()
        if channels != 11 or times not in {128, 256}:
            raise ValueError("Frozen clinic11 long/short input required")
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 63), padding=(0, 31), bias=False), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False), nn.BatchNorm2d(16),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.classifier = nn.Linear(16 * (times // 32), 2)

    def forward(self, x):
        return self.classifier(self.features(x[:, None]).flatten(1))


def _local(root, relative):
    path = Path(relative)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or any(part.lower().startswith(".env") or part.lower() in {".aws", ".ssh", "secrets", "credentials", "legacy_baseline_v1"} for part in path.parts)):
        raise ValueError("Source artifact path must stay in the explicit root")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Source artifact symlinks are forbidden")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Source artifact path escaped root")
    return resolved


def load_verified_source(root, receipt_path, input_contract):
    root = Path(root).resolve(strict=True)
    signal_sha = validate_signal_contract(input_contract)
    receipt = json.loads(_local(root, receipt_path).read_text())
    if (receipt.get("status") != "SOURCE_F0_COMPLETE" or receipt.get("synthetic") is not False or
            receipt.get("signal_contract_sha256") != signal_sha or
            set(receipt.get("training_dataset_ids", [])) != SOURCE_IDS or
            receipt.get("source_lodo_selection_verified") is not True):
        raise ValueError("A verified real four-source fit/selection receipt is required")
    if receipt.get("architecture") != "legacy_compact_eegnet_v1_clinic11":
        raise ValueError("Source architecture mismatch")
    seed = receipt.get("pipeline_seed")
    if type(seed) is not int or seed not in range(20261018, 20261023) or receipt.get("seed") != seed:
        raise ValueError("One predeclared complete pipeline seed is required")
    for key in ("checkpoint", "training_uid_manifest", "source_lodo_receipt"):
        artifact = receipt.get(key, {})
        path = _local(root, artifact.get("path", ""))
        if file_digest(path) != artifact.get("sha256"):
            raise ValueError("Source fit artifact integrity mismatch")
    checkpoint = torch.load(_local(root, receipt["checkpoint"]["path"]), map_location="cpu", weights_only=True)
    identity = checkpoint.get("identity", {})
    job = identity.get("job", {})
    if (checkpoint.get("synthetic") is not False or identity.get("synthetic") is not False
            or identity.get("schema_version") != "paper3-source-fit-v2"
            or checkpoint.get("identity_sha256") != digest(identity)
            or job.get("phase") != "final" or job.get("heldout_dataset") is not None or job.get("seed") != seed
            or identity.get("input_contract_sha256") != signal_sha
            or not receipt.get("source_cache_manifest_sha256")
            or identity.get("cache_manifest_sha256") != receipt["source_cache_manifest_sha256"]):
        raise ValueError("Actual same-input final source fit identity required; synthetic/historical weights are forbidden")
    training = json.loads(_local(root, receipt["training_uid_manifest"]["path"]).read_text())
    uids, indices = training.get("event_uids"), job.get("train_indices")
    if (training.get("schema_version") != "paper3-source-training-uids-v2"
            or training.get("role") != "source_pretrain_train" or training.get("synthetic") is not False
            or training.get("signal_contract_sha256") != signal_sha or training.get("pipeline_seed") != seed
            or not isinstance(uids, list) or not uids or any(not isinstance(v, str) or not v for v in uids)
            or len(set(uids)) != len(uids) or not isinstance(indices, list) or len(indices) != len(uids)
            or any(type(v) is not int or v < 0 for v in indices) or len(set(indices)) != len(indices)
            or job.get("train_uids_sha256") != digest(uids)
            or set(job.get("train_dataset_ids", [])) != SOURCE_IDS or job.get("validation_indices") != []):
        raise ValueError("Actual unique same-seed f0 training UID manifest required")
    lodo = json.loads(_local(root, receipt["source_lodo_receipt"]["path"]).read_text())
    epochs = lodo.get("fold_selected_epochs", [])
    if (lodo.get("status") != "SOURCE_LODO_SELECTION_COMPLETE" or lodo.get("seed") != seed or lodo.get("pipeline_seed") != seed
            or lodo.get("input_contract_sha256") != signal_sha or len(epochs) != 4
            or any(type(e) is not int or not 1 <= e <= 80 for e in epochs)
            or len(lodo.get("fold_receipts", [])) != 4):
        raise ValueError("Four real same-seed source LODO selection receipts required")
    cache_artifact = receipt.get("cache_manifest", {})
    if cache_artifact != lodo.get("cache_manifest") or cache_artifact.get("sha256") != receipt["source_cache_manifest_sha256"]:
        raise ValueError("Source final/LODO/cache manifest identities differ")
    cache_path = _local(root, cache_artifact.get("path", ""))
    if file_digest(cache_path) != cache_artifact["sha256"]:
        raise ValueError("Original source cache manifest bytes changed")
    final_epochs = max(1, int(statistics.median(epochs)))
    if (lodo.get("final_epochs") != final_epochs or checkpoint.get("selected_epoch") != final_epochs
            or identity.get("configuration", {}).get("effective_epochs") != final_epochs):
        raise ValueError("Final source length differs from the predeclared per-seed LODO median")
    heldouts, fold_paths, validation_indices = set(), set(), set()
    for ordinal, artifact in enumerate(lodo["fold_receipts"]):
        path = _local(root, artifact.get("path", ""))
        if file_digest(path) != artifact.get("sha256") or path in fold_paths:
            raise ValueError("Source LODO fit receipt integrity mismatch")
        fold_paths.add(path)
        fold = json.loads(path.read_text())
        if fold.get("status") != "SOURCE_FIT_COMPLETE" or fold.get("synthetic") is not False:
            raise ValueError("Source LODO includes a paused, synthetic or incomplete fit")
        manifest_artifact = fold.get("artifact_manifest", {})
        manifest_path = _local(root, manifest_artifact.get("path", ""))
        if file_digest(manifest_path) != manifest_artifact.get("sha256"):
            raise ValueError("Source LODO completed artifact manifest changed")
        manifest = json.loads(manifest_path.read_text())
        evidence = {}
        for name in ("identity.json", "fit_summary.json"):
            expected_path = path.parent / name
            entries = [item for item in manifest.get("files", []) if item.get("path") == expected_path.relative_to(root).as_posix()]
            if len(entries) != 1 or file_digest(_local(root, entries[0]["path"])) != entries[0].get("sha256"):
                raise ValueError("Source LODO identity/summary lacks full-SHA completion evidence")
            evidence[name] = json.loads(expected_path.read_text())
        fold_identity, summary = evidence["identity.json"], evidence["fit_summary.json"]
        fold_job = fold_identity.get("job", {})
        heldout = fold_job.get("heldout_dataset")
        train_indices, val_indices = fold_job.get("train_indices", []), fold_job.get("validation_indices", [])
        if (fold.get("identity_sha256") != digest(fold_identity) or manifest.get("identity_sha256") != digest(fold_identity)
                or fold_identity.get("synthetic") is not False or fold_identity.get("input_contract_sha256") != signal_sha
                or fold_identity.get("cache_manifest_sha256") != receipt["source_cache_manifest_sha256"]
                or fold_job.get("phase") != "lodo" or fold_job.get("seed") != seed or heldout not in SOURCE_IDS
                or heldout in heldouts or set(fold_job.get("train_dataset_ids", [])) != SOURCE_IDS - {heldout}
                or set(train_indices) & set(val_indices) or set(train_indices) | set(val_indices) != set(indices)
                or validation_indices & set(val_indices)
                or summary.get("complete") is not True or summary.get("synthetic") is not False
                or summary.get("selected_epoch") != epochs[ordinal]
                or summary.get("committed_optimizer_updates") != summary.get("epochs_completed", 0) * 64
                or not epochs[ordinal] <= summary.get("epochs_completed", 0) <= 80):
            raise ValueError("Four distinct same-seed source-held-out completed fits required")
        heldouts.add(heldout)
        validation_indices.update(val_indices)
    if heldouts != SOURCE_IDS or validation_indices != set(indices):
        raise ValueError("Source LODO validation partitions do not cover the final source pool exactly")
    if checkpoint.get("channels") != 11 or checkpoint.get("n_times") != input_contract["n_times"]:
        raise ValueError("Source checkpoint input identity mismatch")
    with torch.random.fork_rng(devices=[]):
        model = CompactEEGNet(11, input_contract["n_times"], checkpoint["config"]["dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    encoder = nn.Sequential(model.features, nn.Flatten(start_dim=1))
    source = FrozenSourceModel(encoder, model.classifier, feature_dim=model.classifier.in_features, input_layout="N1CT")
    return source, receipt
