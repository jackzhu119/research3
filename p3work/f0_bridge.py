"""Receipt-checked baseline EEGNet bridge; this architecture is not the new method.

The compact architecture mirrors the immutable legacy neural.py class. A real
checkpoint is accepted only with a new-protocol source-fit receipt. Historical
synthetic weights or a matching tensor shape alone are insufficient.
"""
from pathlib import Path
import json
import torch
from torch import nn
from .contracts import SOURCE_IDS, file_digest, validate_signal_contract
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
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Source artifact path must stay in the explicit root")
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
    for key in ("checkpoint", "training_uid_manifest", "source_lodo_receipt"):
        artifact = receipt.get(key, {})
        path = _local(root, artifact.get("path", ""))
        if file_digest(path) != artifact.get("sha256"):
            raise ValueError("Source fit artifact integrity mismatch")
    checkpoint = torch.load(_local(root, receipt["checkpoint"]["path"]), map_location="cpu", weights_only=True)
    if checkpoint.get("channels") != 11 or checkpoint.get("n_times") != input_contract["n_times"]:
        raise ValueError("Source checkpoint input identity mismatch")
    model = CompactEEGNet(11, input_contract["n_times"], checkpoint["config"]["dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    encoder = nn.Sequential(model.features, nn.Flatten(start_dim=1))
    source = FrozenSourceModel(encoder, model.classifier, feature_dim=model.classifier.in_features, input_layout="N1CT")
    return source, receipt
