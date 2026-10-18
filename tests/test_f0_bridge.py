import json
import pytest
import torch
from p3work.f0_bridge import CompactEEGNet, load_verified_source
from p3work.adapt import FrozenSourceModel
from p3work.spatial_geometry import joint_trial_rms
from torch import nn


def test_compact_baseline_bridge_keeps_original_double_logit_path():
    torch.manual_seed(4)
    model = CompactEEGNet(11, 128).eval()
    source = FrozenSourceModel(nn.Sequential(model.features, nn.Flatten(1)), model.classifier,
                               feature_dim=model.classifier.in_features, input_layout="N1CT")
    x = torch.randn(3, 11, 128)
    expected = model(joint_trial_rms(x).float())
    torch.testing.assert_close(source.source_logits(x), expected, rtol=0, atol=0)
    source.assert_unchanged()


def test_unfrozen_or_historical_synthetic_checkpoint_never_loads(tmp_path):
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps({"status": "synthetic_complete", "synthetic": True}))
    with pytest.raises(ValueError, match="not frozen"):
        load_verified_source(tmp_path, "receipt.json", {"status": "pending"})
