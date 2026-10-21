"""Native source-extension checks; synthetic BDFs are never scientific fits."""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from p3work.contracts import CHANNELS
from p3work.preprocess import prepare_trial
from p3work.source_extensions import yang2025 as yang


def bdf(path, labels, samples=1000, records=2, values=None, annotations=None):
    n = len(labels)
    fixed = bytearray(b" " * 256)
    fixed[:8] = b"\xffBIOSEMI"
    for start, end, value in ((168, 176, "20.03.19"), (176, 184, "13.53.10"),
                              (184, 192, str(256*(n+1))), (192, 236, "BDF+C"),
                              (236, 244, str(records)), (244, 252, "1"),
                              (252, 256, str(n))):
        fixed[start:end] = value.encode().ljust(end-start)
    columns = [(16, labels), (80, [""]*n), (8, ["?V"]*n),
               (8, ["-375000"]*n), (8, ["375000"]*n),
               (8, ["-8388608"]*n), (8, ["8388607"]*n),
               (80, [""]*n), (8, [str(samples)]*n), (32, [""]*n)]
    header = bytes(fixed) + b"".join(str(v).encode().ljust(width) for width, vals in columns for v in vals)
    with path.open("wb") as f:
        f.write(header)
        for record in range(records):
            for channel in range(n):
                if annotations is not None:
                    f.write(annotations[record].ljust(samples*3, b"\0"))
                else:
                    v = np.zeros(samples, dtype=np.int64) if values is None else values[channel, record*samples:(record+1)*samples]
                    u = v.astype(np.int64) & 0xffffff
                    f.write(np.stack([u & 255, (u >> 8) & 255, (u >> 16) & 255], axis=1).astype(np.uint8).tobytes())
    return path


@pytest.fixture
def profile():
    contract = json.loads((Path(__file__).resolve().parents[1]/"contracts/input_contract.json").read_text())
    return {"schema_version": yang.SCHEMA, "dataset_id": "Yang2025_2C", "source_only": True,
            "context_from_marker_s": [2.25, 3.75], "crop_from_marker_s": [2.5, 3.5],
            "analysis_anchor_from_marker_s": 2.25, "MI_onset_verified": False,
            "prior_analysis_intersection_s": [2, 4], "sfreq_out_hz": 128, "n_times": 128,
            "channels": list(CHANNELS), "input_contract_sha256": yang.INPUT_SHA,
            "native_unit_policy": "common_gain_numeric_unresolved_no_absolute_QC",
            "stored_stage": "pre_rms_car11_temporal_demeaned", "input_contract": contract}


def test_signed_24bit_and_channel_record_boundaries(tmp_path):
    labels = ["unused"] + list(reversed(CHANNELS))
    values = np.arange(len(labels)*2000, dtype=np.int64).reshape(len(labels), 2000)-12000
    values[labels.index("C3"), 998:1003] = [-8388608, -1, 0, 1, 8388607]
    path = bdf(tmp_path/"data.bdf", labels, values=values)
    header = yang._parse_header(path)
    gain = yang._common_gain(header)
    actual = yang._context(path, header, gain["selected_indices"], 995, 1005)
    selected = values[[labels.index(c) for c in CHANNELS], 995:1005]
    scale, offset = Fraction(gain["common_scale_rational"]), Fraction(gain["common_offset_rational"])
    expected = selected.astype(float)*float(scale)+float(offset)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    assert gain["stored_unit"] == "unresolved" and gain["SI_unit_verified"] is False
    assert gain["absolute_amplitude_QC_allowed"] is False


@pytest.mark.parametrize("key,value", [("unit", "uV"), ("physical_max", Fraction(374999)),
                                      ("digital_min", Fraction(-8388607)), ("sfreq", Fraction(999))])
def test_channel_dependent_or_unreviewed_calibration_is_rejected(tmp_path, key, value):
    header = yang._parse_header(bdf(tmp_path/"data.bdf", list(CHANNELS)))
    header["channels"][3][key] = value
    with pytest.raises(yang.YangQualificationError, match="unreviewed_native"):
        yang._common_gain(header)


def test_missing_and_duplicate_channels_fail(tmp_path):
    for labels in [list(CHANNELS[:-1]), list(CHANNELS[:-1])+["C3"]]:
        header = yang._parse_header(bdf(tmp_path/"data.bdf", labels))
        with pytest.raises(yang.YangQualificationError, match="missing_or_duplicate"):
            yang._common_gain(header)


def test_TAL_exact_tokens_original_indices_unknown_and_separator_preserved(tmp_path):
    payloads = [b"+0\x14\x14\0+1.234\x141\x14\0+2.25\x14255\x14\0",
                b"+3.125\x149\x14\0+4.5\x152\x142\x14\0"]
    path = bdf(tmp_path/"evt.bdf", ["BDF Annotations"], samples=100, annotations=payloads)
    rows = yang._read_annotations(path, yang._parse_header(path))
    assert [x["description"] for x in rows] == ["1", "255", "9", "2"]
    assert [x["original_event_index"] for x in rows] == [0, 1, 2, 3]
    assert rows[0]["onset_fraction"] == Fraction(617, 500)
    assert rows[-1]["duration_fraction"] == 2
    assert rows[-1]["native_record_index"] == 1


@pytest.mark.parametrize("body,reason", [(b"+2\x141\x14\0+1\x142\x14\0", "chronological"),
                                         (b"-1\x141\x14\0", "negative_TAL"),
                                         (b"+1bad\0", "malformed_native_TAL")])
def test_noncausal_or_malformed_TAL_fails(tmp_path, body, reason):
    path = bdf(tmp_path/"evt.bdf", ["BDF Annotations"], samples=100, records=1, annotations=[body])
    with pytest.raises(yang.YangQualificationError, match=reason):
        yang._read_annotations(path, yang._parse_header(path))


def test_nearest_native_sample_has_explicit_half_up_and_no_onset_inference():
    assert yang._sample(Fraction("1.234")) == 1234
    assert yang._sample(Fraction("1.2345")) == 1235
    assert yang._sample(Fraction("1.23449")) == 1234


def test_truncated_native_record_fails(tmp_path):
    path = bdf(tmp_path/"data.bdf", list(CHANNELS))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(yang.YangQualificationError, match="file_size_mismatch"):
        yang._parse_header(path)


@pytest.mark.parametrize("field,value", [("source_only", False), ("MI_onset_verified", True),
                                        ("context_from_marker_s", [1.5, 3]),
                                        ("crop_from_marker_s", [2, 4]),
                                        ("stored_stage", "V"), ("n_times", 256)])
def test_source_profile_cannot_silently_reuse_old_window_or_claim_MI_onset(profile, field, value):
    profile[field] = value
    with pytest.raises(yang.YangQualificationError, match="source_profile_changed"):
        yang._profile(profile)


def test_current_native_full_SHA_is_checked_before_manifest_or_parser(tmp_path, profile):
    data = tmp_path/"data.bdf"; evt = tmp_path/"evt.bdf"
    data.write_bytes(b"not-the-frozen-native-signal"); evt.write_bytes(b"not-the-frozen-native-event")
    with pytest.raises(yang.YangQualificationError, match="current_full_native_SHA_mismatch"):
        yang.materialize(data, evt, expected_sha256={"data.bdf": "0"*64, "evt.bdf": "1"*64}, profile=profile)


def test_common_numeric_gain_offset_cancels_only_after_joint_trial_RMS(profile):
    rng = np.random.default_rng(20261019)
    signal = rng.normal(size=(11, 1500))
    outputs = []
    for numeric in [signal, 17*signal+901]:
        x, audit = prepare_trial(numeric, channel_names=list(CHANNELS), sfreq_hz=1000,
                                 task_sample=0, task_end_sample=1500, task_end_verified=True,
                                 contract=profile["input_contract"], native_unit="common_gain_numeric_unresolved",
                                 common_positive_gain_verified=True)
        outputs.append(x/np.sqrt(np.mean(x*x)))
        assert audit["stored_unit"] == "unresolved" and audit["si_conversion_factor"] is None
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-11, rtol=1e-11)
