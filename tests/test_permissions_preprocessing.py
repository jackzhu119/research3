import numpy as np
import pytest
from p3work.contracts import CHANNELS, permit_preparation
from p3work.preprocess import prepare_trial


def contract():
    return {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
            "selected_profile": "long", "context_s": [0, 3], "crop_s": [0.5, 2.5], "n_times": 256,
            "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": "a" * 64,
                "raw_metadata_manifest_sha256": "b" * 64,
                "long": {"participant_mean_coverage": .9, "participant_mean_abs_class_exclusion_gap": .02},
                "short": {"participant_mean_coverage": .95, "participant_mean_abs_class_exclusion_gap": .02}}}


def test_locked_partitions_and_unfrozen_window_fail_closed():
    for dataset, role in [("HSStroke2026_v6", "source_schema"), ("Wang2026", "source_train"),
                          ("Stieger2021", "internal_test"), ("Stieger2021", "meta_validation")]:
        with pytest.raises(PermissionError): permit_preparation(dataset, role)
    permit_preparation("BNCI2014_001", "source_schema")
    c = contract(); c["status"] = "pending"
    with pytest.raises(ValueError, match="not frozen"):
        prepare_trial(np.ones((11, 1000)), channel_names=list(CHANNELS), sfreq_hz=250,
                      task_sample=0, task_end_sample=1000, task_end_verified=True, contract=c, native_unit="V")


def test_future_samples_and_common_reference_cannot_change_input():
    rng = np.random.default_rng(42); x = rng.normal(size=(11, 1200))
    args = dict(channel_names=list(CHANNELS), sfreq_hz=250, task_sample=100, task_end_sample=850, task_end_verified=True,
                contract=contract(), native_unit="uV")
    original, _ = prepare_trial(x, **args)
    changed = x.copy(); changed[:, 850:] = 1e8
    future_changed, _ = prepare_trial(changed, **args)
    offset, _ = prepare_trial(x + rng.normal(size=(1, 1200)), **args)
    np.testing.assert_array_equal(original, future_changed)
    np.testing.assert_allclose(original, offset, atol=1e-18, rtol=1e-10)


def test_no_padding_or_unverified_channel_interpolation():
    args = dict(channel_names=list(CHANNELS), sfreq_hz=250, task_sample=0, task_end_sample=750, task_end_verified=True,
                contract=contract(), native_unit="V")
    with pytest.raises(ValueError, match="padding"):
        prepare_trial(np.ones((11, 700)), **args)
    with pytest.raises(ValueError, match="Missing measured"):
        prepare_trial(np.ones((11, 1000)), **{**args, "channel_names": ["FP1"] + list(CHANNELS)[1:]})


def test_early_task_termination_and_missing_long_audit_cannot_pass():
    from p3work.contracts import validate_signal_contract
    args = dict(channel_names=list(CHANNELS), sfreq_hz=250, task_sample=0,
                task_end_sample=700, task_end_verified=True, contract=contract(), native_unit="V")
    with pytest.raises(ValueError, match="padding"):
        prepare_trial(np.random.default_rng(3).normal(size=(11, 2000)), **args)
    c = contract(); c["window_audit_receipt"]["long"] = {}
    with pytest.raises(ValueError, match="Both candidate"):
        validate_signal_contract(c)
