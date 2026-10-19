"""Synthetic CPU signal checks only; no real EEG, candidate search, or fit.

These check the approved window's software geometry and bounded preprocessing.
Finite sinusoid diagnostics do not prove that real-trial edge bias disappears,
and introduce no new scientific window-selection or edge-acceptance threshold.
"""
from fractions import Fraction

import numpy as np
import pytest
from scipy.signal import sosfreqz
import torch

from p3work.contracts import CHANNELS, digest
from p3work import preprocess
from p3work.spatial_geometry import joint_trial_rms
from test_contracts import software_v21_contract


def software_signal(sfreq=1000, *, anchor=100, extra=80):
    """Deterministic SOFTWARE_TEST_DOUBLE samples, never an EEG observation."""
    stop = anchor + round(1.5 * sfreq)
    signal = np.random.default_rng(91021).normal(0, 8e-6, (11, stop + extra))
    args = {"channel_names": list(CHANNELS), "sfreq_hz": sfreq,
            "task_sample": anchor, "task_end_sample": stop + min(20, extra),
            "task_end_verified": True, "contract": software_v21_contract(), "native_unit": "V"}
    return signal, args, stop


def test_geometry_is_context192_and_exact_half_open_crop32_to160(monkeypatch):
    # Only the DSP stages are test doubles here, so a nonlinear time marker can
    # distinguish the crop location even after the mandatory temporal demeaning.
    signal, args, _ = software_signal(128, anchor=0, extra=0)
    marker_t = np.arange(192, dtype=np.float64)
    marker = (np.arange(11, dtype=np.float64) - 5)[:, None] * (marker_t ** 2 + 3 * marker_t) * 1e-8
    calls = []
    monkeypatch.setattr(preprocess, "sosfiltfilt", lambda sos, x, axis: x)
    def marker_resample(x, up, down, *, axis, padtype):
        calls.append((x.shape, up, down, axis, padtype))
        return marker.copy()
    monkeypatch.setattr(preprocess, "resample_poly", marker_resample)
    actual, qc = preprocess.prepare_trial(signal, **args)
    expected = marker[:, 32:160].copy()
    expected -= expected.mean(axis=-1, keepdims=True)
    np.testing.assert_array_equal(actual, expected)
    assert calls == [((11, 192), 1, 1, -1, "line")]
    assert actual.shape == (11, 128)
    assert qc["filter_context_ends_at_native_sample"] == 192
    assert qc["signal_contract_sha256"] == digest(args["contract"])


@pytest.mark.parametrize("sfreq", [128, 160, 250, 512, 1000])
def test_real_dsp_uses_bounded_context_and_resamples_to192_before_crop(monkeypatch, sfreq):
    signal, args, stop = software_signal(sfreq)
    original_filter, original_resample = preprocess.sosfiltfilt, preprocess.resample_poly
    stages = []
    def bounded_filter(sos, x, *, axis):
        stages.append(("filter", x.shape))
        assert x.shape == (11, round(1.5 * sfreq)) and axis == -1
        return original_filter(sos, x, axis=axis)
    def bounded_resample(x, up, down, *, axis, padtype):
        assert Fraction(up, down) == Fraction(128, sfreq) and padtype == "line"
        result = original_resample(x, up, down, axis=axis, padtype=padtype)
        stages.append(("resample", result.shape))
        return result
    monkeypatch.setattr(preprocess, "sosfiltfilt", bounded_filter)
    monkeypatch.setattr(preprocess, "resample_poly", bounded_resample)
    output, qc = preprocess.prepare_trial(signal, **args)
    assert stages == [("filter", (11, round(1.5 * sfreq))), ("resample", (11, 192))]
    assert output.shape == (11, 128) and output.dtype == np.float64
    assert qc["filter_context_ends_at_native_sample"] == stop and qc["data_after_context_accessed"] is False


def test_extreme_values_before_context_and_after_context_or_task_do_not_change_output():
    signal, args, stop = software_signal()
    reference, qc = preprocess.prepare_trial(signal, **args)
    poisoned = signal.copy()
    poisoned[:, :args["task_sample"]] = np.nan
    poisoned[:, stop:args["task_end_sample"]] = np.inf  # Still inside native task, outside approved context.
    poisoned[:, args["task_end_sample"]:] = -np.inf
    actual, changed_qc = preprocess.prepare_trial(poisoned, **args)
    np.testing.assert_array_equal(actual, reference)
    assert changed_qc == qc
    poisoned[:, stop - 1] = np.nan  # Last included sample is subject to QC.
    with pytest.raises(ValueError, match="Nonfinite context"):
        preprocess.prepare_trial(poisoned, **args)


def test_only_authorized_channel_context_slice_is_read(monkeypatch):
    signal, args, stop = software_signal()
    trace = []
    class BoundedReadArray(np.ndarray):
        def __getitem__(self, key):
            assert isinstance(key, tuple) and len(key) == 2
            channels, span = key
            assert channels == list(range(11))
            assert isinstance(span, slice) and (span.start, span.stop, span.step) == (args["task_sample"], stop, None)
            trace.append((span.start, span.stop))
            return np.ndarray.__getitem__(self, key).view(np.ndarray)
        def __array_ufunc__(self, *a, **kw):
            pytest.fail("A numerical operation traversed the full native signal")
        def __array_function__(self, *a, **kw):
            pytest.fail("An array operation traversed the full native signal")
    guarded = signal.view(BoundedReadArray)
    original_asarray = preprocess.np.asarray
    def bounded_asarray(value, *a, **kw):
        if value is guarded:
            pytest.fail("The full native signal was converted before bounded slicing")
        return original_asarray(value, *a, **kw)
    monkeypatch.setattr(preprocess.np, "asarray", bounded_asarray)
    output, _ = preprocess.prepare_trial(guarded, **args)
    assert trace == [(args["task_sample"], stop)] and output.shape == (11, 128)


@pytest.mark.parametrize("violation", ["short_recording", "early_task_end", "unverified_task_end"])
def test_missing_real_context_blocks_before_filter_without_padding(monkeypatch, violation):
    signal, args, stop = software_signal()
    if violation == "short_recording": signal = signal[:, :stop - 1]
    elif violation == "early_task_end": args["task_end_sample"] = stop - 1
    else: args["task_end_verified"] = False
    monkeypatch.setattr(preprocess, "sosfiltfilt", lambda *a, **kw: pytest.fail("Invalid context reached DSP"))
    with pytest.raises(ValueError, match="context; padding is forbidden"):
        preprocess.prepare_trial(signal, **args)


def test_si_units_car_temporal_mean_and_rms_are_applied_at_declared_positions():
    signal, args, _ = software_signal()
    original = signal.copy()
    output, qc = preprocess.prepare_trial(signal, **args)
    converted, converted_qc = preprocess.prepare_trial(signal * 1e6, **{**args, "native_unit": "uV"})
    np.testing.assert_allclose(output, converted, rtol=2e-13, atol=2e-18)
    np.testing.assert_array_equal(signal, original)
    assert qc["stored_unit"] == converted_qc["stored_unit"] == "V"
    assert qc["si_conversion_factor"] == 1 and converted_qc["si_conversion_factor"] == 1e-6
    # Floating point linear-filter roundoff only; this is not an EEG QC or
    # window-edge acceptance threshold, and scales with the signal amplitude.
    roundoff = 256 * np.finfo(np.float64).eps * np.max(np.abs(output))
    np.testing.assert_allclose(output.mean(0), 0, atol=roundoff, rtol=0)
    np.testing.assert_allclose(output.mean(1), 0, atol=roundoff, rtol=0)
    measured_rms = float(np.sqrt(np.mean(np.square(output), dtype=np.float64)))
    assert qc["joint_rms"] == measured_rms and measured_rms != pytest.approx(1)
    assert qc["normalization_applied"] is False
    normalized = joint_trial_rms(torch.from_numpy(output))
    assert normalized.dtype == torch.float64
    torch.testing.assert_close(normalized.square().mean().sqrt(), torch.tensor(1.0, dtype=torch.float64), rtol=2e-15, atol=2e-15)


def test_clinic11_selection_ignores_unselected_nonfinite_channel_and_native_order():
    signal, args, _ = software_signal()
    reference, _ = preprocess.prepare_trial(signal, **args)
    reordered = np.vstack((np.full((1, signal.shape[1]), np.nan), signal[::-1]))
    actual, _ = preprocess.prepare_trial(reordered, **{**args, "channel_names": ["AUX"] + list(CHANNELS)[::-1]})
    np.testing.assert_array_equal(actual, reference)


def test_unknown_units_are_never_relabeled_volts_and_zero_energy_gets_no_epsilon():
    signal, args, _ = software_signal()
    with pytest.raises(ValueError, match="Unresolved units"):
        preprocess.prepare_trial(signal, **{**args, "native_unit": "unknown"})
    _, qc = preprocess.prepare_trial(signal, **{**args, "native_unit": "unknown", "common_positive_gain_verified": True})
    assert qc["stored_unit"] == "unresolved" and qc["si_conversion_factor"] is None
    common_signal = np.zeros_like(signal)
    with pytest.raises(ValueError, match="Zero/nonfinite trial energy"):
        preprocess.prepare_trial(common_signal, **args)


@pytest.mark.parametrize("frequency", [8, 10, 30])
def test_synthetic_frequency_and_edge_diagnostics_keep_existing_sos_definition(monkeypatch, frequency):
    # These are software diagnostics, not thresholds for accepting real EEG.
    sfreq, anchor = 1000, 100
    t = (np.arange(anchor + 1500 + 40) - anchor) / sfreq
    spatial = (np.arange(11, dtype=np.float64) - 5)[:, None] * 1e-6
    signal = spatial * np.sin(2 * np.pi * frequency * t)[None, :]
    _, args, _ = software_signal(sfreq, anchor=anchor, extra=40)
    original_butter, original_filter = preprocess.butter, preprocess.sosfiltfilt
    coefficients, designs = [], []
    def declared_butter(order, band, *, btype, fs, output):
        designs.append((order, band, btype, fs, output))
        return original_butter(order, band, btype=btype, fs=fs, output=output)
    def bounded_filter(sos, x, *, axis):
        coefficients.append(sos.copy())
        assert x.shape == (11, 1500)
        return original_filter(sos, x, axis=axis)
    monkeypatch.setattr(preprocess, "butter", declared_butter)
    monkeypatch.setattr(preprocess, "sosfiltfilt", bounded_filter)
    actual, _ = preprocess.prepare_trial(signal, **args)
    assert designs == [(4, (8, 30), "bandpass", 1000, "sos")]
    assert coefficients[0].shape == (4, 6)
    _, response = sosfreqz(coefficients[0], worN=np.asarray([8.0, 10.0, 30.0]), fs=sfreq)
    double_pass_gain = np.abs(response) ** 2
    np.testing.assert_allclose(double_pass_gain[[0, 2]], [.5, .5], rtol=2e-11, atol=2e-12)
    assert 0 < double_pass_gain[1] <= 1  # A transfer-function identity, not a chosen EEG performance threshold.
    ideal = spatial * np.sin(2 * np.pi * frequency * np.arange(32, 160) / 128)[None, :] * double_pass_gain[[8, 10, 30].index(frequency)]
    ideal -= ideal.mean(-1, keepdims=True)
    edge_error = np.linalg.norm((actual - ideal)[:, np.r_[0:16, 112:128]])
    relative_error = np.linalg.norm(actual - ideal) / np.linalg.norm(ideal)
    assert np.isfinite(actual).all() and np.isfinite(edge_error) and np.isfinite(relative_error)
    # No error bound is asserted or used to select another window.
