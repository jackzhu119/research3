"""Bounded native-trial preprocessing. Disabled until the W1 window is frozen."""
from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from .contracts import CHANNELS, validate_signal_contract


def prepare_trial(signal, *, channel_names, sfreq_hz, task_sample, task_end_sample,
                  task_end_verified, contract,
                  native_unit, common_positive_gain_verified=False):
    """Return pre-RMS CAR11 trial and QC identity, never statistics from other trials."""
    contract_sha = validate_signal_contract(contract)
    # Accept an ndarray/memmap without converting or traversing a full recording.
    # Only the authorized context slice below is read or converted.
    if not isinstance(signal, np.ndarray):
        raise TypeError("Native reader must supply an ndarray or memmap")
    x = signal
    if x.ndim != 2 or len(channel_names) != x.shape[0] or len(set(channel_names)) != len(channel_names):
        raise ValueError("Invalid channels-by-time native signal")
    missing = set(CHANNELS) - set(channel_names)
    if missing:
        raise ValueError("Missing measured clinic11 channels")
    if not np.isfinite(sfreq_hz) or sfreq_hz <= 60 or not isinstance(task_sample, (int, np.integer)):
        raise ValueError("Invalid native sampling rate/task anchor")
    factors = {"V": 1.0, "uV": 1e-6, "µV": 1e-6, "mV": 1e-3}
    if native_unit not in factors and not common_positive_gain_verified:
        raise ValueError("Unresolved units without verified common positive gain")
    factor = factors.get(native_unit, 1.0)
    stop = int(task_sample) + int(round(contract["context_s"][1] * sfreq_hz))
    if (task_end_verified is not True or not isinstance(task_end_sample, (int, np.integer)) or
            task_sample < 0 or stop > x.shape[1] or stop > task_end_sample):
        raise ValueError("Insufficient original-trial context; padding is forbidden")
    selected = np.asarray(x[[channel_names.index(c) for c in CHANNELS], task_sample:stop],
                          dtype=np.float64) * factor
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite context")
    selected -= selected.mean(axis=0, keepdims=True)
    sos = butter(4, (8, 30), btype="bandpass", fs=sfreq_hz, output="sos")
    filtered = sosfiltfilt(sos, selected, axis=-1)
    ratio = Fraction(128 / float(sfreq_hz)).limit_denominator(100000)
    if abs(float(ratio) * sfreq_hz - 128) > 1e-9:
        raise ValueError("Native rate cannot be represented exactly for resampling")
    resampled = resample_poly(filtered, ratio.numerator, ratio.denominator,
                              axis=-1, padtype="line")
    start, end = (int(round(value * 128)) for value in contract["crop_s"])
    crop = resampled[:, start:end].copy()
    if crop.shape != (11, contract["n_times"]):
        raise ValueError("Crop shape mismatch")
    crop -= crop.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(crop), dtype=np.float64))
    if not np.isfinite(rms) or rms == 0:
        raise ValueError("Zero/nonfinite trial energy")
    return crop, {"signal_contract_sha256": contract_sha, "native_unit": native_unit,
                  "stored_unit": "V" if native_unit in factors else "unresolved",
                  "si_conversion_factor": factor if native_unit in factors else None,
                  "reference": "CAR_exact_clinic11", "joint_rms": float(rms),
                  "normalization_applied": False, "filter_context_ends_at_native_sample": stop,
                  "data_after_context_accessed": False}
