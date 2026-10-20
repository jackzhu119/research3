"""Software numerical regression; no EEG fit, device check or network."""
import numpy as np
import pytest

from p3work.methods.classical import _trials, trace_covariances


def cache_trial():
    x = np.random.default_rng(1).standard_normal((2, 11, 128))
    x -= x.mean(axis=1, keepdims=True)
    x -= x.mean(axis=2, keepdims=True)
    return x.astype(np.float32)


def test_promoted_cache_has_identical_covariances_without_changing_samples():
    native = cache_trial()
    promoted = native.astype(np.float64)
    before = promoted.copy()
    assert np.max(np.abs(promoted.mean(axis=1))) > np.max(np.abs(promoted)) * 1e-10
    np.testing.assert_array_equal(trace_covariances(native), trace_covariances(promoted))
    np.testing.assert_array_equal(promoted, before)
    assert _trials(promoted) is promoted


def test_genuine_float64_retains_the_strict_mean_check():
    x = cache_trial().astype(np.float64)
    x[0, 0, 0] += 1e-11
    assert not np.array_equal(x, x.astype(np.float32).astype(np.float64))
    with pytest.raises(ValueError, match="CAR11"):
        _trials(x)


@pytest.mark.parametrize("axis", [1, 2])
def test_rounded_but_invalid_cache_still_rejected(axis):
    x = cache_trial()
    if axis == 1:
        x[:, 0, :] += np.float32(0.1)
    else:
        x[:, :, 0] += np.float32(0.1)
    with pytest.raises(ValueError, match="CAR11"):
        _trials(x.astype(np.float64))
