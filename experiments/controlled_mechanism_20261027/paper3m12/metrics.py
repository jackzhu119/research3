"""Fixed participant-study scoring; fitting never receives future labels."""
from __future__ import annotations
import numpy as np

FLOOR = 1e-12


def probabilities(values, n=None):
    p = np.asarray(values, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 2 or (n is not None and len(p) != n)
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(1), 1, atol=1e-8, rtol=0)):
        raise ValueError('Finite normalized binary probabilities required')
    p = np.clip(p, FLOOR, 1-FLOOR)
    return p / p.sum(1, keepdims=True)


def balanced_support_bce(labels, predicted):
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError('Binary support labels required')
    p = probabilities(predicted, len(y))
    if not len(y) or set(y.tolist()) != {0, 1}:
        return None
    return float(-sum(.5*np.log(p[y == c, c]).mean() for c in (0, 1)))


def session_metrics(labels, predicted, anchor=None):
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or set(y.tolist()) != {0, 1}:
        raise ValueError('Fixed future query requires both binary classes')
    p = probabilities(predicted, len(y)); counts = np.bincount(y, minlength=2)
    weight = .5 / counts[y]; correct = p.argmax(1) == y
    result = dict(BA=float(weight @ correct),
                  balanced_NLL=float(-weight @ np.log(p[np.arange(len(y)), y])),
                  Brier=float(weight @ ((p[:, 1]-y)**2)),
                  n_query=len(y), n_class0=int(counts[0]), n_class1=int(counts[1]))
    if anchor is not None:
        a = probabilities(anchor, len(y)); before = a.argmax(1) == y
        corrected = ~before & correct; introduced = before & ~correct
        anchor_nll = float(-weight @ np.log(a[np.arange(len(y)), y]))
        result.update(corrected_vs_EA=float(weight @ corrected),
                      introduced_vs_EA=float(weight @ introduced),
                      corrected_error_count_vs_EA=int(corrected.sum()),
                      introduced_error_count_vs_EA=int(introduced.sum()),
                      negative_transfer_session_vs_EA=bool(result['BA'] < float(weight @ before)-1e-12),
                      negative_transfer_NLL_session_vs_EA=bool(result['balanced_NLL'] > anchor_nll+1e-12),
                      query_probability_updated_fraction=float(np.any(np.abs(p-a) > 1e-12, axis=1).mean()),
                      query_logit_RMS=float(np.sqrt(np.mean((np.log(p[:, 1]/p[:, 0])
                                                            - np.log(a[:, 1]/a[:, 0]))**2))))
    return result
