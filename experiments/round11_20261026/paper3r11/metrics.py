"""Participant-unit reporting primitives; fitting code never imports this module."""
from collections import defaultdict
import numpy as np


def probabilities(value, n=None):
    p = np.asarray(value, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 2 or (n is not None and len(p) != n)
            or not np.isfinite(p).all() or np.any(p < 0)
            or np.any(p > 1) or not np.allclose(p.sum(1), 1., atol=1e-8, rtol=0)):
        raise ValueError('Finite normalized binary probabilities required')
    return p


def session_metrics(labels, predicted, *, log_probabilities=None, anchors=None, event_uids=None):
    y = np.asarray(labels)
    p = probabilities(predicted, len(y))
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('Balanced future-query metrics require both binary classes')
    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=2)
    weights = .5 / counts[y]
    if log_probabilities is None:
        logp = np.log(np.maximum(p, np.finfo(np.float64).tiny))
    else:
        logp = np.asarray(log_probabilities, dtype=np.float64)
        if (logp.shape != p.shape or not np.isfinite(logp).all()
                or not np.allclose(np.exp(logp), p, atol=1e-10, rtol=1e-7)):
            raise ValueError('Log-probabilities must reproduce persisted probabilities')
    prediction = p.argmax(1)
    result = {'n_query': len(y), 'n_class0': int(counts[0]), 'n_class1': int(counts[1]),
              'ba': float(np.sum(weights * (prediction == y))),
              'balanced_nll': float(-np.sum(weights * logp[np.arange(len(y)), y])),
              'balanced_brier': float(np.sum(weights * (p[:, 1] - y) ** 2))}
    if event_uids is not None and (len(event_uids) != len(y) or len(set(event_uids)) != len(y)):
        raise ValueError('Unique fixed query UIDs required for paired flips')
    for name, anchor in (anchors or {}).items():
        other = probabilities(anchor, len(y)).argmax(1)
        corrected = (other != y) & (prediction == y)
        introduced = (other == y) & (prediction != y)
        result.update({f'corrected_errors_vs_{name}': int(corrected.sum()),
                       f'introduced_errors_vs_{name}': int(introduced.sum()),
                       f'balanced_corrected_vs_{name}': float(weights @ corrected),
                       f'balanced_introduced_vs_{name}': float(weights @ introduced),
                       f'negative_transfer_vs_{name}': bool((weights @ (prediction == y)) <
                                                           (weights @ (other == y)) - 1e-12)})
    return result


def participant_summaries(rows):
    """Sessions equal within participant/seed/budget; no query-count weighting."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['participant_id'], row['seed'], row['budget'])].append(row)
    result = []
    excluded = {'budget', 'seed', 'n_query', 'n_class0', 'n_class1'}
    for (person, seed, budget), values in sorted(groups.items()):
        numeric = sorted(set.intersection(*[{k for k, v in x.items()
                                            if isinstance(v, (int, float, bool)) and k not in excluded}
                                           for x in values]))
        result.append({'participant_id': person, 'seed': seed, 'budget': budget,
                       'n_sessions': len(values),
                       **{k: float(np.mean([x[k] for x in values])) for k in numeric}})
    return result
