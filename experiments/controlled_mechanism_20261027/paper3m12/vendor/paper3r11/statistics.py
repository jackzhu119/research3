"""Prespecified participant-unit paired statistics for R11, NumPy only."""
from __future__ import annotations

import math
import numpy as np

SEED = 20261026
BUDGETS = (4, 8, 12, 16, 20)
PRIMARY_FAMILY = [
    {'name':'operator_vs_'+reference,'method':'operator','reference':reference,'metric':'ba','budget':20}
    for reference in ('support_ea','sgd100','r9_raw_anchored','r9_ea_anchored','profile','operator_erm_tuned')
]
OPERATOR_CONTROLS = ('operator_no_noise','operator_symmetric','operator_psd',
    'operator_diagonal','operator_identity','operator_feature_noise',
    'operator_fisher_noise','operator_shuffled_pairs','operator_erm')
SECONDARY_ABLATION_FAMILY = [
    {'name':'operator_vs_'+reference,'method':'operator','reference':reference,'metric':metric,'budget':20}
    for reference in OPERATOR_CONTROLS for metric in ('ba','balanced_nll')
]



def finite_vector(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)<1 or not np.isfinite(x).all():
        raise ValueError('A finite nonempty participant vector is required')
    return x


def signflip_pvalue(differences, *, seed=SEED, monte_carlo=100000):
    """Two-sided paired sign randomization; participants, never sessions/trials."""
    d=finite_vector(differences)
    observed=abs(float(d.mean()))
    # Zero coordinates merely duplicate every assignment, so retain exact n in
    # the reported population but enumerate only informative sign coordinates.
    nonzero=d[d!=0.]
    if not len(nonzero):
        return {'pvalue':1.,'method':'exact_sign_flip','assignments':1,'seed':seed}
    threshold=observed*len(d)-1e-12*max(1.,float(np.abs(d).sum()))
    count=0
    if len(d)<=20:
        total=1<<len(nonzero)
        bit=np.arange(len(nonzero),dtype=np.uint64)
        for begin in range(0,total,32768):
            numbers=np.arange(begin,min(total,begin+32768),dtype=np.uint64)
            signs=2*((numbers[:,None]>>bit)&1).astype(np.float64)-1
            count+=int(np.count_nonzero(np.abs(signs@nonzero)>=threshold))
        return {'pvalue':count/total,'method':'exact_sign_flip','assignments':total,'seed':seed}
    if type(monte_carlo) is not int or monte_carlo<1:
        raise ValueError('Positive Monte Carlo sign budget required')
    rng=np.random.default_rng(seed)
    for begin in range(0,monte_carlo,8192):
        n=min(8192,monte_carlo-begin)
        signs=2*rng.integers(0,2,size=(n,len(nonzero)))-1
        count+=int(np.count_nonzero(np.abs(signs@nonzero)>=threshold))
    return {'pvalue':(count+1)/(monte_carlo+1),'method':'monte_carlo_sign_flip_plus_one',
            'assignments':monte_carlo,'seed':seed}


def bootstrap_mean_interval(values, *, seed=SEED, repeats=10000, confidence=.95):
    x=finite_vector(values)
    if len(x)<2:
        return None
    if type(repeats) is not int or repeats<1 or not 0<confidence<1:
        raise ValueError('Valid bootstrap budget and confidence level required')
    rng=np.random.default_rng(seed)
    means=np.empty(repeats)
    for begin in range(0,repeats,4096):
        end=min(repeats,begin+4096)
        indices=rng.integers(0,len(x),size=(end-begin,len(x)))
        means[begin:end]=x[indices].mean(1)
    quantile=(1-confidence)/2
    return [float(v) for v in np.quantile(means,[quantile,1-quantile])]


def paired_effect(method, reference, *, seed=SEED, bootstrap=10000, metric='ba'):
    a,b=finite_vector(method),finite_vector(reference)
    if a.shape!=b.shape:
        raise ValueError('Matched participant vectors must have identical shape')
    d=a-b
    result={'n_participants':len(d),'method_mean':float(a.mean()),'reference_mean':float(b.mean()),
            'effect_mean':float(d.mean()),'ci95':bootstrap_mean_interval(d,seed=seed,repeats=bootstrap),
            'difference_sd':float(d.std(ddof=1)) if len(d)>1 else None,
            'difference_se':float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None,
            **signflip_pvalue(d,seed=seed),
            'fraction_method_lower_than_reference':float(np.mean(d < -1e-12)),
            'independent_unit':'participant',
            'signflip_null_assumption':'Independent paired participant differences with sign exchangeability/symmetry under the null; exact enumeration is not a distribution-free mean-zero guarantee',
            'ci_method':'paired participant percentile bootstrap; not inverse of sign-flip test',
            'bootstrap_repeats':bootstrap,'participant_differences':d.tolist()}
    if metric=='ba':
        result['negative_transfer_fraction']=result['fraction_method_lower_than_reference']
        result['negative_transfer_fraction_ci95']=bootstrap_mean_interval((d < -1e-12).astype(float),seed=seed,repeats=bootstrap)
    return result


def holm_fixed(pvalues):
    """Adjust a fixed family; missing hypotheses remain in its multiplicity."""
    if not isinstance(pvalues,(list,tuple)) or not pvalues:
        return []
    for value in pvalues:
        if value is not None and (not math.isfinite(value) or not 0<=value<=1):
            raise ValueError('P values must be in [0,1] or explicitly missing')
    m=len(pvalues)
    order=sorted(range(m),key=lambda i:(1. if pvalues[i] is None else pvalues[i],i))
    adjusted=[None]*m;previous=0.
    for rank,index in enumerate(order):
        raw=1. if pvalues[index] is None else pvalues[index]
        previous=max(previous,min(1.,(m-rank)*raw))
        if pvalues[index] is not None:
            adjusted[index]=previous
    return adjusted


def normalized_aubc(values, budgets=BUDGETS):
    y=finite_vector(values);x=np.asarray(budgets,dtype=np.float64)
    if y.shape!=x.shape or len(x)<2 or np.any(np.diff(x)<=0):
        raise ValueError('A complete increasing acquired-position budget curve is required')
    return float(np.sum(np.diff(x)*(y[:-1]+y[1:])/2)/(x[-1]-x[0]))
