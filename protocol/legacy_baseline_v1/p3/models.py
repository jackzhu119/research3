"""Inspectable fixed classical baselines with non-pickle saved numerical state.

The tangent baseline uses a log-Euclidean training reference followed by a
whitened matrix logarithm. It is named explicitly and is not an exact replica
of the conventional affine-invariant-mean pyRiemann benchmark pipeline.
"""
import numpy as np
from scipy.linalg import eigh, helmert
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .core import standardized_trials

def covariance(X):
    basis=helmert(X.shape[1],full=False)
    Z=np.einsum("dc,nct->ndt",basis,standardized_trials(X).astype(np.float64))
    cov=Z@Z.transpose(0,2,1)/(Z.shape[-1]-1)
    tr=np.trace(cov,axis1=1,axis2=2)
    if (tr<=0).any(): raise ValueError("Zero covariance")
    cov=cov/tr[:,None,None]
    cov=0.9*cov+0.1*np.eye(Z.shape[1])[None]/Z.shape[1]
    return cov,basis

def matrix_function(A, func):
    w,v=np.linalg.eigh(A)
    if not np.isfinite(w).all(): raise ValueError("Nonfinite matrix spectrum")
    if func in (np.log,np.sqrt) and (w<=0).any(): raise ValueError("Non-SPD covariance")
    return (v*func(w)[...,None,:])@np.swapaxes(v,-1,-2)

def tangent_features(cov, inverse_root):
    white=inverse_root@cov@inverse_root
    logs=matrix_function(white,np.log)
    d=logs.shape[-1]; i,j=np.triu_indices(d)
    return logs[:,i,j]*np.where(i==j,1.0,np.sqrt(2.0))

def balanced_weights(rows,y):
    """Dataset, then person, then class equally weighted; no query statistics."""
    from collections import Counter
    ds=Counter(r["dataset"] for r in rows)
    people={d:{str(r["subject"]) for r in rows if r["dataset"]==d} for d in ds}
    cells=Counter((r["dataset"],str(r["subject"]),int(c)) for r,c in zip(rows,y))
    classes={p:{c for d,s,c in cells if (d,s)==p} for p in {(d,s) for d,s,c in cells}}
    weights=[]
    for row,label in zip(rows,y):
        dataset=row["dataset"];person=(dataset,str(row["subject"]))
        denominator=len(ds)*len(people[dataset])*len(classes[person])*cells[(*person,int(label))]
        weights.append(1.0/denominator)
    w=np.array(weights)
    return w/w.sum()

def csp_features(X,basis,filters):
    Z=np.einsum("dc,nct->ndt",basis,standardized_trials(X).astype(np.float64))
    filtered=np.einsum("kd,ndt->nkt",filters,Z)
    return np.log(np.maximum(np.mean(filtered*filtered,axis=-1),np.finfo(float).tiny))

def fit_classical(kind,X,y,rows):
    if set(y)!={0,1}: raise ValueError("Training requires both classes")
    cov,basis=covariance(X); w=balanced_weights(rows,y)
    if kind=="csp_lda":
        means=[np.average(cov[y==c],axis=0,weights=w[y==c]) for c in (0,1)]
        _,vectors=eigh(means[1],means[0]+means[1])
        components=min(4,basis.shape[0]); low=components//2
        selected=list(range(low))+list(range(basis.shape[0]-(components-low),basis.shape[0]))
        filters=vectors[:,selected].T
        features=csp_features(X,basis,filters)
        model=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto",priors=[0.5,0.5]).fit(features,y)
        # CSP covariance estimation is balanced. sklearn LDA fit itself is not
        # sample-weighted; this distinction remains explicit in the config.
        return {"basis":basis,"filters":filters,"coef":model.coef_,"intercept":model.intercept_}
    if kind=="ts_lr":
        meanlog=np.average(matrix_function(cov,np.log),axis=0,weights=w)
        reference=matrix_function(meanlog,np.exp)
        eig,v=np.linalg.eigh(reference)
        root=(v*(eig**-0.5))@v.T
        features=tangent_features(cov,root)
        scaler=StandardScaler().fit(features,sample_weight=w)
        model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,random_state=0).fit(
            scaler.transform(features),y,sample_weight=w*len(y))
        if np.any(model.n_iter_>=2000): raise RuntimeError("Logistic regression failed to converge")
        return {"basis":basis,"inverse_root":root,"scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_,
                "coef":model.coef_,"intercept":model.intercept_}
    raise ValueError("Unknown classical model")

def predict_classical(kind,state,X):
    if kind=="csp_lda": features=csp_features(X,state["basis"],state["filters"])
    elif kind=="ts_lr":
        cov,_=covariance(X)
        features=(tangent_features(cov,state["inverse_root"])-state["scaler_mean"])/state["scaler_scale"]
    else: raise ValueError("Unknown model")
    score=(features@state["coef"].T+state["intercept"]).ravel()
    p=expit(score)
    # [0, log-odds] is an exact binary-logit representation for these models.
    return np.stack([1-p,p],axis=1),np.stack([np.zeros_like(score),score],axis=1)
