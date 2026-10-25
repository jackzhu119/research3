# Round 8 transport implementation and verification

`paper3r8/transport.py` operates only on an acquired support and a frozen source
reference. It does not accept target query observations during fitting. The
query argument belongs only to the separate, batch-independent prediction API.

## Input convention and controls

The target covariance is the mean of per-trial unit-trace CAR covariances,
matching the preceding `support_EA` control. The existing frozen reference R is
used without re-estimation. C_eta=(1-eta)C+eta I/d, with d=10. `kind='ea'` fixes
alpha=1, eta=.05 and reproduces the previous full recoloring control on CAR
inputs. The residual identity extension retains tiny numerical non-CAR residuals.

`kind='identity'` returns the frozen source exactly. Empty/nonfinite/zero-energy
support gives an explicitly recorded source fallback. A pooled covariance still
uses one-class support. Class-balanced covariance falls back to pooled weights
when a class is missing, with that fact recorded.

## Maps

- Recolor: T=R^(1/2) C_eta^(-1/2).
- Bures: T=C_eta^(-1/2)(C_eta^(1/2) R C_eta^(1/2))^(1/2) C_eta^(-1/2).
- Source-metric Bures: T=R^(1/2)(R^(-1/2) C_eta R^(-1/2))^(-1/2)R^(-1/2).
- Fiber: T=R^(1/2) exp(beta K) C_eta^(-1/2), where K is skew-symmetric.

Each full-strength map matches the **regularized pre-RMS covariance** to R.
This does not assert that the unregularized support covariance, the future
covariance, or the covariance after per-trial RMS equals R. Partial strength
uses I+alpha(T-I) and generally no longer matches R exactly.

Recolor, Bures and source-metric Bures coincide for commuting C_eta and R.
Standard Bures minimizes the squared Euclidean covariance-weighted displacement;
the source-metric version minimizes displacement weighted by R^(-1). These are
known optimal-transport results, not claimed new theorems.

## Fiber direction and bounded step

The worker must provide a fixed training-only, source-whitened class reference
Ds as `meta_parameters['class_reference']`. This module neither estimates Ds
from evaluation data nor silently substitutes it. The per-session class contrast
is Dt=C_eta^(-1/2)(C_1-C_0)C_eta^(-1/2). For a nonzero commutator A=[Ds,Dt],
K=A/||A||F. The alignment F(t)=tr(Ds exp(tK) Dt exp(-tK)) obeys
F'(0)=||A||F. A vanishing commutator selects zero rotation and is recorded as
locally unidentified; no global optimum is asserted.

For L=4||Ds||F ||Dt||F ||K||op^2, |F''(t)|<=L. With
beta=min(beta_max,F'(0)/L), F(beta)-F(0)>=beta F'(0)-L beta^2/2>=beta F'(0)/2.
The implementation also handles tiny, partially normalized commutators by
setting the direction to zero below Frobenius norm 1e-12 and using the actual
directional slope <A,K> in this expression. The exact-arithmetic
bound is for class-contrast alignment only; it is neither a prediction-accuracy
bound nor a future-population guarantee. Fixed-beta experiments need not increase
alignment. Reverse/random controls use the same beta determined from the aligned
direction; the bound is not assigned to those controls.

## Differentiation

Meta-parameters are global alpha/eta logits or a 2-by-4 affine map of three
bounded label-free support descriptors plus intercept. Learned alpha=sigmoid(a)
and eta=.001+.999 sigmoid(e). The three descriptors are covariance shift,
within-prefix covariance dispersion and n/(n+20), not calibrated uncertainty.
Only first-order outer gradients are required.

SPD square-root and inverse-square-root custom backward use spectral divided
differences with finite limits at repeated eigenvalues. The square-root backward
solves the Sylvester equation S X+X S=dA. It never divides by an eigenvalue gap.
Double backward is deliberately unsupported. Diagnostics detach scalars without
detaching the prediction path to learned parameters.

## Local verification

46 numerical tests pass on both the default PyTorch 2.8 CPU environment and the
school-version-compatible Python 3.12 / PyTorch 2.5.1+cu121 CPU environment
(`/tmp/paper3-r8-compat/bin/python`). They cover finite
differences including repeated eigenvalues; finite differences through actual
prediction; exact covariance matching; the stated displacement objectives;
commuting/identity nulls; exact previous-EA agreement; source immutability;
query-batch independence; empty/one-class/invalid support behavior; fiber local
direction, bounded-step inequality, reverse/random controls and commuting nulls.
These software tests do not establish school GPU execution or scientific benefit.
