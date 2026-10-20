# Method software preparation status

This is a software inventory, not W5 completion or evidence of a scientific fit.
The unchanged `protocol/legacy_baseline_v1` is historical source material.

## Implemented software

- Geometry and adaptation: fixed Helmert Q, orthonormal traceless E, hierarchical
  source-training-only Rref, float64 matrix exponential, coefficient projection,
  full/symmetric/skew coordinates, exact joint RMS without epsilon, original
  two-logit margin increments, simultaneous ten-step support SGD and fixed-teacher
  PCT. Source weights, BN buffers and eval mode are checked for mutation.
- `method_registry.py`: B0–B11, P and P+PCT have separate fitting paths and
  registered grids. Learned additive and meta-LoRA require their own learned
  artifacts. Random generators cannot stand in for learned parameters. B4/B5
  require their separately verified source bank, not a legacy model object.
- `episodes.py`: W1/W3 and manifest-bound real episode interface, original paid
  prefix/run/query rules, deterministic participant-then-session sampling,
  split permissions and explicit label capabilities. No missing participant is
  silently removed from the validation denominator. The support-only interface
  does not materialize query arrays or label capabilities. Catalogs, row maps
  and plans are immutable and recheck their issued authority before access.
- `fit_artifacts.py`: stores issued parameters for all fourteen methods using
  tensors/JSON and complete SHA verification. Loading requires the same source,
  reference, method, seed, role and paid prefix. It preserves numerical layout
  and never refits or restores RNG merely to recover predictions.
- `source_training.py` and `classical_training.py`: resumable, budgeted source
  entry points with complete plans and artifact hashes. The source neural plan
  has 15 development or 25 confirmation fits; the historical core math plan has
  ten fits. The new B4/B5 source bank adds six distinct fits. No such fit has run.
- `f0_bridge.py` and `source_reference_io.py`: verify real source training,
  LODO, seed, cache and UID evidence before loading f0 or constructing Rref.
- `meta_training.py`, `training_state.py`, `recovery_budget.py`: engineering and
  scientific jobs remain distinct; full unroll, four episodes per outer update,
  validation every 50 updates, nine-grid screening, participant-equal selection,
  persistent RNG/optimizer/sampler state and conservative hard-interrupt budgets.
  Confirmation requires screening evidence; it cannot invent a selected model.
  `meta_engineering.py` verifies the actual 100-update checkpoint, both scheduled
  validations, full timing/memory ledger and same-device runtime before search.
- `development_evaluation.py`: fixed control-search/confirmation plans, bounded
  calibration-profile and production evaluation entry points. Candidate
  parameters are committed before predictions; recovery reuses a committed
  candidate and counts incomplete dispatched attempts. Formal execution still
  requires real data/source/learned artifacts, a complete measured same-device
  profile and explicit budgets after user boot. No real W6 completion or B*
  selection has occurred.
- `prediction_store.py`: immutable logits/probability/class transactions,
  event identity, paid-label and update/time costs, and full SHA verification
  before internal holdout label release.
- `diagnostics.py`: supplied-matrix symmetry/skew energy, subspace principal
  angles, coefficient/radius diagnostics, exact parameter counts and separate
  whitened versus sensor-space condition numbers. No real learned matrix has
  been analyzed.
- `drift_pca.py`: explicit-configuration mathematical kernels and a read-only
  real-input gate. The protocol does not fully specify its covariance coordinate,
  shrinkage, session-trial scope or weighting recipe. Those fields stay null;
  the formal PCA estimator remains blocked pending a frozen implementation
  amendment. A software example is not an estimated source drift basis.
- `statistics.py`: pure participant-level aggregation, paired bootstrap,
  class-specific correction/error identities and explicitly specified testing
  families. `tools/analyze_statistics.py` remains read-only and blocks real W9
  analysis until W7 and a verified external prediction transaction exist.
- `tools/prepare_stieger_cache.py`: frozen-window, full-registry CPU cache plan,
  serial resource bounds, interruption recovery and complete R2 GET verification.
  The real W1 gate and all598-session CPU cache/independent quality checks have passed; this has not run a scientific fit or GPU pilot.
- `window_acceptance.py` and `source_cache_acceptance.py`: independent read-only
  acceptance of all 396 meta-train session metadata and all 505 source objects.
  Window thresholds use integer counts and exact rational arithmetic. A
  disagreement with the frozen decision blocks downstream work. Source checks
  account for every native LR trial as valid or explicitly excluded, verify
  actual float32 cache bytes, and bind complete R2 GET proofs. Status strings
  cannot substitute for these artifacts.

## Acceptance scope and remaining gates

Tests use explicitly synthetic software fixtures. They cover numerical identity,
full-unroll gradients, reference direction, projection, classical recipes,
permissions, prediction transactions and recovery. CPU resume fixtures do not
establish GPU numerical equivalence or EEG performance.

There are zero real scientific fits, learned G/U0 artifacts or performance
results. W1 must freeze the actual roster and common window; W2/W3 must supply
verified real caches. The user must later boot the existing GPU machine before
the controlled source engineering pilot. Source fitting, Rref estimation, meta
engineering timing, scientific grids, strong-control selection and internal
holdout evaluation each retain their own gates.

The final clinical evaluator, W7 freeze production and external execution remain
separate future technical gates. Sealed external
payloads remain untouched, and `protocol_freeze_sha256` remains null. No test,
import, status flag or input-contract hash can replace those technical gates.

The final current software regression passed3641 tests in557.99 seconds, with all197 software inputs rehashed after the real exit. See the current software acceptance referenced by `contracts/data_plan_final_validation_20261020.json`; original117/189 inventories remain historical evidence. W5 scientific validation remains pending actual user-started GPU execution.
