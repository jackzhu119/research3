# Method software preparation status

These new modules implement the v2 mathematical kernels, not the legacy v1
algorithm and not a claim of a completed W5 or a scientific pilot.

Implemented: deterministic Q/E, source-training-only hierarchical Rref, float64
matrix exponential, norm projection, traceless orthogonal dictionaries,
full/symmetric/skew generator coordinates, exact joint RMS without epsilon,
frozen source encoder/head with all registered buffers hashed and permanent eval
mode, original two-logit margin increments, 10-step simultaneous support SGD,
missing-class fallback, optional fixed-teacher PCT, B0/B1/B8/B9 fitting kernels,
and a distinct meta_train capability with full-unroll objective and one Stiefel
outer step. B8's dictionary needs independent meta-training; P's dictionary is
not supplied or learned by importing or testing these modules.

Receipt-checked source architecture/loading bridge is implemented in ../f0_bridge.py; no real source weights exist.

Not implemented: source scientific model fitting,
the authorized real episode loader/sampler, a resumable 1000-update training
runner with validation and 9-grid selection, B2/B3/B4/B5/B6/B7/B10/B11, drift-PCA,
checkpoint/resume equivalence, final internal/external evaluation, statistics,
and final freeze production. Source/training event identity and actual temporal
run boundaries must be proven by upstream audited manifests; in-memory types
and hash references alone do not establish those facts.

CPU synthetic tests are software/numerical acceptance only. They do not measure
EEG performance, satisfy W1 native-schema requirements, or complete any pilot.
GPU execution remains blocked until the user later starts the existing machine.
