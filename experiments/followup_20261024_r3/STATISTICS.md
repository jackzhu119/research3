# Round 3 statistical reporting

This is an **exploratory internal replication**. The 11 internal participants and
the three fixed pipeline seeds (20261018, 20261019, and 20261020) have already
been used or seen during Round 2. Round 3 must not be described as a fresh
confirmatory experiment, a new blinded test, external validation, or clinical
evidence. Numerical improvements do not remove that limitation.

`paper3r3.report.build_report(plan)` reads only task-local `result.json` and
`evaluations/<partition>/B<budget>/<scenario>/*.json` worker outputs. It writes
`<output>/reports/report.json` and aggregate CSV tables. It never loads a raw
source cache, changes a worker result, submits work, or imports Round 2 code.
Report exceptions are returned as `status: ERROR`; they do not propagate into
training or scheduling.

## Unit of analysis and missing observations

The independent sampling unit is the participant. For a paired comparison,
records are joined by participant and session identity, and exact query trial
identities and labels must agree. Session differences are averaged within each
participant and seed. Those seed averages are then averaged equally over the
available planned seeds for that participant. Finally, participants receive
equal weight. This order matters when seeds have unequal session coverage.
Trials, sessions, and pipeline seeds are not independent sample units.

Each metric uses complete paired session cases and has its own participant
count, excluded session count, and participant seed coverage. No missing value
is imputed as zero. A missing secondary metric does not silently change the BA
denominator. An unavailable participant/seed is omitted from that participant's
descriptive average, with coverage disclosed. Primary inference additionally
requires all planned task seeds and matched candidate/baseline session grids.
The evaluation manifest must exactly match the persisted session files; missing
files, duplicates, invalid probabilities, or inconsistent query signatures are
report errors rather than eligibility exclusions.

## Primary family and inference gate

`plan.comparisons` fixes the entire primary comparison family before results
are read; its length is the Holm family size. The primary coordinate defaults
to holdout, the largest planned budget, and the clean scenario, and can be
specified through `primary_partition`, `primary_budget`, and `primary_scenario`.
Every group in every primary pair must have all planned seeds at that
coordinate, with valid paired data and at least two eligible participants.
Until that gate is met, **every primary p-value and Holm-adjusted p-value is
withheld**. Available effects and descriptive intervals remain visible. A
partially completed family is never substituted for the fixed family, and
there is no significance monitoring over accumulating seeds.

After the entire family is ready, the two-sided paired sign-flip test operates
on the one seed-averaged difference per participant. With 11 participants, all
2,048 sign vectors are enumerated. This test assumes sign-symmetric
participant differences under the null; it does not manufacture randomization
or erase prior reuse of the data. Holm's step-down procedure adjusts all
predeclared primary p-values together. The `positive_evidence` field requires
both a positive BA difference and a Holm-adjusted p-value below 0.05; it remains
an exploratory internal finding.

Intervals use 20,000 deterministic percentile bootstrap replicates, resampling
participants and retaining each participant's already paired, seed-averaged
value. The fixed statistics seed is 20261018. Reported 95% intervals are
individual and unadjusted, conditional on this split, fixed pipeline, and
available seeds. They are not simultaneous Holm intervals, population-wide
clinical uncertainty estimates, or uncertainty across independently sampled
datasets. Fewer than two participants yields no interval or test.

## Descriptive outputs

Budget curves, sensitivity settings, synthetic stress results, transfer
results, probability quality, correction/negative-flip tradeoffs, calibration
timings, and Jacobian/eigenvalue diagnostics remain descriptive. Metadata
`baseline_group` supplies a matched same-coordinate comparator;
`clean_group` supplies a stress comparison against its own clean evaluation.
The predeclared primary pairs are also described at other shared coordinates.
None of these extra tables expands or redefines the primary Holm family.

BA, negative flips, and corrected errors use equal class weighting. Negative
flips and corrections are unconditional class-balanced masses, so BA gain
equals corrected-error mass minus negative-flip mass. The binary Brier score,
negative log likelihood, and ten equal-width confidence-bin ECE also use
class-balanced weights. These probability metrics are measured on saved
probabilities; reporting fits no calibration model. BA and rate differences
are expressed in percentage points, timings in seconds, and probability-score
differences in their native units. The `bce` worker metric is reported as saved;
the independently calculated `nll_balanced` has an explicit definition here.

Calibration times are averaged first across sessions, then seeds, then people.
Nested numeric diagnostics are summarized with the same participant weighting;
saved eigenvalue/singular-value vectors expose minimum, maximum, and sum.
Participant and eligible session counts accompany each diagnostic.

All examples and tests use synthetic prediction records. A passing software
test is evidence for accounting behavior, not evidence of scientific efficacy.
