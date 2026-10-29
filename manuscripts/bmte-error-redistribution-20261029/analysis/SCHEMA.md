# Analysis interface

`participant_metrics.csv` retains study, cohort, analysis_role, participant, native_method, canonical method, reference, budget and budget_kind. C/I/D/G and BA fields are fractions; `_pp` fields multiply by100. Kumar budgets count valid support labels; `full` is19/20, while neural budgets count acquired positions. `budget_order` orders within a study and is not a cross-study equivalence mapping.

`cohort_summary.csv` uses equal-participant means after native session/seed averaging, and `*_ci95_low/high` are new pointwise participant-bootstrap intervals. Cancellation terms and dimensionless ratios carry the same suffixes. Ratio intervals use defined draws only, with counts recorded. `ba_percent` is a percentage; `G_pp` is a percentage-point difference.

`component_summary.csv` preserves three matched conventional comparisons, all budgets and development/main scopes. `neural_paired_contrasts.csv` contains cold-start and fitting-anchor contrasts, including paired BA/NLL differences.

Session tables cover conventional predictions only. Neural sealed participant summaries do not support reconstructing session rows. Source-extension counts are separate. All original inferential families are preserved in evidence; the integrated survey adds no p-values.
