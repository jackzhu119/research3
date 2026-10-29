# R12/R13 validation evidence audit

Audit date: 29 October 2026. Read-only source snapshot: `jackzhu119/research3`, `paper3-results`, commit `a16d06760ae77699a809c9ba36d002d684560c82`. No model was fitted or scientific job started. Large model archives were not downloaded. The 57,282-byte R13 **plan-and-reports** archive was downloaded to restore and SHA-verify the original participant-level CSV; no checkpoints or raw EEG were involved.

## Scientific conclusion for manuscript planning

R12 and R13 are controlled, retrospective **development-cohort experiments**, not external or independent confirmation. Their most useful role is to bound a specific explanation of adaptation: reference-measured spatial sensitivity and persistence of a synthetic nuisance. Neither provides evidence that a new risk gate or a spatial method generally outperforms the anchor. These experiments do not validate a claim of clinical efficacy, new-participant generalization, a natural temporal half-life, or prospective calibration-cost savings.

R12 used a declared stopping rule and returned `STOP`, because support-fit improvement did **not** coincide with deterioration of future NLL. All six paired primary comparisons had Holm-adjusted p > .05. R13 completed its finite pilot matrix, but its primary persistence interaction was close to zero and its 95% interval crossed zero. A transparent manuscript can make a focused positive contribution around a separately supported method/mechanism while using these outcomes to define its scope. Calling these new independent validations would be incorrect.

## R12: registered controlled spatial mechanism study

### What was actually observed

- Source recipe: `b04a1db8d593f2590cb57649ae0dc55f3ac751bc2b36c08cd9782b6d62748343`.
- Eight historical Stieger pilot participants: S1, S2, S3, S6, S14, S26, S50, S53. They were selected within the historical source/meta-training cohort, not recruited independently after method development.
- Three fixed source seeds: 20261018, 20261019, 20261020. Sessions are equally weighted within source seed, then seeds within participant. The inferential sample is **n = 8**, not 24 fits, all session outputs, reference events, or parameter draws.
- Fixed acquired budgets B = 4, 8, 12, 16, 20; common future query follows acquired position 20; no QC replenishment. The primary comparison is B20, 100 optimization steps.
- Frozen source EEGNet, frozen batch normalization/dropout and frozen primary classification head. A four-dimensional spatial generator dictionary is inherited from R7; the finite map is `exp(M)` in frozen reference-whitened coordinates.
- Reference bank: 32 other source participants × two fixed legal events = 64 events; reference-logit RMS is used to match perturbation dose. Five random-subspace draws are metric-averaged, not treated as independent samples and not selected for best performance.
- Primary family: top-vs-random subspace, weak-vs-random deletion, top-vs-scalar shrinkage and top-vs-EA; six two-sided participant-level sign-flip comparisons with Holm adjustment. Reported CIs are pointwise percentile bootstrap intervals, **not simultaneous** and not inversion of the exact randomization test.

### Exact primary table

| ID | Candidate − baseline | Metric | Mean difference | Pointwise 95% CI | Raw p | Holm p |
|---|---|---|---:|---|---:|---:|
| P1 | top_q − random_q | BA, percentage points | +0.0408599 | [0.0107289, 0.0764873] | 0.0390625 | 0.1953125 |
| P2 | top_q − random_q | balanced NLL | −0.0000418632 | [−0.0000739282, −0.0000098885] | 0.0546875 | 0.21875 |
| P3 | drop_weak − drop_random | BA, percentage points | +0.0402539 | [−0.0104112, 0.0865508] | 0.171875 | 0.515625 |
| P4 | drop_weak − drop_random | balanced NLL | −0.0000596952 | [−0.0000979089, −0.0000217308] | 0.0234375 | 0.140625 |
| P5 | top_q − scalar_shrink_full | BA, percentage points | +0.0088354 | [−0.0333760, 0.0547193] | 0.734375 | 0.734375 |
| P6 | top_q − support_EA | BA, percentage points | +0.0527848 | [−0.0151343, 0.1274394] | 0.2265625 | 0.515625 |

All rows are `READY`, complete six-comparison family, n = 8. Lower NLL is favorable, whereas higher BA is favorable. Do not report an uncorrected bootstrap interval as family-adjusted positive evidence.

### Why the later cohorts were skipped

The actual `pilot-decision.json` reports `decision: STOP`, `insufficient_evidence: false`, and no significance claim. The support-fit/future-harm criterion is false: mean support BCE change (full SGD − EA) = −0.007759094655451403, while mean future balanced NLL change = −0.0011800877791132147. Both improve; the intended support-overfit/future-harm conjunction was not observed. Other point-effect criteria were directionally satisfied, but this did not override the stopping rule.

The plan had 217 tasks. The 78 completed tasks comprise three calibrations, one tau freeze, 24 pilot paired blocks, one pilot decision, 24 head-capacity blocks, 24 synthetic-shift blocks and one final report. The remaining 139 tasks were predeclared dependent stages: 138 later cohort fits and one development freeze. They do not constitute missing-upload evidence, but they also do not provide later-cohort results. Current report contains no paired holdout, meta-validation, Pan or Weibo observations; their participant CSV artifacts are one-byte empty files.

The source plan explicitly states: `independent_confirmation.status = NOT_AVAILABLE`; it records planned N90 evaluable/N110 initial participants as **planning**, not completed recruitment.

### Source and restoration evidence

- [Current original R12 report](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-12/studies/b04a1db8d593f259/paper/reports/report.json/part-000000.bin).
- [Original pilot stopping decision](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-12/studies/b04a1db8d593f259/paper/pilot-decision.json/part-000000.bin).
- [Original plan](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-12/studies/b04a1db8d593f259/paper/plan.json).
- Exact CSV restoration paths, original byte counts and SHA256 values are retained in `evidence/r12-restored-provenance.json`. Restored primary CSV SHA256 is `c8f3553ae58173c3f7fdb2a46297ded22de376df05823efbdd9bbdfaad4f52b1` (4,261 bytes); participant CSV SHA256 is `18ae2c5f2da19509c9461695bf3ff7e50ca43e4f7c1c3e05b85a3a37045b90e8` (489,637 bytes).
- Local source `paper3m12/plan.py` and `paper3m12/reports.py` SHA256 values match the code manifest in the uploaded plan. The local initial `EXPERIMENT_CONFIG.json` remains a draft; scientific settings above were taken from the **uploaded effective plan**, whose execution is enabled.

## R13: persistence and additional-observation exploratory replay

### Question and fixed design

R13 asks whether an identical fitted calibration update changes its benefit when a synthetic spatial nuisance persists into the future, and whether later observations help select retain versus revert.

Eight historical pilot participants, three seeds, two nuisance generators and two amplitudes (0.05/0.15) produce 96 task cells. The report verifies 960 task-session outputs and 69,120 metric rows. The independent inferential unit remains eight participants. All sessions are averaged within each generator/amplitude/source-seed stratum, four perturbation strata equally within seed, then seeds equally within participant.

Initial calibration uses the first 12 acquired positions. All arms wait until position 20 before the same future query. Synthetic future operator strength is rho ∈ {0, .25, .5, .75, 1}; rho is **not elapsed time**. The primary spatial map is the raw M12 Full-SGD100 rule with its head frozen. The capacity control called `full_head` trains the 64-feature classification head (65 parameters), with spatial coefficients fixed to zero; it does not fine-tune the full EEGNet encoder.

The same initial fit and support labels are reused across the two endpoint worlds. At rho = 1 the nuisance persists; at rho = 0 the future input returns to original sensor coordinates. Query UIDs/order/labels remain matched. This is a controlled synthetic intervention on real EEG, not a natural session-shift or prospective cohort trial.

### Exact primary outcome

Primary estimand:

`I = (NLL_adapt − NLL_EA)_rho0 − (NLL_adapt − NLL_EA)_rho1`.

| Quantity | Mean | Participant percentile 95% CI | Exact sign-flip p | Alternative |
|---|---:|---|---:|---|
| Persistent excess NLL | −0.0010622570 | [−0.0015185735, −0.0005500599] | 0.01171875 | less |
| Reset excess NLL | −0.0010566143 | [−0.0015130035, −0.0005436179] | 0.9921875 | greater |
| Persistence interaction I | +0.0000056427 | [−0.0000237336, +0.0000333294] | 0.703125 | two-sided |

The prespecified conjunction for a helpful-persistent/harmful-reset sign reversal is not met (`joint_sign_reversal_p = 0.9921875`). The mean excess NLL is slightly negative in both worlds. The primary interaction interval crosses zero. These values do not support the hypothesis that persistence produces the anticipated sign reversal under these tested perturbations.

The original report labels all results `COMPLETE_EXPLORATORY`, `independent_confirmation_available: false`, and explicitly says the pilot subset was in source meta_train. Completeness of a finite exploratory replay is not independent scientific confirmation.

### Additional-observation policies and cost

The fixed heuristic candidates use sentinel inputs only (`candidate_X`) or later revealed labels (`candidate_XY`). The X-only version has 12 revealed labels at either k = 4 or k = 8; XY versions have 16 or 20 revealed labels. Both still wait through original position 20. They are not equal-label-budget comparisons. Ordinary updated EA, holdout keep/revert, scalar shrinkage and all-paid-label refitting are included. These are development-frozen exploratory heuristics; there is no separately trained, independently validated risk predictor.

Descriptive mean BA/NLL tables across all four generator/amplitude strata are supplied in `tables/r13_exploratory_sentinel_policies.csv`. At rho0/k4, for example, candidate_XY BA = 63.7153%, retain_initial = 63.7909%, ordinary holdout = 63.7455% and refit-all-labels = 63.9579%. These are descriptive same-participant means, **not newly significant superiority results**. A claim that the candidate gate improves calibration efficiency over conventional holdout/refitting is unsupported.

### Source and restoration evidence

- [Current original R13 report](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-13/studies/c2bba596810fb259/paper/report.json/part-000000.bin).
- [Published report-and-metrics archive manifest](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-13/archive-groups/61cd0534a9a1403eb972c365c05a06f1666975b310c970e2a06ca42ba37a2fb6/manifest.json).
- [Paper-side reduced plan](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-13/studies/c2bba596810fb259/paper/plan.json).
- Original participant CSV: 317,034 bytes, SHA256 `c27311421bd2ffad05859d5ad234cc1bada6ee06ed33425c20488657b9a656b8`.
- Full report archive report date is 28 October 20:16 UTC; the current paper report date is 29 October 06:21 UTC. All scientific fields are identical after removing `updated_utc`; freshness does not imply new scientific outcomes.
- Local `paper3r13/algorithm.py` and `paper3r13/report.py` SHA256 match the uploaded full plan's code manifest exactly. Protocol framing was checked against `PROTOCOL.json` and the effective plan; implemented scope says new independent cohort unavailable.

## R8/R9 cross-checks

Only the compact original primary-comparison CSVs were restored, not full model archives. R8 has 40 primary rows, all Holm p > .05 (minimum .15625). R9 has 24 primary rows, all Holm p > .05 (minimum .9755859375). Both compare historical external cohorts Pan (14 participants) and Weibo (10 participants), which were reused across development rounds. Dataset separation from Stieger does not make these untouched confirmation cohorts after repeated analysis/model selection.

R8 covariance-preserving class-contrast rotation is a concrete method construction, but its `fiber_bounded` vs support EA BA point effects are +0.26190 pp on Pan and +0.38889 pp on Weibo, with Holm p = 1. R9 EA+anchored-RBF vs support EA gives +0.69048 pp on Pan (Holm .975586) and +0.93651 pp on Weibo (Holm 1). Neither justifies a general superiority or external-confirmation claim. Their original CSVs are retained in evidence for the main audit agent.

## Files ready for the writer

- `tables/r12_verified_primary_comparisons.csv`: original six primary comparisons, concise columns.
- `tables/r13_verified_primary_statistics.csv`: original primary endpoint/interaction statistics.
- `tables/r13_verified_primary_participants.csv`: original eight participant contrast values.
- `tables/r13_exploratory_persistence_curve.csv`: descriptive means on the fixed grid.
- `tables/r13_exploratory_sentinel_policies.csv`: descriptive matched-observation policy/cost table.
- `tables/r13_persistence_participant_values.csv` and `r13_sentinel_participant_values.csv`: auditable plotting rows; seeds/strata already averaged appropriately, n not inflated.
- `research/build_validation_tables.py`: reproducible extraction/aggregation script.
- `evidence/validation_verified.json`: machine-readable summary and source links.
