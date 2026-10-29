# Verified evidence for the manuscript's main scientific argument

Audit date: 29 October 2026. Repository: `jackzhu119/research3`. All remote result reads in this audit were pinned to `a16d06760ae77699a809c9ba36d002d684560c82`. No training, GPU work, model selection, report rebuilding, or scientific-server mutation was performed. Existing source reports were downloaded selectively and checked against their declared SHA256; the new 18-comparison mechanism analysis is explicitly separated below.

## Scientific decision

The most defensible main contribution is the implemented R10 query-functional calibration rule, provisionally called **Evidence-Retaining Functional Calibration (ERFC)**. This is an editorial name for the exact original `profile` implementation (`paper3-r10-query-profile-v1`, selected `a0.3-c0.1`), not a new experiment or renamed external network. The primary evidence supports selective correction under a small acquisition budget, accompanied by fewer introduced errors and fewer corrected errors than matched MAP. The registered primary accuracy family does not support an accuracy-superiority claim. No equivalence or noninferiority test was registered or performed.

R3 is useful as prior mechanism evidence: constraining an otherwise aggressive tangent update restores 4.66 percentage points relative to its unprotected counterpart. It does not establish that meta-learned geometry beats strong spatial SGD. R7 does not establish superiority of the tested spatiotemporal extension.

## R3: exact protection result

The downloaded `reports/primary_comparisons.csv` SHA256 is `77965128761b565aed6437749ec3ce9746b87c5f9e9d70ccfafb6b890d7395e5`.

- Comparison: `legacy:tangent_protected` minus `legacy:tangent_profiled`.
- Participants: 11. Original sessions per seed: 105. Seeds: 20261018, 20261019, 20261020. Paired session–seed rows: 315; these are not 315 independent participants.
- Candidate balanced accuracy: 0.6151325955091348; reference: 0.5685438737622645.
- Difference: **4.658872174687025 percentage points**; individual 95% percentile participant-bootstrap interval **[2.39646724142858, 6.909417253022452]**.
- Two-sided exact paired sign-flip p = 0.0048828125; Holm p = 0.0439453125 over the 10-comparison primary family.
- Nine of eleven participant differences are positive. No unmatched session–seed rows.
- R3 bootstrap: 20,000 draws, statistics seed 20261018. The CI is unadjusted, not a simultaneous family interval or inversion of the sign-flip test. The sign-flip null requires independent participant differences with sign symmetry/exchangeability.
- Strong controls: spatial SGD10 = 61.4910704071% BA, SGD100 = 61.9633694931%; the protected tangent mean is 61.5132595509%. Do not turn the 4.66-point mechanism effect into a claim of superiority to SGD.
- Specifically meta-trained basis comparisons were not positive: `profiled:tangent_profiled` versus `matched_sgd:tangent_profiled`, +0.0449701 points, Holm p=1; `protected:information_protected` versus its matched SGD basis, −0.0053618 points, Holm p=1.
- R3 explicitly records exploratory internal replication on previously exposed participants, not fresh confirmatory evidence.

The `legacy` prefix means a reused existing SGD-meta-trained spatial dictionary. In actual R3 code, the 99-dimensional trace-free generator coordinates are reduced by the learned `99×r` basis; a fixed identity-tangent class-balanced logistic surrogate jointly fits spatial coefficients and residual head. `tangent_protected` then scales the **whole fitted spatial-plus-head update** by powers of one half until the class-balanced support Bernoulli KL from the original source predictions is at most 0.01. It is not a new backbone and is not a new invention of matrix exponentials, trust regions, KL divergence, or Stiefel geometry.

Sources: [R3 source manifest](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-03/paper-data/4c4d33c077891ece00b2cc34b36261a8db1aa33ede9a388e7fe4703034b920f0/manifest.json). Source bytes and local SHA proofs: `evidence/main_download_proofs.json`. Curated values: `tables/r3_primary_comparisons_verified.csv`; participant differences: `tables/r3_protection_participant_differences.csv`.

## R7: exact scope

The final 12 primary BA comparisons are from Pan2023 published-order evaluation: 14 participants, 28 participant/sessions, 3 seeds, 84 session–seed rows. No comparison has positive Holm-adjusted evidence. The stored adjustment family is **24**, incorporating the declared multimetric family; it would be wrong to recompute adjustment as 12 merely because the manuscript prints twelve BA rows. These externally sourced data had prior-round exposure, so this is repeated exploratory external evaluation. Its published trial order is not proof of physical acquisition/run chronology.

Source primary table SHA256: `0b4507c5a62a4981dffbe23a23da8caaf09b9245b8f432ee082c2ae616848467`. The large evaluation-summary table was restored from CSV row parts with repeated headers removed; its final assembled SHA256 equals the original source SHA256 `b2694aa674c156c08e153d5e51c341388fc33a4df96f76af0769a2f4d7766245`. No rows were truncated. [R7 paper manifest](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-07/paper-data/4d44fa6d6f6d754f70d70f1cb4d5b8439ea0721bac938ce54dc85c7138352a24/manifest.json).

## R10 formal exploratory stage: complete primary family

The stage contains 165 sealed tasks: three feature-reference tasks plus 18 methods × 3 cohorts × 3 seeds. Five acquisition budgets are evaluated inside each evaluation task; task count is not participant count or independent experiment count. The signed stage recipe is `3c0482d286bfac9e31df4835898e205815d348a14aef7622681cb5f359d949d1`.

The four primary contrasts, each at budget 20 and metric BA, are ERFC versus support-only EA, spatial SGD100, EA-anchored residual GP, and matched MAP. Across three cohorts these form 12 primary tests; none is Holm-significant. There are separately 24 secondary ablation tests. Exact primary means and intervals are in `tables/r10_primary_comparisons_verified.csv`.

| Cohort | Participants | Sessions | Session–seed pairs | ERFC BA | Support-EA BA | Difference pp | 95% individual CI pp |
|---|---:|---:|---:|---:|---:|---:|---|
| Stieger held-out partition | 11 | 105 | 315 | 62.25462168% | 62.21324283% | +0.04137885 | [−0.01734265, +0.10489067] |
| Pan2023 published order | 14 | 28 | 84 | 63.83333333% | 63.82142857% | +0.01190476 | [−0.09523810, +0.15476190] |
| Weibo2014 published order | 10 | 10 | 30 | 71.93650794% | 71.91269841% | +0.02380952 | [−0.07142857, +0.14285714] |

All three ERFC-versus-EA Holm p values equal 1. ERFC minus matched-MAP BA equals −0.0382953, −0.2738095, and −0.5833333 points on Stieger, Pan, and Weibo, respectively. Their primary Holm p values are 1, 1, and 0.5625. Do not infer equivalence from these tests. A negative individual bootstrap CI and a nonsignificant adjusted sign-flip test can coexist because they are different inferential procedures and family scopes.

The actual report explicitly records `independent_confirmation=false`, `exploratory_cohorts_reused_for_algorithm_design=true`, and `retrospective_protocol_revision=true`. R3 and R10 share the same eleven Stieger held-out participants. These are not independent replications. Target-query labels were not arguments to fitting or prediction, and did not enter R10 episode-specific fitting; this does not erase the researcher's historical exposure to previous cohort results.

The actual development selection was a retrospective amendment using paired EA gain and a one-standard-error rule on preserved mechanism outputs. The selected candidate was `a0.3-c0.1`; selection participants S22, S40, S57, S60; check participants S16, S46, S5, S7. It was not the original untouched confirmatory protocol or a new independent test cohort. The preserved 15-task mechanism report remains distinct from the 165-task exploratory stage.

[Formal exploratory report manifest](https://github.com/jackzhu119/research3/blob/a16d06760ae77699a809c9ba36d002d684560c82/results/paper3/round-10/stages/formal_exploratory/3c0482d286bfac9e/reports/paper-data/1af3f6b478778cd67a086871462125b07e28f26b675cee9955c8071ca1388103/manifest.json).

## New post hoc matched-MAP mechanism analysis

This analysis was performed for manuscript preparation after the original reports existed. It is not substituted for the original primary family. Original participant-level source rows are in `evidence/r10/reports/participant_metrics.csv`, SHA256 checked against its lossless source manifest; the selected 20-position rows for ERFC and matched MAP are saved unchanged with source links in `tables/r10_matched_map_participant_rows.csv`.

At each of the three cohorts, six contrasts were defined: BA; balanced NLL; class-balanced introduced-error mass relative to EA; corrected-error mass relative to EA; query-probability update fraction; and fraction of sessions with BA below the EA anchor. This is one **18-test Holm family**. All tests use independent paired participants after the original equal-session, equal-seed averaging. Two-sided exact sign flips enumerate all 2^n signs. Individual percentile-bootstrap intervals use 10,000 draws and fixed seed 20261029; they are not simultaneous intervals. Reproducible code: `research/analyze_main.py`.

Compared with matched MAP, ERFC decreases introduced-error mass by 0.7466, 0.7143, and 0.6905 percentage points on Stieger, Pan, and Weibo. Corresponding Holm18 p values are 0.013671875, 0.00390625, and 0.03125. The same restriction also decreases corrected-error mass by 0.7849, 0.9881, and 1.2738 points; these differences are also significant. Thus the result is a controlled correction trade-off, not an accuracy gain.

Query probability changes decrease by 83.9497, 84.7976, and 84.8254 points (all Holm18 significant). Session negative-transfer fraction decreases by 27.0366, 17.8571, and 13.3333 points; the Stieger and Pan contrasts are significant, while Weibo has adjusted p=0.5. These are cohort-specific, participant-weighted means; do not pool sessions as if independent or rename this result a clinical no-harm guarantee.

All exact estimates, means, intervals, p values and source links are in `tables/r10_exploratory_matched_map_mechanisms.csv`. `tables/r10_exploratory_paired_values.csv` gives every difference used in the tests.

## Originality and claim boundaries

A new working label for the exact implemented ERFC combination is appropriate. The claimable construction is the evidence-retaining support-objective sublevel set, support-invisible function completion, per-query outer bound, and anchor-relative minimax projection within one small-support EEG calibration procedure. Kernel methods, RKHS decompositions, Schur complements, convex duality, minimax loss, EEGNet, covariance alignment, and the earlier spatial dictionary must retain appropriate attribution. None becomes an original neural backbone by renaming it.

The prediction interval is an objective-compatible model interval, not a GP posterior credible interval, frequentist confidence interval, or coverage-calibrated interval for a true future label probability. Its conditional theorem does not assert future accuracy, distribution-shift robustness, or clinical safety. State the mathematical scope once in the method and test practical outcomes with the measured metrics.
