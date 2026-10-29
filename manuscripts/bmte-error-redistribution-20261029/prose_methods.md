## Materials and methods

### Comparative design and evaluation cohorts

We conducted a retrospective, participant-paired analysis of completed binary left- versus right-hand motor-imagery EEG evaluations. Five cohorts supplied the main analysis: 12 Kumar participants, nine BNCI2014-001 participants, and 11 Stieger, 14 Pan and ten Weibo participants [@kumar2024; @nemar177; @tangermann2012; @stieger2021; @pan2023; @yi2014]. These 56 dataset participants contributed distinct cohort-specific endpoints. Six Kumar development participants were summarized separately. A source-model extension used 85 Dreyer participants and zero target-calibration labels [@dreyer2023]. Source-model replacement and target calibration retained separate analysis roles.

The conventional-decoder evaluation compared update components within Kumar and BNCI. The neural-decoder evaluation compared 18 recorded calibration configurations within Stieger, Pan and Weibo. All contrasts paired methods on the same cohort-specific query observations. Recorded preprocessing, source training and participant partitions defined each evaluation; cohort estimates were reported separately. The Stieger partition contained 40 training-reference, 11 development-validation and 11 historical test participants. Neural evaluation cohorts had prior development exposure. This reanalysis retained their saved outputs and configurations without new fitting or query-driven selection.

### Conventional decoder updates

Kumar2024 contains 18 participants and six sessions. Participants sub-1, 4, 7, 10, 13 and 16 formed the development group; the remaining 12 formed the primary analysis cohort. Common-average-referenced 22-channel signals were filtered at 8–30 Hz and resampled to 128 Hz. Complete cue-anchored five-second epochs contained 22×640 samples [@kumarcodebook; @jayaram2018]. Session 0/runs 0–1 supplied 40 source-training trials. Each subsequent session supplied its first run as support and subsequent whole runs as query. Prefixes contained 4, 8, 12, 16 or the complete 19/20 valid support labels, with query identities fixed across budgets. Single-class prefixes were retained. Of 90 participant/session cells, 14 had 19 full-support labels and 76 had 20.

Frozen decoding fitted four-component common spatial patterns (CSP) and linear discriminant analysis (LDA) on source training. LDA-only updating retained source spatial filters and refitted LDA on source plus support; full updating refitted both on the same union. Covariance alignment separately whitened source and support signals before full CSP/LDA refitting. Kumar CSP used Ledoit–Wolf regularization and LDA automatic shrinkage [@ledoit2004]. Alignment operated in the 21-dimensional common-average-reference subspace using each domain's mean trial covariance. Query transforms used the support reference fixed before evaluation.

BNCI used 22 EEG channels with the recorded reference and cue-relative [0,4) s epochs at 250 Hz after run-wise 8–30 Hz filtering. First-day runs 1–4 supplied 96 hand-imagery source trials. The next day's first run supplied its first 20 chronological hand trials as support; the remaining four were withheld. Runs 2–6 supplied 120 query trials per participant, 60 per class. Its separately versioned implementation used four CSP components and covariance/LDA shrinkage 0.1. All updating pipelines shared the same fitting labels within each dataset. BNCI was also a neural source-training dataset; its conventional evaluation characterizes that pipeline's component contrasts.

### Neural calibration and acquisition budgets

The neural evaluation comprised 105 Stieger sessions, 28 Pan sessions and ten Weibo sessions. Inputs used FC3, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3 and CP4. Common-average referencing, fourth-order 8–30 Hz bidirectional filtering and resampling to 128 Hz produced a one-second 11×128 crop, 0.25–1.25 s after the task anchor. Stieger's anchor occurred 2 s after cue onset. Trials were temporally demeaned and divided by their joint channel–time root-mean-square amplitude.

EEGNet source checkpoints were trained on BNCI2014-001, Cho2017, Lee2019 and PhysioNet MI with three seeds [@lawhern2018; @cho2017; @lee2019; @goldberger2000]. The 18 configurations covered frozen and aligned references, head updates, joint spatial/head updates, protected spatial updates, cross-fitted selection/stacking, bounded geometry and anchored residual controls. Saved method identities, initialization and fitting anchors were retained. Target batch-normalization and dropout states remained in evaluation mode.

Neural support budgets counted the first 4, 8, 12, 16 or 20 acquired positions. Rejected trials consumed positions without replacement. Query observations were fixed after position 20; Stieger additionally excluded the remainder of its boundary run and evaluated subsequent complete runs. Pan and Weibo used published trial order. Neural alignment averaged trace-normalized support covariances in the ten-dimensional common-average subspace, applied 5% isotropic shrinkage, whitened the estimate and recolored it to a fixed source reference. Where required, feature references came from the 40 Stieger training-reference participants. No query statistics updated these references.

### Prediction transitions and component contrasts

For query label \(y_j\), class count \(n_c\) and weight \(w_j=1/(2n_{y_j})\), balanced accuracy (BA) is the weighted correct-prediction sum. Relative to reference \(a\), we measured corrected errors and introduced errors as

\[
C=\sum_jw_j\mathbf{1}\{\widehat y_{aj}\ne y_j,\widehat y_{mj}=y_j\},\qquad
I=\sum_jw_j\mathbf{1}\{\widehat y_{aj}=y_j,\widehat y_{mj}\ne y_j\}.
\]

Class-balanced disagreement was \(D=C+I\), and net gain was \(G=C-I=BA_m-BA_a\). Values are reported in percentage points. These prediction-change measures connect to prediction churn and negative flips [@fard2016churn; @yan2021congruent]. The frozen source decoder was the principal scoring reference. Additional matched component contrasts compared full with LDA-only updating and alignment with full updating. Neural sensitivity analysis used support alignment as the scoring reference while preserving each method's fitting anchor.

Participant summaries averaged sessions equally and, for neural evaluations, averaged the three source seeds. With participant values \(C_i,I_i,D_i,G_i\), total cancellation decomposed exactly as

\[
\overline{D}-|\overline{G}|=
\operatorname{mean}_i[2\min(C_i,I_i)]
+\operatorname{mean}_i|G_i|-|\operatorname{mean}_iG_i|.
\]

The first component describes cancellation within participant summaries, including their session/seed averaging. The second describes cancellation between participant net changes. Fractional components divide by \(\overline{D}\); zero-disagreement ratios remain undefined.

### Statistical analysis and reproducibility

The participant was the independent resampling unit within each cohort. Pointwise 95% percentile intervals used 10,000 paired participant-bootstrap draws, seed 20261029, conditional on the recorded session/seed summaries. All scheduled methods and budgets were retained. New transition and cancellation analyses were descriptive, without additional hypothesis tests or a pooled cross-cohort effect.

Archived endpoint tests retained their original comparison roles and multiplicity families. Kumar full-minus-frozen was the original primary contrast and full-minus-LDA-only secondary; BNCI full-minus-LDA-only was primary. The supplement preserves the historical intervals, tests and separate neural inference families alongside the new descriptive intervals. Query IDs, labels, source-table hashes and the transition identities were checked before analysis. Code regenerates the tables and figures from saved predictions or their verified participant summaries; no additional model training or inference was performed.
