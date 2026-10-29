## Results

### Evaluation coverage and paired prediction records

The main comparative frame contained 56 participants in five cohorts and 212 target sessions (Table 1). Conventional analyses paired four pipelines on fixed later-session queries; neural analyses retained all eighteen configurations and five scheduled acquisition budgets. Kumar's six development participants contributed 30 additional target sessions and were summarized separately. The source-model extension retained its own 85-person cohort.

The conventional input contained 78,600 Kumar prediction rows covering 3,930 distinct target trials and 36 BNCI prediction files covering 1,080 target trials. Methods and budgets matched exactly on ordered query IDs and labels within participant/session. The neural source table contained 3,150 participant–configuration–budget records; two scoring references produced 6,300 paired summaries. These repeated records preserve the participant units in Table 1. The identities C−I=BA difference and C+I=D held throughout the analysis.

**Table 1. Cohorts and information budgets in the main evaluation.** Sessions are target recordings within participants; labels, sessions, seeds and methods do not increase the independent participant count.

| Cohort | Decoder evaluation | Participants | Target sessions | Calibration unit |
| --- | --- | --- | --- | --- |
| Kumar | Conventional | 12 | 60 | 4/8/12/16 valid labels; full 19/20 |
| BNCI2014-001 | Conventional | 9 | 9 | 20 valid labels |
| Stieger | Neural | 11 | 105 | 4/8/12/16/20 acquired positions |
| Pan | Neural | 14 | 28 | 4/8/12/16/20 acquired positions |
| Weibo | Neural | 10 | 10 | 4/8/12/16/20 acquired positions |


Kumar development participants and the zero-target-label source-model extension are reported separately. Neural source checkpoints used BNCI among their training datasets; the BNCI row identifies a conventional pipeline evaluation. Budget units retain their cohort-specific definitions.



![](figures/Fig1_study_design.png){width=95%}

**Figure 1. Study design and paired prediction analysis.** Five cohorts comprise 56 participants in the main evaluation. Conventional CSP/LDA actions and source-trained EEGNet calibration configurations retain their separately defined signal processing, fitting and acquisition contracts. Each update is evaluated on the same queries as its within-cohort reference. Class-balanced corrected errors C and introduced errors I give total prediction disagreement D = C + I and net balanced-accuracy change G = C − I. Sessions, source seeds and configurations are repeated observations within participants. Kumar budgets count valid support labels, whereas neural budgets count acquired positions before quality control. Six previously exposed Kumar development participants and the source-model extension are reported separately.

### Component comparisons locate the prediction changes behind net gains

Kumar frozen CSP/LDA achieved 57.94% BA, classifier-only recalibration 59.89%, full refitting 65.36% and alignment followed by full refitting 70.66% (Figure 2). Full refitting corrected 20.85 pp of source errors while introducing 13.43 pp, giving +7.42 pp net gain [+2.64, +12.42]. Its 34.28-point disagreement was substantially larger than the final score change.

The matched-label spatial-refitting increment beyond classifier-only updating comprised 16.04 points of correction and 10.57 points of introduction, yielding +5.47 points [+2.16, +8.92]. This increment represented 73.7% of the mean full-minus-frozen gain. Alignment beyond full refitting corrected 14.61 points and introduced 9.32 points, adding +5.30 points [+2.82, +8.31]. Each successive operation therefore had a distinguishable error-transition profile under the same source and target fitting labels.

BNCI displayed a different component balance. The spatial-refitting increment was +0.19 points [−1.94, +2.13]: 4.17 corrected versus 3.98 introduced points. Alignment beyond full refitting produced 12.78 corrected and 7.13 introduced points, adding +5.65 points [+3.15, +8.70]. All nine participant alignment differences were positive. The contrast identifies the added alignment operation within that conventional protocol; the full processing paths differ across cohorts.

**Table 2. Error transitions in conventional decoder updates at full support.** C: corrected errors; I: introduced errors; D: class-balanced disagreement; G: net BA change. Intervals are the new exploratory pointwise participant-bootstrap intervals, with no additional hypothesis tests.

| Cohort | Matched comparison | C, pp | I, pp | D, pp | G, pp [95% interval] |
| --- | --- | --- | --- | --- | --- |
| Kumar | Classifier − frozen | 10.70 | 8.75 | 19.46 | +1.95 [−0.97, +5.08] |
| Kumar | Full − frozen | 20.85 | 13.43 | 34.28 | +7.42 [+2.64, +12.42] |
| Kumar | Full − classifier | 16.04 | 10.57 | 26.61 | +5.47 [+2.16, +8.92] |
| Kumar | Aligned full − full | 14.61 | 9.32 | 23.93 | +5.30 [+2.82, +8.31] |
| BNCI | Classifier − frozen | 3.24 | 2.13 | 5.37 | +1.11 [−0.46, +2.96] |
| BNCI | Full − frozen | 5.56 | 4.26 | 9.81 | +1.30 [−1.76, +3.98] |
| BNCI | Full − classifier | 4.17 | 3.98 | 8.15 | +0.19 [−1.94, +2.13] |
| BNCI | Aligned full − full | 12.78 | 7.13 | 19.91 | +5.65 [+3.15, +8.70] |


Kumar full support contains 19 or 20 valid labels; BNCI support contains exactly 20. Historical endpoint intervals, primary/secondary roles and multiplicity adjustments are preserved separately in the Supplementary Materials. Frozen pipelines receive no target labels; the updating pipelines share source/support fitting examples within each cohort.



![](figures/Fig2_component_error_balance.png){width=95%}

**Figure 2. Accuracy and prediction changes along conventional update operations.** A–B: participant trajectories (grey) and participant-equal cohort means for frozen CSP/LDA, classifier-only recalibration, spatial-filter and classifier refitting, and refitting after support-only Euclidean alignment. Kumar uses its full current-session support run of 19/20 valid labels; BNCI uses exactly 20 valid labels. Error bars are pointwise 95% participant-bootstrap intervals from 10,000 resamples. C–D: adjacent-operation comparisons isolate classifier versus frozen decoding, spatial refitting versus classifier-only recalibration, and alignment versus full refitting. Green and orange bars partition prediction disagreement into corrected and introduced class-balanced mass. Separate black diamonds show their net gain and pointwise intervals. The updating pipelines share source/support fitting labels and all four pipelines share query identities within each cohort; frozen decoding uses no target labels. The cohorts retain different preprocessing and fitting implementations.

### Identical neural mean accuracy concealed different prediction exchanges

The Pan bias-only head and cross-fitted selection stack both attained 63.36% BA, relative to 63.18% for the frozen source (Figure 3). Bias-only updating corrected 1.26 pp and introduced 1.08 pp; the stack corrected 5.80 and introduced 5.62 pp. Both consequently gained 0.1786 pp, while disagreement was 2.35 versus 11.42 pp, a 4.87-fold difference. The identical observed means describe this matched-score example, without establishing equivalence.

Participant outcomes also differed. Bias-only calibration improved nine participants and reduced BA in three, with two unchanged. The stack improved ten and reduced BA in three, with one unchanged. Paired participant plots retain the amount of disagreement alongside each net gain, showing which people contributed to the shared cohort mean.



![](figures/Fig3_equal_accuracy_participant_changes.png){width=95%}

**Figure 3. Equal mean accuracy with different prediction changes.** A: participant-by-configuration summaries for all seventeen non-frozen neural configurations at 20 acquired positions, relative to the frozen source. Configuration repetitions do not increase the 11 Stieger, 14 Pan or 10 Weibo participants. B: bias-only calibration and the cross-fitted stack in Pan both attain mean BA of 63.3571% and net gain of +0.1786 percentage points, while their disagreement is 2.3452 and 11.4167 points. C–D: disagreement and net gain for the same fourteen participants; grey lines connect participant pairs and black diamonds indicate participant-equal means. The shared observed mean is an illustrative matched-score case, not a statistical equivalence result.

### Complete neural comparisons separate within-person and between-person cancellation

Across sixteen non-anchor configurations in each of three cohorts at twenty acquired positions, median disagreement against the frozen source was 9.52 pp (IQR 4.80–12.79), whereas median signed net gain was +0.68 pp (IQR +0.39–+0.96). This fixed 48-cell descriptive frame contextualized the Pan example within the complete configuration grid. All eighteen configurations and five budgets are retained in Supplementary Figures S1–S2 and the machine-readable tables.

The total cancellation fraction had a median of 93.02%. Its within-participant-summary component had a median of 84.42%, and its between-participant component a median of 7.07% (Table 3). Additivity holds within each configuration/cohort decomposition; the separately summarized component medians need not add. Including support alignment as a seventeenth non-reference configuration gave 51 cells, median disagreement 9.62 pp and median total cancellation 93.16%.

Using support alignment as the scoring reference retained substantial opposing prediction changes: the common 48-cell frame had median disagreement 8.09 pp and median total cancellation 95.35%. The method's recorded fitting anchor was unchanged by this sensitivity reference. Zero-disagreement ratios remain undefined in the complete tables.

**Table 3. Scoring-reference sensitivity of neural cancellation at twenty acquired positions.** The same sixteen non-anchor configurations in three cohorts define both 48-cell frames. Ratios divide each cell's cancellation component by its mean disagreement. Columns report separate medians across the repeated configuration/cohort cells.

| Scoring reference | Median D, pp | Median G, pp | Within, % | Between, % | Total, % |
| --- | --- | --- | --- | --- | --- |
| Frozen source | 9.52 | +0.68 | 84.42 | 7.07 | 93.02 |
| Support alignment | 8.09 | +0.01 | 82.54 | 11.32 | 95.35 |


### Calibration prefixes changed the contribution of each operation

In Kumar, the mean full-minus-classifier spatial increment was −0.08, +1.18, +2.90, +4.79 and +5.47 pp at 4, 8, 12, 16 and full support, respectively. Alignment beyond full refitting added +1.48, +4.85, +3.28, +3.15 and +5.30 pp. The scheduled curves preserve both the amount of calibration information and each operation's changing correction/introduction balance (Figure 4). The six-person development cohort remains separately visible in Supplementary Figure S3.

Weibo's recorded neural alignment pathway had BA 66.92% after four acquired positions versus 71.54% for the frozen source. The paired difference was −4.62 pp [−7.98, −1.33], with eight of ten participants below the frozen reference. Alignment-minus-source means across the five budgets were −4.62, −1.54, −0.63, −0.08 and +0.37 pp. At twenty positions, six participants improved and four decreased; the interval was [−1.01, +1.64]. Stieger and Pan did not show the same large early cohort-mean deficit. The results locate cohort-specific operating points of the recorded alignment implementations.



![](figures/Fig4_budget_profiles.png){width=95%}

**Figure 4. Error redistribution across calibration budgets.** A–B: participant-equal BA for four conventional update actions in Kumar and frozen/support-aligned EEGNet in Weibo. C–D: corrected errors C, introduced errors I and net gain G for full spatial/classifier refitting in Kumar and support alignment in Weibo, each relative to its frozen reference. Shading shows pointwise 95% participant-bootstrap intervals from 10,000 resamples. Left-column budgets count valid labels, with full support consisting of 19 or 20 labels; right-column budgets count acquired positions before quality control. Frozen references receive no target calibration labels and are repeated on these axes for comparison. The different cohort-specific alignment paths do not constitute a controlled comparison of architectures or covariance estimators.

### A separate source-model replacement extension

On matched Dreyer observations, replacing the nine-person source model with the 60-person source model corrected 18.86 points and introduced 13.01 points, producing 31.87 points of disagreement and +5.85 points net BA change. These summaries average three fitted source seeds within each of 85 participants. The extension uses zero target-calibration labels and remains separate from the calibration-budget comparisons (Supplementary Figure S4).
