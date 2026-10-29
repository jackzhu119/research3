# Evidence-Retaining Functional Calibration for Few-Trial Motor Imagery EEG Decoding

**Ziyuan Zhu¹\*, Sihao Liu¹ and Haoyang Yu¹**

¹ College of Artificial Intelligence Medicine, Chongqing Medical University, Chongqing 400016, China; liu3100853087@gmail.com (S.L.); jiaguca3329@163.com (H.Y.)

\* Correspondence: zzy2630816871@gmail.com  
ORCID (Ziyuan Zhu): <https://orcid.org/0009-0005-1153-4926>

## Abstract

**Background/Objectives:** A short calibration recording can improve a motor imagery decoder while also replacing previously correct predictions with errors. We developed Evidence-Retaining Functional Calibration (ERFC) to determine when a small support recording justifies changing an individual EEG prediction.

**Methods:** ERFC combines a frozen EEGNet decoder and support-based alignment with a regularized residual function. It retains functions preserving 90% of the observed support-objective gain, includes support-invisible query completions, and projects the anchor probability onto a compatible outer interval. A retrospective analysis used three public EEG cohorts comprising 35 participants and 143 sessions, three source-model seeds, and budgets of 4–20 acquired trial positions. Inference used paired participant summaries.

**Results:** At 20 acquired positions, balanced accuracy was 62.25%, 63.83% and 71.94% in the Stieger, Pan and Weibo cohorts, respectively. The 12 archived primary accuracy comparisons had Holm-adjusted p-values of at least 0.5625. ERFC changed 14.37–15.07% of query probabilities by at least 0.0001. In an additional exploratory 18-test analysis, introduced errors were 0.69–0.75 percentage points lower than with a matched point-fit residual (Holm-adjusted p = 0.0039–0.0313). Corrected anchor errors were also 0.78–1.27 percentage points lower.

**Conclusions:** ERFC provides a query-specific rule for translating scarce calibration evidence into prediction changes. The observed correction–preservation trade-off makes the consequences of adaptation measurable alongside aggregate decoding accuracy.

**Keywords:** motor imagery; electroencephalography; brain–computer interface; few-trial calibration; functional uncertainty; kernel methods; negative transfer

<!-- BACK_MATTER -->

## Supplementary Materials

Supplementary Methods and Tables S1–S6 provide the complete primary and exploratory comparison families, configuration and signal-processing details, computational measurements, and related development analyses. Figure S1 documents the earlier protected-tangent comparison. Machine-readable tables, participant summaries, source hashes and executable analysis and plotting scripts accompany the manuscript.

## Author Contributions

Conceptualization, methodology, formal analysis, investigation, visualization and writing—original draft preparation, Z.Z.; software, Z.Z. and S.L.; data curation, Z.Z. and H.Y.; publication-policy verification, S.L.; material organization and repository-record verification, H.Y.; project administration, Z.Z. 

## Funding

This research received no external funding.

## Institutional Review Board Statement

This study was a secondary computational analysis of publicly released EEG datasets and involved no new participant recruitment, contact, intervention or biospecimen collection. The exemption basis considered for this secondary use is Article 32(1) of the *Measures for Ethical Review of Life Science and Medical Research Involving Human Subjects* (Guo Wei Ke Jiao Fa [2023] No. 4), which addresses lawfully obtained public data subject to the conditions specified in that Article [@nhc2023]. The original recording studies report their ethics approvals and participant consent [@stieger2021; @pan2023; @yi2014].

## Informed Consent Statement

Informed consent for the original EEG recordings was obtained by the respective dataset investigators as reported in the cited source studies. No new participants were enrolled in the present analysis.

## Data Availability Statement

Raw EEG data are available from the original repositories cited with the source and evaluation datasets. Their original licenses and access conditions apply. The analyzed results are pinned to `jackzhu119/research3` commit `a16d06760ae77699a809c9ba36d002d684560c82` in the `paper3-results` branch. The analyzed formal exploratory stage has recipe identifier `3c0482d286bfac9e` in the round-10 result partition. The accompanying reproducibility package contains the restored participant summaries, complete comparison tables, deterministic secondary analysis, figure data and source-manifest hashes needed to reproduce this manuscript's numerical analyses and figures. ERFC corresponds to the archived `profile` implementation, version `paper3-r10-query-profile-v1`, and configuration `a0.3-c0.1`; the working manuscript name does not denote a newly trained backbone or a different experimental run. Repository: <https://github.com/jackzhu119/research3>.

## Acknowledgments

We thank the investigators who made the EEG recordings and their documentation publicly available. ChatGPT (OpenAI; accessed on 29 October 2026) assisted with literature retrieval, manuscript organization and drafting, language editing, and preparation of executable analysis and plotting code. The numerical analyses are traceable to the archived data and accompanying scripts. The authors retain responsibility for the content submitted for publication.

## Conflicts of Interest

[All authors must confirm their conflict-of-interest declarations before submission.]

## References
