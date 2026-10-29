# Evidence-Retaining Functional Calibration for Few-Trial Motor Imagery EEG Decoding

**Ziyuan Zhu¹\*, Sihao Liu¹ and Haoyang Yu¹**

¹ College of Artificial Intelligence Medicine, Chongqing Medical University, Chongqing 400016, China; liu3100853087@gmail.com (S.L.); jiaguca3329@163.com (H.Y.)

\* Correspondence: zzy2630816871@gmail.com  
ORCID (Ziyuan Zhu): <https://orcid.org/0009-0005-1153-4926>

## Abstract

**Background/Objectives:** A short calibration recording can improve a motor imagery decoder while also replacing previously correct predictions with errors. We developed Evidence-Retaining Functional Calibration (ERFC) to determine when a small support recording justifies changing an individual EEG prediction.

**Methods:** ERFC combines a frozen EEGNet decoder and support-based alignment with a regularized residual function. It retains functions preserving 90% of the observed support-objective gain, includes support-invisible query completions, and projects the anchor probability onto a compatible outer interval. A retrospective analysis used three public EEG cohorts comprising 35 participants and 143 sessions, three source-model seeds, and budgets of 4–20 acquired trial positions. Inference used paired participant summaries. Every acquisition budget shared the same subsequent query set. The matched control retained the same encoder, anchor, support data and residual objective; it directly applied the fitted correction instead of the query-specific decision.

**Results:** At 20 acquired positions, balanced accuracy was 62.25%, 63.83% and 71.94% in the Stieger, Pan and Weibo cohorts, respectively. The 12 archived primary accuracy comparisons had Holm-adjusted p-values of at least 0.5625. ERFC changed 14.37–15.07% of query probabilities by at least 0.0001. In an additional exploratory 18-test analysis, introduced errors were 0.69–0.75 percentage points lower than with a matched point-fit residual (Holm-adjusted p = 0.0039–0.0313). Corrected anchor errors were also 0.78–1.27 percentage points lower.

**Conclusions:** ERFC provides a query-specific rule for translating scarce calibration evidence into prediction changes. The observed correction–preservation trade-off makes the consequences of adaptation measurable alongside aggregate decoding accuracy.

**Keywords:** motor imagery; electroencephalography; brain–computer interface; few-trial calibration; functional uncertainty; kernel methods; negative transfer



## 1. Introduction

A short calibration recording gives a motor imagery decoder an opportunity to adapt to a new EEG session. It also gives the decoder an opportunity to replace a useful prediction with an error. With twenty acquired trials, fitting a correction is straightforward; establishing where that correction should change subsequent predictions is a different problem. We study this distinction through Evidence-Retaining Functional Calibration (ERFC), a method that asks whether the available calibration evidence supports changing each query prediction. Its purpose is to make the intervention itself depend on the information supplied by the calibration recording.

Motor imagery decoding supports EEG-based communication and control, making session-to-session adaptation a practical component of reusable BCIs [@wolpaw2002; @lotte2018]. Session changes alter the relationship between scalp recordings and a learned decision function. Electrode placement, recording conditions and ongoing brain activity contribute to this variation. Compact convolutional decoders provide reusable representations, while geometric alignment can reduce differences between recording distributions [@lawhern2018; @he2020; @rod2019]. These components provide a practical starting point for adaptation: retain the learned representation, estimate a target-session reference from the available support recording, and learn a small residual correction. The important question then moves from how many network weights to update to which future predictions the support data can justify revising.

Few-shot learning, lightweight personalization and learned geometric references address complementary parts of this problem [@an2024darelation; @an2026resml; @huang2026leapnet]. They establish useful ways to reuse representations or adapt to recording variability. However, a compact update is still an estimated update. A small number of fitted coefficients can influence many query trials, including trials whose feature directions are poorly represented by the support recording. Parameter count and support-set fit therefore do not, by themselves, specify how broadly a correction should be applied. For an adaptation method intended to operate after a short chronological prefix, this distinction determines whether observed support evidence is extended to later trials or supplemented by assumptions about unobserved directions.

ERFC addresses that decision at the level of the query correction. The method uses a frozen EEGNet representation and a support-based Euclidean-alignment anchor, preserving the provenance of those established components [@lawhern2018; @he2020]. A regularized logistic residual model defines a finite feasible support fit. Rather than applying that fitted residual to every query, ERFC retains a set of functions that preserve a specified proportion of its improvement over the anchor. For each query, it accounts for both the support-projected correction and a kernel component outside the support span. It then computes an outer interval of compatible probability corrections and makes the smallest change from the anchor permitted by that interval. When the interval contains the anchor prediction, the original probability is retained.

This construction gives a concrete interpretation to adaptation restraint. An unchanged prediction is a defined outcome of the retained evidence set, rather than a failed optimization step or a confidence threshold appended to a classifier. The interval is derived from the chosen regularized objective and kernel geometry; it is not a posterior credible interval. The contribution lies in combining gain retention, query-functional completion and a minimum-change decision for few-trial EEG calibration. Standard kernel machinery supplies the function space and its geometry [@hofmann2008]; the proposed procedure determines how support evidence is translated into an intervention on each subsequent prediction.

We examine this mechanism using matched predictions from three cohorts comprising 35 participants. ERFC and its fitted-residual control share the frozen anchor, support recording and residual objective, isolating the effect of the query decision. The analysis separates errors introduced by adaptation from errors corrected by adaptation, alongside balanced accuracy and the fraction of query probabilities updated. Across the three cohorts, ERFC reduced the class-balanced rate of newly introduced errors relative to the matched point-fit control. It also corrected fewer anchor errors. These paired outcomes define the operating trade-off examined here: using a short calibration recording to constrain where a decoder changes, with the benefit and cost of that restraint measured separately.



## 2. Materials and Methods

### 2.1. Study design and datasets

We conducted a retrospective methodological study of binary left- versus right-hand motor imagery decoding. The evaluation used a historical test partition of the Stieger longitudinal dataset, together with the publicly released Pan and Weibo datasets [@stieger2021; @pan2023; @yi2014]. The analyzed cohorts comprised 11 participants with 105 eligible Stieger sessions, 14 participants with 28 Pan sessions, and 10 participants with 10 Weibo sessions (Table 1). Three source-model seeds produced 315, 84 and 30 session–seed evaluations, respectively, at each acquisition budget. The participant was the independent unit of inference within each cohort; neither sessions nor repeated source seeds increased the participant count.

The archived Stieger allocation contained 40 training-reference participants, 11 development-validation participants and 11 historical test participants. Feature-reference estimation used the training-reference group. The present evaluation cohorts had also been examined during earlier algorithm development, and the analyzed configuration followed a recorded retrospective development amendment. We therefore treat these results as exploratory evidence about the calibration mechanism. The archived accuracy comparisons and the new mechanism analyses are distinguished below; no new independently recruited or prospectively untouched cohort is claimed.

For each target session, the calibration budget was the first 4, 8, 12, 16 or 20 acquired trial positions. Quality-rejected trials still consumed their positions and were not replaced. Consequently, 20 positions did not necessarily supply 20 usable labeled inputs. The query set was fixed at the boundary defined by the largest budget and remained the same at all five budgets. In Stieger, the remainder of the physical run containing the twentieth acquired trial was excluded; evaluation used subsequent complete runs, with at least 20 valid query trials and five per class. Pan and Weibo were evaluated in the trial order supplied with their public releases; their physical run chronology was not independently established. These latter analyses are published-order transfer evaluations.

**Table 1. Evaluation cohorts and repeated measurements.** B20 denotes 20 acquired positions before quality control. Session–seed cells are repeated measurements rather than independent participant units.

| Cohort | Participants | Sessions | Session–seed cells | Mean valid support at B20 |
| --- | --- | --- | --- | --- |
| Stieger historical test | 11 | 105 | 315 | 15.53 |
| Pan | 14 | 28 | 84 | 20.00 |
| Weibo | 10 | 10 | 30 | 20.00 |

### 2.2. Signal processing, source decoder and alignment anchor

The frozen input contract used 11 channels in the order FC3, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3 and CP4. Processing applied common-average referencing, a fourth-order bidirectional Butterworth band-pass filter at 8–30 Hz within each original trial's fixed 0–1.5 s task-relative context, and polyphase resampling to 128 Hz. The analyzed window extended from 0.25 to 1.25 s after the task anchor, giving an 11 × 128 input. For Stieger, the verified task/feedback anchor was 2 s after cue onset. Each channel was demeaned over time and the complete trial was divided by its joint channel–time root-mean-square amplitude. Nonfinite, zero-energy or insufficient-duration inputs were rejected by the fixed quality rules. Filtering and evaluation were offline.

The existing source decoder used the established EEGNet architecture [@lawhern2018]. It contained eight temporal filters, a depthwise spatial stage with 16 output channels, a depthwise temporal and pointwise stage, ELU nonlinearities and temporal pooling, producing 64 features before a two-logit classifier. Source training used BNCI2014_001, Cho2017, Lee2019_MI and PhysionetMI [@tangermann2012; @cho2017; @lee2019; @goldberger2000]. The three checkpoint seeds were 20261018, 20261019 and 20261020. Their source-only selected training lengths were 39, 30 and 45 epochs, respectively, with 64 steps per epoch and batch size 64. Adam used a learning rate of 0.001, β = (0.9, 0.999), ε = 10⁻⁸ and zero weight decay; dropout during source training was 0.25. Sampling proceeded by source dataset, participant, observed class and trial with replacement. All weights, batch-normalization states and dropout settings were frozen for target calibration. Checkpoint hashes and the detailed input contract are supplied in Supplementary Table S4 and the provenance files.

The principal anchor was support-based covariance alignment, a source-colored adaptation of Euclidean alignment [@he2020; @rod2019]. Only the acquired support inputs were used. In the ten-dimensional effective common-average-reference space, individual support covariances were trace normalized, averaged and shrunk by 0.95 times the mean plus 0.05 times the isotropic covariance. The resulting transform whitened the support covariance and recolored it to the frozen source reference. It used no target labels. The frozen network then supplied anchor logits and features for both ERFC and its matched residual control.

Feature-reference means and whiteners were reused from the same sealed training-side artifacts for all matched methods. They were constructed from 40 Stieger training-reference participants and 396 sessions using valid inputs within each session's first 20 acquired positions. Participants, their sessions and the valid prefix trials within each session were weighted equally at their respective hierarchy levels. The reference feature covariance used fixed 0.1 shrinkage toward an isotropic covariance. Query inputs did not update this reference geometry.


### 2.3. A frozen predictor and a support-only calibration interface

Evidence-Retaining Functional Calibration (ERFC) asks whether the information in a small calibration set supports changing a particular future prediction. Its adaptation object is a residual prediction function. The EEGNet encoder, original classifier, batch-normalization statistics and dropout state remain frozen [@lawhern2018]. The principal anchor applies Euclidean alignment using a covariance transform estimated from the available calibration inputs [@he2020]. Each target episode supplies a support set \(S=\{(x_i,y_i)\}_{i=1}^{n}\), with binary labels and \(n\leq20\). The chronological acquisition budget and quality-control rules are specified in the experimental protocol.

Write the anchor margin as \(m_0(x)\) and its class-1 probability as \(p_0(x)=\sigma(m_0(x))\), where \(\sigma(t)=(1+e^{-t})^{-1}\). A common probability floor of \(10^{-12}\) defines the numerical anchor for the proposed method, matched controls and fallbacks. The residual uses the same binary logistic link as the anchor and conditions on frozen features.

The fitting interface receives support features, anchor logits, labels and a fixed configuration. Prediction receives the resulting state, query features and query anchor logits. It has no query-label argument. No query-batch statistic is estimated: conditional on the supplied frozen features and logits, a query's decision depends on its support set and its own feature vector.

### 2.4. A fixed residual function class

Let \(\phi(x)\in\mathbb R^d\) denote a frozen feature vector. A reference mean \(\mu_{\mathrm{ref}}\) and whitener \(W_{\mathrm{ref}}\), established from training-side reference data, define

\[
z(x)=(\phi(x)-\mu_{\mathrm{ref}})W_{\mathrm{ref}},\qquad
k(x,x')=\tau^2\left[d^{-1}z(x)^\top z(x')+1\right].

\]

(1)


This reference transform is reused during calibration and prediction. The linear kernel and intercept define the residual reproducing-kernel Hilbert space \(\mathcal H_k\) [@aronszajn1950; @hofmann2008]. They are fixed modeling components; ERFC's decision mechanism concerns which support-compatible residuals justify a query correction.

For numerical definiteness, the implemented kernel includes an explicit event-identity nugget,

\[
k_\epsilon(x,x')=k(x,x')+\epsilon\,
\mathbf1\{\operatorname{UID}(x)=\operatorname{UID}(x')\}.

\]

(2)


Support identifiers are unique and disjoint from future-query identifiers. Thus the support Gram matrix has \(\epsilon I\), support–query cross-kernels have no identity increment, and the query diagonal includes \(\epsilon\). All Cholesky directions are retained. The nugget is part of the declared function class, not additional observations.

### 2.5. Retaining an observed calibration gain

Let \(n_c\) be the number of support samples with class \(c\). When both classes are represented, weights \(w_i=n/(2n_{y_i})\) satisfy \(\sum_iw_i=n\). ERFC minimizes the class-balanced logistic objective

\[
J(f)=\sum_{i=1}^{n}w_i\left[
\operatorname{softplus}(m_{0i}+f(x_i))
-y_i(m_{0i}+f(x_i))\right]
+\frac{\lambda}{2}\|f\|_{\mathcal H_{k_\epsilon}}^2.

\]

(3)


The implementation obtains a finite feasible estimate \(\widehat f\) with \(\widehat J=J(\widehat f)\leq J_0=J(0)\). Define its observed regularized support gain as \(G=J_0-\widehat J\). Exact global convergence is not required to define this gain. With a fixed allowance \(c\in[0,1]\), the retained-function set is

\[
T=\widehat J+cG,\qquad
\mathcal C_T=\{f\in\mathcal H_{k_\epsilon}:J(f)\leq T\}.

\]

(4)


Every member retains at least \((1-c)G\) of the observed gain. The analyzed configuration uses \(c=0.1\), retaining at least 90% of that gain. The set expresses compatibility with the measured support objective; it is a deterministic model set rather than a posterior credible region.

### 2.6. Completing a prediction beyond the support span

The retained set includes functions that agree on all support points but differ at a query. This distinction is essential when a small support set leaves feature directions unresolved. Let \(K_{SS}=BB^\top\) be the Cholesky factorization of the nugget-augmented support Gram matrix. For a query \(q\), define

\[
a_q=B^{-1}k_{Sq},\qquad
s_q=k_\epsilon(q,q)-\|a_q\|_2^2.

\]

(5)


The Schur innovation \(s_q\geq0\) is the squared norm of the query representer orthogonal to the support span. The standard RKHS decomposition [@aronszajn1950; @hofmann2008] reduces the extremal query problem to

\[
f_S=Bv,\quad f(q)=a_q^\top v+\sqrt{s_q}\,t,\quad
\|f\|^2=\|v\|_2^2+t^2.

\]

(6)


Only the orthogonal component aligned with the query representer is needed at an extremum. If \(F(v)\) denotes Equation (3) with support residuals \(Bv\), the query range follows from minimizing or maximizing \(a_q^\top v+\sqrt{s_q}t\) subject to \(F(v)+\lambda t^2/2\leq T\). The extra scalar \(t\) represents support-invisible completion, not a parameter learned from query labels.

A direct witness can settle the decision before endpoint computation. Let \(\widehat v\) be the feasible fitted coefficient and \(\mu_q=a_q^\top\widehat v\). For \(s_q>0\), choosing \(t=-\mu_q/\sqrt{s_q}\) preserves every fitted support residual while making the query correction zero. Its additional objective cost is \(\lambda\mu_q^2/(2s_q)\). Therefore,

\[
\frac{\lambda\mu_q^2}{2s_q}\leq cG
\quad\Longrightarrow\quad
0\in\{f(q):f\in\mathcal C_T\}.

\]

(7)


This is a sufficient querywise no-correction certificate. A positive innovation need not always prevent adaptation: its completion cost is assessed against the retained objective allowance. Conversely, \(s_q=0\) removes this particular orthogonal completion direction but does not make every retained coefficient vector identical. Ambiguity among support-compatible values of \(v\) is still assessed by the objective profile. The no-completion control sets the innovation term to zero while preserving the fitted support state, separating this mechanism from a change in the residual fit.

### 2.7. Conservative outer bounds from finite optimization

For a direction \(h\in\{a_q,-a_q\}\) and tilt \(\zeta>0\), consider \(\psi_{h,\zeta}(v)=F(v)-\zeta h^\top v\). Its Hessian is

\[
\nabla^2F(v)=B^\top\operatorname{diag}
\{w_i p_i(v)[1-p_i(v)]\}B+\lambda I
\succeq\lambda I.

\]

(8)


Strong convexity converts any finite Newton iterate \(\widetilde v\) into a lower bound on the tilted minimum. With \(g=\nabla\psi_{h,\zeta}(\widetilde v)\),

\[
M_{h,\zeta}=\psi_{h,\zeta}(\widetilde v)
-\frac{\|g\|_2^2}{2\lambda}
\leq\inf_v\psi_{h,\zeta}(v).

\]

(9)


Convex duality [@boyd2004] then gives

\[
\widehat U_h(\zeta)=\frac{T-M_{h,\zeta}}{\zeta}
+\frac{\zeta s_q}{2\lambda}
\geq\sup_{F(v)+\lambda t^2/2\leq T}
\{h^\top v+\sqrt{s_q}t\}.

\]

(10)


The residual term preserves the real-arithmetic outer-bound direction without treating a convergence flag as an exact optimum. Taking the smallest valid bound over evaluated positive tilts remains conservative. Doubling and bisection locate informative tilts within fixed computational budgets; their role is to tighten a valid bound.

The deployed decision needs one direction. Set \(d_q=\operatorname{sign}(\mu_q)\), evaluate \(h=-d_qa_q\), and form

\[
e_q=-\min_{\zeta\in\mathcal Z}\widehat U_{-d_qa_q}(\zeta),
\qquad
\delta_q=d_q\max(0,e_q).

\]

(11)


A positive \(e_q\) establishes that every retained function corrects in the fitted direction. A nonpositive bound leaves the anchor unchanged. Because \(\widehat f\) is feasible, a bound demanding a correction larger than \(|\mu_q|\) is inconsistent and triggers an anchor return. Computation stays in support-sized systems; the encoder is not optimized per query. This construction also separates decision conservatism from uncertainty magnitude alone. A large completion direction can veto a correction through Equation (7), whereas a tighter objective profile can support a nonzero update in the same function class. The deciding quantity is the nearest permitted query correction relative to zero, not a standalone variance score or the size of the fitted MAP correction.

### 2.8. Projecting the anchor probability

For the ideal residual range \([\ell_q,u_q]\), define

\[
I_q=[L_q,U_q]=[\sigma(m_0(q)+\ell_q),
\sigma(m_0(q)+u_q)].

\]

(12)


The anchor-relative decision under cross-entropy, a strictly proper scoring rule [@gneiting2007], has the closed form

\[
\arg\min_p\sup_{\eta\in I_q}
\{\operatorname{CE}(\eta,p)-\operatorname{CE}(\eta,p_0(q))\}
=\operatorname{clip}(p_0(q),L_q,U_q).

\]

(13)


To see this, regret is affine in \(\eta\). When \(p>p_0\), its worst case occurs at \(L_q\); when \(p<p_0\), it occurs at \(U_q\). Minimizing each branch gives projection onto the interval. If the anchor lies inside, zero regret is attainable by retaining it. Outside, the worst regret at the projected endpoint equals the negative Bernoulli Kullback–Leibler divergence from that endpoint to the anchor and is nonpositive.

The one-sided implementation uses an outer interval with an unbounded far endpoint, yielding Equation (11) followed by \(p(q)=\sigma(m_0(q)+\delta_q)\). Thus a loose valid bound yields a smaller correction. This conditional loss statement concerns probabilities represented by the retained set. Membership of the true future conditional probability is an empirical question; the construction does not supply future-accuracy or coverage guarantees.

### 2.9. Fixed configuration and anchor recovery

The recorded candidate `a0.3-c0.1` uses a linear-plus-intercept kernel with \(\tau=0.3\), \(\lambda=1\), \(c=0.1\) and \(\epsilon=10^{-10}\). CPU float64 solves use at most 60 Newton iterations, 32 bracket steps and 12 bisection steps; gradient and root tolerances are \(10^{-10}\) and \(10^{-7}\). Fixed roundoff padding is \(10^{-10}\). This implements the real-arithmetic bounds numerically, without a formal floating-point interval-arithmetic proof.

Empty, single-class or fewer-than-two-sample supports return the common anchor. Zero kernel amplitude, negligible support gain, zero query cross-information, the witness in Equation (7), invalid joint-kernel geometry or unavailable finite bounds also preserve it. At \(c=1\), the zero residual belongs to \(\mathcal C_T\), so all queries recover the anchor. The matched MAP control uses the same kernel and fitted state but directly applies \(\mu_q\); it removes the query-functional retention decision. The support factor requires \(O(n^2)\) storage. Each Newton step solves an \(n\)-dimensional positive-definite system, and query profiling reuses the support features and factorization. Tilt iterations are temporary computations of an extremum; they do not update the stored fitted state. Numerical diagnostics record the support gain, innovation, witness cost, solver residuals, selected tilt and applied correction for each query. These controls isolate the mechanism that distinguishes ERFC: adaptation requires support evidence that survives the permitted completion of the queried prediction.




![](figures/Fig1_ERFC_framework.png){width=95%}

**Figure 1. Evidence-Retaining Functional Calibration (ERFC) and the acquired-prefix evaluation design.** (A) The acquired calibration prefix supplies the support-based Euclidean-alignment (EA) transform, frozen EEGNet features and anchor logits. A regularized logistic residual objective defines the matched maximum-a-posteriori (MAP) fit and a retained-function set. The selected allowance c = 0.1 retains at least 90% of the observed regularized support-objective gain. (B) Each unlabeled query is profiled using its support projection and the permitted completion outside the support span. The nearest-to-zero correction within a conservative outer interval is applied to the anchor logit; intervals containing zero retain the anchor. Source weights, reference geometry and fitted support state remain fixed during prediction. The construction describes a conditional objective-compatible correction set. (C) Prefix budgets of 4, 8, 12, 16 and 20 acquired positions share the same subsequent query trials. The first four positions are shaded as an illustrative prefix. Pan and Weibo are replayed in published trial order, which does not establish verified physical chronology. The three historical evaluation cohorts comprise 11 Stieger participants/105 sessions, 14 Pan participants/28 sessions and 10 Weibo participants/10 sessions. The diagram is original and schematic; no electrophysiological measurements are illustrated.

### 2.10. Comparators and recorded configuration selection

The archived stage evaluated 18 methods in each of three cohorts and three source seeds, plus three feature-reference tasks, giving 165 completed tasks. Each evaluation task internally covered all five acquisition budgets. Comparators included the frozen decoder, support alignment, bias-only and rank-four or full-head calibration, spatial SGD with 10 or 100 updates, earlier protected-spatial and cross-fitted methods, bounded geometric adaptation, anchored kernel residuals, and the ERFC ablations. Supplementary Table S4 maps every displayed name to its archived implementation key.

The central matched control, referred to here as matched MAP, used the same support inputs, anchor, frozen features, residual kernel, regularization and support-fit procedure as ERFC. It applied the resulting point-fit residual directly. Further ablations removed support-null completion, changed the anchor, used a quadratic profile approximation or replaced the functional decision with a variance-based gate. Configuration selection used development-validation data; the analyzed `a0.3-c0.1` choice and retrospective amendment are retained in the released provenance. No evaluation-query labels entered an episode's residual fitting or prediction interface.

### 2.11. Outcomes and participant-level statistical analysis

Balanced accuracy (BA) was the equally weighted mean of the two class recalls. Balanced negative log-likelihood (NLL) gave each class equal total weight. For each method, we separately measured two transitions relative to the same alignment anchor: an introduced error was an anchor-correct query made incorrect by adaptation; a corrected error was an anchor-incorrect query made correct. The class-balanced rates, denoted I and C, satisfy the identity BA(method) − BA(anchor) = C − I. We verified this identity in the participant summaries used for the present figures.

The query-update fraction counted queries whose class-1 probability changed from the alignment anchor by at least 10⁻⁴. It was distinct from the fraction of predicted class labels that changed. A session was classified as negative transfer when its BA was below its matched anchor BA by more than 10⁻¹². The negative-transfer session fraction was first summarized within participant and source seed.

Session scores were averaged equally within participant and seed, then source seeds were averaged equally within participant. Cohort means and paired inference were calculated over the resulting participant values. The archived main BA family comprised four ERFC contrasts in each cohort—alignment, spatial SGD100, the earlier EA-anchored residual and matched MAP—for 12 comparisons. Its two-sided exact sign-flip tests and 10,000 participant-bootstrap intervals were preserved, including the original statistics seed 20261026. Holm adjustment was applied within that 12-test family [@holm1979]. Sign-flip inference assumes exchangeability of the signs of participant-level contrasts under its null.

For this manuscript, we additionally analyzed the matched-MAP contrast for six outcomes in each cohort: BA, balanced NLL, introduced errors, corrected errors, query-update fraction and negative-transfer session fraction. These 18 tests were explicitly post hoc and exploratory. We used exact two-sided participant sign flips and 10,000 percentile bootstrap resamples of participant pairs, with statistics seed 20261029. Holm adjustment was applied across all 18 tests. Confidence intervals were pointwise rather than simultaneous. Effects on accuracy and event fractions are reported in percentage points; NLL effects are in nats. The two families were not combined or relabeled after observing their results. The full primary and exploratory families are provided in Supplementary Tables S1 and S2.

### 2.12. Reproducibility and computational measurements

The analyzed reports, participant rows and source-model identities were restored from immutable manifests and checked against their recorded SHA-256 hashes. The ERFC algorithm source matched the code manifest of the completed stage. The accompanying analysis script regenerates the new paired statistics; the plotting script reads its saved numerical tables. No additional model training was performed to prepare this manuscript. ChatGPT (OpenAI) assisted with literature synthesis, manuscript drafting and preparation of analysis and plotting code. The numerical calculations were executed by Python scripts using the archived participant data; tool details are disclosed in the Acknowledgments. The evidence files and executable scripts identify the provenance of numerical results.

Residual fit time was recorded on already extracted support features. Incremental prediction time measured the additional residual/profile evaluation on already extracted query features, scaled to 1000 queries. These measurements excluded support-alignment estimation, neural feature extraction and input/output operations. They are reported as component measurements, with no end-to-end real-time claim.

## 3. Results

### 3.1. Accuracy at the twenty-position calibration budget

The eligible cohort counts and acquisition-budget definition are summarized in Table 1. At 20 acquired positions, Stieger supplied a mean of 15.53 valid support inputs after fixed quality control, whereas Pan and Weibo supplied 20. ERFC achieved participant-mean BA of 62.25%, 63.83% and 71.94% in Stieger, Pan and Weibo, respectively (Table 2). The matched alignment anchors achieved 62.21%, 63.82% and 71.91%. The paired ERFC-minus-alignment effects were +0.041 percentage points (95% CI −0.017 to +0.105), +0.012 (−0.095 to +0.155) and +0.024 (−0.071 to +0.143). All three adjusted p-values were 1.0 in the archived 12-test accuracy family.

The matched MAP control achieved BA of 62.29%, 64.11% and 72.52%. ERFC-minus-MAP accuracy differences were −0.038, −0.274 and −0.583 percentage points. None of the 12 archived main accuracy comparisons passed Holm correction; the smallest adjusted p-value was 0.5625. Thus, the accuracy results characterize the operating point rather than establish superiority or equivalence. Figure 2 shows the complete calibration-budget curves and the paired alignment contrasts; the original contrasts and intervals are retained in Supplementary Table S1.

**Table 2. Participant-mean balanced accuracy (%) at 20 acquired positions.** The alignment anchor, matched MAP and ERFC share their source representation. This table reports means; the full 12 paired primary tests and confidence intervals appear in Supplementary Table S1.

| Cohort | Support alignment | Spatial SGD100 | Anchored residual | Matched MAP | ERFC |
| --- | --- | --- | --- | --- | --- |
| Stieger | 62.21 | 61.96 | 62.20 | 62.29 | 62.25 |
| Pan | 63.82 | 63.60 | 64.11 | 64.11 | 63.83 |
| Weibo | 71.91 | 72.59 | 72.54 | 72.52 | 71.94 |



![](figures/Fig2_accuracy_and_budget.png){width=95%}

**Figure 2. Balanced accuracy across acquired-prefix budgets and paired anchor contrasts.** (A–C) Cohort means for ERFC, the same-objective matched MAP control and support EA at 4–20 acquired prefix positions, with future queries held fixed. Values are participant means after averaging eligible sessions within source seed and then the three source seeds within participant. Prefix positions are acquisition budgets, not guaranteed numbers of valid support samples after quality control. Curves show means without uncertainty bands; the paired accuracy distributions at B20 are shown in Figure 4B. (D) The archived primary ERFC-minus-EA balanced-accuracy contrasts at B20, in percentage points, with original participant-level pointwise 95% bootstrap intervals. The corresponding Holm-adjusted p values are 1.0 within the 12-comparison primary accuracy family specified in the archived exploratory plan. Cohort-specific axes in panels A–C display the absolute accuracy levels; their scales differ.

### 3.2. Query-selective application of the fitted correction

ERFC changed the anchor probability by at least 10⁻⁴ for 15.07%, 14.37% and 14.48% of queries in Stieger, Pan and Weibo. Relative to matched MAP, its query-update fraction was lower by 83.95, 84.80 and 84.83 percentage points. These paired reductions passed Holm correction in the exploratory 18-test family (Figure 3C). The result distinguishes the fitted support function from the decision to apply its correction: a common residual fit led to substantially different intervention footprints.

The archived ERFC participant–budget summaries contained no recorded numerical-fallback or certificate-violation events across the 175 participant–budget combinations. This diagnostic count refers to the analyzed runs. The retained-function set and the support-null witness in Figure 1 specify why an unchanged prediction is a valid result of the algorithm.

### 3.3. Introduced errors and recovered anchor errors

At 20 positions, the class-balanced introduced-error rate relative to the alignment anchor was 0.053%, 0.060% and 0.048% for ERFC. In the exploratory matched-MAP analysis, the corresponding paired reductions were −0.747 percentage points for Stieger (95% CI −0.861 to −0.640; adjusted p = 0.0137), −0.714 for Pan (−0.917 to −0.512; adjusted p = 0.0039) and −0.690 for Weibo (−1.000 to −0.381; adjusted p = 0.0313). Table 3 and Figure 3A report these effects with the complete multiplicity context.

ERFC also repaired fewer anchor errors. Its class-balanced correction rates were 0.094%, 0.071% and 0.071%, and the paired ERFC-minus-MAP differences were −0.785, −0.988 and −1.274 percentage points, respectively. All three reductions in corrected-error opportunities passed the same exploratory multiplicity correction. Figures 3A and 3B present the two transitions together. Their difference accounts for the paired net accuracy change; reducing introduced errors did not imply that more anchor errors were corrected.

**Table 3. Additional exploratory matched-control effects.** Negative introduced-error effects denote fewer new errors; negative corrected-error effects denote fewer repaired anchor errors. Intervals are pointwise 95% participant-bootstrap intervals. Holm adjustment covers the complete 18-test post hoc family, not just the nine effects displayed here.

| Cohort | Outcome | ERFC − MAP (pp) | 95% CI (pp) | Holm p (18) |
| --- | --- | --- | --- | --- |
| Stieger | Introduced errors | -0.747 | [-0.861, -0.640] | 0.0137 |
| Stieger | Corrected errors | -0.785 | [-0.932, -0.628] | 0.0137 |
| Stieger | Probability updates | -83.950 | [-86.794, -80.769] | 0.0137 |
| Pan | Introduced errors | -0.714 | [-0.917, -0.512] | 0.0039 |
| Pan | Corrected errors | -0.988 | [-1.167, -0.810] | 0.0022 |
| Pan | Probability updates | -84.798 | [-91.226, -77.857] | 0.0022 |
| Weibo | Introduced errors | -0.690 | [-1.000, -0.381] | 0.0313 |
| Weibo | Corrected errors | -1.274 | [-1.750, -0.845] | 0.0195 |
| Weibo | Probability updates | -84.825 | [-90.040, -79.444] | 0.0195 |



![](figures/Fig3_selectivity_tradeoff.png){width=95%}

**Figure 3. Selective correction trades newly introduced errors against corrected anchor errors.** Participant-paired ERFC-minus-matched-MAP effects at B20 are shown for (A) the class-balanced rate of errors introduced relative to EA, (B) the class-balanced rate of EA errors corrected and (C) the fraction of query probabilities updated by at least 10⁻⁴ in absolute class-1 probability. Points are cohort means; bars are pointwise 95% percentile bootstrap intervals from participant resampling. All effects are in percentage points. Negative values in panel A denote fewer newly introduced errors; negative values in panel B denote fewer corrected errors. Panel C applies the recorded threshold |p₁ − p₁,EA| ≥ 10⁻⁴; it does not count every nonzero numerical correction and does not require a changed class decision. The displayed pH values are two-sided participant sign-flip p values adjusted by Holm over the 18 tests in the newly conducted exploratory matched-control analysis. Stieger, Pan and Weibo have 11, 14 and 10 independent participant units, respectively. The two error-transition rates are shown together because their difference determines the net balanced-accuracy change.

### 3.4. Participant variation and session-level negative transfer

The participant-level distributions show that ERFC and matched MAP occupied different update–error operating regions (Figure 4). With ERFC, the participant-averaged fraction of sessions below alignment-anchor BA was 3.46% in Stieger, 3.57% in Pan and 3.33% in Weibo. Paired reductions relative to MAP were 27.04, 17.86 and 13.33 percentage points, respectively. The first two passed Holm correction in the exploratory family, while the Weibo contrast did not. Individual participant values and the complete tests are available in Supplementary Table S2 and its machine-readable companion.

Balanced NLL at the same budget was 0.6465, 0.6301 and 0.5514 nats for ERFC. The full matched NLL comparisons are reported alongside BA in Supplementary Table S2. The paired results therefore describe an explicit correction–preservation trade-off across probability, classification and session-level measures.



![](figures/Fig4_participant_tradeoffs.png){width=95%}

**Figure 4. Participant-level operating trade-offs of query-selective correction.** (A) Each point relates the fraction of query probabilities updated by at least 10⁻⁴ in absolute class-1 probability to the class-balanced introduced-error rate for one participant at B20. Circles denote ERFC and triangles the matched MAP control. Blue, green and magenta indicate Stieger, Pan and Weibo, respectively. (B) Participant-paired balanced-accuracy differences and (C) participant-paired differences in the fraction of sessions with balanced accuracy below the corresponding EA anchor. Colored points represent individual participants; the small vertical offsets separate overlapping observations and have no quantitative meaning. Black diamonds and horizontal bars show the paired cohort means and their pointwise 95% participant-bootstrap intervals from the same exploratory analysis as Figure 3. Differences are ERFC minus matched MAP, in percentage points. Source seeds and sessions are averaged within participant and do not increase the independent sample size.

### 3.5. Component computation

Mean residual fitting times on extracted support features were 1.75 ms, 1.07 ms and 1.10 ms in Stieger, Pan and Weibo. The additional query-profile calculation required 1.58, 1.56 and 1.66 s per 1000 already encoded queries. Querywise profiling therefore added computation beyond direct point-fit prediction while retaining support-sized linear systems. Supplementary Table S3 reports all methods and budgets using the same component-timing definitions.


## 4. Discussion


### 4.1. Adaptation as an evidence-dependent intervention

ERFC turns few-trial calibration into a decision about where to intervene. The fitted residual identifies a direction of improvement on the support recording, whereas the query profile asks whether that evidence requires a change at a particular query. These are separate operations. Their separation matters when the number of acquired trials is small relative to the feature representation: a support fit may be precise along observed directions while leaving substantial freedom elsewhere. Retaining the anchor whenever the outer interval contains it gives that freedom an explicit consequence in the output.

The empirical comparison with the matched point-fit control supports this mechanism. The class-balanced rate of errors newly introduced relative to the anchor was lower by 0.747, 0.714 and 0.690 percentage points in the three cohorts. The corresponding Holm-adjusted exploratory p-values were 0.0137, 0.0039 and 0.0313. The same residual objective produced these different outcomes depending on whether its point estimate was applied directly or filtered through the evidence-retaining query decision. This contrast locates the observed effect in how the fitted information was used, rather than in a different encoder, a larger support set or an additional source-training stage.

### 4.2. The measurable cost of preserving the anchor

The reduction in introduced errors was accompanied by fewer corrected anchor errors: differences of 0.785, 0.988 and 1.274 percentage points relative to the point-fit control. Both quantities are needed to interpret the result. A broad correction has more opportunities to repair an anchor error and more opportunities to disrupt an initially correct prediction. ERFC intervenes on a smaller set of queries. That choice can be useful when preserving existing decisions is an explicit operating objective, but it does not make the cost of retaining an incorrect anchor disappear.

Balanced accuracy records the net classification outcome of these two changes. ERFC reached 62.25%, 63.83% and 71.94% in the three cohorts, and none of the twelve primary balanced-accuracy comparisons provided positive evidence after the specified multiplicity correction. The empirical contribution is therefore the characterized correction–preservation trade-off, not a claim of superior classification accuracy. Nor does a nonsignificant accuracy difference establish equivalence. Reporting introduced and corrected errors alongside balanced accuracy shows why two adaptation rules can have close aggregate scores while changing different predictions. The choice between them requires an explicit valuation of those changes, rather than treating one summary metric as a complete description of adaptation behavior.

### 4.3. Why query geometry changes the decision

In this implementation, ERFC updated 14.37–15.07% of query probabilities, compared with approximately 99% for the point-fit control. These fractions count class-1 probability changes of at least 10⁻⁴ from the alignment anchor, not the proportion of class labels that changed. Their separation is informative: an update may alter confidence without crossing a decision boundary, and a small probability correction can change the predicted class when the anchor lies near that boundary. The paired introduced-error analysis identifies the classification consequence that an update count alone cannot establish.

The support-null component provides a structural reason for retaining an anchor prediction. A query can contain a feature-space component that the support observations do not constrain directly. Omitting that component treats support projection as a complete account of the query correction. ERFC instead includes compatible continuations under the chosen function-space penalty. When such a continuation supports the unchanged probability, the minimum-change rule retains it. This argument concerns compatibility with the regularized support objective. It does not assign a probability to the true label or certify the future error rate. A useful next evaluation is to examine how the decision changes as acquired support expands the observed feature directions, keeping the acquisition order and all evaluation trials fixed.

### 4.4. A defined role for reusable representations

The frozen decoder and alignment reference are deliberate parts of the design. EEGNet supplies an established compact representation, and Euclidean alignment supplies an established way to construct a session reference [@lawhern2018; @he2020]. ERFC adds a functional rule for using the resulting support evidence; it does not claim those components as new architectures. Keeping them fixed also makes the matched comparison interpretable. A change in the introduced-error rate can be examined without simultaneously attributing it to representation learning, network capacity or a different normalization procedure.

The same reasoning identifies an important condition for this operating point: the anchor must be worth preserving. When its predictions are broadly inappropriate for the new recording, retaining them more often may prevent corrections that the application needs. Conversely, when a reusable decoder already provides useful decisions and calibration is brief, selective intervention offers a way to expose the cost of adaptation. This condition can be tested through controlled anchor quality, rather than inferred from the method name or assumed from the presence of an interval. The current study evaluates prediction-level behavior, not the clinical utility or patient safety of a BCI system.

### 4.5. Interpretation and the next validation step

The three cohorts had prior development exposure, and the mechanism analysis was exploratory. They provide repeated observations of the proposed operating behavior, not three independent confirmations of a final performance claim. Participant-level paired analyses address variation between people; additional sessions and repeated seeds do not create new independent participants. The next confirmatory study should freeze the complete method, its operating point and the introduced-versus-corrected error endpoints before examining a new cohort. It should preserve chronological acquisition and compare the paired predictions generated from the same support recording. That design would test the central question directly: whether an evidence-dependent restriction on intervention produces a useful trade-off under the intended calibration budget.



## 5. Conclusions


Evidence-Retaining Functional Calibration makes the extent of EEG adaptation depend on the support evidence available for each query. It combines a gain-retaining function set, support-null completion and a minimum-change probability decision around a frozen decoder. Across three cohorts, the resulting operating point introduced fewer errors than a matched point-fit correction and also repaired fewer anchor errors. This explicit trade-off provides a focused basis for evaluating few-trial calibration: identify when changing a prediction is supported, and measure what is gained and relinquished by that intervention. The contribution is a defined, inspectable rule for translating a short calibration recording into prediction changes.




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
