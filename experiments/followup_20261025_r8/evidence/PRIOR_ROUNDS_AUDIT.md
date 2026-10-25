# Round 8: evidence audit of prior experiments

Audit date: 2026-10-25 UTC. This document is a read-only interpretation of published summaries and local design code; no school job was submitted, stopped, or modified. Actual numerical result summaries were fetched from the public repository, not inferred from launch success. Full per-trial archives are still being uploaded separately and have not been reanalysed here.

## Source evidence and freshness

Repository: `jackzhu119/research3`.

| Round | Result snapshot | Public source / observed branch head | Evidence scope |
|---|---|---|---|
| 2 | 2026-10-25 02:16:55 UTC | `paper3-v2-progress:monitoring/paper3-followup-v2/status.json`, head `4706017ef85ff2fcf702e55da9258fd101fdbbf0` | 66/66, report COMPLETE, core complete, report_stale=false |
| 3 | 2026-10-25 00:09:10 UTC | `paper3-r3-progress:monitoring/paper3-round3/status.json`, head `b56484bff67fd593b70e972a9e4c62d9d44d98aa` | 174/174; COMPLETE; 10 primary comparisons; all three seeds |
| 7 | 2026-10-24 19:25:04 UTC | `paper3-r7-progress:monitoring/paper3-round7/status.json`, head `f6804b9a72ff118b732643e6b8e05b9f0f6c3cc6` | Top-level count 20/273 conflicts with COMPLETE report, 12 primary BA comparisons and 24 joint BA/NLL comparisons, all three seeds. User's more recent school readout says 273/273. Treat the top-level count as inconsistent telemetry, not the current training count. |

Raw fetched summaries and SHA-256 provenance are stored beside this file: `paper3-round8-evidence-r2.json`, `...-r3.json`, `...-r7.json`, `paper3-round8-evidence-provenance.json`. They contain status and aggregated statistics, not full trial/checkpoint backups. A branch head was observed separately from its raw read; the SHA-256 records the exact bytes actually analysed. No claim of atomic multi-branch snapshot is made.

R1 is still running its native campaign. R4 was running; R5/R6 newly released jobs were queued in the latest supplied school evidence. Their absence of final comparative conclusions must not be replaced by guessed performance.

## What each prior round tested

All rounds freeze the source EEGNet and source batch-normalization state. The central target task is adaptation from the new session's first 20 paid acquired positions, then predicting the fixed future query set. Invalid/QC-rejected prefix positions consume budget. The common signal contract is clinic11 channels, CAR effective dimension 10, 128 Hz, 8–30 Hz, fixed context/crop and pre-RMS cache. R2–R7 use three pipeline seeds 20261018/19/20.

| Round | Main scientific variation | Logical tasks / actual training |
|---|---|---|
| 1 | Original source/pilot/classical/reference and low-rank Lie spatial meta-learning campaign, screening/selection, PCT and existing extension arms | Ongoing native campaign; its phase counts are not whole-experiment percentage |
| 2 | Symmetric/skew support-conditioned gates; soft teacher reliability; future physical-run tilted risk | 66 tasks: 60 meta-fits plus 6 reference evaluations |
| 3 | Exact identity tangent derivatives; finite convex logistic/Schur fitting; head competition; information radii; real-prediction KL protection | 174 tasks: 21 dictionary tasks, 78 core, 72 sensitivity, 3 reference |
| 4 | Prefix-only out-of-fold shrinkage/stacking and one-SE candidate choice; robust-support comparisons | 204 tasks: 21 dictionary, 90 core, 90 sensitivity, 3 reference |
| 5 | Noncommuting joint/KS/SK/symmetric spatial maps, common-tangent and matched-dictionary controls | 105 tasks: 12 dictionary, 48 core, 39 sensitivity, 6 reference |
| 6 | Schur curvature shaped constrained prediction averaging, conditional head compensation, correlated versus independent point rules | 198 tasks: 6 dictionary, 60 core, 48 sensitivity, 12 reference, 72 Pan transfer |
| 7 | Energy-preserving finite-window spatiotemporal Lie operator; head capacity/identifiability; relative versus absolute entropic risks | 273 tasks: 15 dictionary, 72 core, 72 sensitivity, 21 reference, 93 Pan transfer |

The detailed local sources are `/workspace/paper3-v2-design/DESIGN_ZH.md`, `/workspace/paper3-round3/DESIGN_ZH.md` through `/workspace/paper3-round7/DESIGN_ZH.md`, plus each `STATISTICS.md` and the R7 `RESEARCH_OPTIONS_ZH.md`.

## Actual returned findings that should change Round 8

### 1. R2 changes are nearly prediction-equivalent

The 11-participant internal paired summaries show:

| Comparison | Δ balanced accuracy, percentage points | Holm p |
|---|---:|---:|
| dual − P | -0.00806 | 1.0 |
| dual − scalar | -0.00287 | 1.0 |
| softPCT − PCT | +0.00161 | 1.0 |
| temporal − P | 0.00000 | 1.0 |
| temporal − runmean | 0.00000 | 1.0 |
| dual_temporal − dual | 0.00000 | 1.0 |

P averages 61.4911% BA, dual 61.4830%. The existing support covariance-alignment reference reaches 62.2132%; P minus support_EA is -0.7222 points (descriptive comparison). P is only +0.3906 points above frozen f0 and +0.1680 above head-only. These summaries do not prove the gate gradients were exactly zero or the transforms identical; they show no useful aggregate hard-label improvement in the tested settings. Inspect actual transform magnitude, prediction displacement and gate saturation before interpreting a new architecture's nominal complexity as expressivity used in practice.

### 2. R3 stronger support fitting makes future prediction worse

| Method / comparison | Internal BA or ΔBA | Holm p |
|---|---:|---:|
| legacy SGD10 | 61.4911% | — |
| legacy SGD100 | 61.9634% | — |
| tangent_profiled | 56.8544% | — |
| head_newton | 56.8531% | — |
| tangent_profiled − SGD10 | -4.6367 pp | 0.04785 |
| tangent_profiled − SGD100 | -5.1090 pp | 0.02930 |
| tangent_profiled − head_newton | +0.00125 pp | 1.0 |
| tangent_protected − tangent_profiled | +4.6589 pp | 0.04395 |
| information_protected − tangent_information | +4.6430 pp | 0.04395 |

Real-prediction KL protection restores BA to roughly 61.51%; this is recovery from a weak aggressive-fitting baseline, not proof of improvement over SGD100 or support_EA. Jointly trained dictionaries add tiny changes without positive evidence. The risk worth studying is the mismatch between fitting a tiny labelled support set and future prediction, especially overconfident or overlarge functional head motion. These are hypotheses suggested by the negative results, not established causal attribution.

### 3. R7 expanded geometry and relative risks have not produced a clear gain

Pan2023 B20 clean has 14 participants, 28 sessions and 3 pipeline seeds (84 paired session-seed rows). All 12 primary BA comparisons have Holm p=1.0; none of the 24 combined primary BA/NLL comparisons has Holm p<0.05. Examples:

- Spatiotemporal CE/full versus static CE/full: -0.1190 pp.
- Spatiotemporal relative/full versus spatiotemporal CE/full: +0.02381 pp.
- Spatiotemporal relative/budget-head versus relative/full: +0.09524 pp.
- Jointly matched st-rel-budget dictionary versus common dictionary: +0.04762 pp.

| Existing reference / candidate | Stieger internal BA | Pan published-order BA |
|---|---:|---:|
| Frozen f0 | 61.1005% | 63.1786% |
| support_EA | 62.2132% | 63.8214% |
| head_newton | 56.8531% | 60.3929% |
| static CE/full | 56.8918% | 60.4524% |
| spatiotemporal relative/budget | 56.7375% | 60.4524% |

The strongest returned signal is therefore not evidence that extra temporal geometry helps. It is that simple unsupervised support alignment remains competitive while some supervised small-support fitting damages an already useful source model. A Round 8 method should explicitly compare against frozen f0, support_EA and a modest SGD head/space baseline, not merely beat an aggressive Newton baseline.

## Nonduplicative mathematical directions suitable for Round 8

These are proposed investigations, not claims of new published theorems or implemented results.

1. **Reliability-shaped partial covariance transport.** In whitened CAR coordinates, estimate a regularized support covariance C, study `exp(-t log(C)/2)` or an anisotropic spectral shrinkage counterpart, and separate global shrinkage from support-derived spectral reliability. At t=0 this is identity; at t=1 this yields regularized whitening in the whitened coordinates. A determinant-normalized shape variant is equivalent up to a common scale before trial RMS, which is a useful exact null. Main controls: existing EA, scalar partial EA, same-dimensional arbitrary/random spectral weights, and label-free versus label-assisted selection. The innovative question is which spectral shape information can be trusted from the paid prefix, not whether whitening itself is new.

2. **Conditional residual moments with a genuinely restricted critic.** For residual r=y−p and fixed support probe matrix Φ, the ridge-penalized quadratic adversary has closed form `gᵀ(G+γI)⁻¹g`, with g=ΦᵀWr and G=ΦᵀWΦ. Constant, low-capacity physiological, random equal-rank and saturated probes distinguish calibration information from label interpolation. Add source-anchor and support-fold controls, and log effective critic rank. This is a new axis relative to the executed R2–R7 objectives, but conditional moment/GMM methods already exist and full-rank probes at n≤20 can overfit. A low residual moment value is not a future conditional-calibration guarantee.

3. **Function-space continuation / shrinkage after alignment.** Instead of treating Euclidean head ridge and parameter radius as adequate controls, explicitly constrain or regularize changes in source/EA log-odds on support and source-development anchor signals. Compare a calibrated path of functional movement to fixed SGD update count and to R3 KL protection, with a shared optimization budget and same anchor inputs. This must go beyond repeating R3's existing KL protection or R4's OOF mixture; possible new question is whether source-input sensitivity and nuisance-head quotient geometry explain the failure of stronger fitting. Using development anchors changes information available at target adaptation and must be declared separately from the strict support-only deployment arm.

4. **Covariance-preserving ambiguity as a falsifiable mechanism study.** For any covariance match, `A=Rs^(1/2) O Rt^(-1/2)` with orthogonal O is an entire family, not a unique semantic correspondence. Carefully controlled covariance-identical synthetic worlds can test whether a few labels or restricted higher-order moments identify a useful orbit direction. This was proposed but not implemented in R7. It can provide an interpretable negative or positive mechanism conclusion even if no new method wins average BA. It must not claim that covariance match alone identifies physiology.

Given current summaries, (1) with independent (2) is a defensible main direction. More expensive random grids of previously ineffective geometry are less informative than matched counterfactuals, effective movement diagnostics and fresh participants.

## Statistical boundaries and meaningful scale

- Stieger meta-training/validation/internal sets contain 40/11/11 participants; respective known meta catalogs have 396/97/105 sessions. The independent unit of the internal comparison is 11 people, not 105 sessions, 315 seed-session rows or thousands of trials.
- Pan has 14 people, 28 sessions and 3360 cached LR trials. Pan was used in R6/R7 and its results now inform R8. It is repeated external exploration, not a fresh blind confirmation dataset. Its verified order is provider MAT release order, not independently reconstructed physical chronology.
- Existing prepared Weibo/Yi2014 cache contains 10 people, 1580 LR trials and 1380 B20 fixed queries, about 11MB (8.44MB compressed portable bundle). R3/R7 documented optional Weibo stages; current returned R3 transfer tables are empty and R7 transfer is Pan. This suggests a useful extra dataset but does not establish it has never been run elsewhere. Call it a prespecified additional evaluation, not unconditionally a first-ever blind test. Its first20 LR positions are not the first20 positions of the original seven-task experiment; original all-task first20 give only 4–8 LR events. Preserve both scopes explicitly.
- Repeating more seeds or evaluating more methods does not create more independent human participants. Increasing seeds reduces algorithmic Monte Carlo variability, not participant sampling uncertainty. Lock a compact primary comparison family and classify large follow-on grids as discovery/robustness analyses. Do not promote whichever subgroup wins to the main hypothesis after seeing it.
- Save participant-level and session-level paired predictions, support indices, query IDs, source probability, adapted probability, realised functional/input displacement, fallback reason, chosen covariance spectrum, support-loss/future-loss gap and calibration timing. These support mechanism analysis without retraining.
- Keep a finite, versioned task manifest with completed task reuse and append-only extension batches. "Unlimited" should mean no arbitrary cumulative wall-clock stop and easy extension, not an unbounded experiment lacking a completion target or comparable budget.

## Runtime consequences

Round 8 should use `hpc_gpu` / A100-PCIe-40GB, three independent one-GPU Slurm jobs, each 16 CPU and64G host memory (48 CPU/192G total). Node names such as gpu1 must not be confused with partition hpc_gpu1. Small matrix solvers may need fewer compute threads than the requested CPU allocation. Preserve Python3.12.3 / torch2.5.1+cu121 / NumPy2.2.6 and existing source checkpoints; no online pip or raw-data download inside GPU jobs. Bundle already prepared optional small caches or stage them before submission, with retry/resume outside training. GitHub progress and compact summaries are asynchronous and must not terminate training. Full predictions/checkpoints remain local and can be packed to Releases separately. Reuse existing per-task/per-session checkpointing, single-node `NumNodes=1-1` compatibility, exact job-identity adoption and quota retry; do not copy the old multi-round handoff machinery into a fresh round.

