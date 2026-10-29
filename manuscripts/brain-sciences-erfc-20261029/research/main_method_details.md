# Actual implemented method and protocol details

Machine-readable companion: `evidence/main_method_protocol_verified.json`. These details were checked against the current R10 result receipts, signed stage plan, matching source files, and reference provenance. No fresh experiment was run.

## Input and frozen model

- Actual signal contract digest: `5c8ea3f01634c060546af29b9330d314c4f03c265677a74b875ac7c302e31872`. This exactly equals the canonical digest of `/workspace/paper3-r11-repo/contracts/input_contract.json` and all four downloaded R10 scientific/result-reference receipts.
- Measured channels, in fixed order: FC3, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP4.
- Within each original trial, common-average reference; fourth-order bidirectional Butterworth SOS bandpass at 8–30 Hz; polyphase resampling to 128 Hz; crop from 0.25 to 1.25 seconds after the verified task/MI anchor. The only filter context is 0–1.5 seconds after that anchor. No post-context signal, padding of short original trials, or query-derived normalization is allowed.
- A crop has shape 11×128. Demean each channel over time, then normalize all channel/time samples by one joint trial RMS in float64, without an epsilon floor. Nonfinite and zero-energy trials are invalid. Invalid trials still consume their acquired-position budget.
- Stieger's verified MI/feedback anchor is 2000 ms after the cue. Thus do not accidentally describe its crop as 0.25–1.25 s after the cue.
- Four-source existing compact EEGNet: BNCI2014_001, Cho2017, Lee2019_MI, PhysionetMI. It is an existing baseline architecture, not this paper's novel backbone. Its temporal convolution has 8 filters (kernel 63), depthwise spatial stage has 16 output channels, then depthwise temporal kernel15/pointwise16, ELU and mean pools4 then8. For 128 time points its flattened feature dimension is 64, followed by a two-logit classifier. Dropout and batch normalization remain in evaluation mode; source parameters remain frozen during target calibration.
- Actual source F0 receipts were fetched and their exact SHA256 and checkpoint SHA256 matched all three R10 reference receipts. Source-only leave-one-dataset-out selection yielded final effective epochs **39,30,45** for seeds20261018,20261019,20261020, respectively. The stored `epochs=80` is the source-search maximum, not the actual final fit length. Each epoch has64 update steps, minibatch64; Adam learning rate0.001, betas(0.9,0.999), epsilon1e−8, zero weight decay; dropout0.25. Source sampling is dataset→subject→observed class→trial, with replacement. All three actual receipts list BNCI2014_001, Cho2017, Lee2019_MI, PhysionetMI. `evidence/source-f0-receipts-proofs.json` contains full original receipts, SHA256, and immutable source-manifest URLs; `evidence/main_method_protocol_verified.json` provides exact checkpoint/source-LODO descriptors and configuration. Raw source training-trial counts were not fetched, so do not invent them.

## Development references and participant split

The fixed Stieger split comprises 40 meta-training, 11 meta-validation, and 11 internal held-out participants. Split digest `1a4704555760dd1d8956b31268931617b5c12d2441a17f68dd8f34d108dd61fd` is identical in the downloaded R10 data receipts.

Actual held-out IDs: S18, S20, S28, S31, S32, S42, S44, S48, S58, S62, S9. Meta-validation IDs: S16, S22, S33, S37, S40, S46, S5, S57, S60, S7, S8. Full split lists are in the machine-readable companion.

The R10 formal stage used `reference_strategy=reuse_r9_ea`. Its three seed-specific raw and EA reference geometries were reused from sealed R9 artifacts; it did **not** construct them using the separate four-participant/eight-session fallback reference subset. Downloaded `reference.json` receipts bind the same source and data identities and establish **40 meta-training participants and 396 sessions**. Only valid inputs within each session's first20 acquired positions enter the reference; future queries and query labels do not enter. Weighting is equal participant, then equal session within participant, then equal valid acquired-prefix trial within session.

Reference features are centered by this weighted mean and symmetrically whitened by a weighted covariance with fixed 0.1 trace-to-isotropic shrinkage. The R9 implementation has an explicit eigenvalue floor. A lengthscale is computed by a weighted median of pair distances on at most256 evenly indexed reference points; it is not used as a fitted target-query statistic. The primary residual kernel is linear plus intercept, so RBF lengthscale is not its novel component.

Support-EA is the actual source-colored support-covariance alignment implementation, not a claim to reproduce all details of the original EA method. In the ten-dimensional CAR effective space, each support covariance is trace-normalized; their mean is shrunk as 0.95 C + 0.05 I/10 and whitened/recolored to the frozen source reference. It uses acquired support inputs without support labels. The resulting same frozen EEGNet gives the anchor probability and features used by ERFC and its matched-MAP comparison.

## ERFC exact configuration

Working name ERFC maps to original `profile`, VERSION `paper3-r10-query-profile-v1`; algorithm source SHA256 `6fa50fe53ee4e80b3fe9e500b12cc14eb54b5e0bfde8afac1c30611fadba1e48` exactly matches the actual signed stage code manifest.

Selected candidate `a0.3-c0.1`: linear residual kernel with intercept; amplitude0.3; regularization1; evidence-retention fraction0.1; null-completion enabled; unique-event nugget1e−10; probability floor1e−12; maximum support20; Newton budget60; root budget12; bracket budget32; gradient tolerance1e−10; root tolerance1e−7; certification padding1e−10. Full config is saved as `tables/r10_profile_configuration.json`.

The R10 implementation uses class balancing with per-example weights `n/(2n_c)` (sum n), not the R3 mean-loss scale `1/(2n_c)`. The stated regularization λ=1 is tied to the R10 sum-scale objective. Do not silently mix scales in the equations.

1. Fit a regularized logistic residual on frozen features using only the acquired support labels.
2. Let the finite feasible objective be Jhat, the zero-residual objective J0 and observed regularized gain G=J0−Jhat.
3. Retain all model functions whose objective is at most Jhat+cG, including functions invisible at the support inputs.
4. Evaluate each query separately: its support-orthogonal completion scalar is s_q=k(q,q)−k_Sq^T K_SS^−1 k_Sq for the explicit nugget kernel. Use convex tilted problems and conservative finite-solver bounds to enclose its reachable residual/probability range.
5. The anchor-relative minimax proper-loss action is projection of p0 onto the resulting probability interval. If the interval contains p0, retain the anchor.
6. The sufficient zero-update certificate λ μ_q²/(2 s_q)≤cG permits retaining the anchor without solving both query endpoints. It is a function-class/objective statement, not a probability-calibration guarantee about future ground truth.

No new per-query model is fitted from other query features or labels. Query batching/permutation does not redefine the support fit or normalization. Failure or single-class/empty support returns the explicit anchor. The observed `profile` rows at all budgets (175 participant/budget aggregates) contain zero numerical-fallback and certificate-violation counts; this is observed diagnostic scope, not a general no-failure guarantee.

## Evaluation and controls

Original acquired-position budgets:4,8,12,16,20. At each budget use the corresponding unrefilled prefix; invalid acquired trials remain counted, with no class-balanced resampling to force10+10. At budget20 the Stieger mean valid support is15.5301063; Pan and Weibo means are20. Query inputs are always the same original B20 future set at every budget.

For Stieger, the tail of the physical run containing the20th acquired trial is excluded, and only later complete physical runs provide valid queries. Eligibility requires at least20 valid query trials and at least5 per class. For Pan2023 and Weibo2014, evaluation uses supplied published trial order; do not claim that their physical acquisition/run chronology has been established. Prediction probabilities are sealed before future labels are opened by the evaluation function.

The 18 implemented methods are: frozen, support_ea, head_bias, head_rank4, head_full, sgd10, sgd100, r3_protected, r4_xfit_select_stack, r8_fiber_bounded, r9_raw_anchored, r9_ea_anchored, profile, profile_map, profile_no_completion, profile_frozen_anchor, profile_quadratic, profile_variance_gate. The native baseline algorithms and existing learned/reference artifacts are retained; numerical labels in the manuscript must map back to these names. Matched MAP uses the same support kernel, source/EA anchor and frozen feature reference, with direct MAP prediction instead of query-profile projection.

All three declared complete pipeline seeds are20261018/19/20. Compute a session score within each participant/seed, average sessions equally within that seed, average seeds equally within participant, then average/test participants. Never treat315/84/30 session–seed cells as independent participant sample sizes. R10 primary CIs use10,000 participant bootstrap draws; its exact sign-flip test assumes participant sign symmetry/exchangeability. The 12-test primary family,24-test secondary family, and the new18-test post hoc mechanism family remain separate.

## Timing interpretation

`fit_seconds` times residual `algorithm.fit` on already extracted features; it excludes support-EA estimation, encoder feature extraction and IO. `incremental_prediction_seconds_per1000` covers the additional residual/profile prediction on already extracted query features. `elapsed_seconds` includes the worker's actual cache-dependent path. These cannot support a universal end-to-end latency or hardware-normalized speedup claim without a separate controlled benchmark. In particular, residual fitting near1–2ms must not be called the entire EEG calibration latency.

## Research history and contribution wording

The 165-task stage followed an explicitly retrospective development-protocol amendment. It used preserved mechanism outputs, choosing candidate a0.3-c0.1 on selection participants S22/S40/S57/S60 with check participants S16/S46/S5/S7. Internal/external cohorts had prior-round exposure and the report marks independent confirmation false. Computational disjointness of the fitted support and query labels is true; historical research independence is not.

The manuscript may introduce ERFC as a new integrated calibration construction and give original derivations for its objective-compatible query rule. Existing EEGNet, covariance alignment, kernel/RKHS mathematics, Schur innovation, convex duality, trust-region protection, and minimax projection concepts require accurate attribution. Do not describe the frozen backbone, an adopted geometric baseline, or the R3 `legacy` dictionary as newly invented by relabeling them.

## Verified stage and inference counts

R10 formal exploratory contains165 tasks =18methods ×3cohorts ×3complete seeds +3feature-reference tasks. Each evaluate task internally evaluates all five budgets, so 5budgets do not multiply the Slurm/task count. The primary independent sampling population has35 dataset participants and143 unique participant–sessions (11/105 Stieger +14/28 Pan +10/10 Weibo); with3seeds these yield429 session–seed cells (315+84+30), always averaged to participants before inference. The original R10 bootstrap seed is20261026 with10,000 draws; R3 uses seed20261018 with20,000 draws; the new post hoc mechanism family uses seed20261029 with10,000 draws. None of these repeat counts represents biological N.
