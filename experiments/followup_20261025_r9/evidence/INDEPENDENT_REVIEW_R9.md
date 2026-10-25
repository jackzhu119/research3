# Independent R9 review

Reviewed 2026-10-25 UTC. Scope: `runtime/result_channels.py`, native worker finalization and report outputs, followed by `paper3r9/gp.py`, `worker.py`, `plan.py`, report pairing/statistical logic. No school job was launched. The reviewer changed review notes only; code fixes below were made by the runtime author.

## Upload review

**P1 found and fixed:** same-content report atomic replacement/touch updated inode/metadata, but `publish_paper_file` and `archive_file` returned the old signature when SHA matched. Every later final signature check then failed forever. Independently reproduced both failures before correction. Author now refreshes the cached signature on equal SHA, persists it, and added regression coverage. Rechecked corrected code and ran all 10 `test_r9_result_channels.py` tests successfully.

**Completion strengthening implemented:** final report must have COMPLETE status, matching recipe identity, and matching complete/total counts. Full archive additionally requires every current COMPLETE receipt identity to be archived before writing `LATEST.json`. Archive waits until the native task `.lock` is free, covering worker final progress writes after `result.json`.

Verified design properties:

- Paper channel only enumerates declared root reports and one `reports/` level; it does not walk prediction/task trees. Task completion checks read declared receipts only.
- Exact byte chunks are SHA-addressed and reconstructable. Base64 avoids UTF-8 split corruption in Git; complete CSV row views retain multiline records, repeated headers and explicit offsets. Large JSON has no 64MiB rejection.
- Source file signatures are checked before/after capture, and before index publication; source changes during network publication retain blobs without committing a misleading latest pointer. Temporarily orphaned content-addressed pieces can be reused.
- Full archive captures COMPLETE task outputs incrementally, records per-part Release acknowledgments, retains uploaded file/task state after interruption and does not rewalk already-archived immutable completed task trees. The policy depends on completed scientific outputs being immutable.
- Release transport verifies asset size and SHA. Final `LATEST.json` is published only after all required capture/upload steps succeed. Partial `ARCHIVE_PROGRESS.json` and paper pointers remain distinct from scientific completion.
- Channel exceptions are caught in their own detached process, log RETRYING and never call training, cancel jobs or run tests. Credential discovery reads external private paths and does not enumerate them for archival. Task-tree inventory excludes private/runtime/logs/environment/token paths and symlinks; source models/raw datasets/venv are explicitly outside archive scope.

**P2 resolved:** author now filters deleted prior report CSV shards out of the current archive manifest, retaining remote historical objects. Tests cover deleted shard filtering, native task-tail lock waiting and mismatched final report identity. Independently reran the result-channel suite: all 13 tests passed. Reviewed result_channels.py SHA-256: `41b26acb1a9c29b31ec8bcc691b16aab05e423c56c6a5fe21697d660911c2195`.

Operational limitation: report generation and upload are asynchronous; if tables change during network publication, latest index publication retries. Training is unaffected. There is no proof of real school GPU execution or live R9 GitHub uploads yet.

## Algorithm and leakage review

The rank-eigenspace GP Laplace equations are consistent. With support kernel `K=U Λ Uᵀ`, latent basis `B=U sqrt(Λ)`, and query coefficient `a=k(q,S) U Λ^{-1/2}`, the implementation uses:

- posterior mode objective `sum_i softplus(m_i+(Bv)_i)-y_i(m_i+(Bv)_i)+0.5||v||²`;
- precision `Q=I+Bᵀ W B`;
- query residual mean `a v_hat`;
- variance `k(q,q)-||a||²+a Q^{-1} aᵀ`, retaining conditional prior variance outside the observed support span;
- information fraction `(prior_variance-posterior_variance)/prior_variance`, clipped for finite-precision PSD bounds;
- anchored mixture weight `anchor_cap * information_fraction`.

Thus zero query/support kernel information returns exactly the base probability in anchored mode. Unanchored Laplace prediction can still flatten the base logit through unobserved prior variance; preserving the prior conditional variance is essential and correctly implemented. `tau=0` returns the base prediction exactly. In the optional EA preprocessing arm, “base/source” in these statements means the frozen model evaluated under that fixed EA preprocessing, not the raw untransformed frozen baseline; prose should distinguish those cases.

Whitening/reference mean, covariance and deterministic pair-distance scale are learned from **meta-train paid prefix features** with equal participant/session/valid-trial weighting. Target support labels and features enter posterior fitting. Query features enter only their own prediction; no query labels enter GP fit or information gating. Hyperparameters are selected on meta-validation participant-mean balanced NLL and then shared across MAP, Laplace and anchored variants. This is a matched-prior ablation; it is not a claim that each baseline was independently tuned to its own optimum.

The worker preserves all acquired prefix positions, uses only QC-valid rows inside that paid prefix and records both acquired and valid counts. Query IDs, labels and source/data identities are retained for audit and reuse validation; those labels are read for scoring/cache identity, not fed to the fitter. External published-order datasets remain exploratory, especially repeatedly examined Pan.

`report.py` checks matched participant/session/query/source/data grids and all three seeds, averages sessions within participant/seed then seeds, and performs inference on participant differences. Seeds/session rows do not become independent participants. Six declared comparisons × two external datasets × two metrics form one 24-test Holm family. Existing cohorts are explicitly labelled repeated-cohort exploration, not fresh blind confirmation.

Information fraction depends on support labels through posterior likelihood curvature W. That is legitimate posterior conditioning, but its association with future errors is a **mechanism analysis**, not independent evidence that the gate is correct; do not tune thresholds using holdout/external query correctness. Per-query information/variance/mixture diagnostics are saved, while compact report tables retain participant/session/query identity for later analysis.

P2 API note resolved: `GPConfig.jitter` now controls the relative support-kernel eigenvalue cutoff, is validated in (0,1), and is recorded as `kernel_relative_eigenvalue_cutoff`. It is not diagonal jitter, so describe it as relative spectral truncation. Reviewed gp.py SHA-256: `5956354a457c95db4503494c74bb0e1a6e6e2f3cfaf661c5c04ca1d49ee93ad7`.

No new P1 algorithm/leakage issue identified in this inspection. Correct code and absence of leakage do not establish accuracy gain; the proposed R9 combination still needs comparison with EA and frozen/SGD baselines.

## Additional background-archive throughput observation

The current per-file Release-asset scheme can produce many small assets. The inherited transport re-lists all release assets before each upload, which can create quadratic API enumeration overhead at large scale. Runtime author was asked to cache normal successful-path asset inventory while retaining forced re-list after ambiguous POST acknowledgments/conflicts. This affects slow-channel throughput, not training correctness or the fast paper channel. Do not claim all full outputs are already uploaded.

## Final aggregated archive revision

The runtime author replaced per-file Release assets with one deterministic level-1 gzip/tar stream per immutable completed task, split into 32MiB parts. Plan/current reports are captured as their own generation-keyed group. Every group manifest retains source paths, exact source byte counts and SHA256, while the round-level manifest only indexes groups. Packed group/chunk manifests are durable and reused after network interruption. Metadata/signature checks cover each open source file; current report signature sets are compared before final publication. No new P1 inconsistency found in the bounded follow-up review. Independently reran all 15 result-channel tests, including many-small-file aggregation/restoration and interruption continuation: passed. Latest reviewed result_channels.py SHA256 `6b055de420e17af04037a616792ec16e73510bc6b21255d786f28d757f5d6ce7`; transport SHA256 `216ddf5b4b0c714847d2207beea2178d40005213d6c31e6ce5963395d3bc6319`.

This revision resolves the earlier per-file-asset scale concern by aggregating small files; Release-list caching additionally reduces repeated pagination while ambiguous upload acknowledgment paths continue to re-list for correctness. Neither a passing simulated test nor a published package is an assertion of completed school upload.

Independent transport verification after aggregated-archive revision: `python -m unittest discover -s tests -p test_r9_upload_transport.py -q` — all 9 tests passed, including asset-list cache and ambiguous acknowledgment refresh behavior.
