# Verification and reading notes: papers 4–6

Accessed 29 October 2026. This is a research note for the author, not manuscript prose. No performance values from these external papers should be presented as results of the present study. Article structure is described only where the full article was actually read.

## Reading coverage

| Screenshot item | Verified identity | Reading coverage |
|---|---|---|
| RUNet | Jin et al., *IEEE Transactions on Biomedical Engineering* 2026, 73(10), 3384–3399. DOI: [10.1109/TBME.2026.3653024](https://doi.org/10.1109/TBME.2026.3653024). Online first: 12 January 2026; final issue: October 2026. | Publisher/PubMed abstract and Crossref bibliographic record. No legally retrievable public full text located; do not claim full-paper reading. |
| Co-adaptive few-shot calibration | Li et al., *IEEE Journal of Biomedical and Health Informatics*, early access, 1–15; 23 September 2026. DOI: [10.1109/JBHI.2026.3735157](https://doi.org/10.1109/JBHI.2026.3735157). | Publisher/PubMed abstract and Crossref bibliographic record. No legally retrievable public full text located; do not invent body headings or split details. |
| Stiefel-SPD | Tibermacine, Russo and Napoli, *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 2026, 34, 595–606. DOI: [10.1109/TNSRE.2026.3652858](https://doi.org/10.1109/TNSRE.2026.3652858). Online: 12 January 2026. | Full 12-page published article read, including methodology, theory, experiments, discussion and limitations. CC BY 4.0 institutional copy downloaded. |

Machine-readable bibliographic data and exact access scope are in `literature_b.json`; ready-to-import entries are in `literature_b.bib`.

## 4. RUNet — verified abstract-level reading

**Exact title:** *RUNet: A Zero-Calibration Framework for Cross-Domain EEG Decoding via Riemannian and Unsupervised Representation Learning*.

The abstract connects session/subject variability to a unified representation pipeline: multiscale spatiotemporal features, regularized covariance optimization, an affine/log mapping, resting-EEG contrastive pretraining and task-specific retraining. Evaluation covers public MI datasets and a laboratory dataset. The persuasive move is to connect every block to one transfer problem, then test both cross-session and cross-subject settings. This supports an abstract-level structural lesson; the full article's paragraph arrangement, splits and target-data access remain unverified.

For our study, distinguish **zero-calibration** representation learning from adaptation using a fixed 20-trial labeled prefix and temporally later evaluation. Shared covariance geometry does not make the protocols interchangeable. Do not claim a comparison with RUNet unless it was actually rerun under the same information budget. Its published scores are not evidence about our algorithm.

Primary sources: [IEEE publisher record](https://ieeexplore.ieee.org/document/11346009/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41525614/), [Crossref](https://api.crossref.org/works/10.1109/TBME.2026.3653024).

## 5. CFC — verified abstract-level reading

**Exact title:** *A Co-adaptive Few-shot Calibration Method for Improving Motor Imagery BCI Performance*.

The abstract gives two specific obstacles, assigns one intervention to each, and evaluates the resulting co-adaptation in human participants. Its method combines subject-specific MI-pattern selection with meta-learning during calibration. The useful structural lesson is **problem–intervention correspondence**, followed by a practical calibration endpoint. No claim is made here about the article's unseen body-section layout.

Our retrospective fixed-task EEG analysis does not establish improved human MI skill, optimized imagery strategy or online user–machine co-adaptation. A small-label adaptation claim should be tied to classifier behavior on later trials, rather than to behavioral training benefits. The shared few-shot motivation warrants appropriate attribution; renaming meta-learning would not establish originality.

Primary sources: [IEEE publisher record](https://ieeexplore.ieee.org/document/11705632/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/42776881/), [Crossref](https://api.crossref.org/works/10.1109/JBHI.2026.3735157).

## 6. Stiefel-SPD — full-paper reading

**Exact title:** *Stiefel-SPD Manifold Graph Convolution for End-to-End EEG Learning*.

### What the article actually does

Its method forms regularized feature covariances using a compact EEGNet-style front end, reduces their dimension through an orthonormal Stiefel projection, performs reference-centered tangent-space aggregation on a scalp graph, and returns to SPD matrices before log-domain classification. QR retraction enforces the projection constraint. The article explicitly distinguishes the orthogonal-congruence invariance of the log-Euclidean distance from general affine invariance.

The full text contains an introduction, methodology, separate generalization/convergence/stability sections, graph-processing details, experiments, results/discussion and a conclusion. Experimental reporting combines cross-subject evaluation, matched ablations, effect-size/significance analysis and complete-pipeline CPU latency. It evaluates PhysioNet, BNCI 2014-002 and BCI-ERN; the last is an error-related-potential task, so it should not be described as three homogeneous MI datasets. The study is not a fixed 20-trial cross-session prefix-to-future experiment.

### Construction worth learning

The introduction proceeds from a practical decoding problem to three families of established methods, identifies a precise representational gap, and then maps the proposed blocks to that gap. The method develops objects and constraints before optimization. Theory is followed by concrete module-removal experiments and measured computational cost. This ordering gives each experiment an explicit role, rather than presenting a laboratory chronology.

For our manuscript, the analogous structure is to define the information budget and adaptation object first, state the actual protection mechanism, and test that mechanism with paired comparisons. A compact geometric constraint can be explained through its effect on the search space and preservation of the source solution. Any mathematical bound must be derived for the implemented algorithm; this paper's Stiefel results do not automatically apply to a matrix-exponential adapter.

### Boundaries that matter for an independent contribution

Stiefel optimization, SPD covariance construction, matrix logarithms/exponentials, EEGNet and graph aggregation are established components with identifiable prior authors. They cannot be relabeled as newly invented by the present authors. An independent contribution could instead be a specific adaptation/protection rule and its experimentally established behavior under a stricter chronological calibration budget, provided the code and data support it. Present our own measurements; do not reuse the external paper's accuracy, latency, guarantees or neuroscience interpretations.

The article itself notes incomplete session-shift characterization and that log-Euclidean invariance is narrower than affine invariance. This is useful for choosing a precise scientific distinction, not for claiming that our method has already solved every untested setting.

Primary sources: [institutional landing record](https://hdl.handle.net/11573/1765415), [published CC BY PDF](https://iris.uniroma1.it/retrieve/5494d104-7bb3-4f44-9e14-64cf49325d49/Tibermacine_Stiefel-SPD_2026.pdf), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41525558/). Local PDF: `literature/stiefel_spd_2026.pdf`, SHA-256 `abf3e727c35a308e4b5f6a96ad196145071bb5b335de04020b93fb3c940d8530`; extracted reading copy: `literature/stiefel_spd_2026.txt`.

## Attribution and positioning recommendation

Use the article-specific notes to improve logical organization, not to copy sentences or conceal the provenance of reused methods. In a numerical-reference manuscript, necessary attribution can be concise without turning the introduction into six author-by-author summaries. A distinct name for the complete implemented system is appropriate; a claim that its standard mathematical operators or backbone were invented here is not.

A defensible novelty statement must join three elements: the **specific implemented decision rule**, the **defined information/chronology constraint**, and the **result directly measured in this repository**. Whether this is the first such combination requires a broader novelty search and cannot be established by these three sources alone.

## Additional verified foundational references for the writing team

- Congedo, Barachant and Bhatia. Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. DOI [10.1080/2326263X.2017.1297192](https://doi.org/10.1080/2326263X.2017.1297192). [Author publication page](https://alexandre.barachant.org/papers/journals/BCI-Review-2017/).
- Rodrigues, Jutten and Congedo. Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces. DOI [10.1109/TBME.2018.2889705](https://doi.org/10.1109/TBME.2018.2889705). [PubMed](https://pubmed.ncbi.nlm.nih.gov/30596565/).
- He and Wu. Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach. DOI [10.1109/TBME.2019.2913914](https://doi.org/10.1109/TBME.2019.2913914). [PubMed](https://pubmed.ncbi.nlm.nih.gov/31034407/).
- Lawhern et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. DOI [10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c). [PubMed](https://pubmed.ncbi.nlm.nih.gov/29932424/).

These four entries identify methodological provenance; this note does not claim fresh full-text reading of all four.
