# Verified literature A: reading and attribution log

Access date: 29 October 2026. This record separates abstract/preview reading from full-text reading. No publisher PDF of these three target papers has yet been obtained. No claim of full-text review is justified. All numerical performance in these papers belongs to their authors and is excluded from our manuscript results.

## 1. LEAPNet

**Verified citation.** Huang, Jinguo; Xiong, Xiong; Xu, Yunfei; Weng, Xinhua; Su, Li. *Robust Cross-Session EEG Measurement via Trial-Conditioned Log-Euclidean Reference Synthesis*. IEEE Transactions on Instrumentation and Measurement 75 (2026), article 4015813. DOI: [10.1109/TIM.2026.3742096](https://doi.org/10.1109/TIM.2026.3742096). Publisher publication date: 9 October 2026. Screenshot is a shortened description, not the exact title.

**Read scope.** IEEE publisher abstract and publisher-deposited Crossref metadata. Standard publisher PDF request did not return accessible PDF; no verified public accepted manuscript located. Crossref metadata is saved in `literature/crossref_leapnet.json`.

**Technical synopsis (abstract-level).** The problem is longitudinal session drift with costly recalibration. The model learns SPD anchors, routes each trial to a log-Euclidean reference, and uses that reference for tangent projection. Filtering and covariance regularization support the central reference-synthesis mechanism. Evaluation distinguishes zero-shot inference from a separate calibration experiment and includes a primary motor-execution setting and a multiday benchmark.

**Structural lesson, inferred from the abstract rather than unseen sections.** Introduce the deployment burden before the geometry; name one causal design choice; distinguish the main operating condition from supporting validation. Do not claim knowledge of the full introduction or discussion sequence.

**Boundary relevant to our work.** A learned SPD-anchor reference is not the same object as a source-trained adaptation-direction dictionary. If our code updates constrained coefficients from a chronological labeled prefix, define that update explicitly. A zero-shot motor-execution result cannot be imported as a direct MI 20-trial future-run comparison. “First geometric dictionary for EEG” would overlap prior art and should not be claimed.

## 2. Resting-state subject-adaptive meta-learning (ResML)

**Verified citation.** An, Sion; Kang, Myeongkyun; Kim, Soopil; Chikontwe, Philip; Shen, Li; Park, Sang Hyun. *Subject-adaptive meta-learning for personalized BCI: A fusion of resting-state EEG signal and task-specific information*. Information Fusion 125 (January 2026), 103501. DOI: [10.1016/j.inffus.2025.103501](https://doi.org/10.1016/j.inffus.2025.103501). The DOI/author website uses 2025 online-era metadata; the final issue is 2026. Use final-issue year consistently.

**Read scope.** Publisher-indexed article preview, including abstract, introduction, method-section summaries, selected results and discussion/conclusion; DGIST author institution metadata. The [DGIST record](https://scholar.dgist.ac.kr/handle/20.500.11750/58900?mode=full) explicitly lists no attached files and marks the article non-open-access. Author and laboratory publication pages do not link a full article PDF. The article preview must not be described as full-text reading.

**Technical synopsis (preview-level).** The practical gap is collecting task-specific recordings from a new person. The framework fuses resting-state person information with learned task information, then integrates subject-specific adaptation and meta-training. A lightweight update component limits what changes during personalization. The visible article separates problem settings, architecture, fusion/losses, transfer and meta-learning variants, then experimental setup and mechanism-oriented evaluations.

**Structural lesson.** Make the costly resource concrete, connect each module to that bottleneck, and reserve separate ablations for fusion, person information and update capacity. Our adaptation budget should be expressed as labeled trials; it is a different resource from resting-state recordings. Do not infer that their meta-learning framework or lightweight adapter is our invention.

**Additional primary source found.** Sion An's institutional doctoral-thesis record, [DOI 10.22677/THESIS.200000942204](https://scholar.dgist.ac.kr/handle/20.500.11750/59611), publishes chapter headings for both this study and DA-RelationNet. The thesis record itself has no attached PDF. Its table of contents independently verifies the method/experiment/ablation ordering, but is not substitute full article text. The linked official dCollection original displays an embargo notice until **28 February 2029**; no attempt to bypass that restriction was made.

## 3. DA-RelationNet

**Verified citation.** An, Sion; Kim, Soopil; Chikontwe, Philip; Park, Sang Hyun. *Dual Attention Relation Network With Fine-Tuning for Few-Shot EEG Motor Imagery Classification*. IEEE Transactions on Neural Networks and Learning Systems 35(11) (2024), 15479–15493. DOI: [10.1109/TNNLS.2023.3287181](https://doi.org/10.1109/TNNLS.2023.3287181). IEEE reports initial publication on 28 June 2023; final issue is November 2024. PubMed's displayed electronic date differs from IEEE's initial online date, so final bibliographic year 2024 is the least ambiguous citation choice.

**Read scope.** [Publisher metadata/abstract](https://ieeexplore.ieee.org/document/10167679/), [PubMed record](https://pubmed.ncbi.nlm.nih.gov/37379192/), institutional abstract and thesis table of contents. Standard publisher PDF unavailable; author repository [SionAn/FewShotEEG](https://github.com/SionAn/FewShotEEG) is a primary implementation source, not the full article. Author web pages link the DOI, code and news rather than a full PDF.

**Technical synopsis (abstract-level).** The model combines an embedding, temporal attention, support aggregation attention and a relation classifier for two-way few-shot MI. Fine-tuning constructs adaptation queries from the supplied support set. Cross-person and cross-dataset evaluations use BCI IV-2a/2b and GIST. Institutional thesis headings additionally identify attention visualization, person-level differences, support-set robustness and embedding comparisons.

**Structural lesson.** A support/query diagram should show what the model may learn from, followed by targeted rather than decorative ablations. Our chronological prefix/future-run split should not be described as a reproduction of a random-support unseen-person protocol. Attention, relation learning, episodic meta-learning and support-set fine-tuning retain their original attribution if adopted.

### Author-code protocol check

The official repository was read at immutable commit [`de523c0c23208a816ab37dde8afeb4ee1efc91c7`](https://github.com/SionAn/FewShotEEG/tree/de523c0c23208a816ab37dde8afeb4ee1efc91c7). `main.py` configures held-out subjects. In `data.py`, `sampling()` draws `few` support indices **for each of two classes**; consequently `--few 20` requests 40 support entries, not a total 20-trial calibration budget. Test support/query arrays are loaded from external NAS paths and are not supplied in the repository, so test chronology and all exact published split identities cannot be independently reconstructed from this code. These observations describe the inspected implementation, not an unverified claim that every final-paper experiment used that exact default. Saved source and SHA256 provenance are in `literature/FewShotEEG_provenance.json`.

## Use in the manuscript

Use these papers as concise numbered references where they substantiate the relevant background or distinguish the actual adaptation mechanism. Their titles, datasets and performance numbers need not be narrated in the Results section. Do not copy their language, diagrams, equations or model names. A new name for our documented combination can be introduced, but inherited components remain inherited. Novelty must be tied to the implemented constrained update, protection rule, data access boundary and evidence—not to renaming established methods.

## Access and evidence files

`literature_a.json` and `literature_a.bib` contain verified bibliographic fields. Crossref responses, OpenAlex OA-location checks, author-site metadata and publisher-access outcomes are retained under `literature/`. All three target publisher articles currently remain full-text-unavailable through the legitimate routes checked; the previews and thesis headings support the stated scope only.
