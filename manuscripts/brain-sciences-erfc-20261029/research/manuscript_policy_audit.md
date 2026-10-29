# Manuscript policy audit — Brain Sciences

Audited file: `Manuscript.md`, draft observed on **29 October 2026**. Policy authority: the directly retrieved current Brain Sciences author instructions and MDPI ethics policy documented in `journal_requirements.md/json`. This review checks preparation of the draft; no manuscript has been formally submitted.

**Decision:** the draft can be delivered for author review. Its structure follows the journal-specific requirements and it does not invent institutional ethics approval, final author approval or a negative conflict declaration. Two small, concrete edits should be made before producing the review DOCX/PDF: put the substantive AI-use scope directly in Methods as well as Acknowledgments, and unify rounding of one displayed p-value. Outstanding author declarations belong in the separate pre-submission checklist and do not prevent delivery of this clearly labelled draft.

## Verified preparation checks

| Check | Observation | Result |
|---|---|---|
| Journal abstract format | Four headings: Background/Objectives; Methods; Results; Conclusions. About 242 whitespace-delimited words. | **Pass.** Current Brain Sciences calls for approximately 250 words, not a strict maximum or the generic MDPI 200-word format. |
| Keywords | Seven semicolon-separated terms. | **Pass:** permitted range is 3–10. |
| Article structure | Introduction, Materials and Methods, Results, Discussion, Conclusions; numbered subsections. | **Pass.** A separate Related Work section is not mandatory. |
| Highlights | Not included. | **Pass:** optional. |
| Authors and correspondence | Ziyuan Zhu; Sihao Liu; Haoyang Yu, same superscript affiliation. First author marked corresponding, with the supplied address. | **Pass against the user-provided names/emails.** No invented degrees or ORCIDs. |
| Affiliation | College of Artificial Intelligence Medicine, Chongqing Medical University, Chongqing 400016, China. | Consistent with the verified published form in `versions_and_authors.md`; institution's preferred English wording and correspondence campus/postcode can be confirmed separately, without guessing a street address. |
| Author contributions | Software modification for S.L.; material organization/repository checks for H.Y.; draft explicitly reserves final approval by all three. | **Draft acceptable.** Policy checking alone is not substantive scientific review; file transfer alone is not scientific validation. Keep the actual scope, and confirm that Data curation for H.Y. means research-data/material curation rather than merely uploading files. Do not expand to unreported formal analysis, method design or supervision. |
| Funding | No external funding. | **Pass:** confirmed by the user. This does not itself establish absence of financial or nonfinancial conflicts. |
| IRB language | Describes secondary public-EEG analysis, no new recruitment/contact/intervention/specimens, cites Article 32(1), and makes no assertion of a new institutional approval or waiver identifier. | **Pass as a factual draft statement.** It does not certify the project has a committee-issued exemption. Final determination/Article 32 conditions and source-study provenance should be confirmed for submission, as documented in `china_secondary_data_ethics.md`. |
| Original consent | Attributes original consent to the dataset investigators, rather than to the current authors. | **Correct attribution structure.** The actual source-study records must support the collective assertion for all three cohorts; an article citation's existence alone is not proof. |
| Data availability | Provides repository, immutable commit, exact analyzed stage, raw source-repository attribution and scope of reproducibility package. | **Pass.** Ensure the final delivered companion files actually match that package description. |
| Distinguishing data exposure | Methods describes historical/development exposure, retrospective amendment, exploratory inference and participant-level units. | **Pass.** No unsupported preregistration, independent replication, clinical-trial or prospectively untouched-cohort claim. |
| Method provenance | Names EEGNet, Euclidean alignment and RKHS foundations as existing components; ERFC is explicitly a manuscript name for the archived profile implementation. | **Pass.** A newly named method is not presented as a newly trained architecture or a different experimental run. |
| Figures | Five main/supplementary PNGs are 600 dpi to PNG metadata precision (599.9988); raster sizes approximately 3128–4159 px wide. TIFF alternatives are supplied. | **Pass:** journal recommends high-quality images preferably ≥600 dpi. PNG RGBA is usable; RGB TIFF alternatives satisfy the recommended RGB representation. |
| Tables | Markdown tables with headings, captions and statistical units, intended to render as editable Word tables. | **Pass at source stage.** Verify they remain actual tables in DOCX, not screenshots. |
| References | Source uses citation keys and a final References heading; `references.json` and MDPI CSL are supplied for cite processing. | **Rendering check required:** Word/PDF must contain numeric citations and a populated sequential bibliography, not literal `[@key]` tokens or an empty References section. The Markdown source itself need not be the submission artifact. |
| Disclosure placeholders | Conflict statement explicitly requests author confirmation; contribution paragraph does not claim completed all-author approval. | **Pass for review draft; pending before actual submission.** Do not label this a fully submission-ready attested version. |

## Concrete edits for the current render

### 1. Make the Methods AI-use description self-contained

Section 2.12 currently cross-refers the AI assistance to Acknowledgments but gives only the broad terms “manuscript preparation and code drafting.” The current journal/publisher policy calls for details of substantive use in Methods and tool details in Acknowledgments. The task includes prose generation and code assistance, so it does not fall entirely under the superficial-editing exemption.

A suitable replacement in 2.12 is:

> ChatGPT (OpenAI; accessed on 29 October 2026) assisted with literature organization, manuscript drafting and the drafting of analysis and plotting code. Numerical results were obtained by executing the accompanying scripts on the archived data; tool-use details are provided in the Acknowledgments.

This describes the observed scope and does not pretend that all human authors have already reviewed/approved the final manuscript. Keep the existing Acknowledgments disclosure. Add an exact tool/model identifier if it is available from the session record; do not invent a model version to fill the field.

### 2. Use one rounding convention for the Weibo introduced-error p-value

The same exploratory matched-MAP effect is printed as **0.0313** in the Results/Discussion and **0.0312** in Table 3. The saved figure-source table (`Fig3_source_exploratory_effects.csv`, Weibo / balanced_introduced_vs_ea) confirms the underlying value **0.03125**. Use **0.03125** everywhere or consistently round to **0.0313**. This is a display consistency edit, not a change to the statistical test.

## Submission-stage factual confirmations, kept outside the main scientific narrative

- Each author's real substantial contribution, critical intellectual review, final version approval and accountability; email-display consent.
- Actual COI declarations. The current visible placeholder correctly avoids inventing a no-conflict statement.
- Original datasets' ethics/consent/licensing evidence and the conditions supporting reliance on Article 32(1). Do not convert the draft into a fabricated CQMU waiver or write a new approval number.
- Precise institutional English form and any full postal/campus address required by the submission system.
- Prior journal/preprint history and no simultaneous submission. The cover letter must contain the journal's required confirmations once true, rather than presenting them as already verified now.
- Three proposed independent expert reviewers entered in the submission system, not the cover letter.
- The actual AI product/model details where documented, plus human review and responsibility before submission.

These are ordinary journal declarations attached to the concrete draft, not reasons to halt drafting, request payment, contact authors/reviewers, or submit anything automatically.

## Authoritative sources

- [Brain Sciences Instructions for Authors](https://www.mdpi.com/journal/brainsci/instructions), retrieved directly on 29 October 2026.
- [MDPI Research and Publication Ethics](https://www.mdpi.com/ethics), retrieved directly on 29 October 2026.
- [Chinese 2023 ethics-review measures, Article 32](https://www.gov.cn/zhengce/zhengceku/2023-02/28/content_5743658.htm), retrieved directly on 29 October 2026.

Local official-source copies, retrieval details and SHA-256 hashes are recorded in `journal_requirements.json` and `china_ethics_sources.json`.
