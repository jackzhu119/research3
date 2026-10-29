# Error redistribution in motor imagery EEG recalibration

**Error redistribution across conventional and neural motor imagery EEG recalibration**

Ziyuan Zhu, Sihao Liu and Haoyang Yu. Target journal: **Biomedical Engineering / Biomedizinische Technik**. This is an **author-review manuscript**, not a record of journal submission or acceptance.

## Read and download

- [Main manuscript: Word](Manuscript.docx) | [PDF](Manuscript.pdf) | [Markdown](Manuscript.md)
- [Complete public manuscript and reproducibility package](BMTE_Error_Redistribution_Public_Package.zip)
- [All eight scientific figures, vectors and plotting inputs](Scientific_Figures_All_8.zip)
- [Supplementary Materials: PDF](Supplementary_Materials.pdf) | [Word](Supplementary_Materials.docx)
- [Small data/code supplement for journal upload](submission/Supplementary_Data_and_Code.zip)
- [Official ethical and legal declarations](submission/Ethical_and_Legal_Declarations.docx)
- [Cover letter](submission/Cover_Letter.docx) | [Figure captions](submission/Figure_captions.docx)
- [Copyable submission fields](submission/Submission_Fields.txt) | [Chinese guide](README_ZH.md)

## Research scope

The main evaluation comprises 56 participants in five cohorts and 212 target sessions. Six Kumar development participants and an 85-participant source-model replacement extension are reported separately. Four conventional pipelines locate classifier, spatial-refitting and alignment contributions; eighteen neural configurations resolve prediction changes across acquisition budgets. Cohort-specific processing, references and valid-label/acquired-position budgets remain distinct.

Corrected and introduced errors are class-balanced quantities. Their difference gives net balanced-accuracy change, and their sum gives prediction disagreement. Complete method/budget results and participant summaries are retained. The analysis is retrospective and descriptive; original testing families remain separate, and no new model fitting, model inference, p-value tests or pooled architecture effect was introduced.

All four main and four supplementary figures were plotted from the archived data and study design using Matplotlib. Editable SVG/vector PDF and 600dpi RGB PNG/TIFF are supplied. See [figure style review](figures/STYLE_REVIEW.json), [numerical verification](analysis/INDEPENDENT_NUMERICAL_CHECK.json) and [final document validation](submission/FINAL_VALIDATION.json).

## Reproduction

From the extracted public package, use the dependencies in requirements-analysis.txt:

```bash
python analysis/analyze_integrated.py
python analysis/check_analysis.py
python figures/plot_integrated.py
```

Input source commits and SHA-256 hashes are in [evidence/INPUT_MANIFEST.json](evidence/INPUT_MANIFEST.json). Standalone extraction and exact key-table reproduction were verified. Raw EEG remains with the original dataset providers. No training checkpoints or third-party reading fulltexts are included. Suggested-reviewer preparation files remain in the author's local submission package.

Previous manuscript folders and repository history are preserved. See [DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json) for the public files and hashes, and [PUBLICATION_SCOPE.json](PUBLICATION_SCOPE.json) for the relation to the reviewed author package.
