# Data and code for error redistribution in EEG recalibration

Authors: Ziyuan Zhu, Sihao Liu and Haoyang Yu. Correspondence: zzy2630816871@gmail.com.

This archive reproduces a retrospective paired analysis of archived EEG decoder predictions. It contains selected matched predictions, participant summaries, preserved endpoint records, complete new tables and deterministic analysis/plotting scripts. It does not contain raw EEG, model weights or a new model-training task.

## Scope and sources

Five main evaluation cohorts contain 56 participants and 212 target sessions. The six Kumar development participants and the 85-person Dreyer source-model extension remain separate. Cohorts retain their own signal processing, fitting examples and query definitions. Kumar support caps count valid labels (`full` means19/20); neural budgets count acquired positions before quality control. Table rows representing seeds, configurations, references or sessions do not add independent participants.

Sources are pinned to commits `da2a3acee24746981bbcdd779ab88132da6e24c4` in jackzhu119/few-shot-mi-eeg; `a16d06760ae77699a809c9ba36d002d684560c82` in jackzhu119/research3; and `c4ea1f4ee1d078bcb015922f8092f05e263f2399` in jackzhu119/cross-subject-mi-eeg. Exact paths, sizes and SHA-256 hashes are in `evidence/INPUT_MANIFEST.json`. The source-stage logit subset retains every row in the stated two-condition comparison and records its larger parent archive and exact selection receipt.

## Reproduction

Extract all files while preserving their folders. From the extracted root, use the Python dependencies in `requirements-analysis.txt` and run:

```bash
python analysis/analyze_integrated.py
python analysis/check_analysis.py
python figures/plot_integrated.py
```

The first command regenerates the complete participant/cohort tables and pointwise intervals with10,000 participant-bootstrap draws and seed20261029. It does not add hypothesis tests or fit a model. The second checks independent source summaries, transition identities and input hashes. The third regenerates the four main and four supplementary figures using Matplotlib. Scientific plotting input tables and figure values are included. No command submits computing jobs or uploads data.

`analysis/SCHEMA.md` documents table units and fields. The accompanying Supplementary Materials explain the statistical roles, original endpoint families, denominator definitions and acquisition documentation. New common intervals are exploratory and pointwise; the legacy interval and testing families are kept separately. The analysis estimates no pooled cross-cohort effect or controlled architecture comparison.
