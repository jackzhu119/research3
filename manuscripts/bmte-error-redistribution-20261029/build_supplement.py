"""Complete supplementary methods, full result tables and source records."""
from pathlib import Path
import json,re
import pandas as pd
from project_config import TITLE
from assemble_manuscript import table,normalize_citations,f,signed,ci

P=Path(__file__).resolve().parent

def main():
    s=pd.read_csv(P/'tables/cohort_summary.csv')
    main=s[(s.study=='conventional')&(s.budget.astype(str).isin(['full','20']))]
    rows=[]
    labels={'frozen':'Frozen','classifier_only':'Classifier only','full_refit':'Full refit','aligned_full_refit':'Aligned full refit'}
    for _,q in main.iterrows():
        cohort_label='Kumar'if q.cohort=='Kumar2024'else'BNCI'
        role_label='Main'if q.analysis_role=='main'else'Dev.'
        rows.append([cohort_label,role_label,labels[q.method],f(q.ba_percent),f(q.C_pp),f(q.I_pp),f(q.D_pp),signed(q.G_pp)+' '+ci(q)])
    t1=table(['Cohort','Role','Pipeline','BA, %','C, pp','I, pp','D, pp','G, pp [95% interval]'],rows)
    n=s[(s.study=='neural')&(s.reference=='frozen')&(s.budget.astype(str)=='20')]
    neural_labels={'frozen':'Frozen source','support_ea':'Support alignment','head_bias':'Bias-only head','head_rank4':'Rank-4 head','head_full':'Full head','sgd10':'Spatial/head SGD10','sgd100':'Spatial/head SGD100','r3_protected':'Protected spatial','r4_xfit_select_stack':'Cross-fit stack','r8_fiber_bounded':'Bounded geometry','r9_raw_anchored':'Raw anchored residual','r9_ea_anchored':'EA anchored residual','profile_map':'Point-fit residual','profile':'Gain-retaining residual','profile_no_completion':'No-completion residual','profile_frozen_anchor':'Frozen-anchor residual','profile_quadratic':'Quadratic residual','profile_variance_gate':'Variance-gated residual'}
    rows=[[q.cohort,neural_labels[q.native_method],f(q.ba_percent),f(q.C_pp),f(q.I_pp),f(q.D_pp),signed(q.G_pp)]for _,q in n.iterrows()]
    t2=table(['Cohort','Configuration','BA, %','C, pp','I, pp','D, pp','G, pp'],rows)
    methods=(P/'supplement_methods.md').read_text()
    methods=methods.replace('the primary scoring reference','the principal scoring reference')
    captions=(P/'figures/captions.md').read_text()
    supplementary_caps={int(m.group(1)):m.group(0)for m in re.finditer(r'^\*\*Figure S([1-4])\..*$',captions,re.M)}
    stems={1:'FigS1_neural_full_grid',2:'FigS2_neural_probability_scores',3:'FigS3_conventional_development',4:'FigS4_source_model_extension'}
    text=f'''# Supplementary materials

## {TITLE}

Ziyuan Zhu, Sihao Liu and Haoyang Yu

This supplement retains the cohort-specific processing, all scheduled methods and budgets, scoring-reference sensitivities, original inference families and the separate source-model extension. New intervals use the common participant-bootstrap analysis and are distinguished from historical endpoint intervals.

{methods}

## Supplementary results

### Conventional pipelines and separately evaluated development participants

**Table S1. Full-support conventional pipeline summaries.** C and I use frozen CSP/LDA as the scoring reference. Kumar's final support run contains 19 or 20 valid labels; BNCI uses exactly 20. The six Kumar development participants retain their original development role. Values and new pointwise intervals are regenerated from the saved trial labels and predictions.

{t1}

Complete 4/8/12/16/full-support results for both Kumar cohorts are in `tables/cohort_summary.csv`. Session-level and participant-level records retain query pairing and actual support counts. Matched classifier, spatial and alignment operation contrasts at every available budget are in `tables/component_summary.csv` and `tables/component_participants.csv`.

### Complete neural configuration results

**Table S2. All neural configurations at twenty acquired positions, relative to the frozen source.** Every recorded configuration is included. Complete five-budget tables, both scoring references and pointwise intervals are supplied in `tables/cohort_summary.csv`; participant records are in `tables/participant_metrics.csv`. Descriptive row labels identify existing configurations; exact implementation keys are retained in the method inventory and machine-readable tables.

{t2}

### Source-model replacement

The separate source-stage comparison uses the fixed broadband/tail-min-recall S60-versus-S9 models. Frozen-source BA was 58.3186% under S9 and 64.1716% under S60. Corrected, introduced, total-disagreement and net-gain contributions were 18.8627, 13.0098, 31.8725 and +5.8529 percentage points. Total cancellation was 81.6364%. Of the 85 participants, 71 had positive net changes and 14 had negative changes. These participant endpoints average three fitted source seeds. The new transition decomposition is descriptive and does not replace the original weaker-class-recall endpoint.

The selected prediction archive contains all 40,800 rows of the two fixed model conditions (20,400 each), with participant, seed, trial ID, true label and both logits. `evidence/source_extension/SELECTION_RECEIPT.json` identifies its complete 979,200-row parent archive, parent SHA-256, exact selection and source commit. The executable analysis recomputes transitions from these logits and checks them against the independently recovered 85-participant table.

## Original dataset ethics and consent documentation

This article analyzes released, coded records without new recruitment, contact, intervention or re-identification. The statutory secondary-use basis is Article 32(1) of Guo Wei Ke Jiao Fa [2023] No. 4 [@nhc2023]. Original acquisition documentation and secondary-analysis declarations are kept distinct.

| Dataset | Original acquisition record used for documentation |
| --- | --- |
| Kumar | Kumar et al. and the NEMAR release [@kumar2024; @kumar2024correction; @nemar177]. The acquisition article reports University of Texas at Austin protocol 2020-03-0073 and written informed consent. |
| BNCI2014-001 | BCI Competition IV dataset 2a release protocol and competition account [@tangermann2012]. A dataset-specific approval/consent identifier was not identified in the inspected release protocol; statements concerning the separate ECoG dataset are not attributed to 2a. |
| Stieger | Dataset acquisition and release documentation [@stieger2021]. |
| Pan | Original cross-session EEG acquisition and analysis publication [@pan2023]. |
| Weibo | Original simple/compound limb-imagery acquisition publication [@yi2014]. |
| Dreyer | Original large-cohort EEG acquisition and release publication [@dreyer2023]. |

The source-trained neural decoders also used released BNCI2014-001, Cho2017, Lee2019 and PhysioNet MI records [@tangermann2012; @cho2017; @lee2019; @goldberger2000]; their source labels and model identities are inherited from the fixed source-training artifacts. These uses do not constitute new participant recruitment. The national exemption statement does not assert a new institutional approval or committee-issued waiver.

## Machine-readable tables and reproducibility

| File | Analysis unit and purpose |
| --- | --- |
| `tables/participant_metrics.csv` | Class-balanced participant/method/reference/budget summaries; main, development and source-extension roles remain distinct. |
| `tables/cohort_summary.csv` | Separate cohort means, pointwise intervals, participant directions and cancellation components; no pooled effect. |
| `tables/component_participants.csv` | Participant-paired adjacent conventional update operations. |
| `tables/component_summary.csv` | Component contrasts across all available conventional budgets and cohorts. |
| `tables/session_metrics.csv` | Available conventional query-session transition records, with actual label counts. |
| `tables/component_session_metrics.csv` | Session-level conventional component transitions. |
| `tables/neural_paired_contrasts.csv` | Neural reference-path comparisons across all three cohorts and five acquisition budgets. |
| `evidence/neural/comparisons.csv` | Original formal-stage comparison roles and tests, without changing their adjustment families. |
| `evidence/legacy/` | Original endpoint/statistical-family records retained separately from the new common reanalysis. |
| `evidence/INPUT_MANIFEST.json` | Input identities, immutable source commits, paths and SHA-256 hashes. |

Run `python analysis/analyze_integrated.py` to regenerate the complete tables and new pointwise intervals. `python analysis/check_analysis.py` checks numerical identities, preserved source records and agreement with independently recovered summaries. `python figures/plot_integrated.py` regenerates the eight scientific figures from the included plotting inputs. Run `python build_support_materials.py`, `python assemble_manuscript.py`, `python build_supplement.py` and `python build_documents.py` in that order to render the author documents. The analysis and scientific plotting scripts use local records and do not train models, submit computing jobs or upload data. Exact dependencies are listed in `requirements-analysis.txt`.

## Supplementary figures
'''
    for i in range(1,5):text+='\n\n![](figures/'+stems[i]+'.png){width=95%}\n\n'+supplementary_caps[i]+'\n'
    text+='\n## References\n'
    text=normalize_citations(text)
    (P/'Supplementary_Materials.md').write_text(text)
    print(json.dumps({'supplement_markdown_words':len(text.split()),'supplement_tables':6,'supplement_figures':4}))

if __name__=='__main__':main()
