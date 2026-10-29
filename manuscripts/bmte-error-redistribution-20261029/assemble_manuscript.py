"""Assemble one empirical argument from the separately defined evaluation designs."""
from pathlib import Path
import json,re
import pandas as pd
from project_config import *

P=Path(__file__).resolve().parent

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |'for r in rows)+'\n'

def normalize_citations(text):
    aliases=json.loads((P/'research/citation_aliases.json').read_text())
    for old,new in aliases.items():
        if old!=new:text=re.sub(r'@'+re.escape(old)+r'\b','@'+new,text)
    return text

def f(x,d=2):return f'{float(x):.{d}f}'
def signed(x):return f'{float(x):+.2f}'.replace('-','−')
def ci(row,key='G'):
    return f"[{signed(row[key+'_ci95_low_pp'])}, {signed(row[key+'_ci95_high_pp'])}]"

def main():
    s=pd.read_csv(P/'tables/cohort_summary.csv');c=pd.read_csv(P/'tables/component_summary.csv')
    claims=json.loads((P/'tables/key_claims.json').read_text())
    def summary(cohort,method,budget='full',reference='frozen'):
        q=s[(s.cohort==cohort)&(s.method==method)&(s.budget.astype(str)==str(budget))&(s.reference==reference)&(s.analysis_role=='main')]
        assert len(q)==1,(cohort,method,budget,reference,len(q));return q.iloc[0]
    def component(cohort,contrast,budget='full'):
        q=c[(c.cohort==cohort)&(c.comparison==contrast)&(c.budget.astype(str)==str(budget))&(c.analysis_role=='main')]
        assert len(q)==1;return q.iloc[0]
    k=summary('Kumar2024','full_refit');ka=summary('Kumar2024','aligned_full_refit')
    ks=component('Kumar2024','spatial_vs_classifier');ke=component('Kumar2024','alignment_vs_full')
    bs=component('BNCI2014-001','spatial_vs_classifier','20');be=component('BNCI2014-001','alignment_vs_full','20')
    bias=summary('Pan','head_bias','20');stack=summary('Pan','r4_xfit_select_stack','20')
    abstract='''**Objectives:** To characterize the corrected and introduced errors underlying accuracy changes after classifier, spatial-filter and alignment updates in motor imagery EEG recalibration.

**Methods:** A retrospective paired analysis evaluated four conventional pipelines and 18 neural configurations in five cohorts, comprising 56 participants in the main evaluation. Fixed queries supported study-specific budgets of 4–20 valid labels or acquired positions. Class-balanced error transitions were measured against frozen references, with component contrasts and alignment-reference sensitivity analyses. Cancellation was separated within participant summaries and between participants.

**Results:** Full CSP/LDA refitting in Kumar corrected 20.85 percentage points of errors and introduced 13.43 points, giving a net gain of 7.42 points. Spatial refitting beyond classifier-only updating corrected 16.04 points and introduced 10.57 points, yielding 5.47 points. In Pan, two neural configurations both achieved 63.36% balanced accuracy, with label disagreement of 2.35 versus 11.42 points. Across 48 neural configuration–cohort cells at 20 positions, median disagreement was 9.52 points and median net gain 0.68 points. Alignment profiles varied across cohorts and calibration budgets.

**Conclusions:** Decoder operations, prediction exchanges and participant variation jointly explain calibration outcomes. Reporting corrected and introduced errors reveals substantial differences concealed by net accuracy.'''
    words=len(abstract.split());assert words<=200,words
    header=(P/'submission/Title_page.md').read_text()
    intro=(P/'prose_context.md').read_text().split('## Discussion')[0].strip()
    discussion='## Discussion'+(P/'prose_context.md').read_text().split('## Discussion',1)[1]
    intro=intro.replace('the main analyses cover five evaluation cohorts and 56 participants in the main evaluation','the main comparative frame covers five evaluation cohorts and 56 participants')
    intro=intro.replace('a component-resolved account of how calibration gains arise','an empirical account of how calibration gains arise')
    methods=(P/'prose_methods.md').read_text().strip()
    table1=table(['Cohort','Decoder evaluation','Participants','Target sessions','Calibration unit'],[
        ['Kumar','Conventional',12,60,'4/8/12/16 valid labels; full 19/20'],
        ['BNCI2014-001','Conventional',9,9,'20 valid labels'],
        ['Stieger','Neural',11,105,'4/8/12/16/20 acquired positions'],
        ['Pan','Neural',14,28,'4/8/12/16/20 acquired positions'],
        ['Weibo','Neural',10,10,'4/8/12/16/20 acquired positions']])
    t2=[]
    for cohort,budget,label in [('Kumar2024','full','Kumar'),('BNCI2014-001','20','BNCI')]:
        q=summary(cohort,'full_refit',budget)
        comparisons=[('Classifier − frozen',component(cohort,'classifier_vs_frozen',budget)),
                     ('Full − frozen',q),('Full − classifier',component(cohort,'spatial_vs_classifier',budget)),
                     ('Aligned full − full',component(cohort,'alignment_vs_full',budget))]
        for title,row in comparisons:t2.append([label,title,f(row.C_pp),f(row.I_pp),f(row.D_pp),signed(row.G_pp)+' '+ci(row)])
    table2=table(['Cohort','Matched comparison','C, pp','I, pp','D, pp','G, pp [95% interval]'],t2)
    frame=s[(s.study=='neural')&(s.budget.astype(str)=='20')&(~s.method.isin(['frozen','support_ea']))]
    refrows=[]
    for ref,label in [('frozen','Frozen source'),('support_ea','Support alignment')]:
        a=frame[frame.reference==ref];assert len(a)==48
        refrows.append([label,f(a.D_pp.median()),signed(a.G_pp.median()),f(100*a.within_ratio.median()),f(100*a.between_ratio.median()),f(100*a.total_ratio.median())])
    table3=table(['Scoring reference','Median D, pp','Median G, pp','Within, %','Between, %','Total, %'],refrows)
    captext=(P/'figures/captions.md').read_text()
    captions={int(m.group(1)):m.group(0)for m in re.finditer(r'^\*\*Figure ([1-4])\..*$',captext,re.M)}
    def fig(n,stem):return f'\n\n![](figures/{stem}.png){{width=95%}}\n\n'+captions[n]+'\n\n'
    results=f'''## Results

### Evaluation coverage and paired prediction records

The main comparative frame contained 56 participants in five cohorts and 212 target sessions (Table 1). Conventional analyses paired four pipelines on fixed later-session queries; neural analyses retained all eighteen configurations and five scheduled acquisition budgets. Kumar's six development participants contributed 30 additional target sessions and were summarized separately. The source-model extension retained its own 85-person cohort.

The conventional input contained 78,600 Kumar prediction rows covering 3,930 distinct target trials and 36 BNCI prediction files covering 1,080 target trials. Methods and budgets matched exactly on ordered query IDs and labels within participant/session. The neural source table contained 3,150 participant–configuration–budget records; two scoring references produced 6,300 paired summaries. These repeated records preserve the participant units in Table 1. The identities C−I=BA difference and C+I=D held throughout the analysis.

**Table 1. Cohorts and information budgets in the main evaluation.** Sessions are target recordings within participants; labels, sessions, seeds and methods do not increase the independent participant count.

{table1}

Kumar development participants and the zero-target-label source-model extension are reported separately. Neural source checkpoints used BNCI among their training datasets; the BNCI row identifies a conventional pipeline evaluation. Budget units retain their cohort-specific definitions.

### Component comparisons locate the prediction changes behind net gains

Kumar frozen CSP/LDA achieved 57.94% BA, classifier-only recalibration 59.89%, full refitting {f(k.ba_percent)}% and alignment followed by full refitting {f(ka.ba_percent)}% (Figure 2). Full refitting corrected {f(k.C_pp)} pp of source errors while introducing {f(k.I_pp)} pp, giving {signed(k.G_pp)} pp net gain {ci(k)}. Its {f(k.D_pp)}-point disagreement was substantially larger than the final score change.

The matched-label spatial-refitting increment beyond classifier-only updating comprised {f(ks.C_pp)} points of correction and {f(ks.I_pp)} points of introduction, yielding {signed(ks.G_pp)} points {ci(ks)}. This increment represented 73.7% of the mean full-minus-frozen gain. Alignment beyond full refitting corrected {f(ke.C_pp)} points and introduced {f(ke.I_pp)} points, adding {signed(ke.G_pp)} points {ci(ke)}. Each successive operation therefore had a distinguishable error-transition profile under the same source and target fitting labels.

BNCI displayed a different component balance. The spatial-refitting increment was {signed(bs.G_pp)} points {ci(bs)}: {f(bs.C_pp)} corrected versus {f(bs.I_pp)} introduced points. Alignment beyond full refitting produced {f(be.C_pp)} corrected and {f(be.I_pp)} introduced points, adding {signed(be.G_pp)} points {ci(be)}. All nine participant alignment differences were positive. The contrast identifies the added alignment operation within that conventional protocol; the full processing paths differ across cohorts.

**Table 2. Error transitions in conventional decoder updates at full support.** C: corrected errors; I: introduced errors; D: class-balanced disagreement; G: net BA change. Intervals are the new exploratory pointwise participant-bootstrap intervals, with no additional hypothesis tests.

{table2}

Kumar full support contains 19 or 20 valid labels; BNCI support contains exactly 20. Historical endpoint intervals, primary/secondary roles and multiplicity adjustments are preserved separately in the Supplementary Materials. Frozen pipelines receive no target labels; the updating pipelines share source/support fitting examples within each cohort.

### Identical neural mean accuracy concealed different prediction exchanges

The Pan bias-only head and cross-fitted selection stack both attained {f(bias.ba_percent)}% BA, relative to 63.18% for the frozen source (Figure 3). Bias-only updating corrected {f(bias.C_pp)} pp and introduced {f(bias.I_pp)} pp; the stack corrected {f(stack.C_pp)} and introduced {f(stack.I_pp)} pp. Both consequently gained {f(bias.G_pp,4)} pp, while disagreement was {f(bias.D_pp)} versus {f(stack.D_pp)} pp, a 4.87-fold difference. The identical observed means describe this matched-score example, without establishing equivalence.

Participant outcomes also differed. Bias-only calibration improved nine participants and reduced BA in three, with two unchanged. The stack improved ten and reduced BA in three, with one unchanged. Paired participant plots retain the amount of disagreement alongside each net gain, showing which people contributed to the shared cohort mean.

### Complete neural comparisons separate within-person and between-person cancellation

Across sixteen non-anchor configurations in each of three cohorts at twenty acquired positions, median disagreement against the frozen source was 9.52 pp (IQR 4.80–12.79), whereas median signed net gain was +0.68 pp (IQR +0.39–+0.96). This fixed 48-cell descriptive frame contextualized the Pan example within the complete configuration grid. All eighteen configurations and five budgets are retained in Supplementary Figures S1–S2 and the machine-readable tables.

The total cancellation fraction had a median of 93.02%. Its within-participant-summary component had a median of 84.42%, and its between-participant component a median of 7.07% (Table 3). Additivity holds within each configuration/cohort decomposition; the separately summarized component medians need not add. Including support alignment as a seventeenth non-reference configuration gave 51 cells, median disagreement 9.62 pp and median total cancellation 93.16%.

Using support alignment as the scoring reference retained substantial opposing prediction changes: the common 48-cell frame had median disagreement 8.09 pp and median total cancellation 95.35%. The method's recorded fitting anchor was unchanged by this sensitivity reference. Zero-disagreement ratios remain undefined in the complete tables.

**Table 3. Scoring-reference sensitivity of neural cancellation at twenty acquired positions.** The same sixteen non-anchor configurations in three cohorts define both 48-cell frames. Ratios divide each cell's cancellation component by its mean disagreement. Columns report separate medians across the repeated configuration/cohort cells.

{table3}

### Calibration prefixes changed the contribution of each operation

In Kumar, the mean full-minus-classifier spatial increment was −0.08, +1.18, +2.90, +4.79 and +5.47 pp at 4, 8, 12, 16 and full support, respectively. Alignment beyond full refitting added +1.48, +4.85, +3.28, +3.15 and +5.30 pp. The scheduled curves preserve both the amount of calibration information and each operation's changing correction/introduction balance (Figure 4). The six-person development cohort remains separately visible in Supplementary Figure S3.

Weibo's recorded neural alignment pathway had BA 66.92% after four acquired positions versus 71.54% for the frozen source. The paired difference was −4.62 pp [−7.98, −1.33], with eight of ten participants below the frozen reference. Alignment-minus-source means across the five budgets were −4.62, −1.54, −0.63, −0.08 and +0.37 pp. At twenty positions, six participants improved and four decreased; the interval was [−1.01, +1.64]. Stieger and Pan did not show the same large early cohort-mean deficit. The results locate cohort-specific operating points of the recorded alignment implementations.

### A separate source-model replacement extension

On matched Dreyer observations, replacing the nine-person source model with the 60-person source model corrected 18.86 points and introduced 13.01 points, producing 31.87 points of disagreement and +5.85 points net BA change. These summaries average three fitted source seeds within each of 85 participants. The extension uses zero target-calibration labels and remains separate from the calibration-budget comparisons (Supplementary Figure S4).
'''
    # Place the figures at the argument they support rather than appending source projects.
    results=results.replace('### Component comparisons locate',fig(1,'Fig1_study_design')+'### Component comparisons locate',1)
    results=results.replace('### Identical neural mean',fig(2,'Fig2_component_error_balance')+'### Identical neural mean',1)
    results=results.replace('### Complete neural comparisons',fig(3,'Fig3_equal_accuracy_participant_changes')+'### Complete neural comparisons',1)
    results=results.replace('### A separate source-model',fig(4,'Fig4_budget_profiles')+'### A separate source-model',1)
    back=f'''## Supplementary material

Supplementary Figures S1–S4 show the complete neural configuration/budget grid, probability-score profiles, the Kumar development cohort and the source-model extension. Dataset-specific methods, complete participant and cohort tables, reference sensitivities, original endpoint test families and executable code are supplied as separate online supplementary files.

## Acknowledgments

{ACK}

## Research ethics

{ETHICS} [@nhc2023]

## Informed consent

{CONSENT}

## Author contributions

{CONTRIBUTIONS}

## Use of large language models, artificial intelligence and machine learning tools

{AI}

## Conflict of interest

{COI}

## Research funding

{FUNDING}

## Data availability

{DATA}

## References
'''
    doc=header+'\n## Abstract\n\n'+abstract+'\n\n**Keywords:** '+'; '.join(KEYWORDS)+'\n\n'+intro+'\n\n'+methods+'\n\n'+results+'\n\n'+discussion+'\n\n'+back
    doc=normalize_citations(doc)
    (P/'Manuscript.md').write_text(doc)
    (P/'prose_results.md').write_text(results)
    (P/'submission/Figure_captions.md').write_text('# Figure captions\n\n'+'\n\n'.join(captions[n]for n in range(1,5))+'\n')
    metadata={'title':TITLE,'running_head':RUNNING_HEAD,'running_head_characters':len(RUNNING_HEAD),'abstract_words_with_headings':words,'keywords':KEYWORDS,'main_markdown_words':len(doc.split()),'main_figures':4,'main_tables':3,'new_training':False,'new_pvalues':False,'submitted':False}
    (P/'research/manuscript_metrics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    plain=re.sub(r'\*\*','',abstract)
    (P/'submission/Submission_Fields.txt').write_text('TITLE\n'+TITLE+'\n\nRUNNING HEAD\n'+RUNNING_HEAD+'\n\nABSTRACT\n'+plain+'\n\nKEYWORDS (add individually)\n'+'\n'.join(KEYWORDS)+'\n')
    print(json.dumps(metadata,ensure_ascii=False))

if __name__=='__main__':main()
