from pathlib import Path
import json,re
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
ROOT=Path(__file__).resolve().parent
D=ROOT/'drafts';T=ROOT/'tables'

def table(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(str(row[c]).replace('|','/') for c in columns)+' |' for row in frame.to_dict('records')]
    return '\n'.join(lines)

front=(D/'title_abstract_backmatter.md').read_text()
front=front.replace('Inference used paired participant summaries.','Inference used paired participant summaries. Every acquisition budget shared the same subsequent query set. The matched control retained the same encoder, anchor, support data and residual objective; it directly applied the fitted correction instead of the query-specific decision.')
head,back=front.split('<!-- BACK_MATTER -->')
parts=(D/'intro_discussion.md').read_text()
intro,rest=parts.split('# Discussion',1);discussion,conclusion=rest.split('# Conclusion',1)
intro=intro.replace('# Introduction','## 1. Introduction',1)
intro=intro.replace('Session changes alter the relationship between scalp recordings and a learned decision function.','Motor imagery decoding supports EEG-based communication and control, making session-to-session adaptation a practical component of reusable BCIs [@wolpaw2002; @lotte2018]. Session changes alter the relationship between scalp recordings and a learned decision function.')
intro=intro.replace('A regularized logistic residual model defines a best support fit.','A regularized logistic residual model defines a finite feasible support fit.')
discussion=discussion.replace('### ', '### 4.',0)
discussion=discussion.replace('These fractions concern nonzero probability corrections, not the proportion of class labels that changed.','These fractions count class-1 probability changes of at least 10⁻⁴ from the alignment anchor, not the proportion of class labels that changed.')
sections=discussion.split('### ')
discussion=sections[0]+''.join(f'### 4.{i}. '+s for i,s in enumerate(sections[1:],1))
method=(D/'methods_theory.md').read_text().split('\n',1)[1]
for i in range(7,0,-1):method=method.replace(f'### 2.{i}.',f'### 2.{i+2}.')
method=method.replace('[REF:EEGNET]','[@lawhern2018]').replace('[REF:EA]','[@he2020]').replace('[REF:RKHS]','[@aronszajn1950; @hofmann2008]')
method=method.replace("ERFC uses a logistic link, as in established probability-calibration approaches [REF:PLATT], but conditions its residual function on frozen features rather than fitting a global sigmoid of the anchor margin alone.","The residual uses the same binary logistic link as the anchor and conditions on frozen features.")
method=method.replace('Weak duality then gives','Convex duality [@boyd2004] then gives')
method=method.replace('The anchor-relative minimax cross-entropy decision has the closed form','The anchor-relative decision under cross-entropy, a strictly proper scoring rule [@gneiting2007], has the closed form')
# Raw publication math uses \( \) and \[ \]; Pandoc converts these to editable Word equations.
protocol=(D/'protocol_results.md').read_text().replace('<!-- THEORY -->',method)
names={'holdout':'Stieger','pan2023':'Pan','weibo2014':'Weibo'}
cohorts=pd.DataFrame([
 {'Cohort':'Stieger historical test','Participants':11,'Sessions':105,'Session–seed cells':315,'Mean valid support at B20':'15.53'},
 {'Cohort':'Pan','Participants':14,'Sessions':28,'Session–seed cells':84,'Mean valid support at B20':'20.00'},
 {'Cohort':'Weibo','Participants':10,'Sessions':10,'Session–seed cells':30,'Mean valid support at B20':'20.00'}])
(T/'Table1_cohorts.csv').write_text(cohorts.to_csv(index=False))
comp=pd.read_csv(T/'r10_primary_comparisons_verified.csv')
rows=[]
for cohort in names:
 c=comp[comp.cohort==cohort];r={'Cohort':names[cohort]}
 r['Support alignment']=f"{100*c[c.name=='profile_vs_ea'].iloc[0].reference_mean:.2f}"
 r['Spatial SGD100']=f"{100*c[c.name=='profile_vs_spatial_sgd100'].iloc[0].reference_mean:.2f}"
 r['Anchored residual']=f"{100*c[c.name=='profile_vs_r9_ea_anchored'].iloc[0].reference_mean:.2f}"
 r['Matched MAP']=f"{100*c[c.name=='profile_vs_matched_map'].iloc[0].reference_mean:.2f}"
 r['ERFC']=f"{100*c.iloc[0].method_mean:.2f}"
 rows.append(r)
ba=pd.DataFrame(rows);ba.to_csv(T/'Table2_accuracy.csv',index=False)
x=pd.read_csv(T/'r10_exploratory_matched_map_mechanisms.csv')
rows=[]
for cohort in names:
 for metric,label in [('balanced_introduced_vs_ea','Introduced errors'),('balanced_corrected_vs_ea','Corrected errors'),('updated_query_fraction','Probability updates')]:
  r=x[(x.cohort==cohort)&(x.metric==metric)].iloc[0]
  rows.append({'Cohort':names[cohort],'Outcome':label,'ERFC − MAP (pp)':f'{r.effect_reported:+.3f}','95% CI (pp)':f'[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]','Holm p (18)':str(Decimal(str(r.p_holm_18)).quantize(Decimal('0.0001'),rounding=ROUND_HALF_UP))})
mech=pd.DataFrame(rows);mech.to_csv(T/'Table3_mechanism.csv',index=False)
# Tables are placed close to their first callout.
pos='### 2.2. Signal processing, source decoder and alignment anchor'
protocol=protocol.replace(pos,'**Table 1. Evaluation cohorts and repeated measurements.** B20 denotes 20 acquired positions before quality control. Session–seed cells are repeated measurements rather than independent participant units.\n\n'+table(cohorts)+'\n\n'+pos)
pos='### 3.2. Query-selective application of the fitted correction'
protocol=protocol.replace(pos,'**Table 2. Participant-mean balanced accuracy (%) at 20 acquired positions.** The alignment anchor, matched MAP and ERFC share their source representation. This table reports means; the full 12 paired primary tests and confidence intervals appear in Supplementary Table S1.\n\n'+table(ba)+'\n\n'+pos)
pos='### 3.4. Participant variation and session-level negative transfer'
protocol=protocol.replace(pos,'**Table 3. Additional exploratory matched-control effects.** Negative introduced-error effects denote fewer new errors; negative corrected-error effects denote fewer repaired anchor errors. Intervals are pointwise 95% participant-bootstrap intervals. Holm adjustment covers the complete 18-test post hoc family, not just the nine effects displayed here.\n\n'+table(mech)+'\n\n'+pos)
captions=(ROOT/'figures/captions.md').read_text().split('# Source and rendering records')[0]
cap={int(k):v.strip() for k,v in re.findall(r'\*\*Figure ([1-4])\. (.*?)(?=\n\n\*\*Figure|\Z)',captions,re.S)}
# Regex consumes the closing bold marker after the leading sentence; preserve the original caption style.
def figure(n,name):
 text=cap[n]
 return f'\n\n![](figures/{name}.png){{width=95%}}\n\n**Figure {n}. '+text+'\n\n'
protocol=protocol.replace('### 2.10. Comparators and recorded configuration selection',figure(1,'Fig1_ERFC_framework')+'### 2.10. Comparators and recorded configuration selection')
protocol=protocol.replace('### 3.2. Query-selective application of the fitted correction',figure(2,'Fig2_accuracy_and_budget')+'### 3.2. Query-selective application of the fitted correction')
protocol=protocol.replace('### 3.4. Participant variation and session-level negative transfer',figure(3,'Fig3_selectivity_tradeoff')+'### 3.4. Participant variation and session-level negative transfer')
protocol=protocol.replace('### 3.5. Component computation',figure(4,'Fig4_participant_tradeoffs')+'### 3.5. Component computation')
md=head+'\n\n'+intro+'\n\n'+protocol+'\n\n## 4. Discussion\n'+discussion+'\n\n## 5. Conclusions\n'+conclusion+'\n\n'+back
# Use actual threshold terminology consistently.
md=md.replace('approximately 14–15%','14.37–15.07%')
(ROOT/'Manuscript.md').write_text(md)
print(json.dumps({'manuscript_words_whitespace':len(md.split()),'tables':3,'figures':4,'abstract_words_whitespace':len(head.split('## Abstract',1)[1].split('**Keywords:**')[0].split())}))
