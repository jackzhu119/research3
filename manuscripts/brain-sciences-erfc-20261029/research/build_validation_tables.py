from pathlib import Path
import json,shutil
import pandas as pd
P=Path('/workspace/paper3-brain-sciences-20261029');E=P/'evidence';T=P/'tables'
r12=pd.read_csv(E/'r12-pilot-primary_comparisons.csv')
cols=['id','candidate','baseline','metric','n_participants','effect_mean','effect_pp','ci95','ci95_pp','pvalue','holm_pvalue','significant_holm_05','inference_scope','budget','family_size']
r12[cols].to_csv(T/'r12_verified_primary_comparisons.csv',index=False)
r13=json.loads((E/'r13-report.json').read_text());rows=[]
for condition,stats in r13['primary']['statistics'].items():rows.append(dict(contrast=condition,**stats))
pd.DataFrame(rows).to_csv(T/'r13_verified_primary_statistics.csv',index=False)
pd.DataFrame([dict(participant=p,**v) for p,v in r13['primary']['participant_values'].items()]).to_csv(T/'r13_verified_primary_participants.csv',index=False)
d=pd.read_csv(E/'r13-full-participant_metrics.csv')
x=d[(d.information=='prefix_only') & (d.k==0)].groupby(['participant','rho','model','method'])[['BA','balanced_NLL','Brier']].mean().reset_index()
y=x.groupby(['rho','model','method']).agg(n_participants=('participant','nunique'),BA=('BA','mean'),balanced_NLL=('balanced_NLL','mean'),Brier=('Brier','mean')).reset_index();y['BA_percent']=100*y.BA;y['scope']='exploratory_historical_source_trained_pilot';y.to_csv(T/'r13_exploratory_persistence_curve.csv',index=False)
a=d[(d.model=='spatial')&(d.k>0)].groupby(['participant','rho','information','k','method'])[['BA','balanced_NLL','Brier']].mean().reset_index()
b=a.groupby(['rho','information','k','method']).agg(n_participants=('participant','nunique'),BA=('BA','mean'),balanced_NLL=('balanced_NLL','mean'),Brier=('Brier','mean')).reset_index();b['BA_percent']=100*b.BA;b['revealed_labels']=b.apply(lambda r:12+int(r.k) if r.information=='sentinel_XY' else 12,axis=1);b['elapsed_acquired_positions_before_query']=20;b['scope']='exploratory_descriptive_no_new_significance_test';b.to_csv(T/'r13_exploratory_sentinel_policies.csv',index=False)
# Already averaged seeds within participant; preserve participant rows to make plots auditable.
x.to_csv(T/'r13_persistence_participant_values.csv',index=False);a.to_csv(T/'r13_sentinel_participant_values.csv',index=False)
print('tables written',len(list(T.glob('r1[23]*.csv'))))
