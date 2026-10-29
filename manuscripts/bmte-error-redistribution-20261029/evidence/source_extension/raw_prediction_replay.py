from pathlib import Path
import requests,hashlib,json
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent
sha=json.loads((P/'branch.json').read_text())['commit']['sha']
rel='research_runs/PAPER1_PLOS_FINAL_20261018/S2_Neural_Predictions.csv.gz'
f=P/'S2_Neural_Predictions.csv.gz'
if not f.exists():
 r=requests.get('https://raw.githubusercontent.com/jackzhu119/cross-subject-mi-eeg/'+sha+'/'+rel,timeout=90);r.raise_for_status();f.write_bytes(r.content)
parts=[];total=0
for x in pd.read_csv(f,chunksize=100000):
 total+=len(x)
 q=x[(x.pipeline=='BROAD')&(x.selector=='tail_min_recall')&x.source_level.isin(['S9','S60'])].copy()
 if len(q):parts.append(q)
a=pd.concat(parts,ignore_index=True)
keys=['seed','dataset','subject','trial_id']
assert total==979200 and len(a)==40800
b=a[a.source_level=='S9'].set_index(keys).sort_index();m=a[a.source_level=='S60'].set_index(keys).sort_index()
assert len(b)==len(m)==20400 and b.index.equals(m.index)
assert (b.y_true==m.y_true).all()
bp=(b.logit_right>b.logit_left).astype(int);mp=(m.logit_right>m.logit_left).astype(int)
s=pd.DataFrame({'y':b.y_true,'C':((bp!=b.y_true)&(mp==b.y_true)).astype(int),'I':((bp==b.y_true)&(mp!=b.y_true)).astype(int),'base_hit':(bp==b.y_true).astype(int),'method_hit':(mp==m.y_true).astype(int)})
s=s.reset_index()
# Equal class weight, then equal seeds within participant, then equal participants.
c=s.groupby(['subject','seed','y'])[['C','I','base_hit','method_hit']].mean().groupby(['subject','seed']).mean().groupby('subject').mean()
c['D']=c.C+c.I;c['G']=c.C-c.I
assert len(c)==85 and np.max(np.abs(c.G-(c.method_hit-c.base_hit)))<1e-12
c.to_csv(P/'primary_source_expansion_transitions_participants.csv')
means=c.mean();out={'analysis_type':'new post-hoc descriptive reanalysis of an existing frozen primary contrast; no new p-values','source_commit':sha,'source_path':rel,'source_gzip_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'input_rows_read':total,'matched_query_rows_each_configuration':len(b),'contrast':'S60 minus S9, BROAD / tail_min_recall','n_participants':len(c),'n_seeds':3,'n_trials_per_person_per_seed':80,'aggregation':'class-balanced within seed, then mean of 3 seeds per person, then equal people','frozen_primary_outcome':'weaker-class recall; this new metric was not preregistered','base_BA_percent':means.base_hit*100,'method_BA_percent':means.method_hit*100,'corrected_pp':means.C*100,'introduced_pp':means.I*100,'disagreement_pp':means.D*100,'net_BA_gain_pp':means.G*100,'total_cancellation_ratio':(means.D-abs(means.G))/means.D,'n_people_net_improved':int((c.G>1e-12).sum()),'n_people_net_worsened':int((c.G<-1e-12).sum()),'n_people_tied':int((c.G.abs()<=1e-12).sum()),'new_training':False,'new_inference':False,'new_hypothesis_tests':False,'scope':'Different source-trained models evaluated on same Dreyer observations; not target few-shot calibration and not independent of Paper1 participants.'}
(P/'primary_source_expansion_transitions.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
