"""Reproduce R2–R7 paper evidence from pinned raw CSV/JSON/condition summaries.

Participant is independent unit. Sessions average within seed, then seed means
average within participant. Supplemental cross-round tests are retrospective,
not a new confirmatory family; primary Holm families remain historical.
"""
from pathlib import Path
from collections import defaultdict
import csv,gzip,itertools,json,math
import numpy as np
BASE=Path(__file__).resolve().parent
PIN=json.loads((BASE/'PINNED_HEAD.json').read_text())['head']
SEED=20261018
REPS=20000

def table(round_,name):
 p=BASE/('round-%02d'%round_)/name
 if not p.exists() or not p.stat().st_size:return []
 with p.open() as f:return list(csv.DictReader(f))
def value(x):
 if x in ('',None):return None
 try:return json.loads(x) if isinstance(x,str) else x
 except ValueError:return x

def statistics(people,scale=100):
 d=np.asarray([people[p] for p in sorted(people)],float);n=len(d)
 if n<2:return {'n':n,'delta':None,'ci95':None,'p_unadjusted':None}
 rng=np.random.default_rng(SEED);boot=d[rng.integers(0,n,(REPS,n))].mean(1)
 signs=np.asarray(list(itertools.product((-1,1),repeat=n)))
 p=float(np.mean(np.abs((signs*d).mean(1))>=abs(d.mean())-1e-14))
 return {'n':n,'delta':float(scale*d.mean()),'ci95':(scale*np.quantile(boot,[.025,.975])).tolist(),'p_unadjusted':p,'participant_negative_fraction':float(np.mean(d<0)),'participant_deltas':people}

def holm(values):
 order=sorted(range(len(values)),key=lambda i:values[i]);out=[None]*len(values);old=0
 for pos,i in enumerate(order):old=max(old,min(1,values[i]*(len(values)-pos)));out[i]=old
 return out

def primary_audit():
 out=[];valid=[]
 for r in range(2,8):
  rows=table(r,'comparisons.csv' if r==2 else 'reports/primary_comparisons.csv')
  extra=table(r,'reports/primary_multimetric_comparisons.csv')
  family=extra if extra else rows;p_expected=holm([float(x['p']) for x in family]);expected_by_pair={(x['candidate'],x['baseline'],x.get('metric','ba')):v for x,v in zip(family,p_expected)}
  assert all(abs(float(x['p_holm'])-v)<1e-12 for x,v in zip(family,p_expected))
  for i,row in enumerate(rows):
   rec={'round':r,'candidate':row['candidate'],'baseline':row['baseline'],'partition':row.get('partition','holdout'),'budget':int(row.get('budget') or 20),'n_participants':int(row['n_participants']),'effect_ba_pp':float(row['delta_ba_pp']),'ci95_pp':value(row['ci95_pp']),'p':float(row['p']),'p_holm':float(row['p_holm']),'holm_family_size':len(family),'completed_seeds':value(row['completed_seeds']),'candidate_ba':float(row['candidate_ba']),'baseline_ba':float(row['baseline_ba']),'source':'round-%02d/'%r+('comparisons.csv' if r==2 else 'reports/primary_comparisons.csv'),'pinned_head':PIN,'scope':'exploratory_reused_cohort_current_R10_view'}
   assert abs(rec['p_holm']-expected_by_pair[(row['candidate'],row['baseline'],'ba')])<1e-12
   if row.get('participant_deltas_ba'):
    people=value(row['participant_deltas_ba']);s=statistics(people)
    assert s['n']==rec['n_participants'];assert abs(s['delta']-rec['effect_ba_pp'])<1e-10
    assert abs(s['p_unadjusted']-rec['p'])<1e-12
    assert np.allclose(s['ci95'],rec['ci95_pp'],atol=1e-10,rtol=0)
    rec['participant_vector_recomputed']=True;rec['participant_deltas_ba']=people
   else:rec['participant_vector_recomputed']=False;rec['limitation']='R2 light report exports effect/CI/p but not participant delta vector; not re-derived from full prediction archive here.'
   out.append(rec)
  valid.append({'round':r,'primary_ba_rows':len(rows),'additional_primary_NLL_rows':sum(x.get('metric')=='nll_balanced' for x in extra),'holm_family_size':len(family),'stored_holm_reproduced':True,'paired_BA_vector_tests_reproduced':r!=2})
 return out,valid

def conditions(r):
 out={};sources={}
 for l in (BASE/('round-%02d'%r)/'analysis/condition-summaries.jsonl').open():
  x=json.loads(l);k=(x['group'],x['partition'],int(x['budget']),x['scenario'],int(x['seed']))
  assert k not in out;out[k]=x
 for l in (BASE/('round-%02d'%r)/'analysis/task-results.jsonl').open():
  x=json.loads(l);sources[x['task_key']]=x['receipt']
 return out,sources

CON={r:conditions(r) for r in(4,5,6)}

def paired_summary(c_round,c_group,b_round,b_group,partition='holdout',budget=20):
 ca,cr=CON[c_round];ba,br=CON[b_round];means=defaultdict(list);bces=defaultdict(list);nrows=0;checks=[];seeds=[]
 for seed in (20261018,20261019,20261020):
  c=ca[(c_group,partition,budget,'clean',seed)];b=ba[(b_group,partition,budget,'clean',seed)]
  cs,bs=c['summary'],b['summary'];cq={(str(z['participant_id']),str(z['session_id'])):z['query_signature'] for z in cs['expected_sessions']};bq={(str(z['participant_id']),str(z['session_id'])):z['query_signature'] for z in bs['expected_sessions']};assert cq==bq
  assert cs['data_identity']==bs['data_identity']
  assert cr[c['task_key']]['source_identity']==br[b['task_key']]['source_identity']
  cp={str(z['participant_id']):z for z in cs['participants']};bp={str(z['participant_id']):z for z in bs['participants']};assert set(cp)==set(bp)
  for p in sorted(cp):means[p].append(cp[p]['ba']-bp[p]['ba']);bces[p].append(cp[p]['bce']-bp[p]['bce'])
  nrows+=len(cq);seeds.append(seed);checks.append({'seed':seed,'candidate_source':c['source_path'],'baseline_source':b['source_path'],'query_signature_count':len(cq),'source_identity_equal':True,'data_identity_equal':True})
 people={p:float(np.mean(v)) for p,v in means.items()};bce={p:float(np.mean(v)) for p,v in bces.items()}
 return {'candidate':'R%d:%s'%(c_round,c_group),'baseline':'R%d:%s'%(b_round,b_group),'partition':partition,'budget':budget,'n_paired_session_seed_rows':nrows,'seeds':seeds,'BA':statistics(people),'balanced_BCE':statistics(bce,1),'identity_checks':checks,'inference':'retrospective exploratory audit; unadjusted p/individual bootstrap CI; not a new prespecified primary family'}

def supplemental():
 pairs=[(4,'legacy:xfit_select_stack',5,'support_EA'),(4,'legacy:xfit_select_stack',4,'legacy:sgd10'),(4,'legacy:information_protected',5,'support_EA'),(5,'single:single',5,'support_EA'),(6,'map:uq_joint',6,'support_EA'),(6,'map:uq_joint',6,'support_EA','pan2023_published_order')]
 out=[paired_summary(*args) for args in pairs]
 # Original R3 protected-vs-SGD100 is exactly the sum of two matched comparison vectors.
 p3=table(3,'reports/primary_comparisons.csv');a=next(x for x in p3 if x['candidate']=='legacy:tangent_protected' and x['baseline']=='legacy:tangent_profiled');b=next(x for x in p3 if x['candidate']=='legacy:tangent_profiled' and x['baseline']=='legacy:sgd100')
 av,bv=value(a['participant_deltas_ba']),value(b['participant_deltas_ba']);assert set(av)==set(bv)
 out.append({'candidate':'R3:legacy:tangent_protected','baseline':'R3:legacy:sgd100','partition':'holdout','budget':20,'seeds':value(a['completed_seeds']),'n_paired_session_seed_rows':int(a['n_paired_session_seed_rows']),'BA':statistics({p:av[p]+bv[p] for p in av}),'method':'sum of two same-grid published participant-paired BA difference vectors','inference':'retrospective exploratory; unadjusted p, no primary-family claim'})
 # R7 source EA versus head/spatial can be audited directly from its paired CSV.
 q7=table(7,'reports/budget_comparisons.csv');base='static_ce:static:ce:full';ea=next(x for x in q7 if x['candidate']=='support_EA' and x['baseline']==base and x['partition']=='holdout' and x['budget']=='20' and x['scenario']=='clean');eav=value(ea['participant_deltas_ba'])
 out.append({'candidate':'R7:'+base,'baseline':'R7:support_EA','partition':'holdout','budget':20,'n_paired_session_seed_rows':int(ea['n_paired_session_seed_rows']),'BA':statistics({p:-v for p,v in eav.items()}),'inference':'retrospective exploratory audit of exact published paired vector'})
 sens=table(7,'reports/sensitivity_comparisons.csv');rank0=next(x for x in sens if x['candidate']=='sensitivity:head-rank-0' and x['partition']=='holdout');rv=value(rank0['participant_deltas_ba']);st=next(x for x in q7 if x['candidate']==rank0['baseline'] and x['baseline']=='static_ce:st:ce:full' and x['partition']=='holdout' and x['budget']=='20' and x['scenario']=='clean');sv=value(st['participant_deltas_ba']);middle=next(x for x in q7 if x['candidate']=='static_ce:st:ce:full' and x['baseline']==base and x['partition']=='holdout' and x['budget']=='20' and x['scenario']=='clean');mv=value(middle['participant_deltas_ba']);assert set(rv)==set(sv)==set(mv)==set(eav)
 out.append({'candidate':'R7:sensitivity:head-rank-0','baseline':'R7:support_EA','partition':'holdout','budget':20,'n_paired_session_seed_rows':315,'BA':statistics({p:rv[p]+sv[p]+mv[p]-eav[p] for p in rv}),'method':'same-grid paired-difference algebra rank0-st_rel + st_rel-st_ce + st_ce-full - EA-full','inference':'retrospective exploratory audit; not a prespecified superiority test'})
 return out

def mechanism():
 d=table(7,'reports/diagnostics.csv');s=table(7,'reports/sensitivities.csv')+table(7,'reports/budget_curves.csv');groups=['sensitivity:head-rank-0','sensitivity:head-rank-4','sensitivity:head-rank-8','static_ce:static:ce:full'];out=[]
 names=['actual_support_bce','valid_support_count','centered_support_feature_rank','head_parameter_count_including_bias','head_absorbed_spatial_fraction','data_only_spatial_info_rank','weighted_head_span_rank']
 for g in groups:
  row=next(x for x in s if x['group']==g and x['partition']=='holdout' and x['budget']=='20' and x['scenario']=='clean')
  rec={'group':g,'BA':float(row['ba']),'balanced_NLL':float(row['nll_balanced']),'n_participants':int(row['n_participants']),'seeds':value(row['completed_seeds'])}
  for n in names:
   x=next(z for z in d if z['group']==g and z['partition']=='holdout' and z['budget']=='20' and z['scenario']=='clean' and z['diagnostic']==n);rec[n]=float(x['mean']);rec['n_session_seed_rows']=int(x['n_session_seed_rows'])
  out.append(rec)
 return out

def fixed_queries_audit():
 out=[]
 for r,(ca,cr) in CON.items():
  grouped=defaultdict(list)
  for x in ca.values():grouped[(x['task_key'],x['partition'],x['scenario'],x['seed'])].append(x)
  checked=0
  for key,values in grouped.items():
   signatures=[{(str(z['participant_id']),str(z['session_id'])):z['query_signature'] for z in v['summary']['expected_sessions']} for v in values]
   assert all(s==signatures[0] for s in signatures)
   if len(values)>1:checked+=1
  out.append({'round':r,'multi_budget_task_partition_seed_conditions_checked':checked,'all_expected_query_signatures_fixed_across_budget':True})
 return out

if __name__=='__main__':
 primary,validation=primary_audit();sup=supplemental();mech=mechanism();query=fixed_queries_audit()
 artifact={'pinned_head':PIN,'bootstrap_replicates':REPS,'statistics_seed':SEED,'statistical_unit':'participant; sessions within seed, seeds within participant; 3 seeds are not 3N independent people','CI_scope':'individual percentile bootstrap; conditional on reused cohorts/pipeline/seeds; not simultaneous','primary_comparisons':primary,'primary_multimetric_rows':[{**row,'round':r} for r in (6,7) for row in table(r,'reports/primary_multimetric_comparisons.csv')],'primary_validation':validation,'supplemental_exploratory_comparisons':sup,'mechanism_r7':mech,'fixed_query_checks':query}
 (BASE/'EFFECT_AUDIT.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
 with (BASE/'PRIMARY_EFFECTS.csv').open('w',newline='') as f:
  fields=['round','candidate','baseline','partition','budget','n_participants','effect_ba_pp','ci95_pp','p','p_holm','holm_family_size','candidate_ba','baseline_ba','participant_vector_recomputed'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in primary:w.writerow({k:json.dumps(row[k]) if isinstance(row[k],(list,dict)) else row[k] for k in fields})
 with (BASE/'SUPPLEMENTAL_EFFECTS.csv').open('w',newline='') as f:
  fields=['candidate','baseline','partition','budget','n_participants','effect_ba_pp','ci95_pp','p_unadjusted','negative_participant_fraction','scope'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in sup:w.writerow({'candidate':row['candidate'],'baseline':row['baseline'],'partition':row['partition'],'budget':row['budget'],'n_participants':row['BA']['n'],'effect_ba_pp':row['BA']['delta'],'ci95_pp':json.dumps(row['BA']['ci95']),'p_unadjusted':row['BA']['p_unadjusted'],'negative_participant_fraction':row['BA']['participant_negative_fraction'],'scope':row['inference']})
 print(json.dumps({'primary_rows':len(primary),'supplemental_rows':len(sup),'validation':validation,'supplemental':[{'candidate':x['candidate'],'baseline':x['baseline'],'partition':x['partition'],'BA':{k:v for k,v in x['BA'].items() if k!='participant_deltas'}} for x in sup],'fixed_query_checks':query}))
