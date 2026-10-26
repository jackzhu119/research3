"""Reproduce participant-paired late-round evidence from pinned exact sources.
No query labels are used for new algorithm selection; this is historical analysis.
"""
import collections,csv,json,pathlib,sys,hashlib
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source'))
from paper3r9.report import bootstrap_interval,sign_flip,holm
from paper3r7.io import digest
SEEDS=[20261018,20261019,20261020]
METRICS=['ba','nll_balanced','brier_balanced','negative_flip_rate','corrected_error_rate','calibration_seconds','prediction_seconds_per_trial']
PARTITIONS=['holdout','pan2023_published_order','weibo2014_published_order']
def compare(a,b,label):
 assert set(a)==set(b),(label,set(a)^set(b))
 keys=sorted(a);d=tuple(float(a[k]-b[k]) for k in keys);scale=100 if label['metric']=='ba' else 1
 r=dict(label,n_participants=len(d),candidate_mean=float(np.mean(list(a.values()))),baseline_mean=float(np.mean(list(b.values()))),delta=float(np.mean(d))*scale,ci95=bootstrap_interval(d,scale),**sign_flip(d),unit='percentage_points' if scale==100 else 'native_metric_units',direction='candidate_minus_baseline',n_negative_participants=sum(x<0 for x in d),n_positive_participants=sum(x>0 for x in d),participant_ids=keys)
 return r

def r8():
 summaries=[json.loads(l) for l in (BASE/'round-08/analysis/condition-summaries.jsonl').open()]
 index={(v['group'],v['partition'],v['budget'],v['seed'],v['scenario']):v for v in summaries if v['family']=='core'}
 def paired_values(group,partition,budget,metric):
  vals=collections.defaultdict(list)
  for s in SEEDS:
   z=index[(group,partition,budget,s,'clean')]
   for p in z['summary']['participants']:vals[str(p['participant_id'])].append(p['ba' if metric=='ba' else 'bce'])
  assert all(len(v)==3 for v in vals.values())
  return {k:float(np.mean(v)) for k,v in vals.items()}
 report=json.loads((BASE/'round-08/reports/report.json').read_text());checks=[]
 for c in report['primary_multimetric_comparisons']:
  a=paired_values(c['candidate'],c['partition'],20,c['metric']);b=paired_values(c['baseline'],c['partition'],20,c['metric'])
  r=compare(a,b,{k:c[k] for k in ['candidate','baseline','partition','budget','metric']})
  converted_delta=r['delta'] if c['direction']=='candidate_minus_baseline' else -r['delta']
  converted_ci=r['ci95'] if c['direction']=='candidate_minus_baseline' else [-r['ci95'][1],-r['ci95'][0]]
  assert abs(converted_delta-c['delta'])<1e-9 and np.allclose(converted_ci,c['ci95'],atol=1e-9) and abs(r['p']-c['p'])<1e-12,(r,c)
  r.update(original_report_delta=c['delta'],original_report_ci95=c['ci95'],original_report_direction=c['direction'])
  r.update(p_holm=c['p_holm'],holm_family_size=40,inference='original_prespecified_exploratory_family_reproduced',seeds=SEEDS)
  checks.append(r)
 more=[]
 for group in ['fiber_bounded','fiber_fixed','head_sgd100','learned_bures_global_pooled','full_bures']:
  for part in PARTITIONS:
   for budget in [4,8,12,16,20]:
    for metric in ['ba','nll_balanced']:
     a=paired_values(group,part,budget,metric);b=paired_values('support_EA',part,budget,metric)
     r=compare(a,b,dict(candidate=group,baseline='support_EA',partition=part,budget=budget,metric=metric));r.update(inference='additional_exploratory_paired_reanalysis_no_prespecified_claim',seeds=SEEDS);more.append(r)
 (BASE/'R8_PRIMARY_REPRODUCED.json').write_text(json.dumps(checks,indent=2)+'\n')
 (BASE/'R8_ADDITIONAL_COMPARISONS.json').write_text(json.dumps(more,indent=2)+'\n')
 print('R8 all40originalprimary reproducible, minHolm',min(r['p_holm'] for r in checks),flush=True)
 return checks,more

def r9():
 plan=json.loads((BASE/'round-09/plan.json').read_text());tasks={r['key']:r for r in plan['tasks']}
 # Baseline CSV values are independently published original report rows; signatures bind pairs.
 rows=[]
 for f in sorted((BASE/'round-09/reports').glob('session_metrics-*.csv')):
  for r in csv.DictReader(f.open()):
   rows.append({**r,'budget':int(r['budget']),'seed':int(r['seed']),'metrics':{k:float(r[k]) for k in METRICS if r.get(k)}})
 for f in sorted((BASE/'r9-compact-session-metrics').glob('*.jsonl')):
  task=tasks[f.stem]
  for line in f.open():
   r=json.loads(line);r.update(group=task['group'],seed=task['seed'],source_identity=digest(r['source_identity']),data_identity=digest(r['data_identity']));rows.append(r)
 groups=collections.defaultdict(list)
 for r in rows:
  if r['scenario']=='clean':groups[(r['group'],r['partition'],r['budget'])].append(r)
 participant=[];groupmaps={};coverage=[]
 for (group,part,budget),values in groups.items():
  seeds=sorted(set(v['seed'] for v in values));coverage.append({'group':group,'partition':part,'budget':budget,'seeds':seeds,'session_seed_rows':len(values)})
  if seeds!=SEEDS:continue
  seen=set();sig={}
  for r in values:
   key=(r['participant_id'],r['session_id'],r['seed']);assert key not in seen;seen.add(key);sig[key]=(r['query_signature'],r['source_identity'],r['data_identity'])
  expected={s:len([r for r in values if r['seed']==s]) for s in SEEDS};assert len(set(expected.values()))==1
  maps={}
  for metric in METRICS:
   parts=collections.defaultdict(lambda:collections.defaultdict(list))
   for r in values:
    if metric in r['metrics']:parts[r['participant_id']][r['seed']].append(r['metrics'][metric])
   maps[metric]={pid:float(np.mean([np.mean(ss[s]) for s in SEEDS])) for pid,ss in parts.items() if sorted(ss)==SEEDS}
   for pid,v in maps[metric].items():participant.append(dict(group=group,partition=part,budget=budget,metric=metric,participant_id=pid,value=v,seeds=SEEDS))
  groupmaps[(group,part,budget)]=(maps,sig)
 comparisons=[]
 complete_methods=sorted(set(g for g,p,b in groupmaps if 'linear_' in g or 'rbf_' in g))
 for group in complete_methods:
  for part in PARTITIONS:
   for budget in [4,8,12,16,20]:
    if (group,part,budget) not in groupmaps:continue
    maps,sig=groupmaps[(group,part,budget)];base,bsig=groupmaps[('support_EA',part,budget)]
    assert sig==bsig,('query/source/data pairing mismatch',group,part,budget)
    for metric in METRICS:
     if maps[metric] and base[metric]:
      r=compare(maps[metric],base[metric],dict(candidate=group,baseline='support_EA',partition=part,budget=budget,metric=metric));r.update(seeds=SEEDS,session_seed_rows=len(sig),inference='complete_uploaded_raw_GP_vs_EA_reanalysis_original_declared_family_available_separately',original_primary_family_ready=False);comparisons.append(r)
 # Preserve the distinct posthoc linear family even as additional archives arrive.
 fam=[r for r in comparisons if r['budget']==20 and r['metric'] in ['ba','nll_balanced']
      and r['candidate'].startswith('linear_')]
 for r,p in zip(fam,holm([r['p'] for r in fam])):r.update(p_holm_available_linear_family=p,holm_family_size_available_linear=len(fam))
 (BASE/'R9_UPLOAD_COVERAGE.json').write_text(json.dumps(coverage,indent=2)+'\n')
 (BASE/'R9_LINEAR_GP_COMPARISONS.json').write_text(json.dumps([r for r in comparisons if r['candidate'].startswith('linear_')],indent=2)+'\n')
 (BASE/'R9_ALL_AVAILABLE_GP_COMPARISONS.json').write_text(json.dumps(comparisons,indent=2)+'\n')
 original=[];missing=[]
 for candidate,baseline in plan['comparisons']:
  for partition in plan['primary_partitions']:
   for metric in plan['primary_metrics']:
    label=dict(candidate=candidate,baseline=baseline,partition=partition,budget=20,metric=metric)
    if (candidate,partition,20) not in groupmaps or (baseline,partition,20) not in groupmaps:
     missing.append(dict(label,status='NOT_YET_UPLOADED_IN_VERIFIED_ARCHIVES'));continue
    a,asig=groupmaps[(candidate,partition,20)];b,bsig=groupmaps[(baseline,partition,20)]
    assert asig==bsig,('original_pair_query_identity_mismatch',label)
    value=compare(a[metric],b[metric],label)
    value.update(seeds=SEEDS,session_seed_rows=len(asig),inference='original_prespecified_R9_exploratory_family_partial_upload_coverage',original_primary_family_ready=not missing)
    original.append(value)
 padded=[r['p'] for r in original]+[1.]*len(missing)
 for r,p in zip(original,holm(padded)):
  r.update(p_holm_declared_family_with_unavailable_p1=p,declared_family_size=len(padded),
           unavailable_family_hypotheses=len(missing),original_primary_family_ready=not missing)
 (BASE/'R9_ORIGINAL_PRIMARY_AVAILABLE.json').write_text(json.dumps({'available':original,'unavailable':missing,
   'declared_family_size':len(padded),'available_family_size':len(original),
   'unavailable_handling':'No scientific effect assigned; conservative Holm uses p=1 for unuploaded hypotheses.'},indent=2)+'\n')
 with (BASE/'R9_PARTICIPANT_METRICS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(participant[0]));w.writeheader();w.writerows(participant)
 for r in fam:
  if r['metric']=='ba': print('R9',r['partition'],r['candidate'],'BA',r['candidate_mean'],'EA',r['baseline_mean'],'delta',r['delta'],'CI',r['ci95'],'p',r['p'],'Holm24',r['p_holm_available_linear_family'],flush=True)
 for r in comparisons:
  if r['candidate']=='rbf_anchored' and r['budget']==20 and r['metric'] in ('ba','nll_balanced'):
   print('R9_FRESH_RBF',json.dumps(r),flush=True)
 print('R9_ORIGINAL_PRIMARY_COVERAGE',len(original),'/',len(padded),flush=True)
 return comparisons
if __name__=='__main__':r8();r9()
