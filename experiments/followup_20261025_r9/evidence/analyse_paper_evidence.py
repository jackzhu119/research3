#!/usr/bin/env python3
"""Results-backed R9 exploration. Existing reports are not fresh blind tests.
No new significance tests or selection on future query labels are performed here.
"""
import csv,json,pathlib,collections,hashlib,math,datetime
BASE=pathlib.Path(__file__).resolve().parent

def rows(n,name):return list(csv.DictReader((BASE/f'round-{n:02d}'/name).open()))
def numeric(r,fields):
 out={}
 for k in fields:
  if k not in r:continue
  v=r[k]
  if v in ('','None'):out[k]=None;continue
  try:out[k]=json.loads(v)
  except (ValueError,TypeError):out[k]=v
 return out

def main():
 audit=json.loads((BASE/'DOWNLOAD_AUDIT.json').read_text())
 all_rows={}
 for n in [2,3,7]:
  files=sorted((BASE/f'round-{n:02d}').glob('*.csv'))+sorted((BASE/f'round-{n:02d}'/'reports').glob('*.csv'))
  for p in files:all_rows[(n,str(p.relative_to(BASE/f'round-{n:02d}')))]=list(csv.DictReader(p.open()))
 quality=[]
 for (n,p),rr in all_rows.items():
  q={'round':n,'path':p,'rows':len(rr),'partitions':dict(collections.Counter(r.get('partition','holdout') for r in rr)), 'participant_counts':sorted({int(r['n_participants']) for r in rr if r.get('n_participants')})}
  for field in ['unmatched_session_seed_rows','n_excluded_participants','excluded_metric_session_seed_rows']:
   vv=[float(r[field]) for r in rr if r.get(field)]
   if vv:q[field+'_max']=max(vv)
  if rr and {'group','partition','budget','scenario'}.issubset(rr[0]) and not any(x in rr[0] for x in ['diagnostic','candidate','metric']):
   kk=[tuple(r[k] for k in ['group','partition','budget','scenario']) for r in rr]
   q['summary_key_duplicates']=len(kk)-len(set(kk))
  quality.append(q)
 selected_groups=['support_EA','static_ce:static:ce:full','static_ce:st:rel:full','static_ce:st:rel:budget','sensitivity:head-rank-0','sensitivity:head-rank-4','sensitivity:head-rank-8','sensitivity:head-prior-0.01','sensitivity:head-prior-1.0']
 fields=['group','variant','partition','budget','scenario','ba','f0_ba','nll_balanced','f0_nll_balanced','negative_flip_rate','corrected_error_rate','n_participants','all_seeds_complete']
 r7selected=[numeric(r,fields) for r in rows(7,'reports/evaluation_summaries.csv') if r['budget']=='20' and r['scenario']=='clean' and r['group'] in selected_groups]
 diag_keys=['actual_support_bce','valid_support_count','full_head_feature_dim','head_parameter_count_including_bias','centered_support_feature_rank','data_only_spatial_info_rank','head_absorbed_spatial_fraction','head_prior_precision_eta','head_norm']
 diag={g:{r['diagnostic']:float(r['mean']) for r in rows(7,'reports/diagnostics.csv') if r['group']==g and r['budget']=='20' and r['scenario']=='clean' and r['partition']=='holdout' and r['diagnostic'] in diag_keys} for g in selected_groups}
 compfields=['candidate','baseline','partition','budget','scenario','delta_ba_pp','ci95_pp','p','p_holm','p_method','positive_evidence','participant_negative_transfer_fraction','n_participants','n_paired_session_seed_rows','inference']
 primary={str(n):[numeric(r,compfields) for r in rows(n,p)] for n,p in [(2,'comparisons.csv'),(3,'reports/primary_comparisons.csv'),(7,'reports/primary_comparisons.csv')]}
 mm=rows(7,'reports/primary_multimetric_comparisons.csv')
 rank_comps=[numeric(r,compfields) for r in rows(7,'reports/sensitivity_comparisons.csv') if r['candidate'] in selected_groups]
 curves=[]
 for n,p,groups in [(2,'budget_curves.csv',['P','support_EA','head_only','f0']),(7,'reports/budget_curves.csv',['static_ce:static:ce:full','static_ce:st:rel:full','static_ce:st:rel:budget'])]:
  for r in rows(n,p):
   if r.get('group',r.get('variant')) not in groups:continue
   v=numeric(r,fields);v['round']=n;curves.append(v)
 results={'schema':'paper3-r9-evidence-review-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_repository':'jackzhu119/research3','source_branch':'paper3-results','source_snapshots':{str(n):json.loads((BASE/f'r{n}_pointer.json').read_text())['snapshot_id'] for n in [2,3,7]},'verification':{'source_files':len(audit),'verified_parts':sum(r['parts'] for r in audit),'data_tables':len(quality),'new_significance_tests_performed':False,'new_model_training_performed':False},'table_quality':quality,'r7_b20_selected':r7selected,'r7_b20_diagnostics':diag,'existing_primary_ba':primary,'r7_primary_multimetric_family':{'comparisons':len(mm),'holm_family_sizes':sorted({int(r['holm_family_size']) for r in mm}),'minimum_holm_p':min(float(r['p_holm']) for r in mm),'positive_evidence_count':sum(r['positive_evidence']=='True' for r in mm),'independent_participants':sorted({int(r['n_participants']) for r in mm}),'scope':sorted({r['inference'] for r in mm})},'r7_existing_sensitivity_comparisons':rank_comps,'selected_budget_curves':curves,'limitations':['R7 aggregate reports/report.json was skipped by previous uploader due to its 64MiB size limit; the complete 14 CSVs were verified, but these are not all per-session raw records.','R2/R3 internal holdout n=11; R7 internal n=11 and exploratory repeatedly-used Pan n=14. Seeds and sessions are not independent participants.','R3 external transfer tables and R7 cv_selection_frequency.csv are empty, not evidence of cross-dataset confirmation or CV performance.','R7 low-rank/prior curves were inspected after seeing holdout results. Existing sensitivity CIs are descriptive and not a new prespecified confirmation.','Means of diagnostics do not allow participant-level correlation or causal proof; use R7 aggregate supplementation and future compact session metrics for deeper analysis.','Negative flip/corrected error rates are balanced rates; do not interpret them as pooled raw counts.','R7 diagnostics show empirical support head-space absorption, not a theorem that spatial adaptation is globally impossible.','R8 covariance transport/commutator rotation already designed; R9 must isolate head generalization/finite-support evidence rather than rename those directions.']}
 (BASE/'FINDINGS.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'verified_files':len(audit),'verified_parts':sum(r['parts'] for r in audit),'tables':len(quality),'r7_primary_multimetric':results['r7_primary_multimetric_family'],'summary_duplicate_keys':sum(q.get('summary_key_duplicates',0) for q in quality),'unmatched_pair_max':max(q.get('unmatched_session_seed_rows_max',0) for q in quality)},indent=2))
if __name__=='__main__':main()
