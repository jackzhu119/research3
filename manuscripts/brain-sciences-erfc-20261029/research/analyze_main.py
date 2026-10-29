from pathlib import Path
import json,csv,itertools,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1];EV=HERE/'evidence';TABLES=HERE/'tables';TABLES.mkdir(exist_ok=True)
REF='a16d06760ae77699a809c9ba36d002d684560c82';URL='https://github.com/jackzhu119/research3/blob/'+REF+'/'
manifest=json.loads((EV/'r10-reports-manifest.json').read_text());report=json.loads((EV/'r10/reports/report.json').read_text())
r10report=next(x for x in manifest['files'] if x['source_path']=='reports/report.json');r10participants=next(x for x in manifest['files'] if x['source_path']=='reports/participant_metrics.csv')
r10src=URL+r10report['manifest'];partsources={}
fields3=['candidate','baseline','partition','budget','scenario','n_participants','delta_ba_pp','ci95_pp','baseline_ba','candidate_ba','p','p_holm','holm_family_size','n_paired_session_seed_rows','positive_evidence']
for r in ['r3','r7']:
 df=pd.read_csv(EV/r/'reports/primary_comparisons.csv');audit=json.loads((EV/'baseline_completed_audit.json').read_text());mp=audit['rounds'][r[1:]]['paper_manifest'];base=mp.rsplit('/',1)[0];source=URL+base+'/files/reports/primary_comparisons.csv'
 out=df[fields3].copy();out['source_url']=source;out.to_csv(TABLES/(r+'_primary_comparisons_verified.csv'),index=False)
 partsources[r]=source
prim=[x for x in report['comparisons'] if x.get('holm_family_name')=='primary'];out=[]
for x in prim:
 d={k:x.get(k) for k in ['cohort','name','metric','n_participants','n_seeds','session_seed_pairs','method_mean','reference_mean','effect_pp','ci95_pp','pvalue','holm_pvalue','holm_family_size','significant_holm_05']};d['candidate']='profile';d['source_url']=r10src;out.append(d)
pd.DataFrame(out).to_csv(TABLES/'r10_primary_comparisons_verified.csv',index=False)
cols=['cohort','method','budget','n_participants','ba','balanced_nll','balanced_introduced_vs_ea','balanced_corrected_vs_ea','negative_transfer_session_fraction_vs_ea','updated_query_fraction','n_valid_support','fit_seconds','incremental_prediction_seconds_per1000']
methods=pd.DataFrame(report['methods']);methods[cols].assign(source_url=r10src).to_csv(TABLES/'r10_method_budget_summaries_verified.csv',index=False)
people=pd.read_csv(EV/'r10/reports/participant_metrics.csv');selection=people[(people.budget==20)&people.method.isin(['profile','profile_map'])].copy();selection['source_url']=URL+r10participants['manifest'];selection.to_csv(TABLES/'r10_matched_map_participant_rows.csv',index=False)
new=[];paired=[]
metrics=['ba','balanced_nll','balanced_introduced_vs_ea','balanced_corrected_vs_ea','updated_query_fraction','negative_transfer_session_fraction_vs_ea']
for cohort in ['holdout','pan2023','weibo2014']:
 a=selection[(selection.cohort==cohort)&(selection.method=='profile')].set_index('participant_id');b=selection[(selection.cohort==cohort)&(selection.method=='profile_map')].set_index('participant_id');assert a.index.is_unique and b.index.is_unique;assert set(a.index)==set(b.index);ids=sorted(a.index)
 signs=np.asarray(list(itertools.product((-1,1),repeat=len(ids))))
 for metric in metrics:
  av=a.loc[ids,metric].to_numpy();bv=b.loc[ids,metric].to_numpy();d=av-bv;assert np.isfinite(d).all();obs=abs(d.mean());p=float((abs((signs*d).mean(1))>=obs-1e-14).mean());rng=np.random.default_rng(20261029);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1);ci=np.quantile(boot,[.025,.975]);scale=1 if metric=='balanced_nll' else 100
  row=dict(cohort=cohort,comparison='profile_vs_matched_map',metric=metric,n_participants=len(ids),independent_unit='participant',budget_acquired_positions=20,candidate_mean=float(av.mean()),reference_mean=float(bv.mean()),effect_mean=float(d.mean()),effect_reported=float(d.mean()*scale),unit='nats' if scale==1 else'percentage_points',ci95_low=float(ci[0]*scale),ci95_high=float(ci[1]*scale),p_two_sided=p,holm_family_size=18,bootstrap_replicates=10000,statistics_seed=20261029,inference_scope='new_exploratory_posthoc_mechanism_analysis_not_registered_primary',source_url=URL+r10participants['manifest']);new.append(row)
  paired.extend(dict(cohort=cohort,participant_id=person,metric=metric,profile=float(x),matched_map=float(y),difference=float(dd)) for person,x,y,dd in zip(ids,av,bv,d))
order=sorted(range(len(new)),key=lambda i:new[i]['p_two_sided']);previous=0
for pos,i in enumerate(order):
 previous=max(previous,min(1.,(len(new)-pos)*new[i]['p_two_sided']));new[i]['p_holm_18']=previous;new[i]['significant_holm_05']=previous<.05
pd.DataFrame(new).to_csv(TABLES/'r10_exploratory_matched_map_mechanisms.csv',index=False);pd.DataFrame(paired).to_csv(TABLES/'r10_exploratory_paired_values.csv',index=False)
r3=pd.read_csv(EV/'r3/reports/primary_comparisons.csv');r3selected=r3[r3.candidate=='legacy:tangent_protected'].iloc[0].to_dict();r3d=json.loads(r3selected['participant_deltas_ba']);pd.DataFrame([dict(participant_id=k,delta_ba_pp=100*v,source_url=partsources['r3']) for k,v in sorted(r3d.items())]).to_csv(TABLES/'r3_protection_participant_differences.csv',index=False)
value=dict(schema_version='verified-main-evidence-v1',repository='jackzhu119/research3',pinned_results_ref=REF,
 r3=dict(primary_family_size=10,protected_vs_profiled={k:r3selected[k] for k in fields3},statistical_unit='participant; sessions averaged per seed, then seed means per person',seed_count=3,seeds=[20261018,20261019,20261020],unique_sessions_per_seed=105,cohort_previously_used=True,independent_confirmation=False,source=partsources['r3']),
 r7=dict(primary_BA_comparisons=12,adjustment_family_including_multimetric_tests=24,positive_corrected_BA_tests=0,participants=14,unique_published_order_sessions=28,seeds=3,independent_confirmation=False,source=partsources['r7']),
 r10=dict(primary_family_size=12,secondary_family_size=24,positive_corrected_primary_BA_tests=sum(x['significant_holm_05'] for x in prim),primary_comparisons=out,cohorts=[dict(cohort=c,participants=n,unique_sessions=s,session_seed_pairs=s*3) for c,n,s in [('holdout',11,105),('pan2023',14,28),('weibo2014',10,10)]],independent_confirmation=report['independent_confirmation'],historically_reused_cohorts=report['exploratory_cohorts_reused_for_algorithm_design'],retrospective_protocol_revision=report['retrospective_protocol_revision'],selection_lineage=report['selection_lineage'],source_report=r10src,source_participant_table=URL+r10participants['manifest']),
 additional_mechanism_analysis=dict(status='new_exploratory_posthoc',comparison='profile vs matched MAP',metrics=metrics,holm_family=18,statistical_unit='participant',bootstrap_repeats=10000,statistics_seed=20261029,main_claim_restriction='Do not relabel as preregistered primary confirmation; report loss of corrected errors alongside reduction of introduced errors.',rows=new))
(EV/'main_verified.json').write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else x)+'\n')
for x in new:
 if x['metric']in ['balanced_introduced_vs_ea','balanced_corrected_vs_ea','updated_query_fraction','negative_transfer_session_fraction_vs_ea']:
  print(json.dumps({k:x[k] for k in ['cohort','metric','effect_reported','ci95_low','ci95_high','p_holm_18','significant_holm_05']}))
