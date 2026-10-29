"""Reanalyse archived motor-imagery EEG predictions and sealed summaries.

No training, model inference or new hypothesis test is performed. Studies retain
their own cohorts, reference predictions, preprocessing and budget definitions.
All intervals created here are post hoc pointwise participant-bootstrap intervals.
Original endpoint families are retained separately under evidence/legacy and
evidence/neural/comparisons.csv; their intervals/p values are never overwritten.
"""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence';T=ROOT/'tables'
SEED=20261029;B=10000;EPS=1e-12
METRICS=['C','I','D','G','ba','reference_ba']
KMAP={'source_only':'frozen','frozen_csp_lda_recalibration':'classifier_only',
      'refit_csp_lda':'full_refit','ea_csp_lda':'aligned_full_refit'}
BMAP={'frozen':'frozen','lda_only':'classifier_only','full_csp':'full_refit','ea_csp':'aligned_full_refit'}
COMPONENTS=[('classifier_vs_frozen','classifier_only','frozen'),
            ('spatial_vs_classifier','full_refit','classifier_only'),
            ('alignment_vs_full','aligned_full_refit','full_refit')]
GROUP=['study','cohort','analysis_role','method','native_method','reference','budget','budget_kind','budget_order']
PARTKEY=GROUP+['participant']

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x)for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v)if np.isfinite(v)else None
    return v

def transitions(y,ref,pred):
    y=np.asarray(y);ref=np.asarray(ref);pred=np.asarray(pred)
    assert len(y)==len(ref)==len(pred)and set(np.unique(y))=={0,1}
    C=float(np.mean([np.mean((ref[y==c]!=c)&(pred[y==c]==c))for c in [0,1]]))
    I=float(np.mean([np.mean((ref[y==c]==c)&(pred[y==c]!=c))for c in [0,1]]))
    ba=float(np.mean([np.mean(pred[y==c]==c)for c in [0,1]]))
    rba=float(np.mean([np.mean(ref[y==c]==c)for c in [0,1]]))
    assert abs(C-I-(ba-rba))<EPS
    return dict(C=C,I=I,D=C+I,G=C-I,ba=ba,reference_ba=rba,n_query=len(y))

def available_metadata(g):
    return dict(support_available_mean=float(g.k.mean()),support_available_min=int(g.k.min()),
                support_available_max=int(g.k.max()))

def conventional():
    x=pd.read_csv(E/'conventional/kumar_query_predictions.csv.gz',dtype={'budget':str,'subject':str})
    keys=['subject','session','budget','trial_id']
    assert len(x)==78600 and not x.duplicated(keys+['method']).any()
    assert not x[['synthetic','fallback','software_validation_only']].any().any()
    assert set(x.method)==set(KMAP)and set(x.budget)=={'4','8','12','16','full'}
    for _,g in x.groupby(['subject','session']):
        assert len({tuple(zip(a.trial_id,a.label))for _,a in g.groupby(['budget','method'],sort=False)})==1
    s=[];cs=[]
    for k,g in x.groupby(['cohort','subject','session','budget'],sort=True):
        cohort,person,session,budget=k;role='main'if cohort=='remaining12'else'development'
        records={m:a.set_index('trial_id')for m,a in g.groupby('method',sort=False)}
        ref=records['source_only'];ids=ref.index;y=ref.label.to_numpy()
        meta=dict(study='conventional',cohort='Kumar2024',analysis_role=role,participant=person,
                  session=session,budget=budget,budget_kind='valid_support_labels',budget_order=20 if budget=='full'else int(budget),
                  n_seeds=1,native_cohort=cohort,**available_metadata(g))
        for native,a in records.items():
            a=a.loc[ids];assert np.array_equal(a.label.to_numpy(),y)
            s.append(dict(meta,method=KMAP[native],native_method=native,reference='frozen',
                          **transitions(y,ref.prediction,a.prediction)))
        for comparison,candidate,reference in COMPONENTS:
            cn=next(n for n,m in KMAP.items()if m==candidate);rn=next(n for n,m in KMAP.items()if m==reference)
            a=records[cn].loc[ids];r=records[rn].loc[ids]
            cs.append(dict(meta,comparison=comparison,candidate_method=candidate,reference_method=reference,
                           method=candidate,native_method=cn,reference=reference,
                           **transitions(y,r.prediction,a.prediction)))
    assert len(s)==1800 and len(cs)==1350
    bnci_rows=0
    for sub in sorted((E/'conventional/bnci_predictions').iterdir()):
        values={json.loads(f.read_text())['method']:json.loads(f.read_text())for f in sub.glob('*.json')}
        assert set(values)==set(BMAP);ref=values['frozen']
        for native,v in values.items():
            assert v['trial_ids']==ref['trial_ids']and v['y']==ref['y']and v['query_identity']==ref['query_identity']
            assert not v['synthetic']and not v['query_used_for_fitting']and len(v['y'])==120
            meta=dict(study='conventional',cohort='BNCI2014-001',analysis_role='main',participant=str(v['participant']),
                      session='E',budget='20',budget_kind='valid_support_labels',budget_order=20,n_seeds=1,native_cohort='external9',
                      support_available_mean=20.,support_available_min=20,support_available_max=20)
            s.append(dict(meta,method=BMAP[native],native_method=native,reference='frozen',
                          **transitions(v['y'],ref['prediction'],v['prediction'])))
            bnci_rows+=1
        for comparison,candidate,reference in COMPONENTS:
            cn=next(n for n,m in BMAP.items()if m==candidate);rn=next(n for n,m in BMAP.items()if m==reference)
            a=values[cn];r=values[rn]
            cs.append(dict(meta,comparison=comparison,candidate_method=candidate,reference_method=reference,
                           method=candidate,native_method=cn,reference=reference,
                           **transitions(a['y'],r['prediction'],a['prediction'])))
    assert bnci_rows==36
    session=pd.DataFrame(s);component_sessions=pd.DataFrame(cs)
    def aggregate(frame,extra=[]):
        rows=[]
        for k,g in frame.groupby(PARTKEY+extra,sort=True,dropna=False):
            row=dict(zip(PARTKEY+extra,k));row.update({m:float(g[m].mean())for m in METRICS})
            row.update(n_sessions=g.session.nunique(),n_seeds=1,n_query_total=int(g.n_query.sum()),
                       support_available_mean=float(g.support_available_mean.mean()),
                       support_available_min=float(g.support_available_min.min()),support_available_max=float(g.support_available_max.max()))
            rows.append(row)
        return pd.DataFrame(rows)
    participants=aggregate(session)
    components=aggregate(component_sessions,['comparison','candidate_method','reference_method'])
    assert len(participants)==396 and len(components)==297
    support_full=x[(x.method=='source_only')&(x.budget=='full')][['subject','session','k']].drop_duplicates().k.value_counts().to_dict()
    checks={'kumar_prediction_rows':len(x),'kumar_distinct_queries':x.trial_id.nunique(),'kumar_participants':x.subject.nunique(),
            'kumar_full_support_counts':support_full,'all_query_ids_and_labels_match':True,'bnci_prediction_files':bnci_rows,
            'bnci_queries_per_participant':120,'conventional_session_cells':len(session)}
    return session,participants,component_sessions,components,checks

def neural():
    raw=pd.read_csv(E/'neural/participant_metrics.csv',dtype={'participant_id':str})
    assert len(raw)==3150 and not raw.duplicated(['cohort','method','budget','participant_id']).any()
    assert raw.method.nunique()==18 and set(raw.budget)=={4,8,12,16,20}and raw.n_seeds.eq(3).all()
    cmap={'holdout':'Stieger','pan2023':'Pan','weibo2014':'Weibo'}
    arrays=[]
    for reference,suffix in [('frozen','frozen'),('support_ea','ea')]:
        q=raw[['cohort','method','budget','participant_id','n_seeds','ba',f'anchor_{suffix}_ba',
               f'balanced_corrected_vs_{suffix}',f'balanced_introduced_vs_{suffix}','n_valid_support']].copy()
        q=q.rename(columns={'participant_id':'participant',f'anchor_{suffix}_ba':'reference_ba',
                            f'balanced_corrected_vs_{suffix}':'C',f'balanced_introduced_vs_{suffix}':'I','n_valid_support':'support_available_mean'})
        q['native_cohort']=q.cohort;q['cohort']=q.cohort.map(cmap);q['study']='neural';q['analysis_role']='main'
        q['native_method']=q.method;q['reference']=reference;q['budget_order']=q.budget;q['budget']=q.budget.astype(str)
        q['budget_kind']='acquired_positions';q['D']=q.C+q.I;q['G']=q.C-q.I
        q['n_sessions']=q.cohort.map({'Pan':2,'Weibo':1});q['n_query_total']=np.nan
        q['support_available_min']=np.nan;q['support_available_max']=np.nan
        arrays.append(q)
    p=pd.concat(arrays,ignore_index=True)
    assert len(p)==6300 and abs(p.G-(p.ba-p.reference_ba)).max()<EPS
    return p,raw

def extension():
    x=pd.read_csv(E/'source_extension/primary_matched_predictions.csv.gz',dtype={'subject':str})
    assert len(x)==40800 and set(x.source_level)=={'S9','S60'}
    assert x.pipeline.eq('BROAD').all()and x.selector.eq('tail_min_recall').all()
    keys=['dataset','subject','seed','trial_id']
    assert not x.duplicated(keys+['source_level']).any()
    x['prediction']=np.argmax(x[['logit_left','logit_right']].to_numpy(),axis=1)
    ref=x[x.source_level=='S9'][keys+['prediction','y_true']].rename(columns={'prediction':'reference_prediction','y_true':'reference_label'})
    z=x[x.source_level=='S60'].merge(ref,on=keys,how='inner',validate='one_to_one')
    assert len(z)==20400 and z.y_true.eq(z.reference_label).all()
    seed_rows=[]
    for (person,seed),g in z.groupby(['subject','seed']):
        assert len(g)==80 and g.y_true.value_counts().to_dict()=={0:40,1:40}
        seed_rows.append(dict(participant=person,seed=seed,**transitions(g.y_true,g.reference_prediction,g.prediction)))
    seed_rows=pd.DataFrame(seed_rows)
    assert len(seed_rows)==255 and seed_rows.groupby('participant').size().eq(3).all()
    seed_rows.to_csv(T/'source_extension_seed_metrics.csv',index=False)
    d=seed_rows.groupby('participant',as_index=False)[METRICS].mean()
    old=pd.read_csv(E/'source_extension/participant_transitions.csv',dtype={'subject':str}).rename(columns={'subject':'participant','base_hit':'reference_ba','method_hit':'ba'}).set_index('participant')
    assert len(d)==85 and np.max(np.abs(d.set_index('participant')[METRICS].sort_index().to_numpy()-old[METRICS].sort_index().to_numpy()))<EPS
    d['study']='source_extension';d['cohort']='Dreyer';d['analysis_role']='extension'
    d['method']='S60';d['native_method']='S60_BROAD_tail_min_recall';d['reference']='S9'
    d['budget']='0';d['budget_order']=0;d['budget_kind']='zero_target_calibration_labels'
    d['n_seeds']=3;d['n_sessions']=np.nan;d['n_query_total']=80
    d['support_available_mean']=0.;d['support_available_min']=0.;d['support_available_max']=0.
    return d

def bootstrap_indices(p):
    rng=np.random.default_rng(SEED);samples={};people={}
    order=[('neural','Stieger','main'),('neural','Pan','main'),('neural','Weibo','main'),
           ('conventional','Kumar2024','main'),('conventional','Kumar2024','development'),
           ('conventional','BNCI2014-001','main'),('source_extension','Dreyer','extension')]
    for key in order:
        study,cohort,role=key
        ids=sorted(p[(p.study==study)&(p.cohort==cohort)&(p.analysis_role==role)].participant.unique())
        assert len(ids)>0;people[key]=ids;samples[key]=rng.integers(0,len(ids),(B,len(ids)))
    return people,samples

def summarize(p,people,samples,extra=[]):
    rows=[]
    for key,g in p.groupby(GROUP+extra,sort=True,dropna=False):
        row=dict(zip(GROUP+extra,key));scope=(row['study'],row['cohort'],row['analysis_role'])
        g=g.set_index('participant').loc[people[scope]];ix=samples[scope]
        v=g[METRICS].to_numpy(float);vm=v.mean(0);bm=v[ix].mean(1)
        for j,m in enumerate(METRICS):
            lo,hi=np.quantile(bm[:,j],[.025,.975]);row.update({m+'_mean':vm[j],m+'_ci95_low':lo,m+'_ci95_high':hi})
            if m in ['C','I','D','G']:row.update({m+'_pp':vm[j]*100,m+'_ci95_low_pp':lo*100,m+'_ci95_high_pp':hi*100})
            else:row.update({m+'_percent':vm[j]*100,m+'_ci95_low_percent':lo*100,m+'_ci95_high_percent':hi*100})
        within=float((2*np.minimum(g.C,g.I)).mean());between=float(g.G.abs().mean()-abs(g.G.mean()))
        total=float(g.D.mean()-abs(g.G.mean()))
        if abs(between)<EPS:between=0.
        if abs(total)<EPS:total=0.
        assert abs(within+between-total)<EPS and min(within,between,total)>-EPS
        bw=(2*np.minimum(g.C.to_numpy(),g.I.to_numpy()))[ix].mean(1)
        bb=np.abs(g.G.to_numpy())[ix].mean(1)-np.abs(bm[:,3]);bt=bm[:,2]-np.abs(bm[:,3])
        bb[np.abs(bb)<EPS]=0.;bt[np.abs(bt)<EPS]=0.
        assert np.max(np.abs(bw+bb-bt))<EPS
        terms={'within_cancellation':(within,bw),'between_cancellation':(between,bb),'total_cancellation':(total,bt)}
        for name,(value,draws)in terms.items():
            lo,hi=np.quantile(draws,[.025,.975]);row.update({name:value,name+'_ci95_low':lo,name+'_ci95_high':hi})
            ratio_name=name.replace('_cancellation','_ratio');ratio=value/row['D_mean']if row['D_mean']>EPS else np.nan
            good=bm[:,2]>EPS;draw_ratios=draws[good]/bm[good,2]
            lo,hi=np.quantile(draw_ratios,[.025,.975])if len(draw_ratios)else(np.nan,np.nan)
            row.update({ratio_name:ratio,ratio_name+'_ci95_low':lo,ratio_name+'_ci95_high':hi,
                        ratio_name+'_bootstrap_defined_draws':int(good.sum())})
        row.update(n_participants=len(g),n_seeds=int(g.n_seeds.iloc[0]),
                   n_participants_positive_gain=int((g.G>EPS).sum()),n_participants_negative_gain=int((g.G< -EPS).sum()),
                   n_participants_zero_gain=int((g.G.abs()<=EPS).sum()),support_available_mean=float(g.support_available_mean.mean()),
                   support_available_min=float(g.support_available_min.min()),support_available_max=float(g.support_available_max.max()),
                   bootstrap_replicates=B,statistics_seed=SEED,
                   interval_scope='post hoc pointwise participant percentile bootstrap; conditional on archived sessions, seeds and selections; no new p values')
        row['n_sessions_total']=int(g.n_sessions.sum())if g.n_sessions.notna().all()else({'Stieger':105}.get(row['cohort'],np.nan))
        rows.append(row)
    return pd.DataFrame(rows)

def paired_neural_contrasts(raw,people,samples):
    out=[];part=[];cmap={'holdout':'Stieger','pan2023':'Pan','weibo2014':'Weibo'}
    for native,cohort in cmap.items():
        ids=people[('neural',cohort,'main')];ix=samples[('neural',cohort,'main')]
        for budget in [4,8,12,16,20]:
            for contrast,candidate,reference in [('alignment_vs_frozen','support_ea','frozen'),('frozen_anchor_vs_alignment_anchor','profile_frozen_anchor','profile')]:
                a=raw[(raw.cohort==native)&(raw.budget==budget)&(raw.method==candidate)].set_index('participant_id').loc[ids]
                r=raw[(raw.cohort==native)&(raw.budget==budget)&(raw.method==reference)].set_index('participant_id').loc[ids]
                base=dict(study='neural',cohort=cohort,analysis_role='main',budget=str(budget),budget_kind='acquired_positions',
                          comparison=contrast,candidate_method=candidate,reference_method=reference,n_participants=len(ids))
                row=base.copy()
                for metric in ['ba','balanced_nll']:
                    diff=(a[metric]-r[metric]).to_numpy();lo,hi=np.quantile(diff[ix].mean(1),[.025,.975])
                    row.update({metric+'_candidate_mean':float(a[metric].mean()),metric+'_reference_mean':float(r[metric].mean()),
                                metric+'_difference_mean':float(diff.mean()),metric+'_difference_ci95_low':lo,metric+'_difference_ci95_high':hi})
                    if metric=='ba':row.update(delta_ba_pp=float(diff.mean()*100),delta_ba_ci95_low_pp=lo*100,delta_ba_ci95_high_pp=hi*100,
                                               n_positive=int((diff>EPS).sum()),n_negative=int((diff< -EPS).sum()),n_zero=int((abs(diff)<=EPS).sum()))
                for i,person in enumerate(ids):part.append(dict(base,participant=person,delta_ba=float(a.loc[person,'ba']-r.loc[person,'ba']),delta_balanced_nll=float(a.loc[person,'balanced_nll']-r.loc[person,'balanced_nll'])))
                row.update(bootstrap_replicates=B,statistics_seed=SEED,interval_scope='new descriptive pointwise participant bootstrap; no new p values');out.append(row)
    assert len(out)==30 and len(part)==350
    return pd.DataFrame(out),pd.DataFrame(part)

def main():
    T.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((E/'INPUT_MANIFEST.json').read_text())
    for r in manifest['files']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    s,p,cs,cp,checks=conventional();n,raw=neural();ex=extension()
    p=pd.concat([p,n,ex],ignore_index=True)
    for a in [p,cp,s,cs]:
        assert abs(a.D-(a.C+a.I)).max()<EPS and abs(a.G-(a.C-a.I)).max()<EPS
        assert abs(a.G-(a.ba-a.reference_ba)).max()<EPS
        assert (a[['C','I','D']]>=-EPS).all().all()and(a.D<=1+EPS).all()
        a['within_cancellation']=2*np.minimum(a.C,a.I);a['absolute_gain']=a.G.abs()
        for metric in ['C','I','D','G']:a[metric+'_pp']=100*a[metric]
    assert len(p)==6781 and len(s)==1836 and len(cp)==297 and len(cs)==1377
    assert not p.duplicated(PARTKEY).any()
    people,samples=bootstrap_indices(p)
    (ROOT/'analysis/BOOTSTRAP_SCOPE.json').write_text(json.dumps({'seed':SEED,'replicates':B,
        'generator':'numpy.random.default_rng (PCG64)','quantiles':'numpy.quantile, linear method,2.5th/97.5thpercentile',
        'cohort_draw_order':[{'study':k[0],'cohort':k[1],'analysis_role':k[2],'participant_order':people[k],
                              'indices_shape':list(samples[k].shape),'indices_sha256':hashlib.sha256(samples[k].tobytes()).hexdigest()}for k in people]},indent=2)+'\n')
    summary=summarize(p,people,samples);component_summary=summarize(cp,people,samples,['comparison','candidate_method','reference_method'])
    assert len(summary)==585 and len(component_summary)==33
    cold,coldp=paired_neural_contrasts(raw,people,samples)
    distributions=[]
    for ref in ['frozen','support_ea']:
        allcells=summary[(summary.study=='neural')&(summary.reference==ref)&(summary.budget=='20')]
        for frame,g in [('common16',allcells[~allcells.method.isin(['frozen','support_ea'])]),
                        ('all17nonreference',allcells[allcells.method!=ref])]:
            assert len(g)==(48 if frame=='common16'else 51)
            for metric in ['C_mean','I_mean','D_mean','G_mean','within_cancellation','between_cancellation','total_cancellation','within_ratio','between_ratio','total_ratio']:
                v=g[metric].dropna();q=np.quantile(v,[.25,.5,.75]);distributions.append(dict(reference=ref,frame=frame,budget='20',metric=metric,n_cells=len(g),n_defined=len(v),q25=q[0],median=q[1],q75=q[2]))
    dist=pd.DataFrame(distributions)
    for data,name in [(p,'participant_metrics.csv'),(s,'session_metrics.csv'),(summary,'cohort_summary.csv'),
                      (cp,'component_participants.csv'),(cs,'component_session_metrics.csv'),(component_summary,'component_summary.csv'),
                      (cold,'neural_paired_contrasts.csv'),(coldp,'neural_paired_contrast_participants.csv'),(dist,'neural_distributions_B20.csv')]:
        data.to_csv(T/name,index=False)
    main_ids=p[p.analysis_role=='main'][['study','cohort','participant']].drop_duplicates()
    assert len(main_ids)==56 and main_ids.cohort.nunique()==5
    main_counts=main_ids.groupby(['study','cohort']).size().to_dict()
    pan=summary[(summary.study=='neural')&(summary.cohort=='Pan')&(summary.reference=='frozen')&(summary.budget=='20')]
    bias=pan[pan.method=='head_bias'].iloc[0];stack=pan[pan.method=='r4_xfit_select_stack'].iloc[0]
    assert abs(bias.ba_mean-stack.ba_mean)<EPS
    key={'title_scope':'Error redistribution across conventional and neural motor imagery EEG recalibration',
         'analysis_type':'Post hoc archived-prediction reanalysis; separate study/cohort summaries; no pooled architecture effect',
         'counts':{'main_dataset_participants':56,'main_evaluation_cohorts':5,'kumar_development_participants_separate':6,'source_extension_participants_separate':85,
                   'main_target_sessions':212,'conventional_target_sessions_all_kumar':90,'conventional_BNCI_target_sessions':9,'neural_target_sessions':143,
                   'participant_rows':len(p),'session_rows_available':len(s),'cohort_summary_rows':len(summary),'component_participant_rows':len(cp),'component_summary_rows':len(component_summary)},
         'main_cohort_participants':{f'{k[0]}/{k[1]}':int(v)for k,v in main_counts.items()},'input_manifest':'evidence/INPUT_MANIFEST.json',
         'source_commits':manifest['source_commits'],'source_checks':checks,
         'identities':{'C_minus_I_equals_BA_difference_all_rows':True,'C_plus_I_equals_D_all_rows':True,'within_plus_between_equals_total_all_summaries':True},
         'bootstrap':{'replicates':B,'seed':SEED,'unit':'participant within study/cohort/analysis_role','shared_draws_across_methods_budgets_and_metrics':True,
                      'scope':'Pointwise post hoc descriptive; no simultaneous coverage; conditioned on stored session/seed summaries; not a new confirmatory family',
                      'ratios':'Undefined when meanD=0; bootstrap draws with zero D excluded only from that ratio, and defined-draw counts retained'},
         'new_training':False,'new_model_inference':False,'new_p_values':False,'pooled_cross_study_effect':False,
         'cases':{'Kumar_main_full_support':summary[(summary.cohort=='Kumar2024')&(summary.analysis_role=='main')&(summary.budget=='full')].to_dict('records'),
                  'Kumar_component_full_support':component_summary[(component_summary.cohort=='Kumar2024')&(component_summary.analysis_role=='main')&(component_summary.budget=='full')].to_dict('records'),
                  'BNCI_main':summary[summary.cohort=='BNCI2014-001'].to_dict('records'),
                  'BNCI_components':component_summary[component_summary.cohort=='BNCI2014-001'].to_dict('records'),
                  'Pan_equal_accuracy':{'balanced_accuracy_percent':bias.ba_percent,'bias':bias.to_dict(),'stack':stack.to_dict(),'D_ratio':stack.D_mean/bias.D_mean},
                  'Weibo_alignment_curve':cold[(cold.cohort=='Weibo')&(cold.comparison=='alignment_vs_frozen')].to_dict('records'),
                  'neural_common48_distributions':dist[dist.frame=='common16'].to_dict('records'),
                  'Dreyer_source_extension':summary[summary.study=='source_extension'].iloc[0].to_dict()},
         'legacy_inference':{'neural_stage':'evidence/neural/comparisons.csv:12 primary +24secondary_ablation +180unadjustedsecondary; original scopes unchanged',
                             'neural_independent_posthoc18':'evidence/legacy/neural_posthoc18.csv: matched-map contrast ×6outcomes ×3cohorts; separate Holm18',
                             'conventional':'Original endpoints/CI/adjustments preserved in evidence/legacy/conventional_manuscript_source.md and conventional_supplement_source.md; new C/I bootstrap intervals do not replace them'},
         'scope_notes':['Kumar full support19/20 valid labels; neural budgets count acquired positions before QC.',
                        'P3 participant summaries average original sessions/seeds; new session-level P3 data are not reconstructed from aggregates.',
                        'BNCI is a conventional evaluation cohort and a neural source-training dataset; not an unseen neural external-validation cohort.',
                        'Dreyer source-model replacement is zero target calibration and a separate extension, not an extra calibration-budget point.',
                        'Method/cohort/budget cells and repeated sessions/seeds do not create additional independent participants.']}
    (T/'key_claims.json').write_text(json.dumps(clean(key),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_brief(summary,component_summary,dist,cold,bias,stack)
    (ROOT/'analysis/SCHEMA.md').write_text('''# Analysis interface\n\n`participant_metrics.csv` retains study, cohort, analysis_role, participant, native_method, canonical method, reference, budget and budget_kind. C/I/D/G and BA fields are fractions; `_pp` fields multiply by100. Kumar budgets count valid support labels; `full` is19/20, while neural budgets count acquired positions. `budget_order` orders within a study and is not a cross-study equivalence mapping.\n\n`cohort_summary.csv` uses equal-participant means after native session/seed averaging, and `*_ci95_low/high` are new pointwise participant-bootstrap intervals. Cancellation terms and dimensionless ratios carry the same suffixes. Ratio intervals use defined draws only, with counts recorded. `ba_percent` is a percentage; `G_pp` is a percentage-point difference.\n\n`component_summary.csv` preserves three matched conventional comparisons, all budgets and development/main scopes. `neural_paired_contrasts.csv` contains cold-start and fitting-anchor contrasts, including paired BA/NLL differences.\n\nSession tables cover conventional predictions only. Neural sealed participant summaries do not support reconstructing session rows. Source-extension counts are separate. All original inferential families are preserved in evidence; the integrated survey adds no p-values.\n''')
    print(json.dumps(clean({'status':'PASS','counts':key['counts'],'Pan_D_ratio':key['cases']['Pan_equal_accuracy']['D_ratio'],'new_training':False,'new_p_values':False}),ensure_ascii=False))

def write_brief(s,c,d,cold,bias,stack):
    lines=['# Integrated empirical results: exact-value writing interface','',
           'The main analysis contains56 dataset participants in5 evaluation cohorts. Kumar development6 and Dreyer source-extension85 are separate. All new intervals below are exploratory pointwise participant bootstrap10000, seed20261029. Native historical endpoint intervals and p-values remain in their separate source tables.','',
           '## Conventional full-support comparisons against frozen decoding','',
           '| Cohort | Method | BA % | C pp | I pp | D pp | G pp [new CI] |','|---|---|---:|---:|---:|---:|---|']
    for cohort,budget in [('Kumar2024','full'),('BNCI2014-001','20')]:
        for _,r in s[(s.cohort==cohort)&(s.analysis_role=='main')&(s.budget==budget)].iterrows():
            lines.append(f'| {cohort} | {r.method} | {r.ba_percent:.5f} | {r.C_pp:.5f} | {r.I_pp:.5f} | {r.D_pp:.5f} | {r.G_pp:+.5f} [{r.G_ci95_low_pp:+.5f}, {r.G_ci95_high_pp:+.5f}] |')
    lines+=['','## Matched conventional component contrasts','', '| Cohort | Operation | C pp | I pp | G pp [new CI] |','|---|---|---:|---:|---|']
    for cohort,budget in [('Kumar2024','full'),('BNCI2014-001','20')]:
        for _,r in c[(c.cohort==cohort)&(c.analysis_role=='main')&(c.budget==budget)].iterrows():
            lines.append(f'| {cohort} | {r.comparison} | {r.C_pp:.5f} | {r.I_pp:.5f} | {r.G_pp:+.5f} [{r.G_ci95_low_pp:+.5f}, {r.G_ci95_high_pp:+.5f}] |')
    lines+=['','## Neural grid','',f'Pan bias and stack both BA={bias.ba_percent:.8f}%; D={bias.D_pp:.8f}pp and{stack.D_pp:.8f}pp; D ratio={stack.D_mean/bias.D_mean:.8f}. Equal observed means do not establish equivalence.','',
            '| Reference | Metric | Median | Q25 | Q75 |','|---|---|---:|---:|---:|']
    for _,r in d[(d.frame=='common16')&d.metric.isin(['D_mean','G_mean','within_ratio','between_ratio','total_ratio'])].iterrows():
        lines.append(f'| {r.reference} | {r.metric} | {r["median"]:.10f} | {r.q25:.10f} | {r.q75:.10f} |')
    lines+=['','## Weibo alignment minus frozen across acquired positions','', '| Budget | BA candidate % | BA reference % | G pp [CI] | Worse/better |','|---|---:|---:|---|---|']
    for _,r in cold[(cold.cohort=='Weibo')&(cold.comparison=='alignment_vs_frozen')].iterrows():
        lines.append(f'| {r.budget} | {100*r.ba_candidate_mean:.5f} | {100*r.ba_reference_mean:.5f} | {r.delta_ba_pp:+.5f} [{r.delta_ba_ci95_low_pp:+.5f}, {r.delta_ba_ci95_high_pp:+.5f}] | {r.n_negative}/{r.n_positive} |')
    ext=s[s.study=='source_extension'].iloc[0]
    lines+=['','## Separate source-model replacement extension','',f'Dreyer85: C={ext.C_pp:.8f}pp; I={ext.I_pp:.8f}pp; D={ext.D_pp:.8f}pp; G={ext.G_pp:+.8f}pp; total cancellation={100*ext.total_ratio:.8f}%. This is source-model replacement with0target calibration labels; source-trained contrasts are not pooled with target recalibration effects.','']
    (ROOT/'analysis/results_brief.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
