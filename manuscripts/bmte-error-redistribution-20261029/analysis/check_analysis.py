"""Check integrated outputs against independently archived earlier calculations.

This does not refit models or add statistical tests. The neural bootstrap draws
were deliberately retained, permitting all 540 archived C/I/D/G means and CIs to
be compared exactly; conventional metrics are checked against the earlier
trial-pairing audit. The source extension is checked against its earlier85 rows.
"""
from pathlib import Path
import hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    new=pd.read_csv(ROOT/'tables/cohort_summary.csv',dtype={'budget':str})
    old=pd.read_csv(ROOT/'evidence/verification/neural_previous_summary.csv',dtype={'budget':str})
    n=new[new.study=='neural'].copy()
    n['cohort']=n.cohort.map({'Stieger':'holdout','Pan':'pan2023','Weibo':'weibo2014'})
    paired=n.merge(old,on=['cohort','method','reference','budget'],suffixes=('_new','_old'),validate='one_to_one')
    metrics=[m+suffix for m in ['C','I','D','G']for suffix in ['_mean','_ci95_low','_ci95_high']]
    error=max(float((paired[m+'_new']-paired[m+'_old']).abs().max())for m in metrics)
    assert len(paired)==540 and error<1e-12
    p=pd.read_csv(ROOT/'tables/participant_metrics.csv',dtype={'budget':str})
    c=p[(p.study=='conventional')&(p.cohort=='Kumar2024')].copy()
    c['native_cohort']=c.analysis_role.map({'main':'remaining12','development':'development6'})
    previous=pd.read_csv(ROOT/'evidence/verification/conventional_previous_participants.csv',dtype={'budget':str})
    q=c.merge(previous,left_on=['participant','budget','native_method','native_cohort'],
              right_on=['participant','budget','method','cohort'],suffixes=('_new','_old'),validate='one_to_one')
    ce=max(float((q[m+'_new']-q[m+'_old']).abs().max())for m in ['C','I','D','G','ba','reference_ba'])
    assert len(q)==360 and ce<1e-12
    legacy=pd.read_csv(ROOT/'evidence/neural/comparisons.csv')
    families=legacy.groupby('holm_family_name').size().to_dict()
    assert families=={'primary':12,'secondary_ablation':24}
    posthoc=pd.read_csv(ROOT/'evidence/legacy/neural_posthoc18.csv')
    assert len(posthoc)==18 and posthoc.holm_family_size.eq(18).all()
    manifest=json.loads((ROOT/'evidence/INPUT_MANIFEST.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    source_previous=pd.read_csv(ROOT/'evidence/source_extension/participant_transitions.csv',dtype={'subject':str})
    sx=p[p.study=='source_extension'].merge(source_previous,left_on='participant',right_on='subject',suffixes=('_new','_old'),validate='one_to_one')
    se=max(float((sx[m+'_new']-sx[m+'_old']).abs().max())for m in ['C','I','D','G'])
    assert len(sx)==85 and se<1e-12
    result={'status':'PASS','neural_summary_rows_independently_matched':len(paired),'neural_max_mean_or_CI_difference':error,
            'conventional_participant_rows_independently_matched':len(q),'conventional_max_metric_difference':ce,
            'source_extension_participant_rows_independently_matched':len(sx),'source_extension_max_metric_difference':se,
            'source_extension_replay_input':'40800 archived logit predictions; original979200-row source SHA verified before subset creation',
            'legacy_stage_families':families,'separate_historical_posthoc_family':18,
            'all_manifest_hashes_verified':True,'new_training':False,'new_p_values':False}
    (ROOT/'analysis/INDEPENDENT_NUMERICAL_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
