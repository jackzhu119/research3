"""Summarize the SHA-verified R11 read-only evidence snapshot, without models."""
import csv,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('/workspace/paper3-round10/evidence')
REUSE={
 'early/PRIMARY_EFFECTS.csv':'reused/R2_R7_PRIMARY_EFFECTS.csv',
 'early/SUPPLEMENTAL_EFFECTS.csv':'reused/R2_R7_SUPPLEMENTAL_EFFECTS.csv',
 'early/EFFECT_AUDIT.json':'reused/R2_R7_EFFECT_AUDIT.json',
 'late/R8_MAIN_COMPARISONS.csv':'reused/R8_PRIMARY_EFFECTS.csv',
 'late/R8_PRIMARY_REPRODUCED.json':'reused/R8_PRIMARY_REPRODUCED.json',
 'late/R9_B20_FRESH_RBF_COMPARISONS.csv':'reused/R9_RAW_RBF_COMPARISONS.csv',
 'late/R9_B20_AVAILABLE_LINEAR_COMPARISONS.csv':'reused/R9_RAW_LINEAR_COMPARISONS.csv',
 'late/R9_ORIGINAL_PRIMARY_AVAILABLE.json':'reused/R9_ORIGINAL_PRIMARY_AVAILABLE.json'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):return json.loads((ROOT/name).read_text())
def main():
    reused=[]
    for source,target in REUSE.items():
        src=OLD/source;dst=ROOT/target;dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
        reused.append({'source_original_local':str(src),'local':target,'sha256':digest(dst),'bytes':dst.stat().st_size,'new_eeg_results_computed':False})
    (ROOT/'REUSED_EVIDENCE.json').write_text(json.dumps({'scope':'Previously source-SHA audited compact results retained, no new independent observations','r2_r8_results_ref':'68a8d5c5714ad2a4796bfc21b9203ed29027878d','r9_raw_rbf_results_ref':'a752d00a7b56972d67e1786eb122b218828e3616','files':reused},indent=2)+'\n')
    report=load('r9/reports/report.json');full=load('r9/reports/full-report.json');primary=list(csv.DictReader((ROOT/'r9/reports/primary_comparisons-0000.csv').open()));ev=list(csv.DictReader((ROOT/'r9/reports/evaluation_summaries-0000.csv').open()))
    summary={'r9_paper_pointer_complete_tasks':load('R9_PAPER_POINTER.json')['complete_tasks'],'r9_monitor_complete_tasks':load('R9_STATUS.json')['complete_tasks'],'r9_report_actual_complete_tasks':report['complete_tasks'],'r9_report_timestamp_utc':report['updated_utc'],'r9_primary_csv_rows':len(primary),'r9_primary_csv_rows_with_participants':sum(int(x['n_participants'])>0 for x in primary),'r9_primary_csv_ready_rows':sum(x['comparison_ready']=='True' for x in primary),'r9_evaluation_csv_rows':len(ev),'r9_evaluation_groups':sorted({x['group'] for x in ev}),'r9_full_report_matches_report_scope':full['complete_tasks']==report['complete_tasks'] and full['updated_utc']==report['updated_utc'],'r10_complete_tasks':load('R10_STATUS.json')['complete_tasks'],'r10_task_phases':sorted({x['phase'] for x in load('R10_STATUS.json')['tasks']}),'r10_any_actual_attempt':any(x['attempt']>0 for x in load('R10_STATUS.json')['tasks']),'r10_has_empirical_eeg_evidence':False,'r1_completed_by_kind':load('R1_STATUS.json')['completed_by_kind'],'no_new_breakthrough_claim_supported_by_fresh_reports':True,'EA_GP_data_scope':'246group-manifests cover no EA-RBF and only seeds18/19 EA-linear-anchored; missing rawdata is unknown, not a negative result'}
    (ROOT/'EVIDENCE_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    baseline=[]
    for row in ev:
        if row['budget']=='20' and row['group'] in ('support_EA','source_frozen','head_sgd10','head_sgd100'):
            m=json.loads(row['metrics']);baseline.append({'group':row['group'],'cohort':row['partition'],'budget':20,'n_participants':m['ba']['n_participants'],'ba':m['ba']['mean'],'nll_balanced':m['nll_balanced']['mean'],'scope':'R9 actual 120-task report only; descriptive, not fresh confirmation'})
    with (ROOT/'R9_B20_BASELINES_FROM_ACTUAL_CSV.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(baseline[0]));w.writeheader();w.writerows(baseline)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
