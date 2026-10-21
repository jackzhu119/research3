#!/usr/bin/env python3
"""Versioned root-operated local finalization planner for deployment builder v2.

Uses the explicitly versioned software-history adapter; the original planner
and original bf8e builder are preserved. No network, R2 mutation or fit.

Default prints a plan only. --write-local-plan creates immutable local control
files, never uploads. 'post-backup' needs actual verified R2 receipt files from
root's separately executed commands. Data gates are never bypassed.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools import build_data_plan_deploy_manifest_v2 as deployment
from tools import bootstrap_data_plan as bootstrap
from tools import r2_restore
from p3work import source_extension_io as extension


def identity(row):
    return tuple(row[k] for k in extension.ENTRY_FIELDS)


def partition(root,report,extra_receipts=()):
    final=deployment.final_manifest(report)
    expected={r['path']:{**r,'io_profile':profile} for profile,batch in final['backup_manifests'].items() for r in batch['files']}
    groups=final['verified_existing_restore_groups']
    selections={g['receipt']['path']:set(g['paths']) for g in groups}
    descriptors={g['receipt']['path']:g['receipt'] for g in groups}
    for path in extra_receipts:
        item=bootstrap.file_ref(root,path);descriptors[item['path']]=item;selections[item['path']]=None
    # Explicit extras are supplied in current-first order by root. Preserve
    # that priority across post-backup recomputation; filename order must not
    # put a historical mixed receipt before an already available current one.
    priority=[]
    for extra in extra_receipts:
        path=bootstrap.file_ref(root,extra)['path']
        if path not in priority:priority.append(path)
    ordered=priority+[path for path in sorted(descriptors) if path not in priority]
    covered=set();kept=[];parsed={}
    for path in ordered:
        item=descriptors[path];bootstrap.verify_file(root,item)
        value=bootstrap.read(root/path)
        profile='source_extension' if 'source_extension_profile' in value else 'baseline'
        loader=extension._modules()[1] if profile=='source_extension' else r2_restore
        rows,sha=loader.load_verified_receipt(root/path)
        if sha!=item['sha256']:raise ValueError('receipt_changed_during_local_partition')
        selected=selections[path]
        if selected is not None and not selected<={row['path'] for row in rows}:
            raise ValueError('declared_exact_selection_not_in_receipt')
        used=[]
        for row in rows:
            if selected is not None and row['path'] not in selected:continue
            target=expected.get(row['path'])
            if target is not None and target['io_profile']==profile and identity(row)==identity(target):
                covered.add(row['path']);used.append(row['path'])
        if used:kept.append(item);parsed[path]=(profile,rows)
    # freeze_index validates ALL rows of every kept original receipt. A later
    # exact proof cannot rescue an earlier mismatching row. Force such paths
    # into the new current backup, even if another old receipt has exact bytes.
    seen_exact=set();force_current=set();mismatches=[]
    for item in kept:
        profile,rows=parsed[item['path']]
        for row in rows:
            target=expected.get(row['path'])
            if target is None or row['path'] in seen_exact:continue
            if target['io_profile']==profile and identity(row)==identity(target):seen_exact.add(row['path'])
            else:
                force_current.add(row['path'])
                mismatches.append({'path':row['path'],'receipt':item['path'],
                    'reason':'earlier_mixed_receipt_would_conflict_before_current_exact_proof'})
    missing={profile:{'schema_version':1,'files':[]} for profile in final['backup_manifests']}
    for path,row in sorted(expected.items()):
        if path not in covered or path in force_current:
            missing[row['io_profile']]['files'].append({k:row[k] for k in extension.ENTRY_FIELDS})
    return final,missing,kept,mismatches


def make_plan(root,report,output,extra_receipts=()):
    final,missing,kept,mismatches=partition(root,report,extra_receipts)
    closure=report['closure_sha256'];prefix=f'paper3/data-plan-deploy-20261020/{closure[:16]}/'
    commands=[];new_receipts=[]
    for profile,batch in missing.items():
        if not batch['files']:continue
        manifest=(output/(profile+'_uncovered_backup_manifest.json')).relative_to(root).as_posix()
        receipt=(output/(profile+'_uncovered_R2_receipt.json')).relative_to(root).as_posix()
        new_receipts.append(receipt)
        command=['python','-B','tools/r2_backup.py'] if profile=='baseline' else ['python','-B','tools/r2_source_extensions.py','backup']
        commands.append({'io_profile':profile,'objects':len(batch['files']),'bytes':sum(r['bytes'] for r in batch['files']),
            'command':command+['--manifest',manifest,'--root',str(root),'--prefix',prefix+profile+'/',
                               '--receipt',receipt,'--execute']})
    file_bytes=sum(row['bytes'] for row in report['files'])
    return final,missing,{'schema_version':'paper3-root-local-deployment-finalization-plan-v1',
        'status':'LOCAL_PLAN_ACTUAL_R2_COMMANDS_NOT_EXECUTED',
        'closure_sha256':closure,'new_backup_commands':commands,'new_receipt_paths_first':new_receipts,
        'existing_verified_receipt_paths_after_new':[r['path'] for r in kept],
        'existing_nonmatching_metadata_rows_requiring_new_coverage':mismatches,
        'restore_index_path':(output/'bootstrap_index.json').relative_to(root).as_posix(),
        'complete_deployment_file_bytes':file_bytes,'complete_deployment_files':report['file_count'],
        'fresh_full_materialization_minimum_free_bytes':file_bytes+2*1024**3,
        'fresh_receipt_streaming_option':'If full destination plus2GiB exceeds free disk, use independent empty bounded batches, preserve each exact full-GET report before deleting only verified files from the owned fresh destination. Never report the cleaned destination as fully materialized.',
        'server_offline_bootstrap_additional_free_bytes':9*1024**3,
        'current_free_bytes':shutil.disk_usage(root).free,'R2_operations':0,'scientific_fits':0,'GPU_actions':0}


def post_backup(root,report,output,extra_receipts=(),*,write=False):
    final,missing,plan=make_plan(root,report,output,extra_receipts)
    # New receipts go FIRST: old mixed receipts can contain the same metadata
    # path under a historical SHA or role. They must never select old code.
    paths=plan['new_receipt_paths_first']+plan['existing_verified_receipt_paths_after_new']
    manifest_path=output/'deploy_manifest.json'
    if not manifest_path.is_file():raise ValueError('write_and_verify_final_manifest_before_post_backup')
    if bootstrap.read(manifest_path)!=final:raise ValueError('existing_manifest_does_not_match_exact_ready_closure')
    index=bootstrap.freeze_index(root,manifest_path,[root/p for p in paths])
    index_path=output/'bootstrap_index.json'
    if write:bootstrap.write_once(index_path,index)
    elif not index_path.exists():
        return {'status':'REAL_RECEIPTS_VERIFIED_INDEX_LOCAL_WRITE_PENDING','index_sha256':index['index_sha256'],
            'all_index_files':len(index['files']),'R2_operations_this_script':0,'written':False}
    elif bootstrap.read(index_path)!=index:raise ValueError('existing_bootstrap_index_differs_from_actual_final_receipts')
    payload={r['path'] for r in index['files'] if not r['path'].startswith('data/dependency_wheels/') and not (r['path'].startswith('cache/') and Path(r['path']).suffix=='.npy')}
    payload.update(bootstrap.embedded_files(index,index_path.relative_to(root).as_posix()))
    # These original attachments are needed by the unchanged 52-file protocol
    # audit and document provenance. They are documents, never raw EEG.
    originals={
        'archives/Paper3_Work_Handoff_20261018.zip':('0398664faf34682a07fb38de4e46df9829f0e5ad050f657f905d0373dbfaf175',529763),
        'archives/Paper3_Research_Data_Plan_20261019.docx':('74383bb55cd1287ad806e5d83812e95bdb4bf0bccf7a59292b91204f230a073a',45711),
    }
    for path,(sha,size) in originals.items():
        bootstrap.verify_file(root,{'path':path,'sha256':sha,'bytes':size});payload.add(path)
    for path in ['plans/data_plan_20261019/STARTUP_ZH.md',
                 (output/'closure_report.json').relative_to(root).as_posix(),
                 (output/'root_actions.json').relative_to(root).as_posix()]:
        bootstrap.checked_path(root,path,exists=True);payload.add(path)
    files=[bootstrap.file_ref(root,p) for p in sorted(payload)]
    packaging={'schema_version':'paper3-fixed-deployment-package-inputs-v1',
        'status':'LOCAL_EXACT_PAYLOAD_LIST_NOT_AN_ARCHIVE_OR_UPLOAD',
        'closure_sha256':report['closure_sha256'],'bootstrap_index':bootstrap.file_ref(root,index_path),
        'top_level_directory':'paper3_data_plan_'+report['closure_sha256'][:16],
        'files':files,'file_count':len(files),'uncompressed_bytes':sum(r['bytes'] for r in files),
        'full_wheelhouse_embedded':False,'raw_EEG_embedded':False,'all_existing_receipts_preserved':True,
        'GPU_actions':0,'scientific_fits':0,'archive_uploaded':False}
    if write:bootstrap.write_once(output/'package_inputs.json',packaging)
    return {'status':'BOOTSTRAP_INDEX_AND_EXACT_PACKAGE_INPUTS_VALIDATED',
        'index_sha256':index['index_sha256'],'package_files':len(files),'package_input_bytes':packaging['uncompressed_bytes'],
        'all_index_files':len(index['files']),'R2_operations_this_script':0,'written':write}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--report',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('audit/data_plan_20261020/final_deployment'))
    p.add_argument('--extra-receipt',type=Path,action='append',default=[])
    p.add_argument('--stage',choices=['plan','post-backup'],default='plan')
    p.add_argument('--write-local-plan',action='store_true')
    a=p.parse_args(argv);root=a.root.resolve(strict=True)
    report=bootstrap.read(a.report) if a.report else deployment.inspect(root)
    if report.get('status')!='CACHE_CLOSURE_READY_FOR_BACKUP':
        print(json.dumps({'status':'PENDING_ALL_ACTUAL_DATA_GATES_NO_FINALIZATION',
            'gaps':report.get('gaps'),'R2_operations':0,'scientific_fits':0},indent=2));return 2
    output=a.output_dir if a.output_dir.is_absolute() else root/a.output_dir
    bootstrap.checked_path(root,output.relative_to(root).as_posix())
    if a.stage=='post-backup':
        result=post_backup(root,report,output,a.extra_receipt,write=a.write_local_plan)
    else:
        final,missing,result=make_plan(root,report,output,a.extra_receipt)
        if a.write_local_plan:
            bootstrap.write_once(output/'closure_report.json',report)
            bootstrap.write_once(output/'deploy_manifest.json',final)
            bootstrap.write_once(output/'root_actions.json',result)
            for profile,batch in missing.items():
                if batch['files']:bootstrap.write_once(output/(profile+'_uncovered_backup_manifest.json'),batch)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
