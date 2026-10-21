"""Read-only accounting proof for cancelled dependents that never got nodes.

Internal recovery adapter: the caller holds the existing coordinator/data locks.
No cancellation intent is invented; this module never invokes a scheduler write.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import time

SCHEMA = 'paper3-hpc-cancelled-without-allocation-v1'
FIELDS = ('job_id', 'state', 'exit_code', 'start', 'end', 'elapsed', 'account', 'user',
          'name', 'workdir', 'alloc_nodes', 'alloc_tres', 'node_list', 'reason', 'db_index', 'submit')
NULL_TRES = {'', '(null)', 'None'}
NULL_NODES = {'', '(null)', '(unknown)', 'None', 'Unknown', 'N/A', 'None assigned'}
REASONS = {'None', 'Dependency', 'DependencyNeverSatisfied', 'Priority', 'Resources',
           'BeginTime', '', 'OTHER_REASON_NOT_RETAINED'}
FORMAT = ('JobIDRaw%64,State%64,ExitCode,Start,End,ElapsedRaw,Account%64,User%64,'
          'JobName%64,WorkDir%4096,AllocNodes,AllocTRES%4096,NodeList%4096,Reason%80,DBIndex,Submit')


def _api():
    from hpc import recover_failed_chain, start
    return recover_failed_chain, start


def _need(value, code, job_id=None, field=None):
    if not value:
        api, _ = _api()
        diagnostic = {'phase': 'cancelled_without_allocation_accounting',
                      'read_only_diagnostic': True}
        if job_id is not None: diagnostic['job_id'] = job_id
        if field is not None: diagnostic['field'] = field
        raise api.RecoveryError(code, diagnostic)


def _timestamp(value):
    api, _ = _api()
    _need(api._timestamp(value), 'never_allocated_valid_timestamp_required')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _ordered(before, after):
    try:
        return _timestamp(before) <= _timestamp(after)
    except TypeError:
        _need(False, 'never_allocated_timestamp_zone_mismatch')


def _positive_tres(value):
    if value in NULL_TRES: return False
    counts = {}
    for piece in value.split(','):
        match = re.fullmatch(r'([a-zA-Z0-9_/:.-]+)=(\d+(?:\.\d+)?[KMGTPEkmgtpe]?)', piece)
        if not match or match[1] in counts: return False
        counts[match[1]] = match[2]
    return all(re.fullmatch(r'[1-9]\d*', counts.get(key, '')) for key in ('cpu', 'node'))


def _node(value):
    return value not in NULL_NODES and re.fullmatch(r'[A-Za-z0-9_.\[\],:-]{1,4096}', value) is not None


def _validate_snapshot(binding, snapshot, user):
    api, _ = _api()
    rows = snapshot.get('rows') if isinstance(snapshot, dict) else None
    _need(isinstance(rows, list) and 3 <= len(rows) <= 4096, 'never_allocated_complete_accounting_required')
    ids = {binding['jobs'][stage]['job_id']: stage for stage in ('test','prepare','train')}
    allocations, steps, seen = {}, [], set()
    for row in rows:
        _need(isinstance(row, dict) and set(row) == set(FIELDS)
              and all(isinstance(v, str) for v in row.values()), 'never_allocated_accounting_row_invalid')
        job_id = row['job_id']
        _need(job_id not in seen, 'never_allocated_duplicate_job_history')
        seen.add(job_id)
        if job_id in ids:
            stage = ids[job_id]
            _need(row['account'] == binding['account'] and row['user'] == user
                  and row['name'] == 'paper3-' + stage and row['workdir'] == binding['project'],
                  'never_allocated_job_identity_mismatch', job_id)
            _need(re.fullmatch(r'[1-9]\d{0,19}', row['db_index']) is not None,
                  'never_allocated_accounting_database_identity_required', job_id, 'db_index')
            _need(_ordered(row['submit'], row['start']) and _ordered(row['start'], row['end']),
                  'never_allocated_accounting_time_order_invalid', job_id)
            allocations[stage] = row
        else:
            match = re.fullmatch(r'(\d+)\.(batch|extern|\d+)', job_id)
            _need(match is not None and match[1] in ids, 'never_allocated_unexpected_job_or_step')
            _need(ids[match[1]] == 'test', 'never_allocated_descendant_step_found', match[1])
            steps.append(row)
    _need(set(allocations) == {'test','prepare','train'}, 'never_allocated_all_three_allocations_required')
    _need(len({r['db_index'] for r in allocations.values()}) == 3,
          'never_allocated_database_history_ambiguous')
    parent = allocations['test']
    exit_code = re.fullmatch(r'(\d+):(\d+)', parent['exit_code'])
    _need(parent['state'] in api.TERMINAL_TEST and exit_code and any(map(int, exit_code.groups()))
          and re.fullmatch(r'[1-9]\d*', parent['elapsed']) is not None,
          'never_allocated_failed_parent_required', parent['job_id'])
    _need(re.fullmatch(r'[1-9]\d*', parent['alloc_nodes']) is not None
          and _positive_tres(parent['alloc_tres']) and _node(parent['node_list']),
          'never_allocated_positive_parent_allocation_required', parent['job_id'])
    for row in steps:
        _need(row['db_index'] == parent['db_index'] and _ordered(parent['start'], row['start'])
              and _ordered(row['start'], row['end']) and _ordered(row['start'], parent['end']),
              'never_allocated_parent_step_identity_or_time_invalid', row['job_id'])
        if not _ordered(row['end'], parent['end']):
            # Slurm records extern cleanup independently of the batch/job end.
            # Bind this exception to the same completed allocation and the
            # observed downstream cancellations, never an arbitrary grace time.
            _need(row['job_id'] == parent['job_id'] + '.extern'
                  and row['name'] == 'extern' and row['state'] == 'COMPLETED'
                  and row['exit_code'] == '0:0'
                  and row['alloc_nodes'] == parent['alloc_nodes']
                  and row['node_list'] == parent['node_list'] and _positive_tres(row['alloc_tres'])
                  and all(_ordered(row['end'], allocations[stage]['start'])
                          for stage in ('prepare', 'train')),
                  'never_allocated_parent_step_identity_or_time_invalid', row['job_id'])
    _need(any(row['job_id'].endswith(('.batch','.extern'))
              and row['state'] in api.TERMINAL_TEST | {'COMPLETED'}
              and re.fullmatch(r'\d+:\d+', row['exit_code']) is not None
              and re.fullmatch(r'[1-9]\d*', row['alloc_nodes']) is not None
              and _node(row['node_list']) and _positive_tres(row['alloc_tres'])
              and re.fullmatch(r'[1-9]\d*', row['elapsed']) is not None
              and _ordered(row['start'], row['end']) for row in steps),
          'never_allocated_parent_step_positive_control_required')
    for stage in ('prepare','train'):
        row = allocations[stage]
        _need(row['state'] == 'CANCELLED' and row['exit_code'] == '0:0' and row['elapsed'] == '0'
              and row['start'] == row['end'] and _ordered(parent['end'], row['start']),
              'never_allocated_descendant_not_zero_elapsed_cancelled', row['job_id'])
        _need(row['alloc_nodes'] == '0' and row['alloc_tres'] in NULL_TRES
              and row['node_list'] in NULL_NODES,
              'never_allocated_descendant_has_allocation_or_unknown_evidence', row['job_id'])
    _need(all(row['reason'] in REASONS for row in rows), 'never_allocated_reason_not_sanitized')
    return snapshot


def collect(binding, runner=subprocess.run, username=None):
    api, _ = _api()
    user = username or pwd.getpwuid(os.getuid()).pw_name
    ids = ','.join(binding['jobs'][s]['job_id'] for s in ('test','prepare','train'))
    command = ['sacct', '--duplicates', '--jobs=' + ids, '--noheader', '--parsable2', '--format=' + FORMAT]
    raw = api._run(command, runner)
    rows = []
    for line in raw.splitlines():
        if not line.strip(): continue
        cells = [cell.strip() for cell in line.split('|')]
        _need(len(cells) == len(FIELDS), 'never_allocated_accounting_capabilities_required')
        row = dict(zip(FIELDS, cells))
        row['state'] = api._state(row['state'])
        row['reason'] = row['reason'] if row['reason'] in REASONS else 'OTHER_REASON_NOT_RETAINED'
        rows.append(row)
    rows.sort(key=lambda r:r['job_id'])
    return _validate_snapshot(binding, {'rows':rows}, user)


def _no_queue(binding, runner):
    api, start = _api()
    ids = {binding['jobs'][s]['job_id'] for s in ('test','prepare','train')}
    raw = api._run(['squeue','--noheader','--array','--user='+str(os.getuid()),'--format=%i|%T'],runner)
    for line in raw.splitlines():
        if not line.strip(): continue
        cells = [cell.strip() for cell in line.split('|')]
        _need(len(cells)==2 and start.JOB_ID.fullmatch(cells[0]) and cells[1],
              'never_allocated_queue_record_unrecognized')
        _need(cells[0] not in ids, 'never_allocated_original_job_is_queued')


def _bind(root, binding):
    api, start = _api()
    root = start.safe(root)
    _need(root.is_dir() and isinstance(binding, dict) and binding.get('project') == str(root),
          'never_allocated_existing_project_required')
    _need(isinstance(binding.get('jobs'), dict) and set(binding['jobs']) == {'test','prepare','train'}
          and all(isinstance(binding['jobs'][stage], dict)
                  and isinstance(binding['jobs'][stage].get('job_id'), str)
                  and re.fullmatch(r'[0-9]+', binding['jobs'][stage]['job_id'])
                  for stage in ('test','prepare','train')), 'never_allocated_original_binding_invalid')
    _need(not os.environ.get('SLURM_JOB_ID'), 'never_allocated_submission_host_required')
    _need(api._bound(root,binding['jobs']['test']['job_id']) == binding,
          'never_allocated_original_binding_changed')
    for stage in ('prepare','train'):
        for suffix in ('out','err'):
            path=start.safe(root/'.paper3-hpc/logs'/(stage+'-'+binding['jobs'][stage]['job_id']+'.'+suffix))
            _need(not path.exists(),'never_allocated_descendant_execution_log_present',binding['jobs'][stage]['job_id'])
    return root


def validate_proof(root, path, value):
    api, start = _api()
    root=start.safe(root);path=start.safe(path)
    _need(path.parent==root/'.paper3-hpc/recoveries' and isinstance(value,dict)
          and value.get('schema_version')==SCHEMA
          and value.get('status')=='ACCOUNTING_CANCELLED_WITHOUT_ALLOCATION_VERIFIED'
          and value.get('project')==str(root) and type(value.get('uid')) is int
          and value['uid']==os.getuid() and value.get('user')==pwd.getpwuid(os.getuid()).pw_name
          and value.get('scientific_fit_performed') is False and value.get('scientific_budget_changed') is False
          and value.get('new_jobs_submitted') is False and value.get('queue_after')==[],
          'never_allocated_final_proof_invalid')
    binding=value.get('binding',{})
    _bind(root,binding)
    _need(path.name=='test-'+binding['jobs']['test']['job_id']+'.json','never_allocated_final_proof_path_invalid')
    snapshots=value.get('observations',[])
    _need(isinstance(snapshots,list) and len(snapshots)==2 and snapshots[0]==snapshots[1],
          'never_allocated_two_identical_observations_required')
    for snapshot in snapshots:_validate_snapshot(binding,snapshot,value['user'])
    return {'binding':binding,'job_ids':{s:binding['jobs'][s]['job_id'] for s in ('test','prepare','train')},
            'receipt':str(path.relative_to(root))}


def recover(root,binding,*,execute=False,runner=subprocess.run,username=None,sleeper=time.sleep):
    api,start=_api();root=_bind(root,binding)
    user=username or pwd.getpwuid(os.getuid()).pw_name
    _need(user==pwd.getpwuid(os.getuid()).pw_name,'never_allocated_current_owner_required')
    destination=start.safe(root/'.paper3-hpc/recoveries'/('test-'+binding['jobs']['test']['job_id']+'.json'))
    if destination.exists():
        value=start.read(destination);validate_proof(root,destination,value)
        _no_queue(binding,runner)
        return {**value,'reused_existing_proof':True}
    first=collect(binding,runner,user)
    _no_queue(binding,runner)
    if not execute:
        return {'schema_version':SCHEMA,'status':'PLAN_CANCELLED_WITHOUT_ALLOCATION_NO_WRITES',
                'binding':binding,'observation':first,'scientific_fit_performed':False,'new_jobs_submitted':False}
    sleeper(1)
    second=collect(binding,runner,user)
    _need(first==second,'never_allocated_accounting_changed_between_observations')
    _no_queue(binding,runner);_bind(root,binding)
    value={'schema_version':SCHEMA,'status':'ACCOUNTING_CANCELLED_WITHOUT_ALLOCATION_VERIFIED',
           'project':str(root),'failed_test_job':binding['jobs']['test']['job_id'],'binding':binding,
           'uid':os.getuid(),'user':user,'observations':[first,second],'queue_after':[],
           'scientific_fit_performed':False,'scientific_budget_changed':False,'new_jobs_submitted':False,
           'reason_scope':'No allocation or execution established; cancellation actor is not inferred.',
           'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    validate_proof(root,destination,value)
    _need(not destination.exists(),'never_allocated_proof_appeared')
    start.write(destination,value)
    return {**value,'receipt':str(destination.relative_to(root))}
