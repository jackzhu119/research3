"""Operational DAG and thin adapters to the existing Paper3 production kernels.

This module is stdlib-only until ``perform_action``. It neither changes an
experiment source file nor recreates completed fits. Existing job identities,
scientific output paths and numerical checkpoint implementations remain native.
"""
from __future__ import annotations

import ast
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import sys

OUTPUT = 'runs/training_campaign_v1'
OPERATIONAL_OUTPUT = 'runs/training_campaign_parallel_v1'
METHODS = tuple('B%d'%n for n in range(12)) + ('P', 'P+PCT')
LEARNED = tuple(m for m in METHODS if m in {'P','P+PCT','B8','B10','B11'})
SEEDS = (20261018, 20261019, 20261020)
ARMS = ('BASE4_PLUS_YANG', 'BASE4_PLUS_DREYER')
STATUS = {'meta_engineering':'META_ENGINEERING_COMPLETE', 'meta_screening':'META_JOB_COMPLETE',
    'meta_confirmation':'META_JOB_COMPLETE', 'meta_selection':'META_CONFIGURATION_FROZEN',
    'calibration_profile':'DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE',
    'development':'DEVELOPMENT_EVALUATION_COMPLETE',
    'development_selection':'DEVELOPMENT_CONFIGURATION_FROZEN',
    'strongest_control':'STRONGEST_DEVELOPMENT_CONTROL_FROZEN',
    'freeze':'IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE', 'internal':'ONE_TIME_INTERNAL_CHECK_COMPLETE',
    'extension_initialize':None, 'extension_development':'SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN'}
_READ_CACHE = ContextVar('paper3_parallel_planning_read_cache',default=None)
_MISSING = object()


def _read(path, default=None):
    cache = _READ_CACHE.get()
    key = str(path)
    if cache is not None and key in cache:
        return default if cache[key] is _MISSING else cache[key]
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError:
        value = _MISSING
    if cache is not None:
        cache[key] = value
    return default if value is _MISSING else value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _grid_function(root):
    """Compile only the original pure registered-grid function, without Torch.

    The numerical implementation is never compiled or replaced here. Keeping
    this metadata function in its original source avoids hand-copying its 118
    setting declaration into an operational scheduler.
    """
    source = Path(root)/'p3work/method_registry.py'
    tree = ast.parse(source.read_bytes())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name=='registered_grid']
    if len(nodes) != 1:
        raise ValueError('Original registered_grid metadata function missing')
    namespace = {'product':product}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['registered_grid']


def _meta_job(method, configuration, seed=SEEDS[0], phase=None):
    return {'method':method, 'dimension':configuration.get('dimension',configuration.get('rank')),
        'radius':configuration['radius'], 'seed':seed,
        'phase':phase or ('screening' if seed==SEEDS[0] else 'confirmation')}


def _development_job(method, index, configuration, seed=SEEDS[0]):
    job = {'method':method, 'configuration_index':index, 'seed':seed,
        'phase':'screening' if seed==SEEDS[0] else 'confirmation'}
    identity = _digest({**job,'configuration':configuration,'schema_version':'paper3-development-evaluation-v1'})
    return job, identity


def _action(kind, key, receipt, *, serial=False, affinity_slot=None, **fields):
    return {'kind':kind,'key':key,'expected_receipt':receipt,'serial':serial,
            'affinity_slot':affinity_slot, **fields}


def action_complete(root, action):
    value = _read(Path(root)/action['expected_receipt'], {})
    if action['kind'] == 'extension_initialize':
        return isinstance(value,dict) and value.get('arm')==action['arm'] and value.get('schema_version') is not None
    return isinstance(value,dict) and value.get('status')==STATUS[action['kind']]


def _meta_action(kind, job):
    identity = _digest(job)
    return _action(kind,'meta-'+identity,'runs/meta/'+identity+'/receipt.json',method=job['method'],job=job)


def _phase(name, actions, root, *, blocked=None, mainline_complete=False):
    completed = [a for a in actions if action_complete(root,a)]
    missing = [a for a in actions if not action_complete(root,a)]
    return {'phase':name,'ready':missing,'ready_actions':missing,'phase_actions':actions,
        'phase_completed':len(completed),'phase_total':len(actions),
        'complete_counts':dict(Counter(a['kind'] for a in completed)),
        'total_counts':dict(Counter(a['kind'] for a in actions)),
        'blocked':blocked or [],'mainline_complete':mainline_complete,'all_complete':False,
        'scientific_outputs_reused_in_place':True,
        'device_policy':'meta may transfer across equivalent GPUs; all primary calibration/development stays on slot 0'}


def plan_ready(root, slot=None):
    token = _READ_CACHE.set({})
    try:
        return _plan_ready(root,slot)
    finally:
        _READ_CACHE.reset(token)


def _plan_ready(root, slot=None):
    """Read-only phase frontier; no package imports, data payloads or rehashes.

    Strict phase barriers keep catalog hashes fixed for each evaluation phase.
    The original campaign's learned-artifact inventory grows after meta fits;
    pipelining their completion with evaluations would change immutable budget
    evidence on restart. Primary timing/comparison jobs retain one GPU lineage.
    """
    root = Path(root)
    prerequisites = [
        ('runs/controlled_source_pilot_seed20261018/engineering_receipt.json','ENGINEERING_PILOT_COMPLETE'),
        ('runs/source_training_v2/source_profile_development.json','SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING'),
        ('runs/source_classical_v2/receipt.json','SOURCE_CLASSICAL_CORE_COMPLETE'),
    ] + [(OUTPUT+'/references/seed-%s/receipt.json'%seed,'SOURCE_REFERENCE_COMPLETE') for seed in SEEDS]
    gaps = [p for p,s in prerequisites if (_read(root/p,{}) or {}).get('status')!=s]
    if gaps:
        return _phase('EXISTING_PREREQUISITES_PENDING',[],root,blocked=gaps)
    grid = _grid_function(root)
    engineering, screening, selections = [], [], []
    for method in LEARNED:
        engineering_job = _meta_job(method, {'rank':4 if method in {'B10','B11'} else 8,'radius':.35},phase='engineering')
        eng = _meta_action('meta_engineering',engineering_job)
        engineering.append(eng)
        for configuration in grid(method):
            screening.append(_meta_action('meta_screening',_meta_job(method,configuration)))
        selections.append(_action('meta_selection','meta-selection-'+method,
            OUTPUT+'/meta_selections/'+method+'.json',method=method,serial=True,affinity_slot=0))
    missing_eng = [a for a in engineering if not action_complete(root,a)]
    ready_screen = [a for a in screening if action_complete(root,
        next(e for e in engineering if e['method']==a['method'])) and not action_complete(root,a)]
    ready_selections = [a for a in selections if all(action_complete(root,s)
        for s in screening if s['method']==a['method']) and not action_complete(root,a)]
    if missing_eng or any(not action_complete(root,a) for a in screening+selections):
        result = _phase('META_SCREENING',engineering+screening+selections,root)
        result['ready'] = result['ready_actions'] = missing_eng + ready_screen + ready_selections
        return _for_slot(result,slot)
    profiles, development, development_selections = [], [], []
    for method in METHODS:
        profiles.append(_action('calibration_profile','profile-'+method,
            OUTPUT+'/profiles/'+method+'/receipt.json',method=method,affinity_slot=0))
        for index, configuration in enumerate(grid(method)):
            job, identity = _development_job(method,index,configuration)
            development.append(_action('development','development-'+identity,
                OUTPUT+'/development/'+identity+'/receipt.json',method=method,job=job,affinity_slot=0))
        development_selections.append(_action('development_selection','selection-'+method,
            OUTPUT+'/selections/'+method+'.json',method=method,serial=True,affinity_slot=0))
    if any(not action_complete(root,a) for a in profiles+development+development_selections):
        result = _phase('DEVELOPMENT_SCREENING',profiles+development+development_selections,root)
        ready = [a for a in profiles if not action_complete(root,a)]
        ready += [a for a in development if not action_complete(root,a) and action_complete(root,
            next(p for p in profiles if p['method']==a['method']))]
        ready += [a for a in development_selections if not action_complete(root,a) and
                  all(action_complete(root,d) for d in development if d['method']==a['method'])]
        result['ready'] = result['ready_actions'] = ready
        return _for_slot(result,slot)
    confirmations, confirmed_development, disagreement = [], [], []
    for method in METHODS:
        selected_path = OUTPUT+'/selections/'+method+'.json'
        selected = _read(root/selected_path)
        index, configuration = selected['configuration_index'], selected['configuration']
        if method in LEARNED:
            meta_path = OUTPUT+'/meta_selections/'+method+'.json'
            meta = _read(root/meta_path)
            expected = configuration.get('dimension',configuration.get('rank'))
            if meta.get('dimension') != expected or meta.get('radius') != configuration.get('radius'):
                disagreement.append('META_AND_DEVELOPMENT_SELECTION_DISAGREE:'+method)
                continue
        for seed in SEEDS[1:]:
            if method in LEARNED:
                meta_action = _meta_action('meta_confirmation',_meta_job(method,configuration,seed))
                meta_action['meta_selection'] = meta_path
                confirmations.append(meta_action)
            job, identity = _development_job(method,index,configuration,seed)
            item = _action('development','development-'+identity,OUTPUT+'/development/'+identity+'/receipt.json',
                method=method,job=job,affinity_slot=0)
            # Original descriptor SHA is derived only when executing this task.
            item['configuration_receipt_path'] = selected_path
            confirmed_development.append(item)
    if any(not action_complete(root,a) for a in confirmations):
        return _for_slot(_phase('META_CONFIRMATION',confirmations,root,blocked=disagreement),slot)
    if disagreement:
        return _phase('SCIENTIFIC_SELECTION_REQUIRES_REVIEW',[],root,blocked=disagreement)
    if any(not action_complete(root,a) for a in confirmed_development):
        return _for_slot(_phase('DEVELOPMENT_CONFIRMATION',confirmed_development,root),slot)
    final = [
        _action('strongest_control','strongest-control',OUTPUT+'/strongest_control.json',serial=True,affinity_slot=0),
        _action('freeze','freeze',OUTPUT+'/pre_internal_freeze.json',serial=True,affinity_slot=0),
        _action('internal','internal','runs/internal_validation_once/receipt.json',serial=True,affinity_slot=0),
    ]
    for action in final:
        if not action_complete(root,action):
            return _for_slot(_phase('MAINLINE_FINALIZATION',[action],root),slot)
    extensions = []
    for slot_index, arm in enumerate(ARMS,1):
        launch = 'runs/source_extension_downstream/'+arm+'/launch.json'
        initialize = _action('extension_initialize','extension-initialize-'+arm,launch,
            arm=arm,affinity_slot=slot_index)
        if not action_complete(root,initialize):
            extensions.append(initialize)
        else:
            extensions.append(_action('extension_development','extension-development-'+arm,
                'runs/source_extension_downstream/'+arm+'/pre_internal_development_freeze.json',
                arm=arm,affinity_slot=slot_index))
    result = _phase('EXISTING_SOURCE_EXTENSIONS',extensions,root,mainline_complete=True)
    if not result['ready_actions']:
        result.update(phase='ALL_ORIGINAL_WORK_COMPLETE',all_complete=True)
    return _for_slot(result,slot)


def _for_slot(result,slot):
    if slot is not None:
        if slot not in (0,1,2):
            raise ValueError('Registered operational worker slot required')
        result = dict(result)
        result['ready'] = result['ready_actions'] = [a for a in result['ready_actions']
            if a.get('affinity_slot') in (None,slot)]
    return result


def activate_runtime(root, payload_manifest, *, direct_package=None, allocation_seconds=529200):
    """Activate already deployed operational overlays and original root code."""
    root, payload_manifest = Path(root).resolve(), Path(payload_manifest).resolve()
    paths = [str(payload_manifest.parent), str(root)]
    if direct_package is not None:
        paths.insert(0,str(Path(direct_package).resolve()))
    sys.path[:0] = [p for p in paths if p not in sys.path]
    os.environ.update(PAPER3_HPC_ROOT=str(root), PAPER3_HPC_STAGE='train',
        PAPER3_HPC_RUNTIME_PROFILE='school-cu121-v1',
        PAPER3_EPISODE_JSON_COMPAT_ROOT=str(root), PAPER3_EPISODE_JSON_COMPAT_ENABLED='1',
        PAPER3_EPISODE_JSON_COMPAT_SOURCE_SHA256='6ee44b0adafdf080885be943a45a3df84cb7407c602a64bf8dcd549746f092cb',
        PAPER3_HPC_RUN_UNTIL_COMPLETE='1', PAPER3_HPC_WALL_TIME_SECONDS=str(allocation_seconds),
        PAPER3_CONTINUOUS_REPAIR_MANIFEST=str(payload_manifest),
        PAPER3_RUNTIME_TRANSFER_ENABLED='1', PAPER3_RUNTIME_TRANSFER_ROOT=str(root),
        PAPER3_RUNTIME_TRANSFER_MANIFEST=str(payload_manifest),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    from episode_json_compat import install_from_environment as episode_hook
    from runtime_policy import install_from_environment as policy_hook
    from runtime_transfer import install_from_environment as transfer_hook
    episode_hook(); policy_hook(); transfer_hook()
    from audited_runtime import install as migration_hook
    migration_hook(root)
    from p3work import training_campaign as campaign
    from campaign_recovery import install
    install(campaign,root)
    from hardware_comparison import install as comparison_hook
    comparison_hook(root)
    return campaign


def _maximum(root, action, allocation_seconds):
    """Rolling wall allowance only; retain all original scientific work caps."""
    folder = Path(root)/Path(action['expected_receipt']).parent
    ledger = _read(folder/'wall_budget.json',{}) or {}
    values = [ledger.get(k,0) for k in ('durable_wall_seconds','checkpoint_wall_high_water_seconds','max_wall_seconds')]
    floor = max([0]+[v for v in values if type(v) in (int,float) and math.isfinite(v) and v>=0])
    return math.ceil(floor)+int(allocation_seconds)+600


def perform_action(root, action, payload_manifest, *, allocation_seconds=529200, direct_package=None, stop_requested=None, on_ready=None):
    """Execute exactly one native action; never call the global campaign loop.

    Caller owns operational per-action deduplication/claim and Slurm lifecycle.
    SIGINT is the original kernel's checkpoint unwind signal; do not use SIGKILL
    to migrate a live action. A changed physical GPU is audited by the existing
    UUID-only equivalent-runtime overlay, never hidden in observed properties.
    """
    root = Path(root).resolve()
    if action_complete(root, action):
        return {'status': 'REUSED_COMPLETE', 'receipt': {'path': action['expected_receipt']}, 'new_fit': False}
    campaign = activate_runtime(root, payload_manifest, direct_package=direct_package, allocation_seconds=allocation_seconds)
    'PAPER3_ACTION_STARTUP_CONTEXT_V1'
    import action_startup
    with action_startup.active(root, action):
        if on_ready is not None:
            on_ready()
        if action['kind'].startswith('extension_'):
            from p3work import source_extension_downstream as downstream
            from campaign_recovery import install_downstream
            install_downstream(downstream, root)
            arm = action['arm']
            if action['kind'] == 'extension_initialize':
                release = _read(root / 'hpc/data_release.json')
                index = _read(root / release['bootstrap_index']['path'])
                receipt = downstream.initialize_catalog(root, arm, OUTPUT + '/pre_internal_freeze.json', index['extension_manifests'][arm], execute=True)
                return {'status': 'COMPLETE', 'receipt': receipt, 'new_fit': False}
            result = downstream.auto_development(root, 'runs/source_extension_downstream/' + arm + '/launch.json', execute=True, user_started_gpu=True, stop_requested=stop_requested, stop_grace_seconds=120)
            return {'status': 'COMPLETE' if action_complete(root, action) else 'PAUSED', 'receipt': {'path': action['expected_receipt']} if action_complete(root, action) else None, 'native_result': result}
        native = {k: v for k, v in action.items() if k not in {'serial', 'affinity_slot', 'configuration_receipt_path'}}
        if action.get('configuration_receipt_path'):
            native['configuration_receipt'] = campaign.descriptor(root, root / action['configuration_receipt_path'])
        campaign.validate_registered_action(native)
        receipt = campaign.Production(root, _maximum(root, action, allocation_seconds)).perform(native)
        return {'status': 'COMPLETE' if receipt and action_complete(root, action) else 'PAUSED', 'receipt': receipt, 'new_fit': action['kind'] not in {'meta_selection', 'development_selection', 'strongest_control', 'freeze'}}


# Explicit aliases for the controller/worker integration.
list_ready_actions = plan_ready
run_action = perform_action
