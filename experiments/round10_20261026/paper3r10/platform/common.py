"""Versioned R10 plan and result contracts; standard library only."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

PLAN_SCHEMA = 'paper3-r10-plan-v1'
STAGES = frozenset(('mechanism', 'formal_exploratory', 'formal_confirmatory'))
SAFE_KEY = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\Z')


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path, default=None, max_bytes=16*1024*1024):
    try:
        with Path(path).open('rb') as handle:
            data = handle.read(max_bytes+1)
        if len(data)>max_bytes:
            raise ValueError('R10 metadata exceeds bounded JSON limit')
        return json.loads(data)
    except (OSError, ValueError):
        return default


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic(path, value):
    atomic_bytes(path, encoded(value))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=utc(), **fields), ensure_ascii=False,
                     allow_nan=False), flush=True)


@contextmanager
def locked(path, *, nonblocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield handle


def task_identity(plan, task):
    return digest({'schema': 'paper3-r10-task-v1', 'recipe_sha256': plan['recipe_sha256'],
                   'stage': plan['stage'], 'task': task})


def plan_recipe(plan):
    """Bind the complete scientific stage plan; only explicit runtime notes are excluded."""
    return digest({k:v for k,v in plan.items() if k not in ('recipe_sha256','operational_metadata')})


def sign_plan(plan):
    """Return a signed copy after all tasks, identities, paths and code are fixed."""
    value=dict(plan)
    manifest=value.get('code_manifest')
    if not isinstance(manifest,dict) or not manifest:
        raise ValueError('R10 plan requires its explicit code manifest before signing')
    value['code_manifest_sha256']=digest(manifest)
    if value.get('worker_entry'):
        value['worker_entry_sha256']=sha(value['worker_entry'])
    value['recipe_sha256']=plan_recipe(value)
    return value


def verify_code(plan):
    manifest=plan.get('code_manifest')
    if (not isinstance(manifest,dict) or not manifest
            or plan.get('code_manifest_sha256')!=digest(manifest)):
        raise ValueError('R10 code manifest identity changed')
    for relative,expected in manifest.items():
        if not re.fullmatch(r'[0-9a-f]{64}',str(expected)):
            raise ValueError('R10 code manifest needs full SHA256 values')
        path=local(plan['package'],relative)
        if not path.is_file() or sha(path)!=expected:
            raise ValueError('Declared R10 code changed; create and freeze a new recipe')
    if plan.get('worker_entry') and sha(plan['worker_entry'])!=plan.get('worker_entry_sha256'):
        raise ValueError('Explicit CPU fixture entry changed after plan signing')


def local(base, relative):
    value = Path(relative)
    if (value.is_absolute() or not value.parts or '..' in value.parts
            or any(p.lower() in ('private','.git','.ssh','.aws','secrets','credentials') or p.lower().startswith('.env') for p in value.parts)
            or value.name.lower() in ('github.json','credentials.json')):
        raise ValueError('R10 artifact path must be relative to its declared base')
    base = Path(base).resolve(); path = base/value
    if not path.resolve().is_relative_to(base) or any(p.is_symlink() for p in (path, *path.parents) if p!=base.parent):
        raise ValueError('R10 artifact path escaped its base or contains a symlink')
    return path


def verify_descriptor(base, item):
    if (not isinstance(item, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', '')))
            or type(item.get('bytes')) is not int or item['bytes']<0):
        raise ValueError('R10 artifact needs a SHA256 and byte count')
    path = local(base, item['path'])
    if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
        raise ValueError('R10 artifact byte/SHA identity changed')
    return path


def validate_stage(plan):
    if plan.get('stage') not in STAGES:
        raise ValueError('Unknown R10 stage')
    if plan['stage'].startswith('formal_') and plan.get('synthetic_fixture'):
        raise ValueError('Synthetic software fixtures cannot be used in formal R10 experiments')
    for task in plan['tasks']:
        if 'method_config' in task and task['method_config']!=task.get('config'):
            raise ValueError('method_config differs from the canonical task config')
    if plan['stage']=='mechanism':
        for task in plan['tasks']:
            role = task.get('role', task.get('partition'))
            if role not in ('meta_train', 'meta_validation'):
                raise ValueError('Mechanism tasks may use only meta_train/meta_validation')
        return
    freeze = plan.get('freeze')
    if not isinstance(freeze, dict):
        raise ValueError('Formal R10 evaluation requires an explicit freeze descriptor')
    base = Path(plan['root'])
    path = verify_descriptor(base, freeze)
    receipt = read(path, {})
    if (receipt.get('code_manifest_sha256')!=plan.get('code_manifest_sha256')
            or not receipt.get('code_manifest_sha256')):
        raise ValueError('Formal code identity differs from the development freeze')
    registry_sha=plan.get('study_registry',{}).get('sha256')
    if not registry_sha or receipt.get('study_registry_sha256')!=registry_sha:
        raise ValueError('Formal study registry differs from the development freeze')
    if receipt.get('scientific_scope')!='REAL_DATA':
        raise ValueError('Only a REAL_DATA freeze can authorize formal R10 science')
    if (receipt.get('status')!='FROZEN' or receipt.get('target_query_labels_used_for_selection') is not False
            or receipt.get('development_query_labels_used') is not True
            or receipt.get('selection_roles')!=['meta_train', 'meta_validation']
            or not isinstance(receipt.get('selected_configuration'), dict)
            or not receipt.get('comparison_family')):
        raise ValueError('R10 freeze lacks development-only selection and fixed comparisons')
    if plan.get('comparison_family')!=receipt['comparison_family']:
        raise ValueError('Formal comparison family differs from the frozen family')
    if (not isinstance(receipt.get('secondary_ablation_family'),list)
            or plan.get('secondary_ablation_family')!=receipt['secondary_ablation_family']):
        raise ValueError('Formal secondary ablation family differs from the frozen family')
    if (not receipt.get('reference_strategy')
            or receipt['reference_strategy']!=plan.get('reference_strategy')):
        raise ValueError('Formal feature reference strategy differs from the development freeze')
    primary=0
    for task in plan['tasks']:
        if 'method_config' in task and task['method_config']!=task.get('config'):
            raise ValueError('Formal method_config differs from the frozen task config')
        if task.get('is_primary_algorithm'):
            primary+=1
            if task.get('config')!=receipt['selected_configuration']:
                raise ValueError('Primary R10 configuration differs from the development freeze')
        if task.get('ablation'):
            allowed=receipt.get('allowed_ablations',{})
            if (task.get('method')!=task['ablation'] or not isinstance(allowed,dict)
                    or allowed.get(task['ablation'])!=task.get('config')):
                raise ValueError('Formal ablation was not exactly specified in the freeze')
    if primary==0:
        raise ValueError('Formal plan must identify the frozen primary algorithm tasks')
    if plan['stage']=='formal_confirmatory':
        item = plan.get('confirmation_manifest')
        if not isinstance(item, dict):
            raise ValueError('Independent confirmation data have not been provided')
        data = read(verify_descriptor(base, item), {})
        if (data.get('cohort_kind')!='independent_confirmation'
                or data.get('previously_used_for_algorithm_design') is not False
                or not data.get('dataset_identity') or not data.get('participants')):
            raise ValueError('Confirmation manifest must declare genuinely unused data')
        previous = set(plan.get('previously_used_participants', []))
        if previous.intersection(data['participants']):
            raise ValueError('Confirmation participants overlap previously used participants')
        registry=read(verify_descriptor(base,plan.get('study_registry')), {})
        from paper3r10.confirmation import validate_confirmation_metadata
        validate_confirmation_metadata(plan,registry)


def load_plan(path, *, validate_science_stage=True):
    path = Path(path).expanduser().resolve()
    plan = read(path)
    if not isinstance(plan, dict) or plan.get('schema_version')!=PLAN_SCHEMA or plan.get('round')!=10:
        raise ValueError('A versioned Round10 plan is required')
    if not re.fullmatch(r'[0-9a-f]{64}', str(plan.get('recipe_sha256', ''))):
        raise ValueError('R10 recipe SHA256 is required')
    if plan['recipe_sha256']!=plan_recipe(plan):
        raise ValueError('R10 scientific plan content changed without a new recipe')
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('R10 stage plan must contain a nonempty finite task list')
    if any(not isinstance(task,dict) for task in tasks):
        raise ValueError('Every R10 task must be a versioned metadata object')
    keys = [t.get('key') for t in tasks]
    if any(not isinstance(k, str) or not SAFE_KEY.fullmatch(k) for k in keys) or len(keys)!=len(set(keys)):
        raise ValueError('R10 task keys must be unique safe directory names')
    known = set(keys); done = set(); remaining = {t['key']:set(t.get('requires', [])) for t in tasks}
    if any(not deps<=known for deps in remaining.values()):
        raise ValueError('R10 dependency is absent from the stage plan')
    while remaining:
        ready = {key for key,deps in remaining.items() if deps<=done}
        if not ready:
            raise ValueError('R10 task dependency graph contains a cycle')
        done.update(ready)
        for key in ready:
            remaining.pop(key)
    for key in ('root', 'output', 'package'):
        if not isinstance(plan.get(key), str) or not Path(plan[key]).is_absolute():
            raise ValueError('R10 plan root/output/package must be explicit absolute paths')
    output = Path(plan['output']).resolve()
    if output==Path(plan['root']).resolve() or not any(s in output.name.lower() for s in ('r10','round10')):
        raise ValueError('R10 output must use its own new round10/r10 directory')
    verify_code(plan)
    if validate_science_stage:
        validate_stage(plan)
    return path, plan


def task_folder(plan, task):
    return Path(plan['output'])/task['key']


def verify_result(plan, task, *, sealed=True, verify_artifacts=True):
    folder = task_folder(plan, task)
    result_path = folder/'result.json'
    result = read(result_path, {})
    expected_scope='SOFTWARE_TEST_ONLY' if plan.get('synthetic_fixture') else 'REAL_DATA'
    if (not isinstance(result,dict) or result.get('status')!='COMPLETE' or result.get('recipe_sha256')!=plan['recipe_sha256']
            or result.get('task_key')!=task['key'] or result.get('task_identity')!=task_identity(plan, task)):
        return False
    if result.get('scientific_scope')!=expected_scope:
        return False
    artifacts = result.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        return False
    try:
        if any(not isinstance(item,dict) or item.get('path') in ('result.json','completion.json') for item in artifacts):
            return False
        if verify_artifacts:
            for item in artifacts:
                verify_descriptor(folder, item)
        if sealed:
            proof = read(folder/'completion.json', {})
            if (not isinstance(proof,dict) or proof.get('schema_version')!='paper3-r10-completion-v1'
                    or proof.get('task_identity')!=task_identity(plan, task)
                    or proof.get('result_sha256')!=sha(result_path)
                    or proof.get('artifacts')!=artifacts):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def seal_result(plan, task):
    if not verify_result(plan, task, sealed=False):
        raise ValueError('Worker did not publish a complete exact-identity R10 result')
    folder = task_folder(plan, task); result = read(folder/'result.json')
    proof = {'schema_version': 'paper3-r10-completion-v1', 'task_identity':task_identity(plan,task),
             'recipe_sha256':plan['recipe_sha256'], 'task_key':task['key'],
             'result_sha256':sha(folder/'result.json'), 'artifacts':result['artifacts'],
             'completed_utc':utc(), 'scientific_result_verified':True}
    atomic(folder/'completion.json', proof)
    return proof
