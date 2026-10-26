"""Versioned R11 plan and result contracts; standard library only."""
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

PLAN_SCHEMA = 'paper3-r11-plan-v1'
STAGES = frozenset(('campaign','development', 'exploratory', 'confirmatory'))
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
            raise ValueError('R11 metadata exceeds bounded JSON limit')
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
    return digest({'schema': 'paper3-r11-task-v1', 'recipe_sha256': plan['recipe_sha256'],
                   'stage': plan['stage'], 'task': task})


def plan_recipe(plan):
    """Bind the complete scientific stage plan; only explicit runtime notes are excluded."""
    return digest({k:v for k,v in plan.items() if k not in ('recipe_sha256','operational_metadata')})


def sign_plan(plan):
    """Return a signed copy after all tasks, identities, paths and code are fixed."""
    value=dict(plan)
    manifest=value.get('code_manifest')
    if not isinstance(manifest,dict) or not manifest:
        raise ValueError('R11 plan requires its explicit code manifest before signing')
    value['code_manifest_sha256']=digest(manifest)
    if value.get('worker_entry'):
        value['worker_entry_sha256']=sha(value['worker_entry'])
    value['recipe_sha256']=plan_recipe(value)
    return value


def verify_code(plan, *, check_files=True):
    manifest=plan.get('code_manifest')
    if (not isinstance(manifest,dict) or not manifest
            or plan.get('code_manifest_sha256')!=digest(manifest)):
        raise ValueError('R11 code manifest identity changed')
    for relative,expected in manifest.items():
        if not re.fullmatch(r'[0-9a-f]{64}',str(expected)):
            raise ValueError('R11 code manifest needs full SHA256 values')
        if check_files:
            path=local(plan['package'],relative)
            if not path.is_file() or sha(path)!=expected:
                raise ValueError('Declared R11 code changed; create a new recipe')
    if check_files and plan.get('worker_entry') and sha(plan['worker_entry'])!=plan.get('worker_entry_sha256'):
        raise ValueError('Explicit CPU fixture entry changed after plan signing')


def local(base, relative):
    value = Path(relative)
    if (value.is_absolute() or not value.parts or '..' in value.parts
            or any(p.lower() in ('private','.git','.ssh','.aws','secrets','credentials') or p.lower().startswith('.env') for p in value.parts)
            or value.name.lower() in ('github.json','credentials.json')):
        raise ValueError('R11 artifact path must be relative to its declared base')
    base = Path(base).resolve(); path = base/value
    if not path.resolve().is_relative_to(base) or any(p.is_symlink() for p in (path, *path.parents) if p!=base.parent):
        raise ValueError('R11 artifact path escaped its base or contains a symlink')
    return path


def verify_descriptor(base, item):
    if (not isinstance(item, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', '')))
            or type(item.get('bytes')) is not int or item['bytes']<0):
        raise ValueError('R11 artifact needs a SHA256 and byte count')
    path = local(base, item['path'])
    if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
        raise ValueError('R11 artifact byte/SHA identity changed')
    return path


def validate_stage(plan):
    """Local role/no-leak policy; no foreign runtime profile or freeze transfer gate."""
    stage=plan.get('stage')
    if stage not in STAGES:raise ValueError('Unknown R11 stage')
    for task in plan['tasks']:
        if 'method_config' in task and task['method_config']!=task.get('config'):
            raise ValueError('method_config differs from the canonical task config')
        role=task.get('role',task.get('partition'))
        family=str(task.get('family',task.get('kind',''))).lower()
        if (family.endswith('train') or family in ('train','feature_reference')) and role!='meta_train':
            raise ValueError('Training tasks must use meta_train participants')
        if family in ('select','selection') and role!='meta_validation':
            raise ValueError('Selection tasks must use meta_validation participants')
        if stage=='development' and role not in ('meta_train','meta_validation'):
            raise ValueError('Development tasks may use only meta_train/meta_validation')
        if (task.get('target_query_labels_used_for_selection') is True
                or plan.get('target_query_labels_used_for_selection') is True):
            raise ValueError('Target query labels cannot select or fit an R11 update')
        if stage=='confirmatory':
            development_validation=role=='meta_validation' and task.get('phase')=='development'
            if role not in ('meta_train','independent_confirmation') and not development_validation:
                raise ValueError('Confirmation tasks require independent targets or development-only selection')
            if role=='independent_confirmation' and task.get('phase')!='confirmatory':
                raise ValueError('Independent target tasks must declare phase confirmatory')
    if stage=='confirmatory' and plan.get('synthetic_fixture'):
        raise ValueError('Synthetic software fixtures cannot claim independent confirmation')


def load_plan(path, *, validate_science_stage=True):
    path = Path(path).expanduser().resolve()
    plan = read(path,max_bytes=64*1024*1024)
    if not isinstance(plan, dict) or plan.get('schema_version')!=PLAN_SCHEMA or plan.get('round')!=11:
        raise ValueError('A versioned Round11 plan is required')
    if not re.fullmatch(r'[0-9a-f]{64}', str(plan.get('recipe_sha256', ''))):
        raise ValueError('R11 recipe SHA256 is required')
    if plan['recipe_sha256']!=plan_recipe(plan):
        raise ValueError('R11 scientific plan content changed without a new recipe')
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('R11 stage plan must contain a nonempty finite task list')
    if any(not isinstance(task,dict) for task in tasks):
        raise ValueError('Every R11 task must be a versioned metadata object')
    keys = [t.get('key') for t in tasks]
    if any(not isinstance(k, str) or not SAFE_KEY.fullmatch(k) for k in keys) or len(keys)!=len(set(keys)):
        raise ValueError('R11 task keys must be unique safe directory names')
    known = set(keys); done = set(); remaining = {t['key']:set(t.get('requires', []))|set(t.get('requires_terminal',[])) for t in tasks}
    if any(not deps<=known for deps in remaining.values()):
        raise ValueError('R11 dependency is absent from the stage plan')
    while remaining:
        ready = {key for key,deps in remaining.items() if deps<=done}
        if not ready:
            raise ValueError('R11 task dependency graph contains a cycle')
        done.update(ready)
        for key in ready:
            remaining.pop(key)
    for key in ('root', 'output', 'package'):
        if not isinstance(plan.get(key), str) or not Path(plan[key]).is_absolute():
            raise ValueError('R11 plan root/output/package must be explicit absolute paths')
    output = Path(plan['output']).resolve()
    if output==Path(plan['root']).resolve() or not any(s in output.name.lower() for s in ('r11','round11')):
        raise ValueError('R11 output must use its own new round11/r11 directory')
    verify_code(plan,check_files=validate_science_stage)
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
            if (not isinstance(proof,dict) or proof.get('schema_version')!='paper3-r11-completion-v1'
                    or proof.get('task_identity')!=task_identity(plan, task)
                    or proof.get('result_sha256')!=sha(result_path)
                    or proof.get('artifacts')!=artifacts):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def seal_result(plan, task):
    if not verify_result(plan, task, sealed=False):
        raise ValueError('Worker did not publish a complete exact-identity R11 result')
    folder = task_folder(plan, task); result = read(folder/'result.json')
    proof = {'schema_version': 'paper3-r11-completion-v1', 'task_identity':task_identity(plan,task),
             'recipe_sha256':plan['recipe_sha256'], 'task_key':task['key'],
             'result_sha256':sha(folder/'result.json'), 'artifacts':result['artifacts'],
             'completed_utc':utc(), 'scientific_result_verified':True}
    atomic(folder/'completion.json', proof)
    return proof
