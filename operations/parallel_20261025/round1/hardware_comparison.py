"""Audited equivalent-GPU comparisons after native artifact verification.

Only physical device UUID may differ. Original artifacts, observed runtime
profiles and numerical selection rules remain unchanged. The returned copies
carry an explicit hardware-class comparison identity, never a fabricated UUID.
"""
from __future__ import annotations

import ast
import copy
from contextvars import ContextVar
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

SCHEMA='paper3-equivalent-hardware-comparison-v1'
AUDIT='runs/training_campaign_parallel_v1/runtime_transfer_comparison.json'
PINNED={'p3work/development_evaluation.py':'dc5de7376b5b774dd7d846cdce2cf6d1ba4f5f51df228130c275e13ec6724618',
        'p3work/internal_validation.py':'a847fc9d3d6ac0aa42d47992ce4e22335794854332ca5ad449e6f1e9c537af3b'}
_CONTROL_OBSERVATIONS=ContextVar('paper3_control_native_verified_observations',default=None)
_OBSERVED_CACHE={}


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def _digest(value):return hashlib.sha256(_canonical(value)).hexdigest()


def _raw_profile(value):
    if not isinstance(value,dict):raise ValueError('Complete native runtime profile required')
    if not all(isinstance(value.get(k),str) and value[k] for k in ('python','torch','device_name')):
        raise ValueError('Complete native Python/Torch/GPU profile required')
    if not any(isinstance(value.get(k),str) and value[k] for k in ('cuda','cuda_runtime')):
        raise ValueError('Native CUDA runtime required')
    if not any(type(value.get(k)) is int and value[k]>0 for k in ('device_total_memory','device_total_memory_bytes')):
        raise ValueError('Native GPU memory profile required')
    if not any(isinstance(value.get(k),list) and len(value[k])==2 for k in ('compute_capability','device_compute_capability')):
        raise ValueError('Native GPU architecture profile required')
    if not re.fullmatch(r'(?:GPU-)?[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}',str(value.get('device_uuid',''))):
        raise ValueError('Observed physical GPU UUID required')
    result=copy.deepcopy(value);result.pop('device_uuid')
    # Native nvidia-smi enumerates the whole node, not only the allocated GPU.
    # Repeated copies of the exact same version describe inventory cardinality.
    drivers=result.get('driver_versions')
    if isinstance(drivers,list) and drivers and all(isinstance(v,str) and v for v in drivers):
        result['driver_versions']=sorted(set(drivers))
    return result


def _profile(value):
    result = _raw_profile(value)
    # Only official naming aliases are administrative. Memory, form factor,
    # software and numerical settings remain separate comparison evidence.
    from audited_runtime import hardware_type
    result['device_name'] = hardware_type(value)
    return result


def _existing_comparison_identity(value, normalized):
    previous = value.get('hardware_class_profile')
    if previous != normalized:
        # Old journals store the uncanonicalized model string. Bind that class
        # to a full original observation instead of fabricating a device UUID.
        matched = any(
            _raw_profile(row.get('runtime_profile')) == previous
            and _profile(row.get('runtime_profile')) == normalized
            for row in value.get('observations', {}).values()
        )
        if not matched:
            raise ValueError('Observed runtime does not match this comparison hardware class')
    original = _digest({'schema_version': SCHEMA, 'runtime_profile_without_device_uuid': previous})
    if value.get('hardware_class_identity_sha256') != original:
        raise ValueError('Existing comparison identity does not match its recorded profile')
    return original


def equivalent_runtime(root,before,actual,context):
    if _profile(before)!=_profile(actual):
        raise ValueError('GPU model, software, driver or numerical policy changed; not UUID-only migration')
    observe(root,before,{'context':context,'role':'historical_runtime'})
    observe(root,actual,{'context':context,'role':'actual_runtime'})
    return True


def observe(root,runtime,reference):
    root=Path(root).resolve()
    normalized=_profile(runtime)
    identity=_digest({'schema_version':SCHEMA,'runtime_profile_without_device_uuid':normalized})
    path=root/AUDIT;path.parent.mkdir(parents=True,exist_ok=True)
    observation={'runtime_profile':copy.deepcopy(runtime),'native_runtime_identity_sha256':_digest(runtime),
                 'reference':copy.deepcopy(reference)}
    key=_digest(observation)
    cached=(str(root),identity,key)
    if cached in _OBSERVED_CACHE:
        return _OBSERVED_CACHE[cached]
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            value=json.loads(path.read_text())
            if value.get('schema_version') != SCHEMA:
                raise ValueError('Observed runtime comparison journal schema changed')
            # Retain the previously published comparison ID and class bytes.
            # This keeps already frozen selection evidence valid across aliases.
            identity = _existing_comparison_identity(value, normalized)
        else:
            value={'schema_version':SCHEMA,'authorization':'2026-10-25 continue original first-round work in three GPU jobs with checkpoint reuse',
                'scope':'Equivalent-hardware comparison; physical UUID and repeated node-driver inventory count may differ, exact driver versions and actual measurements remain recorded',
                'hardware_class_profile':normalized,'hardware_class_identity_sha256':identity,
                'original_receipts_and_checkpoint_bytes_changed':False,'actual_device_UUID_overridden':False,
                'BA_BCE_seed_split_update_or_selection_tie_rules_changed':False,
                'cross_device_bitwise_resume_empirically_verified':False,'observations':{}}
        if key not in value['observations']:
            value['observations'][key]=observation
            fd,temp=tempfile.mkstemp(prefix='.comparison-',dir=path.parent)
            try:
                with os.fdopen(fd,'wb') as handle:
                    handle.write(_canonical(value)+b'\n');handle.flush();os.fsync(handle.fileno())
                os.replace(temp,path)
            finally:
                Path(temp).unlink(missing_ok=True)
    _OBSERVED_CACHE[cached] = identity
    return identity


def normalize_verified(root,value,descriptor):
    """Called only on the successful return value of the native verifier."""
    runtime=value.get('inputs',{}).get('runtime_profile')
    if value.get('inputs',{}).get('device_identity_sha256')!=_digest(runtime):
        raise ValueError('Native receipt runtime identity does not match observed profile')
    identity=observe(root,runtime,{'kind':'native_verified_development_receipt','descriptor':descriptor})
    result=copy.deepcopy(value)
    inputs=result['inputs']
    inputs['observed_device_identity_sha256']=inputs['device_identity_sha256']
    inputs['device_identity_sha256']=identity
    inputs['device_comparison_identity_sha256']=identity
    inputs['device_comparison_scope']='equivalent-hardware-class; actual runtime_profile retained unchanged'
    return result


def _preserved_control(root,kwargs,result,observations):
    """Preserve an existing legacy B* artifact after a full equivalent rebuild.

    A previously frozen file can differ only in its copied comparison-identity
    hashes. Every method, seed, score, timing, source, choice and reference must
    still equal the native reconstruction; otherwise the existing freeze fails.
    """
    output=kwargs.get('output') or 'runs/training_campaign_v1/strongest_control.json'
    path=Path(root)/output
    if not path.is_file():return result
    prior=json.loads(path.read_text())
    normalized=copy.deepcopy(prior)
    old_rows=normalized.get('evidence_rows',[])
    new_rows=result.get('evidence_rows',[])
    if len(old_rows)!=len(new_rows):raise ValueError('Existing Bstar evidence coverage differs')
    for before,after in zip(old_rows,new_rows):
        observed=observations.get((after.get('method'),after.get('seed')))
        if before.get('device_identity_sha256') not in {after.get('device_identity_sha256'),observed}:
            raise ValueError('Existing Bstar device hash was not observed in its native verified receipt')
        before['device_identity_sha256']=after['device_identity_sha256']
    if normalized!=result:
        raise ValueError('Existing Bstar differs beyond equivalent-device administrative comparison')
    return prior


def _internal_identity(root,folder,candidate):
    path=Path(root)/'runs/internal_validation_once_claim.json'
    if not path.is_file():return candidate
    original=json.loads(path.read_text())['identity']
    before,after=copy.deepcopy(original),copy.deepcopy(candidate)
    before_runtime=before.pop('runtime',None);actual=after.pop('runtime',None)
    if before!=after:raise ValueError('Internal one-time claim differs beyond physical GPU UUID')
    equivalent_runtime(root,before_runtime,actual,'existing_internal_once_claim_lineage')
    return original


def _rewrite_internal(module,root):
    source=Path(root)/'p3work/internal_validation.py';raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINNED['p3work/internal_validation.py']:
        raise ValueError('Original internal validation implementation changed')
    tree=ast.parse(raw)
    expressions={
        'execute_internal': [('runtime != row["runtime_profile"]',
            'not _paper3_parallel_equivalent_runtime(root, row["runtime_profile"], runtime, "internal_development_profile")')],
        '_load_fit': [('warmup.get("runtime_profile") != runtime',
            'not _paper3_parallel_equivalent_runtime(root, warmup.get("runtime_profile"), runtime, "internal_committed_warmup")'),
            ('timing.get("runtime_profile") != runtime',
            'not _paper3_parallel_equivalent_runtime(root, timing.get("runtime_profile"), runtime, "internal_committed_timing")')],
    }
    definitions=[]
    for name,edits in expressions.items():
        definition=copy.deepcopy(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name))
        for before,after in edits:
            old=ast.dump(ast.parse(before,mode='eval').body,include_attributes=False)
            replacement=ast.parse(after,mode='eval').body
            matches=[0]
            class Rewrite(ast.NodeTransformer):
                def generic_visit(self,node):
                    if ast.dump(node,include_attributes=False)==old:
                        matches[0]+=1;return ast.copy_location(copy.deepcopy(replacement),node)
                    return super().generic_visit(node)
            definition=Rewrite().visit(definition)
            if matches[0]!=1:raise ValueError('Native internal administrative comparison changed')
        if name=='execute_internal':
            marker=ast.dump(ast.parse('destination.mkdir(parents=True, exist_ok=True)').body[0],include_attributes=False)
            inserts=0
            for index,node in enumerate(definition.body):
                if ast.dump(node,include_attributes=False)==marker:
                    definition.body.insert(index+1,ast.parse('identity = _paper3_parallel_internal_identity(root,destination,identity)').body[0])
                    inserts+=1;break
            if inserts!=1:raise ValueError('Native internal identity publication changed')
        definitions.append(definition)
    module.__dict__.update(_paper3_parallel_equivalent_runtime=equivalent_runtime,
                           _paper3_parallel_internal_identity=_internal_identity)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[])),str(source),'exec'),module.__dict__)


def install(root,*,development_module=None,internal_module=None):
    root=Path(root).resolve()
    if development_module is None:
        from p3work import development_evaluation as development_module
    dev=development_module
    if getattr(dev,'_paper3_parallel_hardware_comparison',False):
        if dev._paper3_parallel_hardware_root!=root:raise ValueError('Conflicting comparison project root')
        return
    for relative in PINNED:
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=PINNED[relative]:
            raise ValueError('Original source changed before hardware-comparison amendment')
    original_verify=dev.verify_development_receipt
    @functools.wraps(original_verify)
    def verify(current_root,descriptor):
        # The full native parameter/prediction/metric verifier runs first.
        value=original_verify(current_root,descriptor)
        result=normalize_verified(current_root,value,descriptor)
        observations=_CONTROL_OBSERVATIONS.get()
        if observations is not None:
            observations[(result.get('method'),result.get('seed'))]=result['inputs']['observed_device_identity_sha256']
        return result
    dev.verify_development_receipt=verify
    original_control=dev.select_strongest_control
    @functools.wraps(original_control)
    def control(current_root,**kwargs):
        options=dict(kwargs);execute=options.pop('execute',False)
        observations={};token=_CONTROL_OBSERVATIONS.set(observations)
        try:
            result=original_control(current_root,execute=False,**options)
        finally:
            _CONTROL_OBSERVATIONS.reset(token)
        preserved=_preserved_control(current_root,options,result,observations)
        if execute:
            output=options.get('output')
            if not output or not output.startswith('runs/'):
                raise ValueError('Explicit immutable control output required')
            path=dev._local(Path(current_root).resolve(),output,exists=False)
            path.parent.mkdir(parents=True,exist_ok=True)
            dev._put(path,preserved)
        return preserved
    dev.select_strongest_control=control
    if internal_module is None:
        from p3work import internal_validation as internal_module
    _rewrite_internal(internal_module,root)
    dev._paper3_parallel_hardware_comparison=True
    dev._paper3_parallel_hardware_root=root

