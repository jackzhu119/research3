"""Typed A100 80GB-to-40GB checkpoint migration, separate from timing equality.

Scientific identities/checkpoint bytes remain historical. Actual observations
are journaled; hardware-transfer permission is never a bitwise-equivalence or
equal-speed claim. Software, code, data and scientific limits stay exact.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

SCHEMA='paper3-authorized-a100-hardware-migration-v1'
AUDIT='runs/training_campaign_parallel_v1/hardware_migrations.json'
HARDWARE_KEYS=frozenset({'device_name','device_uuid','device_total_memory','device_total_memory_bytes'})


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def _sha(value):return hashlib.sha256(_encoded(value)).hexdigest()


def hardware_type(profile):
    if not isinstance(profile,dict):raise ValueError('Complete runtime object required')
    name=profile.get('device_name')
    if not isinstance(name,str):raise ValueError('Observed GPU model required')
    normalized=name.upper()
    kind={'NVIDIA A100-SXM4-80GB':'A100_SXM4_80GB','A100-SXM4-80GB':'A100_SXM4_80GB',
          'NVIDIA A100-PCIE-40GB':'A100_PCIE_40GB','A100-PCIE-40GB':'A100_PCIE_40GB'}.get(normalized)
    if kind is None:raise ValueError('Only the explicitly authorized full A100 80GB/40GB hardware is transferable')
    capability=profile.get('device_compute_capability',profile.get('compute_capability'))
    if capability!=[8,0]:raise ValueError('A100 compute capability 8.0 required')
    memory=profile.get('device_total_memory_bytes',profile.get('device_total_memory'))
    if type(memory) is not int:raise ValueError('Observed GPU memory required')
    nominal=80 if kind=='A100_SXM4_80GB' else 40
    if not (nominal-5)*1024**3 < memory < (nominal+5)*1024**3:
        raise ValueError('Observed A100 memory is inconsistent with the model')
    if not re.fullmatch(r'(?:GPU-)?[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}',str(profile.get('device_uuid',''))):
        raise ValueError('Actual physical GPU UUID required')
    if not all(isinstance(profile.get(k),str) and profile[k] for k in ('python','torch')):
        raise ValueError('Complete Python and Torch runtime required')
    if not any(isinstance(profile.get(k),str) and profile[k] for k in ('cuda','cuda_runtime')):
        raise ValueError('Complete native CUDA runtime required')
    return kind


def validate_typed_transfer(before,actual):
    before_type,actual_type=hardware_type(before),hardware_type(actual)
    if set(before)!=set(actual):raise ValueError('Runtime fields differ beyond authorized hardware')
    def software_value(key,value):
        if key=='driver_versions' and isinstance(value,list) and value and all(isinstance(v,str) and v for v in value):
            return sorted(set(value))
        return value
    if any(software_value(key,before[key])!=software_value(key,actual[key]) for key in before if key not in HARDWARE_KEYS):
        raise ValueError('Software, code, data, numerical policy or scientific limits changed')
    return {'historical_hardware':before_type,'actual_hardware':actual_type,
            'allowed_differences':sorted(key for key in HARDWARE_KEYS if before.get(key)!=actual.get(key)),
            'identical_driver_versions_with_different_node_GPU_inventory_count':before.get('driver_versions')!=actual.get('driver_versions')}


def _journal(root,before,actual,context,artifact_refs,policy,payload):
    path=Path(root)/AUDIT;path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=json.loads(path.read_text()) if path.exists() else {
            'schema_version':SCHEMA,'authorization':'2026-10-25 move original first round from hpc_gpu1 A100 80GB to hpc_gpu A100 PCIe 40GB in three jobs',
            'original_identity_and_checkpoint_bytes_changed':False,'actual_device_properties_overridden':False,
            'scientific_caps_seed_split_and_algorithm_changed':False,
            'cross_hardware_bitwise_resume_empirically_verified':False,
            'old_engineering_timing_is_an_operational_estimate_not_a_new_40GB_benchmark':True,
            'primary_80GB_and_40GB_calibration_timings_may_be_pooled':False,'transfers':{}}
        if value.get('schema_version')!=SCHEMA:raise ValueError('Hardware migration journal schema changed')
        record={'context':context,'historical_runtime':copy.deepcopy(before),'actual_runtime':copy.deepcopy(actual),
                'policy':policy,'artifact_refs':copy.deepcopy(artifact_refs or []),'source_payload_manifest':payload,
                'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
        key=_sha(record)
        if key not in value['transfers']:
            value['transfers'][key]=record
            fd,temp=tempfile.mkstemp(prefix='.hardware-transfer-',dir=path.parent)
            try:
                with os.fdopen(fd,'wb') as handle:handle.write(_encoded(value)+b'\n');handle.flush();os.fsync(handle.fileno())
                os.replace(temp,path)
            finally:Path(temp).unlink(missing_ok=True)


def install(root,*,transfer_module=None):
    root=Path(root).resolve()
    if transfer_module is None:
        import runtime_transfer as transfer_module
    module=transfer_module
    if getattr(module,'_paper3_authorized_a100_transfer',False):
        if module._paper3_authorized_a100_root!=root:raise ValueError('Hardware migration root conflict')
        return
    original=module.validate_transfer
    def validate(current_root,before,actual,context,artifact_refs=None,*,record=True):
        if Path(current_root).resolve()!=root:raise ValueError('Hardware migration requires the same project')
        policy=validate_typed_transfer(before,actual)
        # Equivalent cards continue through the existing audited UUID-only path.
        if (all(before.get(k)==actual.get(k) for k in HARDWARE_KEYS if k!='device_uuid')
                and before.get('driver_versions')==actual.get('driver_versions')):
            return original(current_root,before,actual,context,artifact_refs,record=record)
        if record:
            payload=module._manifest()  # Retain the already deployed adapter's source binding.
            _journal(root,before,actual,context,artifact_refs,policy,payload)
        return True
    module.validate_transfer=validate
    module._paper3_authorized_a100_transfer=True
    module._paper3_authorized_a100_root=root


def inspect_primary_measurements(root,target='A100_PCIE_40GB'):
    """Metadata only: never silently merge old 80GB timing with new 40GB timing."""
    base=Path(root)/'runs/training_campaign_v1'
    incompatible=[]
    for relative in ('profiles','development'):
        directory=base/relative
        if not directory.is_dir():continue
        for path in sorted(directory.glob('*/receipt.json')):
            try:value=json.loads(path.read_text())
            except (OSError,ValueError):continue
            if value.get('status') not in {'DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE','DEVELOPMENT_EVALUATION_COMPLETE'}:
                continue
            runtime=value.get('runtime_profile') or value.get('inputs',{}).get('runtime_profile')
            if runtime is None:
                identity=value.get('identity')
                if isinstance(identity,dict) and isinstance(identity.get('path'),str):
                    try:runtime=json.loads((Path(root)/identity['path']).read_text()).get('runtime_profile')
                    except (OSError,ValueError):pass
            if runtime is None:continue
            if hardware_type(runtime)!=target:
                incompatible.append({'path':str(path.relative_to(root)),'status':value['status'],
                    'observed_hardware':hardware_type(runtime),'required_measurement_hardware':target,
                    'source_file_retained':True,'source_or_meta_retraining_required':False,
                    'target_calibration_remeasurement_required':True})
    return {'target_hardware':target,'incompatible_measurements':incompatible,
            'measurement_hardware_matches':not incompatible,
            'historical_80GB_timings_normalized_as_40GB':False}

