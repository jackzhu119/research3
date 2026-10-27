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


def _hardware_error(message,profile):
    """A failed classification must expose the observed device, not a bare gate."""
    observation={key:profile.get(key) for key in ('device_name','device_uuid',
        'device_compute_capability','compute_capability','device_total_memory_bytes','device_total_memory')}
    raise ValueError(message+'; observed_gpu='+json.dumps(observation,ensure_ascii=False,sort_keys=True,default=str))


def hardware_type(profile):
    """Identify an actual full A100 without depending on one driver spelling.

    CUDA/NVML have used names such as ``NVIDIA A100-PCIE-40GB``,
    ``Tesla A100-SXM4-40GB`` and ``NVIDIA A100 80GB PCIe``.  The
    physical allocation is identified jointly from the A100 name, architecture,
    memory and physical UUID.  The original runtime object remains untouched.
    """
    if not isinstance(profile,dict):raise ValueError('Complete runtime object required')
    name=profile.get('device_name')
    if not isinstance(name,str) or not name.strip():_hardware_error('Observed GPU model required',profile)
    normalized=name.upper().strip()
    if not re.search(r'(?<![A-Z0-9])A100(?![A-Z0-9])',normalized):
        _hardware_error('Expected a full A100 40GB or 80GB allocation',profile)
    if (re.search(r'\bMIG\b',normalized) or str(profile.get('device_uuid','')).upper().startswith('MIG-')
            or profile.get('is_multi_instance_gpu') is True):
        _hardware_error('A MIG slice is not the requested full A100 allocation',profile)
    capability=profile.get('device_compute_capability',profile.get('compute_capability'))
    if not isinstance(capability,(list,tuple)) or tuple(capability)!=(8,0):
        _hardware_error('A100 compute capability 8.0 required',profile)
    memory=profile.get('device_total_memory_bytes',profile.get('device_total_memory'))
    if type(memory) is not int:_hardware_error('Observed GPU memory in bytes required',profile)
    nominal=next((size for size in (40,80) if (size-5)*1024**3 < memory < (size+5)*1024**3),None)
    if nominal is None:_hardware_error('Observed memory is not a full A100 40GB/80GB allocation',profile)
    named_sizes={int(size) for size in re.findall(r'(?<![0-9])(40|80)\s*G(?:B)?\b',normalized)}
    if named_sizes and named_sizes!={nominal}:
        _hardware_error('A100 model memory label differs from observed allocation',profile)
    compact=re.sub(r'[^A-Z0-9]','',normalized)
    form='SXM4' if 'SXM4' in compact else ('SXM' if 'SXM' in compact else ('PCIE' if 'PCIE' in compact else 'FULL'))
    kind='A100_%s_%dGB'%(form,nominal)
    if not re.fullmatch(r'(?:GPU-)?[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}',str(profile.get('device_uuid','')),re.IGNORECASE):
        _hardware_error('Actual physical GPU UUID required',profile)
    if not all(isinstance(profile.get(k),str) and profile[k] for k in ('python','torch')):
        _hardware_error('Complete Python and Torch runtime required',profile)
    if not any(isinstance(profile.get(k),str) and profile[k] for k in ('cuda','cuda_runtime')):
        _hardware_error('Complete native CUDA runtime required',profile)
    return kind


def validate_typed_transfer(before,actual):
    try:before_type=hardware_type(before)
    except ValueError as error:raise ValueError('Historical runtime: '+str(error)) from None
    try:actual_type=hardware_type(actual)
    except ValueError as error:raise ValueError('Actual runtime: '+str(error)) from None
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


def _latest_engineering_hardware(root):
    """Read only the five registered engineering receipt paths, never data trees."""
    observations=[]
    warnings=[]
    for method in ('B8','B10','B11','P','P+PCT'):
        job={'method':method,'dimension':4 if method in {'B10','B11'} else 8,
             'radius':.35,'seed':20261018,'phase':'engineering'}
        path=Path(root)/'runs/meta'/_sha(job)/'receipt.json'
        try:
            value=json.loads(path.read_text())
            if value.get('status')!='META_ENGINEERING_COMPLETE' or value.get('job')!=job:continue
            runtime=value.get('runtime_profile')
            kind=hardware_type(runtime)
            observations.append({'path':str(path.relative_to(root)),
                'hardware_type':kind,'device_name':runtime['device_name'],
                'mtime_ns':path.stat().st_mtime_ns})
        except FileNotFoundError:continue
        except (OSError,ValueError,TypeError,AttributeError) as error:
            warnings.append({'path':str(path.relative_to(root)),'error_type':type(error).__name__})
    latest=max(observations,key=lambda row:(row['mtime_ns'],row['path'])) if observations else None
    return latest,warnings


def inspect_primary_measurements(root,target=None):
    """Metadata only; infer the target from completed actual GPU engineering.

    The resource-group label is not device evidence. SXM4 40GB, PCIe 40GB and
    80GB measurements remain distinct. No observed target means pending, not a
    fictional PCIe allocation or a reason to stop training.
    """
    target_source='explicit' if target is not None else None
    target_warnings=[]
    if target is None:
        observed,target_warnings=_latest_engineering_hardware(root)
        if observed is not None:
            target=observed['hardware_type']
            target_source=observed
        else:
            return {'target_hardware':None,'target_hardware_source':None,'target_pending':True,
                'target_observation_warnings':target_warnings,'incompatible_measurements':[],
                'measurement_hardware_matches':None,'historical_80GB_timings_normalized_as_40GB':False}
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
    return {'target_hardware':target,'target_hardware_source':target_source,'target_pending':False,
            'target_observation_warnings':target_warnings,'incompatible_measurements':incompatible,
            'measurement_hardware_matches':not incompatible,
            'historical_80GB_timings_normalized_as_40GB':False}

