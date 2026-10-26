"""Allow the observed R1 CUDA12.1 driver transition without changing identities.

CUDA12.x Linux minor-version compatibility requires driver525.60.13 or newer.
This operational adapter permits only the two versions actually observed on the
school nodes, with the same observed framework. Every other software, source,
data, numerical-policy, and scientific-limit field still passes the original
exact comparator. Original profiles and checkpoint/identity files are untouched.
"""
from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
import time

VERSION='paper3-r1-observed-cu121-driver-transition-v1'
OBSERVED_DRIVERS=frozenset(('535.104.12','550.54.14'))
MINIMUM_CUDA12_LINUX_DRIVER=(525,60,13)


def _unique(value):
    if (not isinstance(value,list) or not value
            or not all(isinstance(item,str) and item for item in value)):
        return None
    return frozenset(value)


def _cuda(profile):
    return profile.get('cuda',profile.get('cuda_runtime'))


def _observed_python(before,actual):
    # Meta/development records platform.python_version(); source-extension
    # records the full sys.version string. Retain exact equality of either
    # representation, including every build/compiler suffix when present.
    value=before.get('python')
    return (isinstance(value,str) and value==actual.get('python')
            and value.split()[:1]==['3.12.3'])


def _observed_transition(before,actual):
    if not isinstance(before,dict) or not isinstance(actual,dict):return False
    historical,current=_unique(before.get('driver_versions')),_unique(actual.get('driver_versions'))
    if (historical is None or current is None or len(historical)!=1 or len(current)!=1
            or historical==current or historical|current!=OBSERVED_DRIVERS):return False
    if any(tuple(map(int,driver.split('.')))<MINIMUM_CUDA12_LINUX_DRIVER for driver in historical|current):
        return False
    return (_observed_python(before,actual)
        and before.get('torch')==actual.get('torch')=='2.5.1+cu121'
        and _cuda(before)==_cuda(actual)=='12.1'
        and before.get('cudnn')==actual.get('cudnn')==90100)


def _emit(event,callback=None,**fields):
    try:
        if callback is None:
            print(json.dumps(dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                  **fields),sort_keys=True),flush=True)
        else:callback(event,**fields)
    except Exception:pass


def install(root,*,audited_module=None,emit=None):
    """Amend only the external typed comparator; existing validators still run."""
    root=str(Path(root).resolve())
    if audited_module is None:
        import audited_runtime as audited_module
    current=audited_module.validate_typed_transfer
    if getattr(current,'_paper3_r1_driver_compat_version',None)==VERSION:
        if getattr(current,'_paper3_r1_driver_compat_root',None)!=root:
            raise ValueError('Observed R1 driver compatibility is bound to a different project root')
        return {'status':'ALREADY_INSTALLED','version':VERSION}

    @functools.wraps(current)
    def validate(before,actual):
        if not _observed_transition(before,actual):return current(before,actual)
        # Normalize one field in a temporary comparison copy. The existing
        # journal receives the actual untouched dictionaries from its caller.
        comparison=copy.deepcopy(actual)
        comparison['driver_versions']=copy.deepcopy(before['driver_versions'])
        policy=dict(current(before,comparison))
        transition={'schema_version':VERSION,'historical_driver_versions':copy.deepcopy(before['driver_versions']),
            'actual_driver_versions':copy.deepcopy(actual['driver_versions']),
            'cuda_runtime':'12.1','torch':'2.5.1+cu121','cudnn':90100,
            'minimum_cuda12_linux_driver':'525.60.13','observed_driver_pair_only':True,
            'all_other_original_comparisons_preserved':True,
            'bitwise_equivalence_or_equal_speed_claimed':False}
        policy.update(driver_transition=transition,
            allowed_differences=sorted(set(policy.get('allowed_differences',[]))|{'driver_versions'}),
            identical_driver_versions_with_different_node_GPU_inventory_count=False)
        _emit('R1_OBSERVED_CUDA_DRIVER_TRANSFER_ALLOWED',emit,
              historical_driver_versions=sorted(set(before['driver_versions'])),
              actual_driver_versions=sorted(set(actual['driver_versions'])),
              runtime_identity_changed=False,checkpoint_changed=False,
              bitwise_equivalence_claimed=False)
        return policy

    validate._paper3_r1_driver_compat_version=VERSION
    validate._paper3_r1_driver_compat_root=root
    audited_module.validate_typed_transfer=validate
    _emit('R1_OBSERVED_DRIVER_COMPATIBILITY_INSTALLED',emit,version=VERSION,
          observed_driver_versions=sorted(OBSERVED_DRIVERS),native_sources_changed=False)
    return {'status':'INSTALLED','version':VERSION}
