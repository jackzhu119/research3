"""Bounded explicit software checks; GPU use requires an existing allocation."""
from __future__ import annotations
import importlib.metadata
import os
from pathlib import Path
import platform
import time


def environment():
    versions={'python':platform.python_version()}
    for name in ('numpy','torch','pytest'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return versions


def cpu_check():
    import numpy as np
    from .algorithm import Config,fit,predict,anchor_probabilities,save_state,load_state
    import tempfile
    x=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
    y=np.array([0,0,1,1]);z=np.zeros((4,2));ids=tuple('s%d'%i for i in range(4))
    cfg=Config(amplitude=3.,evidence_fraction=.01)
    state=fit(x,z,y,cfg,support_ids=ids)
    q=np.array([[-1.,0.],[1.,0.],[0.,4.]])
    out,diagnostics=predict(state,q,np.zeros((3,2)),query_ids=('q0','q1','q2'))
    assert out.shape==(3,2) and np.isfinite(out).all()
    assert np.allclose(out.sum(1),1.) and np.all((out>0)&(out<1))
    anchor_state=fit(x,z,y,Config(amplitude=3.,evidence_fraction=1.),support_ids=ids)
    anchored,_=predict(anchor_state,q,np.zeros((3,2)),query_ids=('q0','q1','q2'))
    assert np.array_equal(anchored,anchor_probabilities(np.zeros((3,2))))
    with tempfile.TemporaryDirectory(prefix='r10-cpu-software-check-') as temp:
        p=Path(temp)/'state.npz';save_state(state,p)
        restored=load_state(p,expected_fit_identity=state.fit_identity)
        resumed,_=predict(restored,q,np.zeros((3,2)),query_ids=('q0','q1','q2'))
        assert np.array_equal(out,resumed)
    return {'status':'CPU_SOFTWARE_CHECK_PASSED','checks':[
        'finite_support_fit','proper_probabilities','c1_exact_anchor','pickle_free_exact_resume'],
        'environment':environment(),'gpu_initialized':False,'jobs_submitted':0,
        'real_EEG_performance_validated':False,'diagnostics':diagnostics}


def gpu_check(root,registry_path):
    """The caller supplies an already allocated GPU; this never calls Slurm."""
    if not os.environ.get('SLURM_JOB_ID'):
        return {'status':'NOT_RUN_NO_EXISTING_GPU_ALLOCATION','gpu_used':False,'jobs_submitted':0}
    import torch
    if str(torch.__version__)!='2.5.1+cu121' or str(torch.version.cuda)!='12.1':
        raise ValueError('Finite school GPU check requires the pinned Torch2.5.1+cu121 runtime')
    if not torch.cuda.is_available():
        return {'status':'NOT_RUN_CUDA_UNAVAILABLE','gpu_used':False,'jobs_submitted':0}
    from .study import describe
    from .data import load_registry,load_source,native
    plan={'root':str(Path(root).resolve()),'study_registry':describe(registry_path,root)}
    registry,_=load_registry(plan)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    properties=torch.cuda.get_device_properties(0)
    if 'A100' not in properties.name.upper() or properties.total_memory<35*1024**3:
        raise ValueError('Expected one full A10040/80GB from the declared hpc_gpu allocation')
    source,geometry,identity=load_source(plan,registry,20261018,'cuda:0')
    torch.manual_seed(20261026)
    x=torch.randn((4,11,128),dtype=torch.float64).to('cuda:0')
    before=[p.detach().clone() for p in source.parameters()]
    with torch.inference_mode():
        values=source.source_logits(x)
        assert values.shape==(4,2) and bool(torch.isfinite(values).all())
        a=geometry.rref_sqrt@geometry.rref_invsqrt
        assert bool(torch.isfinite(a).all())
    assert all(torch.equal(a,b) for a,b in zip(before,source.parameters()))
    torch.cuda.synchronize()
    return {'status':'FINITE_GPU_PRECHECK_PASSED','device':properties.name,
        'uuid':str(getattr(properties,'uuid','unavailable')),'environment':environment(),
        'source_identity':identity,'synthetic_signal_trials':4,
        'source_parameters_changed':False,'real_EEG_performance_validated':False,
        'new_allocations_created':0,'jobs_submitted':0}
