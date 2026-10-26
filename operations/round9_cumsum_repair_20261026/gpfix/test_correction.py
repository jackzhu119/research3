"""CPU regression and explicit CUDA routing contract; no GPU execution claim."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from correction import corrected_source, evidence, ORIGINAL_GP_SHA256, OLD_CUMSUM, NEW_CUMSUM

NATIVE=Path('/workspace/paper3-round9/paper3r9/gp.py')


def load_source(source, name):
    module=types.ModuleType(name);module.__file__=str(NATIVE)
    sys.modules[name]=module
    exec(compile(source,str(NATIVE),'exec'),module.__dict__)
    return module


@pytest.fixture(autouse=True)
def settings():
    prior=torch.get_num_threads();det=torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(det);torch.set_num_threads(prior)


def test_exact_scoped_correction_is_idempotent_and_native_bytes_unchanged():
    raw=NATIVE.read_bytes();source=raw.decode()
    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_GP_SHA256
    fixed,labels=corrected_source('paper3r9.gp',source)
    assert labels==['weighted_reference_deterministic_cpu_cumsum']
    assert source.replace(OLD_CUMSUM,NEW_CUMSUM,1)==fixed
    assert corrected_source('paper3r9.gp',fixed)==(fixed,[])
    assert corrected_source('paper3r8.gp',source)==(source,[])
    assert NATIVE.read_bytes()==raw
    with pytest.raises(RuntimeError):corrected_source('paper3r9.gp',source.replace('pairweights[order],0','pairweights,0'))


def test_cuda_routing_contract_old_raises_fixed_copies_only_prefix_weights():
    """Simulate only PyTorch's documented CUDA deterministic cumsum guard.

    A fake CUDA vector has an actual CPU payload. A CPU cumsum is permitted;
    returning to its requested device is observed without executing CUDA.
    """
    actual=torch.tensor([.125,.125,.25,.5],dtype=torch.float64)
    route=[]
    class CUDAWeights:
        device='cuda:0'
        def __getitem__(self,index):return self
        def detach(self):route.append('detach_cuda_weights');return self
        def cpu(self):route.append('copy_weights_to_cpu');return actual
    class Prefix:
        def __init__(self,value):self.value=value
        def to(self,device):
            route.append(('return_to_device',device));return self
    class Torch:
        @staticmethod
        def cumsum(value,dim):
            if isinstance(value,CUDAWeights):
                raise RuntimeError('cumsum_cuda_kernel does not have a deterministic implementation')
            assert value.device.type=='cpu' and value.dtype==torch.float64 and dim==0
            route.append('cpu_float64_cumsum');return Prefix(torch.cumsum(value,dim))
    namespace={'torch':Torch,'pairweights':CUDAWeights(),'order':None}
    with pytest.raises(RuntimeError,match='cumsum_cuda_kernel'):
        exec(OLD_CUMSUM.strip(),namespace)
    exec(NEW_CUMSUM.strip(),namespace)
    torch.testing.assert_close(namespace['cumulative'].value,torch.tensor([.125,.25,.5,1.],dtype=torch.float64))
    assert route==['detach_cuda_weights','copy_weights_to_cpu','cpu_float64_cumsum',('return_to_device','cuda:0')]
    # Left-inclusive half-mass tie is unchanged: searchsorted(..., right=False).
    cumulative=namespace['cumulative'].value
    assert int(torch.searchsorted(cumulative,cumulative[-1]*.5))==2


@pytest.mark.parametrize('n,d',[(4,1),(19,64),(7920,64)])
def test_reference_end_to_end_same_weighted_geometry_and_lengthscale(n,d):
    source=NATIVE.read_text();fixed,_=corrected_source('paper3r9.gp',source)
    before=load_source(source,'r9_original_gp_'+str(n))
    after=load_source(fixed,'r9_fixed_gp_'+str(n))
    generator=torch.Generator().manual_seed(19)
    features=torch.randn(n,d,generator=generator,dtype=torch.float64)
    weights=torch.arange(1,n+1,dtype=torch.float64);weights/=weights.sum()
    a=before.fit_reference_geometry(features,weights=weights,provenance={'role':'meta_train'})
    b=after.fit_reference_geometry(features,weights=weights,provenance={'role':'meta_train'})
    torch.testing.assert_close(a.mean,b.mean,rtol=0,atol=0)
    torch.testing.assert_close(a.whitener,b.whitener,rtol=0,atol=0)
    assert a.lengthscale==b.lengthscale
    assert a.diagnostics==b.diagnostics
    # Independently calculate the same weighted lower median from selected pairs.
    white=(features-b.mean)@b.whitener
    indices=torch.linspace(0,n-1,min(n,256)).round().long().unique()
    distances=torch.cdist(white[indices],white[indices]).square()/d
    rows,cols=torch.triu_indices(len(indices),len(indices),offset=1)
    pairs=sorted(zip(distances[rows,cols].tolist(),(weights[indices[rows]]*weights[indices[cols]]).tolist()))
    half=sum(w for _,w in pairs)*.5;running=0.
    for distance,w in pairs:
        running+=w
        if running>=half:
            expected=max(math.sqrt(distance),1e-6);break
    assert b.lengthscale==pytest.approx(expected,rel=1e-12,abs=1e-12)


def test_tie_boundary_and_reference_state_restoration():
    fixed,_=corrected_source('paper3r9.gp',NATIVE.read_text())
    module=load_source(fixed,'r9_fixed_gp_tie')
    x=torch.tensor([[-3.],[-1.],[1.],[3.]],dtype=torch.float64)
    reference=module.fit_reference_geometry(x,weights=torch.full((4,),.25,dtype=torch.float64))
    assert reference.lengthscale==pytest.approx(math.sqrt(.8),rel=1e-12)
    restored=module.ReferenceGeometry.from_state_dict(reference.state_dict())
    torch.testing.assert_close(reference.transform(x),restored.transform(x),rtol=0,atol=0)
    fit=module.fit_gp(restored.transform(x),torch.zeros(4,dtype=torch.float64),torch.tensor([0,1,0,1]),
        module.GPConfig(kernel='rbf',lengthscale=restored.lengthscale))
    p=fit.predict_proba(restored.transform(x),torch.zeros(4,dtype=torch.float64))
    assert bool(torch.isfinite(p).all())
    torch.testing.assert_close(p.sum(1),torch.ones(4,dtype=torch.float64))


def test_fix_keeps_determinism_enabled():
    fixed,_=corrected_source('paper3r9.gp',NATIVE.read_text())
    assert 'use_deterministic_algorithms(False)' not in fixed
    module=load_source(fixed,'r9_fixed_gp_deterministic')
    module.fit_reference_geometry(torch.eye(4,dtype=torch.float64))
    assert torch.are_deterministic_algorithms_enabled()
