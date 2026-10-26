"""Explicit bounded CPU checks. Actual GPU precheck is optional and allocation-local."""
from __future__ import annotations
import importlib.metadata,os,platform
from pathlib import Path

def environment():
 result={'python':platform.python_version()}
 for name in ('numpy','torch','pytest'):
  try:result[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:result[name]=None
 return result

def cpu_check():
 import tempfile,numpy as np
 from . import algorithm as a
 x=np.array([[-2.,0.],[-1.,.2],[1.,-.2],[2.,0.]])
 y=np.array([0,0,1,1]);z=np.zeros((4,2))
 reference={'mean':np.zeros(2),'whitener':np.eye(2),'eigenvectors':np.eye(2),'provenance':{'role':'meta_train'}}
 pairs=[{'training_role':'meta_train','participant_id':'software-fixture','session_id':str(i),
  'support_features':x,'support_logits':z,'support_y':y,'future_features':x*.9,'future_logits':z,'future_y':y,
  'support_ids':tuple('s%d-%d'%(i,j) for j in range(4)),
  'future_ids':tuple('q%d-%d'%(i,j) for j in range(4))} for i in range(2)]
 op=a.train_operator(pairs,a.Config(feature_rank=2,ridge=.001),reference=reference)
 state=a.adapt(op,x,z,y,support_ids=('t0','t1','t2','t3'))
 p,diag=a.predict(state,x*.8,z,query_ids=('u0','u1','u2','u3'))
 assert np.isfinite(p).all() and np.allclose(p.sum(1),1.)
 assert np.any(abs(p[:,1]-.5)>1e-6)
 with tempfile.TemporaryDirectory(prefix='r11-cpu-check-') as temp:
  path=Path(temp)/'operator.npz';a.save_operator(op,path)
  loaded=a.load_operator(path,expected_operator_identity=op.operator_identity)
  path=Path(temp)/'calibration.npz';a.save_state(state,path)
  resumed=a.load_state(path,expected_fit_identity=state.fit_identity)
  p2,_=a.predict(resumed,x*.8,z,query_ids=('u0','u1','u2','u3'));assert np.array_equal(p,p2)
 return {'status':'CPU_SOFTWARE_CHECK_PASSED','checks':['meta_train_only_operator_fit','nonzero_support_to_future_update',
  'proper_probabilities','exact_pickle_free_resume'],'environment':environment(),'operator_identity':op.operator_identity,
  'diagnostics':diag,'gpu_initialized':False,'jobs_submitted':0,'real_EEG_performance_validated':False}

def gpu_check(root,registry_path):
 if not os.environ.get('SLURM_JOB_ID'):return {'status':'NOT_RUN_NO_EXISTING_ALLOCATION','gpu_used':False,'jobs_submitted':0}
 import torch
 if not torch.cuda.is_available():return {'status':'NOT_RUN_CUDA_UNAVAILABLE','gpu_used':False,'jobs_submitted':0}
 from .study import describe
 from .data import load_registry,load_source
 plan={'root':str(Path(root).resolve()),'study_registry':describe(registry_path,root)}
 registry,_=load_registry(plan)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 source,geometry,identity=load_source(plan,registry,20261018,'cuda:0')
 before={k:v.detach().clone() for k,v in source.state_dict().items()}
 x=torch.arange(4*11*128,device='cuda:0',dtype=torch.float64).reshape(4,11,128)/10000
 with torch.inference_mode():logits=source.source_logits(x)
 assert logits.shape==(4,2) and bool(torch.isfinite(logits).all())
 assert all(torch.equal(v,before[k]) for k,v in source.state_dict().items())
 return {'status':'FINITE_GPU_PRECHECK_PASSED','device':torch.cuda.get_device_name(0),
  'environment':environment(),'source_identity':identity,'synthetic_trials':4,'source_weights_changed':False,
  'new_allocations_created':0,'jobs_submitted':0,'real_EEG_performance_validated':False}
