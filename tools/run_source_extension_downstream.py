#!/usr/bin/env python3
"""Independent five-source 40/11 development; no internal/external execution."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from p3work import source_extension_downstream as run


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--arm',choices=tuple(run.ARMS),required=True)
    p.add_argument('--stage',choices=('plan','initialize','budget','f0','reference','source-bank','meta-engineering',
        'meta','profile','evaluate','select','freeze','auto-development'),default='plan')
    p.add_argument('--catalog')
    p.add_argument('--primary-freeze')
    p.add_argument('--extension-manifest')
    p.add_argument('--seed',type=int,choices=run.DEVELOPMENT_SEEDS,default=run.SEEDS[0])
    p.add_argument('--method',choices=run.PRIMARY_METHODS)
    p.add_argument('--configuration-index',type=int,default=0)
    p.add_argument('--phase',choices=('screening','confirmation'),default='screening')
    p.add_argument('--max-actions',type=int)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--user-started-gpu',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.execute and not a.user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
        if a.stage=='plan':result=run.readiness(a.root,a.arm)
        elif a.stage=='initialize':
            if not a.primary_freeze or not a.extension_manifest:raise run.DownstreamError('real_primary_freeze_and_complete_extension_manifest_required')
            result=run.initialize_catalog(a.root,a.arm,a.primary_freeze,a.extension_manifest,execute=a.execute)
        else:
            catalog=a.catalog or f'runs/source_extension_downstream/{a.arm}/launch.json'
            kwargs={'execute':a.execute,'user_started_gpu':a.user_started_gpu}
            if a.stage=='budget':result=run.prepare_budget(a.root,catalog,execute=a.execute)
            elif a.stage=='f0':result=run.execute_stage(a.root,catalog,'f0',args=(a.seed,),**kwargs)
            elif a.stage=='reference':result=run.execute_stage(a.root,catalog,'reference',args=(a.seed,),**kwargs)
            elif a.stage=='source-bank':result=run.execute_stage(a.root,catalog,'source-bank',**kwargs)
            elif a.stage in {'meta-engineering','meta'}:
                if a.method not in run.LEARNED:raise run.DownstreamError('learned_method_required')
                config=run.configuration(a.method,a.configuration_index)
                phase='engineering' if a.stage=='meta-engineering' else a.phase
                if phase=='engineering':config=max(run.registered_grid(a.method),key=lambda c:(c.get('rank',c.get('dimension')),c['radius']))
                job=run.MetaJob(a.method,config.get('dimension',config.get('rank')),config['radius'],a.seed,phase)
                result=run.execute_stage(a.root,catalog,'meta',args=(job,),**kwargs)
            elif a.stage=='profile':result=run.execute_stage(a.root,catalog,'profile',**kwargs)
            elif a.stage=='evaluate':
                if not a.method:raise run.DownstreamError('method_required')
                result=run.execute_stage(a.root,catalog,'evaluate',args=(a.method,a.configuration_index,a.seed,a.phase),**kwargs)
            elif a.stage=='select':result=run.execute_stage(a.root,catalog,'select',**kwargs)
            elif a.stage=='freeze':result=run.execute_stage(a.root,catalog,'freeze',**kwargs)
            else:result=run.auto_development(a.root,catalog,max_actions=a.max_actions,**kwargs)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result.get('status') not in {'PAUSED_NOT_COMPLETE','BLOCKED'} else 3
    except (ValueError,PermissionError,OSError,KeyError,TypeError,RuntimeError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc) if isinstance(exc,(run.DownstreamError,PermissionError)) else 'dependency_or_runtime_failed_details_suppressed',
            'scientific_fit_completion_claim':False,'GPU_actions_this_invocation':None if a.execute else 0,
            'server_actions':False,'external_access':False,'internal_access':False},indent=2))
        return 2


if __name__=='__main__':raise SystemExit(main())
