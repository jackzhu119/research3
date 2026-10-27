"""R11 direct install/prepare/explicit launch and observation commands."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path('/public/home/zhuziyuan/projects/paper3')
PACKAGE=Path(__file__).resolve().parents[1]
STAGES=('campaign','development','exploratory','confirmatory')

def _print(value):print(json.dumps(value,ensure_ascii=False,indent=2,default=str),flush=True)

def parser():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command')
 q=sub.add_parser('validate');q.add_argument('--cpu',action='store_true');q.add_argument('--full-tests',action='store_true')
 for name in ('inputs','plan','launch','gpu-precheck'):
  q=sub.add_parser(name);q.add_argument('--root',type=Path,default=ROOT);q.add_argument('--registry',type=Path)
  q.add_argument('--registry-overrides',type=Path)
  if name in ('plan','launch'):
   q.add_argument('--stage',choices=STAGES,default='campaign');q.add_argument('--output',type=Path)
   q.add_argument('--plan-file',type=Path);q.add_argument('--grid',type=Path);q.add_argument('--confirmation',type=Path)
   if name=='launch':q.add_argument('--submit',action='store_true')
 for name in ('submit','resume','status','report','export','select'):
  q=sub.add_parser(name);q.add_argument('--plan',type=Path,required=True)
  if name in ('submit','resume'):
   q.add_argument('--submit',action='store_true');q.add_argument('--retry-blocked',action='store_true')
 return p

def _registry(args):
 from .registry import build_registry,write_registry
 from .study import describe
 root=args.root.expanduser().resolve()
 if args.registry:
  path=args.registry.expanduser().resolve();value=json.loads(path.read_text())
  if Path(value['root']).resolve()!=root:raise ValueError('Study registry root differs')
  return value,describe(path,root),path
 override=json.loads(args.registry_overrides.read_text()) if args.registry_overrides else None
 value=build_registry(root,overrides=override);descriptor=write_registry(value)
 return value,descriptor,root/descriptor['path']

def prepare_research(args):
 from .study import build_plan,describe,write_plan,METHODS,SEEDS
 from .platform.runtime import prepare
 # An existing exact local plan is resumed without repeated registry discovery.
 path=Path(args.plan_file or args.root/'runs/round11_preparation'/('plan-'+args.stage+'.json')).expanduser().resolve()
 if path.is_file():return path,prepare(path)
 registry,descriptor,_=_registry(args)
 from .registry import validate_registry
 cohorts=('holdout','pan2023','weibo2014') if args.stage in ('campaign','exploratory') else ()
 availability=validate_registry(registry,stage=args.stage,methods=list(METHODS),seeds=list(SEEDS),cohorts=list(cohorts))
 if availability['status']!='READY':
  raise ValueError('Required source/cache data are unavailable: '+json.dumps(availability['missing_artifacts'],ensure_ascii=False))
 grid=json.loads(args.grid.read_text()) if args.grid else None
 confirmation=describe(args.confirmation,args.root) if args.confirmation else None
 plan=build_plan(args.root,descriptor,stage=args.stage,output=args.output,grid=grid,
                 confirmation=confirmation,python=sys.executable)
 write_plan(plan,path)
 result=prepare(path)
 result['unavailable_optional_baselines']=availability.get('unavailable_baselines',[])
 return path,result

def main(argv=None):
 args=parser().parse_args(argv)
 if not args.command:parser().print_help();return 0
 try:
  if args.command=='validate':
   from .selfcheck import cpu_check
   _print(cpu_check())
   if args.full_tests:return subprocess.run([sys.executable,'-m','pytest','-q',str(PACKAGE/'tests')],cwd=PACKAGE).returncode
   return 0
  if args.command=='inputs':
   from .registry import validate_registry
   registry,descriptor,_=_registry(args);_print({'registry':descriptor,'availability':validate_registry(registry)});return 0
  if args.command=='gpu-precheck':
   from .selfcheck import gpu_check
   _,_,path=_registry(args);_print(gpu_check(args.root,path));return 0
  if args.command in ('plan','launch'):
   path,result=prepare_research(args)
   if args.command=='launch' and args.submit:
    from .platform.runtime import start
    result=start(path,submit=True)
   _print(result)
   return 2 if result.get('status')=='CONTROLLER_FAILED_TO_START' else 0
  if args.command in ('submit','resume'):
   from .platform.runtime import start,retry_blocked
   if args.retry_blocked:
    if not args.submit:raise ValueError('Explicit --submit required to retry isolated failed tasks')
    retry_blocked(args.plan)
   result=start(args.plan,submit=args.submit);_print(result)
   return 2 if result.get('status')=='CONTROLLER_FAILED_TO_START' else 0
  if args.command=='status':
   from .platform.runtime import status
   _print(status(args.plan));return 0
  if args.command=='report':
   from .report import generate_report
   _print(generate_report(args.plan));return 0
  if args.command=='select':
   from .selection import execute_selection
   _print(execute_selection(args.plan));return 0
  if args.command=='export':
   from .platform.uploads import start_uploads
   _print(start_uploads(args.plan));return 0
 except (OSError,ValueError,KeyError,RuntimeError) as error:
  _print({'status':'R11_COMMAND_ERROR','error_type':type(error).__name__,'reason':str(error),
          'implicit_gpu_allocation':False,'implicit_job_submission':False});return 2
 return 0
if __name__=='__main__':raise SystemExit(main())
