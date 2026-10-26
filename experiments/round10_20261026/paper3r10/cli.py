"""R10 install/validation/planning/freeze/submission boundaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/public/home/zhuziyuan/projects/paper3')
PACKAGE=Path(__file__).resolve().parents[1]


def _print(value):
    print(json.dumps(value,ensure_ascii=False,indent=2,default=str),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command')
    v=sub.add_parser('validate');v.add_argument('--cpu',action='store_true');v.add_argument('--full-tests',action='store_true')
    for name in ('inputs','plan','gpu-precheck'):
        q=sub.add_parser(name);q.add_argument('--root',type=Path,default=ROOT)
        q.add_argument('--registry',type=Path);q.add_argument('--registry-overrides',type=Path)
        if name=='plan':
            q.add_argument('--stage',choices=('mechanism','formal_exploratory','formal_confirmatory'),default='mechanism')
            q.add_argument('--output',type=Path);q.add_argument('--plan-file',type=Path)
            q.add_argument('--freeze',type=Path);q.add_argument('--confirmation',type=Path)
    for name in ('submit','resume','status','report','export'):
        q=sub.add_parser(name);q.add_argument('--plan',required=True,type=Path)
        if name in ('submit','resume'):
            q.add_argument('--submit',action='store_true');q.add_argument('--retry-blocked',action='store_true')
    q=sub.add_parser('freeze');q.add_argument('--plan',required=True,type=Path);q.add_argument('--output',type=Path)
    return p


def _registry(args):
    from .registry import build_registry,write_registry
    from .study import describe
    root=args.root.expanduser().resolve()
    if args.registry:
        path=args.registry.expanduser().resolve();value=json.loads(path.read_text())
        if Path(value['root']).resolve()!=root:raise ValueError('Registry root differs')
        return value,describe(path,root),path
    overrides=json.loads(args.registry_overrides.read_text()) if args.registry_overrides else None
    value=build_registry(root,overrides=overrides);descriptor=write_registry(value)
    return value,descriptor,root/descriptor['path']


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command is None:
        parser().print_help();return 0
    try:
        if args.command=='validate':
            from .selfcheck import cpu_check
            result=cpu_check();_print(result)
            if args.full_tests:
                import importlib.util
                if importlib.util.find_spec('pytest') is None:
                    raise ValueError('Full tests require pytest8.4.2; no dependency is installed automatically')
                return subprocess.run([sys.executable,'-m','pytest','-q',str(PACKAGE/'tests')],cwd=PACKAGE).returncode
            return 0
        if args.command in ('inputs','plan','gpu-precheck'):
            registry,descriptor,registry_path=_registry(args)
            if args.command=='gpu-precheck':
                from .selfcheck import gpu_check
                _print(gpu_check(args.root,registry_path));return 0
            from .registry import validate_registry
            if args.command=='inputs':
                _print({'registry':descriptor,'availability':validate_registry(registry)});return 0
            from .study import build_plan,describe,write_plan,METHODS,SEEDS
            formal=args.stage!='mechanism'
            cohorts=['holdout','pan2023','weibo2014'] if args.stage=='formal_exploratory' else []
            availability=validate_registry(registry,stage=args.stage,methods=list(METHODS) if formal else [],
                                           seeds=list(SEEDS) if formal else [SEEDS[0]],cohorts=cohorts)
            if availability['status']!='READY':
                _print({'status':'BLOCKED_MISSING_INPUTS','registry':descriptor,**availability});return 2
            freeze=describe(args.freeze,args.root) if args.freeze else None
            confirmation=describe(args.confirmation,args.root) if args.confirmation else None
            plan=build_plan(args.root,descriptor,stage=args.stage,output=args.output,
                            freeze=freeze,confirmation=confirmation,python=sys.executable)
            path=args.plan_file or args.root/'runs/round10_preparation'/('plan-'+args.stage+'.json')
            path=write_plan(plan,path)
            from .platform.runtime import prepare
            _print(prepare(path));return 0
        if args.command in ('submit','resume'):
            from .platform.runtime import start
            if args.retry_blocked:
                from .platform.runtime import retry_blocked
                if not args.submit:raise ValueError('Retrying blocked tasks requires explicit --submit')
                retry_blocked(args.plan)
            _print(start(args.plan,submit=args.submit));return 0
        if args.command=='status':
            from .platform.runtime import status
            _print(status(args.plan));return 0
        if args.command=='export':
            from .platform.uploads import start_uploads
            _print(start_uploads(args.plan));return 0
        if args.command=='report':
            from .report import generate_report
            _print(generate_report(args.plan));return 0
        if args.command=='freeze':
            from .selection import freeze_development
            _print(freeze_development(args.plan,args.output));return 0
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        _print({'status':'R10_COMMAND_BLOCKED','error_type':type(error).__name__,
                'reason':str(error),'implicit_gpu_allocation':False,'implicit_job_submission':False})
        return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
