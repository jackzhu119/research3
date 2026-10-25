"""Independent existing-result report process. No torch, network or training imports."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import sysconfig
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from three_jobs import atomic,emit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--operational',required=True)
    parser.add_argument('--package',default=str(HERE.parent))
    args=parser.parse_args(argv)
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text())
    operational=Path(args.operational);operational.mkdir(parents=True,exist_ok=True)
    with (operational/'report-build.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            emit('ROUND_8_REPORT_EXISTING_PROCESS_REUSED');return 73
        sys.path.insert(0,str(Path(args.package).resolve()))
        prefix=Path(sys.executable).absolute().parent.parent
        version='python%d.%d'%sys.version_info[:2]
        candidates=[prefix/lib/version/'site-packages' for lib in ('lib','lib64')]
        candidates += [Path(sysconfig.get_paths()[key]) for key in ('purelib','platlib')]
        for path in candidates:
            if path.is_dir() and str(path) not in sys.path:sys.path.append(str(path))
        emit('ROUND_8_REPORT_STARTED',plan=str(plan_path),training_controlled=False)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[key]='1'
        from paper3r8.report import build_report
        value=build_report(plan,output=Path(plan['output']))
        if not isinstance(value,dict):
            raise ValueError('Report builder must return its written summary')
        state={'status':'BUILT','recipe_sha256':plan['recipe_sha256'],
            'complete_tasks':value.get('complete_tasks'),
            'total_tasks':value.get('total_tasks'),'report_status':value.get('status'),
            'built_at':time.time(),'report':str(Path(plan['output'])/'reports/report.json')}
        atomic(operational/'report-state.json',state)
        emit('ROUND_8_REPORT_BUILT',**state,training_controlled=False)
        return 0


if __name__=='__main__':raise SystemExit(main())
