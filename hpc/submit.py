#!/usr/bin/env python3
"""Render or submit one school Slurm job; never execute a scientific stage here."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
STAGES = ('test', 'prepare', 'pilot', 'train', 'backup')


class SubmissionError(ValueError):
    pass


def need(test, code):
    if not test:
        raise SubmissionError(code)


def parse_time(value):
    """Slurm-compatible D-HH:MM:SS or HH:MM:SS, returned as seconds."""
    need(isinstance(value, str), 'wall_time_string_required')
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', value)
    need(match is not None, 'wall_time_requires_HH_MM_SS_or_D_HH_MM_SS')
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    need(minutes < 60 and seconds < 60 and (not match[1] or hours < 24), 'wall_time_fields_invalid')
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    need(total > 0, 'wall_time_must_be_positive')
    return total


def absolute(value):
    path = Path(os.path.abspath(os.fspath(value)))
    need(not any(ord(c) < 32 or ord(c) == 127 for c in str(path)), 'control_character_in_path')
    return path


def load_config(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    need(isinstance(value, dict) and value.get('schema_version') == 'paper3-hpc-resources-v1',
         'wrong_hpc_resource_configuration')
    need(set(value.get('stages', {})) == set(STAGES), 'five_hpc_stage_profiles_required')
    return value


def build_request(args, config):
    stage = args.stage
    defaults = config['stages'][stage]
    project = absolute(args.root or config['project'])
    python = absolute(args.python or config['python'])
    account = args.account or config['account']
    partition = args.partition or defaults['partition']
    cpus = args.cpus if args.cpus is not None else defaults['cpus']
    memory = args.mem_gib if args.mem_gib is not None else defaults['mem_gib']
    gpus = args.gpus if args.gpus is not None else defaults['gpus']
    wall_time = args.time or defaults['time']
    need(re.fullmatch(r'[A-Za-z0-9_-]+', account) is not None, 'account_name_invalid')
    need(partition in config.get('partitions', {}), 'partition_requires_confirmed_configuration')
    capacity = config['partitions'][partition]
    for number, label in ((cpus, 'cpus'), (memory, 'mem_gib'), (gpus, 'gpus')):
        need(type(number) is int and number >= (0 if label == 'gpus' else 1), 'invalid_' + label)
    need(cpus <= capacity['node_cpus'] and memory <= capacity['node_memory_gib']
         and gpus <= capacity['node_gpus'], 'request_exceeds_reported_single_node_hardware')
    need((stage in ('pilot', 'train') and gpus >= 1) or (stage not in ('pilot', 'train') and gpus == 0),
         'GPU_allocation_must_match_stage')
    wall_seconds = parse_time(wall_time)
    limits = config.get('recommendation_limits', {})
    # A configured school limit is a hard cap. Null means unknown, never unlimited
    # evidence: both scheduler test and submission still have to accept it.
    for name, actual in (('max_cpus', cpus), ('max_mem_gib', memory), ('max_gpus', gpus),
                         ('max_time_hours', wall_seconds / 3600)):
        maximum = limits.get(name)
        if maximum is not None:
            need(type(maximum) in (int, float) and maximum >= 0 and actual <= maximum,
                 'request_exceeds_confirmed_school_limit_' + name)
    if args.dependency:
        need(re.fullmatch(r'(?:afterok|afterany|afternotok):\d+(?::\d+)*', args.dependency),
             'dependency_expression_invalid')
    logs = project / '.paper3-hpc' / 'logs'
    common = ['--account=' + account, '--partition=' + partition, '--nodes=1', '--ntasks=1',
              '--cpus-per-task=' + str(cpus), '--mem=' + str(memory) + 'G', '--time=' + wall_time,
              '--job-name=paper3-' + stage, '--signal=B:USR1@180', '--open-mode=append',
              '--output=' + str(logs / (stage + '-%j.out')),
              '--error=' + str(logs / (stage + '-%j.err')), '--chdir=' + str(project), '--export=ALL']
    if gpus:
        common.append('--gres=gpu:' + str(gpus))
    if args.dependency:
        common.append('--dependency=' + args.dependency)
    tail = [str(project / 'hpc' / 'job.sbatch'), '--root', str(project), '--python', str(python),
            '--stage', stage, '--wall-seconds', str(wall_seconds)]
    return {'schema_version': 'paper3-slurm-submission-v1', 'stage': stage,
            'project': str(project), 'python': str(python),
            'resources': {'account': account, 'partition': partition, 'nodes': 1, 'tasks': 1,
                          'cpus': cpus, 'mem_gib': memory, 'gpus': gpus, 'time': wall_time},
            'school_quota_known': any(value is not None for value in limits.values()),
            'scheduler_preflight': ['sbatch', '--test-only', *common, *tail],
            'command': ['sbatch', '--parsable', *common, *tail],
            'scientific_budget_changed': False, 'scientific_threads_changed': False,
            'multi_gpu_execution_validated': False, 'credentials_recorded': False}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.submission-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def submit(request, *, runner=subprocess.run):
    root = Path(request['project'])
    python = Path(request['python'])
    need(root.is_dir() and (root / 'hpc/job.sbatch').is_file()
         and (root / 'hpc/run_stage.py').is_file(), 'deploy_hpc_code_to_project_first')
    need(python.is_file() and os.access(python, os.X_OK), 'existing_school_python_not_executable')
    need(shutil.which('sbatch') is not None, 'sbatch_not_available_on_submission_host')
    (root / '.paper3-hpc/logs').mkdir(parents=True, exist_ok=True)
    preflight = runner(request['scheduler_preflight'], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, check=False, timeout=60)
    need(preflight.returncode == 0, 'Slurm_test_only_rejected_request_no_job_submitted')
    result = runner(request['command'], stdin=subprocess.DEVNULL, capture_output=True,
                    text=True, check=False, timeout=60)
    need(result.returncode == 0, 'Slurm_submission_rejected_no_success_claim')
    # --parsable can append a federation cluster after a semicolon.
    job_id = result.stdout.strip().split(';')[0]
    need(re.fullmatch(r'\d+', job_id) is not None, 'Slurm_submission_id_not_verified')
    receipt = {**request, 'status': 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE', 'job_id': job_id,
               'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
               'scientific_completion_claimed': False, 'upload_claimed': False}
    path = root / '.paper3-hpc/submissions' / (job_id + '-' + uuid.uuid4().hex[:8] + '.json')
    atomic_json(path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description='Submit one Paper3 stage through school Slurm; never train on login1.')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, default=HERE / 'resources.json')
    parser.add_argument('--root', type=Path, help='Deployed shared project directory')
    parser.add_argument('--python', type=Path, help='Existing school environment interpreter; no install performed')
    parser.add_argument('--account')
    parser.add_argument('--partition')
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--mem-gib', type=int)
    parser.add_argument('--gpus', type=int, help='Allocation only; scientific runtime remains single GPU without validated scaling')
    parser.add_argument('--time', help='Administrative Slurm wall time (HH:MM:SS or D-HH:MM:SS), not scientific budget')
    parser.add_argument('--dependency', help='For example afterok:123456')
    parser.add_argument('--dry-run', action='store_true', help='Only render request, do not call sbatch or create a job')
    args = parser.parse_args(argv)
    try:
        request = build_request(args, load_config(args.config))
        if args.dry_run:
            print(json.dumps({**request, 'status': 'DRY_RUN_NO_JOB_SUBMITTED',
                              'command_display': shlex.join(request['command'])}, ensure_ascii=False, indent=2))
            return 0
        receipt = submit(request)
        print(json.dumps({'status': receipt['status'], 'job_id': receipt['job_id'],
                          'logs': str(Path(receipt['project']) / '.paper3-hpc/logs'),
                          'resources': receipt['resources']}, ensure_ascii=False, indent=2))
        return 0
    except (SubmissionError, OSError, subprocess.TimeoutExpired, ValueError) as exc:
        code = str(exc) if isinstance(exc, SubmissionError) else type(exc).__name__
        print('Submission blocked: ' + code + '; existing checkpoints and jobs were not changed.')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
