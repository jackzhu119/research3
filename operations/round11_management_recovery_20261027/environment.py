"""Shared stdlib-only bootstrap for independent R11 management processes."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import legacy_controller as legacy
import resume_io
from state_io import install_state_patch, required_json, StateUnavailable, StateIdentityError

DEFAULT_PLAN = resume_io.DEFAULT_PLAN
DEFAULT_RECIPE = legacy.DEFAULT_RECIPE
REPAIR_ID = 'paper3-r11-management-recovery-v2'


def load_env(package, plan_path, *, expected_recipe=DEFAULT_RECIPE):
    env = resume_io.load_env(package, plan_path, expected_recipe=expected_recipe)
    env['state_patch'] = install_state_patch(env)
    env['repair_id'] = REPAIR_ID
    return env


def validate_history(env):
    """Read required authorization/allocation history with retry, never defaults."""
    h, runtime = env['header'], env['runtime']
    authorization = required_json(runtime/'SUBMIT_AUTHORIZATION.json')
    if (authorization.get('explicit_submit') is not True
            or authorization.get('plan_sha256') != h['plan_sha256']
            or authorization.get('recipe_sha256') != h['recipe_sha256']):
        raise StateIdentityError('Original exact R11 submit authorization differs')
    state = required_json(runtime/'scheduler-state.json')
    marker = legacy.scheduler_identity(runtime, h['recipe_sha256'])
    if state.get('marker') != marker:
        raise StateIdentityError('Original R11 scheduler marker differs')
    slots = state.get('slots')
    if not isinstance(slots, dict) or set(slots) != {'0','1','2','3'}:
        raise StateUnavailable(runtime/'scheduler-state.json', ValueError('Four retained slots required'), 1)
    jobs = []
    for slot in range(4):
        row = slots[str(slot)]
        attempts = row.get('attempts') if isinstance(row, dict) else None
        if not isinstance(attempts, list) or not attempts:
            raise StateUnavailable(runtime/'scheduler-state.json', ValueError('Allocation history required'), 1)
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise StateUnavailable(runtime/'scheduler-state.json', ValueError('Attempt object required'), 1)
            if attempt.get('marker') != marker:
                raise StateIdentityError('Retained R11 allocation marker differs')
            job = attempt.get('job_id')
            if job is not None:
                if not str(job).isdigit():
                    raise StateUnavailable(runtime/'scheduler-state.json', ValueError('Allocation job ID invalid'), 1)
                jobs.append(str(job))
    if not jobs:
        raise StateUnavailable(runtime/'scheduler-state.json', ValueError('Existing scientific allocations required'), 1)
    return dict(marker=marker, known_job_ids=sorted(set(jobs)), prior_authorization_verified=True,
                tasks_reset=False, scientific_results_changed=False)


def load_launch(env):
    launch = required_json(env['runtime']/'launch.json')
    h = env['header']
    if (launch.get('plan_sha256') != h['plan_sha256']
            or launch.get('campaign_identity') != h['recipe_sha256']
            or Path(launch.get('runtime','')).resolve() != env['runtime'].resolve()):
        raise StateIdentityError('Original R11 launch history differs')
    if not isinstance(launch.get('worker_command'),list) or not launch['worker_command']:
        raise StateUnavailable(env['runtime']/'launch.json', ValueError('Original worker command required'), 1)
    return launch
