"""Two explicitly authorized retries for the observed R1 B5 profile only.

The native profile owns every checkpoint, completed fit, cumulative charge and
budget-amendment history update. This wrapper only supplies a new immutable
operational budget to a future B5 action; it never edits progress or fit files.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import functools
import importlib
import json
from pathlib import Path
import time

VERSION = 'paper3-r1-b5-two-lost-attempt-retries-v1'
IDENTITY = '3a22efc255ca65bf5e9abd4f4fcc0477ad3626d0538f19ad622108ef2b3516e5'
OUTPUT = 'runs/training_campaign_v1/profiles/B5'
BASE_LIMIT = 9
RETRY_LIMIT = 11
KNOWN_ATTEMPTS = (2, 3, 4, 5, 6, 8, 9)
KNOWN_PLAN_SHA256 = '4418a0ee18aac64932a1c2f880dbecd1434be62f1595fa328fca74fa2685aae2'
KNOWN_JOB_IDS = (
    'c392e1651efee1252ed3133b012fc2360a9a0f6ed6153375654fbbeff1455596',
    'b8aa0b33565f9f67cc247f91ceb6352385930df7670cd415008969773bee5738',
    'd20070a29f3d7231222afb0f35609f5096bd6571f5e38506129293479f0df777',
    'fadb668e0319588e762f50a45cf0cf27841012830ef61fff8934501c19d139ac',
    '67e62ebe9212e5feb91f6da9d0ee23ce147a1e3d90503b255865da35fd3f4702',
    'e287b437b05912b2b2d68a79e72377e854913c7f5acb67224610ac63e4763d5c',
    '57e212aa7d5019eb62a38ac3df4905164e6978487a710e740e981511455cc210',
    'd1b159817836c6ca0f4c0b405d7a139ab5b12402c8acd789efd57d3382b994fd',
    '3f7794064e53a811f8ebfd953e63dbc0c7af940aeb61a3b6c44580e34e68978a')
KNOWN_GRID = tuple(dict(shrinkage=s,l2_to_source=l2)
                   for s in (.01,.05,.1) for l2 in (.01,.1,1.))
NOTE = ('User authorized R1 repair on 2026-10-27. Preserve the nine registered B5 '
        'configurations, all seven completed calibrations and all nine cumulative '
        'dispatch charges. Two historical dispatches (1 and 7) did not yield a '
        'completed calibration. Permit exactly two additional dispatches, cap 11; '
        'no completed fit is repeated and no scientific identity or per-fit calibration limit changes.')
_STATE = '_paper3_r1_b5_retry_budget_v1'


class B5RetryBudgetError(ValueError):
    pass


def _require(ok, reason):
    if not ok:
        raise B5RetryBudgetError(reason)


def _notice(emit, event, **fields):
    value = dict(repair_id=VERSION,
                 utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    try:
        if emit is None:
            print(json.dumps(dict(event=event, **value), ensure_ascii=False), flush=True)
        else:
            emit(event, **value)
    except Exception:
        pass


def amended_budget(state, budget, plan, *, original_budget=None):
    """Pure validated transform; neither arguments nor checkpoint are mutated."""
    _require(isinstance(state, dict) and isinstance(budget, dict) and isinstance(plan, dict),
             'B5 retry requires readable native state, budget and plan')
    jobs = plan.get('jobs')
    _require(plan.get('methods') == ['B5'] and plan.get('registered_calibrations') == 9
             and plan.get('plan_sha256') == KNOWN_PLAN_SHA256
             and isinstance(jobs, list) and len(jobs) == 9,
             'B5 retry requires the complete native nine-configuration plan')
    by_id = {}
    for index, row in enumerate(jobs):
        job = row.get('job', {}) if isinstance(row, dict) else {}
        _require(isinstance(row, dict) and job.get('method') == 'B5'
                 and type(job.get('configuration_index')) is int and job['configuration_index'] == index
                 and job.get('seed') == 20261018 and job.get('phase') == 'screening'
                 and row.get('job_sha256') == KNOWN_JOB_IDS[index]
                 and row.get('configuration') == KNOWN_GRID[index],
                 'B5 registered job order, seed or method changed')
        by_id[row['job_sha256']] = row
    _require(len(by_id) == 9, 'B5 registered job identities are not unique')
    _require(budget.get('stage') == 'development_calibration_profile'
             and budget.get('plan_sha256') == plan.get('plan_sha256')
             and isinstance(plan.get('plan_sha256'), str)
             and budget.get('max_jobs') == 1 and budget.get('max_download_bytes') == 0
             and type(budget.get('max_fit_attempts')) is int
             and budget['max_fit_attempts'] in (BASE_LIMIT, RETRY_LIMIT),
             'B5 budget stage, plan or attempt allowance differs')
    _require(state.get('identity_sha256') == IDENTITY,
             'B5 checkpoint identity differs from the observed repair incident')
    dispatched = state.get('fit_attempts_dispatched')
    cap = state.get('fit_attempt_limit')
    _require(type(dispatched) is int and BASE_LIMIT <= dispatched <= RETRY_LIMIT
             and state.get('initial_fit_attempt_limit') == BASE_LIMIT
             and type(cap) is int and cap in (BASE_LIMIT, RETRY_LIMIT)
             and dispatched <= cap,
             'B5 cumulative dispatch or retained initial/current limit differs')
    amendments = state.get('fit_limit_amendments')
    _require(isinstance(amendments, list), 'B5 native amendment history is missing')
    if cap == BASE_LIMIT:
        _require(amendments == [] and dispatched == BASE_LIMIT,
                 'B5 original cap has unexpected amendment/dispatch history')
    else:
        _require(len(amendments) == 1, 'B5 retry cap has unexpected additional amendments')
        change = amendments[0]
        _require(isinstance(change, dict) and change.get('previous') == BASE_LIMIT
                 and change.get('new') == RETRY_LIMIT
                 and change.get('scientific_identity_changed') is False
                 and isinstance(change.get('note'), str) and NOTE in change['note'],
                 'B5 retry amendment history is not this authorized repair')
    completed = state.get('completed')
    _require(isinstance(completed, dict) and 7 <= len(completed) <= 9
             and set(completed) <= set(by_id), 'B5 completed configuration identities differ')
    attempts = set()
    completed_indices = set()
    for key, record in completed.items():
        _require(isinstance(record, dict) and record.get('job') == by_id[key]['job'],
                 'B5 completed calibration job differs from the native plan')
        index, attempt = record['job']['configuration_index'], record.get('attempt')
        _require(type(attempt) is int and 0 < attempt <= dispatched and attempt not in attempts,
                 'B5 completed calibration attempt history differs')
        if index < 7:
            _require(attempt == KNOWN_ATTEMPTS[index], 'B5 historical completed fit was replaced')
        else:
            _require(attempt in (10, 11), 'B5 remaining fit has an unknown dispatch')
        attempts.add(attempt); completed_indices.add(index)
    _require(set(range(7)) <= completed_indices,
             'One of the seven observed completed B5 calibrations is absent')
    active = state.get('active')
    if active is not None:
        _require(isinstance(active, dict) and active.get('job_sha256') in by_id,
                 'B5 active dispatch has an unknown job')
        index = by_id[active['job_sha256']]['job']['configuration_index']
        attempt = active.get('attempt')
        _require(index in (7, 8) and index not in completed_indices
                 and type(attempt) is int and attempt == dispatched and attempt in (10, 11)
                 and active.get('phase') == 'FIT_DISPATCHED'
                 and active.get('directory') == OUTPUT + '/fits/' + active['job_sha256']
                     + '/attempt-%06d' % attempt,
                 'B5 active dispatch is not a retained remaining retry')
    result = copy.deepcopy(budget)
    result['max_fit_attempts'] = RETRY_LIMIT
    previous_note = result.get('operational_amendment_note', '')
    _require(isinstance(previous_note, str), 'B5 prior operational note is malformed')
    if NOTE not in previous_note:
        result['operational_amendment_note'] = (previous_note + '\n' + NOTE).strip()
    result['r1_b5_retry_repair'] = dict(
        repair_id=VERSION, observed_identity_sha256=IDENTITY,
        previous_cumulative_attempt_cap=BASE_LIMIT, cumulative_attempt_cap=RETRY_LIMIT,
        observed_completed_configurations=list(range(7)), lost_historical_dispatches=[1, 7],
        maximum_registered_configurations=9, completed_fits_repeated=False,
        scientific_identity_changed=False)
    if original_budget is not None:
        result['r1_b5_retry_original_budget'] = copy.deepcopy(original_budget)
    return result


def is_b5_action(action):
    return (isinstance(action, dict) and action.get('kind') == 'calibration_profile'
            and action.get('key') == 'profile-B5' and action.get('method') == 'B5'
            and action.get('expected_receipt') == OUTPUT + '/receipt.json')


@contextmanager
def active(root, action, *, campaign=None, development=None, emit=None):
    if not is_b5_action(action):
        yield None
        return
    root = Path(root).resolve(strict=True)
    campaign = campaign or importlib.import_module('p3work.training_campaign')
    development = development or importlib.import_module('p3work.development_evaluation')
    previous = campaign.Production.budget

    @functools.wraps(previous)
    def budget(self, action_key, *args, **kwargs):
        descriptor = previous(self, action_key, *args, **kwargs)
        if action_key != 'profile-B5':
            return descriptor
        _require(Path(self.root).resolve(strict=True) == root,
                 'B5 budget producer root differs from action root')
        if (root / OUTPUT / 'receipt.json').exists():
            # The original native consumer performs its full receipt proof.
            return descriptor
        plan = development.build_calibration_profile_plan(('B5',))
        limits = development._read_budget(root, descriptor,
            stage='development_calibration_profile', expected_identity=plan['plan_sha256'])
        path = root / OUTPUT / 'progress.json'
        state = json.loads(path.read_text())
        replacement = amended_budget(state, limits, plan, original_budget=descriptor)
        relative = campaign.OUTPUT + '/budgets/operational/b5-retry-' + campaign.digest(replacement) + '.json'
        result = campaign.immutable(root, relative, replacement)
        _notice(emit, 'R1_B5_TWO_LOST_DISPATCH_RETRY_BUDGET_SELECTED',
                cumulative_attempt_cap=RETRY_LIMIT, previous_attempt_cap=BASE_LIMIT,
                cumulative_dispatches_preserved=state['fit_attempts_dispatched'],
                completed_calibrations_preserved=len(state['completed']),
                progress_files_changed_by_adapter=0, original_budget=descriptor,
                amended_budget=result, per_fit_and_wall_gpu_limits_changed=False)
        return result

    campaign.Production.budget = budget
    try:
        yield dict(repair_id=VERSION, future_action_only=True)
    finally:
        if campaign.Production.budget is budget:
            campaign.Production.budget = previous


def install(startup_module, root, *, campaign=None, development=None, emit=None):
    expected_root = Path(root).resolve(strict=True)
    old = getattr(startup_module, _STATE, None)
    function=startup_module.active; seen=set(); installed=False
    while old and function is not None and id(function) not in seen:
        if function is old['wrapper']:
            installed=True;break
        seen.add(id(function));function=getattr(function,'__wrapped__',None)
    if installed:
        _require(old['root'] == expected_root, 'B5 installed root differs')
        return dict(status='ALREADY_INSTALLED', repair_id=VERSION)
    previous = startup_module.active

    @contextmanager
    @functools.wraps(previous)
    def wrapped(root, action, *args, **kwargs):
        if is_b5_action(action):
            _require(Path(root).resolve(strict=True) == expected_root,
                     'B5 future action root differs from installation')
        with active(root, action, campaign=campaign, development=development,
                    emit=emit if emit is not None else kwargs.get('emit')):
            with previous(root, action, *args, **kwargs) as result:
                yield result

    startup_module.active = wrapped
    setattr(startup_module, _STATE, dict(wrapper=wrapped, original=previous, root=expected_root))
    return dict(status='R1_B5_FUTURE_RETRY_BUDGET_INSTALLED', repair_id=VERSION,
                future_action_only=True, progress_files_changed=0, signals_sent=0,
                operational_budget_published=False)
