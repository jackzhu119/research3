"""Strict, retryable reads of existing operational records; never reset history.

The original reader intentionally returned a default for malformed JSON and
transient filesystem failures. That default must not turn an unreadable live
task/scheduler record into a new empty state. These records are required after
an R11 campaign has started. No writes or scientific imports occur here.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time

MAX_STATE_BYTES = 256 * 1024 * 1024
RETRY_DELAYS = (.04, .12, .3, .6, 1.)


class StateUnavailable(RuntimeError):
    """A required operational observation is unavailable; do not submit/reset."""

    def __init__(self, path, cause, attempts):
        self.path = str(Path(path))
        self.cause = type(cause).__name__
        self.attempts = attempts
        super().__init__('Required operational state is temporarily unreadable: ' + Path(path).name)


class StateIdentityError(ValueError):
    """A readable record belongs to a different campaign; never adopt it."""


def required_json(path, *, max_bytes=MAX_STATE_BYTES, attempts=6, sleep=time.sleep,
                  opener=None):
    path = Path(path)
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError('Positive state read attempts required')
    original = None
    for attempt in range(attempts):
        try:
            with (opener or Path.open)(path, 'rb') as handle:
                before = os.fstat(handle.fileno())
                size = before.st_size
                if size < 0 or size > max_bytes:
                    raise ValueError('Operational state exceeds bounded size')
                blocks = []
                count = 0
                while count <= size:
                    block = handle.read(min(1024 * 1024, size + 1 - count))
                    if not block:
                        break
                    blocks.append(block)
                    count += len(block)
                after = os.fstat(handle.fileno())
            if count != size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Operational state changed during observation')
            value = json.loads(b''.join(blocks))
            if not isinstance(value, dict):
                raise ValueError('Operational state must be a JSON object')
            return value
        except (OSError, ValueError, UnicodeError) as error:
            original = error
            if attempt + 1 < attempts:
                sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS)-1)])
    raise StateUnavailable(path, original, attempts) from None


def install_state_patch(env, *, attempts=6, sleep=time.sleep):
    """Patch only live process aliases, keeping original package bytes intact."""
    rt, common = env['runtime_module'], env['common']
    runtime = Path(env['runtime']).resolve()
    existing = getattr(rt._state, '_r11_management_runtime', None)
    if existing == str(runtime):
        return rt._state._r11_management_patch
    state = dict(reads=0, retries_exhausted=0, last_error=None,
                 required_task_state=True, existing_history_reset=False,
                 source_files_changed=False)

    def load_state(plan):
        path = Path(plan['output']) / 'runtime/tasks.json'
        try:
            value = required_json(path, attempts=attempts, sleep=sleep)
        except StateUnavailable as error:
            state['retries_exhausted'] += 1
            state['last_error'] = dict(error_type=error.cause, path=path.name,
                                       read_attempts=error.attempts)
            raise
        if value.get('recipe_sha256') != plan['recipe_sha256']:
            raise StateIdentityError('Operational task state belongs to another R11 recipe')
        if not isinstance(value.get('tasks'), dict):
            raise StateUnavailable(path, ValueError('Task map required'), 1)
        if any(not isinstance(key, str) or not isinstance(row, dict)
               for key, row in value['tasks'].items()):
            raise StateUnavailable(path, ValueError('Task history rows must be objects'), 1)
        tasks, identities = rt.plan_index(plan)
        for key, row in value['tasks'].items():
            if key not in tasks:
                raise StateIdentityError('Operational task history contains an unregistered R11 task')
            if row.get('phase', 'NEW') not in {'NEW','RUNNING','COMPLETE','PAUSED','RETRYING',
                    'QUARANTINED','BLOCKED_CONFIGURATION','BLOCKED_DEPENDENCY'}:
                raise StateUnavailable(path, ValueError('Task phase invalid'), 1)
            for field in ('attempt','consecutive_failures'):
                if field in row and (type(row[field]) is not int or row[field] < 0):
                    raise StateUnavailable(path, ValueError('Retained task counter invalid'), 1)
            for field in ('retry_at','started_at','finished_at'):
                if field in row and (type(row[field]) not in (int,float) or not math.isfinite(row[field])):
                    raise StateUnavailable(path, ValueError('Retained task clock invalid'), 1)
            if ('failure_history' in row and not isinstance(row['failure_history'],list)):
                raise StateUnavailable(path, ValueError('Retained task failure history invalid'), 1)
            if row.get('phase') == 'COMPLETE' and (
                    row.get('task_identity') != identities[key]
                    or any(not re.fullmatch('[0-9a-f]{64}',str(row.get(field,'')))
                           for field in ('result_sha256','completion_sha256'))):
                raise StateUnavailable(path, ValueError('Complete task identity cannot be adopted'), 1)
        state['reads'] += 1
        state['last_error'] = None
        return value

    load_state._r11_management_runtime = str(runtime)
    load_state._r11_management_patch = state
    rt._state = load_state
    # Scheduler._load must also distinguish an unavailable required record
    # from a brand-new campaign. Keep optional per-job outcome semantics.
    scheduler = __import__('paper3r11.platform.scheduler', fromlist=['_read'])
    previous = scheduler._read
    target = runtime / 'scheduler-state.json'

    def scheduler_read(path, default=None):
        if Path(path).resolve() == target:
            return required_json(path, attempts=attempts, sleep=sleep)
        return previous(path, default)

    scheduler_read._r11_management_runtime = str(runtime)
    scheduler._read = scheduler_read
    old_load = scheduler.FourJobScheduler._load

    def scheduler_load(instance):
        if Path(instance.runtime).resolve() != runtime:
            return old_load(instance)
        # Do not call Path.exists(): a transient negative metadata lookup
        # must never create four fresh slots over retained allocation intents.
        value = required_json(instance.state_path, attempts=attempts, sleep=sleep)
        if value.get('marker') != instance.marker:
            raise StateIdentityError('Scheduler state belongs to another R11 campaign')
        slots = value.get('slots')
        if not isinstance(slots, dict) or set(slots) != {'0', '1', '2', '3'}:
            raise StateUnavailable(instance.state_path, ValueError('Four retained slots required'), 1)
        def number(item):
            return type(item) in (int, float) and math.isfinite(item)
        def check_attempt(item, slot, source):
            if (type(item.get('sequence')) is not int or item['sequence'] < 1
                    or item.get('slot') != slot
                    or item.get('marker') != instance.marker
                    or item.get('name') != '%s-s%d-a%d' % (instance.marker, slot, item['sequence'])
                    or item.get('status') not in {'INTENT','ACKNOWLEDGED','UNCERTAIN','FINISHED','REJECTED'}
                    or not number(item.get('intent_at'))
                    or (item.get('job_id') is not None and
                        (not isinstance(item['job_id'], str) or not item['job_id'].isdigit()))):
                raise StateUnavailable(source, ValueError('Allocation attempt identity/schema invalid'), 1)
        if not number(value.get('created_at')):
            raise StateUnavailable(instance.state_path, ValueError('Scheduler creation clock invalid'), 1)
        for slot_name, row in slots.items():
            if not isinstance(row, dict) or not isinstance(row.get('attempts'), list) or not row['attempts']:
                raise StateUnavailable(instance.state_path, ValueError('Retained allocation history required'), 1)
            if any(not isinstance(item, dict) for item in row['attempts']):
                raise StateUnavailable(instance.state_path, ValueError('Allocation attempts must be objects'), 1)
            row.setdefault('worker_failures', 0)
            if (not number(row.get('next_submit_at'))
                    or type(row.get('rejections')) is not int or row['rejections'] < 0
                    or type(row.get('worker_failures')) is not int or row['worker_failures'] < 0):
                raise StateUnavailable(instance.state_path, ValueError('Retained slot counters invalid'), 1)
            for item in row['attempts']:
                check_attempt(item, int(slot_name), instance.state_path)
            if len({item['sequence'] for item in row['attempts']}) != len(row['attempts']):
                raise StateUnavailable(instance.state_path, ValueError('Duplicate attempt sequences'), 1)
        instance.state = value
        # Receipts can contain an acknowledged RPC newer than scheduler-state.
        # An unreadable listed receipt defers observation instead of silently
        # discarding that acknowledgement and risking a duplicate submission.
        for receipt_path in sorted(instance.receipts.glob('*.json')):
            receipt = required_json(receipt_path, attempts=attempts, sleep=sleep)
            if receipt.get('marker') != instance.marker or receipt.get('slot') not in (0, 1, 2, 3):
                continue
            if not isinstance(receipt.get('sequence'), int):
                raise StateUnavailable(receipt_path, ValueError('Receipt sequence required'), 1)
            check_attempt(receipt, receipt['slot'], receipt_path)
            attempts_list = instance.state['slots'][str(receipt['slot'])]['attempts']
            previous_attempt = next((item for item in attempts_list
                                     if item['sequence'] == receipt['sequence']), None)
            if previous_attempt is None:
                attempts_list.append(receipt)
            else:
                previous_attempt.update(receipt)
            attempts_list.sort(key=lambda item: item['sequence'])

    scheduler_load._r11_management_runtime = str(runtime)
    scheduler.FourJobScheduler._load = scheduler_load
    old_query = scheduler.FourJobScheduler._query
    def scheduler_query(instance):
        result = old_query(instance)
        if Path(instance.runtime).resolve() == runtime:
            # Native scheduler retains observation_error after a later good
            # query. Clear that stale marker only when both RPCs succeeded.
            instance.state.pop('observation_error', None)
            instance.state['last_successful_queue_observation_utc'] = common.utc()
        return result
    scheduler.FourJobScheduler._query = scheduler_query
    return state
