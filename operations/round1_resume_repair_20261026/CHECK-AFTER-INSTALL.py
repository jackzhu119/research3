"""Read current R1 attempts and numerical progress; no downloads or Torch."""
import datetime
import hashlib
import json
from pathlib import Path
import time

root = Path('/public/home/zhuziyuan/projects/paper3')
runtime = root/'runs/training_campaign_parallel_v1'

def read(path):
    try:
        with path.open('rb') as f:
            data = f.read(1024*1024+1)
        return json.loads(data) if len(data)<=1024*1024 else {}
    except (OSError, ValueError):
        return {}

def tail(path):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell()-131072))
            return f.read().decode('utf8', 'replace').splitlines()
    except OSError:
        return []

def age(path):
    return round(time.time()-path.stat().st_mtime, 1) if path.is_file() else None

def show(label, value):
    print(label, json.dumps(value, ensure_ascii=False, separators=(',', ':')), flush=True)

fields = ('event', 'utc', 'phase', 'status', 'error_type', 'reason',
          'elapsed_seconds', 'outer_updates_completed', 'outer_update',
          'outer_attempts_dispatched', 'confirmed_outer_updates_observed',
          'validation_checkpoints', 'updated_utc')
show('CHECK_UTC', datetime.datetime.now(datetime.timezone.utc).isoformat())
for slot in range(3):
    row = read(runtime/'worker-status'/('slot-%d.json'%slot))
    show('SLOT', {'slot': slot, **{k:row.get(k) for k in
        ('job_id','status','action_key','child_pid','claim_token')}})
    key, token = row.get('action_key'), row.get('claim_token')
    if not key or not token:
        continue
    attempt = runtime/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/str(token)
    claim = read(attempt/'action.json').get('claim', {})
    started = claim.get('claimed_utc', 0)
    show('STARTUP', {k:v for k,v in read(attempt/'startup-progress.json').items() if k in fields})
    receipt = row.get('expected_receipt')
    if receipt:
        p = (root/receipt).parent/'progress.json'
        value = read(p)
        show('NUMERIC', {'age_seconds':age(p),
            'older_than_current_attempt':bool(p.is_file() and isinstance(started,(int,float)) and p.stat().st_mtime<started),
            **{k:v for k,v in value.items() if k in fields}})
    events = []
    for line in tail(attempt/'kernel.log'):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value,dict) and (str(value.get('event','')).startswith(('R1_OBSERVED_', 'R1_CROSS_NODE_'))
            or value.get('event') in ('R1_INPUT_VERIFY_FINISHED','R1_INPUT_VERIFY_FAILED','R1_ACTION_ERROR','R1_ACTION_RETURNED')
            or value.get('stage') in ('FAILED','COMPLETE')):
            events.append({k:v for k,v in value.items() if k in fields or k=='stage'})
    show('CURRENT_ATTEMPT_EVENTS', events[-8:])
    show('LOG_END', tail(attempt/'kernel.log')[-2:])
print('CHECK_DONE', flush=True)
