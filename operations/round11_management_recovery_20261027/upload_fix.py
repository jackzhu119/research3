"""Observe live R11 uploaders and resume missing channels without training control.

An occupied legacy upload lock is preserved. Its durable per-group receipts
and bounded recent log tail provide fresh operational observations while the
original process continues its potentially long archive cycle.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import threading
import time

from network import TransportStopped, error_code, make_bounded_client_class

REPAIR_ID = 'paper3-r11-management-upload-observer-v2'
_CLIENT_CLASSES = {}
_ERROR_NAMES = {'ValueError','RuntimeError','OSError','FileNotFoundError','PermissionError',
                'MemoryError','ValidationError','ApiError','URLError','HTTPError','TimeoutError',
                'ConnectionError','TransportRetryLater','TransportStopped','PaperRepairValidationError'}
_EVENTS = {'ARCHIVE_PART_UPLOADING','ARCHIVE_PART_VERIFIED','ARCHIVE_PART_RETRY',
           'ARCHIVE_PART_VERIFY_RETRY','ASSET_VERIFY_RETRY','RESULT_INDEX_PUBLISHED',
           'RESULT_INDEX_UNCHANGED','RESULT_INDEX_REBASE_RETRY','GIT_BLOB_FULL_GET_SHA256_VERIFIED',
           'GITHUB_NETWORK_RETRY','GITHUB_RETRY','R11_UPLOAD_RETRY','R11_PAPER_FIX_DIAGNOSTIC'}


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _stamp(path):
    try:
        value = Path(path).stat()
        return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    except OSError:
        return None


def _put(common, path, value):
    try: common.atomic(path, value); return True
    except Exception: return False


def _read(path, limit=64*1024*1024):
    with Path(path).open('rb') as handle:
        size = os.fstat(handle.fileno()).st_size
        if not 0 <= size <= limit: raise ValueError('UPLOAD_METADATA_SIZE_LIMIT')
        data = handle.read(size+1)
    if len(data) != size: raise ValueError('UPLOAD_METADATA_CHANGED_DURING_READ')
    value = json.loads(data)
    if not isinstance(value, dict): raise ValueError('UPLOAD_METADATA_OBJECT_REQUIRED')
    return value


def _time(value):
    try:
        if not isinstance(value, str): return None
        parsed = dt.datetime.fromisoformat(value.replace('Z','+00:00'))
        if parsed.tzinfo is None: return None
        return parsed.timestamp()
    except (ValueError, TypeError, OverflowError): return None


def _safe_event(record):
    if not isinstance(record, dict) or record.get('event') not in _EVENTS:
        return None
    timestamp = record.get('utc', record.get('updated_utc'))
    if _time(timestamp) is None: return None
    result = dict(event=record['event'],utc=timestamp)
    if re.fullmatch(r'part-[0-9]{5}\.bin', str(record.get('name',''))): result['part_name']=record['name']
    if type(record.get('bytes')) is int and 0 <= record['bytes'] <= 128*1024*1024: result['bytes']=record['bytes']
    if type(record.get('reused')) is bool: result['reused']=record['reused']
    if re.fullmatch('[0-9a-f]{40}', str(record.get('commit',''))): result['commit']=record['commit']
    if record.get('error_type') in _ERROR_NAMES: result['error_type']=record['error_type']
    return result


class LegacyObserver:
    def __init__(self, env, channel, work, *, log_paths=None):
        self.env,self.channel,self.work=env,channel,Path(work)
        runtime=env['runtime']; recipe=env['header']['recipe_sha256']
        self.channel_work=runtime/('upload-'+channel)
        paths=[runtime/(channel+'-upload.log')]
        if channel=='paper':
            paths.append(Path.home()/'paper3-r11-operational-repair/work'/recipe[:16]/'paper.log')
        paths.extend(Path(p) for p in (log_paths or ()))
        self.logs=list(dict.fromkeys(paths));self.log_stamps={};self.events={}
        self.group_stamp=None;self.group_count=None;self.group_errors=0

    def observe(self, now=None):
        now=time.time() if now is None else now
        group_path=self.channel_work/('groups.json' if self.channel=='full' else 'published-groups.json')
        stamp=_stamp(group_path)
        if stamp is not None and stamp!=self.group_stamp:
            try:
                value=_read(group_path)
                if value.get('recipe_sha256')!=self.env['header']['recipe_sha256']:
                    raise ValueError('UPLOAD_CACHE_RECIPE_DIFFERS')
                tasks=value.get('tasks')
                if not isinstance(tasks,dict): raise ValueError('UPLOAD_TASK_MAP_REQUIRED')
                def verified(item):
                    if not isinstance(item,dict): return False
                    proof=item if self.channel=='full' else item.get('receipt',{})
                    return isinstance(proof,dict) and bool(proof.get('remote_full_GET_SHA256_verified')
                        or proof.get('remote_complete_blob_GET_SHA256_verified'))
                self.group_count=sum(verified(item) for item in tasks.values())
                self.group_stamp=stamp
            except (OSError,ValueError,TypeError):
                # Preserve the last good observation, never turn an I/O gap
                # into zero progress or a fresh upload campaign.
                self.group_errors+=1
        for path in self.logs:
            stamp=_stamp(path)
            if stamp is None or self.log_stamps.get(path)==stamp: continue
            try:
                with path.open('rb') as handle:
                    size=os.fstat(handle.fileno()).st_size
                    handle.seek(max(0,size-256*1024));data=handle.read(256*1024)
                for line in data.splitlines():
                    try: record=json.loads(line)
                    except (ValueError,UnicodeError): continue
                    event=_safe_event(record)
                    if event is not None:
                        old=self.events.get(event['event'])
                        if old is None or _time(event['utc'])>_time(old['utc']):self.events[event['event']]=event
                self.log_stamps[path]=stamp
            except OSError: pass
        latest=max(self.events.values(),key=lambda item:_time(item['utc']),default=None)
        verified=self.events.get('ARCHIVE_PART_VERIFIED')
        published=self.events.get('RESULT_INDEX_PUBLISHED')
        blob=self.events.get('GIT_BLOB_FULL_GET_SHA256_VERIFIED')
        proofs=[value for value in (verified,published,blob) if value is not None]
        newest_proof=max(proofs,key=lambda item:_time(item['utc']),default=None)
        age=max(0.,now-_time(latest['utc'])) if latest else None
        result=dict(observer_heartbeat_not_a_new_remote_verification=True,
            current_transport_activity=('LEGACY_UPLOADER_ACTIVE' if age is not None and age<=1200
                                        else 'WAITING_FOR_LEGACY_EVENT_PROOF'),
            observed_verified_groups=self.group_count,upload_metadata_observation_errors=self.group_errors,
            verification_scope='Original uploader persisted whole-GET verification receipts; no new re-fetch claimed',
            latest_legacy_event=latest,legacy_event_age_seconds=age,
            last_verified_part_utc=verified['utc'] if verified else None,
            last_index_publish_utc=published['utc'] if published else None,
            last_blob_verified_utc=blob['utc'] if blob else None,
            newest_verified_event_utc=newest_proof['utc'] if newest_proof else None,
            legacy_status_is_not_current_transport_proof=True)
        try:
            status=_read(self.channel_work/'status.json',limit=4*1024*1024)
            result['legacy_cycle_status']=status.get('status')
            result['legacy_cycle_updated_utc']=status.get('updated_utc')
            if status.get('error_type') in _ERROR_NAMES:
                error_time=_time(status.get('updated_utc'))
                superseded=bool(newest_proof and error_time is not None
                    and _time(newest_proof['utc'])>error_time)
                result['last_cycle_error']=dict(error_type=status['error_type'],
                    updated_utc=status.get('updated_utc'),historical_not_a_new_failure=True,
                    superseded_by_verified_event=superseded)
                result['legacy_error_superseded_by_verified_event']=superseded
                result['legacy_error_observation']=('OLDER_ERROR_WITH_LATER_VERIFIED_SUCCESS' if superseded
                                                     else 'ERROR_NOT_YET_SUPERSEDED_BY_VERIFIED_EVENT')
        except (OSError,ValueError): pass
        return result


def _uploader(env):
    return importlib.import_module('paper3r11.platform.uploads')


def _token(u, explicit):
    try: return u.discover_token(explicit)
    except RuntimeError:
        for name in ('paper3-round9','paper3-round10','paper3-round11'):
            base=Path.home()/name
            paths=[base/'private/github.json']
            if base.is_dir():paths.extend(sorted(base.glob('*/private/github.json')))
            for path in paths:
                try: value=_read(path,limit=64*1024)
                except (OSError,ValueError): continue
                token=value.get('token',value.get('github_token'))
                if isinstance(token,str) and token.strip():return token.strip()
        raise RuntimeError('EXISTING_GITHUB_CREDENTIAL_UNAVAILABLE') from None


def make_client(env, channel, work, credentials=None, *, stop=None, progress=None):
    u=_uploader(env);base=u.FullVerifiedGitHub if channel=='full' else u.VerifiedGitHub
    if base not in _CLIENT_CLASSES:_CLIENT_CLASSES[base]=make_bounded_client_class(base)
    return _CLIENT_CLASSES[base](_token(u,credentials),u.emit,
        branch='paper3-r11-progress' if channel=='monitor' else 'paper3-results',
        cache_path=Path(work)/'git-blobs.json',stop=stop,progress=progress)


def _compact(result):
    allowed={'schema_version','channel','recipe_sha256','status','collection_status','complete_tasks',
             'total_tasks','commit','manifest','files','bytes','round','stage','updated_utc',
             'remote_full_GET_SHA256_verified','remote_complete_blob_GET_SHA256_verified',
             'exact_original_bytes','exact_original_bytes_scope','full_raw_results_uploaded'}
    return {key:value for key,value in result.items() if key in allowed and not isinstance(value,(dict,list))}


def run_channel(env, channel, work, *, stop=None, credentials=None, launch_token=None,
                once=False, gh_factory=None, interval=120, heartbeat_interval=5,
                legacy_log_paths=None):
    """Preserve a live old uploader; acquire its lock only after natural release."""
    if channel not in {'paper','full'}:raise ValueError('EXPECTED_PAPER_OR_FULL_CHANNEL')
    stop=stop or threading.Event();work=Path(work);work.mkdir(parents=True,exist_ok=True)
    common=env['common'];runtime=env['runtime'];cache=runtime/('upload-'+channel);cache.mkdir(parents=True,exist_ok=True)
    recipe=env['header']['recipe_sha256'];u=_uploader(env);guard=threading.RLock();own_upload=[False]
    state=dict(schema_version='paper3-r11-upload-observation-v2',repair_id=REPAIR_ID,
        channel=channel,role=channel,recipe_sha256=recipe,pid=os.getpid(),
        status='STARTING',training_controlled=False,scientific_files_changed=False,
        scientific_completion_claimed=False,remote_completion_claimed_by_heartbeat=False)
    observer=LegacyObserver(env,channel,work,log_paths=legacy_log_paths)
    heartbeat_stop=threading.Event()

    def publish():
        with guard:
            state['updated_utc']=_utc();state['observer_heartbeat_utc']=state['updated_utc']
            value=dict(state)
        _put(common,cache/'observer-status.json',value)
        _put(common,work/(channel+'-status.json'),value)
        if own_upload[0]:_put(common,cache/'status.json',value)

    def notice(**values):
        with guard:
            state.update(values);state['last_local_activity_utc']=_utc()
            if values.get('last_network_success'):state['last_network_response_utc']=_utc()

    def heartbeat():
        while not stop.is_set() and not heartbeat_stop.is_set():
            publish()
            if heartbeat_stop.wait(max(.05,heartbeat_interval)):break

    def ready(scope):
        payload=dict(status='READY',pid=os.getpid(),launch_token=launch_token,
            recipe_sha256=recipe,role=channel,lock_acquired=True,lock_scope=scope,
            original_upload_lock_owned=own_upload[0],existing_uploader_preserved=not own_upload[0],
            updated_utc=_utc(),training_controlled=False,remote_upload_complete=False)
        _put(common,work/(channel+'-ready.json'),payload)

    with (work/(channel+'-observer.lock')).open('a') as management_lock:
        try:fcntl.flock(management_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='EXISTING_REPAIR_HELPER_REUSED',channel=channel)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        lease=(runtime/(channel+'-upload.lock')).open('a')
        try:
            while not stop.is_set():
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    with guard:
                        state.update(observer.observe(),status='OBSERVING_EXISTING_UPLOADER',
                            lock_scope='repair_observer',original_upload_lock_owned=False,
                            existing_uploader_preserved=True)
                    ready('repair_observer');publish()
                    if once:return dict(state)
                    if stop.wait(max(.05,heartbeat_interval)):break
                    continue
                own_upload[0]=True
                with guard:state.update(status='UPLOAD_CHANNEL_LOCK_ACQUIRED',lock_scope='original_upload',
                    original_upload_lock_owned=True,existing_uploader_preserved=False)
                ready('original_upload')
                _put(common,cache/'process.json',dict(pid=os.getpid(),channel=channel,
                    started_utc=_utc(),repair_id=REPAIR_ID,launch_token=launch_token))
                if channel=='paper':
                    import paper_fix
                    paper_fix.install_patch(u)
                # Observe only this process's durable upload-group checkpoint
                # writes; no scientific state or other process is patched.
                previous_atomic=u.atomic
                def checkpoint_atomic(path,value):
                    result=previous_atomic(path,value)
                    path=Path(path)
                    if path.parent==cache and path.name in {'groups.json','published-groups.json'} and isinstance(value,dict):
                        notice(committed_uploaded_groups=len(value.get('tasks',{})),
                               complete_tasks=len(value.get('tasks',{})),
                               total_tasks=env['header'].get('total_tasks'),
                               coverage_origin='Durable per-group upload receipts, not heartbeat counts',
                               last_verified_group_checkpoint_utc=_utc())
                    return result
                u.atomic=checkpoint_atomic
                try:
                    while not stop.is_set():
                        with guard:
                            state.update(status='RUNNING_UPLOAD_CYCLE',current_transport_activity='OWNED_UPLOADER_ACTIVE',
                                cycle_started_utc=_utc(),observer_heartbeat_not_a_new_remote_verification=True)
                        publish()
                        try:
                            if gh_factory:
                                gh=gh_factory(_token(u,credentials),u.emit,
                                    branch='paper3-results',cache_path=cache/'git-blobs.json',stop=stop,progress=notice)
                            else:gh=make_client(env,channel,cache,credentials,stop=stop,progress=notice)
                            cycle=u.paper_cycle if channel=='paper' else u.full_cycle
                            result=cycle(gh,env['path'],cache)
                            with guard:state.update(_compact(result),last_cycle_completed_utc=_utc(),
                                last_cycle_error=None,status=result.get('status','CYCLE_FINISHED'))
                        except TransportStopped:
                            break
                        except Exception as error:
                            kind=type(error).__name__
                            with guard:
                                state.update(status='RETRYING',last_cycle_error=dict(
                                    error_type=kind if kind in _ERROR_NAMES else 'UploadOperationError',
                                    fixed_code=error_code(error),http_status=getattr(error,'status',None),
                                    updated_utc=_utc(),historical_not_a_new_failure=False))
                        publish()
                        if once:return dict(state)
                        if state.get('status') in {'COMPLETE','PAPER_DATA_FILES_VERIFIED'}:return dict(state)
                        if stop.wait(max(.05,interval)):break
                finally:u.atomic=previous_atomic
                break
            with guard:state.update(status='STOPPED',management_stop_only=True)
            publish();return dict(state)
        finally:
            # A helper finishing does not clear another process's upload state.
            # Use a private Event only for the heartbeat thread, rather than
            # signalling a worker or the externally supplied stop Event.
            heartbeat_stop.set();thread.join(timeout=2)
            lease.close()
