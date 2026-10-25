"""Asynchronous automatic existing-result summaries for Paper3 rounds 1/5/6.

This is a browsable result-summary transfer, not a full prediction/model backup.
No experiment code, aggregate builder, or training process is imported/called.
"""
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import threading
import time

REPOSITORY = 'jackzhu119/research3'
BRANCH = 'paper3-results'
DEFAULT_LIMIT = 16 * 1024 * 1024
EXTENSION_ARMS = ('BASE4_PLUS_YANG', 'BASE4_PLUS_DREYER')
_PRIVATE_KEYS = {
    'token', 'access_token', 'github_token', 'authorization', 'password', 'secret', 'pat',
    'api_key', 'private_key', 'participants', 'participant', 'participant_id', 'participant_ids',
    'participant_deltas_ba', 'participant_seed_counts', 'participant_results', 'per_participant',
    'sessions', 'session_id', 'session_ids', 'session_results', 'per_session', 'expected_sessions',
    'event_uids', 'event_uid', 'trial_uids', 'trial_uid', 'trial_indices', 'trials', 'labels',
    'predictions', 'probabilities', 'log_probabilities', 'log_probs', 'f0_predictions',
    'f0_probabilities', 'f0_log_probabilities', 'f0_log_probs', 'signals', 'raw_eeg', 'samples'}


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False)+'\n').encode()


def _emit(event, **fields):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                      **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def _read(path, default=None):
    try:
        with Path(path).open('rb') as handle:
            raw = handle.read(DEFAULT_LIMIT+1)
        if len(raw)>DEFAULT_LIMIT:
            return default
        return json.loads(raw)
    except (OSError, ValueError):
        return default


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('wb') as handle:
        handle.write(_bytes(value))
    os.replace(temporary,path)


def _sanitize(value):
    if isinstance(value, dict):
        return {str(key): _sanitize(child) for key,child in value.items()
                if str(key).lower() not in _PRIVATE_KEYS}
    if isinstance(value, list):
        return [_sanitize(child) for child in value]
    if isinstance(value, tuple):
        return [_sanitize(child) for child in value]
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def _csv_summary(data):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    fields = [key for key in (reader.fieldnames or []) if key.lower() not in _PRIVATE_KEYS]
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output,fieldnames=fields,lineterminator='\n')
    writer.writeheader()
    for row in reader:
        result = {}
        for key in fields:
            value = row.get(key,'') or ''
            if value[:1] in ('{','['):
                try:
                    value = json.dumps(_sanitize(json.loads(value)),ensure_ascii=False,sort_keys=True,allow_nan=False)
                except (ValueError,TypeError):
                    pass
            result[key] = value
        writer.writerow(result)
    return output.getvalue().encode()


def _safe(root,path):
    try:
        path = Path(path)
        if '..' in path.parts:
            return None
        if not path.is_absolute():
            path=root/path
        resolved=path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            return None
        if any(part in {'private','.git','credentials','__pycache__'} for part in resolved.parts):
            return None
        if path.name in {'github.json','credentials.json','.env','token','token.txt'}:
            return None
        return path
    except (OSError,ValueError,TypeError):
        return None


def _source_paths(root,round_number,progress):
    paths=[]
    if round_number==1:
        tasks=progress.get('tasks',[])
        if isinstance(tasks,dict):tasks=list(tasks.values())
        for task in tasks if isinstance(tasks,list) else []:
            if isinstance(task,dict) and task.get('expected_receipt'):
                paths.append(root/task['expected_receipt'])
        patterns = ('runs/meta/*/receipt.json',
            'runs/training_campaign_v1/meta_selections/*.json',
            'runs/training_campaign_v1/selections/*.json',
            'runs/training_campaign_v1/profiles/*/receipt.json',
            'runs/training_campaign_v1/development/*/receipt.json',
            'runs/internal_validation_once/reports/*.csv')
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))
        paths.extend(root/name for name in (
            'runs/training_campaign_v1/strongest_control.json',
            'runs/training_campaign_v1/pre_internal_freeze.json',
            'runs/internal_validation_once/receipt.json',
            'runs/internal_validation_once/report.json'))
        for arm in EXTENSION_ARMS:
            paths.extend(root/'runs/source_extension_downstream'/arm/name for name in (
                'pre_internal_development_freeze.json','catalog.json'))
        return paths,None
    output=root/('runs/followup_20261024_r%d'%round_number)
    plan=_read(output/'plan.json',{})
    paths.extend((output/'plan.json',output/'reports/report.json',output/'report.json'))
    paths.extend(sorted((output/'reports').glob('*.csv')))
    paths.extend(sorted(output.glob('*.csv')))
    for task in plan.get('tasks',[]) if isinstance(plan,dict) else []:
        if isinstance(task,dict) and isinstance(task.get('key'),str):
            paths.append(output/task['key']/'result.json')
    return paths,plan


def collect_summary(root,round_number,operational,progress_snapshot=None,*,max_file_bytes=DEFAULT_LIMIT):
    if round_number not in (1,5,6):
        raise ValueError('Automatic summaries are limited to rounds 1, 5 and 6')
    root=Path(root).absolute()
    operational=Path(operational)
    progress=_read(operational/'progress.json',{})
    if not isinstance(progress,dict):progress={}
    # A supplied observation is only a fallback declaration. Heartbeats/timestamps
    # and live counters are not part of the summary content identity.
    if not progress and isinstance(progress_snapshot,dict):
        progress=progress_snapshot
    paths,plan=_source_paths(root,round_number,progress)
    files,records,seen={},[],set()
    complete_results=0
    for original in paths:
        path=_safe(root,original)
        if path is None:
            continue
        relative=path.relative_to(root).as_posix()
        if relative in seen:continue
        seen.add(relative)
        try:
            with path.open('rb') as handle:
                before=os.fstat(handle.fileno())
                if before.st_size>max_file_bytes:
                    records.append({'path':relative,'bytes':before.st_size,'status':'LOCAL_ONLY_OVER_SIZE_LIMIT',
                                    'local_file_retained':True})
                    continue
                data=handle.read(max_file_bytes+1)
                after=os.fstat(handle.fileno())
            if len(data)>max_file_bytes:
                records.append({'path':relative,'status':'LOCAL_ONLY_OVER_SIZE_LIMIT','local_file_retained':True})
                continue
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                records.append({'path':relative,'status':'CHANGED_DURING_READ_RETRY_LATER','local_file_retained':True})
                continue
            if path.suffix.lower()=='.csv':
                captured=_csv_summary(data)
            else:
                value=json.loads(data)
                if path.name=='result.json' and isinstance(value,dict) and value.get('status')=='COMPLETE':
                    complete_results+=1
                captured=_bytes(_sanitize(value))
            if len(captured)>max_file_bytes:
                records.append({'path':relative,'bytes':len(data),
                    'status':'LOCAL_ONLY_SANITIZED_SUMMARY_OVER_SIZE_LIMIT','local_file_retained':True})
                continue
            files['source/'+relative]=captured
            records.append({'path':relative,'bytes':len(data),'source_sha256':hashlib.sha256(data).hexdigest(),
                            'summary_sha256':hashlib.sha256(captured).hexdigest(),'status':'CAPTURED_SANITIZED_SUMMARY'})
        except FileNotFoundError:
            continue
        except (OSError,ValueError,UnicodeError,TypeError,csv.Error) as error:
            records.append({'path':relative,'status':'READ_RETRY_LATER','error_type':type(error).__name__,
                            'local_file_retained':True})
    manifest={'schema_version':'paper3-automatic-existing-result-summary-v1','round':round_number,
        'source_files':sorted(records,key=lambda row:row['path']),
        'report_capture':'Existing files captured independently, sanitized and never recomputed; report status is preserved.',
        'report_recomputed':False,'trial_prediction_directories_scanned':False,
        'full_raw_results_uploaded':False,'raw_predictions_or_checkpoints_included':False,
        'scope':'Browsable existing report/CSV/receipt summary; raw trial predictions, checkpoints and over-limit files remain local.'}
    summary={'round':round_number,'kind':'AUTOMATIC_EXISTING_RESULT_SUMMARY',
        'recipe_sha256':plan.get('recipe_sha256') if isinstance(plan,dict) else progress.get('recipe_sha256'),
        'source_complete_result_receipts':complete_results if round_number in (5,6) else None,
        'source_summary_files':len(files),'report_recomputed':False,
        'raw_predictions_or_checkpoints_included':False}
    files['SUMMARY.json']=_bytes(summary)
    files['manifest.json']=_bytes(manifest)
    identity=[{'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
              for path,data in sorted(files.items())]
    content_hash=hashlib.sha256(_bytes(identity)).hexdigest()
    return {'round':round_number,'files':files,'manifest':manifest,'content_hash':content_hash,
            'source_complete_tasks':complete_results if round_number in (5,6) else None,
            'source_summary_files':len(files)-2}


def publish_summary(bundle,round_number,client,*,emit=_emit):
    if bundle.get('round')!=round_number:
        raise ValueError('Summary round mismatch')
    prefix='results/paper3/round-%02d'%round_number
    version=prefix+'/automatic-summary/'+bundle['content_hash']
    files={version+'/'+path:data for path,data in bundle['files'].items()}
    pointer={'schema_version':'paper3-automatic-summary-pointer-v1','round':round_number,
        'status':'AUTOMATIC_EXISTING_RESULT_SUMMARY_UPLOADED','content_hash':bundle['content_hash'],
        'summary_directory':version,'manifest':version+'/manifest.json','summary':version+'/SUMMARY.json',
        'source_summary_files':bundle.get('source_summary_files'),
        'full_raw_results_uploaded':False,'raw_predictions_or_checkpoints_included':False,
        'report_recomputed':False,
        'scope':'Existing result summary only. This is not the full-output LATEST.json archive.'}
    files[prefix+'/AUTO_SUMMARY_LATEST.json']=_bytes(pointer)
    files[version+'/README_ZH.md']=(
        '# 第%d轮自动结果摘要\n\n'
        '此处为已有结果的可浏览摘要，在独立后台线程同步，文件内容未变则不重复提交。\n\n'
        '- [摘要概览](SUMMARY.json)\n- [来源与校验清单](manifest.json)\n- [现成报告、CSV和回执](source/)\n\n'
        '报告状态按原文件保留，不现场重算统计。逐试次数据、模型checkpoint和超出摘要单文件大小限制的文件仍在学校本地。'
        '本入口不能代替完整结果归档的`LATEST.json`。\n') % round_number
    files[version+'/README_ZH.md']=files[version+'/README_ZH.md'].encode()
    commit=client.git_publish(files,'Update Paper3 round %d automatic result summary %s'%(round_number,bundle['content_hash'][:16]))
    emit('AUTO_SUMMARY_SYNCED',round=round_number,content_hash=bundle['content_hash'],commit=commit,
         full_raw_results_uploaded=False,training_controlled=False)
    return {'round':round_number,'content_hash':bundle['content_hash'],'commit':commit,'pointer':pointer}


def _credentials(explicit=None):
    for name in ('PAPER3_GITHUB_TOKEN','GITHUB_TOKEN','GH_TOKEN'):
        value=os.environ.get(name,'').strip()
        if value:return value
    candidates=[]
    for value in (explicit,os.environ.get('PAPER3_GITHUB_CREDENTIALS')):
        if value:candidates.append(Path(value).expanduser())
    candidates.append(Path.home()/'.config/paper3/github.json')
    for name in ('paper3-followup','paper3-round3','paper3-round4','paper3-round5','paper3-round6',
                 'paper3-round7','paper3-round4-hpcgpu','paper3-round5-hpcgpu','paper3-round6-hpcgpu'):
        base=Path.home()/name
        candidates.extend((base/'private/github.json',*sorted(base.glob('*/private/github.json'))))
    for path in candidates:
        value=_read(path,{})
        token=value.get('token') if isinstance(value,dict) else None
        if isinstance(token,str) and token.strip():return token.strip()
    raise ValueError('Existing GitHub credential not available')


def _github_class():
    path=Path(__file__).with_name('summary_transport.py')
    spec=importlib.util.spec_from_file_location('paper3_automatic_summary_transport',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GitHub


class SummaryDispatcher:
    def __init__(self,root,round_number,operational,credentials=None,*,client_factory=None,
                 emit=_emit,collector=collect_summary,publisher=publish_summary):
        self.root=Path(root).absolute();self.round_number=round_number
        self.operational=Path(operational);self.credentials=credentials
        self.client_factory=client_factory;self.emit=emit;self.collector=collector;self.publisher=publisher
        self.thread=None;self._mutex=threading.Lock();self._client=None
        self.state_path=self.operational/'auto_summary_state.json'
    def kick(self,snapshot=None):
        # This foreground method does no filesystem/network work and never joins.
        with self._mutex:
            if self.thread is not None and self.thread.is_alive():return False
            self.thread=threading.Thread(target=self._run,args=(snapshot,),daemon=True,
                                         name='paper3-auto-summary-r%d'%self.round_number)
            self.thread.start()
            return True
    def _run(self,snapshot):
        try:
            self.operational.mkdir(parents=True,exist_ok=True)
            with (self.operational/'auto_summary.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return
                bundle=self.collector(self.root,self.round_number,self.operational,snapshot)
                previous=_read(self.state_path,{})
                if previous.get('content_hash')==bundle['content_hash']:
                    return
                if self._client is None:
                    token=_credentials(self.credentials)
                    self._client=self.client_factory(token) if self.client_factory else _github_class()(
                        token,self.emit,repo=REPOSITORY,branch=BRANCH,
                        cache_path=self.operational/'auto_summary_blob_cache.json')
                result=self.publisher(bundle,self.round_number,self._client,emit=self.emit)
                _atomic(self.state_path,{'content_hash':bundle['content_hash'],'commit':result['commit'],
                    'round':self.round_number,'summary_only':True})
        except Exception as error:
            self._client=None
            self.emit('AUTO_SUMMARY_RETRY',round=self.round_number,error_type=type(error).__name__,
                      training_controlled=False,full_raw_results_uploaded=False)
