import collections, getpass, hashlib, json, pathlib, signal, subprocess, time

ROOT = pathlib.Path('/public/home/zhuziyuan/projects/paper3')
errors = []
def read(p):
    try:
        v = json.loads(p.read_text())
        return v if isinstance(v, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        errors.append(str(p.relative_to(ROOT)) + ':' + type(e).__name__)
        return {}

def brief(v):
    keys = ('status', 'stage', 'outer_update', 'total_updates', 'session', 'sessions_total',
            'budget', 'outer_updates_completed', 'inner_updates_actual',
            'last_observed_trajectory_update', 'observed_optimizer_calls_total')
    return json.dumps({k:v[k] for k in keys if k in v}, ensure_ascii=False)

def age(p):
    try: return str(round(max(0, time.time()-p.stat().st_mtime))) + '秒前'
    except OSError: return '未知'

class CheckTimeout(Exception): pass
def expire(*args): raise CheckTimeout('本轮读取超过60秒，跳过；不影响训练')
signal.signal(signal.SIGALRM, expire)
print('CHECK_STARTED', time.strftime('%Y-%m-%d %H:%M:%S %Z'), flush=True)
queue, queue_ok = [], False
try:
    q = subprocess.run(['squeue','--array','--noheader','--user',getpass.getuser(),
        '--format=%i|%k|%T|%j|%P|%M|%R|%N'], capture_output=True, text=True, timeout=15)
    queue_ok = q.returncode == 0
    queue = [[x.strip() for x in s.split('|',7)] for s in q.stdout.splitlines() if s.count('|') >= 7]
    print('当前账户队列：作业ID|轮次标记|状态|作业名|分区|已运行|原因|节点')
    print(q.stdout.strip() or (q.stderr.strip() if not queue_ok else '当前没有排队或运行作业'))
except (OSError, subprocess.TimeoutExpired) as e:
    print('队列查询失败：', type(e).__name__)

paths = {1:'training_campaign_v1', 2:'followup_20261023_v2'}
paths.update({n:'followup_20261024_r%d'%n for n in range(3,8)})
for n in range(1,8):
    print('\n========== 第%d轮 =========='%n, flush=True)
    errors.clear()
    signal.alarm(60)
    try:
        out = ROOT/'runs'/paths[n]
        if n == 1:
            p = out/'campaign_progress.json'
            v = read(p)
            print('编排记录：', v.get('status','尚无记录'), '更新于', age(p))
            print('已记录完成阶段：', dict(collections.Counter(a.get('kind','未知') for a in v.get('actions',[]))))
            active = v.get('active') or {}
            print('当前阶段记录：', json.dumps(active, ensure_ascii=False))
            expected = active.get('expected_receipt')
            if expected:
                folder = ROOT/pathlib.Path(expected).parent
                for name in ('progress.json','execution_progress.json'):
                    p = folder/name
                    z = read(p)
                    if z: print('阶段进度：', brief(z), age(p))
            for name in ('source_training_v2/source_training_progress.json', 'source_classical_v2/receipt.json',
                         'internal_validation_once/receipt.json'):
                z = read(ROOT/'runs'/name)
                if z: print(name, z.get('status'), '源神经完成数=',z.get('actual_neural_fits_complete','-'))
            print('原作业队列：', ['|'.join(r) for r in queue if r[0]=='770043'] or '未见770043；其他作业见上方总队列')
            continue
        plan = read(out/'plan.json')
        tasks = plan.get('tasks',[])
        if not tasks:
            print('未读到任务计划，不能判断本轮完成情况')
            continue
        state = read(out/'runtime/state.json')
        allocation = read(out/'allocation_progress.json')
        token = hashlib.sha256(json.dumps([plan.get('recipe_sha256'),str(out.resolve())],separators=(',',':')).encode()).hexdigest()[:16]
        marker = ('paper3-followup-' if n==2 else 'paper3r%d-'%n) + token
        ids = set()
        def collect_ids(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k in ('job_id','slurm_job_id') and isinstance(x,(str,int)): ids.add(str(x))
                    elif isinstance(x,(dict,list)): collect_ids(x)
            elif isinstance(v,list):
                for x in v: collect_ids(x)
        collect_ids(state)
        if allocation.get('job_id'): ids.add(str(allocation['job_id']))
        owned = [r for r in queue if r[1]==marker or r[3].startswith(marker) or r[0] in ids or r[0].split('_')[0] in ids]
        print('本轮实际队列：', '\n'.join('|'.join(r) for r in owned) or ('未匹配到；请对照上方总队列，不能据此判失败' if queue_ok else '查询失败，未知'))
        done, groups, recent = 0, {}, []
        for t in tasks:
            folder = out/t['key']
            complete = read(folder/'result.json').get('status') == 'COMPLETE'
            done += complete
            family = t.get('family',t.get('arm','其他'))
            counts = groups.setdefault(family,[0,0])
            counts[0] += complete
            counts[1] += 1
            if not complete:
                p = folder/'progress.json'
                z = read(p)
                if z: recent.append((p.stat().st_mtime,t['key'],z,p))
        print('本地完成：%d/%d（%.1f%%）；未完成或回执不可读：%d'%(done,len(tasks),100*done/len(tasks),len(tasks)-done))
        print('分类完成/总数：', '；'.join('%s %d/%d'%(k,*v) for k,v in groups.items()))
        print('控制器记录：', state.get('controller_status',state.get('status','无')))
        for slot,row in allocation.get('slots',{}).items():
            print('GPU槽',slot,row.get('task_key'),row.get('status'),brief(row.get('worker_progress') or {}))
        for _,key,z,p in sorted(recent,key=lambda x:x[0],reverse=True)[:6]:
            print('最近未完成任务记录：',key,brief(z),'更新于',age(p))
        print('日志目录：',out/'logs','；控制器日志：',out/'controller.log')
    except Exception as e:
        print('本轮检查未完整结束：',type(e).__name__,str(e))
    finally:
        signal.alarm(0)
        if errors: print('读取异常：',len(errors),errors[:3])
print('\nCHECK_DONE；任务进度记录可能滞后，运行/排队以本次队列为准。',flush=True)
