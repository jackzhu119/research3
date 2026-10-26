"""Read-only pinned GitHub research-evidence snapshot; never runs training."""
import base64,concurrent.futures,datetime,hashlib,json,os,requests
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TOKEN=json.loads(Path(os.environ.get('PAPER3_GITHUB_CREDENTIALS','/workspace/paper3-v2-design/private/github.json')).read_text())['token']
PREFIX='https://api.github.com/repos/jackzhu119/research3'
HEADERS={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json'}
def get(endpoint,params=None,missing=False):
    r=requests.get(PREFIX+endpoint,headers=HEADERS,params=params,timeout=(8,40))
    if missing and r.status_code==404:return None
    r.raise_for_status();return r.json()
def raw(path,head):
    d=get('/contents/'+path,{'ref':head})
    if d.get('encoding')=='base64':b=base64.b64decode(d['content'])
    else:b=base64.b64decode(get('/git/blobs/'+d['sha'])['content'])
    return b,{'path':path,'git_blob':d['sha'],'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'ref':head}
def save_json(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
    branches={'results':'paper3-results','r1':'paper3-r1-progress','r9':'paper3-r9-progress','r10':'paper3-r10-progress'}
    pins={k:get('/git/ref/heads/'+v)['object']['sha'] for k,v in branches.items()}
    sources=[]
    def capture(path,head,target):
        b,meta=raw(path,head);(ROOT/target).parent.mkdir(exist_ok=True,parents=True);(ROOT/target).write_bytes(b);sources.append(dict(meta,local=target));return json.loads(b)
    r1=capture('monitoring/paper3-round1/status.json',pins['r1'],'R1_STATUS.json')
    r9=capture('monitoring/paper3-round9/three-jobs-status.json',pins['r9'],'R9_STATUS.json')
    r10=capture('monitoring/paper3-round10/status.json',pins['r10'],'R10_STATUS.json')
    pointer=capture('results/paper3/round-09/PAPER_DATA_LATEST.json',pins['results'],'R9_PAPER_POINTER.json')
    manifest=capture(pointer['manifest'],pins['results'],'R9_PAPER_MANIFEST.json')
    index=capture('results/paper3/round-10/INDEX.json',pins['results'],'R10_INDEX.json')
    wanted=['plan.json','reports/report.json','reports/full-report.json','reports/primary_comparisons-0000.csv','reports/evaluation_summaries-0000.csv']
    files=[f for f in manifest['files'] if f['source_path'] in wanted]
    def restore(f):
        chunks=[];checks=[]
        for part in sorted(f['parts'],key=lambda x:x['byte_offset']):
            text,info=raw(part['path'],pins['results']);b=base64.b64decode(text) if part.get('encoding')=='base64' else text
            if len(b)!=part['bytes'] or hashlib.sha256(b).hexdigest()!=part['sha256']:raise ValueError('PartSHA differs '+f['source_path'])
            chunks.append(b);checks.append(info)
        b=b''.join(chunks)
        if len(b)!=f['bytes'] or hashlib.sha256(b).hexdigest()!=f['sha256']:raise ValueError('FileSHA differs '+f['source_path'])
        local=ROOT/'r9'/f['source_path'];local.parent.mkdir(exist_ok=True,parents=True);local.write_bytes(b)
        return {'source_path':f['source_path'],'bytes':len(b),'sha256':f['sha256'],'local':str(local.relative_to(ROOT)),'parts':checks,'verified':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:restored=list(pool.map(restore,files))
    save_json('PROVENANCE.json',{'schema_version':'paper3-r11-evidence-read-v1','queried_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository':'jackzhu119/research3','branches':branches,'pins':pins,'sources':sources,'restored_files':restored,'jobs_changed':0,'gpu_used':False})
    print(json.dumps({'pins':pins,'r1_timestamp':r1.get('updated_utc'),'r9_keys':list(r9),'r10_keys':list(r10),'r9_pointer':pointer,'restored_files':[{k:v for k,v in x.items() if k!='parts'} for x in restored]},ensure_ascii=False))
if __name__=='__main__':main()
