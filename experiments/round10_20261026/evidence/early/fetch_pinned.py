"""Fetch exact evidence bytes from one immutable GitHub commit; never writes remote."""
from pathlib import Path
import base64, concurrent.futures, datetime, hashlib, json, sys, urllib.parse
sys.path.insert(0,'/workspace/paper3-fast-paper-upload')
from paper3_result_transport import GitHub
BASE=Path(__file__).resolve().parent
SECRET=Path('/workspace/paper3-v2-design/private/github.json')
def client():
    return GitHub(json.loads(SECRET.read_text())['token'],lambda *a,**k:None,branch='paper3-results',timeout=25)
def read_repo(path,ref,expected=None):
    gh=client(); row=gh.api(gh.prefix+'/contents/'+urllib.parse.quote(path,safe='/')+'?ref='+urllib.parse.quote(ref,safe=''),allow404=True)
    if row is None:return None,None
    if row.get('encoding')=='base64' and row.get('content'):
        raw=base64.b64decode(row['content'])
    else:
        with gh._request('https://api.github.com/'+gh.prefix+'/git/blobs/'+row['sha'],accept='application/vnd.github.raw+json') as response:raw=response.read()
    assert len(raw)==row['size'],(path,'size')
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha'],(path,'git_sha')
    sha=hashlib.sha256(raw).hexdigest()
    if expected:assert sha==expected,(path,'source_sha256')
    record={'path':path,'ref':ref,'git_blob_sha':row['sha'],'bytes':len(raw),'sha256':sha}
    return raw,record

def main():
    gh=client();sha=gh.api(gh.prefix+'/git/ref/heads/paper3-results')['object']['sha'];
    metadata={'head':sha,'queried_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':'paper3-results'}
    (BASE/'PINNED_HEAD.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def one(r):
        entries=gh.api(gh.prefix+'/contents/results/paper3/round-%02d?ref='%r+sha)
        pointers={}
        for name in ('LATEST.json','PAPER_DATA_LATEST.json','REPORT_JSON_LATEST.json','AUTO_SUMMARY_LATEST.json'):
            path='results/paper3/round-%02d/'%r+name
            raw,row=read_repo(path,sha)
            if raw is not None:
                p=BASE/('round-%02d'%r)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
                pointers[name]={'value':json.loads(raw),'provenance':row}
        return {'round':r,'directory':[{'name':x['name'],'size':x.get('size'),'type':x['type']} for x in entries],'pointers':pointers}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:values=list(pool.map(one,range(2,8)))
    (BASE/'POINTER_AUDIT.json').write_text(json.dumps({'pin':metadata,'rounds':values},indent=2)+'\n')
    print(json.dumps({'pin':metadata,'rounds':[{'round':v['round'],'names':[x['name'] for x in v['directory']],'pointers':{k:d['value'] for k,d in v['pointers'].items()}} for v in values]}))
if __name__=='__main__':main()
