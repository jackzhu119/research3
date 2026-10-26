import base64, concurrent.futures, hashlib, json, pathlib, sys, urllib.parse, urllib.request
BASE=pathlib.Path(__file__).resolve().parent
from evidence_reader import GitHubRead,credential
TOKEN=credential()
GH=GitHubRead(TOKEN)
REFS=json.loads((BASE/'pinned_refs.json').read_text())

def fetch(path,ref=None,sha=None):
 ref=ref or REFS['paper3-results']; cache=BASE/'remote'/ref/path
 if cache.exists():
  b=cache.read_bytes()
  if not sha or hashlib.sha256(b).hexdigest()==sha:return b
 url='https://raw.githubusercontent.com/jackzhu119/research3/'+ref+'/'+path
 h={'User-Agent':'paper3-r10-evidence'}
 if TOKEN:h['Authorization']='Bearer '+TOKEN
 req=urllib.request.Request(url,headers=h)
 with urllib.request.urlopen(req,timeout=60) as f:b=f.read()
 if sha and hashlib.sha256(b).hexdigest()!=sha:raise ValueError('SHA256 mismatch: '+path)
 cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(b)
 return b

def getj(path,ref=None): return json.loads(fetch(path,ref))

def main():
 out={}
 for n in [8,9]:
  out[str(n)]={}
  for name in ['PAPER_DATA_LATEST.json','AUTO_SUMMARY_LATEST.json','LATEST.json']:
   path=f'results/paper3/round-{n:02d}/'+name
   try:v=getj(path)
   except Exception as e:v={'unavailable':type(e).__name__}
   out[str(n)][name]=v
  mp=f'monitoring/paper3-round{n}/three-jobs-status.json'
  out[str(n)]['monitor']=getj(mp,REFS[f'paper3-r{n}-progress'])
 (BASE/'latest_pointers_and_status.json').write_text(json.dumps(out,indent=2))
 for n,x in out.items():
  print(json.dumps({'round':n,'pointers':{k:v for k,v in x.items() if k!='monitor'},'monitor':{k:x['monitor'].get(k) for k in ['updated_utc','complete_tasks','total_tasks','status','families'] }},indent=2))
if __name__=='__main__':main()
