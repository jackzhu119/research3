import sys,json,base64,hashlib,concurrent.futures
from pathlib import Path
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE));(BASE/'archive-manifests').mkdir(parents=True,exist_ok=True)
from evidence_reader import GitHubRead,credential
from restore_archived_gp import restore
api=GitHubRead(credential());ref=json.loads((BASE/'final_recheck_refs.json').read_text())['paper3-results']
dirs=api.api(api.prefix+'/contents/results/paper3/round-09/archive-groups?ref='+ref)
old=json.loads((BASE/'r9_available_archive_groups.json').read_text());known={x['manifest_path'] for x in old}
new=[x for x in dirs if x['path']+'/manifest.json' not in known]
def get(item):
 path=item['path']+'/manifest.json';v=api.api(api.prefix+'/contents/'+path+'?ref='+ref);b=base64.b64decode(v['content']);assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==v['sha'];m=json.loads(b);m.update(manifest_path=path,manifest_ref=ref,manifest_bytes_sha256=hashlib.sha256(b).hexdigest(),manifest_git_blob_sha=v['sha']);(BASE/'archive-manifests'/item['name']).write_bytes(b);return m
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:ms=list(pool.map(get,new))
selected=[m for m in ms if m['key'].startswith('core-') and any('-'+x+'-' in m['key'] for x in ['rbf_map','rbf_pp','rbf_anchored','rbf_global_mixture'])]
print(json.dumps({'fresh_RBF_tasks':len(selected),'bytes':sum(x['archive_bytes'] for x in selected)}),flush=True)
(BASE/'R9_FRESH_RBF_MANIFESTS.json').write_text(json.dumps([{k:v for k,v in m.items() if k!='files'} for m in selected],indent=2)+'\n')
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for m,r in zip(selected,pool.map(restore,selected)):
  r.update(manifest_ref=ref,manifest_bytes_sha256=m['manifest_bytes_sha256'],manifest_git_blob_sha=m['manifest_git_blob_sha']);rows.append(r);print(json.dumps({'task':r['task'],'files':len(r['files']),'archive_bytes':r['archive_bytes_verified']}),flush=True)
(BASE/'R9_FRESH_RBF_RESTORE_AUDIT.json').write_text(json.dumps(rows,indent=2)+'\n')
print('FRESH_RBF_RESTORE_COMPLETE',len(rows),flush=True)
