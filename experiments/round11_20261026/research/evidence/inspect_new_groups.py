import concurrent.futures,json,sys,hashlib
from pathlib import Path
from fetch_latest import ROOT,raw,save_json
head=json.loads((ROOT/'PROVENANCE.json').read_text())['pins']['results']
old=Path('/workspace/paper3-round10/evidence/late');known={x['manifest_path'] for x in json.loads((old/'r9_available_archive_groups.json').read_text())+json.loads((old/'R9_FRESH_RBF_MANIFESTS.json').read_text())}
new=[x for x in json.loads((ROOT/'R9_CURRENT_GROUP_DIRS.json').read_text()) if x['path']+'/manifest.json' not in known]
(ROOT/'new-group-manifests').mkdir(exist_ok=True)
def get(item):
    b,p=raw(item['path']+'/manifest.json',head);(ROOT/'new-group-manifests'/item['name']).write_bytes(b);v=json.loads(b);v.update(manifest_path=p['path'],manifest_ref=head,manifest_sha256=p['sha256']);return v
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:ms=list(pool.map(get,new))
save_json('R9_NEW_GROUP_MANIFESTS.json',[{k:v for k,v in m.items() if k!='files'} for m in ms])
print(json.dumps({'new_manifests':len(ms),'tasks':[{'key':m['key'],'archive_bytes':m['archive_bytes'],'source_files':len(m['files'])} for m in ms]},indent=2))
