import json,urllib.request,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'evidence';ref='a16d06760ae77699a809c9ba36d002d684560c82';base='https://raw.githubusercontent.com/jackzhu119/research3/'+ref+'/'
def get(path):
 with urllib.request.urlopen(base+path,timeout=30) as r:return r.read()
m=json.loads((p/'r10-paper-manifest.json').read_text());tasks=m['tasks']
keys=[next(k for k in tasks if 'holdout-profile-seed20261018' in k)]+[k for k in tasks if k.startswith('feature_reference')]
proof=[]
for key in keys:
 path=tasks[key]['receipt']['manifest'];manifest=json.loads(get(path))
 for f in manifest['files']:
  if f['source_path'].endswith(('/result.json','/reference.json')):
   source=json.loads(get(f['manifest']));data=b''.join(get(x['path']) for x in source['parts']);assert hashlib.sha256(data).hexdigest()==source['source_sha256'];name=f['source_path'].split('/')[-1];target=p/'r10-task-metadata'/key/name;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(data);proof.append(dict(task=key,path=f['source_path'],source_manifest=base+f['manifest'],sha256=source['source_sha256']))
(p/'r10-task-metadata-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
for row in proof:print(json.dumps(row))
