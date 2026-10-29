from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures
ROOT=Path('/workspace/paper3-brain-sciences-20261029');E=ROOT/'evidence';REF='a16d06760ae77699a809c9ba36d002d684560c82';RAW='https://raw.githubusercontent.com/jackzhu119/research3/'+REF+'/'
BASE='results/paper3/round-12/studies/b04a1db8d593f259/paper/'
def get(path):
 with urllib.request.urlopen(RAW+path,timeout=30) as r:return r.read()
def restore(row):
 name=row['path'];m=json.loads(get(BASE+name+'/manifest.json'));parts=[]
 for i,p in enumerate(m['parts']):
  b=get(p['path']);assert hashlib.sha256(b).hexdigest()==p['sha256'];parts.append(b if i==0 else b[int(p.get('header_bytes',0)):])
 b=b''.join(parts);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],(name,len(b),row['bytes'])
 label='r12-'+('secondary-'+name.split('/secondary/')[1].replace('/','-') if '/secondary/' in name else 'pilot-'+Path(name).name)
 (E/label).write_bytes(b)
 return {'file':label,'source_path':BASE+name,'source_url':RAW+BASE+name+'/manifest.json','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
r=json.loads((E/'r12-report.json').read_text());selected=[]
for a in r['artifacts']:
 p=a['path']
 if '/snapshots/'+r['snapshot_sha256']+'/' not in p:continue
 if p.endswith(('pilot/primary_comparisons.csv','pilot/participant_metrics.csv','secondary/head_block/participant_metrics.csv','secondary/shift_block/participant_metrics.csv')):selected.append(a)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(restore,selected))
(E/'r12-restored-provenance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
