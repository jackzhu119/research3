"""Restore all R2–R7 paper tables and R4–R6 unsanitized data from pinned manifests."""
from fetch_pinned import *
import csv,io,shutil
pin=json.loads((BASE/'PINNED_HEAD.json').read_text())['head'];audit=json.loads((BASE/'POINTER_AUDIT.json').read_text())
items=[];manifest_jobs=[]
for row in audit['rounds']:
 r=row['round'];pointer=row['pointers']['PAPER_DATA_LATEST.json']['value'];folder=BASE/('round-%02d'%r);m=json.loads((folder/'manifest.json').read_text());prefix=pointer['manifest'].rsplit('/',1)[0]
 if r in (4,5,6):
  for source in m['files']:manifest_jobs.append((r,source))
 else:
  for rep in m['github_representations']:
   if rep['source_path'].endswith('.csv') or rep['source_path']=='plan.json' or (r==2 and rep['source_path']=='report.json'):
    items.append((r,rep,prefix,False))
def source_manifest(job):
 r,source=job;raw,prov=read_repo(source['manifest'],pin,source['manifest_sha256']);m=json.loads(raw)
 p=BASE/('round-%02d'%r)/'source-manifests'/source['source_path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 return r,m,'',True
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:items.extend(pool.map(source_manifest,manifest_jobs))
jobs=[]
for r,rep,prefix,v2 in items:
 for p in rep['parts']:
  path=p['path'] if v2 else prefix+'/'+p['path'];cache=BASE/('round-%02d'%r)/'download'/p['path']
  jobs.append((r,p,path,cache))
def part(job):
 r,p,path,cache=job
 old=Path('/workspace/paper3-round9/evidence')/('round-%02d'%r)/'download'/p['path']
 candidates=[cache,old]
 for candidate in candidates:
  if candidate.is_file():
   raw=candidate.read_bytes()
   if len(raw)==p['bytes'] and hashlib.sha256(raw).hexdigest()==p['sha256']:
    cache.parent.mkdir(parents=True,exist_ok=True)
    if candidate!=cache:cache.write_bytes(raw)
    return {'round':r,'path':path,'bytes':len(raw),'sha256':p['sha256'],'method':'existing_local_bytes_bound_to_current_verified_manifest'}
 raw,prov=read_repo(path,pin,p['sha256']);assert len(raw)==p['bytes']
 cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(raw)
 return {'round':r,**prov,'method':'fetched_from_pinned_commit'}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(part,jobs))
records=[]
for r,rep,prefix,v2 in items:
 target=BASE/('round-%02d'%r)/rep['source_path'];target.parent.mkdir(parents=True,exist_ok=True)
 def block(p):return (BASE/('round-%02d'%r)/'download'/p['path']).read_bytes()
 if v2 or rep['storage']=='original':
  raw=b''.join(block(p) for p in rep['parts']);assert len(raw)==rep['source_bytes'];assert hashlib.sha256(raw).hexdigest()==rep['source_sha256'];target.write_bytes(raw)
  reconstruction='exact_original_bytes'
 else:
  assert rep['storage']=='csv-row-parts',rep['storage'];header=None;rows=[];offset=0
  for p in rep['parts']:
   values=list(csv.reader(io.StringIO(block(p).decode('utf-8-sig'))));assert p['data_row_offset']==offset
   if header is None:header=values[0]
   assert header==values[0];assert len(values)-1==p['data_rows'];rows.extend(values[1:]);offset+=len(values)-1
  assert offset==rep['data_rows']
  with target.open('w',newline='') as f:
   w=csv.writer(f);w.writerow(header);w.writerows(rows)
  reconstruction='CSV_parsed_fields_and_row_order_equal_formatting_may_differ'
 records.append({'round':r,'path':rep['source_path'],'source_bytes':rep['source_bytes'],'source_sha256':rep['source_sha256'],'reconstruction':reconstruction,'parts':len(rep['parts']),'local_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'pinned_head':pin})
(BASE/'DATA_RESTORATION_AUDIT.json').write_text(json.dumps({'pin':pin,'source_files':records,'parts':parts},indent=2)+'\n')
print(json.dumps({'source_files':len(records),'parts':len(parts),'exact_source_bytes':sum(x['reconstruction']=='exact_original_bytes' for x in records),'csv_semantic_sources':sum(x['reconstruction']!='exact_original_bytes' for x in records),'download_bytes':sum(x['bytes'] for x in parts)}))
