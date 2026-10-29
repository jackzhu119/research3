from pathlib import Path
import json,urllib.request,hashlib,concurrent.futures,io,csv
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'evidence';REF='a16d06760ae77699a809c9ba36d002d684560c82';BASE='https://raw.githubusercontent.com/jackzhu119/research3/'+REF+'/'
def get(path):
 with urllib.request.urlopen(BASE+path,timeout=35) as r:return r.read()
def fetch_old(roundkey,relative):
 m=json.loads((OUT/(roundkey+'-paper-manifest.json')).read_text());entry=next(x for x in m['github_representations'] if x['source_path']==relative)
 manifestpath=json.loads(Path('/workspace/paper3-results-completeness-20261029/completed-rounds.json').read_text())['rounds'][roundkey[1:]]['paper_manifest'];prefix=manifestpath.rsplit('/',1)[0]
 path=OUT/roundkey/relative;path.parent.mkdir(exist_ok=True,parents=True)
 if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==entry['source_sha256']:
  raw=path.read_bytes()
 else:
  blocks=[get(prefix+'/'+x['path']) for x in entry['parts']]
  for block,part in zip(blocks,entry['parts']):
   assert len(block)==part['bytes'] and hashlib.sha256(block).hexdigest()==part['sha256']
  if entry['storage']=='csv-row-parts':
   rows=[];header=None
   for block in blocks:
    partrows=list(csv.reader(io.StringIO(block.decode('utf-8-sig'))));header=partrows[0] if header is None else header;assert partrows[0]==header;rows+=partrows[1:]
   assert len(rows)==entry['data_rows'];out=io.StringIO();writer=csv.writer(out);writer.writerows([header,*rows]);raw=out.getvalue().encode()
  else:
   raw=b''.join(blocks);assert len(raw)==entry['source_bytes'];assert hashlib.sha256(raw).hexdigest()==entry['source_sha256']
  path.write_bytes(raw)
 return dict(round=roundkey,source_path=relative,bytes=len(raw),sha256=entry['source_sha256'],representation=entry['storage'],exact_original_bytes=entry['storage']=='original',repository_manifest=BASE+manifestpath,parts=[BASE+prefix+'/'+x['path'] for x in entry['parts']])
def fetch_new(roundkey,relative):
 m=json.loads((OUT/(roundkey+'-reports-manifest.json' if roundkey=='r10' else roundkey+'-paper-manifest.json')).read_text());entry=next(x for x in m['files'] if x['source_path']==relative)
 s=json.loads(get(entry['manifest']));path=OUT/roundkey/relative;path.parent.mkdir(exist_ok=True,parents=True)
 raw=path.read_bytes() if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==s['source_sha256'] else b''.join(get(x['path']) for x in s['parts'])
 assert len(raw)==s['source_bytes'];assert hashlib.sha256(raw).hexdigest()==s['source_sha256'];path.write_bytes(raw)
 return dict(round=roundkey,source_path=relative,bytes=len(raw),sha256=s['source_sha256'],source_manifest=BASE+entry['manifest'],parts=[BASE+x['path'] for x in s['parts']])
items=[]
for r in ('r3','r7'):
 for f in ('plan.json','reports/primary_comparisons.csv','reports/evaluation_summaries.csv','reports/budget_curves.csv','reports/timings.csv'):
  items.append((fetch_old,r,f))
for r in ('r10','r10mechanism'):
 for f in ('plan.json','reports/report.json','reports/comparisons.csv','reports/method_summary.csv','reports/participant_metrics.csv','reports/report-status.json'):
  items.append((fetch_new,r,f))
items.append((fetch_new,'r10','reports/development_amendment.json'))
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
 futures=[ex.submit(fun,r,f) for fun,r,f in items];rows=[]
 for future in concurrent.futures.as_completed(futures):
  row=future.result();rows.append(row);print(json.dumps({k:row[k] for k in ('round','source_path','bytes')}),flush=True)
(OUT/'main_download_proofs.json').write_text(json.dumps(rows,indent=2)+'\n')
