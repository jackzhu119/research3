import sys,base64,json,hashlib,concurrent.futures,csv,io
from fetch_late import *
def restore(n,r):
 source=r['source_path'];target=BASE/f'round-{n:02d}'/source
 expected=r.get('source_sha256',r.get('sha256'));size=r.get('source_bytes',r.get('bytes'))
 if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:return {'round':n,'path':source,'sha256':expected,'bytes':size,'cached':True}
 if 'manifest' in r:
  m=json.loads(fetch(r['manifest'],sha=r['manifest_sha256']));parts=m['parts']
 else:parts=r['parts']
 chunks=[];offset=0
 for part in parts:
  b=fetch(part['path'],sha=None if part.get('encoding')=='base64' else part['sha256'])
  if part.get('encoding')=='base64':b=base64.b64decode(b,validate=False)
  assert hashlib.sha256(b).hexdigest()==part['sha256'],source
  assert len(b)==part['bytes'] and part.get('offset_bytes',part.get('byte_offset'))==offset
  chunks.append(b);offset+=len(b)
 b=b''.join(chunks);assert offset==size and hashlib.sha256(b).hexdigest()==expected,source
 target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 return {'round':n,'path':source,'sha256':expected,'bytes':size,'parts':len(parts),'exact_bytes_sha256_verified':True}
def main():
 todo=[]
 for n in [8,9]:
  m=json.loads((BASE/f'r{n}_manifest.json').read_text())
  for r in m['files']:
   if r['source_path']=='reports/report-cache.json':continue
   todo.append((n,r))
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
  futures=[ex.submit(restore,*r) for r in todo];out=[]
  for f in concurrent.futures.as_completed(futures):
   r=f.result();out.append(r);print(json.dumps({k:r[k] for k in ['round','path','bytes']}),flush=True)
 (BASE/'RESTORED_SOURCE_AUDIT.json').write_text(json.dumps(sorted(out,key=lambda r:(r['round'],r['path'])),indent=2)+'\n')
if __name__=='__main__':main()
