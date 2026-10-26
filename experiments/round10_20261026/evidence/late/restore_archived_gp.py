import sys,json,hashlib,tarfile,io,concurrent.futures
from fetch_late import *

def manifest(d):
 path=d['path']+'/manifest.json';m=getj(path);target=BASE/'archive-manifests'/d['name'];target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(m));return dict(m,manifest_path=path)
def download(part):
 p=BASE/'archive-parts'/part['sha256'];p.parent.mkdir(exist_ok=True)
 if p.exists() and p.stat().st_size==part['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==part['sha256']:return p.read_bytes()
 with GH._request('https://api.github.com/'+GH.prefix+'/releases/assets/'+str(part['asset_id']),accept='application/octet-stream') as f:b=f.read()
 assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256'],'archiveSHA'
 return b

def restore(m):
 import numpy as np
 sys.path.insert(0,str(BASE/'source'))
 native=BASE/'source/paper3r7/report.py'
 assert hashlib.sha256(native.read_bytes()).hexdigest()=='a4392ce1faa303f55c16603100684be5b60e3d14eca8fdf30017f2df7aa6eed7'
 from paper3r7.report import row_metrics, _query_signature
 records={r['source_path']:r for r in m['files']}
 wanted={k for k in records if k.endswith('.json') and '/evaluations/' in k or k.endswith('/result.json')}
 out=[];compact=[]
 b=b''.join(download(p) for p in sorted(m['parts'],key=lambda p:p['index']))
 with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as tf:
  for t in tf:
   if t.name not in wanted:continue
   raw=tf.extractfile(t).read();r=records[t.name];assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
   if t.name.endswith('/summary.json') or t.name.endswith('/result.json'):
    p=BASE/'r9-archived-task-summaries'/t.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
   else:
    v=json.loads(raw);metrics=row_metrics(v)
    for k in ['nll_balanced','brier_balanced']:
     if v.get(k) is not None: assert abs(v[k]-metrics[k])<2e-6,(m['key'],k)
    compact.append({'task':m['key'],'partition':v['partition'],'budget':v['budget'],'scenario':v['scenario'],'participant_id':str(v['participant_id']),'session_id':str(v['session_id']),'query_signature':_query_signature(v),'source_identity':v['source_identity'],'data_identity':v['data_identity'],'metrics':metrics,'prediction_diagnostics':v.get('prediction_diagnostics',{}),'query_mechanism_bins':v.get('query_mechanism_bins',[]),'source_path':t.name,'source_sha256':r['sha256']})
   out.append(r)
 assert len(out)==len(wanted),(m['key'],len(out),len(wanted))
 target=BASE/'r9-compact-session-metrics'/(m['key']+'.jsonl');target.parent.mkdir(exist_ok=True)
 target.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in compact))
 return {'task':m['key'],'manifest':m['manifest_path'],'archive_parts_verified':len(m['parts']),'archive_bytes_verified':len(b),'files':out,'compact_sessions':len(compact)}
def main():
 dirs=json.loads((BASE/'r9_archive_group_dirs.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:ms=list(ex.map(manifest,dirs))
 (BASE/'r9_available_archive_groups.json').write_text(json.dumps([{k:v for k,v in m.items() if k!='files'} for m in ms],indent=2))
 selected=[m for m in ms if m['key'].startswith('core-') and any('-'+v+'-' in m['key'] for v in ['linear_map','linear_pp','linear_anchored','linear_global_mixture','rbf_map','rbf_pp','rbf_anchored','rbf_global_mixture','ea_linear_map','ea_linear_pp','ea_linear_anchored','ea_linear_global_mixture','ea_rbf_map','ea_rbf_pp','ea_rbf_anchored','ea_rbf_global_mixture'])]
 print(json.dumps({'selected':len(selected),'archive_MB':sum(m['archive_bytes'] for m in selected)/1e6,'tasks':[m['key'] for m in selected]}),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  out=[]
  for r in ex.map(restore,selected):out.append(r);print(json.dumps({'restored_task':r['task'],'files':len(r['files']),'archive_MB':r['archive_bytes_verified']/1e6}),flush=True)
 (BASE/'R9_GP_ARCHIVE_RESTORE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
