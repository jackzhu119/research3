from fetch_pinned import *
gh=client();jobs=[];refs={}
for r in range(3,8):
 branch='paper3-r%d-progress'%r;head=gh.api(gh.prefix+'/git/ref/heads/'+branch)['object']['sha'];refs[str(r)]={'branch':branch,'head':head}
 for name in('worker.py','report.py','algorithm.py','diagnostics.py'):
  path='experiments/followup_20261024_r%d/paper3r%d/'%(r,r)+name;jobs.append((r,path,head,name))
def one(job):
 r,path,head,name=job;raw,row=read_repo(path,head)
 if raw is None:return {'round':r,'path':path,'missing':True}
 p=BASE/'code'/('round-%02d'%r)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 local=Path('/workspace/paper3-round%d/paper3r%d/'%(r,r))/name
 return {'round':r,**row,'same_as_preexisting_local_source':local.is_file() and local.read_bytes()==raw,'local':str(p.relative_to(BASE))}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,jobs))
(BASE/'CODE_PROVENANCE.json').write_text(json.dumps({'refs':refs,'files':rows},indent=2)+'\n')
print(json.dumps({'refs':refs,'files':len(rows),'same_as_local':sum(x.get('same_as_preexisting_local_source',False) for x in rows),'missing':[x for x in rows if x.get('missing')]}))
