"""Build immutable, documented native baseline copies (read-only upstream)."""
from fetch_pinned import *
OUT=BASE.parents[1]/'paper3r10/vendor/native';OUT.mkdir(parents=True,exist_ok=True)
metadata=json.loads((BASE/'CODE_PROVENANCE.json').read_text());refs=metadata['refs'];gh=client()
for r,branch in ((8,'paper3-round8-20261025'),(9,'paper3-round9-20261025')):refs[str(r)]={'branch':branch,'head':gh.api(gh.prefix+'/git/ref/heads/'+branch)['object']['sha']}
sources=[(3,'algorithm.py'),(4,'core.py'),(4,'algorithm.py'),(7,'baseline.py'),(7,'head_capacity.py'),(8,'transport.py'),(9,'gp.py')]
rows=[]
for r,name in sources:
 head=refs[str(r)]['head'];date='20261025' if r>=8 else '20261024';path='experiments/followup_%s_r%d/paper3r%d/'%(date,r,r)+name
 raw,record=read_repo(path,head);assert raw is not None
 local=Path('/workspace/paper3-round%d/paper3r%d/'%(r,r))/name
 assert local.is_file() and raw==local.read_bytes(),(r,name,'local/upstream mismatch')
 modified=raw;changes=[]
 if r==9:
  old=b'cumulative=torch.cumsum(pairweights[order],0)';new=b'cumulative=torch.cumsum(pairweights[order].detach().cpu(),0).to(pairweights.device)'
  assert raw.count(old)==1;modified=raw.replace(old,new);changes=[{'kind':'existing_runtime_bugfix','old':old.decode(),'new':new.decode(),'reason':'Same previously validated R9 deterministic CUDA float cumsum metadata fix; GP fit/predict algorithm unchanged.'}]
 target=OUT/('paper3r%d'%r)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(modified)
 init=target.parent/'__init__.py'
 if not init.exists():init.write_text('"""Pinned original baseline source package for isolated R10 workers."""\n')
 rows.append({**record,'round':r,'vendored_path':str(target.relative_to(OUT.parent)),'vendored_sha256':hashlib.sha256(modified).hexdigest(),'transformations':changes})
(OUT.parent/'SOURCE_BASELINES.json').write_text(json.dumps({'schema_version':'r10-native-baseline-provenance-v1','files':rows,'source_branch_refs':refs,'original_paths_modified':False,'new_algorithm_claimed':False},indent=2)+'\n')
print(json.dumps({'files':len(rows),'bytes':sum((OUT.parent/x['vendored_path']).stat().st_size for x in rows),'manifest':str(OUT.parent/'SOURCE_BASELINES.json')}))
