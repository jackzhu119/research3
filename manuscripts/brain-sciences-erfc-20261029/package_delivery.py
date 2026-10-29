from pathlib import Path
import hashlib,json,zipfile,datetime
import numpy,pandas,matplotlib,docx
P=Path(__file__).resolve().parent
version={'Python':'3.12','NumPy':numpy.__version__,'pandas':pandas.__version__,'Matplotlib':matplotlib.__version__,'python-docx':docx.__version__}
(P/'submission/software_versions.json').write_text(json.dumps(version,indent=2)+'\n')
(P/'requirements-analysis.txt').write_text('numpy=='+numpy.__version__+'\npandas=='+pandas.__version__+'\nmatplotlib=='+matplotlib.__version__+'\n')
allowed=[]
for path in P.rglob('*'):
 if not path.is_file() or path.is_symlink():continue
 rel=path.relative_to(P)
 if any(part in ('__pycache__','_previous') for part in rel.parts):continue
 if path.suffix.lower() in ('.zip','.dot','.html','.txt') and path.name not in ('requirements-analysis.txt','README_FIGURES.txt'):continue
 if path.name in ('DELIVERY_MANIFEST.json','PACKAGE_RECEIPT.json'):continue
 if rel.parts[0]=='literature':continue
 if rel.parts[0]=='research' and path.suffix.lower() not in ('.md','.json','.py','.bib','.csl'):continue
 if rel.parts[0]=='evidence' and path.suffix.lower() not in ('.md','.json','.csv'):continue
 if rel.parts[0] not in ('drafts','evidence','figures','research','submission','tables') and path.suffix.lower() not in ('.md','.docx','.pdf','.py','.json','.txt'):continue
 allowed.append(path)
rows=[{'path':x.relative_to(P).as_posix(),'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sorted(allowed)]
manifest={'version':'D1-20261029-ERFC','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_result_ref':'a16d06760ae77699a809c9ba36d002d684560c82','content_scope':'Author-review manuscript, original data-derived figures, verified numerical inputs, reproducible analyses and submission-policy notes. Raw EEG and training checkpoints are not embedded.','files':rows}
(P/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');allowed.append(P/'DELIVERY_MANIFEST.json')
archive=P/'Brain_Sciences_ERFC_Author_Review_Package.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(allowed):z.write(f,arcname='Brain_Sciences_ERFC/'+f.relative_to(P).as_posix())
with zipfile.ZipFile(archive) as z:
 bad=z.testzip();assert bad is None
 assert not any('/_previous/' in n or n.endswith(('.dot','.html')) for n in z.namelist())
receipt={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(allowed),'zip_crc_verified':True,'contents_manifest':'DELIVERY_MANIFEST.json'}
(P/'PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
