"""Build a self-contained stdlib-only operational sidecar; no old scientific code."""
from pathlib import Path
import base64,hashlib,io,json,stat,zipfile
ROOT=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()
def build(root=ROOT):
 root=Path(root);paths=[]
 for name in ('controller_fix.py','repair.py','paper_fix.py','build_delivery.py','installer_template.py'):
  p=root/name
  if not p.is_file():raise ValueError('Repair source missing: '+name)
  paths.append(p)
 for p in root.iterdir():
  if p.is_file() and not p.is_symlink() and p.suffix.lower() in ('.md','.json') and p.stat().st_size<=1024*1024 and p.name.lower() not in ('github.json','credentials.json'):paths.append(p)
 for p in (root/'tests').glob('*.py'):paths.append(p)
 for p in (root/'proof').glob('*.json'):
  if p.is_file() and not p.is_symlink() and p.stat().st_size<=1024*1024:paths.append(p)
 values={str(p.relative_to(root)):p.read_bytes() for p in sorted(set(paths))}
 manifest=dict(schema_version='paper3-r11-operational-sidecar-package-v1',scope='Operational code/tests/proof only; no original scientific package, model, EEG, plan or credentials',
  files={name:dict(bytes=len(data),sha256=sha(data)) for name,data in values.items()})
 m=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();values['MANIFEST.json']=m
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name,data in sorted(values.items()):
   info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.create_system=3;info.external_attr=(stat.S_IFREG|0o600)<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
 raw=buf.getvalue();entry=(root/'installer_template.py').read_text().replace('__PAYLOAD_B85__',base64.b85encode(raw).decode()).replace('__PAYLOAD_SHA256__',sha(raw)).replace('__MANIFEST_SHA256__',sha(m)).encode()
 compile(entry,'REPAIR-PAPER3-R11-OPERATIONS.py','exec');out=root/'delivery';out.mkdir(exist_ok=True)
 (out/'REPAIR-PAPER3-R11-OPERATIONS.py').write_bytes(entry);(out/'paper3-r11-operational-repair.zip').write_bytes(raw);(out/'MANIFEST.json').write_bytes(m)
 record=dict(schema_version='paper3-r11-operational-sidecar-delivery-v1',entry='REPAIR-PAPER3-R11-OPERATIONS.py',entry_sha256=sha(entry),entry_bytes=len(entry),payload_sha256=sha(raw),payload_bytes=len(raw),manifest_sha256=sha(m),files=len(manifest['files']),default='install_only',actual_school_jobs_changed=0)
 (out/'PACKAGE.json').write_text(json.dumps(record,indent=2)+'\n');return record
if __name__=='__main__':print(json.dumps(build(),indent=2))
