import base64,hashlib,json,re,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
FILES=('delivery.py','group_archive.py','paper_credentials.py','receipt_projection.py','r1_export.py','r11_export.py','r11_paper_fix.py','transport/__init__.py','transport/lossless_publish.py','transport/paper3_result_transport.py')
def build():
 document={};hashes={}
 for name in FILES:
  raw=(HERE/name).read_bytes();compile(raw,name,'exec')
  if re.search(rb'(?:github_pat_[A-Za-z0-9_]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})',raw):raise ValueError('PUBLIC_SOURCE_CONTAINS_CREDENTIAL_LITERAL')
  document[name]=raw.decode();hashes[name]=hashlib.sha256(raw).hexdigest()
 payload=zlib.compress(json.dumps(document,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode(),9)
 digest=hashlib.sha256(payload).hexdigest();template=(HERE/'bootstrap_template.py').read_text()
 entry=template.replace('__PAYLOAD_SHA256__',digest).replace('__PAYLOAD_B85__',base64.b85encode(payload).decode()).replace('__FILE_HASHES__',repr(hashes)).encode()
 compile(entry,'START-PAPER3-R1-R11-RESULTS.py','exec');(HERE/'START-PAPER3-R1-R11-RESULTS.py').write_bytes(entry)
 result=dict(entrypoint='START-PAPER3-R1-R11-RESULTS.py',entrypoint_bytes=len(entry),entrypoint_sha256=hashlib.sha256(entry).hexdigest(),payload_sha256=digest,files=hashes,dependencies='Python 3.12 standard library plus existing school R11 operational package; no pip or Torch import',scientific_workers_modified=False,slurm_calls=0)
 (HERE/'RELEASE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':build()
