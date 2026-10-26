from fetch_pinned import *
pin=json.loads((BASE/'PINNED_HEAD.json').read_text())['head'];audit=json.loads((BASE/'POINTER_AUDIT.json').read_text())
def one(row):
 r=row['round'];pointer=row['pointers']['PAPER_DATA_LATEST.json']['value'];path=pointer['manifest'];raw,prov=read_repo(path,pin);manifest=json.loads(raw)
 folder=BASE/('round-%02d'%r);(folder/'manifest.json').write_bytes(raw);(folder/'manifest.provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
 return {'round':r,'keys':list(manifest),'files':manifest.get('files'),'source_files':manifest.get('source_files'),'source_entries':manifest.get('source_entries'),'parts':manifest.get('parts'),'manifest_size':len(raw)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: values=list(pool.map(one,audit['rounds']))
for v in values:
 print(json.dumps({'round':v['round'],'keys':v['keys'],'manifest_size':v['manifest_size'],'file_count':len(v.get('files') or []),'file_preview':(v.get('files') or [])[:2],'source_files_preview':(v.get('source_files') or [])[:2]}))
