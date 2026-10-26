from pathlib import Path
import hashlib
import json
import subprocess

root=Path(__file__).resolve().parent
source=(root/'launch_template.sh').read_text()
proof=[]
for number, folder, primary in (
    (9, '/workspace/paper3-r9-cumsum-repair', 'https://paste.rs/lhcCa'),
    (1, '/workspace/paper3-r1-startup-cache-repair', 'https://paste.rs/IipNg'),
):
    directory=Path(folder)
    package=json.loads((directory/'PACKAGE.json').read_text())
    publication=json.loads((directory/'PUBLICATION.json').read_text())
    mirror='https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/'+publication['commit']+'/'+publication['prefix']+package['entry']
    for key,value in {'SHA':package['sha256'], 'PRIMARY':primary, 'MIRROR':mirror}.items():
        source=source.replace('__R%d_%s__'%(number,key),value)
    proof.append({'round':number,**package,'primary':primary,'mirror':mirror})
assert '__R' not in source
path=root/'FIX-PAPER3-R19.sh';path.write_text(source)
subprocess.run(['bash','-n',str(path)],check=True)
metadata={'entry':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'components':proof,'school_jobs_changed_by_assistant':0}
(root/'PACKAGE.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata))
