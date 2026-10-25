#!/usr/bin/env python3
"""Fetch declared R2/R3/R7 paper tables, validate SHA and reconstruct CSV rows.
Uses immutable paper snapshot paths; never accesses school server or training state.
"""
import csv,hashlib,io,json,pathlib,time,urllib.request,concurrent.futures,os
BASE=pathlib.Path(__file__).resolve().parent
REPO='https://raw.githubusercontent.com/jackzhu119/research3/paper3-results/'
TOKEN=json.loads(pathlib.Path(os.environ.get('PAPER3_GITHUB_CREDENTIALS','/workspace/paper3-v2-design/private/github.json')).read_text())['token']
def fetch(path):
    for i in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(REPO+path,headers={'Authorization':'Bearer '+TOKEN}),timeout=45).read()
        except Exception:
            if i==4:raise
            time.sleep(min(2**i,8))
def cached(path,local,sha=None):
    if local.exists() and (not sha or hashlib.sha256(local.read_bytes()).hexdigest()==sha):return local.read_bytes()
    b=fetch(path)
    if sha and hashlib.sha256(b).hexdigest()!=sha:raise ValueError('SHA mismatch '+path)
    local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(b);return b

def main():
    audit=[];todo=[]
    for n in [2,3,7]:
        prefix=f'results/paper3/round-{n:02d}/'
        p=json.loads(cached(prefix+'PAPER_DATA_LATEST.json',BASE/f'r{n}_pointer.json'))
        m=json.loads(cached(p['manifest'],BASE/f'r{n}_manifest.json'))
        snapshot=p['manifest'].rsplit('/',1)[0]
        for r in m['github_representations']:
            if not (r['source_path'].endswith('.csv') or r['source_path']=='plan.json'):continue
            for part in r['parts']:
                local=BASE/f'round-{n:02d}'/'download'/part['path']
                todo.append((snapshot+'/'+part['path'],local,part['sha256']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(cached,*args) for args in todo]
        for f in concurrent.futures.as_completed(fs):f.result()
    for n in [2,3,7]:
        p=json.loads((BASE/f'r{n}_pointer.json').read_text());m=json.loads((BASE/f'r{n}_manifest.json').read_text())
        for r in m['github_representations']:
            if not (r['source_path'].endswith('.csv') or r['source_path']=='plan.json'):continue
            target=BASE/f'round-{n:02d}'/r['source_path'];target.parent.mkdir(parents=True,exist_ok=True)
            parts=r['parts'];row_count=None
            if r['storage']=='original':
                b=(BASE/f'round-{n:02d}'/'download'/parts[0]['path']).read_bytes();target.write_bytes(b)
                assert hashlib.sha256(b).hexdigest()==r['source_sha256']
                if target.suffix=='.csv':row_count=max(0,sum(1 for _ in csv.reader(io.StringIO(b.decode('utf-8-sig'))))-1)
            else:
                assert target.suffix=='.csv',r
                header=None;rows=[];offset=0
                for part in parts:
                    assert part['data_row_offset']==offset
                    b=(BASE/f'round-{n:02d}'/'download'/part['path']).read_bytes();v=list(csv.reader(io.StringIO(b.decode('utf-8-sig'))))
                    if header is None:header=v[0]
                    assert v[0]==header
                    assert len(v)-1==part['data_rows'];offset+=len(v)-1;rows+=v[1:]
                assert offset==r['data_rows']
                with target.open('w',newline='') as out:
                    w=csv.writer(out);w.writerow(header);w.writerows(rows)
                row_count=offset
            audit.append({'round':n,'path':r['source_path'],'source_sha256':r['source_sha256'],'source_bytes':r['source_bytes'],'parts':len(parts),'verified_part_sha256':True,'rows':row_count,'reconstruction':'exact_source_bytes' if r['storage']=='original' else 'CSV_field_values_and_order_identical_not_original_format_bytes','snapshot_id':p['snapshot_id']})
    (BASE/'DOWNLOAD_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'files':len(audit),'sha256_verified_parts':len(todo),'rounds':[2,3,7]},indent=2))
if __name__=='__main__':main()
