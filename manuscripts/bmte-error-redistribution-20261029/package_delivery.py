"""Create a self-contained authored working package with verified member hashes."""
from pathlib import Path
import json,re,zipfile,hashlib,datetime

P=Path(__file__).resolve().parent

def main():
    names=[]
    for folder in ['analysis','evidence','tables','figures','submission']:
        names.extend(str(f.relative_to(P))for f in (P/folder).rglob('*')if f.is_file()and '__pycache__'not in f.parts and f.suffix!='.pyc')
    names+=[str(f.relative_to(P))for f in P.iterdir()if f.is_file()and f.suffix in ['.md','.py','.json','.txt','.docx','.pdf']and f.name not in ['PACKAGE_RECEIPT.json','DELIVERY_MANIFEST.json','NARRATIVE_NOTES_ZH.md']]
    research=['bmte-vancouver.csl','journal_requirements.md','journal_requirements.json','address_ethics_verification.md','address_ethics_verification.json','citation_aliases.json','additional_dataset_citations.json','manuscript_metrics.json','Manuscript_render.json','Supplementary_Materials_render.json','INDEPENDENT_REVIEW.md','INDEPENDENT_REVIEW.json','EXTRACTED_SUPPLEMENT_REPRODUCTION.json']
    names+=['research/'+n for n in research if(P/'research'/n).is_file()]
    names=sorted(set(names))
    for n in names:
        f=P/n
        if f.suffix in ['.md','.py','.json','.txt','.csv']:
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|sk-proj-[A-Za-z0-9_-]{20,})',f.read_text()),n
    manifest={'title':'Error redistribution across conventional and neural motor imagery EEG recalibration','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status':'AUTHOR_REVIEW_PACKAGE','submitted':False,'third_party_reading_fulltexts_included':False,'new_training':False,
              'files':[{'path':n,'bytes':(P/n).stat().st_size,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()}for n in names]}
    (P/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    out=P/'BMTE_Error_Redistribution_Manuscript_Package.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for n in names+['DELIVERY_MANIFEST.json']:z.write(P/n,n)
    with zipfile.ZipFile(out)as z:
        assert z.testzip()is None
        for r in manifest['files']:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    receipt={'file':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'members':len(names)+1,'all_member_hashes_verified':True,'submitted':False}
    (P/'PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
