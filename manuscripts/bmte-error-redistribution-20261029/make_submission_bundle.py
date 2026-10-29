"""Build the journal-sized reanalysis supplement and exact upload manifest."""
from pathlib import Path
import json,zipfile,hashlib

P=Path(__file__).resolve().parent

def main():
    names=[]
    for folder in ['analysis','evidence','tables','figures/input']:
        names.extend(str(f.relative_to(P))for f in (P/folder).rglob('*')if f.is_file()and '__pycache__'not in f.parts and f.suffix!='.pyc')
    names+=['figures/plot_integrated.py','requirements-analysis.txt','SUPPLEMENT_README.md']
    names+= [str(f.relative_to(P))for f in (P/'figures').glob('*values*.csv')]
    names=sorted(set(names));out=P/'submission/Supplementary_Data_and_Code.zip'
    manifest={'scope':'Archived-prediction reanalysis, separate cohort estimates and scientific figure regeneration; no raw EEG or new model fitting.',
              'files':[{'path':n,'bytes':(P/n).stat().st_size,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()}for n in names]}
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for n in names:z.write(P/n,n)
        z.writestr('SUPPLEMENT_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out)as z:
        assert z.testzip()is None
        for r in manifest['files']:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    assert out.stat().st_size<10_000_000,out.stat().st_size
    uploads=['Manuscript.docx','submission/Ethical_and_Legal_Declarations.docx','submission/Cover_Letter.docx','Supplementary_Materials.pdf']
    uploads += [str(f.relative_to(P))for f in sorted((P/'figures').glob('Fig[1-4]_*.tif'))]
    uploads += ['submission/Figure_captions.docx','submission/Supplementary_Data_and_Code.zip']
    for n in ['Supplementary_Materials.pdf','Supplementary_Materials.docx','submission/Supplementary_Data_and_Code.zip']:assert (P/n).stat().st_size<10_000_000
    receipt={'target_journal':'Biomedical Engineering / Biomedizinische Technik','submission_portal':'https://mc.manuscriptcentral.com/bmt','submitted':False,
        'files':[{'path':n,'bytes':(P/n).stat().st_size,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()}for n in uploads],
        'supplement_file_limit_bytes':10_000_000,'supplement_members':len(names),'all_supplement_member_hashes_verified':True}
    (P/'submission/UPLOAD_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'supplement_bytes':out.stat().st_size,'supplement_members':len(names),'upload_files':len(uploads),'submitted':False}))

if __name__=='__main__':main()
