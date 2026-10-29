"""Validate source identities, numerical definitions and BMTE document constraints."""
from pathlib import Path
import json,re,hashlib,zipfile,datetime
import fitz,pandas as pd
from docx import Document
from PIL import Image
from lxml import etree

P=Path(__file__).resolve().parent

def main():
    metrics=json.loads((P/'research/manuscript_metrics.json').read_text())
    main=(P/'Manuscript.md').read_text();supp=(P/'Supplementary_Materials.md').read_text()
    refs={r['id']:r for r in json.loads((P/'references.json').read_text())};keys=set()
    for block in re.findall(r'\[[^\]]*@[a-zA-Z][^\]]*\]',main+'\n'+supp):keys.update(re.findall(r'@([A-Za-z0-9_:-]+)',block))
    assert not keys-refs.keys(),keys-refs.keys()
    assert len(keys)<=35,len(keys)
    assert metrics['abstract_words_with_headings']<=200 and metrics['running_head_characters']<=75
    assert 3<=len(metrics['keywords'])<=6 and metrics['keywords']==sorted(metrics['keywords'],key=str.casefold)
    assert not re.search(r'^#{2,5}\s+\d',main,re.M)
    assert len(re.findall(r'^\*\*Table [1-3]\.',main,re.M))==3
    assert len(re.findall(r'^!\[\]\(figures/Fig[1-4]_',main,re.M))==4
    assert '401331'in main and '400016'not in main
    assert 'The authors state no conflict of interest.'in main and 'Article 32(1)'in main
    assert 'All original studies obtained informed consent'not in main
    assert 'earlier conventional-decoder manuscript'in (P/'submission/Cover_Letter.md').read_text()
    manifest=json.loads((P/'evidence/INPUT_MANIFEST.json').read_text())
    for r in manifest['files']:assert hashlib.sha256((P/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    p=pd.read_csv(P/'tables/participant_metrics.csv',dtype={'participant':str,'budget':str})
    s=pd.read_csv(P/'tables/cohort_summary.csv')
    assert len(p)==6781 and len(s)==585
    assert not p.duplicated(['study','cohort','analysis_role','participant','method','reference','budget']).any()
    assert (p.C+p.I-p.D).abs().max()<1e-12
    assert (p.C-p.I-p.G).abs().max()<1e-12
    assert (p.G-(p.ba-p.reference_ba)).abs().max()<1e-12
    assert (s.within_cancellation+s.between_cancellation-s.total_cancellation).abs().max()<1e-12
    assert s.loc[s.D_mean.abs()<1e-12,['within_ratio','between_ratio','total_ratio']].isna().all().all()
    claims=json.loads((P/'tables/key_claims.json').read_text())
    assert claims['counts']['main_dataset_participants']==56 and claims['counts']['main_target_sessions']==212
    assert json.loads((P/'analysis/INDEPENDENT_NUMERICAL_CHECK.json').read_text())['status']=='PASS'
    docs={};pdfs={}
    ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for name,images in [('Manuscript',4),('Supplementary_Materials',4)]:
        pdf=fitz.open(P/(name+'.pdf'));text='\n'.join(pg.get_text()for pg in pdf)
        pdfs[name]={'pages':len(pdf),'words':len(text.split())}
        assert not re.search(r'\[@[A-Za-z]',text)
        with zipfile.ZipFile(P/(name+'.docx'))as z:
            root=etree.fromstring(z.read('word/document.xml'))
            obj={'editable_math':len(root.findall('.//m:oMath',ns)),'editable_tables':len(root.findall('.//w:tbl',ns)),'embedded_figures':len(root.findall('.//a:blip',ns))}
            assert obj['embedded_figures']==images,(name,obj)
            assert obj['editable_math']>0
            docs[name]=obj
        if name=='Manuscript':
            assert obj['editable_tables']==3 and len(text.split())<=6000
            for value in ['Ziyuan Zhu','Sihao Liu','Haoyang Yu','0009-0005-1153-4926','401331','ChatGPT','Article 32(1)']:
                assert value in text,value
            assert re.search(r'Lawhern\s+VJ',text) and re.search(r'et\s+al\.',text)
    figureproof=[]
    for f in sorted((P/'figures').glob('Fig*.tif')):
        with Image.open(f)as im:
            dpi=im.info.get('dpi',[]);assert im.mode=='RGB'and min(dpi,default=0)>=599
            figureproof.append({'path':str(f.relative_to(P)),'mode':im.mode,'dpi':[float(x)for x in dpi]})
    assert len(figureproof)==8
    for f in (P/'figures').glob('Fig*.svg'):assert '<text'in f.read_text()
    declarations=Document(P/'submission/Ethical_and_Legal_Declarations.docx')
    assert all(r.cells[1].text.strip()for r in declarations.tables[0].rows[1:9])
    authors=pd.read_csv(P/'submission/authors_for_submission.csv')
    assert len(authors)==3 and authors.affiliation.str.contains('401331').all()
    for name in ['Supplementary_Materials.docx','Supplementary_Materials.pdf','submission/Supplementary_Data_and_Code.zip']:assert (P/name).stat().st_size<10_000_000
    receipt={'status':'PASS_FOR_AUTHOR_REVIEW','validated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_journal':'Biomedical Engineering / Biomedizinische Technik',
             'submitted':False,'metrics':metrics,'cited_reference_count_main_and_supplement':len(keys),'cited_keys':sorted(keys),'documents':docs,'pdfs':pdfs,'figures':figureproof,
             'main_participants':56,'main_target_sessions':212,'separate_Kumar_development':6,'separate_source_extension':85,
             'input_files_hash_verified':len(manifest['files']),'new_training':False,'new_pvalues':False,'pooled_effect':False,'numerical_independent_check':'PASS',
             'files':{name:{'bytes':(P/name).stat().st_size,'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest()}for name in ['Manuscript.md','Manuscript.docx','Manuscript.pdf','Supplementary_Materials.docx','Supplementary_Materials.pdf','submission/Cover_Letter.docx','submission/Ethical_and_Legal_Declarations.docx']}}
    (P/'submission/FINAL_VALIDATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:receipt[k]for k in ['status','cited_reference_count_main_and_supplement','documents','pdfs','input_files_hash_verified','new_training','new_pvalues']}))

if __name__=='__main__':main()
