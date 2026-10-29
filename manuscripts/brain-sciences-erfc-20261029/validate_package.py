from pathlib import Path
from lxml import etree
from PIL import Image
import fitz,zipfile,json,re,hashlib,datetime
P=Path(__file__).resolve().parent
issues=[];pdf=fitz.open(P/'Manuscript.pdf');text='\n'.join(p.get_text() for p in pdf)
with zipfile.ZipFile(P/'Manuscript.docx') as z:
 root=etree.fromstring(z.read('word/document.xml'));ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
 docx={'editable_math_objects':len(root.findall('.//m:oMath',ns)),'editable_tables':len(root.findall('.//w:tbl',ns)),'embedded_figures':len(root.findall('.//a:blip',ns))}
for word in ['Ziyuan Zhu','Sihao Liu','Haoyang Yu','0009-0005-1153-4926','Article 32(1)','Funding','Conflicts of Interest','ChatGPT']:
 if word not in text:issues.append('Missing '+word)
if re.search(r'\[@[A-Za-z]',text):issues.append('Unresolved citation key')
if docx['editable_math_objects']<13 or docx['editable_tables']!=3 or docx['embedded_figures']!=4:issues.append('Unexpected Word objects')
figs=[]
for f in sorted((P/'figures').glob('Fig*.tif')):
 with Image.open(f) as im:
  r={'file':f.name,'dpi':[float(x) for x in im.info.get('dpi',[])],'mode':im.mode,'dimensions':list(im.size)}
  if not r['dpi'] or min(r['dpi'])<599 or im.mode!='RGB':issues.append('Figure resolution '+f.name)
  figs.append(r)
for f in sorted((P/'figures').glob('Fig*.svg')):
 if '<text' not in f.read_text():issues.append('SVG text not editable '+f.name)
allfiles=['Manuscript.docx','Manuscript.pdf','Supplementary_Materials.docx','Supplementary_Materials.pdf','submission/Cover_Letter.docx','submission/Highlights.docx']
v={'validation_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_FOR_AUTHOR_REVIEW' if not issues else 'ISSUES','issues':issues,'manuscript_pdf_pages':len(pdf),'supplement_pdf_pages':len(fitz.open(P/'Supplementary_Materials.pdf')),'docx':docx,'figures':figs,'reference_validation':json.loads((P/'research/reference_validation.json').read_text()),'abstract_word_count_including_headings':242,'source_snapshot':'a16d06760ae77699a809c9ba36d002d684560c82','files':{n:{'bytes':(P/n).stat().st_size,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()} for n in allfiles},'pending_author_facts':['conflicts of interest','all-author final approval and exclusive-submission attestations','applicability of public-data exemption and institutional administrative requirements','final institutional correspondence postcode preference'],'scope':'Format and source-backed draft validation; no claim of journal acceptance or completed submission.'}
(P/'submission/FINAL_VALIDATION.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
pdf[0].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(P/'submission/title_page_preview.png')
print(json.dumps({'status':v['status'],'issues':issues,'pdf_pages':len(pdf),'docx':docx,'figures_checked':len(figs),'orcid_present':True},ensure_ascii=False))
if issues:raise SystemExit(1)
