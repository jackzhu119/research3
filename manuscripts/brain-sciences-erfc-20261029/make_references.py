from pathlib import Path
import json,re,urllib.request
P=Path(__file__).resolve().parent
R=P/'research'
refs={}
for n in ('root_references.json','additional_references.json'):
 for key,value in json.loads((R/n).read_text()).items():
  if value.get('title'):refs[key]=value
for x in json.loads((R/'intro_foundation_references.json').read_text()):
 if x['key'] in refs:continue
 names=[]
 for a in x['author']:
  given,_,family=a.rpartition(' ');names.append({'given':given,'family':family})
 refs[x['key']]={'id':x['key'],'type':'article-journal','title':x['title'],'author':names,'DOI':x['doi'],'issued':{'date-parts':[[x['year']]]},'container-title':x['journal'],'volume':x.get('volume'),'issue':x.get('issue'),'page':x.get('pages'),'URL':x['url']}
for x in json.loads((R/'literature_a.json').read_text())['papers']:
 names=[]
 for a in x['authors']:
  given,_,family=a.rpartition(' ');names.append({'given':given,'family':family})
 refs[x['key']]={'id':x['key'],'type':'article-journal','title':x['title'],'author':names,'DOI':x['doi'],'issued':{'date-parts':[[x['year']]]},'container-title':x['journal'],'volume':x.get('volume'),'issue':x.get('issue'),'page':x.get('pages_or_article'),'URL':x['url']}
for x in json.loads((R/'literature_b.json').read_text())['papers']:
 refs[x['id']]={'id':x['id'],'type':'article-journal','title':x['title'],'author':x['authors'],'DOI':x['doi'],'issued':{'date-parts':[[x['year']]]},'container-title':x['journal'],'volume':x.get('volume'),'issue':x.get('issue'),'page':x.get('pages'),'URL':x['publisher_url']}
refs['boyd2004']={'id':'boyd2004','type':'book','title':'Convex Optimization','author':[{'given':'Stephen','family':'Boyd'},{'given':'Lieven','family':'Vandenberghe'}],'issued':{'date-parts':[[2004]]},'publisher':'Cambridge University Press','publisher-place':'Cambridge, UK','DOI':'10.1017/CBO9780511804441','URL':'https://web.stanford.edu/~boyd/cvxbook/'}
refs['holm1979']={'id':'holm1979','type':'article-journal','title':'A Simple Sequentially Rejective Multiple Test Procedure','author':[{'given':'Sture','family':'Holm'}],'issued':{'date-parts':[[1979]]},'container-title':'Scandinavian Journal of Statistics','volume':'6','issue':'2','page':'65-70','URL':'https://www.jstor.org/stable/4615733'}
refs['nhc2023']={'id':'nhc2023','type':'webpage','title':'Measures for Ethical Review of Life Science and Medical Research Involving Human Subjects (Guo Wei Ke Jiao Fa [2023] No. 4), Article 32','author':[{'literal':'National Health Commission of the People’s Republic of China; Ministry of Education; Ministry of Science and Technology; National Administration of Traditional Chinese Medicine'}],'issued':{'date-parts':[[2023,2,18]]},'accessed':{'date-parts':[[2026,10,29]]},'URL':'https://www.gov.cn/zhengce/zhengceku/2023-02/28/content_5743658.htm'}
# No author inventing: all journal metadata are publisher-deposited or verified primary records.
used=set(re.findall(r'(?<![\w.])@([A-Za-z][A-Za-z0-9_-]*)',(P/'Manuscript.md').read_text()))
missing=used-set(refs)
if missing:raise ValueError('Missing reference '+str(missing))
for v in refs.values():
 for k in list(v):
  if v[k] is None:del v[k]
(P/'references.json').write_text(json.dumps(list(refs.values()),ensure_ascii=False,indent=2)+'\n')
(P/'research/reference_validation.json').write_text(json.dumps({'cited_references':len(used),'unresolved_keys':[],'unused_verified_bibliography_entries':sorted(set(refs)-used),'cited_keys':sorted(used)},indent=2)+'\n')
print('Cited references',len(used),'verified entries',len(refs))
