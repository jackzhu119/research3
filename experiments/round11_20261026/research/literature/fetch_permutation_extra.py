import requests,datetime,hashlib,json,pathlib,concurrent.futures,re
from html.parser import HTMLParser
import html
class DocParser(HTMLParser):
 def __init__(self):super().__init__();self.metas={};self.text=[];self.title=[];self.intitle=False
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='title':self.intitle=True
  if tag=='meta':
   n=a.get('name') or a.get('property')
   if n and n.startswith(('citation_','dc.','og:')):self.metas.setdefault(n,[]).append(a.get('content'))
 def handle_endtag(self,tag):
  if tag=='title':self.intitle=False
 def handle_data(self,data):
  self.text.append(data)
  if self.intitle:self.title.append(data)
B=pathlib.Path('/workspace/paper3-round11/research/literature');C=pathlib.Path('/tmp/r11-lit-primary')
URLS=[('bishop_author_page', 'https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/'), ('bishop_publisher_page', 'https://direct.mit.edu/neco/article/7/1/108-116/5828'), ('neyman_euclid', 'https://projecteuclid.org/journals/statistical-science/volume-5/issue-4/On-the-Application-of-Probability-Theory-to-Agricultural-Experiments-Essay/10.1214/ss/1177012031.full'), ('neyman_euclid_pdf', 'https://projecteuclid.org/journals/statistical-science/volume-5/issue-4/On-the-Application-of-Probability-Theory-to-Agricultural-Experiments-Essay/10.1214/ss/1177012031.pdf'), ('freedman_euclid_pdf', 'https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-1/On-regression-adjustments-in-experiments-with-several-treatments/10.1214/07-AOAS143.pdf'), ('bohning_discovery', 'https://api.crossref.org/works?query.title=Multinomial%20logistic%20regression%20algorithm&rows=3')]

def one(pair):
 k,u=pair;t=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  r=requests.get(u,timeout=(10,35),headers={'User-Agent':'ResearchLiteratureAudit/1.0'});content=r.content;sha=hashlib.sha256(content).hexdigest();p=C/(k+('.pdf' if content.startswith(b'%PDF') else '.html'));p.write_bytes(content)
  soup=DocParser();soup.feed(content.decode('utf-8',errors='replace'));title=' '.join(soup.title);metas=soup.metas
  txt=' '.join(' '.join(soup.text).split());(C/(k+'.txt')).write_text(txt)
  return {'key':k,'requested_url':u,'final_url':r.url,'status':r.status_code,'fetched_utc':t,'bytes':len(content),'sha256':sha,'content_type':r.headers.get('content-type'),'cache_path':str(p),'title':title,'meta':metas,'text_excerpt':txt[:650]}
 except Exception as e:return {'key':k,'requested_url':u,'fetched_utc':t,'error':type(e).__name__+': '+str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:out=list(ex.map(one,URLS))
old=json.loads((B/'primary_source_fetch_manifest.json').read_text());(B/'primary_source_fetch_manifest.json').write_text(json.dumps(old+out,ensure_ascii=False,indent=2)+'\n')
for x in out:print(json.dumps({k:x.get(k) for k in ['key','status','bytes','title','meta','error']},ensure_ascii=False))
