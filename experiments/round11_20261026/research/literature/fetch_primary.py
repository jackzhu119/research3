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
URLS=[
('maml','https://proceedings.mlr.press/v70/finn17a.html'),
('hierarchical_bayes','https://arxiv.org/abs/1801.08930'),
('dkt','https://arxiv.org/abs/1910.05199'),
('alpaca','https://proceedings.mlr.press/v87/harrison18a.html'),
('deep_kernel','https://proceedings.mlr.press/v51/wilson16.html'),
('r2d2','https://arxiv.org/abs/1805.08136'),
('metaoptnet','https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html'),
('pmaml','https://arxiv.org/abs/1806.02817'),
('metacurvature','https://arxiv.org/abs/1902.03399'),
('mtl_feature','https://proceedings.neurips.cc/paper/2006/hash/0afa92fc0f8a9cf051bf2961b06ac56b-Abstract.html'),
('arxiv_kernel_search','https://export.arxiv.org/api/query?search_query=all:%22kernel%20meta%20learning%22&start=0&max_results=8'),
('kml_search','https://arxiv.org/search/?query=kernel+modulation&searchtype=all&abstracts=show&order=-announced_date_first&size=50')]
def one(pair):
 k,u=pair;t=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  r=requests.get(u,timeout=(10,35),headers={'User-Agent':'ResearchLiteratureAudit/1.0'});content=r.content;sha=hashlib.sha256(content).hexdigest();p=C/(k+('.pdf' if content.startswith(b'%PDF') else '.html'));p.write_bytes(content)
  soup=DocParser();soup.feed(content.decode('utf-8',errors='replace'));title=' '.join(soup.title);metas=soup.metas
  txt=' '.join(' '.join(soup.text).split());(C/(k+'.txt')).write_text(txt)
  return {'key':k,'requested_url':u,'final_url':r.url,'status':r.status_code,'fetched_utc':t,'bytes':len(content),'sha256':sha,'content_type':r.headers.get('content-type'),'cache_path':str(p),'title':title,'meta':metas,'text_excerpt':txt[:650]}
 except Exception as e:return {'key':k,'requested_url':u,'fetched_utc':t,'error':type(e).__name__+': '+str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:out=list(ex.map(one,URLS))
(B/'primary_source_fetch_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for x in out:print(json.dumps({k:x.get(k) for k in ['key','status','bytes','title','meta','error']},ensure_ascii=False))
