"""Read-only GitHub evidence reader; credentials are selected, never printed."""
import json,os,ssl,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path
class SafeRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  if urllib.parse.urlsplit(newurl).scheme!='https':raise ValueError('Non-HTTPS evidence redirect rejected')
  out=super().redirect_request(req,fp,code,msg,headers,newurl)
  if out is not None and urllib.parse.urlsplit(newurl).hostname!='api.github.com':
   for collection in [out.headers,out.unredirected_hdrs]:
    for key in list(collection):
     if key.lower()=='authorization':del collection[key]
  return out
class GitHubRead:
 prefix='repos/jackzhu119/research3'
 def __init__(self,token=None):
  self.token=token;self.opener=urllib.request.build_opener(SafeRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
 def _request(self,url,accept='application/vnd.github+json'):
  if urllib.parse.urlsplit(url).hostname!='api.github.com':raise ValueError('GitHub API evidence endpoint required')
  h={'Accept':accept,'User-Agent':'paper3-r10-evidence-readonly','X-GitHub-Api-Version':'2022-11-28'}
  if self.token:h['Authorization']='Bearer '+self.token
  for i in range(5):
   try:return self.opener.open(urllib.request.Request(url,headers=h),timeout=90)
   except (OSError,urllib.error.URLError):
    if i==4:raise
    time.sleep(min(2**i,8))
 def api(self,path):
  url=path if path.startswith('https://') else 'https://api.github.com/'+path
  with self._request(url) as f:return json.load(f)
def credential():
 explicit=os.environ.get('PAPER3_GITHUB_CREDENTIALS')
 paths=[Path(explicit)] if explicit else [Path.home()/'.config/paper3/github.json',Path('/workspace/paper3-v2-design/private/github.json')]
 for p in paths:
  if p.is_file():
   v=json.loads(p.read_text());token=v.get('token')
   if isinstance(token,str) and token:return token
 return None
