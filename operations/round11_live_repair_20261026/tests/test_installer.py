import importlib.util,json,subprocess,sys
from pathlib import Path
import build_delivery


def fixture(tmp_path):
 source=tmp_path/'source';source.mkdir();origin=Path(__file__).resolve().parents[1]
 for name in ('build_delivery.py','installer_template.py','controller_fix.py','paper_fix.py'):(source/name).write_bytes((origin/name).read_bytes())
 (source/'repair.py').write_text("import json,sys\nprint(json.dumps({'software_fixture_route':sys.argv[1:]}))\n")
 return source,build_delivery.build(source)


def test_default_actual_installer_is_offline_and_all_files_match(tmp_path):
 source,record=fixture(tmp_path);entry=source/'delivery'/record['entry'];home=tmp_path/'home'
 guard='''import importlib.abc,runpy,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,n,p=None,t=None):
  if n.split('.')[0] in {'numpy','torch','socket','requests','subprocess'}:raise AssertionError('Forbidden installation import '+n)
sys.meta_path.insert(0,Guard());sys.argv=[sys.argv[1],'--home',sys.argv[2]];runpy.run_path(sys.argv[0],run_name='__main__')
'''
 done=subprocess.run([sys.executable,'-I','-S','-B','-c',guard,str(entry),str(home)],capture_output=True,text=True)
 assert done.returncode==0,done.stderr
 receipt=json.loads((home/'paper3-r11-operational-repair/CURRENT.json').read_text());package=Path(receipt['package']);manifest=json.loads((package/'MANIFEST.json').read_text())
 for name,item in manifest['files'].items():assert build_delivery.sha((package/name).read_bytes())==item['sha256']
 assert receipt['jobs_submitted_during_install']==0 and receipt['scientific_plan_changed'] is False


def test_repair_submit_is_an_explicit_forwarded_dispatch_only_fixture(tmp_path):
 source,record=fixture(tmp_path);entry=source/'delivery'/record['entry']
 result=subprocess.run([sys.executable,'-I','-S','-B',str(entry),'--home',str(tmp_path/'home'),'--python',sys.executable,'--repair','--submit','--plan','/fixture/plan.json'],capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 route=json.loads(result.stdout.splitlines()[-1])['software_fixture_route']
 assert route[0]=='start' and route[route.index('--plan')+1]=='/fixture/plan.json' and '--submit' in route
 refused=subprocess.run([sys.executable,'-I','-S','-B',str(entry),'--home',str(tmp_path/'another'),'--submit'],capture_output=True,text=True)
 assert refused.returncode!=0 and not (tmp_path/'another').exists()
