import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE=Path(__file__).resolve().parent
PYTHON='/tmp/paper3-r8-compat/bin/python'
spec=importlib.util.spec_from_file_location('resolver',HERE/'source/cache_resolution.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fake_package(root, *, include_source=False):
    root=Path(root);wanted={}
    for folder in ('pan2023','weibo'):
        session={}
        for field,suffix in (('X','.npy'),('events','.json'),('schema','.json')):
            content=(folder+'-'+field).encode()
            relative='S/'+field+suffix
            session[field]={'path':relative,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            full=Path('data')/folder/relative
            wanted[full.as_posix()]=content
            if include_source:
                target=root/full;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        manifest=root/'data'/folder/'manifest.json';manifest.parent.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps({'sessions':[session]}))
    return wanted


class Response(io.BytesIO):
    status=200
    headers={}


class DeliveryTests(unittest.TestCase):
    def test_real_cached_prepare_same_recipe_and_no_submit(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);project=d/'project';project.mkdir();home=d/'home';home.mkdir()
            command=[PYTHON,'-I','-S','-B','-u',str(HERE/'START-PAPER3-R9-SMALL.py'),
                '--install-base',str(d/'install'),'--root',str(project),'--cache-home',str(home),
                '--cache-root','/workspace/paper3-round9','--prepare-only']
            answer=subprocess.run(command,capture_output=True,text=True,timeout=60)
            self.assertEqual(answer.returncode,0,answer.stderr+answer.stdout)
            events=[json.loads(s) for s in answer.stdout.splitlines() if s.startswith('{')]
            prepared=next(e for e in events if e.get('event')=='ROUND_9_THREE_JOBS_PREPARED')
            self.assertEqual(prepared['tasks'],1548);self.assertEqual(prepared['jobs'],3)
            self.assertNotIn('BACKGROUND_PROCESSES_STARTED',answer.stdout)
            package=Path(next(e for e in events if e.get('status')=='PACKAGE_READY')['package'])
            receipt=json.loads((package/'LOCAL_DATA_READY.json').read_text())
            self.assertEqual(receipt['arrays'],114)
            self.assertEqual(len(receipt['files']),114)
            for entry in receipt['files']:
                target=package/entry['path']
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),entry['sha256'])
            actual=json.loads((project/'runs/followup_20261025_r9/plan.json').read_text())
            script="import sys,json;sys.path.insert(0,'/workspace/paper3-round9');from paper3r9.plan import build;print(json.dumps(build(sys.argv[1])))"
            expected=json.loads(subprocess.check_output([PYTHON,'-c',script,str(project)],text=True))
            self.assertEqual(actual['recipe_sha256'],expected['recipe_sha256'])
            self.assertEqual(actual['code_sha256'],expected['code_sha256'])
            # A normal repeat checks 114 metadata signatures, not another data
            # scan or internet probe. No PyTorch import at cache resolution.
            with patch.object(m,'exact_bytes',side_effect=AssertionError('unexpected reread')):
                cached=m.ensure_data(package,cache_home=home)
            self.assertEqual(cached['arrays'],114)
            plan_path=project/'runs/followup_20261025_r9/plan.json'
            before=plan_path.read_bytes()
            again=subprocess.run(command,capture_output=True,text=True,timeout=60)
            self.assertEqual(again.returncode,0,again.stderr)
            self.assertEqual(plan_path.read_bytes(),before)

    def test_discover_r8_direct_install_and_no_tree_walk(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);package=d/'package';wanted=fake_package(package)
            old=d/'home/paper3-round8/hash';fake_package(old,include_source=True)
            with patch('os.walk',side_effect=AssertionError('full scan')), patch.object(Path,'rglob',side_effect=AssertionError('full scan')):
                result=m.ensure_data(package,cache_home=d/'home',opener=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('network')))
            self.assertEqual(result['arrays'],6)
            for relative,value in wanted.items():self.assertEqual((package/relative).read_bytes(),value)

    def test_corrupt_cache_does_not_override_good_next_root(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);wanted=fake_package(d/'package')
            fake_package(d/'wrong',include_source=True);fake_package(d/'good',include_source=True)
            for relative in wanted:(d/'wrong'/relative).write_bytes(b'bad')
            result=m.ensure_data(d/'package',cache_home=d/'home',cache_roots=[d/'wrong',d/'good'])
            self.assertTrue(all('good' in r['origin'] for r in result['files']))

    def test_per_file_fallback_uses_only_missing_declared_file(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);wanted=fake_package(d/'package',include_source=True)
            missing=next(iter(wanted));(d/'package'/missing).unlink();requested=[]
            def opener(req,**kwargs):
                requested.append(req.full_url)
                return Response(wanted[missing])
            result=m.ensure_data(d/'package',cache_home=d/'home',data_bases=['https://test.invalid'],opener=opener,max_cycles=1)
            self.assertEqual(len(requested),1);self.assertTrue(requested[0].endswith(missing))
            self.assertEqual(result['arrays'],6)

    def test_archive_fallback_extracts_only_exact_declared_files(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);wanted=fake_package(d/'package')
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w') as archive:
                for rel,value in wanted.items():archive.writestr(rel,value)
                archive.writestr('../not-allowed.txt',b'ignore')
            content=stream.getvalue();sha=hashlib.sha256(content).hexdigest()
            result=m.ensure_data(d/'package',cache_home=d/'home',data_archives=['https://test.invalid/data.zip#sha256='+sha],opener=lambda *a,**k:Response(content),max_cycles=1)
            self.assertEqual(result['arrays'],6);self.assertFalse((d/'not-allowed.txt').exists())
            for rel,value in wanted.items():self.assertEqual((d/'package'/rel).read_bytes(),value)

    def test_archive_network_interrupt_resumes_exact_byte_offset(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);content=os.urandom(600000);sha=hashlib.sha256(content).hexdigest();calls=[]
            class Interrupt(Response):
                def read(self,*args):
                    if self.tell()>=262144:raise OSError('disconnected')
                    return super().read(*args)
            def opener(req,**kwargs):
                calls.append(dict(req.headers))
                if len(calls)==1:return Interrupt(content)
                response=Response(content[262144:]);response.status=206
                response.headers={'Content-Range':'bytes 262144-599999/600000'}
                return response
            spec='https://test.invalid/cache.zip#sha256='+sha
            with self.assertRaises(OSError):m.archive_cache(d,spec,opener=opener)
            result=m.archive_cache(d,spec,opener=opener)
            self.assertEqual(result.read_bytes(),content)
            self.assertEqual(calls[1]['Range'],'bytes=262144-')

    def test_missing_cache_cannot_claim_submission(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);fake_package(d/'package')
            with self.assertRaisesRegex(RuntimeError,'cache-root'):
                m.ensure_data(d/'package',cache_home=d/'home')
            self.assertFalse((d/'package/LOCAL_DATA_READY.json').exists())

    def test_bootstrap_source_only_install_never_requires_data(self):
        with tempfile.TemporaryDirectory() as d:
            answer=subprocess.run([PYTHON,'-I','-S','-B','-u',str(HERE/'START-PAPER3-R9-SMALL.py'),
                '--install-base',str(Path(d)/'install'),'--install-only'],capture_output=True,text=True,timeout=60)
            self.assertEqual(answer.returncode,0,answer.stderr)
            self.assertNotIn('R9_DATA_CACHE_MISSING',answer.stdout)


if __name__=='__main__':unittest.main()
