"""Operational regressions: isolated startup, bounded arrays and independent sync."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import subprocess
import sys
import venv
import pytest
import START
import entry
from paper3v2 import monitor
from paper3v2.io import put


def test_entry_loads_venv_packages_without_site_hooks(tmp_path):
    virtual = tmp_path / 'venv'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(virtual)
    site = virtual / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    site.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / 'UNWANTED_STARTUP_HOOK'
    (site / 'sitecustomize.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("bad")\n')
    (site / 'unwanted.pth').write_text('import pathlib; pathlib.Path('+repr(str(marker))+').write_text("bad-pth")\n')
    (site / 'school_dependency.py').write_text('VALUE="existing-venv-package"\n')
    package = tmp_path / 'package'; package.mkdir()
    shutil.copyfile(START.PACKAGE / 'entry.py', package / 'entry.py')
    (package / 'paper3v2').mkdir(); (package / 'paper3v2/__init__.py').write_text('')
    (package / 'paper3v2/probe.py').write_text('import school_dependency\ndef main():\n    print(school_dependency.VALUE)\n    return 0\n')
    result = subprocess.run([str(virtual / 'bin/python'), '-I', '-S', '-B', str(package/'entry.py'), str(tmp_path), 'probe'],
                            capture_output=True, text=True, timeout=20, check=True)
    assert result.stdout.strip() == 'existing-venv-package'
    assert not marker.exists()


def test_expanded_array_parse_ignores_other_experiments():
    arrays, indices = START.active_array_tasks('123_0|our-run\n123_4|our-run\n124_2|our-run\n125_7|other-run\n', 'our-run')
    assert arrays == {'123','124'} and indices == {0,2,4}


def test_partial_retry_is_queued_without_exceeding_global_gpu_cap(tmp_path, monkeypatch):
    root = tmp_path / 'project'; root.mkdir()
    output = tmp_path / 'output'
    plan = START.make_plan(root, output)
    marker = START.job_marker(plan)
    put(output/plan['tasks'][1]['key']/'result.json', {'status':'COMPLETE'})
    queue = ['321_0|'+marker, '321_2|'+marker]
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == 'squeue':
            assert '--array' in command and '%i|%k' in command
            return SimpleNamespace(returncode=0, stdout='\n'.join(queue)+'\n', stderr='')
        assert command[0] == 'sbatch'
        return SimpleNamespace(returncode=0, stdout='322\n', stderr='')
    launched = []
    monkeypatch.setattr(START.subprocess, 'run', fake_run)
    monkeypatch.setattr(START.subprocess, 'Popen', lambda command, **kwargs: launched.append(command))
    monkeypatch.setattr(sys, 'argv', ['START.py','--root',str(root),'--output',str(output),'--python',sys.executable,'--submit'])
    START.main()
    submit = next(c for c in calls if c[0]=='sbatch')
    assert '--dependency=afterany:321' in submit
    actual = next(c for c in submit if c.startswith('--array='))
    assert actual == '--array='+','.join(map(str,range(3,len(plan['tasks']))))+'%3'
    assert '-S' in launched[0]
    # A repeated invocation sees both arrays and schedules no duplicate element.
    queue += [f'322_{i}|'+marker for i in range(3,len(plan['tasks']))]
    START.main()
    assert len([c for c in calls if c[0]=='sbatch']) == 1


def test_local_sync_io_failure_does_not_terminate_observation(tmp_path, monkeypatch):
    plan=START.make_plan(tmp_path,tmp_path)
    class FakeGitHub:
        published=[]
        def ensure_branch(self): pass
        def publish(self,v): self.published.append(v)
    gh=FakeGitHub()
    def fail(*args,**kwargs): raise OSError('transient shared filesystem failure')
    monkeypatch.setattr(monitor,'put',fail)
    _, ready, finished = monitor.cycle(plan, None, gh, False)
    assert ready and not finished and len(gh.published)==1


def test_report_failure_is_retried_after_successful_upload(tmp_path, monkeypatch):
    from paper3v2 import report
    plan=START.make_plan(tmp_path,tmp_path)
    for task in plan['tasks']:
        put(tmp_path/task['key']/'result.json', {'status':'COMPLETE'})
    class FakeGitHub:
        def ensure_branch(self): pass
        def publish(self,v): pass
    def failing_report(plan): raise OSError('transient result read failure')
    monkeypatch.setattr(report,'aggregate',failing_report)
    gh, ready, finished = monitor.cycle(plan, None, FakeGitHub(), False)
    assert ready and not finished
    monkeypatch.setattr(report,'aggregate',lambda p:{'status':'COMPLETE','comparisons':[]})
    assert monitor.cycle(plan, None, gh, ready)[2]


def test_partial_report_refreshes_on_completed_count_and_retries(tmp_path, monkeypatch):
    from paper3v2 import report
    plan = START.make_plan(tmp_path, tmp_path)
    cache, calls, published = {}, [], []
    class FakeGitHub:
        def ensure_branch(self): pass
        def publish(self, value): published.append(value)
    gh = FakeGitHub()
    failures = {1}
    def aggregate(plan):
        count = monitor.collect(plan)['complete_tasks']
        calls.append(count)
        if count in failures:
            failures.remove(count)
            raise OSError('transient result read')
        return {'status': 'PARTIAL', 'complete_tasks': count, 'total_tasks': len(plan['tasks']),
                'budget_comparisons': [{'candidate': 'dual', 'baseline': 'P', 'delta_ba_pp': count}]}
    monkeypatch.setattr(report, 'aggregate', aggregate)
    assert not monitor.cycle(plan, None, gh, False, cache)[2]
    assert not monitor.cycle(plan, None, gh, True, cache)[2]
    assert calls == [0]
    put(tmp_path/plan['tasks'][0]['key']/'result.json', {'status': 'COMPLETE'})
    assert not monitor.cycle(plan, None, gh, True, cache)[2]
    assert published[-1]['complete_tasks'] == 1
    assert published[-1]['report_stale']
    assert published[-1]['report_error_type'] == 'OSError'
    assert not monitor.cycle(plan, None, gh, True, cache)[2]
    assert calls == [0, 1, 1]
    assert published[-1]['report_complete_tasks'] == 1
    assert not published[-1]['report_stale']
    assert published[-1]['budget_comparisons'][0]['delta_ba_pp'] == 1
    assert not monitor.cycle(plan, None, gh, True, cache)[2]
    assert calls == [0, 1, 1]


@pytest.mark.parametrize('status_path', [None, 'monitoring/paper3-followup-v2/status.json'])
def test_github_uses_separate_configured_status_path(status_path):
    config = {'repo': 'owner/project', 'branch': 'progress'}
    if status_path is not None:
        config['status_path'] = status_path
    github = monitor.GitHub(config)
    calls = []
    def api(path, body=None, method=None):
        calls.append((path, body, method))
        return {'sha': 'prior'}
    github.api = api
    github.publish({'complete_tasks': 1})
    expected = 'repos/owner/project/contents/' + (status_path or 'monitoring/paper3-followup/status.json')
    assert calls[0][0] == expected + '?ref=progress'
    assert calls[1][0] == expected and calls[1][2] == 'PUT'
    assert calls[1][1]['sha'] == 'prior'


def test_published_report_has_bounded_aggregate_tables_without_trial_data():
    row = {'candidate': 'dual', 'baseline': 'P', 'delta_ba_pp': 1.2, 'ci95_pp': [-.1, 2.5],
           'n_participants': 10, 'labels': [0, 1], 'predictions': [0, 0],
           'probabilities': [[.8, .2], [.6, .4]], 'event_uids': ['trial-a', 'trial-b'],
           'participant_id': 'private-person',
           'per_seed': [{'seed': 5, 'delta_ba_pp': 1.3, 'labels': [0, 1]}],
           'secondary_tradeoffs': {'negative_flip_rate': {'delta': -.2, 'ci95': [-.4, .1]},
                                   'labels': {'delta': 100}},
           'change_from_own_clean': {'delta_ba_pp': -.5, 'event_uids': ['private-trial']}}
    report = {'status': 'PARTIAL', 'complete_tasks': 12, 'total_tasks': 60,
              'trial_rows': [{'labels': [0, 1]}], **{key: [row] for key in monitor.REPORT_TABLES}}
    summary = monitor.report_summary(report)
    assert summary['report_complete_tasks'] == 12
    for key in monitor.REPORT_TABLES:
        assert summary[key][0]['delta_ba_pp'] == 1.2
        assert summary[key][0]['secondary_tradeoffs']['negative_flip_rate']['delta'] == -.2
        assert summary[key][0]['change_from_own_clean']['delta_ba_pp'] == -.5
    encoded = json.dumps(summary)
    for excluded in ('labels', 'predictions', 'probabilities', 'event_uids', 'participant_id', 'trial_rows'):
        assert excluded not in encoded
    assert len(encoded) < 10000
    report['diagnostics'] = [row] * 250
    bounded = monitor.report_summary(report)
    assert len(bounded['diagnostics']) == 200
    assert bounded['report_table_rows']['diagnostics'] == {'total': 250, 'published': 200}


def test_optional_corrupt_cache_falls_back_to_shared_site(tmp_path):
    (tmp_path/'bad').mkdir();(tmp_path/'bad/manifest.json').write_text('[]')
    assert entry.reuse_node_runtime(tmp_path) == []


def test_explicit_termination_saves_but_does_not_requeue(tmp_path):
    import os, signal
    bash = Path('/workspace/paper3-bash42-portability/bash-4.2/bash')
    if not bash.exists(): pytest.skip('Bash 4.2 fixture not installed')
    (tmp_path/'entry.py').write_text('import signal,time,sys\nsignal.signal(signal.SIGUSR1,lambda *_:sys.exit(75))\nprint("READY",flush=True)\nwhile True: time.sleep(.1)\n')
    control=tmp_path/'scontrol';control.write_text('#!/bin/bash\nprintf "%s\\n" "$*" > "$PAPER3_TEST_REQUEUE"\n');control.chmod(0o700)
    env={**os.environ,'PATH':str(tmp_path)+os.pathsep+os.environ['PATH'],
         'SLURM_ARRAY_TASK_ID':'0','SLURM_JOB_ID':'123','PAPER3_TEST_REQUEUE':str(tmp_path/'requeue')}
    p=subprocess.Popen([str(bash),str(START.PACKAGE/'train.sbatch'),str(tmp_path),sys.executable,str(tmp_path),'unused'],
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    try:
        assert p.stdout.readline().startswith('TRAIN_PROCESS_STARTED')
        assert p.stdout.readline().strip()=='READY'
        p.send_signal(signal.SIGTERM)
        output=p.communicate(timeout=10)[0]
        assert p.returncode==75 and 'resume explicitly' in output
        assert not (tmp_path/'requeue').exists()
    finally:
        if p.poll() is None:p.kill();p.wait()
