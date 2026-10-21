import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('paper3_check', Path(__file__).parents[1] / 'CHECK_PAPER3.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def no_gpu(*args, **kwargs):
    return SimpleNamespace(returncode=1, stdout='do not print fixture-secret', stderr='fixture-secret')


def test_count_skips_links_and_partial(tmp_path):
    (tmp_path / 'regular').write_text('x')
    (tmp_path / '.paper3-x.partial').write_text('partial')
    (tmp_path / 'link').symlink_to(tmp_path / 'regular')
    assert check.count_files(tmp_path) == 1


def test_root_link_refused(tmp_path):
    (tmp_path / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        check.count_files(tmp_path / 'link')


def test_process_scope_and_safe_fields(tmp_path):
    workspace = tmp_path / 'workspace'
    cwd = workspace / 'Paper3_OneClick'
    cwd.mkdir(parents=True)
    proc = tmp_path / 'proc'
    row = proc / '321'
    row.mkdir(parents=True)
    (row / 'cwd').symlink_to(cwd, target_is_directory=True)
    (row / 'cmdline').write_bytes(b'python3\0paper3_one_click.py\0fixture-secret\0')
    fields = ['S'] + ['0'] * 40
    fields[11], fields[12] = '40', '2'
    (row / 'stat').write_text('321 (python process) ' + ' '.join(fields))
    assert check.controllers(workspace, proc) == [{'pid': 321, 'state': 'S', 'cpu_ticks': 42}]
    (row / 'cmdline').write_bytes(b'python3\0../../outside/paper3_one_click.py\0')
    assert check.controllers(workspace, proc) == []


def test_collect_whitelist_no_writes(tmp_path):
    work = tmp_path / 'Paper3_OneClick' / ('work_' + check.RELEASE[:16])
    work.mkdir(parents=True)
    (work / 'RUN_PROGRESS.json').write_text(json.dumps({'closure_sha256': check.RELEASE,
        'credential': 'fixture-secret', 'r2_objects': {name: {'status': 'R2_FULL_GET_SHA_VERIFIED',
        'complete_body_read': True, 'get_attempted': True} for name in
        ['ready', 'main_inputs', 'control_inputs', 'main_archive', 'control_archive']}}))
    proc = tmp_path / 'emptyproc'
    proc.mkdir()
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    value = check.collect(tmp_path, proc, no_gpu)
    assert value['full_GET_objects'] == 5
    assert '解包' in value['stage'] and 'fixture-secret' not in check.render(value)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_activity_is_not_completed_claim():
    value = {'observed_at': 'test', 'stage': '解包', 'sdk_files': 2208, 'project_files': 5,
             'full_GET_objects': 5, 'cached_objects': 0, 'controllers': [{'pid': 3, 'state': 'S', 'cpu_ticks': 4}],
             'gpu': {'devices': [[20, 30, 40]], 'compute_processes': [[3, 30]]}}
    older = {**value, 'project_files': 4}
    assert '检测到' in check.render(value, older)
    assert '不能据此认定失败' in check.render(value, value)
    assert '训练已完成' not in check.render(value)


def test_gpu_only_safe_numeric_output():
    def fake(command, **kwargs):
        assert kwargs['timeout'] == 5
        return SimpleNamespace(returncode=0, stdout='1, 2, 3\n' if '--query-gpu=' in command[1] else '33, 44\n')
    assert check.gpu_query(fake) == {'devices': [[1, 2, 3]], 'compute_processes': [[33, 44]]}
