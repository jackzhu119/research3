from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import types

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_r1_selection_installer_tests', HERE/'install_selection.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def env(tmp_path):
    root, home, package = (tmp_path/x for x in ('project', 'home', 'package'))
    for path in (root, home, package):
        path.mkdir()
    old_cache = home/'old-cache.py'; old_cache.write_text('OLD = True\n')
    source = (
        'from contextlib import contextmanager\n'
        '@contextmanager\n'
        'def active(*args, **kwargs):\n    yield "NATIVE_STARTUP"\n'
        'def _paper3_r1_complete_helper(path, expected):\n'
        '    import hashlib, importlib.util\n'
        '    from pathlib import Path\n'
        '    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected\n'
        '    spec = importlib.util.spec_from_file_location("_hook_fixture_helper", path)\n'
        '    module = importlib.util.module_from_spec(spec)\n'
        '    spec.loader.exec_module(module)\n'
        '    return module\n'
        '_paper3_r1_complete_specs = '+repr({'verified_context_cache.py':
            (str(old_cache), repair.sha(old_cache.read_bytes()))})+'\n').encode()
    helper = package/'selection_cache.py'
    helper.write_text('def install(module, old_cache, root):\n'
                      '    module.selection_installed = (old_cache.OLD, root)\n')
    b5helper=package/'b5_retry_budget.py'
    b5helper.write_text('def install(module, root):\n'
                        '    module.b5_retry_installed = root\n')
    put(package/'MANIFEST.json', {'files':{p.name:{
        'bytes':p.stat().st_size, 'sha256':repair.sha(p.read_bytes())} for p in (helper,b5helper)}})
    directories = [home/x for x in repair.TARGETS]
    for directory in directories:
        directory.mkdir(parents=True)
        (directory/'action_startup.py').write_bytes(source)
        for name in ('worker.py', 'kernel_bridge.py', 'audited_runtime.py', 'scheduler.py', 'START.py'):
            (directory/name).write_text('# original operational source '+name+'\n')
    runtime = root/repair.RUNTIME
    put(runtime/'launch.json', {'package':str(directories[0]),
        'worker_command':['python', 'worker.py', '--package', str(directories[1])]})
    put(runtime/'scheduler-state.json', {'slots':{'0':{'attempts':[
        {'command':['python', 'worker.py', '--package', str(directories[0])]}]}}})
    put(runtime/'claims.json', {'selection-B2':{'token':'current-B2', 'pid':100}})
    put(runtime/'progress.json', {'status':'RUNNING', 'phase':'selection-B2'})
    for name in ('development_evaluation', 'classical_training', 'method_registry', 'meta_training'):
        path=root/'p3work'/(name+'.py');path.parent.mkdir(exist_ok=True)
        path.write_text('# scientific source '+name+'\n')
    return types.SimpleNamespace(root=root, home=home, package=package, source=source,
        base_sha=repair.sha(source), targets=[p/'action_startup.py' for p in directories],
        old_cache=old_cache, helper=helper, b5helper=b5helper, runtime=runtime)


def apply(e):
    return repair.apply(e.root, e.home, e.package, expected_base_sha=e.base_sha)


def inventory(e):
    allowed = set(e.targets)
    return {str(p):repair.sha(p.read_bytes()) for base in (e.root,e.home,e.package)
            for p in base.rglob('*') if p.is_file() and p not in allowed
            and not p.is_relative_to(e.home/'paper3-r1-selection-continuation')}


def test_applies_both_retained_hooks_and_nothing_else(env, monkeypatch):
    before=inventory(env); sent=[]
    monkeypatch.setattr(repair.os,'kill',lambda *args:sent.append(args))
    receipt=apply(env)
    assert receipt['activation']=='FUTURE_ACTIONS_ONLY'
    assert receipt['signals_sent']==receipt['jobs_submitted']==receipt['scientific_files_changed']==0
    assert receipt['existing_children_modified'] is False
    assert inventory(env)==before and sent==[]
    assert all(row['changed'] for row in receipt['operational_files'])
    for path in env.targets:
        assert repair.strip_own_block(path.read_text()).encode()==env.source
        assert path.read_text().count(repair.MARKER)==2
    assert json.loads((env.runtime/'claims.json').read_text())['selection-B2']['token']=='current-B2'


def test_reapply_is_byte_identical_no_duplicate_hook(env):
    first=apply(env); before={p:p.read_bytes() for p in env.targets}
    second=apply(env)
    assert all(not row['changed'] for row in second['operational_files'])
    assert before=={p:p.read_bytes() for p in env.targets}
    assert first['helpers']==second['helpers']


def test_future_import_uses_existing_loader_and_old_cache(env):
    apply(env)
    module=types.ModuleType('_future_r1_startup_fixture')
    sys.modules[module.__name__]=module
    try:
        exec(compile(env.targets[0].read_bytes(),str(env.targets[0]),'exec'),module.__dict__)
        assert module.selection_installed==(True,str(env.root))
        assert module.b5_retry_installed==str(env.root)
        with module.active() as result:
            assert result=='NATIVE_STARTUP'
    finally:
        sys.modules.pop(module.__name__,None)


@pytest.mark.parametrize('suffix', [
    '\n# '+repair.MARKER+'\n',
    '\n# END_'+repair.MARKER+'\n',
    '\n# '+repair.MARKER+'\n# END_'+repair.MARKER+'\n\n# third-party tail\n'])
def test_incomplete_or_nonterminal_own_block_is_refused(env,suffix):
    env.targets[1].write_bytes(env.source+suffix.encode())
    with pytest.raises(ValueError,match='OWN_BLOCK'):
        apply(env)
    assert env.targets[0].read_bytes()==env.source


def test_duplicate_own_blocks_are_refused_before_writes(env):
    fixed=repair.patch_source(env.source,repair.checked_helpers(env.package),env.root,
                              expected_base_sha=env.base_sha)
    trailer=fixed[len(env.source):]
    env.targets[0].write_bytes(fixed+trailer)
    with pytest.raises(ValueError,match='DUPLICATE_OWN_BLOCK'):
        apply(env)
    assert env.targets[1].read_bytes()==env.source


def test_unknown_base_and_third_party_edit_are_refused(env):
    env.targets[1].write_bytes(env.source+b'# unrecognized edit\n')
    with pytest.raises(ValueError,match='UNKNOWN_STARTUP_BASE'):
        apply(env)
    assert env.targets[0].read_bytes()==env.source


def test_helper_manifest_hash_mismatch_refuses_all_target_writes(env):
    env.helper.write_text('# corrupt helper\n')
    with pytest.raises(ValueError,match='HELPER_SHA256'):
        apply(env)
    assert all(path.read_bytes()==env.source for path in env.targets)


def test_existing_old_cache_hash_mismatch_refuses_before_writes(env):
    env.old_cache.write_text('# changed old validator\n')
    with pytest.raises(ValueError,match='EXISTING_CACHE_SHA256'):
        apply(env)
    assert all(path.read_bytes()==env.source for path in env.targets)


def test_b5_helper_manifest_hash_mismatch_refuses_before_writes(env):
    env.b5helper.write_text('# corrupt B5 budget helper\n')
    with pytest.raises(ValueError,match='HELPER_SHA256'):
        apply(env)
    assert all(path.read_bytes()==env.source for path in env.targets)


def test_default_production_base_cannot_be_overridden_by_manifest(env):
    put(env.package/'MANIFEST.json',{'expected_base_sha256':env.base_sha,
        'files':json.loads((env.package/'MANIFEST.json').read_text())['files']})
    with pytest.raises(ValueError,match='UNKNOWN_STARTUP_BASE'):
        repair.apply(env.root,env.home,env.package)
    assert all(path.read_bytes()==env.source for path in env.targets)


def test_helper_changed_after_first_write_rolls_back_own_hook(env,monkeypatch):
    native=repair.atomic
    def changing(path,data,**kwargs):
        result=native(path,data,**kwargs)
        if Path(path)==env.targets[0] and data!=env.source:
            env.b5helper.write_text('# concurrently changed helper\n')
        return result
    monkeypatch.setattr(repair,'atomic',changing)
    with pytest.raises(ValueError,match='HELPER_SHA256'):
        apply(env)
    assert all(path.read_bytes()==env.source for path in env.targets)


@pytest.mark.parametrize('which',['target','helper','target_parent'])
def test_symlink_files_or_components_are_refused(env,which,tmp_path):
    if which=='target':
        saved=tmp_path/'saved.py';saved.write_bytes(env.source)
        env.targets[0].unlink();env.targets[0].symlink_to(saved)
    elif which=='helper':
        saved=tmp_path/'saved-helper.py';saved.write_bytes(env.helper.read_bytes())
        env.helper.unlink();env.helper.symlink_to(saved)
    else:
        original=env.targets[0].parent
        saved=original.with_name('saved-round1');original.rename(saved);original.symlink_to(saved,target_is_directory=True)
    with pytest.raises(ValueError,match='REGULAR|SYMLINK'):
        apply(env)


def test_partial_apply_failure_rolls_back_only_own_first_write(env,monkeypatch):
    native=repair.atomic
    def failing(path,data,**kwargs):
        if Path(path)==env.targets[1] and data!=env.source:
            raise OSError('fixture write failed')
        return native(path,data,**kwargs)
    monkeypatch.setattr(repair,'atomic',failing)
    with pytest.raises(OSError,match='fixture write'):
        apply(env)
    assert all(path.read_bytes()==env.source for path in env.targets)


def test_rollback_preserves_third_party_later_write(env,monkeypatch):
    native=repair.atomic; third_party=env.source+b'# third party concurrent change\n'
    def failing(path,data,**kwargs):
        if Path(path)==env.targets[1] and data!=env.source:
            env.targets[0].write_bytes(third_party)
            raise OSError('fixture later failure')
        return native(path,data,**kwargs)
    monkeypatch.setattr(repair,'atomic',failing)
    with pytest.raises(OSError,match='fixture later'):
        apply(env)
    assert env.targets[0].read_bytes()==third_party
    assert env.targets[1].read_bytes()==env.source


def test_read_before_replace_prevents_overwriting_concurrent_source(env,monkeypatch):
    native=repair.atomic; other=env.source+b'# changed after preparation\n'
    def racing(path,data,**kwargs):
        if Path(path)==env.targets[0] and data!=env.source:
            env.targets[0].write_bytes(other)
        return native(path,data,**kwargs)
    monkeypatch.setattr(repair,'atomic',racing)
    with pytest.raises(RuntimeError,match='SOURCE_CHANGED_BEFORE_REPLACE'):
        apply(env)
    assert env.targets[0].read_bytes()==other
    assert env.targets[1].read_bytes()==env.source


def test_atomic_shared_filesystem_rename_error_after_success_is_reconciled(env,monkeypatch):
    native=repair.os.replace; attempts=[]
    def ambiguous(source,target):
        native(source,target);attempts.append(target)
        if len(attempts)==1:
            raise OSError(5,'fixture server applied rename but returned EIO')
    monkeypatch.setattr(repair.os,'replace',ambiguous)
    receipt=apply(env)
    assert receipt['activation']=='FUTURE_ACTIONS_ONLY'
    assert all(path.read_bytes()!=env.source for path in env.targets)
    assert not list(env.home.rglob('.r1-selection-stage-*'))


def test_known_fallbacks_only_and_unknown_active_target_refused(env,tmp_path):
    extra=env.home/'unknown-active';extra.mkdir()
    (extra/'action_startup.py').write_bytes(env.source)
    put(env.runtime/'launch.json',{'package':str(extra)})
    with pytest.raises(ValueError,match='UNRECOGNIZED_ACTIVE'):
        apply(env)
    assert all(path.read_bytes()==env.source for path in env.targets)


def test_private_backup_directory_symlink_refused_before_staging(env,tmp_path):
    base=env.home/'paper3-r1-selection-continuation';base.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (base/'backups').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='ATOMIC_PARENT_IS_SYMLINK'):
        apply(env)
    assert list(outside.iterdir())==[]
    assert all(path.read_bytes()==env.source for path in env.targets)


def test_actual_helpers_install_without_eager_budget_or_scientific_writes(env):
    selection=HERE/'selection_cache.py'
    b5=HERE/'b5_retry_budget.py'
    if not selection.is_file() or not b5.is_file():
        pytest.skip('Actual helper sources are not available in this fixture environment')
    env.helper.write_bytes(selection.read_bytes());env.b5helper.write_bytes(b5.read_bytes())
    put(env.package/'MANIFEST.json', {'files':{p.name:{
        'bytes':p.stat().st_size,'sha256':repair.sha(p.read_bytes())}
        for p in (env.helper,env.b5helper)}})
    apply(env)
    before=inventory(env)
    module=types.ModuleType('_future_r1_actual_helpers_fixture')
    sys.modules[module.__name__]=module
    try:
        exec(compile(env.targets[0].read_bytes(),str(env.targets[0]),'exec'),module.__dict__)
        assert hasattr(module,'_paper3_r1_selection_source_snapshot_v1')
        assert inventory(env)==before
    finally:
        sys.modules.pop(module.__name__,None)


def test_receipt_failure_reports_applied_hooks_truthfully_and_rerun_recovers(env,monkeypatch):
    native=repair.atomic
    def failing(path,data,**kwargs):
        if Path(path).name.startswith('applied-'):
            raise OSError('fixture telemetry unavailable')
        return native(path,data,**kwargs)
    monkeypatch.setattr(repair,'atomic',failing)
    receipt=apply(env)
    assert receipt['status']=='R1_FUTURE_STARTUP_CONTINUATION_INSTALLED'
    assert receipt['receipt_persisted'] is False
    assert receipt['receipt_error_type']=='OSError'
    assert all(path.read_bytes()!=env.source for path in env.targets)
    monkeypatch.setattr(repair,'atomic',native)
    assert apply(env)['receipt_persisted'] is True
