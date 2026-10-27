from contextlib import contextmanager
import functools
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import newdiagnostics as diagnostics


class Faults:
    def __init__(self, enabled=False, failure=None):
        self.enabled = enabled
        self.failure = failure
        self.calls = []

    def is_enabled(self):
        return self.enabled

    def enable(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        self.enabled = True


def fixture():
    calls = []
    @contextmanager
    def active(root, action, *, traceback_interval=300, label=None):
        calls.append(('enter', traceback_interval, root, action, label))
        try:
            yield 'original-context-result'
        finally:
            calls.append(('exit', traceback_interval))
    return types.SimpleNamespace(active=active), calls


@pytest.mark.parametrize('explicit', [None, 300, 30, 0])
def test_only_optional_native_dumps_are_disabled(explicit):
    startup, calls = fixture(); faults = Faults()
    diagnostics.install(startup, faults=faults)
    options = {} if explicit is None else {'traceback_interval': explicit}
    action = {'kind': 'development', 'key': 'unchanged'}
    with startup.active('/original/root', action, label='preserved', **options) as value:
        assert value == 'original-context-result'
    assert calls == [('enter', 0, '/original/root', action, 'preserved'), ('exit', 0)]
    assert len(faults.calls) == 1
    assert faults.calls[0]['all_threads'] is True


def test_original_exception_cleanup_and_wrapper_restoration():
    startup, calls = fixture(); original = startup.active
    diagnostics.install(startup, faults=Faults(True))
    with pytest.raises(ValueError, match='scientific failure retained'):
        with startup.active('root', {'key': 'action'}):
            raise ValueError('scientific failure retained')
    assert calls[-1] == ('exit', 0)
    assert diagnostics.restore(startup) is True
    assert startup.active is original
    assert diagnostics.restore(startup) is False


def test_idempotent_install_even_if_followed_by_another_normal_wrapper():
    startup, calls = fixture(); faults = Faults()
    diagnostics.install(startup, faults=faults); first = startup.active
    @functools.wraps(first)
    def later(*a, **kw):
        return first(*a, **kw)
    startup.active = later
    record = diagnostics.install(startup, faults=faults)
    assert record['status'] == 'ALREADY_INSTALLED'
    assert startup.active is later and len(faults.calls) == 1
    assert diagnostics.restore(startup) is False
    with startup.active('root', {}):
        pass
    assert calls[0][1] == 0


@pytest.mark.parametrize('error', [OSError('closed log'), ValueError('bad descriptor'), RuntimeError('unavailable')])
def test_diagnostic_failure_does_not_block_original_work(error):
    startup, calls = fixture()
    record = diagnostics.install(startup, faults=Faults(failure=error))
    assert record['diagnostic_error'] == type(error).__name__
    assert record['fatal_fault_reporting'] is False
    with startup.active('root', {}):
        pass
    assert calls[-1] == ('exit', 0)


def test_real_original_context_no_native_timer_or_signal_change(tmp_path, monkeypatch):
    source = (ROOT/'vendor/action_startup.py')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('original_r1_startup_diag_fixture', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module.faulthandler, 'dump_traceback_later',
        lambda *a, **k: pytest.fail('periodic native dump was armed'))
    monkeypatch.setattr(module.faulthandler, 'cancel_dump_traceback_later',
        lambda *a, **k: pytest.fail('no native timer should have been armed'))
    records = []
    diagnostics.install(module, faults=Faults(True))
    with module.active(tmp_path, {'kind': 'meta_selection', 'key': 'exact-identity'},
                       emit=lambda event, **fields: records.append((event, fields))):
        records.append(('SCIENCE_FIXTURE_EXECUTED', {}))
    assert [event for event, _ in records] == [
        'R1_STARTUP_SCOPE_ACTIVE', 'SCIENCE_FIXTURE_EXECUTED', 'R1_STARTUP_SCOPE_RETURNED']
    assert records[-1][1]['status'] == 'RETURNED'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_failed_diagnostic_sink_cannot_break_install():
    startup, _ = fixture()
    def failed_emit(*a, **k):
        raise OSError('log sink unavailable')
    record = diagnostics.install(startup, faults=Faults(True), emit=failed_emit)
    assert record['periodic_native_stack_dump'] is False
    assert record['native_crash_root_cause_proven'] is False
