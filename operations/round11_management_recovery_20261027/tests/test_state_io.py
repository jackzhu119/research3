from __future__ import annotations
import errno
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import state_io as s


def test_required_read_allocates_actual_bytes_not_maximum(tmp_path):
    target = tmp_path/'tasks.json'; target.write_bytes(b'{"tasks":{}}')
    requests = []
    class Reader:
        def __init__(self): self.handle = target.open('rb')
        def __enter__(self): return self
        def __exit__(self, *args): self.handle.close()
        def fileno(self): return self.handle.fileno()
        def read(self, size): requests.append(size); return self.handle.read(size)
    value = s.required_json(target, opener=lambda *_:Reader())
    assert value == {'tasks':{}}
    assert max(requests) == target.stat().st_size+1
    assert all(size <= 13 for size in requests)


def test_transient_file_visibility_and_truncated_json_retry_without_rewrite(tmp_path):
    target = tmp_path/'tasks.json'; original = b'{"recipe_sha256":"fixed","tasks":{"a":{"attempt":9}}}'
    target.write_bytes(original)
    calls = []
    def opener(path, mode):
        calls.append(path)
        if len(calls) == 1: raise OSError(errno.ESTALE, 'temporary stale inode')
        return Path.open(path, mode)
    assert s.required_json(target, opener=opener, sleep=lambda _:None)['tasks']['a']['attempt'] == 9
    assert len(calls) == 2 and target.read_bytes() == original


def test_malformed_then_recovered_json_is_real_observation(tmp_path):
    target = tmp_path/'tasks.json'; target.write_bytes(b'{')
    waits = []
    def repair(delay):
        waits.append(delay)
        target.write_text('{"tasks":{"kept":{"failure_history":[1,2]}}}')
    value = s.required_json(target, sleep=repair)
    assert value['tasks']['kept']['failure_history'] == [1,2]
    assert len(waits) == 1


@pytest.mark.parametrize('contents', [None, b'{', b'null', b'[]', b''])
def test_required_missing_or_invalid_never_becomes_empty_state(tmp_path, contents):
    target = tmp_path/'tasks.json'
    if contents is not None: target.write_bytes(contents)
    with pytest.raises(s.StateUnavailable) as caught:
        s.required_json(target, attempts=3, sleep=lambda _:None)
    assert caught.value.attempts == 3
    assert not target.exists() if contents is None else target.read_bytes() == contents


def test_oversize_state_stays_unavailable_without_payload_allocation(tmp_path):
    target = tmp_path/'tasks.json'; target.write_bytes(b'{}' + b' '*100)
    with pytest.raises(s.StateUnavailable): s.required_json(target, max_bytes=4, attempts=1)
    assert target.stat().st_size == 102
