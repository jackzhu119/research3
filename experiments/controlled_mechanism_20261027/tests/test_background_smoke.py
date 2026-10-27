"""Sidecar startup uses independent local CPU helpers; no archive RPCs."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper3m12 import background, common


def fixture_plan(tmp_path):
    helper = tmp_path/'local helper with spaces.py'
    helper.write_text('import time\nwhile True: time.sleep(.05)\n')
    path = tmp_path/'plan.json'
    common.atomic_json(path, dict(root=str(tmp_path), output=str(tmp_path/'output'),
        worker_entry=str(helper), python=sys.executable, recipe_sha256='a'*64,
        tasks=[dict(key='fixture', kind='paired_block')], synthetic_fixture=True))
    return path, common.load_plan(path)[1]


def test_real_three_helpers_reuse_pid_and_recover_lost_telemetry_receipt(tmp_path):
    path, plan = fixture_plan(tmp_path)
    owned = {}
    try:
        first = background.ensure_sidecars(path)
        owned = {key: row['pid'] for key, row in first.items()}
        assert len(set(owned.values())) == 3
        assert all(row['status'] == 'HELPER_STARTED' for row in first.values())
        assert all(row['training_controlled'] is False for row in first.values())
        second = background.ensure_sidecars(path)
        assert {key: row['pid'] for key, row in second.items()} == owned
        assert all(row['status'] == 'EXISTING_HELPER_REUSED' for row in second.values())
        (Path(plan['runtime'])/'paper-upload-process.json').unlink()
        third = background.ensure_sidecars(path)
        assert {key: row['pid'] for key, row in third.items()} == owned
        assert third['paper-upload']['status'] == 'EXISTING_HELPER_DISCOVERED'
        assert len(list(Path(plan['runtime']).glob('*-process.json'))) == 3
    finally:
        # Only this test's CPU idle helpers. Production helpers are never sent signals.
        for pid in owned.values():
            try:
                os.kill(pid, signal.SIGTERM)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass


def test_helper_launch_failure_stays_noncontrolling_and_has_compact_record(tmp_path):
    path, plan = fixture_plan(tmp_path)
    calls = []
    def failing(command, **kwargs):
        calls.append(command)
        raise OSError('fixture missing helper interpreter')
    result = background.ensure_sidecars(path, popen=failing)
    assert len(calls) == 3
    assert all(row['status'] == 'HELPER_START_RETRY_REQUIRED' for row in result.values())
    assert all(row['training_controlled'] is False for row in result.values())
    assert not list((Path(plan['output'])/'tasks').glob('*/result.json'))
    assert len(list(Path(plan['runtime']).glob('*-start-error.json'))) == 3
