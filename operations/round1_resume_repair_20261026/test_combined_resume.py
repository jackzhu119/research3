"""Native CPU checkpoint + real operational adapters; never an A100/EEG fit.

The framework/driver and cross-boot journal values reproduce the user-reported
administrative mismatch. GPU identities and the two-value numerical model are
explicit software fixtures; no hardware equivalence measurement is inferred.
"""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_clock_recovery import (Clock, Sampler, actual_cross_node_state, meta,
                                 native_module, score, seed, step, write)
from p3work.recovery_budget import WallBudgetBlocked
import r1_clock_recovery as clock_repair
import r1_driver_compat as driver_repair


def external_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profiles():
    historical = dict(python='3.12.3', torch='2.5.1+cu121', cuda='12.1', cudnn=90100,
        device_name='NVIDIA A100-SXM4-80GB', device_uuid='GPU-11111111-2222-3333-4444-555555555555',
        device_total_memory_bytes=80 * 1024**3, device_compute_capability=[8, 0],
        driver_versions=['550.54.14'] * 8, full_unroll=True, autocast=False, tf32=False,
        float32_matmul_precision='highest', deterministic_algorithms=True,
        cublas_workspace_config=':4096:8')
    actual = dict(deepcopy(historical), device_name='NVIDIA A100-PCIE-40GB',
        device_uuid='GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        device_total_memory_bytes=40 * 1024**3, driver_versions=['535.104.12'] * 5)
    return historical, actual


def test_real_identity_reconcile_then_verified_native_clock_resume_preserves_checkpoint(
        tmp_path, native_module, monkeypatch):
    module = native_module
    historical, actual = profiles()
    job = module.MetaJob('P', 8, .35, phase='engineering')
    identity = dict(historical, job=asdict(job), code_sha256='a' * 64,
                    scope='SOFTWARE_TEST_DOUBLE_NOT_A_GPU_OBSERVATION')
    actual_candidate = dict(actual, job=asdict(job), code_sha256='a' * 64,
                            scope='SOFTWARE_TEST_DOUBLE_NOT_A_GPU_OBSERVATION')
    original_identity, original_candidate = deepcopy(identity), deepcopy(actual_candidate)
    output = tmp_path / 'runs/meta' / job.identity
    write(output / 'identity.json', identity)
    identity_bytes = (output / 'identity.json').read_bytes()

    attempts = []
    seed()
    def interrupted(value, episodes):
        attempts.append(1)
        if len(attempts) == 8:
            raise KeyboardInterrupt('SOFTWARE_FIXTURE interrupted native outer loop')
        return step(value, episodes)

    with pytest.raises(KeyboardInterrupt):
        module.run_outer_loop(parameter=torch.zeros(2, dtype=torch.float64, requires_grad=True),
            sampler=Sampler(), step=interrupted, evaluate=score, identity=identity,
            output=output, total_updates=100, max_wall_seconds=1330200, **Clock().arguments())
    latest_path = output / 'latest_checkpoint.json'
    latest_bytes = latest_path.read_bytes()
    latest = json.loads(latest_bytes)
    checkpoint = output / latest['path']
    checkpoint_bytes = checkpoint.read_bytes()
    sidecar_bytes = checkpoint.with_suffix(checkpoint.suffix + '.sha256.json').read_bytes()
    journal_state = actual_cross_node_state(clock_repair._digest(identity))
    write(output / 'wall_budget.json', journal_state)

    # This is the same operational install order used by the R1 bridge: native
    # runtime-transfer hooks, then the audited hardware adapter, then the narrow
    # driver compatibility policy. No native source function is recompiled here.
    transfer = external_module('combined_native_transfer',
        '/workspace/paper3-continuous-repair/runtime_transfer.py')
    audited = external_module('combined_audited_runtime',
        '/workspace/paper3-r1-hardware-fix/audited_runtime.py')
    payload = tmp_path / 'software-fixture-operational-payload.json'
    write(payload, {'scope': 'SOFTWARE_FIXTURE_SOURCE_BINDING',
                    'files': [transfer._file(transfer.__file__)]})
    monkeypatch.setenv('PAPER3_RUNTIME_TRANSFER_MANIFEST', str(payload))
    audited.install(tmp_path, transfer_module=transfer)
    with pytest.raises(ValueError, match='Software, code, data, numerical policy or scientific limits changed'):
        transfer.reconcile_identity(tmp_path, output, actual_candidate, 'meta')
    assert not (tmp_path / audited.AUDIT).exists()
    assert driver_repair.install(tmp_path, audited_module=audited)['status'] == 'INSTALLED'
    reconciled = transfer.reconcile_identity(tmp_path, output, actual_candidate, 'meta')
    assert reconciled == identity == original_identity
    assert actual_candidate == original_candidate
    assert (output / 'identity.json').read_bytes() == identity_bytes
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert checkpoint.with_suffix(checkpoint.suffix + '.sha256.json').read_bytes() == sidecar_bytes
    assert latest_path.read_bytes() == latest_bytes
    transfers = json.loads((tmp_path / audited.AUDIT).read_text())['transfers']
    assert len(transfers) == 1
    record = next(iter(transfers.values()))
    assert record['historical_runtime'] == original_identity
    assert record['actual_runtime'] == original_candidate
    assert record['policy']['driver_transition']['historical_driver_versions'] == ['550.54.14'] * 8
    assert record['policy']['driver_transition']['actual_driver_versions'] == ['535.104.12'] * 5
    assert record['policy']['driver_transition']['bitwise_equivalence_or_equal_speed_claimed'] is False

    call = dict(parameter=torch.zeros(2, dtype=torch.float64, requires_grad=True),
        sampler=Sampler(), step=step, evaluate=score, identity=reconciled,
        output=output, total_updates=100, max_wall_seconds=1330200,
        resume=checkpoint, **Clock().arguments())
    with pytest.raises(WallBudgetBlocked):
        module.run_outer_loop(**call)
    assert checkpoint.read_bytes() == checkpoint_bytes

    native_loop = module.run_outer_loop
    seed()
    expected, _ = native_loop(parameter=call['parameter'], sampler=Sampler(), step=step,
        evaluate=score, identity=identity, output=tmp_path / 'separate-software-baseline',
        total_updates=100, max_wall_seconds=1330200, **Clock().arguments())
    assert clock_repair.install(tmp_path)['status'] == 'INSTALLED'
    call['sampler'] = Sampler()
    observed, result = module.run_outer_loop(**call)
    assert torch.equal(observed, expected)
    assert result['status'] == 'UPDATE_BUDGET_COMPLETED'
    assert result['outer_updates_completed'] == 100
    assert result['validation_checkpoints'] == 2
    assert call['sampler'].cursor == 400
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert checkpoint.with_suffix(checkpoint.suffix + '.sha256.json').read_bytes() == sidecar_bytes
    assert (output / 'identity.json').read_bytes() == identity_bytes
    saved = json.loads((output / 'wall_budget.json').read_text())
    recovery = saved['authorized_clock_recoveries'][0]
    assert recovery['original_clock_block'] == journal_state['clock_block']
    assert recovery['original_unsettled_interval'] == journal_state['interval']
    assert recovery['verified_checkpoint']['outer_cursor'] == 7
    assert saved['durable_wall_seconds'] >= journal_state['durable_wall_seconds']
    assert saved['max_wall_seconds'] == journal_state['max_wall_seconds']
    assert recovery['unknown_additional_elapsed_seconds'] is None
    assert saved['clock_history_qualification']['exact_cumulative_elapsed_claimed'] is False
