"""Read-only monitoring of tiny synthetic control receipts; never a real job."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hpc import monitor


def write(root, relative, value, mtime=995):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    os.utime(path, (mtime, mtime))
    return path


def scheduler(lines='', code=0):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == 'squeue'
        assert '--user=' + str(os.getuid()) in command and kwargs['timeout'] == 5
        assert kwargs['stdin'] == subprocess.DEVNULL
        return SimpleNamespace(returncode=code, stdout=lines, stderr='never display service stderr')
    run.calls = calls
    return run


def job_line(job='100', stage='train', state='RUNNING', reason='gpu3', dependency='(null)'):
    return '|'.join((job, 'paper3-' + stage, state, '00:01:20', '24', '128G', 'gpu:1', reason, dependency))


def submit(root, job='100', stage='train', **kwargs):
    value = {'status': 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE', 'job_id': job, 'stage': stage}
    value.update(kwargs)
    return write(root, '.paper3-hpc/submissions/' + job + '-abcdef.json', value)


def live(root, job='100', mtime=995, **kwargs):
    value = {'stage': 'train', 'allocation': {'job_id': job, 'cpus': 24},
             'used_cpu_cores': 2.5, 'allocated_CPU_util_percent': 10.4166,
             'sampled_tree_rss_bytes': 3 * 1024**3, 'sampled_tree_peak_rss_bytes': 4 * 1024**3,
             'gpu_observation_available': True, 'owned_process_gpu_memory_mib': 2048,
             'owned_process_gpu_peak_memory_mib': 4096,
             'owned_gpus': [{'device_utilization_percent': 72, 'total_memory_mib': 81920,
                             'owned_process_memory_mib': 2048}]}
    value.update(kwargs)
    return write(root, '.paper3-hpc/jobs/' + job + '/resources.live.json', value, mtime)


def final(root, job='100', mtime=990, **kwargs):
    value = {'stage': 'train', 'allocation': {'job_id': job, 'cpus': 24}, 'actual_exit_code': 0,
             'measurements': {'wall_seconds': 100, 'cpu_seconds': 200,
                 'peak_rss_bytes': 4 * 1024**3, 'gpu_observation_available': False,
                 'owned_process_gpu_peak_memory_mib': 0}}
    value.update(kwargs)
    return write(root, '.paper3-hpc/jobs/' + job + '/resources.json', value, mtime)


def test_failed_fetch_is_stopped_and_never_training_started(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json',
          {'operation': 'fetch', 'status': 'failed', 'error_code': 'R2_GET_FAILED',
           'release_object_GETs': [{'status': 'R2_FULL_GET_SHA_VERIFIED', 'complete_body_read': True},
                                  {'status': 'LOCAL_VERIFIED_CACHE_HIT'}]})
    runner = scheduler()
    snapshot = monitor.collect(tmp_path, runner=runner, now=1000)
    shown = monitor.render(snapshot)
    assert 'STOPPED' in shown and 'R2_GET_FAILED' in shown
    assert '固定对象已核验 2/5' in shown
    assert '完整 GET 1' in shown and '本地 SHA 命中 1' in shown
    assert snapshot['queue_count'] == 0 and snapshot['jobs'] == []
    assert '先按错误修复' in shown
    assert len(runner.calls) == 1


def test_running_data_receipt_alone_never_proves_liveness(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json',
          {'operation': 'fetch', 'status': 'running'}, mtime=400)
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    assert 'UNCONFIRMED' in monitor.render(snapshot)
    assert '不能证明进程仍活跃' in monitor.render(snapshot)
    assert snapshot['data']['age_seconds'] == 600


def test_transfer_progress_is_not_completion_and_storage_errno_is_visible(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json',
          {'operation': 'fetch', 'status': 'failed', 'error_code': 'LOCAL_IO_ENOSPC',
           'active_transfer': {'phase': 'body_read', 'object': 'main_archive',
               'attempt': 1, 'bytes_read': 1048576, 'expected_bytes': 2097152,
               'bucket': 'SECRET_BUCKET', 'endpoint': 'SECRET_ENDPOINT'},
           'failure_detail': {'phase': 'local_write', 'errno': 28, 'errno_name': 'ENOSPC',
                              'raw_message': 'SECRET_ERROR'}})
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    shown = monitor.render(snapshot)
    assert 'STOPPED' in shown and '1.0MiB / 2.0MiB' in shown
    assert '固定对象已核验 0/5' in shown and '不等于校验成功' in shown
    assert 'local_write' in shown and 'ENOSPC' in shown and 'errno=28' in shown
    assert 'SECRET' not in json.dumps(snapshot) and 'SECRET' not in shown


def test_true_file_inspection_counters_do_not_claim_data_ready_or_liveness(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {
        'operation': 'materialize', 'status': 'running',
        'active_transfer': {'phase': 'frozen_file_SHA_verification',
            'inspection': 'after_restore', 'processed_files': 21262, 'total_files': 21262,
            'verified_files': 21260, 'missing_files_count': 2,
            'verified_bytes': 1048576, 'total_bytes': 2097152, 'completed': True,
            'updated_at_utc': '2026-10-21T07:18:15.279981+00:00'}}, mtime=400)
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    shown = monitor.render(snapshot)
    assert snapshot['data']['status'] == 'running' and snapshot['data']['age_seconds'] == 600
    assert snapshot['queue_count'] == 0 and 'UNCONFIRMED' in shown
    assert '恢复后复核' in shown and '本地冻结文件 SHA 核验' in shown
    assert '21262 / 21262' in shown and 'SHA 通过 21260 文件；缺失 2 文件' in shown
    assert '1.0MiB / 2.0MiB' in shown and '累计核验量，非速率' in shown
    assert '整体数据就绪仍以最终回执为准' in shown
    assert snapshot['data']['transfer']['completed'] is True
    assert snapshot['data']['transfer']['updated_at_utc'] == '2026-10-21T07:18:15.279981+00:00'


@pytest.mark.parametrize('phase', ['archive_member_scan', 'archive_header_verification',
    'materializing_verified_archive', 'index_descriptor_verification', 'frozen_scope_selection'])
def test_local_stage_counts_are_not_presented_as_network_transfer_bytes(tmp_path, phase):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {
        'status': 'running', 'active_transfer': {'phase': phase, 'object': 'control',
            'processed_files': 100, 'total_files': 12483, 'completed': False,
            'materialized_files': 40, 'already_present_files': 59, 'excluded_files': 1}})
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    shown = monitor.render(snapshot)
    assert phase in shown and '100 / 12483' in shown
    assert '本包新落盘 40；原有 SHA 命中 59；范围外条目 1' in shown
    assert '本阶段尚未返回' in shown and 'UNCONFIRMED' in shown
    assert 'GET 正文' not in shown


def test_index_receipt_binding_has_its_own_count_scope(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {
        'status': 'running', 'active_transfer': {'phase': 'index_receipt_binding_verification',
            'processed_groups': 1234, 'total_groups': 2657, 'covered_files': 10000,
            'completed': False}})
    shown = monitor.render(monitor.collect(tmp_path, runner=scheduler(), now=1000))
    assert '原恢复收据绑定核验' in shown
    assert '原收据已核验组 1234 / 2657；已绑定文件 10000' in shown
    assert 'UNCONFIRMED' in shown


def test_group_file_count_is_plan_size_not_completed_restore_count(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {
        'operation': 'fetch', 'status': 'running',
        'group_restore_receipts': [{'status': 'restored_verified'}],
        'active_transfer': {'phase': 'original_profile_restore', 'group': 2,
            'groups_total': 1538, 'files': 448, 'io_profile': 'baseline'}})
    shown = monitor.render(monitor.collect(tmp_path, runner=scheduler(), now=1000))
    assert '原权限恢复组 2/1538 | baseline | 本组需恢复 448 文件' in shown
    assert '成功恢复组 1/1' in shown and 'UNCONFIRMED' in shown


def test_offline_pass_does_not_imply_missing_historical_GET_proofs(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {
        'operation': 'materialize', 'status': 'running',
        'active_transfer': {'phase': 'frozen_file_SHA_verification',
            'processed_files': 100, 'total_files': 21262, 'completed': False}})
    shown = monitor.render(monitor.collect(tmp_path, runner=scheduler(), now=1000))
    assert '本次为离线核查，未发起固定对象 GET' in shown
    assert '历史取回证据保留在原 fetch 回执' in shown
    assert '100 / 21262' in shown and 'UNCONFIRMED' in shown


def test_running_Slurm_prepare_takes_precedence_over_old_stage_counters(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {
        'status': 'running', 'active_transfer': {'phase': 'original_profile_restore',
            'group': 1538, 'groups_total': 1538}}, mtime=200)
    snapshot = monitor.collect(tmp_path, runner=scheduler(job_line(stage='prepare')), now=1000)
    shown = monitor.render(snapshot)
    assert snapshot['data']['age_seconds'] == 800
    assert '当前：Slurm 已观察到 RUNNING/COMPLETING' in shown
    assert 'UNCONFIRMED' not in shown
    assert 'prepare 成功' in monitor.next_step(snapshot)


def test_stage_metadata_is_strictly_whitelisted_and_malformed_counters_are_NA(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {
        'status': 'running', 'active_transfer': {'phase': ['SECRET_PHASE'],
            'object': {'key': 'SECRET_OBJECT'}, 'io_profile': 'SECRET_PROFILE',
            'inspection': 'SECRET_SCOPE', 'updated_at_utc': 'SECRET_TIMESTAMP',
            'completed': 'SECRET_COMPLETED', 'processed_files': True, 'total_files': -1,
            'verified_bytes': 10**400, 'verified_files': 'SECRET_COUNTER',
            'raw_exception': 'SECRET_SDK', 'file_path': 'SECRET_PATH'}})
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    shown = monitor.render(snapshot)
    transfer = snapshot['data']['transfer']
    assert transfer['phase'] == transfer['object'] == transfer['inspection'] == 'N/A'
    assert transfer['processed_files'] is transfer['verified_bytes'] is transfer['completed'] is None
    assert 'SECRET' not in json.dumps(snapshot) and 'SECRET' not in shown
    assert 'UNCONFIRMED' in shown


def test_latest_data_uses_mtime_not_attempt_name(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-fff/receipt.json', {'status': 'running'}, mtime=900)
    write(tmp_path, '.paper3-hpc/data/attempt-aaa/receipt.json', {'status': 'failed'}, mtime=999)
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    assert snapshot['data']['status'] == 'failed'


def test_live_owned_resources_display_actual_values(tmp_path):
    live(tmp_path)
    snapshot = monitor.collect(tmp_path, runner=scheduler(job_line()), now=1000)
    resource = snapshot['jobs'][0]['resources']
    assert resource['source'] == 'LIVE' and resource['current'] is True
    assert resource['cpu_cores'] == 2.5 and resource['rss_bytes'] == 3 * 1024**3
    shown = monitor.render(snapshot)
    assert '3.0GiB' in shown and '2.0GiB' in shown and '72.0%' in shown
    assert '设备整体利用率可能含邻居作业' in shown
    assert snapshot['queue_running_count'] == 1


def test_first_CPU_sample_and_unavailable_GPU_are_NA_not_zero(tmp_path):
    live(tmp_path, used_cpu_cores=None, allocated_CPU_util_percent=None,
         gpu_observation_available=False, owned_process_gpu_memory_mib=None,
         owned_process_gpu_peak_memory_mib=0, owned_gpus=[], cpu_sample_status='FIRST_SAMPLE')
    snapshot = monitor.collect(tmp_path, runner=scheduler(job_line()), now=1000)
    shown = monitor.render(snapshot)
    assert '采样占核 N/A' in shown and 'CPU 利用率 N/A' in shown
    assert '本作业 GPU 采样显存 N/A / 峰值 N/A' in shown
    assert 'FIRST_SAMPLE' in shown


def test_old_live_peak_fields_are_not_presented_as_current_CPU_or_RSS(tmp_path):
    live(tmp_path, used_cpu_cores=None, allocated_CPU_util_percent=None,
         sampled_tree_rss_bytes=None, gpu_observation_available=None,
         owned_process_gpu_memory_mib=None, owned_gpus=[], cpu_seconds=None)
    resource = monitor.collect(tmp_path, runner=scheduler(job_line()), now=1000)['jobs'][0]['resources']
    assert resource['cpu_cores'] is None and resource['rss_bytes'] is None
    assert resource['peak_rss_bytes'] == 4 * 1024**3
    assert resource['owned_gpu_memory_mib'] is None and resource['owned_gpu_peak_mib'] == 4096


def test_stale_live_sample_not_current_even_with_running_queue(tmp_path):
    live(tmp_path, mtime=800)
    resource = monitor.collect(tmp_path, runner=scheduler(job_line()), now=1000)['jobs'][0]['resources']
    assert resource['source'] == 'LAST_SAMPLE_NOT_CURRENT'
    assert resource['age_seconds'] == 200 and resource['current'] is False


def test_latest_final_is_historical_not_running_even_if_live_file_remains(tmp_path):
    live(tmp_path, mtime=980)
    final(tmp_path, mtime=995)
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    job = snapshot['jobs'][0]
    assert job['state'] == 'NOT_IN_CURRENT_QUEUE'
    assert job['resources']['source'] == 'FINAL_HISTORY'
    assert job['resources']['cpu_cores'] == 2
    assert snapshot['queue_running_count'] == 0
    assert '平均占核 2.0' in monitor.render(snapshot)


def test_requeue_with_newer_live_chooses_current_observation(tmp_path):
    final(tmp_path, mtime=900)
    live(tmp_path, mtime=995)
    resource = monitor.collect(tmp_path, runner=scheduler(job_line()), now=1000)['jobs'][0]['resources']
    assert resource['source'] == 'LIVE' and resource['actual_exit_code'] is None


def test_pending_dependency_and_resource_history_are_explicit(tmp_path):
    live(tmp_path, mtime=995)
    snapshot = monitor.collect(tmp_path, runner=scheduler(job_line(state='PENDING', reason='(DependencyNeverSatisfied)',
                                                                  dependency='afterok:99(failed)')), now=1000)
    shown = monitor.render(snapshot)
    assert 'DependencyNeverSatisfied' in shown and 'afterok:99(failed)' in shown
    assert snapshot['jobs'][0]['resources']['current'] is False
    assert '等待 Slurm 调度/依赖' in shown


def test_three_latest_actual_jobs_only_and_foreign_names_filtered(tmp_path):
    for number in range(100, 105):
        submit(tmp_path, str(number))
    submit(tmp_path, '999', status='DRY_RUN_NO_JOB_SUBMITTED')
    runner = scheduler(job_line('104') + '\n' + '990|someone-else|RUNNING|1|1|1G|N/A|gpu4|(null)')
    snapshot = monitor.collect(tmp_path, runner=runner, now=1000)
    assert [row['job_id'] for row in snapshot['jobs']] == ['104', '103', '102']
    assert snapshot['queue_count'] == 1


def test_queue_failure_is_unknown_not_no_jobs(tmp_path):
    submit(tmp_path)
    live(tmp_path)
    snapshot = monitor.collect(tmp_path, runner=scheduler(code=1), now=1000)
    assert snapshot['queue_status'] == 'unavailable'
    assert snapshot['jobs'][0]['state'] == 'QUEUE_UNKNOWN'
    assert snapshot['jobs'][0]['resources']['current'] is False
    assert 'UNKNOWN' in monitor.render(snapshot)
    assert '重复提交' in monitor.next_step(snapshot)


def test_queue_timeout_does_not_expose_exception_message(tmp_path):
    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired('DO_NOT_PRINT_SECRET', 5)
    snapshot = monitor.collect(tmp_path, runner=unavailable, now=1000)
    shown = monitor.render(snapshot)
    assert 'DO_NOT_PRINT_SECRET' not in shown and 'UNKNOWN' in shown


def test_campaign_only_known_fields_and_safe_characters_are_rendered(tmp_path):
    write(tmp_path, 'runs/training_campaign_v1/campaign_progress.json',
          {'status': 'RUNNING_LOCAL_USER_STARTED_GPU\033[2J', 'active': {'kind': 'source', 'key': 'B7-20261018\nINJECT'},
           'credential': 'DO_NOT_PRINT_SECRET', 'active_process': {'cmdline': 'DO_NOT_PRINT_PROCESS'}})
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    shown = monitor.render(snapshot)
    assert 'kind=source' in shown and '不证明 fit 仍活跃' in shown
    assert '\033' not in shown and 'DO_NOT_PRINT' not in shown
    assert 'B7-20261018?INJECT' in shown


def test_oversized_and_symlink_receipts_are_skipped_without_payload_reads(tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_text('{"status":"failed"}')
    linked = tmp_path / '.paper3-hpc/data/attempt-abc'
    linked.mkdir(parents=True)
    (linked / 'receipt.json').symlink_to(outside)
    large = tmp_path / '.paper3-hpc/data/attempt-def/receipt.json'
    large.parent.mkdir()
    with large.open('wb') as stream:
        stream.truncate(monitor.MAX_JSON_BYTES + 1)
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    assert snapshot['data'] is None


def test_symlink_directory_and_root_are_refused(tmp_path):
    actual = tmp_path / 'actual'
    actual.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        monitor.collect(alias, runner=scheduler(), now=1000)
    (actual / '.paper3-hpc').symlink_to(tmp_path, target_is_directory=True)
    assert monitor.collect(actual, runner=scheduler(), now=1000)['data'] is None


def test_malformed_resource_shape_becomes_NA_without_crash(tmp_path):
    write(tmp_path, '.paper3-hpc/jobs/100/resources.live.json', {'stage': 'train', 'allocation': None})
    snapshot = monitor.collect(tmp_path, runner=scheduler(job_line()), now=1000)
    assert snapshot['jobs'][0]['resources']['source'] == 'N/A'
    assert '未采样项为 N/A' in monitor.render(snapshot)


def test_no_mutations_and_no_cache_raw_query_traversal(tmp_path, monkeypatch):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {'status': 'failed'})
    originals = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    real_iterdir = Path.iterdir
    def guarded_iterdir(path):
        assert not any(part in {'cache', 'raw', 'query', 'data'} for part in path.relative_to(tmp_path).parts
                       if part != 'data' or '.paper3-hpc' not in path.parts)
        return real_iterdir(path)
    monkeypatch.setattr(Path, 'iterdir', guarded_iterdir)
    monkeypatch.setattr(Path, 'write_text', lambda *args, **kwargs: pytest.fail('monitor attempted write'))
    monitor.collect(tmp_path, runner=scheduler(), now=1000)
    after = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert originals == after


def test_actual_failed_stage_gives_fix_before_resubmit_hint(tmp_path):
    submit(tmp_path, stage='prepare')
    write(tmp_path, '.paper3-hpc/jobs/100/stage-prepare.json',
          {'job_id': '100', 'stage': 'prepare', 'returncode': 2})
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    assert '最近 prepare 作业停止/失败' in monitor.next_step(snapshot)


@pytest.mark.parametrize('stage,expected', [('test', 'test 成功后 prepare'), ('prepare', 'prepare 成功'),
                                           ('train', '原冻结门禁'), ('backup', '完整 GET/SHA')])
def test_next_step_predicts_conditions_not_completion(tmp_path, stage, expected):
    snapshot = monitor.collect(tmp_path, runner=scheduler(job_line(stage=stage)), now=1000)
    assert expected in monitor.next_step(snapshot)


def test_once_non_TTY_has_no_clear_escape_and_exits(tmp_path, monkeypatch):
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    monkeypatch.setattr(monitor, 'collect', lambda *args, **kwargs: snapshot)
    output = io.StringIO()
    monkeypatch.setattr(monitor.sys, 'stdout', output)
    monkeypatch.setattr(monitor.time, 'sleep', lambda *args: pytest.fail('--once slept'))
    assert monitor.main(['--root', str(tmp_path), '--once']) == 0
    assert '\033' not in output.getvalue() and '只读' in output.getvalue()


def test_default_loop_interval_ten_seconds_and_ctrlC_exits(tmp_path, monkeypatch):
    observed = []
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    def collect(*args, **kwargs):
        observed.append(kwargs['interval'])
        return snapshot
    def stop(seconds):
        observed.append(seconds)
        raise KeyboardInterrupt
    monkeypatch.setattr(monitor, 'collect', collect)
    monkeypatch.setattr(monitor.time, 'sleep', stop)
    monkeypatch.setattr(monitor.sys, 'stdout', io.StringIO())
    assert monitor.main(['--root', str(tmp_path)]) == 0
    assert observed == [10, 10]


def test_R2_HEAD_diagnostic_prints_only_allowlisted_status_fields(tmp_path):
    write(tmp_path, '.paper3-hpc/diagnostics/r2-head-abc.json', {
        'schema_version': 'paper3-hpc-r2-head-diagnostic-v1', 'status': 'R2_ACCESS_DIAGNOSTIC_BLOCKED',
        'objects': [{'object': 'main_archive', 'status': 'HEAD_FAILED',
                     'error': {'category': 'S3_REQUEST_REJECTED', 'http_status': 403, 'service_code': 'AccessDenied'},
                     'endpoint': 'DO_NOT_PRINT_ENDPOINT', 'bucket': 'DO_NOT_PRINT_BUCKET'}],
        'next_step': 'DO_NOT_PRINT_ARBITRARY_TEXT'})
    shown = monitor.render(monitor.collect(tmp_path, runner=scheduler(), now=1000))
    assert 'HTTP 403' in shown and 'AccessDenied' in shown and 'S3_REQUEST_REJECTED' in shown
    assert 'DO_NOT_PRINT' not in shown and 'HEAD 不等于完整 GET/SHA' in shown


def test_R2_GET_failure_does_not_guess_cause_or_initiate_probe(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json',
          {'status': 'failed', 'operation': 'fetch', 'error_code': 'R2_GET_FAILED'})
    runner = scheduler()
    shown = monitor.render(monitor.collect(tmp_path, runner=runner, now=1000))
    assert '尚不足判定网络、权限或密钥原因' in shown
    assert 'hpc/r2_diagnose.py --network-host' in shown
    assert len(runner.calls) == 1


def test_HEAD_success_does_not_clear_failed_data_gate(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json',
          {'status': 'failed', 'operation': 'fetch', 'error_code': 'R2_GET_FAILED'})
    write(tmp_path, '.paper3-hpc/diagnostics/r2-head-abc.json', {
        'schema_version': 'paper3-hpc-r2-head-diagnostic-v1',
        'status': 'FIXED_OBJECT_HEADS_ACCESSIBLE_FULL_GET_STILL_REQUIRED',
        'objects': [{'object': 'main_archive', 'status': 'HEAD_LENGTH_MATCH_NOT_FULL_GET', 'http_status': 200}]})
    snapshot = monitor.collect(tmp_path, runner=scheduler(), now=1000)
    assert 'STOPPED' in monitor.render(snapshot) and '先按错误修复' in monitor.next_step(snapshot)


def test_diagnostic_unknown_service_code_and_arbitrary_text_are_not_printed(tmp_path):
    write(tmp_path, '.paper3-hpc/diagnostics/r2-head-abc.json', {
        'schema_version': 'paper3-hpc-r2-head-diagnostic-v1', 'status': 'R2_DIAGNOSTIC_CONFIGURATION_BLOCKED',
        'error': {'category': 'DO_NOT_PRINT_CATEGORY', 'http_status': 403, 'service_code': 'DO_NOT_PRINT_KEY'}})
    shown = monitor.render(monitor.collect(tmp_path, runner=scheduler(), now=1000))
    assert 'DO_NOT_PRINT' not in shown and 'HTTP 403' in shown


def test_current_running_stage_takes_precedence_over_separate_failed_fetch(tmp_path):
    write(tmp_path, '.paper3-hpc/data/attempt-abc/receipt.json', {'status': 'failed', 'error_code': 'R2_GET_FAILED'})
    snapshot = monitor.collect(tmp_path, runner=scheduler(job_line(stage='train')), now=1000)
    assert '原冻结门禁' in monitor.next_step(snapshot)
    assert 'STOPPED' not in monitor.render(snapshot)
    assert snapshot['data']['status'] == 'failed'


def test_final_device_peaks_explicitly_label_history(tmp_path):
    report = {'wall_seconds': 100, 'cpu_seconds': 200, 'gpu_observation_available': True,
              'owned_process_gpu_peak_memory_mib': 2000, 'owned_gpu_peaks': [
                  {'device_utilization_percent': 92, 'owned_process_memory_mib': 2000,
                   'total_memory_mib': 81920}]}
    final(tmp_path, measurements=report)
    shown = monitor.render(monitor.collect(tmp_path, runner=scheduler(), now=1000))
    assert '设备历史峰值利用率 92.0%' in shown
    assert '本作业 GPU 历史显存' in shown


def test_stale_device_metrics_explicitly_label_historical_sampling(tmp_path):
    live(tmp_path, mtime=500)
    shown = monitor.render(monitor.collect(tmp_path, runner=scheduler(job_line()), now=1000))
    assert '设备历史采样利用率' in shown and 'GPU 历史采样显存' in shown
