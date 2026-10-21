#!/usr/bin/env python3
"""Diagnose access to the five fixed release objects with HEAD only.

No object body is downloaded. A successful HEAD never means a complete GET,
SHA-256 verification, backup, data readiness or training completion. Endpoint,
bucket, credentials, raw exception messages and request headers are not saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.handoff import HandoffError, local, r2_client, safe_root, write_json
from hpc.stage_data import load_release

AWS_CODES = frozenset({'AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch',
    'NoSuchBucket', 'NoSuchKey', 'NotFound', '404', '403', 'RequestTimeout',
    'SlowDown', 'ExpiredToken', 'InvalidToken', 'AuthorizationHeaderMalformed',
    'InvalidRequest', 'InternalError', 'ServiceUnavailable', 'PermanentRedirect'})
NETWORK_ERRORS = {
    'EndpointConnectionError': 'NETWORK_DNS_OR_CONNECTION_FAILED',
    'ConnectTimeoutError': 'CONNECTION_TIMEOUT', 'ReadTimeoutError': 'RESPONSE_TIMEOUT',
    'SSLError': 'TLS_CERTIFICATE_FAILED', 'ProxyConnectionError': 'PROXY_CONNECTION_FAILED',
    'ConnectionClosedError': 'CONNECTION_CLOSED', 'HTTPClientError': 'HTTP_TRANSPORT_FAILED',
    'NoCredentialsError': 'R2_CREDENTIAL_CONFIGURATION_MISSING',
    'PartialCredentialsError': 'R2_CREDENTIAL_CONFIGURATION_INCOMPLETE',
}
CONFIG_CODES = frozenset({'existing_r2_configuration_required_on_network_allowed_node',
    'existing_environment_missing_minimal_r2_sdk', 'r2_https_account_endpoint_required',
    'interactive_hidden_input_required', 'invalid_hidden_credential'})


def safe_error(exc):
    """Fixed machine codes only; never stringify an SDK exception."""
    result = {'category': NETWORK_ERRORS.get(type(exc).__name__, 'UNCLASSIFIED_ERROR'),
              'http_status': None, 'service_code': None}
    response = getattr(exc, 'response', None)
    if isinstance(response, dict):
        metadata = response.get('ResponseMetadata', {})
        status = metadata.get('HTTPStatusCode') if isinstance(metadata, dict) else None
        service = response.get('Error', {})
        code = service.get('Code') if isinstance(service, dict) else None
        result.update(category='S3_REQUEST_REJECTED',
                      http_status=status if type(status) is int and 100 <= status <= 599 else None,
                      service_code=code if isinstance(code, str) and code in AWS_CODES else 'OTHER_S3_ERROR')
    if isinstance(exc, HandoffError):
        result['category'] = exc.code if exc.code in CONFIG_CODES else 'INVALID_LOCAL_DIAGNOSTIC_CONFIGURATION'
    return result


def next_step(rows):
    bad = [row for row in rows if row['status'] != 'HEAD_LENGTH_MATCH_NOT_FULL_GET']
    if not bad:
        return 'HEAD可访问；仍须由原fetch完成GET及SHA验证。若GET再失败，保留该失败回执；HEAD不能证明大文件正文可完整下载。'
    error = bad[0].get('error', {})
    category = error.get('category', '')
    status, code = error.get('http_status'), error.get('service_code')
    if category == 'TLS_CERTIFICATE_FAILED':
        return '检查学校认可的CA证书及代理配置；保留TLS验证，不使用verify=False。'
    if category.startswith(('NETWORK_', 'CONNECTION_', 'RESPONSE_', 'PROXY_', 'HTTP_')):
        return '在学校允许访问R2的传输节点排查DNS、代理与出网；或在允许联网的机器取回后手动转移固定包和缓存。'
    if status == 403 or code in {'AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch', 'InvalidToken', 'ExpiredToken'}:
        return '检查原账号/原bucket对应的有效凭证和固定对象读取权限；403本身不能单独判定是密钥错误。'
    if status == 404 or code in {'NoSuchKey', 'NoSuchBucket', 'NotFound', '404'}:
        return '确认使用保存这些冻结对象的原账号与原bucket；对象不存在时需找回原对象，不能用空文件替代。'
    if bad[0]['status'] == 'HEAD_SIZE_MISMATCH':
        return '远端长度不符冻结描述；保持数据门阻塞，检查是否连接了正确的原存储。'
    return '保留固定诊断码并核对现有R2配置或环境依赖；不打印或发送密钥。'


def diagnose(root, *, network_host=False, client=None, bucket=None, prompt=False,
             client_factory=r2_client, release_loader=load_release, emit=None):
    root = safe_root(root)
    if network_host is not True:
        return {'status': 'READONLY_PLAN_NO_NETWORK', 'network_requests': 0,
                'instruction': 'Use --network-host only on a school-permitted network or transfer host.'}
    os.umask(0o077)
    report = {'schema_version': 'paper3-hpc-r2-head-diagnostic-v1', 'status': 'DIAGNOSTIC_RUNNING',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'method': 'HEAD',
              'network_requests': 0, 'object_body_GETs': 0, 'downloaded_body_bytes': 0,
              'full_GET_SHA256_verified': False, 'data_ready_claimed': False,
              'training_started': False, 'external_query_accessed': False,
              'credentials_recorded': False, 'objects': []}
    relative = '.paper3-hpc/diagnostics/r2-head-' + uuid.uuid4().hex + '.json'
    path = local(root, relative)
    report['report_path'] = relative
    created_client = False
    try:
        release = release_loader()
        if client is None:
            client, bucket = client_factory(prompt=prompt, metadata_probe=True)
            created_client = True
        if not bucket:
            raise HandoffError('existing_r2_configuration_required_on_network_allowed_node')
        for label, fixed in release['objects'].items():
            if emit:
                emit({'status': 'CHECKING_FIXED_OBJECT_HEAD', 'object': label})
            row = {'object': label, 'expected_bytes': fixed['bytes'],
                   'status': 'HEAD_FAILED', 'full_GET_SHA256_verified': False}
            report['network_requests'] += 1
            try:
                response = client.head_object(Bucket=bucket, Key=fixed['key'])
                http = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
                size = response.get('ContentLength')
                row.update(http_status=http if type(http) is int and 100 <= http <= 599 else None,
                           remote_bytes=size if type(size) is int and size >= 0 else None)
                if http == 200 and type(size) is int:
                    row['status'] = 'HEAD_LENGTH_MATCH_NOT_FULL_GET' if size == fixed['bytes'] else 'HEAD_SIZE_MISMATCH'
                else:
                    row['error'] = {'category': 'INVALID_HEAD_RESPONSE', 'http_status': row['http_status'], 'service_code': None}
            except Exception as exc:
                row['error'] = safe_error(exc)
            report['objects'].append(row)
            write_json(path, report)
            if emit:
                emit(row)
            # Stop at a real failure; never loop indefinitely or retry a body.
            if row['status'] != 'HEAD_LENGTH_MATCH_NOT_FULL_GET':
                break
        report['status'] = ('FIXED_OBJECT_HEADS_ACCESSIBLE_FULL_GET_STILL_REQUIRED'
            if len(report['objects']) == len(release['objects'])
            and all(row['status'] == 'HEAD_LENGTH_MATCH_NOT_FULL_GET' for row in report['objects'])
            else 'R2_ACCESS_DIAGNOSTIC_BLOCKED')
        report['next_step'] = next_step(report['objects'])
    except KeyboardInterrupt:
        report.update(status='DIAGNOSTIC_INTERRUPTED_NOT_COMPLETE', next_step='诊断已停止；没有下载正文或启动训练。')
    except Exception as exc:
        error = safe_error(exc)
        report.update(status='R2_DIAGNOSTIC_CONFIGURATION_BLOCKED', error=error,
                      next_step=next_step([{'status': 'HEAD_FAILED', 'error': error}]))
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(path, report)
        if created_client:
            try:
                client.close()
            except Exception:
                pass
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--network-host', action='store_true')
    args = parser.parse_args(argv)
    def emit(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)
    try:
        report = diagnose(args.root, network_host=args.network_host, prompt=sys.stdin.isatty(), emit=emit)
        emit(report)
        return 0 if report['status'] in {'READONLY_PLAN_NO_NETWORK', 'FIXED_OBJECT_HEADS_ACCESSIBLE_FULL_GET_STILL_REQUIRED'} else 2
    except Exception:
        emit({'status': 'DIAGNOSTIC_BLOCKED_UNSAFE_OR_UNWRITABLE_PROJECT',
              'training_started': False, 'credentials_recorded': False})
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
