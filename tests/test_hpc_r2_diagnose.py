"""Only fake HEAD transports; no data download, credentials file or fit."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hpc import handoff
from hpc import r2_diagnose as diag


def release():
    return {'objects': {f'object_{i}': {'key': f'paper3/fixture/{i}', 'bytes': 7 + i}
                        for i in range(5)}}


class HeadClient:
    def __init__(self, failure=None, wrong_length=False):
        self.calls = []
        self.failure = failure
        self.wrong_length = wrong_length

    def head_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        size = int(kwargs['Key'].split('/')[-1]) + 7
        return {'ResponseMetadata': {'HTTPStatusCode': 200},
                'ContentLength': size + int(self.wrong_length)}

    def get_object(self, **kwargs):
        pytest.fail('a diagnostic must not download any object body')


def test_default_plan_never_creates_client_or_writes(tmp_path):
    def forbidden(**kwargs):
        pytest.fail('no networking allowed without explicit network host flag')
    value = diag.diagnose(tmp_path, client_factory=forbidden)
    assert value['network_requests'] == 0
    assert not list(tmp_path.iterdir())


def test_five_head_successes_never_claim_full_get_sha_or_training(tmp_path):
    client = HeadClient()
    report = diag.diagnose(tmp_path, network_host=True, client=client,
                           bucket='SECRET_BUCKET_MUST_NOT_BE_RECORDED', release_loader=release)
    assert report['status'] == 'FIXED_OBJECT_HEADS_ACCESSIBLE_FULL_GET_STILL_REQUIRED'
    assert len(client.calls) == 5 and report['object_body_GETs'] == 0
    assert report['full_GET_SHA256_verified'] is False
    assert report['training_started'] is False and report['data_ready_claimed'] is False
    saved = (tmp_path / report['report_path']).read_text()
    assert 'SECRET_BUCKET_MUST_NOT_BE_RECORDED' not in saved
    assert json.loads(saved) == report


@pytest.mark.parametrize('status,service', [(403, 'AccessDenied'), (404, 'NoSuchKey'),
    (403, 'SignatureDoesNotMatch'), (400, 'SECRET_KEY_IN_UNTRUSTED_SERVICE_CODE')])
def test_service_errors_are_whitelisted_without_secret_message(tmp_path, status, service):
    class ClientError(Exception):
        response = {'ResponseMetadata': {'HTTPStatusCode': status}, 'Error': {
            'Code': service, 'Message': 'SECRET_TOKEN_IN_SERVER_MESSAGE'}}
    client = HeadClient(ClientError('SECRET_CREDENTIAL_IN_EXCEPTION'))
    report = diag.diagnose(tmp_path, network_host=True, client=client,
                           bucket='fixture', release_loader=release)
    assert len(client.calls) == 1 and report['status'] == 'R2_ACCESS_DIAGNOSTIC_BLOCKED'
    assert report['objects'][0]['error']['http_status'] == status
    assert 'SECRET_' not in json.dumps(report)


@pytest.mark.parametrize('name,category', list(diag.NETWORK_ERRORS.items()))
def test_network_diagnostics_report_type_without_exception_payload(tmp_path, name, category):
    failure = type(name, (Exception,), {})('SECRET_ENDPOINT_AND_SIGNATURE')
    report = diag.diagnose(tmp_path, network_host=True, client=HeadClient(failure),
                           bucket='fixture', release_loader=release)
    assert report['objects'][0]['error']['category'] == category
    assert 'SECRET_' not in json.dumps(report)
    assert report['network_requests'] == 1


def test_remote_size_mismatch_stays_blocked_without_download(tmp_path):
    report = diag.diagnose(tmp_path, network_host=True, client=HeadClient(wrong_length=True),
                           bucket='fixture', release_loader=release)
    assert report['status'] == 'R2_ACCESS_DIAGNOSTIC_BLOCKED'
    assert report['objects'][0]['status'] == 'HEAD_SIZE_MISMATCH'
    assert report['full_GET_SHA256_verified'] is False


def test_factory_requests_bounded_head_timeouts_and_closes_owned_client(tmp_path):
    seen = []
    client = HeadClient()
    client.close = lambda: seen.append('closed')
    def factory(**kwargs):
        seen.append(kwargs)
        return client, 'fixture'
    report = diag.diagnose(tmp_path, network_host=True, prompt=False,
                           client_factory=factory, release_loader=release)
    assert seen == [{'prompt': False, 'metadata_probe': True}, 'closed']
    assert report['network_requests'] == 5


def test_missing_sdk_is_a_minimum_gap_not_false_head_success(tmp_path):
    def missing(**kwargs):
        raise handoff.HandoffError('existing_environment_missing_minimal_r2_sdk')
    report = diag.diagnose(tmp_path, network_host=True, client_factory=missing, release_loader=release)
    assert report['status'] == 'R2_DIAGNOSTIC_CONFIGURATION_BLOCKED'
    assert report['network_requests'] == 0
    assert report['error']['category'] == 'existing_environment_missing_minimal_r2_sdk'


def test_symlink_report_tree_does_not_escape_project(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (tmp_path / '.paper3-hpc').symlink_to(outside, target_is_directory=True)
    with pytest.raises(handoff.HandoffError, match='symlink'):
        diag.diagnose(tmp_path, network_host=True, client=HeadClient(), bucket='fixture', release_loader=release)
    assert not list(outside.iterdir())


@pytest.mark.parametrize('probe,connect,read,attempts', [(False, 15, 120, 2), (True, 5, 10, 0)])
def test_only_head_diagnostic_changes_transport_timeouts(monkeypatch, probe, connect, read, attempts):
    for name, value in zip(handoff.R2_NAMES, ('https://fixture.r2.cloudflarestorage.com',
                                           'fixture', 'test-access', 'test-secret')):
        monkeypatch.setenv(name, value)
    seen = {}
    def config(**kwargs):
        seen.update(kwargs)
        return object()
    def client(*args, **kwargs):
        return object()
    monkeypatch.setitem(sys.modules, 'boto3', SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules, 'botocore.config', SimpleNamespace(Config=config))
    handoff.r2_client(metadata_probe=probe)
    assert seen == {'connect_timeout': connect, 'read_timeout': read,
                    'retries': {'max_attempts': attempts, 'mode': 'standard'}}
