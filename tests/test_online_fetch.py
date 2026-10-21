import hashlib
import io
import json
from pathlib import Path
import sys
from urllib import error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import online_fetch as fetch

REPO = 'owner/research3'
TAG = 'paper3-online-v1'
NAME = 'bundle.zip'
BODY = b'fixture complete body\x00\x01'
SHA = hashlib.sha256(BODY).hexdigest()
TOKEN = 'fixture-token-never-persist'


@pytest.fixture(autouse=True)
def no_real_http_by_default(monkeypatch):
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('Real HTTP forbidden in unit fixtures'))


class Response(io.BytesIO):
    def __init__(self, raw, status=200, headers=None):
        super().__init__(raw)
        self.status, self.headers = status, headers or {}

    def getcode(self):
        return self.status


def metadata(**changes):
    value = {'tag_name': TAG, 'draft': False, 'assets': [
        {'id': 123, 'name': NAME, 'size': len(BODY), 'state': 'uploaded'}]}
    value.update(changes)
    return Response(json.dumps(value).encode())


def call(path, **kwargs):
    return fetch.fetch_release_asset(REPO, TAG, NAME, SHA, len(BODY), path, **kwargs)


def opener(monkeypatch, responses):
    calls = []
    def open_(req):
        calls.append(req)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(fetch, '_open_once', open_)
    return calls


def test_public_complete_get_and_atomic_commit(tmp_path, monkeypatch):
    calls = opener(monkeypatch, [metadata(), Response(b'', 302, {'Location': 'https://release-assets.githubusercontent.com/asset?signed=fixture'}),
                                Response(BODY, headers={'Content-Length': str(len(BODY))})])
    destination = tmp_path / NAME
    result = call(destination)
    assert destination.read_bytes() == BODY
    assert result['get_attempted'] and result['complete_body_read']
    assert result['network_bytes_read'] == len(BODY)
    assert result['atomic_commit'] == 'link_no_overwrite'
    assert not list(tmp_path.glob('*.partial'))
    assert not any(req.get_header('Authorization') for req in calls)


def test_private_auth_only_api_never_forwarded_to_cdn(tmp_path, monkeypatch):
    calls = opener(monkeypatch, [metadata(), Response(b'', 302, {'Location': 'https://objects.githubusercontent.com/asset?sig=fixture'}),
                                Response(BODY)])
    result = call(tmp_path / NAME, token=TOKEN)
    assert result['complete_body_read']
    assert calls[0].get_header('Authorization') == 'Bearer ' + TOKEN
    assert calls[1].get_header('Authorization') == 'Bearer ' + TOKEN
    assert calls[2].get_header('Authorization') is None
    assert TOKEN not in json.dumps(result)
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes()


def test_cache_hit_no_network(tmp_path, monkeypatch):
    destination = tmp_path / NAME
    destination.write_bytes(BODY)
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('Network on cache hit'))
    result = call(destination)
    assert result['status'] == 'LOCAL_VERIFIED_CACHE_HIT' and not result['get_attempted']


def test_wrong_existing_is_preserved_no_network(tmp_path, monkeypatch):
    destination = tmp_path / NAME
    destination.write_bytes(b'keep foreign bytes')
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('Network before refusal'))
    with pytest.raises(fetch.FetchError, match='EXISTING_GITHUB_ASSET_DIFFERS_PRESERVED'):
        call(destination)
    assert destination.read_bytes() == b'keep foreign bytes'


@pytest.mark.parametrize('response,code', [
    (Response(b'x' * len(BODY)), 'GITHUB_RELEASE_BODY_SHA_OR_LENGTH_MISMATCH'),
    (Response(BODY[:-1]), 'GITHUB_RELEASE_BODY_SHA_OR_LENGTH_MISMATCH'),
    (Response(BODY + b'!'), 'GITHUB_RELEASE_BODY_LENGTH_MISMATCH'),
    (Response(BODY, headers={'Content-Length': '1'}), 'GITHUB_RELEASE_DECLARED_LENGTH_MISMATCH'),
    (Response(BODY, 206), 'GITHUB_RELEASE_HTTP_NOT_200'),
])
def test_bad_body_never_committed(tmp_path, monkeypatch, response, code):
    opener(monkeypatch, [metadata(), response])
    with pytest.raises(fetch.FetchError, match=code):
        call(tmp_path / NAME)
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob('*.partial'))


@pytest.mark.parametrize('url', ['http://github.com/file', 'https://evil.invalid/file',
    'https://user:secret@github.com/file', 'https://github.com:444/file',
    'https://github.com.evil.invalid/file', 'https://release-assets.githubusercontent.com/file#fragment'])
def test_unsafe_redirect_refused_and_no_credentials_sent(tmp_path, monkeypatch, url):
    calls = opener(monkeypatch, [metadata(), Response(b'', 302, {'Location': url})])
    with pytest.raises(fetch.FetchError, match='GITHUB_RELEASE_REDIRECT_DESTINATION_REFUSED'):
        call(tmp_path / NAME, token=TOKEN)
    assert len(calls) == 2


def test_metadata_wrong_tag_blocks_asset_get(tmp_path, monkeypatch):
    calls = opener(monkeypatch, [metadata(tag_name='different')])
    with pytest.raises(fetch.FetchError, match='GITHUB_RELEASE_METADATA_IDENTITY_MISMATCH'):
        call(tmp_path / NAME)
    assert len(calls) == 1


def test_duplicate_matching_asset_blocks_download(tmp_path, monkeypatch):
    row = {'id': 123, 'name': NAME, 'size': len(BODY), 'state': 'uploaded'}
    opener(monkeypatch, [metadata(assets=[row, row])])
    with pytest.raises(fetch.FetchError, match='GITHUB_RELEASE_EXACT_ASSET_MISSING'):
        call(tmp_path / NAME)


def test_symlink_destination_refused(tmp_path, monkeypatch):
    foreign = tmp_path / 'foreign'
    foreign.write_bytes(BODY)
    destination = tmp_path / NAME
    destination.symlink_to(foreign)
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('Network before path check'))
    with pytest.raises(fetch.FetchError, match='GITHUB_DOWNLOAD_SYMLINK_REFUSED'):
        call(destination)


def test_service_exception_details_suppressed(tmp_path, monkeypatch):
    opener(monkeypatch, [RuntimeError('Secret: ' + TOKEN)])
    with pytest.raises(fetch.FetchError) as error:
        call(tmp_path / NAME, token=TOKEN)
    assert error.value.code == 'GITHUB_RELEASE_CONNECTION_FAILED'
    assert TOKEN not in str(error.value)


def test_urllib_http_redirect_handled_manually(tmp_path, monkeypatch):
    redirect = error.HTTPError('https://github.com/fixture', 302, 'Found',
                                {'Location': 'https://release-assets.githubusercontent.com/fixture'}, io.BytesIO())
    calls = opener(monkeypatch, [metadata(), redirect, Response(BODY)])
    assert call(tmp_path / NAME)['complete_body_read']
    assert len(calls) == 3


def test_private_cli_getpass_never_falls_back_to_echo(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'stdin', type('TTY', (), {'isatty': lambda self: True})())
    monkeypatch.setattr(fetch, 'fetch_release_asset',
                        lambda *a, **kw: (_ for _ in ()).throw(fetch.FetchError('GITHUB_RELEASE_AUTH_REQUIRED')))
    def unsafe_getpass(prompt):
        import warnings
        warnings.warn('Fixture cannot hide input', fetch.getpass.GetPassWarning)
        pytest.fail('getpass warning must stop before a fallback read')
    monkeypatch.setattr(fetch.getpass, 'getpass', unsafe_getpass)
    assert fetch.main(['--tag', TAG]) == 2
    assert 'INTERACTIVE_HIDDEN_INPUT_REQUIRED' in capsys.readouterr().out


COMMIT = '4e4fa1504e01bf2d04875c9ce04d543f3248f826'
PINNED_PATH = 'launcher_assets/' + NAME


def pinned_call(destination, **kwargs):
    values = dict(repo=REPO, commit=COMMIT, path=PINNED_PATH,
                  expected_sha=SHA, expected_bytes=len(BODY), destination=destination)
    values.update(kwargs)
    return fetch.fetch_pinned_file(**values)


def test_pinned_raw_complete_get_without_api_or_authorization(tmp_path, monkeypatch):
    calls = opener(monkeypatch, [Response(BODY, headers={'Content-Length': str(len(BODY))})])
    destination = tmp_path / NAME
    result = pinned_call(destination)
    assert len(calls) == 1
    assert calls[0].full_url == 'https://raw.githubusercontent.com/' + REPO + '/' + COMMIT + '/' + PINNED_PATH
    assert calls[0].get_header('Authorization') is None
    assert 'api.github.com' not in calls[0].full_url
    assert destination.read_bytes() == BODY
    assert result['status'] == 'PINNED_GITHUB_FILE_COMPLETE_GET_SHA256_VERIFIED'
    assert result['get_attempted'] is True and result['complete_body_read'] is True
    assert result['http_status'] == 200 and result['network_bytes_read'] == len(BODY)
    assert result['commit'] == COMMIT and result['path'] == PINNED_PATH
    assert result['sha256'] == SHA and result['atomic_commit'] == 'link_no_overwrite'
    assert not list(tmp_path.glob('*.partial'))


def test_pinned_cache_exact_local_no_api_or_network(tmp_path, monkeypatch):
    destination = tmp_path / NAME
    destination.write_bytes(BODY)
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('No request on cache hit'))
    result = pinned_call(destination)
    assert result['status'] == 'LOCAL_VERIFIED_CACHE_HIT'
    assert not result['get_attempted'] and result['network_bytes_read'] == 0


def test_pinned_cache_wrong_content_preserved_without_network(tmp_path, monkeypatch):
    destination = tmp_path / NAME
    destination.write_bytes(b'foreign bytes to preserve')
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('No request before mismatch refusal'))
    with pytest.raises(fetch.FetchError, match='EXISTING_GITHUB_ASSET_DIFFERS_PRESERVED'):
        pinned_call(destination)
    assert destination.read_bytes() == b'foreign bytes to preserve'


@pytest.mark.parametrize('response,code', [
    (Response(b'x' * len(BODY)), 'GITHUB_RELEASE_BODY_SHA_OR_LENGTH_MISMATCH'),
    (Response(BODY[:-1]), 'GITHUB_RELEASE_BODY_SHA_OR_LENGTH_MISMATCH'),
    (Response(BODY + b'!'), 'GITHUB_RELEASE_BODY_LENGTH_MISMATCH'),
    (Response(BODY, headers={'Content-Length': '1'}), 'GITHUB_RELEASE_DECLARED_LENGTH_MISMATCH'),
    (Response(BODY, 206), 'GITHUB_RELEASE_HTTP_NOT_200'),
])
def test_pinned_bad_http_body_never_commits(tmp_path, monkeypatch, response, code):
    calls = opener(monkeypatch, [response])
    with pytest.raises(fetch.FetchError, match=code):
        pinned_call(tmp_path / NAME)
    assert len(calls) == 1 and 'raw.githubusercontent.com' in calls[0].full_url
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob('*.partial'))


@pytest.mark.parametrize('value', [-1, 0, True, '12', None])
def test_pinned_invalid_size_rejected_before_network(tmp_path, monkeypatch, value):
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('Invalid size must not request'))
    with pytest.raises(fetch.FetchError, match='GITHUB_RELEASE_IDENTITY_INVALID'):
        pinned_call(tmp_path / NAME, expected_bytes=value)


@pytest.mark.parametrize('value', ['../bundle.zip', '/bundle.zip', 'x/../bundle.zip',
                                  'x//bundle.zip', './bundle.zip', 'x/./bundle.zip',
                                  'x\\bundle.zip', 'x/\nbundle.zip', ''])
def test_pinned_path_traversal_rejected_before_network(tmp_path, monkeypatch, value):
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('Unsafe path must not request'))
    with pytest.raises(fetch.FetchError, match='PINNED_GITHUB_PATH_INVALID'):
        pinned_call(tmp_path / NAME, path=value)


@pytest.mark.parametrize('value', ['main', 'paper3-online-v1', 'a' * 39, 'A' * 40, 'a' * 40 + '/x'])
def test_pinned_requires_exact_immutable_commit(tmp_path, monkeypatch, value):
    monkeypatch.setattr(fetch, '_open_once', lambda req: pytest.fail('Mutable ref must not request'))
    with pytest.raises(fetch.FetchError, match='IMMUTABLE_GITHUB_COMMIT_REQUIRED'):
        pinned_call(tmp_path / NAME, commit=value)


def test_pinned_redirect_still_never_sends_authorization(tmp_path, monkeypatch):
    calls = opener(monkeypatch, [Response(b'', 302, {'Location': 'https://objects.githubusercontent.com/raw-fixture'}),
                                Response(BODY)])
    assert pinned_call(tmp_path / NAME)['complete_body_read']
    assert len(calls) == 2 and all(req.get_header('Authorization') is None for req in calls)


def test_pinned_wrong_sha_leaves_no_destination(tmp_path, monkeypatch):
    opener(monkeypatch, [Response(BODY)])
    with pytest.raises(fetch.FetchError, match='GITHUB_RELEASE_BODY_SHA_OR_LENGTH_MISMATCH'):
        pinned_call(tmp_path / NAME, expected_sha='0' * 64)
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob('*.partial'))


def test_pinned_atomic_commit_does_not_replace_concurrent_file(tmp_path, monkeypatch):
    opener(monkeypatch, [Response(BODY)])
    destination = tmp_path / NAME
    def race(source, target):
        Path(target).write_bytes(b'concurrent foreign file')
        raise FileExistsError('fixture race')
    monkeypatch.setattr(fetch.os, 'link', race)
    with pytest.raises(fetch.FetchError, match='GITHUB_DOWNLOAD_TARGET_CREATED_CONCURRENTLY'):
        pinned_call(destination)
    assert destination.read_bytes() == b'concurrent foreign file'
    assert not list(tmp_path.glob('*.partial'))


def test_default_cli_uses_pinned_file_not_release_api(tmp_path, monkeypatch, capsys):
    calls = []
    def fake_pinned(*args):
        calls.append(args)
        return {'status': 'FIXTURE_PINNED_GET_VERIFIED', 'get_attempted': True}
    monkeypatch.setattr(fetch, 'fetch_pinned_file', fake_pinned)
    monkeypatch.setattr(fetch, 'fetch_release_asset', lambda *args, **kwargs: pytest.fail('Default must not call release API'))
    monkeypatch.setattr(fetch.getpass, 'getpass', lambda *args: pytest.fail('Public pinned default must not prompt for GitHub token'))
    destination = tmp_path / NAME
    assert fetch.main(['--destination', str(destination)]) == 0
    assert len(calls) == 1
    assert calls[0][0] == 'jackzhu119/research3' and calls[0][1] == COMMIT
    assert calls[0][2] == 'launcher_assets/Paper3_OneClick_20261021.zip'
    assert calls[0][-1] == destination
    assert 'FIXTURE_PINNED_GET_VERIFIED' in capsys.readouterr().out
