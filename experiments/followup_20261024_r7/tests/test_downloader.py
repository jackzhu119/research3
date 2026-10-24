"""Fault-injected transport checks for the small resumable Round7 bootstrap."""
import hashlib
import http.client
import importlib.util
import io
from pathlib import Path
import subprocess
import urllib.error

import pytest


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[1] / "online/DOWNLOAD-AND-START-R7.py"
    spec = importlib.util.spec_from_file_location("paper3_r7_download", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    value.CHUNK_BYTES = 4
    return value


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None, error_after=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(content))}
        self.error_after = error_after

    def read(self, size=-1):
        if self.error_after is not None and self.tell() >= self.error_after:
            raise TimeoutError("synthetic socket timeout")
        return super().read(size)


def test_partial_disconnect_is_resumed_with_exact_range(downloader, tmp_path):
    payload = b"0123456789abcdef"
    target = tmp_path / "START.py"
    requests = []

    def opener(request, **kwargs):
        requested = request.get_header("Range")
        requests.append(requested)
        assert kwargs["timeout"] == 30
        if len(requests) == 1:
            return Response(payload, error_after=8)
        assert requested == "bytes=8-"
        return Response(payload[8:], 206, {"Content-Range": "bytes 8-15/16", "Content-Length": "8"})

    result = downloader.obtain(target, len(payload), digest(payload), urls=[("raw", "https://raw.test/file")],
                               attempts=2, opener=opener, sleep=lambda _: None)
    assert result == target and target.read_bytes() == payload
    assert requests == [None, "bytes=8-"]


def test_ignored_range_restarts_only_its_own_mirror(downloader, tmp_path):
    payload = b"the pinned full bundle"
    target = tmp_path / "START.py"
    raw = target.with_name(target.name + ".raw.part")
    api = target.with_name(target.name + ".api.part")
    raw.write_bytes(payload[:7])
    api.write_bytes(payload[:3])

    def opener(request, **_):
        assert request.get_header("Range") == "bytes=7-"
        return Response(payload)

    downloader.obtain(target, len(payload), digest(payload), urls=[("raw", "https://raw.test/file")],
                      attempts=1, opener=opener)
    assert target.read_bytes() == payload
    assert api.read_bytes() == payload[:3]


@pytest.mark.parametrize("headers", [
    {"Content-Range": "bytes 1-9/10", "Content-Length": "9"},
    {"Content-Range": "bytes 4-10/11", "Content-Length": "7"},
    {"Content-Range": "bytes 4-9/10", "Content-Length": "5"},
    {"Content-Length": "6"},
])
def test_wrong_206_cannot_corrupt_retained_prefix(downloader, tmp_path, headers):
    payload = b"0123456789"
    part = tmp_path / "file.raw.part"
    part.write_bytes(payload[:4])
    with pytest.raises(downloader.DownloadError):
        downloader.transfer_once("https://test/file", part, len(payload), digest(payload),
                                 opener=lambda *_, **__: Response(payload[4:], 206, headers))
    assert part.read_bytes() == payload[:4]


def test_incomplete_read_partial_bytes_are_not_lost(downloader, tmp_path):
    payload = b"abcdefghij"
    part = tmp_path / "file.part"

    class Interrupted(Response):
        def read(self, size=-1):
            raise http.client.IncompleteRead(payload[:3], 7)

    with pytest.raises(downloader.DownloadError, match="interrupted"):
        downloader.transfer_once("https://test/file", part, len(payload), digest(payload),
                                 opener=lambda *_, **__: Interrupted(payload))
    assert part.read_bytes() == payload[:3]


def test_bad_sha_is_never_executed_and_rejected_bytes_are_kept(downloader, tmp_path, monkeypatch):
    expected = b"print('correct')\n"
    wrong = b"print('wrong!!')\n"
    assert len(wrong) == len(expected)
    monkeypatch.setattr(downloader, "SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(downloader, "WRAPPER_BYTES", len(expected))
    monkeypatch.setattr(downloader, "WRAPPER_SHA256", digest(expected))
    monkeypatch.setattr(downloader, "tls_context", lambda _: None)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_, **__: Response(wrong))
    monkeypatch.setattr(downloader.subprocess, "call", lambda *_: pytest.fail("Unverified bytes executed"))
    with pytest.raises(downloader.AttemptsExhausted):
        downloader.main(["--download-dir", str(tmp_path), "--attempts", "1", "--submit"])
    assert not (tmp_path / downloader.FILE_NAME).exists()
    assert len(list(tmp_path.glob("*.invalid"))) == 2


def test_verified_cache_runs_offline_and_forwards_school_arguments(downloader, tmp_path, monkeypatch):
    payload = b"print('already downloaded')\n"
    path = tmp_path / downloader.FILE_NAME
    path.write_bytes(payload)
    monkeypatch.setattr(downloader, "SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(downloader, "WRAPPER_BYTES", len(payload))
    monkeypatch.setattr(downloader, "WRAPPER_SHA256", digest(payload))
    monkeypatch.setattr(downloader, "tls_context", lambda _: None)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_args, **_kw: pytest.fail("Network on verified cache"))
    invoked = []
    monkeypatch.setattr(downloader.subprocess, "call", lambda args: invoked.append(args) or 0)
    assert downloader.main(["--download-dir", str(tmp_path), "--partition", "hpc_gpu", "--submit"]) == 0
    assert invoked[0][-3:] == ["--partition", "hpc_gpu", "--submit"]
    assert invoked[0][1:5] == ["-I", "-S", "-B", "-u"]
    assert invoked[0][5] == str(path)


def test_truncated_eof_and_http416_preserve_partial_for_next_command(downloader, tmp_path):
    payload = b"0123456789"
    target = tmp_path / "START.py"
    part = target.with_name(target.name + ".raw.part")
    with pytest.raises(downloader.AttemptsExhausted):
        downloader.obtain(target, len(payload), digest(payload), urls=[("raw", "https://test/file")], attempts=1,
                          opener=lambda *_a, **_k: Response(payload[:6], headers={"Content-Length": "10"}))
    assert part.read_bytes() == payload[:6]

    def unavailable(request, **_):
        assert request.get_header("Range") == "bytes=6-"
        raise urllib.error.HTTPError(request.full_url, 416, "Range unavailable", {}, None)

    with pytest.raises(downloader.AttemptsExhausted):
        downloader.obtain(target, len(payload), digest(payload), urls=[("raw", "https://test/file")],
                          attempts=1, opener=unavailable)
    assert part.read_bytes() == payload[:6]


def test_whole_verified_partial_publishes_without_network(downloader, tmp_path):
    payload = b"whole bytes from previous interrupted process"
    target = tmp_path / "START.py"
    target.with_name(target.name + ".raw.part").write_bytes(payload)
    downloader.obtain(target, len(payload), digest(payload), urls=[("raw", "https://test/file")], attempts=1,
                      opener=lambda *_a, **_k: pytest.fail("Should verify whole partial locally"))
    assert target.read_bytes() == payload


def test_default_retry_uses_bounded_wait_and_never_claims_submission(downloader, tmp_path, capsys):
    payload = b"12345678"
    count, delays = [0], []

    def opener(*_, **__):
        count[0] += 1
        if count[0] == 1:
            raise TimeoutError("offline now")
        return Response(payload)

    downloader.obtain(tmp_path / "START.py", len(payload), digest(payload),
                      urls=[("raw", "https://test/file")], opener=opener, sleep=delays.append)
    assert delays == [2] and count[0] == 2
    text = capsys.readouterr().out
    assert '"status": "WAITING_DOWNLOAD_RETRY"' in text
    assert '"jobs_submitted": false' in text and '"training_started": false' in text
