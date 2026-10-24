#!/usr/bin/env python3
"""Download and verify an immutable per-round hpc_gpu migration bundle.

Round4/5 reuse existing school data. Round6 also includes its prepared Pan cache.
Persistent independent raw/API partials resume before installation. Only an
explicit forwarded --submit starts checkpoint migration of that selected round.
"""
import argparse
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

SOURCE_COMMIT = "79eabe6c456aa443e18c35c53b6c39a33d27a639"
REPOSITORY = "jackzhu119/research3"
ROUND_PACKAGES = {'4': {'path': 'operations/hpc_gpu_migrations_20261024/round4/online/START-PAPER3-R4-HPCGPU.py', 'bytes': 323081, 'sha256': 'a1ef49e5e888d67477b59f8db0ec8ee798bdba51335b667ae8631cc595c9516a', 'payload_sha256': 'd2501fa7ecbbf630adbea08353f28ad5a7b38c1d5f53005d2fa11522c8ff0c7e', 'install_directory_name': 'd2501fa7ecbbf630adbe'}, '5': {'path': 'operations/hpc_gpu_migrations_20261024/round5/online/START-PAPER3-R5-HPCGPU.py', 'bytes': 324043, 'sha256': 'b32ba85331f25ec4e10d0ba41348d033eab7a60e29b8bcce59c0f5aca2d01e66', 'payload_sha256': '02955e93c9180dadeb6ea44ebaaae92a064d698f444e7b56fe295dcacb4ebec3', 'install_directory_name': '02955e93c9180dadeb6e'}, '6': {'path': 'operations/hpc_gpu_migrations_20261024/round6/online/START-PAPER3-R6-HPCGPU.py', 'bytes': 24659326, 'sha256': 'd993c7408b1510a061c715a32ee45002903534b148e9adfe8e18798e26dc053e', 'payload_sha256': '45fb15d5c45bf0af03729d70786527e5baece9e5c28c3dfd33b82f84455d4c5e', 'install_directory_name': '45fb15d5c45bf0af0372'}}
REMOTE_PATH = ""
FILE_NAME = ""
WRAPPER_SHA256 = ""
WRAPPER_BYTES = 0
CHUNK_BYTES = 64 * 1024
SOCKET_TIMEOUT_SECONDS = 30
PROGRESS_SECONDS = 5


class DownloadError(Exception):
    pass


class AttemptsExhausted(DownloadError):
    pass


def emit(status, **values):
    print(json.dumps({"status": status, **values}, ensure_ascii=False,
                     sort_keys=True), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path, expected_size, expected_sha):
    path = Path(path)
    return (path.is_file() and path.stat().st_size == expected_size
            and sha256(path) == expected_sha)


def publication_urls(commit=None):
    commit = SOURCE_COMMIT if commit is None else commit
    return (
        ("raw", f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{REMOTE_PATH}"),
        ("api", f"https://api.github.com/repos/{REPOSITORY}/contents/{REMOTE_PATH}?ref={commit}"),
    )


def tls_context(ca_file=None):
    # Keep the runtime's normal/proxy CA trust, and add the school's installed
    # botocore Mozilla bundle if available. TLS verification is never disabled.
    context = ssl.create_default_context()
    candidates = [Path(ca_file).expanduser()] if ca_file else [
        Path(sys.executable).parent.parent / "lib/python3.12/site-packages/botocore/cacert.pem",
        Path("/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            context.load_verify_locations(cafile=str(candidate))
            break
    if ca_file and not candidates[0].is_file():
        raise DownloadError("Requested CA file does not exist: " + str(candidates[0]))
    return context


def _quarantine_bad_partial(path, reason):
    # Preserve the rejected bytes for inspection, while allowing the next
    # attempt to start cleanly. Only this bootstrap's cache file is touched.
    rejected = path.with_name(path.name + ".invalid")
    os.replace(path, rejected)
    emit("DOWNLOADED_FILE_REJECTED", mirror_cache=path.name,
         retained_file=str(rejected), reason=reason)


def transfer_once(url, part, expected_size, expected_sha, *, context=None,
                  opener=None, clock=None):
    """Fetch/resume one mirror. Return True only for a verified whole file.

    There is no overall transfer-time limit. A stalled socket times out; bytes
    already received are flushed and retained for the next Range request.
    """
    part = Path(part)
    opener = opener or urllib.request.urlopen
    clock = clock or time.monotonic
    if part.is_file() and part.stat().st_size >= expected_size:
        if verified(part, expected_size, expected_sha):
            return True
        _quarantine_bad_partial(part, "cached_partial_size_or_sha256_mismatch")
    offset = part.stat().st_size if part.is_file() else 0
    headers = {"Accept": "application/vnd.github.raw+json",
               "Accept-Encoding": "identity", "User-Agent": "paper3-round7-resumable-download/1"}
    if offset:
        headers["Range"] = "bytes=" + str(offset) + "-"
    request = urllib.request.Request(url, headers=headers)
    emit("DOWNLOAD_RESUME" if offset else "DOWNLOAD_START", mirror_cache=part.name,
         received_bytes=offset, total_bytes=expected_size,
         percent=round(100 * offset / expected_size, 2))
    try:
        response = opener(request, timeout=SOCKET_TIMEOUT_SECONDS, context=context)
    except urllib.error.HTTPError as error:
        if error.code == 416 and verified(part, expected_size, expected_sha):
            error.close()
            return True
        error.close()
        raise DownloadError("HTTP " + str(error.code) + " for mirror " + part.name) from error
    with response:
        status = getattr(response, "status", None) or response.getcode()
        if response.headers.get("Content-Encoding", "identity").lower() not in ("identity", ""):
            raise DownloadError("Unexpected content encoding; cache retained")
        content_length = response.headers.get("Content-Length")
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                 response.headers.get("Content-Range", ""))
            if not match:
                raise DownloadError("Missing/invalid Content-Range; cache retained")
            first, last, total = map(int, match.groups())
            if first != offset or total != expected_size or last < first or last >= total:
                raise DownloadError("Content-Range differs from pinned bundle; cache retained")
            response_bytes = last - first + 1
            if content_length is not None and int(content_length) != response_bytes:
                raise DownloadError("Content-Length/Range disagree; cache retained")
            mode = "ab"
        elif status == 200:
            if content_length is not None and int(content_length) != expected_size:
                raise DownloadError("Full response size differs from pinned bundle; cache retained")
            if offset:
                emit("DOWNLOAD_RANGE_IGNORED_RESTART", mirror_cache=part.name,
                     retained_other_mirror=True)
            offset, mode, response_bytes = 0, "wb", expected_size
        else:
            raise DownloadError("Unexpected HTTP status " + str(status))
        received, transferred, next_progress = offset, 0, clock() + PROGRESS_SECONDS
        with part.open(mode, buffering=0) as handle:
            try:
                while True:
                    incomplete = None
                    try:
                        block = response.read(CHUNK_BYTES)
                    except http.client.IncompleteRead as error:
                        block, incomplete = error.partial, error
                    if block:
                        if received + len(block) > expected_size or transferred + len(block) > response_bytes:
                            raise DownloadError("Response exceeded pinned length; cache retained")
                        handle.write(block)
                        received += len(block)
                        transferred += len(block)
                        if clock() >= next_progress:
                            emit("DOWNLOADING", mirror_cache=part.name,
                                 received_bytes=received, total_bytes=expected_size,
                                 percent=round(100 * received / expected_size, 2))
                            next_progress = clock() + PROGRESS_SECONDS
                    if incomplete is not None:
                        raise DownloadError("Transfer interrupted; received prefix retained") from incomplete
                    if not block:
                        break
            finally:
                os.fsync(handle.fileno())
        if transferred != response_bytes or received != expected_size:
            raise DownloadError("Transfer ended before whole bundle; received prefix retained")
    if not verified(part, expected_size, expected_sha):
        _quarantine_bad_partial(part, "download_sha256_mismatch")
        raise DownloadError("Downloaded bytes failed SHA256; never executed")
    return True


def obtain(target, expected_size, expected_sha, *, urls=None, attempts=0,
           context=None, opener=None, sleep=None):
    """Obtain a whole cached bundle. attempts=0 retries until success/Ctrl+C."""
    target = Path(target)
    if (expected_size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or attempts < 0):
        raise DownloadError("Invalid pinned bundle metadata")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    urls = tuple(urls or publication_urls())
    if not urls or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name, _ in urls):
        raise DownloadError("Invalid mirror names")
    sleep = sleep or time.sleep
    emit("CHECKING_DOWNLOAD_CACHE", file=str(target))
    with target.with_name(target.name + ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if verified(target, expected_size, expected_sha):
            emit("DOWNLOAD_CACHE_VERIFIED", file=str(target), bytes=expected_size,
                 network_accessed=False)
            return target
        round_number = 0
        while True:
            round_number += 1
            for mirror, url in urls:
                part = target.with_name(target.name + "." + mirror + ".part")
                try:
                    transfer_once(url, part, expected_size, expected_sha,
                                  context=context, opener=opener)
                    # A second hash at publication is cheap and ensures only
                    # the exact immutable file can become the executable input.
                    if not verified(part, expected_size, expected_sha):
                        raise DownloadError("Completed partial changed before publication")
                    os.replace(part, target)
                    target.chmod(0o600)
                    directory = os.open(str(target.parent), os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                    emit("DOWNLOADED_VERIFIED", file=str(target), bytes=expected_size,
                         sha256=expected_sha, attempts=round_number)
                    return target
                except (DownloadError, OSError, ValueError, urllib.error.URLError,
                        http.client.HTTPException) as error:
                    emit("DOWNLOAD_TRANSFER_RETRY", mirror=mirror, attempt=round_number,
                         partial_bytes=part.stat().st_size if part.is_file() else 0,
                         reason=str(error))
            if attempts and round_number >= attempts:
                raise AttemptsExhausted("No verified bundle yet; partial caches retained. Run the same command to continue.")
            delay = min(30, 2 ** min(round_number, 5))
            emit("WAITING_DOWNLOAD_RETRY", attempt=round_number,
                 retry_in_seconds=delay, training_started=False, jobs_submitted=False)
            sleep(delay)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--round", type=int, choices=(4, 5, 6), required=True)
    parser.add_argument("--download-dir", default=str(Path.home() / "paper3-hpcgpu-downloads"))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--attempts", type=int, default=0)
    parser.add_argument("--ca-file")
    args, forwarded = parser.parse_known_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT):
        raise DownloadError("Publication source commit has not been filled")
    info = ROUND_PACKAGES[str(args.round)]
    path = info["path"]
    urls = (("raw", f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/{path}"),
            ("api", f"https://api.github.com/repos/{REPOSITORY}/contents/{path}?ref={SOURCE_COMMIT}"))
    target = obtain(Path(args.download_dir).expanduser() / Path(path).name,
                    info["bytes"], info["sha256"], urls=urls, attempts=args.attempts,
                    context=tls_context(args.ca_file))
    emit("ROUND_MIGRATION_PACKAGE_VERIFIED", round=args.round, file=str(target),
         partition="hpc_gpu", nodes=1, slurm_jobs=1, gpus=3, cpus_total=36, memory_total="128G")
    if args.download_only:
        return 0
    return subprocess.call([sys.executable, "-I", "-S", "-B", "-u", str(target),
        "--partition", "hpc_gpu", "--gpus", "3", "--cpus", "12", "--memory-total", "128G", *forwarded])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        emit("DOWNLOAD_PAUSED_CACHE_PRESERVED", training_started=False)
        raise SystemExit(130)
    except (DownloadError, OSError, ValueError) as error:
        print("DOWNLOAD_NOT_COMPLETE: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
