"""Stdlib GitHub result transport; independent of training and reporting code."""
import base64
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class ApiError(Exception):
    def __init__(self, status, message="GitHub rejected the request"):
        self.status = status
        # Never propagate server response bodies, request headers or credentials.
        super().__init__("%s (HTTP %s)" % (message, status))


class _TransportRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ApiError(code, "Non-HTTPS GitHub redirect refused")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(newurl).hostname not in (
                "api.github.com", "uploads.github.com"):
            for key in list(redirected.headers):
                if key.lower() == "authorization":
                    del redirected.headers[key]
            for key in list(redirected.unredirected_hdrs):
                if key.lower() == "authorization":
                    del redirected.unredirected_hdrs[key]
        return redirected


def _transport_context():
    context = ssl.create_default_context()
    prefixes = {Path(sys.executable).absolute().parent.parent,
                Path(sys.executable).resolve().parent.parent}
    candidates = []
    for prefix in prefixes:
        for directory in prefix.glob("lib/python*/site-packages"):
            candidates.extend((directory / "botocore/cacert.pem", directory / "certifi/cacert.pem"))
    # -S deliberately omits site-packages, but its CA bundle remains usable.
    for directory in sys.path:
        if directory:
            candidates.extend((Path(directory) / "botocore/cacert.pem",
                               Path(directory) / "certifi/cacert.pem"))
    for path in candidates:
        if path.is_file():
            context.load_verify_locations(cafile=str(path))
    return context


class GitHub:
    def __init__(self, token, emit, *, repo="jackzhu119/research3", branch="paper3-results",
                 context=None, timeout=60, cache_path=None):
        self._token = token
        self.emit = emit
        self.repo, self.branch = repo, branch
        self.prefix = "repos/" + repo
        self.timeout = timeout
        self.context = context or _transport_context()
        self.opener = urllib.request.build_opener(
            _TransportRedirect(), urllib.request.HTTPSHandler(context=self.context))
        self.cache_path = Path(cache_path) if cache_path else None
        try:
            self.blob_cache = json.loads(self.cache_path.read_text()) if self.cache_path else {}
            if not isinstance(self.blob_cache, dict):
                self.blob_cache = {}
        except (OSError, ValueError):
            self.blob_cache = {}

    def _emit(self, event, **fields):
        self.emit(event, **fields)

    def _delay(self, attempt, headers=None):
        delay = min(60, 2 ** min(attempt, 6))
        headers = headers or {}
        try:
            requested = float(headers.get("Retry-After", "0"))
            delay = min(60, max(delay, requested))
        except (TypeError, ValueError):
            pass
        self._emit("GITHUB_RETRY", attempt=attempt, delay_seconds=delay)
        time.sleep(delay)

    @staticmethod
    def _retryable(error, body=b""):
        if error.code in (408, 425, 429, 500, 502, 503, 504):
            return True
        if error.code == 403:
            return (error.headers.get("X-RateLimit-Remaining") == "0"
                    or bool(error.headers.get("Retry-After"))
                    or b"secondary rate limit" in body.lower()
                    or b"rate limit exceeded" in body.lower())
        return False

    def _request(self, url, data=None, method=None, *, accept="application/vnd.github+json",
                 content_type="application/json"):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in ("api.github.com", "uploads.github.com"):
            raise ValueError("Unsupported GitHub endpoint")
        request = urllib.request.Request(url, data=data, method=method,
            headers={"Authorization": "Bearer " + self._token,
                     "User-Agent": "paper3-results-archive-uploader",
                     "Accept": accept, "Content-Type": content_type,
                     "X-GitHub-Api-Version": "2022-11-28"})
        return self.opener.open(request, timeout=self.timeout)

    def api(self, path, body=None, method=None, allow404=False):
        url = path if path.startswith("https://") else "https://api.github.com/" + path.lstrip("/")
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
        attempt = 0
        while True:
            try:
                with self._request(url, encoded, method) as response:
                    raw = response.read()
                return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as error:
                raw = error.read(65536)
                if allow404 and error.code == 404:
                    return None
                if not self._retryable(error, raw):
                    raise ApiError(error.code) from None
                attempt += 1
                self._delay(attempt, error.headers)
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                attempt += 1
                self._emit("GITHUB_NETWORK_RETRY", error_type=type(error).__name__)
                self._delay(attempt)

    def ensure_branch(self):
        endpoint = self.prefix + "/git/ref/heads/" + urllib.parse.quote(self.branch, safe="/")
        found = self.api(endpoint, allow404=True)
        if found is not None:
            return found["object"]["sha"]
        default = self.api(self.prefix)["default_branch"]
        head = self.api(self.prefix + "/git/ref/heads/" + urllib.parse.quote(default, safe="/"))["object"]["sha"]
        try:
            return self.api(self.prefix + "/git/refs",
                {"ref": "refs/heads/" + self.branch, "sha": head}, "POST")["object"]["sha"]
        except ApiError as error:
            if error.status not in (409, 422):
                raise
            # Another uploader, or a lost creation acknowledgement, can create it.
            return self.api(endpoint)["object"]["sha"]

    def ensure_release(self, tag, title, body):
        endpoint = self.prefix + "/releases/tags/" + urllib.parse.quote(tag, safe="")
        found = self.api(endpoint, allow404=True)
        if found is not None:
            return found
        self.ensure_branch()
        try:
            return self.api(self.prefix + "/releases", {
                "tag_name": tag, "target_commitish": self.branch, "name": title,
                "body": body, "draft": False, "prerelease": False,
                "make_latest": "false"}, "POST")
        except ApiError as error:
            if error.status not in (409, 422):
                raise
            found = self.api(endpoint, allow404=True)
            if found is None:
                raise
            return found

    def _assets(self, release):
        assets, page = [], 1
        while True:
            batch = self.api(self.prefix + "/releases/%s/assets?per_page=100&page=%s" % (release["id"], page))
            assets.extend(batch)
            if len(batch) < 100:
                return assets
            page += 1

    def _asset_hash(self, asset):
        digest = asset.get("digest") or ""
        if digest.startswith("sha256:"):
            return digest[7:]
        attempt = 0
        while True:
            try:
                digest = hashlib.sha256()
                with self._request(self.prefix_url + "/releases/assets/%s" % asset["id"],
                                   accept="application/octet-stream") as response:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                return digest.hexdigest()
            except urllib.error.HTTPError as error:
                raw = error.read(65536)
                if not self._retryable(error, raw):
                    raise ApiError(error.code, "Cannot verify uploaded archive part") from None
                attempt += 1
                self._delay(attempt, error.headers)
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                attempt += 1
                self._emit("ASSET_VERIFY_RETRY", error_type=type(error).__name__)
                self._delay(attempt)

    @property
    def prefix_url(self):
        return "https://api.github.com/" + self.prefix

    def upload_part(self, release, path, name, sha256):
        path = Path(path)
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise ValueError("Local archive part hash differs from its manifest")
        upload_url = release["upload_url"].split("{", 1)[0]
        upload_url += "?" + urllib.parse.urlencode({"name": name})
        attempt = 0
        while True:
            # Always re-list before POST, including after a lost POST response.
            matches = [a for a in self._assets(release) if a.get("name") == name]
            for asset in matches:
                if (asset.get("state") == "uploaded" and asset.get("size") == size
                        and self._asset_hash(asset) == sha256):
                    self._emit("ARCHIVE_PART_VERIFIED", name=name, bytes=size, reused=True)
                    return {"id": asset["id"], "name": name, "bytes": size, "sha256": sha256,
                            "url": asset.get("browser_download_url"), "reused": True}
                # This asset was found in this exact release with this exact name.
                self.api(self.prefix + "/releases/assets/%s" % asset["id"], method="DELETE", allow404=True)
            try:
                self._emit("ARCHIVE_PART_UPLOADING", name=name, bytes=size)
                with self._request(upload_url, path.read_bytes(), "POST",
                                   content_type="application/octet-stream") as response:
                    created = json.load(response)
                if (created.get("state") != "uploaded" or created.get("size") != size
                        or self._asset_hash(created) != sha256):
                    self._emit("ARCHIVE_PART_VERIFY_RETRY", name=name)
                    attempt += 1
                    self._delay(attempt)
                    continue
                self._emit("ARCHIVE_PART_VERIFIED", name=name, bytes=size, reused=False)
                return {"id": created["id"], "name": name, "bytes": size, "sha256": sha256,
                        "url": created.get("browser_download_url"), "reused": False}
            except urllib.error.HTTPError as error:
                raw = error.read(65536)
                # 422 can acknowledge an already completed POST whose reply was lost.
                if not self._retryable(error, raw) and error.code not in (409, 422):
                    raise ApiError(error.code, "Archive part upload rejected") from None
                attempt += 1
                self._delay(attempt, error.headers)
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                attempt += 1
                self._emit("ARCHIVE_PART_RETRY", name=name, error_type=type(error).__name__)
                self._delay(attempt)

    def _save_cache(self):
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.cache_path.with_name(self.cache_path.name + ".tmp.%s" % os.getpid())
        temp.write_text(json.dumps(self.blob_cache, sort_keys=True))
        os.replace(temp, self.cache_path)

    def _blob(self, content):
        if not isinstance(content, bytes):
            raise TypeError("Git content must be bytes")
        # Git SHA is deterministic and works across repository branches/reruns.
        expected = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if self.blob_cache.get(expected) == expected:
            return expected
        result = self.api(self.prefix + "/git/blobs", {
            "content": base64.b64encode(content).decode(), "encoding": "base64"}, "POST")
        if result.get("sha") != expected:
            raise ValueError("GitHub blob identity mismatch")
        self.blob_cache[expected] = expected
        self._save_cache()
        return expected

    def git_publish(self, files, message):
        self.ensure_branch()
        entries = []
        for path, content in sorted(files.items()):
            pure = Path(path)
            if pure.is_absolute() or ".." in pure.parts or not path:
                raise ValueError("Unsafe result repository path")
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": self._blob(content)})
        endpoint = self.prefix + "/git/ref/heads/" + urllib.parse.quote(self.branch, safe="/")
        attempt = 0
        while True:
            head = self.api(endpoint)["object"]["sha"]
            base_tree = self.api(self.prefix + "/git/commits/" + head)["tree"]["sha"]
            tree = self.api(self.prefix + "/git/trees", {"base_tree": base_tree, "tree": entries}, "POST")["sha"]
            if tree == base_tree:
                self._emit("RESULT_INDEX_UNCHANGED", branch=self.branch, commit=head)
                return head
            commit = self.api(self.prefix + "/git/commits", {
                "message": message, "tree": tree, "parents": [head]}, "POST")["sha"]
            try:
                self.api(self.prefix + "/git/refs/heads/" + urllib.parse.quote(self.branch, safe="/"),
                         {"sha": commit, "force": False}, "PATCH")
                self._emit("RESULT_INDEX_PUBLISHED", branch=self.branch, commit=commit)
                return commit
            except ApiError as error:
                if error.status not in (409, 422):
                    raise
                attempt += 1
                self._emit("RESULT_INDEX_REBASE_RETRY", branch=self.branch)
                self._delay(attempt)
