"""Publish only an owned, fixed Paper3 code snapshot to a new GitHub branch.

No credentials are persisted. The caller supplies an in-memory PAT obtained
through hidden input; no API in this module prompts or launches training.
"""
from __future__ import annotations

import hashlib
import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from urllib import error, parse, request


class PublishError(ValueError):
    """A fixed, non-sensitive error code; never SDK/Git/HTTP exception text."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


_DEFAULT_URLOPEN = request.urlopen
_ROOT_FILES = frozenset({
    "README.md", "pyproject.toml", "requirements-cpu.lock",
    "requirements-gpu.lock", "paper3_one_click.py", "paper3_transport.py",
    "paper3_github.py", "release_manifest.json", "code_snapshot_manifest.json",
})
_FINALIZER = "audit/data_plan_20261020/deployment_finalize_review/prepare_finalize_plan_v2.py"
_STATE = "paper3_publish_state.json"
_INITIALIZING = "paper3_initializing.json"
_GIT_OPTIONS = [
    "-c", "credential.helper=", "-c", "http.followRedirects=false",
    "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
    "-c", "commit.gpgsign=false", "-c", "protocol.allow=never",
    "-c", "protocol.https.allow=always", "-c", "protocol.file.allow=never",
    "-c", "transfer.fsckobjects=true",
]


def _need(condition, code):
    if not condition:
        raise PublishError(code)


def validate_repo(text):
    _need(isinstance(text, str) and text and text == text.strip()
          and not any(ord(c) < 33 or ord(c) == 127 for c in text), "GITHUB_REPO_INVALID")
    if "://" in text:
        try:
            value = parse.urlsplit(text)
        except ValueError:
            raise PublishError("GITHUB_REPO_INVALID") from None
        _need(value.scheme == "https" and value.hostname == "github.com"
              and value.netloc == "github.com" and not value.username
              and not value.password and not value.query and not value.fragment,
              "GITHUB_REPO_INVALID")
        text = value.path.removeprefix("/")
    if text.endswith(".git"):
        text = text[:-4]
    _need(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9_.-]{1,100}", text)
          is not None, "GITHUB_REPO_INVALID")
    _need(not text.split("/")[1].startswith(".")
          and not text.split("/")[1].endswith(".")
          and text.split("/")[1] not in {".", ".."}, "GITHUB_REPO_INVALID")
    return text


def validate_branch(text):
    _need(isinstance(text, str) and 1 < len(text) <= 200
          and text.startswith(("research/", "paper3/"))
          and re.fullmatch(r"[A-Za-z0-9_./-]+", text) is not None
          and not any(x in text for x in ("..", "//", "@{"))
          and not text.endswith(("/", ".", ".lock"))
          and all(part and not part.startswith(".") and not part.endswith(".lock")
                  for part in text.split("/")), "GITHUB_RESEARCH_BRANCH_INVALID")
    return text


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PublishError("GITHUB_REDIRECT_FORBIDDEN")


def _api(path, token, urlopen):
    url = "https://api.github.com" + path
    req = request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "Paper3-fixed-code-launcher",
        "Authorization": "Bearer " + token, "X-GitHub-Api-Version": "2022-11-28",
    })
    response = None
    try:
        if urlopen is _DEFAULT_URLOPEN:
            # Default ProxyHandler and HTTPS trust remain inherited. No redirect
            # can forward the Authorization header to another destination.
            response = request.build_opener(_NoRedirect()).open(req, timeout=30)
        else:
            response = urlopen(req, timeout=30)
        _need(response.getcode() == 200 and response.geturl() == url,
              "GITHUB_API_RESPONSE_INVALID")
        body = response.read(1024 * 1024 + 1)
        _need(len(body) <= 1024 * 1024, "GITHUB_API_RESPONSE_INVALID")
        value = json.loads(body)
        _need(isinstance(value, dict), "GITHUB_API_RESPONSE_INVALID")
        return value
    except PublishError:
        raise
    except error.HTTPError as exc:
        if exc.code == 401:
            raise PublishError("GITHUB_TOKEN_REJECTED") from None
        if exc.code == 404:
            raise PublishError("GITHUB_REPOSITORY_NOT_FOUND_OR_NOT_AUTHORIZED") from None
        if exc.code == 403:
            raise PublishError("GITHUB_PERMISSION_OR_RATE_LIMIT") from None
        raise PublishError("GITHUB_API_HTTP_FAILED") from None
    except Exception:
        raise PublishError("GITHUB_API_CONNECTION_OR_JSON_FAILED") from None
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _hash(path):
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        blob = hashlib.sha1(b"blob " + str(before.st_size).encode() + b"\0", usedforsecurity=False)
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
            blob.update(data)
            n += len(data)
        after = os.fstat(stream.fileno())
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    _need(identity(before) == identity(after) == identity(path.stat()), "CODE_CHANGED_DURING_READ")
    return h.hexdigest(), n, blob.hexdigest()


def _inventory(root):
    _need(root.is_absolute() and root.is_dir()
          and not any(p.is_symlink() for p in (root, *root.parents)), "CODE_ROOT_INVALID")
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            continue
        _need(not path.is_symlink(), "CODE_SYMLINK_FORBIDDEN")
        if path.is_dir():
            continue
        _need(path.is_file(), "CODE_REGULAR_FILE_REQUIRED")
        allowed = (relative in _ROOT_FILES or relative == _FINALIZER
                   or (path.relative_to(root).parts[0] in {"p3work", "tools", "tests"}
                       and path.suffix == ".py"))
        _need(allowed and not any(part.startswith(".") for part in path.relative_to(root).parts),
              "NON_CODE_OR_PRIVATE_FILE_FORBIDDEN")
        _need(re.fullmatch(r"[A-Za-z0-9_./-]+", relative) is not None,
              "CODE_FILENAME_INVALID")
        sha, size, blob = _hash(path)
        rows.append({"path": relative, "sha256": sha, "bytes": size, "git_blob_sha1": blob})
    _need(rows and any(x["path"].startswith("p3work/") for x in rows), "CODE_SNAPSHOT_EMPTY")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rows, digest


def _git_environment(helper, token):
    trust_paths = {"GIT_SSL_CAINFO", "GIT_SSL_CAPATH", "GIT_PROXY_SSL_CAINFO", "GIT_PROXY_SSL_CAPATH"}
    env = {k: v for k, v in os.environ.items()
           if (not k.startswith(("GIT_", "R2_", "AWS_", "GH_", "GITHUB_")) or k in trust_paths)
           and k not in {"PAPER3_GITHUB_PAT", "PYTHONHOME", "PYTHONPATH"}}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_ASKPASS=str(helper),
               PAPER3_GITHUB_PAT=token, GIT_LFS_SKIP_SMUDGE="1")
    return env


def _git(root, env, runner, args):
    try:
        result = runner(["git", *_GIT_OPTIONS, *args], cwd=root, env=env,
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, check=False)
        _need(result.returncode == 0, "GITHUB_GIT_COMMAND_FAILED")
        return result.stdout
    except PublishError:
        raise
    except Exception:
        raise PublishError("GITHUB_GIT_COMMAND_FAILED") from None


def _remote(root, env, runner, repo, branch):
    text = _git(root, env, runner, ["ls-remote", "--heads", "https://github.com/" + repo + ".git",
                                   "refs/heads/" + branch]).strip()
    if not text:
        return None
    parts = text.split()
    _need(len(parts) == 2 and re.fullmatch(r"[0-9a-f]{40}", parts[0])
          and parts[1] == "refs/heads/" + branch, "GITHUB_REMOTE_SHA_INVALID")
    return parts[0]


def _verify_index(root, env, runner, rows):
    entries = _git(root, env, runner, ["ls-files", "--stage", "-z"]).split("\0")
    found = {}
    for entry in filter(None, entries):
        match = re.fullmatch(r"(100644|100755) ([0-9a-f]{40}) 0\t(.+)", entry)
        _need(match is not None and match[3] not in found, "EXACT_CODE_ONLY_INDEX_REQUIRED")
        found[match[3]] = match[2]
    _need(found == {r["path"]: r["git_blob_sha1"] for r in rows},
          "EXACT_CODE_ONLY_INDEX_REQUIRED")


def _write_state(path, value):
    _need(not path.is_symlink(), "PUBLISH_STATE_INVALID")
    body = (json.dumps(value, indent=2) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix="paper3-state-", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _initialization_marker(root, repo, branch, inventory_sha, code_files):
    identity = root.stat()
    return {"schema_version": "paper3-owned-git-initializing-v1", "root": str(root),
            "root_dev": identity.st_dev, "root_ino": identity.st_ino,
            "repo": repo, "branch": branch, "inventory_sha256": inventory_sha,
            "code_files": code_files}


def _check_initialization_marker(gitdir, marker):
    path = gitdir / _INITIALIZING
    _need(path.is_file() and not path.is_symlink(), "UNOWNED_GIT_DIRECTORY")
    try:
        with path.open("rb") as stream:
            raw = stream.read(16385)
        _need(len(raw) <= 16384 and json.loads(raw) == marker,
              "PUBLISH_INITIALIZATION_MARKER_INVALID")
    except PublishError:
        raise
    except Exception:
        raise PublishError("PUBLISH_INITIALIZATION_MARKER_INVALID") from None


def _renameat2_syscall_number():
    # glibc before 2.28 has no renameat2 wrapper. These are Linux LP64 syscall
    # ABIs, not arbitrary CPU labels: x32 and unknown ABIs must stay closed.
    if (not sys.platform.startswith("linux") or ctypes.sizeof(ctypes.c_void_p) != 8
            or ctypes.sizeof(ctypes.c_long) != 8):
        return None
    return {"x86_64": 316, "amd64": 316, "aarch64": 276, "arm64": 276}.get(
        platform.machine().lower())


def _rename_no_replace(source, destination):
    # The training target is Linux. Plain rename can replace an unknown empty
    # destination directory; Linux RENAME_NOREPLACE atomically forbids that.
    try:
        library = ctypes.CDLL(None, use_errno=True)
    except (AttributeError, OSError):
        raise PublishError("ATOMIC_GIT_INITIALIZATION_UNSUPPORTED") from None
    try:
        rename = library.renameat2
    except AttributeError:
        number = _renameat2_syscall_number()
        if number is None:
            raise PublishError("ATOMIC_GIT_INITIALIZATION_UNSUPPORTED") from None
        try:
            syscall = library.syscall
            syscall.argtypes = [ctypes.c_long]
            syscall.restype = ctypes.c_long
        except (AttributeError, OSError):
            raise PublishError("ATOMIC_GIT_INITIALIZATION_UNSUPPORTED") from None
        # Use exactly the same kernel RENAME_NOREPLACE operation. A missing
        # kernel/filesystem feature never falls back to a replacing rename.
        result = syscall(ctypes.c_long(number), ctypes.c_int(-100),
                         ctypes.c_char_p(os.fsencode(source)), ctypes.c_int(-100),
                         ctypes.c_char_p(os.fsencode(destination)), ctypes.c_uint(1))
    except OSError:
        raise PublishError("ATOMIC_GIT_INITIALIZATION_UNSUPPORTED") from None
    else:
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise PublishError("UNOWNED_GIT_DIRECTORY")
        if number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
            raise PublishError("ATOMIC_GIT_INITIALIZATION_UNSUPPORTED")
        raise PublishError("GITHUB_LOCAL_INITIALIZATION_FAILED")


def _install_initializing_git(root, marker):
    # A hard kill before the rename leaves no .git; a hard kill after it leaves
    # a fully written marker that can prove ownership before initialization.
    with tempfile.TemporaryDirectory(prefix=".paper3-git-handoff-", dir=root.parent) as staging:
        candidate = Path(staging) / "git"
        candidate.mkdir(mode=0o700)
        _write_state(candidate / _INITIALIZING, marker)
        _sync_directory(candidate)
        _rename_no_replace(candidate, root / ".git")
        _sync_directory(root)
    identity = (root / ".git").stat()
    return identity.st_dev, identity.st_ino


def _discard_initializing_git(root, marker, expected_identity=None):
    gitdir = root / ".git"
    _need(not gitdir.is_symlink() and gitdir.is_dir(), "UNOWNED_GIT_DIRECTORY")
    identity = gitdir.stat()
    _need(expected_identity is None or (identity.st_dev, identity.st_ino) == expected_identity,
          "GITHUB_INITIALIZATION_DIRECTORY_CHANGED")
    _check_initialization_marker(gitdir, marker)
    _need(not (gitdir / _STATE).exists(), "OWNED_PUBLISH_STATE_MUST_BE_PRESERVED")
    # Move ownership-verified partial metadata out atomically before deletion.
    # A hard kill during removal cannot leave a half-deleted .git in code_dir.
    with tempfile.TemporaryDirectory(prefix=".paper3-git-discard-", dir=root.parent) as discarded:
        _rename_no_replace(gitdir, Path(discarded) / "git")
        _sync_directory(root)


def publish_code(repo, branch, token, code_dir, *, runner=subprocess.run,
                 urlopen=request.urlopen):
    repo, branch = validate_repo(repo), validate_branch(branch)
    _need(isinstance(token, str) and 1 <= len(token) <= 4096
          and all(33 <= ord(c) <= 126 for c in token), "GITHUB_TOKEN_INVALID")
    root = Path(os.path.abspath(code_dir))
    rows, inventory_sha = _inventory(root)
    initializing = _initialization_marker(root, repo, branch, inventory_sha, len(rows))
    user = _api("/user", token, urlopen)
    metadata = _api("/repos/" + repo, token, urlopen)
    login, uid = user.get("login"), user.get("id")
    _need(isinstance(login, str) and re.fullmatch(r"[A-Za-z0-9-]{1,39}", login)
          and type(uid) is int and uid > 0, "GITHUB_USER_METADATA_INVALID")
    permissions = metadata.get("permissions")
    _need(isinstance(metadata.get("full_name"), str)
          and metadata["full_name"].lower() == repo.lower()
          and isinstance(permissions, dict) and permissions.get("push") is True,
          "GITHUB_REPOSITORY_WRITE_PERMISSION_REQUIRED")
    is_default_branch = branch == metadata.get("default_branch")
    gitdir = root / ".git"
    existing = gitdir.exists()
    _need(not gitdir.is_symlink() and (not existing or gitdir.is_dir()), "UNOWNED_GIT_DIRECTORY")
    statefile = gitdir / _STATE
    state = None
    if existing and not statefile.exists():
        _discard_initializing_git(root, initializing)
        existing = False
    if existing:
        _need(statefile.is_file() and not statefile.is_symlink(), "UNOWNED_GIT_DIRECTORY")
        try:
            state = json.loads(statefile.read_bytes())
        except Exception:
            raise PublishError("PUBLISH_STATE_INVALID") from None
        _need(isinstance(state, dict), "PUBLISH_STATE_INVALID")
        _need(state.get("schema_version") == "paper3-owned-code-publish-v1"
              and state.get("repo") == repo and state.get("branch") == branch
              and state.get("inventory_sha256") == inventory_sha
              and state.get("code_files") == len(rows), "PUBLISH_STATE_OR_SOURCE_CHANGED")
        _need((gitdir / "config").is_file() and not (gitdir / "config").is_symlink()
              and _hash(gitdir / "config")[0] == state.get("local_config_sha256"),
              "PUBLISH_GIT_CONFIG_CHANGED")
        if (gitdir / _INITIALIZING).exists():
            _check_initialization_marker(gitdir, initializing)
            (gitdir / _INITIALIZING).unlink()
            _sync_directory(gitdir)
    # GitHub may make the first branch of an empty repository its default.
    # An exact owned commit may still be verified without another push.
    _need(not is_default_branch or existing, "GITHUB_DEFAULT_BRANCH_FORBIDDEN")
    with tempfile.TemporaryDirectory(prefix="paper3-git-auth-") as directory:
        helper = Path(directory) / "askpass.py"
        # Only a variable NAME is persisted, never its value. stdout here is
        # Git's private askpass pipe, not this launcher's terminal or logs.
        helper.write_text("#!" + sys.executable + "\n"
            "import os,re,sys\n"
            "p=sys.argv[1] if len(sys.argv)>1 else ''\n"
            "if not re.search(r'https://(?:[^/@\\s]+@)?github\\.com(?:[:/\\\']|$)',p):sys.exit(1)\n"
            "if p.lower().startswith('username'):print('x-access-token')\n"
            "elif p.lower().startswith('password'):print(os.environ['PAPER3_GITHUB_PAT'])\n"
            "else:sys.exit(1)\n")
        helper.chmod(0o700)
        env = _git_environment(helper, token)
        created_git_identity = None
        owned_state_saved = existing
        try:
            remote_sha = _remote(root, env, runner, repo, branch)
            if not existing:
                _need(remote_sha is None, "GITHUB_BRANCH_ALREADY_EXISTS_CHOOSE_NEW_BRANCH")
                created_git_identity = _install_initializing_git(root, initializing)
                _git(root, env, runner, ["init"])
                _git(root, env, runner, ["symbolic-ref", "HEAD", "refs/heads/" + branch])
                _git(root, env, runner, ["config", "user.name", login])
                _git(root, env, runner, ["config", "user.email", f"{uid}+{login}@users.noreply.github.com"])
                _git(root, env, runner, ["add", "--", *[r["path"] for r in rows]])
                _verify_index(root, env, runner, rows)
                _git(root, env, runner, ["commit", "-m", "Add fixed Paper3 code and one-click launcher"])
                commit = _git(root, env, runner, ["rev-parse", "HEAD"]).strip()
                _need(re.fullmatch(r"[0-9a-f]{40}", commit), "LOCAL_COMMIT_SHA_INVALID")
                state = {"schema_version": "paper3-owned-code-publish-v1", "repo": repo,
                         "branch": branch, "commit": commit, "code_files": len(rows),
                         "inventory_sha256": inventory_sha,
                         "local_config_sha256": _hash(gitdir / "config")[0],
                         "remote_sha_verified": False}
                _write_state(statefile, state)
                owned_state_saved = True
                (gitdir / _INITIALIZING).unlink()
                _sync_directory(gitdir)
            else:
                commit = _git(root, env, runner, ["rev-parse", "HEAD"]).strip()
                _need(commit == state.get("commit")
                      and _git(root, env, runner, ["symbolic-ref", "--short", "HEAD"]).strip() == branch
                      and not _git(root, env, runner, ["status", "--porcelain"]).strip(),
                      "OWNED_PUBLISH_WORKTREE_CHANGED")
                _verify_index(root, env, runner, rows)
                _need(remote_sha in {None, commit}, "GITHUB_REMOTE_BRANCH_CHANGED_NO_OVERWRITE")
                _need(not is_default_branch or remote_sha == commit,
                      "GITHUB_DEFAULT_BRANCH_FORBIDDEN")
            _need(_inventory(root)[1] == inventory_sha, "CODE_CHANGED_BEFORE_PUSH")
            if remote_sha is None:
                _git(root, env, runner, ["push", "https://github.com/" + repo + ".git",
                                        "HEAD:refs/heads/" + branch])
            verified = _remote(root, env, runner, repo, branch)
            _need(verified == commit, "GITHUB_PUSH_REMOTE_SHA_NOT_VERIFIED")
            state["remote_sha_verified"] = True
            _write_state(statefile, state)
            return {"status": "GITHUB_CODE_PUSH_AND_REMOTE_SHA_VERIFIED", "repo": repo,
                    "branch": branch, "commit": commit, "code_files": len(rows),
                    "inventory_sha256": inventory_sha, "reused_verified_remote": remote_sha == commit}
        finally:
            env.pop("PAPER3_GITHUB_PAT", None)
            if created_git_identity is not None and not owned_state_saved:
                # The outer launcher holds its lock. Remove only the fresh local
                # metadata this call created before a resumable commit existed;
                # never delete an existing Git directory or any source file.
                try:
                    _discard_initializing_git(root, initializing, created_git_identity)
                except PublishError:
                    raise
                except Exception:
                    raise PublishError("GITHUB_INITIALIZATION_CLEANUP_REQUIRED") from None
