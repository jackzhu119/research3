"""Local Git and injected HTTP fixtures: this test suite never contacts GitHub."""
import io
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper3_github as gh


TOKEN = "fixture-pat-never-persist"
REPO = "owner/research3"
BRANCH = "research/paper3-test-123456"


def _without_renameat2(real_cdll):
    """Hide only the new glibc wrapper, retaining the real kernel syscall."""
    def load(*args, **kwargs):
        library = real_cdll(*args, **kwargs)
        class Glibc217:
            def __getattr__(self, name):
                if name == "renameat2":
                    raise AttributeError(name)
                return getattr(library, name)
        return Glibc217()
    return load


def test_publish_error_exposes_only_the_fixed_failure_code():
    error = gh.PublishError('GITHUB_TOKEN_REJECTED')
    assert error.code == 'GITHUB_TOKEN_REJECTED' and str(error) == error.code


def test_git_environment_preserves_explicit_trust_only(monkeypatch):
    monkeypatch.setattr(gh.os, "environ", {
        "PATH": "/usr/bin:/bin", "HTTPS_PROXY": "https://proxy.invalid",
        "SSL_CERT_FILE": "/fixture/ca.pem", "GIT_SSL_CAINFO": "/fixture/git-ca.pem",
        "GIT_SSL_CAPATH": "/fixture/certs", "GIT_PROXY_SSL_CAINFO": "/fixture/proxy-ca.pem",
        "GIT_PROXY_SSL_CAPATH": "/fixture/proxy-certs", "GIT_TRACE": "remove-me",
        "GIT_CONFIG_COUNT": "1", "GITHUB_PAT": "remove-old-github-pat",
        "GH_ENTERPRISE_TOKEN": "remove-old-enterprise-pat", "PYTHONHOME": "/old/python",
        "PYTHONPATH": "/old/site", "AWS_SESSION_TOKEN": "remove-old-aws-session"})
    env = gh._git_environment(Path('/fixture/askpass'), TOKEN)
    assert env['GIT_SSL_CAINFO'] == '/fixture/git-ca.pem'
    assert env['GIT_SSL_CAPATH'] == '/fixture/certs'
    assert env['GIT_PROXY_SSL_CAINFO'] == '/fixture/proxy-ca.pem'
    assert env['GIT_PROXY_SSL_CAPATH'] == '/fixture/proxy-certs'
    assert env['HTTPS_PROXY'] == 'https://proxy.invalid'
    assert env['SSL_CERT_FILE'] == '/fixture/ca.pem'
    assert not set(env) & {'GIT_TRACE', 'GIT_CONFIG_COUNT', 'GITHUB_PAT',
                           'GH_ENTERPRISE_TOKEN', 'PYTHONHOME', 'PYTHONPATH', 'AWS_SESSION_TOKEN'}


class Response(io.BytesIO):
    def __init__(self, value, url):
        super().__init__(json.dumps(value).encode())
        self.url = url

    def getcode(self):
        return 200

    def geturl(self):
        return self.url


class API:
    def __init__(self, overrides=None, redirect=False):
        self.overrides = overrides or {}
        self.redirect = redirect
        self.urls = []

    def __call__(self, req, timeout):
        assert timeout == 30
        assert req.get_header("Authorization") == "Bearer " + TOKEN
        self.urls.append(req.full_url)
        path = req.full_url.removeprefix("https://api.github.com")
        values = {"/user": {"login": "paper3-user", "id": 123},
                  "/repos/owner/research3": {"full_name": REPO, "permissions": {"push": True},
                                             "default_branch": "main"}}
        value = self.overrides.get(path, values[path])
        return Response(value, "https://evil.invalid/" if self.redirect else req.full_url)


class LocalGit:
    """Run Git locally; intercept every command which could contact a network."""
    def __init__(self, *, remote=None, wrong_verified_sha=False, fail_push=False, mutate_add=False):
        self.remote = remote
        self.wrong_verified_sha = wrong_verified_sha
        self.fail_push = fail_push
        self.mutate_add = mutate_add
        self.calls = []
        self.helpers = set()
        self.pushes = 0

    def __call__(self, argv, **kw):
        assert argv[0] == "git"
        assert TOKEN not in repr(argv)
        assert "--force" not in argv and "--delete" not in argv
        assert "-A" not in argv
        assert kw["stdout"] == subprocess.PIPE and kw["stderr"] == subprocess.PIPE
        assert kw["stdin"] == subprocess.DEVNULL and kw["check"] is False
        env = kw["env"]
        assert env["PAPER3_GITHUB_PAT"] == TOKEN
        assert not any(k.startswith(("R2_", "AWS_", "GIT_TRACE")) for k in env)
        assert env["HTTPS_PROXY"] == "https://proxy.invalid:8080"
        assert env["SSL_CERT_FILE"] == "/fixture/ca.pem"
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
        helper = Path(env["GIT_ASKPASS"])
        helper_body = helper.read_text()
        assert TOKEN not in helper_body
        compile(helper_body, str(helper), "exec")
        self.helpers.add(helper)
        self.calls.append(list(argv))
        args = argv[1 + len(gh._GIT_OPTIONS):]
        if args[0] == "ls-remote":
            assert args == ["ls-remote", "--heads", "https://github.com/owner/research3.git",
                            "refs/heads/" + BRANCH]
            value = self.remote
            if self.pushes and self.wrong_verified_sha:
                value = "b" * 40
            output = "" if value is None else value + "\trefs/heads/" + BRANCH + "\n"
            return subprocess.CompletedProcess(argv, 0, output, "")
        if args[0] == "push":
            assert args == ["push", "https://github.com/owner/research3.git",
                            "HEAD:refs/heads/" + BRANCH]
            self.pushes += 1
            if self.fail_push:
                return subprocess.CompletedProcess(argv, 1, "", "fatal: " + TOKEN)
            local = subprocess.run(["git", *gh._GIT_OPTIONS, "rev-parse", "HEAD"],
                                   cwd=kw["cwd"], env=env, capture_output=True, text=True, check=True)
            self.remote = local.stdout.strip()
            return subprocess.CompletedProcess(argv, 0, "", "")
        if args[0] == "add" and self.mutate_add:
            (Path(kw["cwd"]) / "p3work" / "source.py").write_text("print('changed')\n")
        assert args[0] in {"init", "symbolic-ref", "config", "add", "ls-files", "commit",
                           "rev-parse", "status"}, "Unexpected command might contact network"
        return subprocess.run(argv, **kw)


@pytest.fixture
def code(tmp_path, monkeypatch):
    # Replace rather than inspect the real environment. All credentials are fake.
    monkeypatch.setattr(gh.os, "environ", {
        "PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "LANG": "C.UTF-8",
        "HTTPS_PROXY": "https://proxy.invalid:8080", "SSL_CERT_FILE": "/fixture/ca.pem",
        "AWS_ACCESS_KEY_ID": "fixture-r2-id", "AWS_SECRET_ACCESS_KEY": "fixture-r2-secret",
        "R2_ENDPOINT": "fixture-r2-endpoint", "R2_BUCKET": "fixture-r2-bucket",
        "GIT_TRACE": "/fixture/wrong-log", "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "credential.helper", "GIT_CONFIG_VALUE_0": "bad-helper",
        "GH_TOKEN": "fixture-old-token", "GITHUB_TOKEN": "fixture-old-token",
    })
    root = tmp_path / "code"
    (root / "p3work").mkdir(parents=True)
    (root / "p3work" / "source.py").write_text("print('fixture source')\n")
    (root / "README.md").write_text("Fixture code only.\n")
    (root / "release_manifest.json").write_text('{"fixture":true}\n')
    return root


def publish(code, runner=None, api=None):
    return gh.publish_code(REPO, BRANCH, TOKEN, code, runner=runner or LocalGit(), urlopen=api or API())


def local_git(code, *args):
    env = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    return subprocess.run(["git", *gh._GIT_OPTIONS, *args], cwd=code, env=env,
                          capture_output=True, text=True, check=True).stdout


def test_publish_orphan_exact_source_and_owned_resume(code, capsys):
    runner = LocalGit()
    api = API()
    result = publish(code, runner, api)
    assert result["status"] == "GITHUB_CODE_PUSH_AND_REMOTE_SHA_VERIFIED"
    assert result["commit"] == runner.remote and result["code_files"] == 3
    assert runner.pushes == 1 and not result["reused_verified_remote"]
    assert local_git(code, "rev-list", "--count", "HEAD").strip() == "1"
    assert len(local_git(code, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 1
    assert set(local_git(code, "ls-tree", "-r", "--name-only", "HEAD").splitlines()) == {
        "p3work/source.py", "README.md", "release_manifest.json"}
    assert local_git(code, "config", "user.email").strip() == "123+paper3-user@users.noreply.github.com"
    assert "[remote " not in (code / ".git" / "config").read_text()
    assert all(not path.exists() for path in runner.helpers)
    for path in code.rglob("*"):
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes()
    again = publish(code, runner, api)
    assert again["commit"] == result["commit"] and again["reused_verified_remote"]
    assert runner.pushes == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert api.urls == ["https://api.github.com/user", "https://api.github.com/repos/owner/research3"] * 2


@pytest.mark.parametrize("value", ["https://user:pat@github.com/owner/repo", "http://github.com/a/b",
    "https://github.com.evil.invalid/a/b", "https://github.com/a/b?token=x", "owner/repo\n",
    "owner/../repo", "owner/.repo", "owner/repo.", "https://[bad/a/b", " owner/repo"])
def test_reject_repo(value):
    with pytest.raises(gh.PublishError, match="^GITHUB_REPO_INVALID$"):
        gh.validate_repo(value)


@pytest.mark.parametrize("value", ["owner/research3", "https://github.com/owner/research3",
                                    "https://github.com/owner/research3.git"])
def test_normalize_repo(value):
    assert gh.validate_repo(value) == REPO


@pytest.mark.parametrize("value", ["main", "master", "refs/heads/research/test", "research/a..b",
    "research/.a", "research/a.lock/b", "research/a//b", "research/a b", "research/", "research/a\n"])
def test_reject_branch(value):
    with pytest.raises(gh.PublishError, match="^GITHUB_RESEARCH_BRANCH_INVALID$"):
        gh.validate_branch(value)


@pytest.mark.parametrize("path", ["data/source.csv", "cache/data.npz", ".env", "credential.txt",
    "audit/secret.json", "tests/secret.json", "public_code.zip"])
def test_forbid_data_and_private_files_before_http(code, path):
    private = code / path
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("fixture")
    api = API()
    with pytest.raises(gh.PublishError, match="^NON_CODE_OR_PRIVATE_FILE_FORBIDDEN$"):
        publish(code, api=api)
    assert not api.urls and not (code / ".git").exists()


def test_forbid_symlink(code):
    (code / "p3work" / "link.py").symlink_to("source.py")
    with pytest.raises(gh.PublishError, match="^CODE_SYMLINK_FORBIDDEN$"):
        publish(code)


def test_no_existing_branch_overwrite(code):
    runner = LocalGit(remote="a" * 40)
    with pytest.raises(gh.PublishError, match="^GITHUB_BRANCH_ALREADY_EXISTS_CHOOSE_NEW_BRANCH$"):
        publish(code, runner)
    assert runner.pushes == 0 and not (code / ".git").exists()


def test_no_default_branch_overwrite(code):
    api = API({"/repos/owner/research3": {"full_name": REPO, "permissions": {"push": True},
                                       "default_branch": BRANCH}})
    with pytest.raises(gh.PublishError, match="^GITHUB_DEFAULT_BRANCH_FORBIDDEN$"):
        publish(code, api=api)
    assert not (code / ".git").exists()


def test_owned_first_branch_becoming_default_is_read_only_verified(code):
    runner = LocalGit()
    first = publish(code, runner)
    api = API({"/repos/owner/research3": {"full_name": REPO, "permissions": {"push": True},
                                       "default_branch": BRANCH}})
    second = publish(code, runner, api)
    assert second['commit'] == first['commit'] and second['reused_verified_remote']
    assert runner.pushes == 1


def test_owned_default_branch_missing_remote_never_repushes(code):
    runner = LocalGit()
    publish(code, runner)
    runner.remote = None
    api = API({"/repos/owner/research3": {"full_name": REPO, "permissions": {"push": True},
                                       "default_branch": BRANCH}})
    with pytest.raises(gh.PublishError, match='^GITHUB_DEFAULT_BRANCH_FORBIDDEN$'):
        publish(code, runner, api)
    assert runner.pushes == 1


def test_foreign_git_refused(code):
    (code / ".git").mkdir()
    with pytest.raises(gh.PublishError, match="^UNOWNED_GIT_DIRECTORY$"):
        publish(code)


def test_changed_remote_resume_refused(code):
    runner = LocalGit()
    publish(code, runner)
    runner.remote = "c" * 40
    with pytest.raises(gh.PublishError, match="^GITHUB_REMOTE_BRANCH_CHANGED_NO_OVERWRITE$"):
        publish(code, runner)
    assert runner.pushes == 1


def test_changed_source_resume_refused(code):
    runner = LocalGit()
    publish(code, runner)
    (code / "p3work" / "source.py").write_text("print('different')\n")
    with pytest.raises(gh.PublishError, match="^PUBLISH_STATE_OR_SOURCE_CHANGED$"):
        publish(code, runner)
    assert runner.pushes == 1


def test_changed_git_config_resume_refused(code):
    runner = LocalGit()
    publish(code, runner)
    with (code / ".git" / "config").open("a") as stream:
        stream.write("\n[credential]\n\thelper = fixture-hostile\n")
    with pytest.raises(gh.PublishError, match="^PUBLISH_GIT_CONFIG_CHANGED$"):
        publish(code, runner)
    assert runner.pushes == 1


def test_source_mutation_between_inventory_add_fails(code):
    runner = LocalGit(mutate_add=True)
    with pytest.raises(gh.PublishError, match="^EXACT_CODE_ONLY_INDEX_REQUIRED$"):
        publish(code, runner)
    assert runner.pushes == 0


def test_failed_push_is_safe_and_resume_reuses_commit(code, capsys):
    runner = LocalGit(fail_push=True)
    with pytest.raises(gh.PublishError, match="^GITHUB_GIT_COMMAND_FAILED$") as caught:
        publish(code, runner)
    assert TOKEN not in str(caught.value)
    state = json.loads((code / ".git" / gh._STATE).read_text())
    assert state["remote_sha_verified"] is False
    assert all(not helper.exists() for helper in runner.helpers)
    runner.fail_push = False
    result = publish(code, runner)
    assert result["commit"] == state["commit"]
    assert TOKEN not in capsys.readouterr().out


def test_wrong_final_remote_sha_does_not_claim_success(code):
    runner = LocalGit(wrong_verified_sha=True)
    with pytest.raises(gh.PublishError, match="^GITHUB_PUSH_REMOTE_SHA_NOT_VERIFIED$"):
        publish(code, runner)
    assert json.loads((code / ".git" / gh._STATE).read_text())["remote_sha_verified"] is False


def test_api_wrong_final_url_refused(code):
    with pytest.raises(gh.PublishError, match="^GITHUB_API_RESPONSE_INVALID$"):
        publish(code, api=API(redirect=True))


def test_api_redirect_handler_always_refuses():
    with pytest.raises(gh.PublishError, match="^GITHUB_REDIRECT_FORBIDDEN$"):
        gh._NoRedirect().redirect_request(None, None, 302, "fixture", {}, "https://evil.invalid")


@pytest.mark.parametrize(("status", "expected"), [(401, "GITHUB_TOKEN_REJECTED"),
    (403, "GITHUB_PERMISSION_OR_RATE_LIMIT"), (404, "GITHUB_REPOSITORY_NOT_FOUND_OR_NOT_AUTHORIZED")])
def test_fixed_api_errors(code, status, expected):
    def fail(req, timeout):
        raise HTTPError(req.full_url, status, TOKEN, {}, None)
    with pytest.raises(gh.PublishError, match="^" + expected + "$"):
        publish(code, api=fail)


@pytest.mark.parametrize("permissions", [None, [], {"push": False}])
def test_missing_push_permission_fails_closed(code, permissions):
    api = API({"/repos/owner/research3": {"full_name": REPO, "permissions": permissions}})
    with pytest.raises(gh.PublishError, match="^GITHUB_REPOSITORY_WRITE_PERMISSION_REQUIRED$"):
        publish(code, api=api)


def test_askpass_only_trusted_github_prompt(code):
    class CheckAskpass(LocalGit):
        def __call__(self, argv, **kw):
            result = super().__call__(argv, **kw)
            helper = kw["env"]["GIT_ASKPASS"]
            for prompt, expected in [("Username for 'https://github.com': ", "x-access-token\n"),
                ("Password for 'https://x-access-token@github.com': ", TOKEN + "\n")]:
                proc = subprocess.run([sys.executable, helper, prompt], env=kw["env"],
                                      capture_output=True, text=True)
                assert proc.returncode == 0 and proc.stdout == expected and not proc.stderr
            proc = subprocess.run([sys.executable, helper,
                "Password for 'https://github.com.evil.invalid': "], env=kw["env"],
                capture_output=True, text=True)
            assert proc.returncode != 0 and TOKEN not in proc.stdout
            return result
    publish(code, CheckAskpass())


@pytest.mark.parametrize("failure_command", ["init", "config", "add", "ls-files", "commit", "rev-parse"])
def test_pre_state_failure_cleans_only_owned_git_and_can_retry(code, failure_command):
    class FailLocal(LocalGit):
        def __call__(self, argv, **kw):
            args = argv[1 + len(gh._GIT_OPTIONS):]
            if args[0] == failure_command:
                # Simulate a process failing after it may have modified local Git.
                result = super().__call__(argv, **kw)
                return subprocess.CompletedProcess(argv, 1, result.stdout, "fixture failure " + TOKEN)
            return super().__call__(argv, **kw)
    with pytest.raises(gh.PublishError, match="^GITHUB_GIT_COMMAND_FAILED$"):
        publish(code, FailLocal())
    assert not (code / ".git").exists()
    assert (code / "p3work" / "source.py").read_text() == "print('fixture source')\n"
    result = publish(code)
    assert result["status"] == "GITHUB_CODE_PUSH_AND_REMOTE_SHA_VERIFIED"


@pytest.mark.parametrize("step", ["handoff", "init", "add", "commit", "state_before", "state_after"])
@pytest.mark.parametrize("missing_glibc_wrapper", [False, True])
def test_hard_exit_residue_is_owned_and_recovers(code, step, missing_glibc_wrapper, monkeypatch):
    # os._exit runs no Python finally/context cleanup, like SIGKILL. Network
    # commands remain intercepted in the child and only fixture credentials exist.
    if missing_glibc_wrapper:
        monkeypatch.setattr(gh.ctypes, "CDLL", _without_renameat2(gh.ctypes.CDLL))
    build = Path(__file__).resolve().parents[1]
    script = r'''
import os,sys,json,importlib
sys.path[:0]=json.loads(sys.argv[4])
sys.path.insert(0,sys.argv[1])
sys.path.insert(0,sys.argv[1]+"/tests")
fixture=importlib.import_module(sys.argv[5])
gh,LocalGit,publish,Path=fixture.gh,fixture.LocalGit,fixture.publish,fixture.Path
if sys.argv[6] == "missing":
    gh.ctypes.CDLL=fixture._without_renameat2(gh.ctypes.CDLL)
step=sys.argv[3]
original_install=gh._install_initializing_git
original_write=gh._write_state
if step == "handoff":
    def install(*a,**k):
        original_install(*a,**k)
        os._exit(73)
    gh._install_initializing_git=install
if step.startswith("state_"):
    def write(path,value):
        if path.name == gh._STATE:
            if step == "state_before":
                (path.parent/"paper3-state-interrupted").write_text('{"partial":')
            else:
                original_write(path,value)
            os._exit(73)
        return original_write(path,value)
    gh._write_state=write
class KillGit(LocalGit):
    def __call__(self,argv,**kw):
        result=super().__call__(argv,**kw)
        args=argv[1+len(gh._GIT_OPTIONS):]
        if args[0] == step:
            os._exit(73)
        return result
publish(Path(sys.argv[2]),KillGit())
raise RuntimeError("Expected a hard exit")
'''
    killed = subprocess.run([sys.executable, "-B", "-c", script, str(build), str(code), step,
                             json.dumps(sys.path), Path(__file__).stem,
                             "missing" if missing_glibc_wrapper else "native"],
                            env=dict(gh.os.environ), capture_output=True, text=True)
    assert killed.returncode == 73, killed.stderr
    assert (code / ".git" / gh._INITIALIZING).is_file()
    old_commit = None
    if step == "state_after":
        old_commit = json.loads((code / ".git" / gh._STATE).read_text())["commit"]
    result = publish(code)
    assert result["status"] == "GITHUB_CODE_PUSH_AND_REMOTE_SHA_VERIFIED"
    assert not (code / ".git" / gh._INITIALIZING).exists()
    if old_commit:
        assert result["commit"] == old_commit


@pytest.mark.parametrize("field", ["root", "root_dev", "root_ino", "repo", "branch",
                                    "inventory_sha256", "schema_version", "code_files"])
def test_initialization_marker_mismatch_never_deletes_unknown_git(code, field):
    rows, sha = gh._inventory(code)
    marker = gh._initialization_marker(code, REPO, BRANCH, sha, len(rows))
    marker[field] = "fixture-wrong-value"
    gitdir = code / ".git"
    gitdir.mkdir()
    (gitdir / "keep-unknown-file").write_text("preserve this unknown Git directory")
    gh._write_state(gitdir / gh._INITIALIZING, marker)
    with pytest.raises(gh.PublishError, match="^PUBLISH_INITIALIZATION_MARKER_INVALID$"):
        publish(code)
    assert (gitdir / "keep-unknown-file").read_text() == "preserve this unknown Git directory"


@pytest.mark.parametrize("missing_glibc_wrapper", [False, True])
def test_atomic_install_cannot_replace_unknown_empty_git(code, missing_glibc_wrapper, monkeypatch):
    if missing_glibc_wrapper:
        monkeypatch.setattr(gh.ctypes, "CDLL", _without_renameat2(gh.ctypes.CDLL))
    stage = code.parent / "stage-git"
    stage.mkdir()
    (stage / "marker").write_text("owned staged metadata")
    unknown = code / ".git"
    unknown.mkdir()
    before = unknown.stat()
    with pytest.raises(gh.PublishError, match="^UNOWNED_GIT_DIRECTORY$"):
        gh._rename_no_replace(stage, unknown)
    assert unknown.is_dir() and list(unknown.iterdir()) == []
    assert unknown.stat().st_ino == before.st_ino
    assert (stage / "marker").read_text() == "owned staged metadata"


def test_missing_glibc_wrapper_uses_actual_no_replace_syscall(tmp_path, monkeypatch):
    monkeypatch.setattr(gh.ctypes, "CDLL", _without_renameat2(gh.ctypes.CDLL))
    monkeypatch.setattr(gh.os, "rename", lambda *a, **k: pytest.fail("replacing rename forbidden"))
    source, target = tmp_path / "源目录", tmp_path / "已安装.git"
    source.mkdir(); (source / "ownership-marker").write_bytes(b"owned metadata")
    identity = source.stat()
    gh._rename_no_replace(source, target)
    assert not source.exists() and target.stat().st_ino == identity.st_ino
    assert (target / "ownership-marker").read_bytes() == b"owned metadata"


@pytest.mark.parametrize("number,code", [
    (gh.errno.ENOSYS, "ATOMIC_GIT_INITIALIZATION_UNSUPPORTED"),
    (gh.errno.EOPNOTSUPP, "ATOMIC_GIT_INITIALIZATION_UNSUPPORTED"),
    (gh.errno.EINVAL, "ATOMIC_GIT_INITIALIZATION_UNSUPPORTED"),
    (gh.errno.EEXIST, "UNOWNED_GIT_DIRECTORY"),
    (gh.errno.ENOTEMPTY, "UNOWNED_GIT_DIRECTORY"),
    (gh.errno.EACCES, "GITHUB_LOCAL_INITIALIZATION_FAILED"),
])
def test_raw_rename_syscall_errors_remain_closed(tmp_path, monkeypatch, number, code):
    calls = []
    def syscall(*args):
        calls.append(args)
        gh.ctypes.set_errno(number)
        return -1
    class OldLibc:
        pass
    library = OldLibc(); library.syscall = syscall
    monkeypatch.setattr(gh.ctypes, "CDLL", lambda *a, **k: library)
    monkeypatch.setattr(gh.os, "rename", lambda *a, **k: pytest.fail("replacing rename forbidden"))
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(); (source / "marker").write_bytes(b"keep")
    with pytest.raises(gh.PublishError, match="^" + code + "$"):
        gh._rename_no_replace(source, target)
    assert len(calls) == 1 and calls[0][-1].value == 1
    assert source.is_dir() and (source / "marker").read_bytes() == b"keep"
    assert not target.exists()


@pytest.mark.parametrize("platform_name,machine,pointer_bytes,long_bytes", [
    ("darwin", "x86_64", 8, 8), ("linux", "riscv64", 8, 8),
    ("linux", "x86_64", 4, 4), ("linux", "x86_64", 8, 4),
])
def test_missing_wrapper_unknown_kernel_abi_never_attempts_syscall(
        tmp_path, monkeypatch, platform_name, machine, pointer_bytes, long_bytes):
    class OldLibc:
        def syscall(self, *args):
            pytest.fail("unknown syscall ABI must remain closed")
    monkeypatch.setattr(gh.ctypes, "CDLL", lambda *a, **k: OldLibc())
    monkeypatch.setattr(gh.sys, "platform", platform_name)
    monkeypatch.setattr(gh.platform, "machine", lambda: machine)
    original_sizeof = gh.ctypes.sizeof
    monkeypatch.setattr(gh.ctypes, "sizeof", lambda value:
        pointer_bytes if value is gh.ctypes.c_void_p else
        long_bytes if value is gh.ctypes.c_long else original_sizeof(value))
    with pytest.raises(gh.PublishError, match="^ATOMIC_GIT_INITIALIZATION_UNSUPPORTED$"):
        gh._rename_no_replace(tmp_path / "source", tmp_path / "target")


@pytest.mark.parametrize("machine,number", [("x86_64", 316), ("amd64", 316),
                                           ("aarch64", 276), ("arm64", 276)])
def test_raw_rename_syscall_abi_selection_is_explicit(monkeypatch, machine, number):
    monkeypatch.setattr(gh.sys, "platform", "linux")
    monkeypatch.setattr(gh.platform, "machine", lambda: machine)
    assert gh._renameat2_syscall_number() == number


def test_missing_both_rename_and_syscall_symbols_preserves_unsupported_error(tmp_path, monkeypatch):
    monkeypatch.setattr(gh.ctypes, "CDLL", lambda *a, **k: object())
    with pytest.raises(gh.PublishError, match="^ATOMIC_GIT_INITIALIZATION_UNSUPPORTED$"):
        gh._rename_no_replace(tmp_path / "source", tmp_path / "target")
