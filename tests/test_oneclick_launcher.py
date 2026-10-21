"""Small orchestration fixtures: no real network, credentials, fits or GPUs."""
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper3_one_click as launch


FAKE_CREDENTIALS = {
    "endpoint": "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
    "bucket": "fixture-paper3", "access_key": "FAKE_ACCESS_VALUE",
    "secret_key": "FAKE_SECRET_VALUE", "github_token": "FAKE_GH_VALUE",
}


def state_and_paths(tmp_path, *, runtime=True, symlink=False):
    work = tmp_path / "owned-work"
    work.mkdir()
    state = launch.current_state(work)
    launch.save_progress(work, state)
    project = work / "release" / "fixed-project"
    project.mkdir(parents=True)
    if runtime:
        bin_dir = project / ".paper3-deployment" / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        python = bin_dir / "python"
        if symlink:
            target = project / ".paper3-deployment" / "runtime" / "bin" / "python3.12"
            target.parent.mkdir(parents=True)
            target.write_text("not an actual interpreter\n")
            python.symlink_to(target)
        else:
            python.write_text("not an actual interpreter\n")
    return work, project, state


class FakeChildren:
    def __init__(self, *returncodes):
        self.returncodes = iter(returncodes)
        self.calls = []

    def __call__(self, command, cwd, env, log_path):
        self.calls.append({"command": command, "cwd": cwd, "env": env,
                           "log_path": log_path})
        log_path.write_text("fixture child producer only\n")
        return {"actual_exit_code": next(self.returncodes),
                "interrupted_by_user": False, "log_file": log_path.name}


def run_entry(tmp_path, children, *, runtime=True, symlink=False, **kwargs):
    work, project, state = state_and_paths(tmp_path, runtime=runtime, symlink=symlink)
    code = launch.training_and_backup(project, work, FAKE_CREDENTIALS, state,
                                     child_runner=children, **kwargs)
    return code, work, project, state


def test_non_tty_secret_input_fails_before_reading(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("must never read this"))
    monkeypatch.setattr(launch.getpass, "getpass",
                        lambda *args: pytest.fail("getpass called without TTY"))
    with pytest.raises(launch.LaunchError,
                       match="^INTERACTIVE_TERMINAL_REQUIRED_NO_ECHO_FALLBACK$"):
        launch.secret_input("fixture")


def test_hidden_input_never_prints_secret(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", type("TTY", (), {"isatty": lambda self: True})())
    calls = []
    def fake_getpass(prompt):
        calls.append(prompt)
        return "FAKE_INPUT_VALUE"
    monkeypatch.setattr(launch.getpass, "getpass", fake_getpass)
    assert launch.secret_input("fixture") == "FAKE_INPUT_VALUE"
    assert calls == ["fixture："]
    assert "FAKE_INPUT_VALUE" not in capsys.readouterr().out


@pytest.mark.parametrize("value", ["", "x\nvalue", "x\x00value", "x\x7fvalue"])
def test_hidden_input_rejects_empty_or_control_values(monkeypatch, value):
    monkeypatch.setattr(sys, "stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr(launch.getpass, "getpass", lambda *args: value)
    with pytest.raises(launch.LaunchError, match="^EMPTY_OR_INVALID_CREDENTIAL$"):
        launch.secret_input("fixture")


@pytest.mark.parametrize("value,expected", [
    ("A" * 32, "https://" + "a" * 32 + ".r2.cloudflarestorage.com"),
    (FAKE_CREDENTIALS["endpoint"] + "/", FAKE_CREDENTIALS["endpoint"]),
    (FAKE_CREDENTIALS["endpoint"] + ":443", FAKE_CREDENTIALS["endpoint"]),
])
def test_r2_endpoint_normalization(value, expected):
    assert launch.normalize_endpoint(value) == expected


@pytest.mark.parametrize("value", [
    "https://r2.cloudflarestorage.com.evil.example",
    "http://" + "a" * 32 + ".r2.cloudflarestorage.com",
    "https://user:password@" + "a" * 32 + ".r2.cloudflarestorage.com",
    FAKE_CREDENTIALS["endpoint"] + "/other",
    FAKE_CREDENTIALS["endpoint"] + "?key=secret",
    FAKE_CREDENTIALS["endpoint"] + "#secret",
    FAKE_CREDENTIALS["endpoint"] + ":444",
    "0123456789abcdef",
])
def test_r2_endpoint_rejects_wrong_origin_or_secret_locations(value):
    with pytest.raises(launch.LaunchError,
                       match="^R2_ACCOUNT_ID_OR_HTTPS_S3_ENDPOINT_REQUIRED$"):
        launch.normalize_endpoint(value)


def test_training_env_has_no_github_credentials_and_preserves_proxy_trust(monkeypatch):
    inherited = {
        "GITHUB_TOKEN": "old-secret", "GH_TOKEN": "old-secret", "GITHUB_PAT": "old-secret",
        "GH_ENTERPRISE_TOKEN": "old-secret", "PAPER3_GITHUB_PAT": "old-secret",
        "GIT_CONFIG_COUNT": "1", "GIT_TRACE": "1", "GIT_ASKPASS": "/tmp/foreign-helper",
        "HTTPS_PROXY": "http://proxy.example:8080", "SSL_CERT_FILE": "/etc/custom-ca.pem",
        "REQUESTS_CA_BUNDLE": "/etc/custom-ca.pem", "PATH": "/usr/bin",
        "R2_BUCKET": "old-bucket", "OMP_NUM_THREADS": "128",
        "AWS_SESSION_TOKEN": "old-session-token", "AWS_SECURITY_TOKEN": "old-security-token",
    }
    monkeypatch.setattr(launch.os, "environ", inherited)
    result = launch.training_environment(FAKE_CREDENTIALS)
    assert result["HTTPS_PROXY"] == inherited["HTTPS_PROXY"]
    assert result["SSL_CERT_FILE"] == inherited["SSL_CERT_FILE"]
    assert result["REQUESTS_CA_BUNDLE"] == inherited["REQUESTS_CA_BUNDLE"]
    assert result["PATH"] == "/usr/bin"
    assert not any(k.startswith("GIT_") or k in launch.GH_NAMES for k in result)
    assert "AWS_SESSION_TOKEN" not in result and "AWS_SECURITY_TOKEN" not in result
    assert result["R2_BUCKET"] == FAKE_CREDENTIALS["bucket"]
    assert result["AWS_SECRET_ACCESS_KEY"] == FAKE_CREDENTIALS["secret_key"]
    assert result["OMP_NUM_THREADS"] == "1"
    assert result["PYTHONDONTWRITEBYTECODE"] == "1"
    assert inherited["R2_BUCKET"] == "old-bucket"


@pytest.mark.parametrize("relative", ["../elsewhere", "/absolute", "x/../y", "x//y", "x\\y", "./x"])
def test_safe_payload_paths_reject_traversal(tmp_path, relative):
    with pytest.raises(launch.LaunchError, match="^UNSAFE_LOCAL_PATH$"):
        launch.safe_path(tmp_path, relative)


def test_safe_payload_path_does_not_follow_existing_symlink(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(launch.LaunchError, match="^SYMLINK_FORBIDDEN$"):
        launch.safe_path(tmp_path, "link/payload")


def test_same_work_dir_rejects_second_launcher_and_releases_own_lock(tmp_path):
    work = tmp_path / "lock-work"
    with launch.owner_lock(work):
        with pytest.raises(launch.LaunchError, match="^ONE_LAUNCHER_ALREADY_ACTIVE$"):
            with launch.owner_lock(work):
                pytest.fail("second owner entered")
    with launch.owner_lock(work):
        assert (work / "launcher.owner.lock").is_file()


@pytest.mark.parametrize("training_exit", [1, 2, 17, 130, -2])
def test_failed_or_paused_training_exit_is_preserved_after_successful_backup(tmp_path, training_exit):
    children = FakeChildren(training_exit, 0)
    code, work, project, state = run_entry(tmp_path, children)
    assert code == training_exit
    assert len(children.calls) == 2
    training, backup = [x["command"] for x in children.calls]
    assert "--launch-training" in training and "--user-started-gpu" in training
    assert launch.RESULT_HELPER in backup and "--launch-training" not in backup
    assert state["training"]["actual_exit_code"] == training_exit
    assert state["result_backup"]["actual_exit_code"] == 0
    saved = json.loads((work / "RUN_PROGRESS.json").read_text())
    assert saved["training"]["actual_exit_code"] == training_exit
    assert [r["kind"] for r in saved["attempts"]] == ["training_entry", "result_backup_only"]


def test_failed_backup_preserves_successful_training_and_returns_backup_failure(tmp_path):
    children = FakeChildren(0, 23)
    code, work, project, state = run_entry(tmp_path, children)
    assert code == 23
    assert state["training"]["actual_exit_code"] == 0
    assert state["result_backup"]["actual_exit_code"] == 23


def test_successful_scientific_entry_retry_never_runs_training_again(tmp_path):
    work, project, state = state_and_paths(tmp_path)
    state["training"] = {"actual_exit_code": 0, "log_file": "previous-training.log"}
    state["result_backup"] = {"actual_exit_code": 23}
    children = FakeChildren(0)
    result = launch.training_and_backup(project, work, FAKE_CREDENTIALS, state,
                                        child_runner=children)
    assert result == 0
    assert len(children.calls) == 1
    assert launch.RESULT_HELPER in children.calls[0]["command"]
    assert state["training"]["log_file"] == "previous-training.log"
    assert [r["kind"] for r in state["attempts"]] == ["result_backup_only"]


def test_prepare_only_has_no_fit_flags_and_no_future_backup_claim(tmp_path):
    children = FakeChildren(0)
    code, work, project, state = run_entry(tmp_path, children, prepare_only=True)
    assert code == 0 and len(children.calls) == 1
    command = children.calls[0]["command"]
    assert "tools/bootstrap_data_plan.py" in command and "--execute" in command
    assert "--launch-training" not in command and "--user-started-gpu" not in command
    assert state["preparation"]["actual_exit_code"] == 0
    assert "training" not in state and "result_backup" not in state
    assert not state["attempts"]


def test_backup_only_never_runs_bootstrap_or_fit_even_without_training_state(tmp_path):
    children = FakeChildren(0)
    code, work, project, state = run_entry(tmp_path, children, backup_only=True)
    assert code == 0 and len(children.calls) == 1
    command = children.calls[0]["command"]
    assert launch.RESULT_HELPER in command and "--execute" in command
    assert "tools/bootstrap_data_plan.py" not in command
    assert "--launch-training" not in command
    assert "training" not in state


def test_no_runtime_preserves_training_failure_and_claims_no_upload(tmp_path):
    children = FakeChildren(9)
    code, work, project, state = run_entry(tmp_path, children, runtime=False)
    assert code == 9 and len(children.calls) == 1
    assert state["result_backup"] == {
        "status": "NOT_RUN_NO_PREPARED_ENVIRONMENT", "uploaded_claim": False}


def test_expected_venv_python_symlink_confined_to_owned_runtime_is_allowed(tmp_path):
    children = FakeChildren(0)
    code, work, project, state = run_entry(tmp_path, children, symlink=True, backup_only=True)
    assert code == 0 and len(children.calls) == 1
    interpreter = Path(children.calls[0]["command"][0])
    assert interpreter.is_symlink()
    assert interpreter.resolve().is_relative_to(project / ".paper3-deployment")


def test_venv_python_symlink_outside_owned_runtime_never_runs_backup(tmp_path):
    work, project, state = state_and_paths(tmp_path, runtime=False)
    bin_dir = project / ".paper3-deployment" / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    foreign = tmp_path / "foreign-interpreter"
    foreign.write_text("foreign\n")
    (bin_dir / "python").symlink_to(foreign)
    children = FakeChildren(0)
    with pytest.raises(launch.LaunchError,
                       match="^RESULT_BACKUP_PYTHON_OUTSIDE_OWNED_RUNTIME$"):
        launch.training_and_backup(project, work, FAKE_CREDENTIALS, state,
                                    backup_only=True, child_runner=children)
    assert not children.calls


def test_training_exception_never_starts_concurrent_backup_or_claims_fit(tmp_path):
    work, project, state = state_and_paths(tmp_path)
    calls = []
    def interrupted_child(command, cwd, env, log_path):
        calls.append(command)
        raise launch.LaunchError("OWNED_TRAINING_CHILD_STILL_STOPPING_NO_BACKUP_OR_RESTART")
    with pytest.raises(launch.LaunchError,
                       match="^OWNED_TRAINING_CHILD_STILL_STOPPING_NO_BACKUP_OR_RESTART$"):
        launch.training_and_backup(project, work, FAKE_CREDENTIALS, state,
                                    child_runner=interrupted_child)
    assert len(calls) == 1
    saved = json.loads((work / "RUN_PROGRESS.json").read_text())
    assert "entry_issued_at_utc" in saved
    assert "training" not in saved and "result_backup" not in saved


def test_progress_and_logs_do_not_persist_credential_values(tmp_path, capsys):
    children = FakeChildren(0, 0)
    code, work, project, state = run_entry(tmp_path, children)
    assert code == 0
    output = capsys.readouterr().out
    for path in work.rglob("*"):
        if path.is_file():
            text = path.read_text()
            for secret in FAKE_CREDENTIALS.values():
                assert secret not in text
    for secret in FAKE_CREDENTIALS.values():
        assert secret not in output
    assert FAKE_CREDENTIALS["github_token"] not in children.calls[0]["env"].values()


def fixture_orchestrator(tmp_path, monkeypatch, failure):
    """Exercise the real stage order while every external action is a fixture."""
    calls = []
    names = ("ready", "main_inputs", "control_inputs", "main_archive", "control_archive")
    manifest = {
        "sdk_wheels": [], "objects": {name: {"path": name} for name in names},
        "project_topdir": "fixed-project",
    }
    def stop_at(stage):
        calls.append(stage)
        if stage == failure:
            raise launch.LaunchError("FIXTURE_STAGE_FAILED")
    def fetch(client, bucket, row, downloads):
        stop_at("GET_" + row["path"])
        return {"status": "FIXTURE_GET"}
    def publish(repo, branch, token, code):
        assert token == FAKE_CREDENTIALS["github_token"]
        stop_at("GITHUB_PUSH")
        return {"repo": repo, "branch": branch, "commit": "a" * 40}
    def extract(archive, inputs, root, topdir):
        stop_at("EXTRACT")
        (root / topdir).mkdir(exist_ok=True)
        return {"status": "FIXTURE_EXTRACT"}
    transport = SimpleNamespace(
        ensure_sdk=lambda *args: stop_at("SDK"),
        create_client=lambda *args: object(), fetch_object=fetch, extract_exact=extract,
        probe_write=lambda *args: stop_at("R2_WRITE_READ_PROBE"))
    github = SimpleNamespace(validate_repo=lambda x: x, validate_branch=lambda x: x,
                             publish_code=publish)
    monkeypatch.setattr(launch, "load_bundle", lambda bundle: manifest)
    monkeypatch.setattr(launch.importlib, "import_module",
                        lambda name: {"paper3_transport": transport, "paper3_github": github}[name])
    monkeypatch.setattr(launch, "collect_credentials", lambda required: dict(FAKE_CREDENTIALS))
    monkeypatch.setattr(launch, "prepare_code_directory", lambda bundle, work, manifest: work / "publish-code")
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/git")
    monkeypatch.setattr(launch, "copy_exact", lambda *args: None)
    monkeypatch.setattr(launch, "verify_deployed_release", lambda *args: stop_at("FIXED_RELEASE_VERIFY"))
    monkeypatch.setattr(launch, "training_and_backup", lambda *args, **kwargs: stop_at("FIT_ENTRY"))
    monkeypatch.setattr(sys, "stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr(sys, "path", sys.path[:])
    args = SimpleNamespace(check_bundle=False, run_on_this_gpu=True, prepare_only=False,
                           backup_only=False, work_dir=tmp_path / "owned-launcher",
                           repo="fixture/fixture", branch=None)
    return calls, args


@pytest.mark.parametrize("stage", [
    "GET_ready", "R2_WRITE_READ_PROBE", "GITHUB_PUSH", "EXTRACT", "FIXED_RELEASE_VERIFY"])
def test_real_stage_order_blocks_fit_after_failed_prerequisite(tmp_path, monkeypatch, stage):
    calls, args = fixture_orchestrator(tmp_path, monkeypatch, stage)
    with pytest.raises(launch.LaunchError, match="^FIXTURE_STAGE_FAILED$"):
        launch.run(args)
    assert calls[-1] == stage
    assert "FIT_ENTRY" not in calls
    state = json.loads((args.work_dir / "RUN_PROGRESS.json").read_text())
    assert "training" not in state and not state.get("fixed_release_verified")
    if stage in {"GET_ready", "R2_WRITE_READ_PROBE"}:
        assert "GITHUB_PUSH" not in calls
    if stage == "GITHUB_PUSH":
        assert "GET_main_archive" not in calls
        assert "github" not in state


def test_check_bundle_mode_never_prompts_or_loads_network_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "load_bundle", lambda bundle: {})
    monkeypatch.setattr(launch, "collect_credentials", lambda *args: pytest.fail("credential prompt"))
    monkeypatch.setattr(launch.importlib, "import_module", lambda *args: pytest.fail("network module"))
    args = SimpleNamespace(check_bundle=True)
    assert launch.run(args) == 0


def test_failed_publish_retry_uses_persisted_same_branch(tmp_path, monkeypatch):
    calls, args = fixture_orchestrator(tmp_path, monkeypatch, "GITHUB_PUSH")
    branches = []
    github = launch.importlib.import_module("paper3_github")
    github.validate_branch = lambda value: branches.append(value) or value
    with pytest.raises(launch.LaunchError, match="^FIXTURE_STAGE_FAILED$"):
        launch.run(args)
    first = json.loads((args.work_dir / "RUN_PROGRESS.json").read_text())
    assert first["github_target"]["repo"] == args.repo
    assert first["github_target"]["branch"] == branches[0]
    monkeypatch.setattr(launch.uuid, "uuid4", lambda: pytest.fail("retry generated another branch"))
    with pytest.raises(launch.LaunchError, match="^FIXTURE_STAGE_FAILED$"):
        launch.run(args)
    assert branches == [first["github_target"]["branch"]] * 2
    assert calls.count("GITHUB_PUSH") == 2
    second = json.loads((args.work_dir / "RUN_PROGRESS.json").read_text())
    assert second["github_target"] == first["github_target"]
    assert "github" not in second and "training" not in second


@pytest.mark.parametrize("field,value,code", [
    ("repo", "fixture/different", "RESUME_REPOSITORY_CHANGED_USE_NEW_WORK_DIRECTORY"),
    ("branch", "research/different", "RESUME_BRANCH_CHANGED_USE_NEW_WORK_DIRECTORY"),
])
def test_existing_work_target_change_is_rejected_before_new_credentials(
        tmp_path, monkeypatch, field, value, code):
    calls, args = fixture_orchestrator(tmp_path, monkeypatch, "GITHUB_PUSH")
    with pytest.raises(launch.LaunchError, match="^FIXTURE_STAGE_FAILED$"):
        launch.run(args)
    setattr(args, field, value)
    monkeypatch.setattr(launch, "collect_credentials", lambda *args: pytest.fail("new credential prompt"))
    with pytest.raises(launch.LaunchError, match="^" + code + "$"):
        launch.run(args)
    assert calls.count("GITHUB_PUSH") == 1
