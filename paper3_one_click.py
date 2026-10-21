#!/usr/bin/env python3
"""Paper3: hidden credentials, code-only GitHub publication, frozen GPU entry.

Run only on the Linux GPU machine the user has manually started. No RunPod,
SSH, browser, instance management, new credentials, or external-query action.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import uuid
import warnings
import zipfile

RELEASE = "603d299351fa6605aa0bc258bbbe2c421b2048e5becbeb33521e119a62f20fd7"
BASE = "audit/data_plan_20261020/final_deployment_v2"
INDEX = BASE + "/bootstrap_index.json"
INDEX_SHA = "ef220ccb0affd9ac7ce256284de6a1c6dbf2634d945eb4f538355cb86a67b3e7"
INDEX_BYTES = 11184232
INDEX_SEMANTIC = "2533371924f90b97ec5315cd2298bc7b1be575f66adc792934b3cdb44e748db1"
RESULT_HELPER = BASE + "/BACKUP_TRAINING_RESULTS.py.txt"
RESULT_HELPER_SHA = "2d448546e8510749a44e16d2cc513ca4410629c6ffd7fe438a3715bb5bd93a69"
R2_NAMES = ("R2_ENDPOINT", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
GH_NAMES = {"PAPER3_GITHUB_PAT", "GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT", "GH_ENTERPRISE_TOKEN"}


class LaunchError(ValueError):
    """Fixed non-sensitive code only; never stringify an SDK/Git exception."""


def need(condition, code):
    if not condition:
        raise LaunchError(code)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    h = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def safe_path(root, relative):
    need(isinstance(relative, str) and relative and "\\" not in relative,
         "UNSAFE_LOCAL_PATH")
    path = PurePosixPath(relative)
    need(not path.is_absolute() and path.as_posix() == relative
         and all(p not in {"", ".", ".."} for p in relative.split("/")), "UNSAFE_LOCAL_PATH")
    target = Path(root)
    for part in path.parts:
        target /= part
        need(not target.is_symlink(), "SYMLINK_FORBIDDEN")
    return target


def canonical_root(path):
    path = Path(os.path.abspath(path))
    need(not any(p.is_symlink() for p in (path, *path.parents)), "SYMLINK_FORBIDDEN")
    return path


def verify_ref(root, row):
    need(isinstance(row, dict) and set(row) == {"path", "sha256", "bytes"},
         "EXACT_FILE_REFERENCE_REQUIRED")
    need(type(row["bytes"]) is int and row["bytes"] >= 0
         and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is not None,
         "INVALID_FILE_REFERENCE")
    path = safe_path(root, row["path"])
    need(path.is_file() and digest_file(path) == (row["sha256"], row["bytes"]),
         "BUNDLE_FILE_SHA_MISMATCH")
    return path


def write_json(path, value):
    """Atomic progress metadata. Callers never pass credentials to this API."""
    path = canonical_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".paper3-state-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_bundle(bundle):
    manifest_path = safe_path(bundle, "release_manifest.json")
    need(manifest_path.is_file(), "RELEASE_MANIFEST_MISSING")
    value = json.loads(manifest_path.read_bytes())
    need(value.get("schema_version") == "paper3-one-click-release-v1"
         and value.get("closure_sha256") == RELEASE, "WRONG_FIXED_RELEASE")
    need(value.get("index") == {"path": INDEX, "sha256": INDEX_SHA, "bytes": INDEX_BYTES}
         and value.get("index_semantic_sha256") == INDEX_SEMANTIC, "WRONG_FROZEN_INDEX")
    need(value.get("project_topdir") == "paper3_data_plan_" + RELEASE[:16], "WRONG_PROJECT_DIRECTORY")
    for row in value.get("launcher_files", []):
        verify_ref(bundle, row)
    names = {r["path"] for r in value.get("launcher_files", [])}
    need(names == {"paper3_one_click.py", "paper3_github.py", "paper3_transport.py", "START.sh"},
         "EXACT_LAUNCHER_MODULES_REQUIRED")
    verify_ref(bundle, value["public_code_snapshot"])
    need(len(value.get("sdk_wheels", [])) == 8, "EIGHT_PINNED_SDK_WHEELS_REQUIRED")
    for row in value["sdk_wheels"]:
        need(row["path"].startswith("sdk/") and row["path"].endswith(".whl"), "SDK_SCOPE_INVALID")
        verify_ref(bundle, row)
    need(set(value.get("objects", {})) == {"main_archive", "main_inputs", "control_archive", "control_inputs", "ready"},
         "EXACT_FIVE_APPROVED_R2_OBJECTS_REQUIRED")
    expected = {
        "main_archive": ("77da487a0f14f779943cc62f00cfa4c8fd6119540b4285b822ec4bbef285a984", 166234777),
        "main_inputs": ("d2511c0ba9ad7e94ad9b90b94aaaeea9fa14b6c87d9b59ba5046d05b741e45c8", 4704814),
        "control_archive": ("9e65447ea1064d418bb518d22381e7cb43f22941e5ae7f8b8968fca2dcdb5b13", 19126869),
        "control_inputs": ("cd74fa9824357a0515d74e40435a7a07fdd0d0bdad47601e6c5c1a87462f5e3a", 3141376),
        "ready": ("cd95a5574fc05fc80aeb52645eafd0453368bd8da6af387bde69c0cd9da3223a", 7282),
    }
    for name, pair in expected.items():
        row = value["objects"][name]
        need(set(row) == {"path", "key", "sha256", "bytes"}
             and (row["sha256"], row["bytes"]) == pair
             and row["key"].startswith("paper3/") and row["key"].endswith("/" + row["path"]),
             "R2_OBJECT_IDENTITY_OR_SCOPE_INVALID")
        safe_path(bundle, row["path"])
    return value


def secret_input(label):
    need(sys.stdin.isatty(), "INTERACTIVE_TERMINAL_REQUIRED_NO_ECHO_FALLBACK")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            value = getpass.getpass(label + "：").strip()
        except (EOFError, getpass.GetPassWarning):
            raise LaunchError("INTERACTIVE_TERMINAL_REQUIRED_NO_ECHO_FALLBACK") from None
    need(value and not any(ord(c) < 32 or ord(c) == 127 for c in value), "EMPTY_OR_INVALID_CREDENTIAL")
    return value


def normalize_endpoint(text):
    from urllib.parse import urlsplit
    if re.fullmatch(r"[a-fA-F0-9]{32}", text):
        return "https://" + text.lower() + ".r2.cloudflarestorage.com"
    value = urlsplit(text)
    host = value.hostname or ""
    need(value.scheme == "https" and host.endswith(".r2.cloudflarestorage.com")
         and value.username is None and value.password is None and value.port in {None, 443}
         and value.path in {"", "/"} and not value.query and not value.fragment,
         "R2_ACCOUNT_ID_OR_HTTPS_S3_ENDPOINT_REQUIRED")
    return "https://" + host


def collect_credentials(github_required):
    result = {}
    if github_required:
        result["github_token"] = secret_input("GitHub Token（仓库 Contents 读写权限，输入不回显）")
    result["endpoint"] = normalize_endpoint(secret_input("R2 Account ID 或 S3 Endpoint（输入不回显）"))
    result["bucket"] = secret_input("R2 Bucket 名称（输入不回显）")
    need(re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", result["bucket"]) is not None,
         "INVALID_R2_BUCKET_NAME")
    result["access_key"] = secret_input("R2 Access Key ID（输入不回显）")
    result["secret_key"] = secret_input("R2 Secret Access Key（输入不回显）")
    return result


def training_environment(credentials):
    env = dict(os.environ)
    for name in list(env):
        if name in GH_NAMES or name.startswith("GIT_") or name in {
                "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN", "PYTHONHOME", "PYTHONPATH"}:
            env.pop(name, None)
    env.update(zip(R2_NAMES, (credentials["endpoint"], credentials["bucket"],
                            credentials["access_key"], credentials["secret_key"])))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    return env


@contextmanager
def owner_lock(work):
    work = canonical_root(work)
    work.mkdir(parents=True, exist_ok=True)
    path = safe_path(work, "launcher.owner.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LaunchError("ONE_LAUNCHER_ALREADY_ACTIVE") from None
        yield


def current_state(work):
    path = safe_path(work, "RUN_PROGRESS.json")
    if not path.exists():
        need({p.name for p in work.iterdir()} <= {"launcher.owner.lock"},
             "NEW_WORK_DIRECTORY_MUST_BE_EMPTY")
        return {"schema_version": "paper3-oneclick-progress-v1", "closure_sha256": RELEASE,
                "created_at_utc": utc(), "attempts": []}
    value = json.loads(path.read_bytes())
    need(value.get("schema_version") == "paper3-oneclick-progress-v1"
         and value.get("closure_sha256") == RELEASE, "WORK_DIRECTORY_BELONGS_TO_DIFFERENT_RELEASE")
    return value


def save_progress(work, state):
    state["updated_at_utc"] = utc()
    write_json(work / "RUN_PROGRESS.json", state)


def copy_exact(source, destination, expected):
    destination = canonical_root(destination)
    need(digest_file(source) == (expected["sha256"], expected["bytes"]), "SOURCE_SHA_CHANGED")
    if destination.exists():
        need(destination.is_file() and digest_file(destination) == (expected["sha256"], expected["bytes"]),
             "EXISTING_FILE_DIFFERS_NO_OVERWRITE")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".paper3-copy-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as out, Path(source).open("rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        need(digest_file(name) == (expected["sha256"], expected["bytes"]), "COPY_SHA_MISMATCH")
        os.link(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def prepare_code_directory(bundle, work, manifest):
    code = safe_path(work, "publish-code")
    code.mkdir(exist_ok=True)
    refs = manifest["code_files"]
    need(len(refs) == len({r["path"] for r in refs}), "DUPLICATE_PUBLIC_CODE_PATH")
    expected = {r["path"]: r for r in refs}
    snapshot = verify_ref(bundle, manifest["public_code_snapshot"])
    with zipfile.ZipFile(snapshot) as archive:
        members = archive.infolist()
        need(len(members) == len(expected) and {m.filename for m in members} == set(expected),
             "PUBLIC_CODE_SNAPSHOT_MEMBER_SET_MISMATCH")
        for member in members:
            need(not member.is_dir() and stat.S_IFMT(member.external_attr >> 16) != stat.S_IFLNK,
                 "PUBLIC_CODE_SNAPSHOT_REGULAR_FILES_ONLY")
            row = expected[member.filename]
            target = safe_path(code, member.filename)
            raw = archive.read(member)
            need((hashlib.sha256(raw).hexdigest(), len(raw)) == (row["sha256"], row["bytes"]),
                 "PUBLIC_CODE_SNAPSHOT_SHA_MISMATCH")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                need(digest_file(target) == (row["sha256"], row["bytes"]), "PUBLIC_CODE_CHANGED_NO_OVERWRITE")
            else:
                with target.open("xb") as stream:
                    stream.write(raw)
    for name in ("paper3_one_click.py", "paper3_transport.py", "paper3_github.py", "release_manifest.json"):
        src = bundle / name
        sha, size = digest_file(src)
        copy_exact(src, safe_path(code, name), {"sha256": sha, "bytes": size})
    public_index = [{k: row[k] for k in ("path", "sha256", "bytes")} for row in refs]
    index_file = safe_path(code, "code_snapshot_manifest.json")
    if not index_file.exists():
        write_json(index_file, public_index)
    else:
        need(json.loads(index_file.read_bytes()) == public_index, "PUBLIC_CODE_INDEX_CHANGED")
    return code


def verify_deployed_release(project, manifest):
    verify_ref(project, manifest["index"])
    value = json.loads((project / INDEX).read_bytes())
    digest = hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != "index_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    need(value.get("index_sha256") == digest == INDEX_SEMANTIC
         and value.get("closure_sha256") == RELEASE and len(value["files"]) == 21573,
         "DEPLOYED_INDEX_SEMANTIC_MISMATCH")
    need(digest_file(project / RESULT_HELPER)[0] == RESULT_HELPER_SHA,
         "RESULT_BACKUP_ENTRY_SHA_MISMATCH")
    return value


def run_child(command, cwd, env, log_path):
    """Capture trusted producer output; propagate Ctrl-C to our own child only."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        interrupted = False
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                returncode = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                raise LaunchError("OWNED_TRAINING_CHILD_STILL_STOPPING_NO_BACKUP_OR_RESTART") from None
        stream.flush()
        os.fsync(stream.fileno())
    return {"actual_exit_code": returncode, "interrupted_by_user": interrupted,
            "log_file": str(log_path.relative_to(cwd.parent)) if log_path.is_relative_to(cwd.parent) else log_path.name}


def training_and_backup(project, work, credentials, state, *, prepare_only=False, backup_only=False,
                        child_runner=run_child):
    env = training_environment(credentials)
    attempt = uuid.uuid4().hex[:12]
    logs = safe_path(work, "logs")
    logs.mkdir(exist_ok=True)
    if not backup_only:
        previous = state.get("training", {})
        # A result-upload failure never turns a successful scientific entry
        # into a request to repeat it. A paused campaign can use core resume.
        if previous.get("actual_exit_code") != 0 or prepare_only:
            command = [sys.executable, "-B", "tools/bootstrap_data_plan.py", "--index", INDEX, "--execute"]
            if not prepare_only:
                command += ["--launch-training", "--user-started-gpu"]
            print("正在准备固定环境和数据。训练日志：" + str(logs / (attempt + "_training.log")), flush=True)
            state["entry_issued_at_utc"] = utc()
            save_progress(work, state)
            result = child_runner(command, project, env, logs / (attempt + "_training.log"))
            result["returned_at_utc"] = utc()
            if prepare_only:
                state["preparation"] = result
                save_progress(work, state)
                return result["actual_exit_code"]
            state["training"] = result
            state["attempts"].append({"kind": "training_entry", **result})
            save_progress(work, state)
    python = safe_path(project, ".paper3-deployment/venv/bin") / "python"
    # The official venv python is intentionally a symlink to the pinned
    # portable interpreter. Verify its confined resolution instead of
    # applying payload no-symlink checks to this expected interpreter link.
    if not python.exists():
        state["result_backup"] = {"status": "NOT_RUN_NO_PREPARED_ENVIRONMENT", "uploaded_claim": False}
        save_progress(work, state)
        return state.get("training", {}).get("actual_exit_code", 2) or 2
    resolved = python.resolve(strict=True)
    need(resolved.is_relative_to(project / ".paper3-deployment") and resolved.is_file(),
         "RESULT_BACKUP_PYTHON_OUTSIDE_OWNED_RUNTIME")
    command = [str(python), "-B", RESULT_HELPER, "--root", str(project), "--execute"]
    result = child_runner(command, project, env, logs / (attempt + "_backup.log"))
    result["returned_at_utc"] = utc()
    state["result_backup"] = result
    state["attempts"].append({"kind": "result_backup_only", **result})
    save_progress(work, state)
    original = state.get("training", {}).get("actual_exit_code", 0 if backup_only else 2)
    return original if original != 0 else result["actual_exit_code"]


def run(args):
    bundle = canonical_root(Path(__file__).parent)
    manifest = load_bundle(bundle)
    if args.check_bundle:
        print("启动包与固定 SHA256 校验通过；未联网、未推送、未训练。")
        return 0
    need(args.run_on_this_gpu or args.prepare_only or args.backup_only,
         "CHOOSE_RUN_ON_THIS_GPU_PREPARE_ONLY_OR_BACKUP_ONLY")
    need(sys.platform.startswith("linux") and sys.version_info >= (3, 10), "LINUX_PYTHON310_OR_NEWER_REQUIRED")
    need(sys.stdin.isatty(), "INTERACTIVE_TERMINAL_REQUIRED_NO_ECHO_FALLBACK")
    work = canonical_root(args.work_dir or bundle / ("work_" + RELEASE[:16]))
    need(not bundle.is_relative_to(work), "WORK_DIRECTORY_MUST_NOT_CONTAIN_LAUNCHER_BUNDLE")
    with owner_lock(work):
        state = current_state(work)
        save_progress(work, state)
        sys.path.insert(0, str(bundle))
        transport = importlib.import_module("paper3_transport")
        github = importlib.import_module("paper3_github")
        transport.ensure_sdk(bundle, manifest["sdk_wheels"], work)
        if args.backup_only:
            repo = branch = None
        else:
            need(shutil.which("git") is not None, "GIT_EXECUTABLE_REQUIRED")
            repo = github.validate_repo(args.repo)
            target = state.get("github_target", state.get("github", {}))
            need(not target or target.get("repo") == repo, "RESUME_REPOSITORY_CHANGED_USE_NEW_WORK_DIRECTORY")
            branch = github.validate_branch(args.branch or target.get("branch")
                or "research/paper3-oneclick-20261021-" + uuid.uuid4().hex[:6])
            need(not target or target.get("branch") == branch, "RESUME_BRANCH_CHANGED_USE_NEW_WORK_DIRECTORY")
            state["github_target"] = {"repo": repo, "branch": branch}
            save_progress(work, state)
        credentials = collect_credentials(not args.backup_only)
        print("凭证已仅在内存中接收。正在核验已授权的固定 R2 对象……", flush=True)
        client = transport.create_client(credentials["endpoint"], credentials["bucket"],
                                         credentials["access_key"], credentials["secret_key"])
        downloads = safe_path(work, "verified-downloads")
        downloads.mkdir(exist_ok=True)
        state.setdefault("r2_objects", {})
        state["r2_objects"]["ready"] = transport.fetch_object(client, credentials["bucket"],
            manifest["objects"]["ready"], downloads)
        save_progress(work, state)
        state["r2_write_read_probe"] = transport.probe_write(client, credentials["bucket"],
            safe_path(work, "credential-probes"), manifest)
        save_progress(work, state)
        if not args.backup_only:
            need(shutil.which("git") is not None, "GIT_EXECUTABLE_REQUIRED")
            code = prepare_code_directory(bundle, work, manifest)
            state["github"] = github.publish_code(repo, branch, credentials.pop("github_token"), code)
            save_progress(work, state)
            print("GitHub 已实际推送并核验提交：" + state["github"]["commit"], flush=True)
        for name in ("main_inputs", "control_inputs", "main_archive", "control_archive"):
            state["r2_objects"][name] = transport.fetch_object(client, credentials["bucket"],
                manifest["objects"][name], downloads)
            save_progress(work, state)
        releases = safe_path(work, "release")
        releases.mkdir(exist_ok=True)
        for archive_name, input_name in (("main_archive", "main_inputs"), ("control_archive", "control_inputs")):
            state.setdefault("extraction", {})[archive_name] = transport.extract_exact(
                downloads / manifest["objects"][archive_name]["path"],
                downloads / manifest["objects"][input_name]["path"], releases, manifest["project_topdir"])
            save_progress(work, state)
        project = releases / manifest["project_topdir"]
        for name in ("main_inputs", "control_inputs", "ready"):
            row = manifest["objects"][name]
            copy_exact(downloads / row["path"], safe_path(project, row["path"]), row)
        verify_deployed_release(project, manifest)
        state["fixed_release_verified"] = True
        save_progress(work, state)
        print("固定包、精确成员 SHA 和训练索引已核验。阶段门禁由原训练入口执行。", flush=True)
        code = training_and_backup(project, work, credentials, state, prepare_only=args.prepare_only,
                                   backup_only=args.backup_only)
        credentials.clear()
        if code == 0:
            print("本次入口正常返回。训练状态与结果上传以 RUN_PROGRESS.json 及原 producer/R2 回执为准。")
        else:
            print("入口暂停或未完成；原 claim/checkpoint/回执已保留。重试上传请使用 --backup-only。")
        return code if code >= 0 else 130


def main(argv=None):
    parser = argparse.ArgumentParser(description="论文3：隐藏输入凭证、自动推送代码、固定包恢复与训练。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-on-this-gpu", action="store_true", help="在用户已手动开机的本机 GPU 上运行原 pilot 和训练阶段")
    mode.add_argument("--prepare-only", action="store_true", help="推送代码、恢复数据和依赖，暂不训练")
    mode.add_argument("--backup-only", action="store_true", help="只重试已产生结果的 R2 备份，不启动训练")
    mode.add_argument("--check-bundle", action="store_true", help="仅本地启动包 SHA 核验，无凭证输入或网络")
    parser.add_argument("--repo", default="jackzhu119/research3")
    parser.add_argument("--branch", help="新研究分支；默认自动生成 research/paper3-oneclick-20261021-xxxxxx")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    if not any((args.run_on_this_gpu, args.prepare_only, args.backup_only, args.check_bundle)):
        parser.print_help()
        return 0
    try:
        return run(args)
    except KeyboardInterrupt:
        print("已中断；保留原数据和训练 claim。请先确认原协调器已停止，再重跑同一命令。")
        return 130
    except Exception as exc:
        code = str(exc) if isinstance(exc, LaunchError) else getattr(exc, "code", type(exc).__name__)
        if not isinstance(code, str) or re.fullmatch(r"[A-Za-z0-9_]+", code) is None:
            code = type(exc).__name__
        print("本步骤未完成：" + code + "。未完成的推送、训练或上传不会记为成功。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
