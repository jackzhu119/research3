#!/usr/bin/env python3
"""Attach or launch the existing frozen Paper3 bundle; never manage a Pod."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

ZIP_NAME = 'Paper3_OneClick_20261021.zip'
ZIP_SHA = '5bb056d39254854707e91c4eb2ce2a4cc6e8e922a9337a52f3daea08b0fb5489'
ZIP_BYTES = 17717251
TOP = 'Paper3_OneClick'
WORK = 'work_603d299351fa6605'
RELEASE = '603d299351fa6605aa0bc258bbbe2c421b2048e5becbeb33521e119a62f20fd7'
MONITOR_SOURCE = '#!/usr/bin/env python3\n"""Read-only Paper3 terminal status. Never reads credentials or producer logs."""\nfrom __future__ import annotations\n\nimport argparse\nfrom datetime import datetime, timedelta, timezone\nimport json\nimport os\nfrom pathlib import Path\nimport re\nimport stat\nimport time\n\nRELEASE = "603d299351fa6605aa0bc258bbbe2c421b2048e5becbeb33521e119a62f20fd7"\nWORK_NAME = "work_" + RELEASE[:16]\nPROJECT_NAME = "paper3_data_plan_" + RELEASE[:16]\nEXPECTED_SDK_FILES = 2208\nSHANGHAI = timezone(timedelta(hours=8))\nOBJECTS = ("ready", "main_inputs", "control_inputs", "main_archive", "control_archive")\nMAX_STATE_BYTES = 4 * 1024 * 1024\n\n\nclass StatusError(ValueError):\n    def __init__(self, code):\n        self.code = code\n        super().__init__(code)\n\n\ndef _safe_path(value):\n    path = Path(os.path.abspath(os.fspath(value)))\n    for current in (path, *path.parents):\n        try:\n            mode = current.lstat().st_mode\n        except FileNotFoundError:\n            continue\n        if stat.S_ISLNK(mode):\n            raise StatusError("STATUS_SYMLINK_FORBIDDEN")\n    return path\n\n\ndef _regular_count(root, not_directory_code):\n    """Count completed regular files; do not follow directory/file symlinks."""\n    root = _safe_path(root)\n    if not root.exists():\n        return 0\n    if not root.is_dir():\n        raise StatusError(not_directory_code)\n    count = 0\n    pending = [root]\n    while pending:\n        current = pending.pop()\n        try:\n            with os.scandir(current) as entries:\n                for entry in entries:\n                    try:\n                        if entry.is_symlink():\n                            continue\n                        if entry.is_dir(follow_symlinks=False):\n                            pending.append(Path(entry.path))\n                        elif entry.is_file(follow_symlinks=False):\n                            if not (entry.name.startswith(".paper3-") and entry.name.endswith(".partial")):\n                                count += 1\n                    except FileNotFoundError:\n                        # The owning writer may have atomically renamed a temp.\n                        continue\n        except FileNotFoundError:\n            continue\n    return count\n\n\ndef _sdk_count(root):\n    return _regular_count(root, "STATUS_SDK_NOT_DIRECTORY")\n\n\ndef _read_state(path):\n    path = _safe_path(path)\n    if not path.exists():\n        return None, "missing"\n    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)\n    with os.fdopen(descriptor, "rb") as stream:\n        mode = os.fstat(stream.fileno())\n        if not stat.S_ISREG(mode.st_mode) or mode.st_size > MAX_STATE_BYTES:\n            raise StatusError("STATUS_PROGRESS_FILE_INVALID")\n        raw = stream.read(MAX_STATE_BYTES + 1)\n    if len(raw) > MAX_STATE_BYTES:\n        raise StatusError("STATUS_PROGRESS_FILE_TOO_LARGE")\n    try:\n        value = json.loads(raw)\n    except (ValueError, UnicodeError):\n        return None, "temporarily_invalid"\n    if (not isinstance(value, dict)\n            or value.get("schema_version") != "paper3-oneclick-progress-v1"\n            or value.get("closure_sha256") != RELEASE):\n        raise StatusError("STATUS_WRONG_PROGRESS_RELEASE")\n    return value, "valid"\n\n\ndef _dict(value):\n    return value if isinstance(value, dict) else {}\n\n\ndef _exit_code(value):\n    code = _dict(value).get("actual_exit_code")\n    return code if type(code) is int else None\n\n\ndef _timestamp(value):\n    if not isinstance(value, str) or len(value) >= 100:\n        return None\n    try:\n        parsed = datetime.fromisoformat(value)\n        return parsed if parsed.tzinfo is not None else None\n    except (ValueError, OverflowError):\n        return None\n\n\ndef collect_status(bundle):\n    """Return only allowlisted, nonsensitive fields from the existing progress."""\n    bundle = _safe_path(bundle)\n    work = _safe_path(bundle / WORK_NAME)\n    sdk_files = _sdk_count(work / "sdk_site")\n    release_files = _regular_count(work / "release" / PROJECT_NAME,\n                                  "STATUS_RELEASE_NOT_DIRECTORY")\n    state, state_kind = _read_state(work / "RUN_PROGRESS.json")\n    value = state or {}\n    github = _dict(value.get("github"))\n    commit = github.get("commit")\n    github_verified = (github.get("status") == "GITHUB_CODE_PUSH_AND_REMOTE_SHA_VERIFIED"\n                       and isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None)\n    objects = _dict(value.get("r2_objects"))\n    network = cached = 0\n    for name in OBJECTS:\n        row = _dict(objects.get(name))\n        if (row.get("status") == "R2_FULL_GET_SHA_VERIFIED"\n                and row.get("get_attempted") is True\n                and row.get("complete_body_read") is True\n                and row.get("http_status") == 200):\n            network += 1\n        elif row.get("status") == "LOCAL_VERIFIED_CACHE_HIT" and row.get("get_attempted") is False:\n            cached += 1\n    probe = (_dict(value.get("r2_write_read_probe")).get("status")\n             == "R2_WRITE_AND_FULL_GET_PROBE_VERIFIED")\n    training_exit = _exit_code(value.get("training"))\n    backup_exit = _exit_code(value.get("result_backup"))\n    prepare_exit = _exit_code(value.get("preparation"))\n    issued = isinstance(value.get("entry_issued_at_utc"), str)\n    issued_at = _timestamp(value.get("entry_issued_at_utc"))\n    returned_at = [_timestamp(_dict(value.get(name)).get("returned_at_utc"))\n                   for name in ("training", "preparation")\n                   if _exit_code(value.get(name)) is not None]\n    returned_at = [date for date in returned_at if date is not None]\n    pending = issued and training_exit is None and prepare_exit is None\n    if issued_at is not None and returned_at:\n        pending = issued_at > max(returned_at)\n    fixed = value.get("fixed_release_verified") is True\n    target_chosen = isinstance(value.get("github_target"), dict)\n    if pending:\n        stage = "原部署/训练入口已派发，尚无本次返回回执"\n    elif backup_exit is not None:\n        stage = "结果备份入口已返回；请以原 R2 完整回读回执核对结果"\n    elif training_exit is not None:\n        stage = "原训练入口已返回，等待或执行实际结果备份"\n    elif prepare_exit is not None:\n        stage = "原准备入口已返回，尚无训练返回回执"\n    elif issued and training_exit is None and prepare_exit is None:\n        stage = "原部署/训练入口已派发，尚无返回回执"\n    elif fixed:\n        stage = "固定部署包已校验，等待原入口"\n    elif network + cached == len(OBJECTS):\n        stage = "固定对象已取得，正在核验解包"\n    elif github_verified:\n        stage = "代码推送已核验，恢复固定部署包"\n    elif probe:\n        stage = "R2 写入及完整回读已实测，等待或执行代码推送"\n    elif target_chosen:\n        stage = "SDK 准备已返回，等待凭证输入或连接核验"\n    elif sdk_files:\n        stage = "凭证提示前：离线 SDK 展开/校验"\n    elif work.exists():\n        stage = "本地启动包及 SDK 准备"\n    else:\n        stage = "等待启动器创建本次工作目录"\n    age = None\n    timestamp = _timestamp(value.get("updated_at_utc"))\n    if timestamp is not None:\n        age = max(0, int((datetime.now(timezone.utc) - timestamp).total_seconds()))\n    return {\n        "observed_at_shanghai": datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S"),\n        "stage": stage, "sdk_files": sdk_files, "sdk_expected_files": EXPECTED_SDK_FILES,\n        "release_files_on_disk": release_files,\n        "state_kind": state_kind, "github_verified": github_verified,\n        "github_commit_short": commit[:12] if github_verified else None,\n        "r2_network_full_GET_objects": network, "r2_local_cache_objects": cached,\n        "r2_expected_objects": len(OBJECTS), "r2_write_read_probe_verified": probe,\n        "fixed_release_verified": fixed, "entry_issued_without_return": pending,\n        "training_entry_exit_code": training_exit, "backup_entry_exit_code": backup_exit,\n        "preparation_entry_exit_code": prepare_exit, "progress_update_age_seconds": age,\n    }\n\n\ndef format_status(value):\n    github = ("已推送并核验 " + value["github_commit_short"]) if value["github_verified"] else "未见已推送回执"\n    training = value["training_entry_exit_code"]\n    backup = value["backup_entry_exit_code"]\n    training_text = ("已返回，退出码 " + str(training)) if training is not None else "未见返回回执"\n    if value["entry_issued_without_return"] and training is not None:\n        training_text = "本次未见返回；上次退出码 " + str(training)\n    backup_text = ("已返回，退出码 " + str(backup)) if backup is not None else "未见返回回执"\n    delta = value.get("release_files_delta")\n    change = ("；相邻快照 " + format(delta, "+d")) if type(delta) is int else "；首个快照"\n    lines = [\n        "[北京时间 " + value["observed_at_shanghai"] + "] " + value["stage"],\n        "SDK 文件：" + str(value["sdk_files"]) + "/" + str(value["sdk_expected_files"])\n        + "；已有文件仍会重新校验，数量不变不等于卡死。",\n        "已落盘文件：" + str(value["release_files_on_disk"]) + change\n        + "（含后续恢复/结果，不单独代表验证通过）。",\n        "GitHub：" + github,\n        "R2 固定对象：完整网络 GET " + str(value["r2_network_full_GET_objects"])\n        + "，本地 SHA 命中 " + str(value["r2_local_cache_objects"])\n        + "，目标共 " + str(value["r2_expected_objects"])\n        + "；写读探针：" + ("已通过" if value["r2_write_read_probe_verified"] else "未见通过回执"),\n        "训练入口：" + training_text + "；备份入口：" + backup_text,\n    ]\n    if value["progress_update_age_seconds"] is not None:\n        lines.append("进度回执最近更新：" + str(value["progress_update_age_seconds"]) + " 秒前。")\n    if value["state_kind"] == "temporarily_invalid":\n        lines.append("进度 JSON 暂不可解析，等待下次只读刷新。")\n    lines.append("每10秒读取已有回执；入口状态不等于科学完成。凭证在上方提示处输入。")\n    return "\\n".join(lines)\n\n\ndef main(argv=None):\n    parser = argparse.ArgumentParser(description="论文3只读进度：不启动、不终止或重跑任务。")\n    parser.add_argument("--bundle", type=Path, default=Path("/workspace/Paper3_OneClick"))\n    parser.add_argument("--interval", type=float, default=10.0)\n    parser.add_argument("--once", action="store_true")\n    args = parser.parse_args(argv)\n    if not 2 <= args.interval <= 60:\n        parser.error("刷新间隔需在 2–60 秒内")\n    try:\n        previous_files = None\n        while True:\n            value = collect_status(args.bundle)\n            value["release_files_delta"] = (value["release_files_on_disk"] - previous_files\n                                             if previous_files is not None else None)\n            previous_files = value["release_files_on_disk"]\n            print(format_status(value), flush=True)\n            if args.once:\n                return 0\n            time.sleep(args.interval)\n    except KeyboardInterrupt:\n        return 0\n    except (OSError, StatusError):\n        print("只读监视暂不可用：目录权限、路径或进度身份未通过检查；未改变运行任务。", flush=True)\n        return 2\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
FETCH_SOURCE = '#!/usr/bin/env python3\n"""Fetch one pinned GitHub release asset with whole-body SHA256 verification.\n\nNo service exception or signed redirect URL is printed. Authorization is sent\nonly to api.github.com; redirects to GitHub asset hosts never carry that header.\n"""\nfrom __future__ import annotations\n\nimport argparse\nimport getpass\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport re\nimport stat\nimport sys\nimport tempfile\nimport warnings\nfrom urllib import error, parse, request\n\nCHUNK = 1024 * 1024\nMAX_REDIRECTS = 6\nALLOWED_HOSTS = frozenset({\'api.github.com\', \'github.com\',\n    \'release-assets.githubusercontent.com\', \'objects.githubusercontent.com\',\n    \'github-releases.githubusercontent.com\'})\n\n\nclass FetchError(ValueError):\n    def __init__(self, code):\n        self.code = code\n        super().__init__(code)\n\n\ndef _need(value, code):\n    if not value:\n        raise FetchError(code)\n\n\ndef _safe_url(url):\n    try:\n        value = parse.urlsplit(url)\n        safe = (value.scheme == \'https\' and value.hostname in ALLOWED_HOSTS\n                and value.username is None and value.password is None\n                and value.port in {None, 443} and not value.fragment)\n    except (TypeError, ValueError):\n        safe = False\n    _need(safe, \'GITHUB_RELEASE_REDIRECT_DESTINATION_REFUSED\')\n    return value\n\n\nclass _NoRedirect(request.HTTPRedirectHandler):\n    def redirect_request(self, req, fp, code, msg, headers, newurl):\n        return None\n\n\ndef _open_once(req):\n    # Default ProxyHandler/HTTPS trust are preserved.\n    return request.build_opener(_NoRedirect()).open(req, timeout=60)\n\n\ndef _open_get(url, accept, token):\n    for _ in range(MAX_REDIRECTS + 1):\n        parsed = _safe_url(url)\n        headers = {\'Accept\': accept, \'User-Agent\': \'Paper3-fixed-online-bootstrap\',\n                   \'X-GitHub-Api-Version\': \'2022-11-28\'}\n        if token is not None and parsed.hostname == \'api.github.com\':\n            headers[\'Authorization\'] = \'Bearer \' + token\n        req = request.Request(url, headers=headers, method=\'GET\')\n        response = None\n        try:\n            response = _open_once(req)\n            status = response.getcode()\n        except error.HTTPError as exc:\n            response, status = exc, exc.code\n        except Exception:\n            raise FetchError(\'GITHUB_RELEASE_CONNECTION_FAILED\') from None\n        if status in {301, 302, 303, 307, 308}:\n            target = response.headers.get(\'Location\')\n            response.close()\n            _need(isinstance(target, str) and target, \'GITHUB_RELEASE_REDIRECT_LOCATION_MISSING\')\n            url = parse.urljoin(url, target)\n            _safe_url(url)\n            continue\n        if status == 200:\n            return response\n        response.close()\n        if status == 401:\n            raise FetchError(\'GITHUB_RELEASE_AUTH_REQUIRED\')\n        if status == 404:\n            raise FetchError(\'GITHUB_RELEASE_NOT_FOUND_OR_AUTH_REQUIRED\')\n        if status == 403:\n            raise FetchError(\'GITHUB_RELEASE_ACCESS_DENIED_OR_RATE_LIMIT\')\n        if status == 429:\n            raise FetchError(\'GITHUB_RELEASE_RATE_LIMIT\')\n        raise FetchError(\'GITHUB_RELEASE_HTTP_NOT_200\')\n    raise FetchError(\'GITHUB_RELEASE_TOO_MANY_REDIRECTS\')\n\n\ndef _path(path):\n    path = Path(os.path.abspath(path))\n    _need(not any(p.is_symlink() for p in (path, *path.parents)), \'GITHUB_DOWNLOAD_SYMLINK_REFUSED\')\n    if path.exists():\n        _need(stat.S_ISREG(path.lstat().st_mode), \'GITHUB_DOWNLOAD_REGULAR_FILE_REQUIRED\')\n    return path\n\n\ndef _hash(path):\n    h, n = hashlib.sha256(), 0\n    with path.open(\'rb\') as stream:\n        for data in iter(lambda: stream.read(CHUNK), b\'\'):\n            h.update(data)\n            n += len(data)\n    return h.hexdigest(), n\n\n\ndef fetch_release_asset(repo, tag, name, expected_sha, expected_bytes, destination, token=None):\n    """Fetch to a final FILE path. Six positional arguments suffice for public GitHub.\n\n    Existing identical content is reported as a local cache hit, never as a new\n    GET. Wrong existing content is preserved and rejected. `token`, if used, is\n    memory-only and is attached exclusively to api.github.com requests.\n    """\n    _need(isinstance(repo, str) and re.fullmatch(r\'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}\', repo),\n          \'GITHUB_RELEASE_REPOSITORY_INVALID\')\n    _need(isinstance(tag, str) and tag and len(tag) <= 200\n          and not any(ord(c) < 33 or ord(c) == 127 for c in tag), \'GITHUB_RELEASE_TAG_INVALID\')\n    _need(isinstance(name, str) and name and name not in {\'.\', \'..\'} and \'/\' not in name\n          and \'\\\\\' not in name and not any(ord(c) < 32 or ord(c) == 127 for c in name),\n          \'GITHUB_RELEASE_ASSET_NAME_INVALID\')\n    _need(isinstance(expected_sha, str) and re.fullmatch(r\'[a-f0-9]{64}\', expected_sha)\n          and type(expected_bytes) is int and expected_bytes > 0, \'GITHUB_RELEASE_IDENTITY_INVALID\')\n    _need(token is None or (isinstance(token, str) and 1 <= len(token) <= 4096\n          and all(33 <= ord(c) <= 126 for c in token)), \'GITHUB_RELEASE_TOKEN_INVALID\')\n    target = _path(destination)\n    identity = {\'repository\': repo, \'tag\': tag, \'asset_name\': name,\n                \'sha256\': expected_sha, \'bytes\': expected_bytes, \'credentials_recorded\': False}\n    if target.exists():\n        _need(_hash(target) == (expected_sha, expected_bytes), \'EXISTING_GITHUB_ASSET_DIFFERS_PRESERVED\')\n        return dict(identity, status=\'LOCAL_VERIFIED_CACHE_HIT\', get_attempted=False,\n                    network_bytes_read=0, http_status=None, complete_body_read=None)\n    metadata_url = \'https://api.github.com/repos/\' + repo + \'/releases/tags/\' + parse.quote(tag, safe=\'\')\n    response = _open_get(metadata_url, \'application/vnd.github+json\', token)\n    try:\n        raw = response.read(2 * CHUNK + 1)\n        _need(len(raw) <= 2 * CHUNK, \'GITHUB_RELEASE_METADATA_TOO_LARGE\')\n        value = json.loads(raw)\n    except FetchError:\n        raise\n    except Exception:\n        raise FetchError(\'GITHUB_RELEASE_METADATA_INVALID\') from None\n    finally:\n        response.close()\n    _need(isinstance(value, dict) and value.get(\'tag_name\') == tag\n          and value.get(\'draft\') is False and isinstance(value.get(\'assets\'), list),\n          \'GITHUB_RELEASE_METADATA_IDENTITY_MISMATCH\')\n    assets = [row for row in value[\'assets\'] if isinstance(row, dict) and row.get(\'name\') == name]\n    _need(len(assets) == 1, \'GITHUB_RELEASE_EXACT_ASSET_MISSING\')\n    asset = assets[0]\n    _need(type(asset.get(\'id\')) is int and asset[\'id\'] > 0\n          and type(asset.get(\'size\')) is int and asset[\'size\'] == expected_bytes\n          and asset.get(\'state\') == \'uploaded\', \'GITHUB_RELEASE_ASSET_METADATA_MISMATCH\')\n    # For public releases the browser endpoint avoids API asset JSON negotiation.\n    # A private asset must use the API endpoint with the in-memory token.\n    if token is None:\n        asset_url = \'https://github.com/\' + repo + \'/releases/download/\' + parse.quote(tag, safe=\'\') + \'/\' + parse.quote(name, safe=\'\')\n    else:\n        asset_url = \'https://api.github.com/repos/\' + repo + \'/releases/assets/\' + str(asset[\'id\'])\n    response = _open_get(asset_url, \'application/octet-stream\', token)\n    temporary = None\n    try:\n        declared = response.headers.get(\'Content-Length\')\n        if declared is not None:\n            _need(str(declared).isdigit() and int(declared) == expected_bytes,\n                  \'GITHUB_RELEASE_DECLARED_LENGTH_MISMATCH\')\n        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)\n        _path(target)\n        descriptor, temporary = tempfile.mkstemp(prefix=\'.paper3-download-\', suffix=\'.partial\', dir=target.parent)\n        h, n = hashlib.sha256(), 0\n        with os.fdopen(descriptor, \'wb\') as output:\n            while data := response.read(CHUNK):\n                _need(isinstance(data, bytes), \'GITHUB_RELEASE_NON_BINARY_BODY\')\n                n += len(data)\n                _need(n <= expected_bytes, \'GITHUB_RELEASE_BODY_LENGTH_MISMATCH\')\n                h.update(data)\n                output.write(data)\n            _need((h.hexdigest(), n) == (expected_sha, expected_bytes), \'GITHUB_RELEASE_BODY_SHA_OR_LENGTH_MISMATCH\')\n            output.flush()\n            os.fsync(output.fileno())\n        try:\n            os.link(temporary, target)\n        except FileExistsError:\n            raise FetchError(\'GITHUB_DOWNLOAD_TARGET_CREATED_CONCURRENTLY\') from None\n        return dict(identity, status=\'GITHUB_RELEASE_COMPLETE_GET_SHA256_VERIFIED\',\n                    get_attempted=True, network_bytes_read=n, http_status=200,\n                    complete_body_read=True, asset_id=asset[\'id\'], atomic_commit=\'link_no_overwrite\')\n    except FetchError:\n        raise\n    except Exception:\n        raise FetchError(\'GITHUB_RELEASE_DOWNLOAD_FAILED\') from None\n    finally:\n        response.close()\n        if temporary is not None:\n            Path(temporary).unlink(missing_ok=True)\n\n\ndef main(argv=None):\n    parser = argparse.ArgumentParser(description=\'论文3固定 GitHub 发布包下载与完整 SHA256 核验。\')\n    parser.add_argument(\'--repo\', default=\'jackzhu119/research3\')\n    parser.add_argument(\'--tag\', default=\'paper3-online-20261021-v1\')\n    parser.add_argument(\'--name\', default=\'Paper3_OneClick_20261021.zip\')\n    parser.add_argument(\'--sha256\', default=\'5bb056d39254854707e91c4eb2ce2a4cc6e8e922a9337a52f3daea08b0fb5489\')\n    parser.add_argument(\'--bytes\', type=int, default=17717251)\n    parser.add_argument(\'--destination\', type=Path, default=Path(\'/workspace/Paper3_OneClick_20261021.zip\'))\n    args = parser.parse_args(argv)\n    token = None\n    try:\n        try:\n            result = fetch_release_asset(args.repo, args.tag, args.name, args.sha256, args.bytes, args.destination)\n        except FetchError as exc:\n            if exc.code not in {\'GITHUB_RELEASE_AUTH_REQUIRED\', \'GITHUB_RELEASE_NOT_FOUND_OR_AUTH_REQUIRED\'} or not sys.stdin.isatty():\n                raise\n            with warnings.catch_warnings():\n                warnings.simplefilter(\'error\', getpass.GetPassWarning)\n                try:\n                    token = getpass.getpass(\'若该发布需要授权，请输入 GitHub Token（不回显）：\').strip()\n                except getpass.GetPassWarning:\n                    raise FetchError(\'INTERACTIVE_HIDDEN_INPUT_REQUIRED\') from None\n            result = fetch_release_asset(args.repo, args.tag, args.name, args.sha256, args.bytes, args.destination, token)\n        print(json.dumps(result, ensure_ascii=False))\n        return 0\n    except Exception as exc:\n        code = exc.code if isinstance(exc, FetchError) else type(exc).__name__\n        print(\'固定发布包下载未完成：\' + code)\n        return 2\n    finally:\n        token = None\n\n\nif __name__ == \'__main__\':\n    raise SystemExit(main())\n'
ONLINE_REPO = 'jackzhu119/research3'
ONLINE_TAG = 'paper3-online-20261021-v1'


class AutoError(ValueError):
    pass


def need(test, code):
    if not test:
        raise AutoError(code)


def checked_path(path):
    path = Path(os.path.abspath(path))
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINK_PATH_REFUSED')
    return path


def run(command):
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, check=False)
    need(result.returncode == 0, 'LOCAL_TOOL_COMMAND_FAILED')
    return result.stdout.strip()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def active_jobs(workspace, proc_root=Path('/proc')):
    """Exact argv names plus cwd scope; never inspect process environments."""
    jobs = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = [os.fsdecode(a) for a in (entry / 'cmdline').read_bytes().split(b'\0') if a]
            cwd = (entry / 'cwd').resolve(strict=True)
            for arg in args[1:]:
                name = Path(arg).name
                candidate = Path(os.path.abspath(Path(arg) if Path(arg).is_absolute() else cwd / arg))
                launcher = name == 'paper3_one_click.py' and candidate.parent.name == TOP
                core = name in {'bootstrap_data_plan.py', 'run_training_campaign.py',
                                'run_source_extension_downstream.py'} and \
                       ('paper3_data_plan_603d299351fa6605' in candidate.parts)
                inside = name == 'PAPER3_AUTO_RUN.py' and '--inside' in args
                if (launcher or core or inside) and candidate.is_relative_to(workspace):
                    jobs.append({'pid': int(entry.name), 'bundle': candidate.parent if launcher else None})
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
    return jobs


def ancestor_ids(pid, proc_root=Path('/proc')):
    result = set()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            lines = (proc_root / str(pid) / 'status').read_text().splitlines()
            pid = int(next(row.split()[1] for row in lines if row.startswith('PPid:')))
        except (OSError, ValueError, StopIteration):
            break
    return result


def pane_for_jobs(jobs, panes, proc_root=Path('/proc')):
    for job in jobs:
        ancestors = ancestor_ids(job['pid'], proc_root)
        for pane in panes:
            if pane['pid'] in ancestors:
                return pane
    return None


def list_panes():
    result = subprocess.run(['tmux', 'list-panes', '-a', '-F', '#{session_id}\t#{window_id}\t#{pane_id}\t#{pane_pid}'],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    panes = []
    for line in result.stdout.splitlines():
        fields = line.split('\t')
        if len(fields) == 4 and fields[3].isdigit():
            panes.append(dict(zip(('session', 'window', 'pane', 'pid'), fields[:3] + [int(fields[3])])))
    return panes


@contextmanager
def short_manager_lock(workspace):
    path = checked_path(workspace / '.paper3-auto-manager.lock')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AutoError('ANOTHER_AUTO_ENTRY_IS_CHECKING_PLEASE_RETRY') from None
        yield


def original_locks_busy(bundle):
    root = bundle / WORK
    project = root / 'release' / 'paper3_data_plan_603d299351fa6605'
    paths = [root / 'launcher.owner.lock', project / '.paper3-deployment/owner.lock',
             project / 'runs/.training_campaign_v1.lock']
    for path in paths:
        checked_path(path)
        if not path.exists():
            continue
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        with os.fdopen(fd, 'a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    return False


def ensure_bundle(workspace):
    """Use the pinned ZIP; compare existing members, never overwrite work."""
    source = checked_path(workspace / ZIP_NAME)
    if not source.exists():
        download_bundle(workspace)
    need(source.is_file(), 'FIXED_ONLINE_BUNDLE_NOT_AVAILABLE')
    print('正在核验已有启动 ZIP（17.7 MB）……', flush=True)
    need(source.stat().st_size == ZIP_BYTES and file_sha(source) == ZIP_SHA,
         'ORIGINAL_ZIP_SHA_MISMATCH')
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        need(len(names) == len(set(names)), 'ZIP_DUPLICATE_MEMBER')
        for member in members:
            parts = member.filename.split('/')
            need(parts[0] == TOP and all(p not in {'', '.', '..'} for p in parts)
                 and '\\' not in member.filename and not member.is_dir()
                 and stat.S_IFMT(member.external_attr >> 16) in {0, stat.S_IFREG}, 'ZIP_MEMBER_REFUSED')
            target = checked_path(workspace / member.filename)
            payload = archive.read(member)
            if target.exists():
                need(target.is_file() and target.stat().st_size == len(payload)
                     and file_sha(target) == hashlib.sha256(payload).hexdigest(),
                     'EXISTING_LAUNCHER_FILE_DIFFERS_PRESERVED')
                continue
            write_new_file(target, payload)
    print('启动包已核验；已有工作目录、缓存和断点均保留。', flush=True)
    return checked_path(workspace / TOP)


def download_bundle(workspace):
    need(isinstance(FETCH_SOURCE, str), 'USE_BUILT_ONLINE_SINGLE_FILE_HELPER')
    print('正在从 GitHub 固定发布自动下载启动器与SDK（17.7 MB）……', flush=True)
    namespace = {'__name__': 'paper3_online_release_fetch'}
    exec(compile(FETCH_SOURCE, 'embedded_online_fetch.py', 'exec'), namespace)
    try:
        return namespace['fetch_release_asset'](ONLINE_REPO, ONLINE_TAG, ZIP_NAME,
                ZIP_SHA, ZIP_BYTES, checked_path(workspace / ZIP_NAME))
    except Exception as exc:
        code = getattr(exc, 'code', 'GITHUB_FIXED_RELEASE_DOWNLOAD_FAILED')
        raise AutoError(code) from None


def write_new_file(target, payload):
    target = checked_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.paper3-auto-copy-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


def install_helpers(workspace):
    source = Path(__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    root = checked_path(workspace / '.paper3-auto-tools' / digest[:16])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    need(isinstance(MONITOR_SOURCE, str), 'USE_BUILT_SINGLE_FILE_HELPER')
    for name, payload in [('PAPER3_AUTO_RUN.py', source), ('paper3_status.py', MONITOR_SOURCE.encode())]:
        path = checked_path(root / name)
        if path.exists():
            need(path.is_file() and path.read_bytes() == payload, 'AUTO_HELPER_COPY_DIFFERS')
        else:
            write_new_file(path, payload)
    return root


def add_status_pane(pane, helpers, bundle):
    marker = subprocess.run(['tmux', 'show-options', '-wqv', '-t', pane['window'], '@paper3_status_pane'],
                            capture_output=True, text=True, check=False).stdout.strip()
    current = {row['pane']: row for row in list_panes()}
    command = shlex.join([sys.executable, '-B', str(helpers / 'paper3_status.py'), '--bundle', str(bundle)])
    if marker in current:
        if marker == pane['pane']:
            return
        # Upgrade only an identified read-only monitor, never the input/fit pane.
        try:
            pid = current[marker]['pid']
            args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            paths = [Path(os.fsdecode(arg)) for arg in args[1:] if arg]
            script = next((path for path in paths if path.suffix == '.py'), None)
            owned = (script is not None and script.name == 'paper3_status.py'
                     and '.paper3-auto-tools' in script.parts and b'-c' not in args and b'-m' not in args)
            current_version = subprocess.run(['tmux', 'show-options', '-wqv', '-t', pane['window'],
                                               '@paper3_status_version'], capture_output=True, text=True).stdout.strip()
            if owned and current_version != helpers.name:
                run(['tmux', 'respawn-pane', '-k', '-t', marker, command])
                run(['tmux', 'set-option', '-w', '-t', pane['window'], '@paper3_status_version', helpers.name])
        except (OSError, KeyError, ValueError):
            pass
        return
    result = subprocess.run(['tmux', 'split-window', '-d', '-v', '-l', '9', '-t', pane['pane'],
                             '-P', '-F', '#{pane_id}', command], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        run(['tmux', 'set-option', '-w', '-t', pane['window'], '@paper3_status_pane', result.stdout.strip()])
        run(['tmux', 'set-option', '-w', '-t', pane['window'], '@paper3_status_version', helpers.name])
    else:
        print('终端空间不足以分屏；已保留原任务并接入其窗口。', flush=True)


def create_job(helpers, workspace, bundle, repo):
    command = shlex.join([sys.executable, '-B', str(helpers / 'PAPER3_AUTO_RUN.py'), '--inside',
                          '--workspace', str(workspace), '--bundle', str(bundle), '--repo', repo])
    sessions = subprocess.run(['tmux', 'has-session', '-t', '=paper3'], capture_output=True).returncode
    action = ['new-window', '-d', '-t', 'paper3:', '-n', 'train'] if sessions == 0 else \
             ['new-session', '-d', '-s', 'paper3', '-n', 'train']
    text = run(['tmux', *action, '-P', '-F', '#{session_id}\t#{window_id}\t#{pane_id}\t#{pane_pid}', command])
    fields = text.split('\t')
    need(len(fields) == 4 and fields[3].isdigit(), 'TMUX_NEW_PANE_NOT_IDENTIFIED')
    pane = dict(zip(('session', 'window', 'pane', 'pid'), fields[:3] + [int(fields[3])]))
    run(['tmux', 'set-option', '-p', '-t', pane['pane'], '@paper3_launcher_bundle', str(bundle)])
    return pane


def reserved_pane(bundle):
    for pane in list_panes():
        value = subprocess.run(['tmux', 'show-options', '-pqv', '-t', pane['pane'], '@paper3_launcher_bundle'],
                               capture_output=True, text=True, check=False).stdout.strip()
        if value == str(bundle):
            return pane
    return None


def attach(pane):
    run(['tmux', 'select-window', '-t', pane['window']])
    run(['tmux', 'select-pane', '-t', pane['pane']])
    if os.environ.get('TMUX'):
        run(['tmux', 'switch-client', '-t', pane['session']])
        return 0
    os.execvp('tmux', ['tmux', 'attach-session', '-t', pane['session']])


def inside(args):
    print('论文3自动入口：准备进度见下方状态面板；上方出现凭证提示后逐项输入。', flush=True)
    print('正在沿用固定训练入口：GitHub → R2 校验 → 受控 pilot → 科学门禁 → 结果备份。', flush=True)
    result = subprocess.run(['bash', str(args.bundle / 'START.sh'), '--repo', args.repo], cwd=args.bundle)
    print('原启动入口返回码：' + str(result.returncode) + '。完成状态以真实 producer 和 R2 回执为准。', flush=True)
    try:
        input('按 Enter 关闭这个窗口；已有数据、回执和断点保留。')
    except (EOFError, KeyboardInterrupt):
        pass
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description='论文3自动接入/启动，不创建、开机或重启云实例。')
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--repo', default='jackzhu119/research3')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        need(sys.platform.startswith('linux') and sys.version_info >= (3, 10), 'LINUX_PYTHON310_REQUIRED')
        args.workspace = checked_path(args.workspace)
        need(args.workspace.is_dir(), 'WORKSPACE_DIRECTORY_MISSING')
        if args.inside:
            return inside(args)
        need(sys.stdin.isatty(), 'RUN_IN_RUNPOD_TERMINAL')
        missing_tools = [name for name in ('tmux', 'git') if shutil.which(name) is None]
        if missing_tools:
            need(os.geteuid() == 0 and shutil.which('apt-get'), 'REQUIRED_TOOLS_MISSING_NO_AUTOMATIC_INSTALLER')
            print('正在自动安装启动工具：' + ', '.join(missing_tools) + '……', flush=True)
            for command in [['apt-get', 'update'], ['apt-get', 'install', '-y', *missing_tools]]:
                subprocess.run(command, check=True, env={**os.environ, 'DEBIAN_FRONTEND': 'noninteractive'})
        with short_manager_lock(args.workspace):
            helpers = install_helpers(args.workspace)
            jobs = active_jobs(args.workspace)
            monitor_only = False
            if jobs:
                print('发现已有论文3进程；直接接入，不重复启动或重新解压。', flush=True)
                pane = pane_for_jobs(jobs, list_panes())
                bundle = next((job['bundle'] for job in jobs if job['bundle']), args.workspace / TOP)
                if pane is None:
                    print('现有任务位于 tmux 外，保持运行；这里只显示状态，请在原终端输入凭证。', flush=True)
                    # No restart or attempt to move an already-running process.
                    monitor_only = True
            else:
                bundle = checked_path(args.workspace / TOP)
                pane = reserved_pane(bundle)
                if pane is None:
                    need(not original_locks_busy(bundle), 'EXISTING_TRAINING_LOCK_BUSY_NO_DUPLICATE_STARTED')
                    # Recheck all processes under the short manager lock before starting.
                    need(not active_jobs(args.workspace), 'TASK_APPEARED_DURING_CHECK_RETRY_TO_ATTACH')
                    bundle = ensure_bundle(args.workspace)
                    pane = create_job(helpers, args.workspace, bundle, args.repo)
            if not monitor_only:
                add_status_pane(pane, helpers, bundle)
        # Do not let the long-lived tmux client inherit the short manager lock.
        if monitor_only:
            return subprocess.call([sys.executable, '-B', str(helpers / 'paper3_status.py'), '--bundle', str(bundle)])
        return attach(pane)
    except KeyboardInterrupt:
        print('已退出自动入口；未停止或重启已有训练任务。')
        return 130
    except Exception as exc:
        code = str(exc) if isinstance(exc, AutoError) else type(exc).__name__
        print('自动入口未完成：' + code + '。原任务和工作目录保留。')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
