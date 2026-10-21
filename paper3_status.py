#!/usr/bin/env python3
"""Read-only Paper3 terminal status. Never reads credentials or producer logs."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import stat
import time

RELEASE = "603d299351fa6605aa0bc258bbbe2c421b2048e5becbeb33521e119a62f20fd7"
WORK_NAME = "work_" + RELEASE[:16]
PROJECT_NAME = "paper3_data_plan_" + RELEASE[:16]
EXPECTED_SDK_FILES = 2208
SHANGHAI = timezone(timedelta(hours=8))
OBJECTS = ("ready", "main_inputs", "control_inputs", "main_archive", "control_archive")
MAX_STATE_BYTES = 4 * 1024 * 1024


class StatusError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _safe_path(value):
    path = Path(os.path.abspath(os.fspath(value)))
    for current in (path, *path.parents):
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise StatusError("STATUS_SYMLINK_FORBIDDEN")
    return path


def _regular_count(root, not_directory_code):
    """Count completed regular files; do not follow directory/file symlinks."""
    root = _safe_path(root)
    if not root.exists():
        return 0
    if not root.is_dir():
        raise StatusError(not_directory_code)
    count = 0
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            if not (entry.name.startswith(".paper3-") and entry.name.endswith(".partial")):
                                count += 1
                    except FileNotFoundError:
                        # The owning writer may have atomically renamed a temp.
                        continue
        except FileNotFoundError:
            continue
    return count


def _sdk_count(root):
    return _regular_count(root, "STATUS_SDK_NOT_DIRECTORY")


def _read_state(path):
    path = _safe_path(path)
    if not path.exists():
        return None, "missing"
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        mode = os.fstat(stream.fileno())
        if not stat.S_ISREG(mode.st_mode) or mode.st_size > MAX_STATE_BYTES:
            raise StatusError("STATUS_PROGRESS_FILE_INVALID")
        raw = stream.read(MAX_STATE_BYTES + 1)
    if len(raw) > MAX_STATE_BYTES:
        raise StatusError("STATUS_PROGRESS_FILE_TOO_LARGE")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        return None, "temporarily_invalid"
    if (not isinstance(value, dict)
            or value.get("schema_version") != "paper3-oneclick-progress-v1"
            or value.get("closure_sha256") != RELEASE):
        raise StatusError("STATUS_WRONG_PROGRESS_RELEASE")
    return value, "valid"


def _dict(value):
    return value if isinstance(value, dict) else {}


def _exit_code(value):
    code = _dict(value).get("actual_exit_code")
    return code if type(code) is int else None


def _timestamp(value):
    if not isinstance(value, str) or len(value) >= 100:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def collect_status(bundle):
    """Return only allowlisted, nonsensitive fields from the existing progress."""
    bundle = _safe_path(bundle)
    work = _safe_path(bundle / WORK_NAME)
    sdk_files = _sdk_count(work / "sdk_site")
    release_files = _regular_count(work / "release" / PROJECT_NAME,
                                  "STATUS_RELEASE_NOT_DIRECTORY")
    state, state_kind = _read_state(work / "RUN_PROGRESS.json")
    value = state or {}
    github = _dict(value.get("github"))
    commit = github.get("commit")
    github_verified = (github.get("status") == "GITHUB_CODE_PUSH_AND_REMOTE_SHA_VERIFIED"
                       and isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None)
    objects = _dict(value.get("r2_objects"))
    network = cached = 0
    for name in OBJECTS:
        row = _dict(objects.get(name))
        if (row.get("status") == "R2_FULL_GET_SHA_VERIFIED"
                and row.get("get_attempted") is True
                and row.get("complete_body_read") is True
                and row.get("http_status") == 200):
            network += 1
        elif row.get("status") == "LOCAL_VERIFIED_CACHE_HIT" and row.get("get_attempted") is False:
            cached += 1
    probe = (_dict(value.get("r2_write_read_probe")).get("status")
             == "R2_WRITE_AND_FULL_GET_PROBE_VERIFIED")
    training_exit = _exit_code(value.get("training"))
    backup_exit = _exit_code(value.get("result_backup"))
    prepare_exit = _exit_code(value.get("preparation"))
    issued = isinstance(value.get("entry_issued_at_utc"), str)
    issued_at = _timestamp(value.get("entry_issued_at_utc"))
    returned_at = [_timestamp(_dict(value.get(name)).get("returned_at_utc"))
                   for name in ("training", "preparation")
                   if _exit_code(value.get(name)) is not None]
    returned_at = [date for date in returned_at if date is not None]
    pending = issued and training_exit is None and prepare_exit is None
    if issued_at is not None and returned_at:
        pending = issued_at > max(returned_at)
    fixed = value.get("fixed_release_verified") is True
    target_chosen = isinstance(value.get("github_target"), dict)
    if pending:
        stage = "原部署/训练入口已派发，尚无本次返回回执"
    elif backup_exit is not None:
        stage = "结果备份入口已返回；请以原 R2 完整回读回执核对结果"
    elif training_exit is not None:
        stage = "原训练入口已返回，等待或执行实际结果备份"
    elif prepare_exit is not None:
        stage = "原准备入口已返回，尚无训练返回回执"
    elif issued and training_exit is None and prepare_exit is None:
        stage = "原部署/训练入口已派发，尚无返回回执"
    elif fixed:
        stage = "固定部署包已校验，等待原入口"
    elif network + cached == len(OBJECTS):
        stage = "固定对象已取得，正在核验解包"
    elif github_verified:
        stage = "代码推送已核验，恢复固定部署包"
    elif probe:
        stage = "R2 写入及完整回读已实测，等待或执行代码推送"
    elif target_chosen:
        stage = "SDK 准备已返回，等待凭证输入或连接核验"
    elif sdk_files:
        stage = "凭证提示前：离线 SDK 展开/校验"
    elif work.exists():
        stage = "本地启动包及 SDK 准备"
    else:
        stage = "等待启动器创建本次工作目录"
    age = None
    timestamp = _timestamp(value.get("updated_at_utc"))
    if timestamp is not None:
        age = max(0, int((datetime.now(timezone.utc) - timestamp).total_seconds()))
    return {
        "observed_at_shanghai": datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage, "sdk_files": sdk_files, "sdk_expected_files": EXPECTED_SDK_FILES,
        "release_files_on_disk": release_files,
        "state_kind": state_kind, "github_verified": github_verified,
        "github_commit_short": commit[:12] if github_verified else None,
        "r2_network_full_GET_objects": network, "r2_local_cache_objects": cached,
        "r2_expected_objects": len(OBJECTS), "r2_write_read_probe_verified": probe,
        "fixed_release_verified": fixed, "entry_issued_without_return": pending,
        "training_entry_exit_code": training_exit, "backup_entry_exit_code": backup_exit,
        "preparation_entry_exit_code": prepare_exit, "progress_update_age_seconds": age,
    }


def format_status(value):
    github = ("已推送并核验 " + value["github_commit_short"]) if value["github_verified"] else "未见已推送回执"
    training = value["training_entry_exit_code"]
    backup = value["backup_entry_exit_code"]
    training_text = ("已返回，退出码 " + str(training)) if training is not None else "未见返回回执"
    if value["entry_issued_without_return"] and training is not None:
        training_text = "本次未见返回；上次退出码 " + str(training)
    backup_text = ("已返回，退出码 " + str(backup)) if backup is not None else "未见返回回执"
    delta = value.get("release_files_delta")
    change = ("；相邻快照 " + format(delta, "+d")) if type(delta) is int else "；首个快照"
    lines = [
        "[北京时间 " + value["observed_at_shanghai"] + "] " + value["stage"],
        "SDK 文件：" + str(value["sdk_files"]) + "/" + str(value["sdk_expected_files"])
        + "；已有文件仍会重新校验，数量不变不等于卡死。",
        "已落盘文件：" + str(value["release_files_on_disk"]) + change
        + "（含后续恢复/结果，不单独代表验证通过）。",
        "GitHub：" + github,
        "R2 固定对象：完整网络 GET " + str(value["r2_network_full_GET_objects"])
        + "，本地 SHA 命中 " + str(value["r2_local_cache_objects"])
        + "，目标共 " + str(value["r2_expected_objects"])
        + "；写读探针：" + ("已通过" if value["r2_write_read_probe_verified"] else "未见通过回执"),
        "训练入口：" + training_text + "；备份入口：" + backup_text,
    ]
    if value["progress_update_age_seconds"] is not None:
        lines.append("进度回执最近更新：" + str(value["progress_update_age_seconds"]) + " 秒前。")
    if value["state_kind"] == "temporarily_invalid":
        lines.append("进度 JSON 暂不可解析，等待下次只读刷新。")
    lines.append("每10秒读取已有回执；入口状态不等于科学完成。凭证在上方提示处输入。")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="论文3只读进度：不启动、不终止或重跑任务。")
    parser.add_argument("--bundle", type=Path, default=Path("/workspace/Paper3_OneClick"))
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not 2 <= args.interval <= 60:
        parser.error("刷新间隔需在 2–60 秒内")
    try:
        previous_files = None
        while True:
            value = collect_status(args.bundle)
            value["release_files_delta"] = (value["release_files_on_disk"] - previous_files
                                             if previous_files is not None else None)
            previous_files = value["release_files_on_disk"]
            print(format_status(value), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except (OSError, StatusError):
        print("只读监视暂不可用：目录权限、路径或进度身份未通过检查；未改变运行任务。", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
