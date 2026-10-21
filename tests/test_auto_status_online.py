from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper3_status as monitor


def put_state(bundle, **fields):
    work = bundle / monitor.WORK_NAME
    work.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": "paper3-oneclick-progress-v1", "closure_sha256": monitor.RELEASE,
             "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
    (work / "RUN_PROGRESS.json").write_text(json.dumps(value))
    return work


def test_missing_bundle_waits_and_never_claims_fit(tmp_path):
    value = monitor.collect_status(tmp_path / "missing")
    assert value["sdk_files"] == 0 and value["training_entry_exit_code"] is None
    assert "等待启动器" in value["stage"]
    assert "入口状态不等于科学完成" in monitor.format_status(value)


def test_sdk_counts_only_regular_completed_files(tmp_path):
    work = put_state(tmp_path)
    sdk = work / "sdk_site"
    (sdk / "package").mkdir(parents=True)
    (sdk / "package" / "module.py").write_text("fixture")
    (sdk / ".paper3-one.partial").write_text("not completed")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("ignore outside")
    (sdk / "link-dir").symlink_to(outside, target_is_directory=True)
    (sdk / "link-file").symlink_to(outside / "file")
    value = monitor.collect_status(tmp_path)
    assert value["sdk_files"] == 1 and value["sdk_expected_files"] == 2208
    assert "凭证提示前" in value["stage"]


@pytest.mark.parametrize("where", ["bundle", "work", "sdk", "state", "release", "project"])
def test_symlink_roots_and_state_are_refused(tmp_path, where):
    bundle = tmp_path / "bundle"
    work = put_state(bundle)
    external = tmp_path / "elsewhere"
    external.mkdir()
    if where == "bundle":
        link = tmp_path / "bundle-link"
        link.symlink_to(bundle, target_is_directory=True)
        bundle = link
    elif where == "work":
        (work / "RUN_PROGRESS.json").unlink()
        work.rmdir()
        work.symlink_to(external, target_is_directory=True)
    elif where == "sdk":
        (work / "sdk_site").symlink_to(external, target_is_directory=True)
    elif where == "release":
        (work / "release").symlink_to(external, target_is_directory=True)
    elif where == "project":
        (work / "release").mkdir()
        (work / "release" / monitor.PROJECT_NAME).symlink_to(external, target_is_directory=True)
    else:
        state = work / "RUN_PROGRESS.json"
        state.unlink()
        (external / "state.json").write_text("{}")
        state.symlink_to(external / "state.json")
    with pytest.raises(monitor.StatusError, match="STATUS_SYMLINK_FORBIDDEN"):
        monitor.collect_status(bundle)


def test_unknown_and_secret_fields_never_reach_status(tmp_path, monkeypatch):
    secret = "fixture-secret-not-for-display"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    put_state(tmp_path, github={"status": secret, "commit": secret, "token": secret},
              credentials={"secret": secret}, entry_issued_at_utc=secret,
              training={"actual_exit_code": "0 " + secret, "reason": secret},
              r2_objects={"unknown-secret": secret}, note=secret)
    value = monitor.collect_status(tmp_path)
    assert secret not in json.dumps(value)
    assert secret not in monitor.format_status(value)
    assert value["training_entry_exit_code"] is None


def test_network_and_local_cache_are_distinct(tmp_path):
    put_state(tmp_path, github={"status": "GITHUB_CODE_PUSH_AND_REMOTE_SHA_VERIFIED", "commit": "a" * 40},
              r2_objects={"ready": {"status": "R2_FULL_GET_SHA_VERIFIED", "get_attempted": True,
                                     "complete_body_read": True, "http_status": 200},
                          "main_inputs": {"status": "LOCAL_VERIFIED_CACHE_HIT", "get_attempted": False},
                          "control_inputs": {"status": "R2_FULL_GET_SHA_VERIFIED", "get_attempted": False}},
              r2_write_read_probe={"status": "R2_WRITE_AND_FULL_GET_PROBE_VERIFIED"})
    value = monitor.collect_status(tmp_path)
    assert value["github_verified"] and value["github_commit_short"] == "a" * 12
    assert value["r2_network_full_GET_objects"] == 1 and value["r2_local_cache_objects"] == 1
    assert value["r2_write_read_probe_verified"]


def test_zero_exit_never_claims_scientific_completion(tmp_path):
    put_state(tmp_path, training={"actual_exit_code": 0}, result_backup={"actual_exit_code": 0})
    value = monitor.collect_status(tmp_path)
    text = monitor.format_status(value)
    assert "退出码 0" in text
    assert "入口状态不等于科学完成" in text
    assert "训练已完成" not in text and "科学训练已完成" not in text


def test_prepare_only_is_not_labeled_returned_training(tmp_path):
    put_state(tmp_path, entry_issued_at_utc="fixture-issued", preparation={"actual_exit_code": 0})
    value = monitor.collect_status(tmp_path)
    assert "原准备入口已返回" in value["stage"]
    assert value["training_entry_exit_code"] is None


def test_resumed_entry_does_not_look_like_old_backup_return(tmp_path):
    put_state(tmp_path, entry_issued_at_utc="2026-10-21T02:00:00+00:00",
              training={"actual_exit_code": 2, "returned_at_utc": "2026-10-21T01:00:00+00:00"},
              result_backup={"actual_exit_code": 0, "returned_at_utc": "2026-10-21T01:30:00+00:00"})
    value = monitor.collect_status(tmp_path)
    assert value["entry_issued_without_return"]
    assert "尚无本次返回" in value["stage"]
    assert "本次未见返回；上次退出码 2" in monitor.format_status(value)


def test_partial_json_can_be_retried_without_exposing_body(tmp_path):
    work = put_state(tmp_path)
    (work / "RUN_PROGRESS.json").write_text('{"token":"fixture-secret"')
    value = monitor.collect_status(tmp_path)
    assert value["state_kind"] == "temporarily_invalid"
    assert "fixture-secret" not in monitor.format_status(value)


def test_wrong_release_refused(tmp_path):
    put_state(tmp_path, closure_sha256="a" * 64)
    with pytest.raises(monitor.StatusError, match="STATUS_WRONG_PROGRESS_RELEASE"):
        monitor.collect_status(tmp_path)


def test_cli_once_is_read_only(tmp_path, capsys):
    put_state(tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert monitor.main(["--bundle", str(tmp_path), "--once"]) == 0
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert "北京时间" in capsys.readouterr().out


def test_release_counts_real_files_without_reading_payloads_or_following_links(tmp_path):
    work = put_state(tmp_path)
    project = work / "release" / monitor.PROJECT_NAME
    (project / "cache").mkdir(parents=True)
    (project / "cache" / "cache.npy").write_bytes(b"fixture-not-read-as-data")
    (project / "runs").mkdir()
    (project / "runs" / "checkpoint.pt").write_bytes(b"fixture-not-fit")
    (project / ".paper3-x.partial").write_bytes(b"uncommitted-temp")
    external = tmp_path / "unrelated"
    external.mkdir()
    (external / "do-not-count").write_bytes(b"fixture-secret")
    (project / "external-directory").symlink_to(external, target_is_directory=True)
    (project / "external-file").symlink_to(external / "do-not-count")
    before = {p: p.read_bytes() for p in project.rglob("*") if p.is_file() and not p.is_symlink()}
    value = monitor.collect_status(tmp_path)
    assert value["release_files_on_disk"] == 2
    text = monitor.format_status(value)
    assert "已落盘文件：2" in text and "不单独代表验证通过" in text
    assert "fixture-secret" not in text and "训练已完成" not in text
    after = {p: p.read_bytes() for p in project.rglob("*") if p.is_file() and not p.is_symlink()}
    assert before == after


def test_cli_adjacent_snapshots_show_release_growth_without_state_json_updates(tmp_path, monkeypatch, capsys):
    work = put_state(tmp_path)
    project = work / "release" / monitor.PROJECT_NAME
    project.mkdir(parents=True)
    (project / "first.py").write_text("fixture")
    state_before = (work / "RUN_PROGRESS.json").read_bytes()
    sleeps = []
    def fake_sleep(interval):
        sleeps.append(interval)
        if len(sleeps) == 1:
            (project / "second.py").write_text("fixture")
        else:
            raise KeyboardInterrupt
    monkeypatch.setattr(monitor.time, "sleep", fake_sleep)
    assert monitor.main(["--bundle", str(tmp_path), "--interval", "10"]) == 0
    text = capsys.readouterr().out
    assert "已落盘文件：1；首个快照" in text
    assert "已落盘文件：2；相邻快照 +1" in text
    assert (work / "RUN_PROGRESS.json").read_bytes() == state_before
    assert sleeps == [10.0, 10.0]
