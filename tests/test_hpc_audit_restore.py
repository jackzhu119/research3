"""Tiny metadata-only archive fixtures; no EEG, network, GPU or fit."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from tools import hpc_restore_audit as audit


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture_archive(tmp_path, *, rows=None, kind=None, duplicate=False):
    root = tmp_path / "project"
    root.mkdir()
    archives = tmp_path / "packages"
    archives.mkdir()
    payloads = {"audit/software_fixture/original.py.txt": b"# original SOFTWARE FIXTURE\n",
                "contracts/software_fixture.json": b'{"SOFTWARE_FIXTURE":true}\n'}
    source = archives / "fixed.tar.gz"
    with tarfile.open(source, "w:gz") as stream:
        for index, (name, raw) in enumerate(payloads.items()):
            member = tarfile.TarInfo("fixed/" + name)
            member.size = len(raw)
            if kind and index == 0:
                member.type = kind
                member.linkname = "/etc/hostname"
                member.size = 0
            stream.addfile(member, io.BytesIO(raw) if member.isfile() else None)
            if duplicate and index == 0:
                stream.addfile(member, io.BytesIO(raw))
    descriptors = [{"path": name, "sha256": sha(raw), "bytes": len(raw), "archive": "main"}
                   for name, raw in payloads.items()]
    if rows is not None:
        descriptors = rows
    manifest = {"schema_version": audit.SCHEMA, "top_level_directory": "fixed",
                "sources": {"main": {"filename": source.name, "sha256": sha(source.read_bytes()),
                                      "bytes": source.stat().st_size}},
                "files": descriptors, "file_count": len(descriptors),
                "bytes": sum(row["bytes"] for row in descriptors)}
    index = tmp_path / "inputs.json"
    index.write_text(json.dumps(manifest))
    return root, archives, index, source, payloads


def test_actual_manifest_keeps_bound_history_and_original_protocol():
    manifest = audit.load_manifest(Path(__file__).parents[1] / "hpc/audit_inputs.json")
    rows = {row["path"]: row for row in manifest["files"]}
    assert rows["audit/data_plan_20261020/dreyer_structure_case_review/v5_historical_code/dreyer2023.py"]["sha256"] == (
        "ec2f7dbc10903c308cd30564c30b04c4653c64173fdb0331ede318fd1c3b1489")
    assert rows["audit/data_plan_20261020/deployment_finalize_review/prepare_finalize_plan.py"]["sha256"] == (
        "6d40cb72f60470ed0fbd3401dbe711628d4820c72d5cb1255a222c9a2c0314fa")
    assert "audit/data_plan_20261020/classical_precision_fix/native_regression.json" in rows
    assert len([path for path in rows if path.startswith("protocol/")]) == 53
    # The original review binds 128 parent captures and 392 subsequent captures;
    # the latter were originally written as native_structure.json.
    assert rows["audit/data_plan_20261020/dreyer_structure_case_review/final_native_TAL_and_backup_review.json"]["sha256"] == (
        "2a769a20353f1f478a57fd06333c6b1745938e99b820d88e417df96891d5cc37")
    assert len([path for path in rows if path.endswith(("/native_metadata.json", "/native_structure.json"))]) == 520
    assert not any(path.startswith(("p3work/", "tests/", "tools/", "data/")) for path in rows)
    assert all(Path(path).suffix == ".json" for path in rows if path.startswith("cache/"))


def test_readonly_reports_missing_and_never_materializes(tmp_path):
    root, archives, index, _, _ = fixture_archive(tmp_path)
    report = audit.restore(root, index, archives)
    assert report["missing_files"] == 2 and report["restored_files"] == 0
    assert report["network_operations"] == report["scientific_fits"] == 0
    assert not list(root.iterdir())


def test_real_archive_SHA_and_each_selected_file_are_verified_then_resume_is_readonly(tmp_path):
    root, archives, index, _, payloads = fixture_archive(tmp_path)
    report = audit.restore(root, index, archives, execute=True)
    assert report["verified_files"] == report["restored_files"] == 2
    for name, raw in payloads.items():
        assert (root / name).read_bytes() == raw
    missing_archive_directory = tmp_path / "does-not-exist"
    before = {path: path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    again = audit.restore(root, index, missing_archive_directory, execute=True)
    assert again["restored_files"] == 0 and again["missing_files"] == 0
    assert before == {path: path.stat().st_mtime_ns for path in before}


def test_corrupt_original_archive_blocks_before_any_write(tmp_path):
    root, archives, index, source, _ = fixture_archive(tmp_path)
    with source.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(audit.RestoreError, match="original_archive_SHA_or_length_mismatch"):
        audit.restore(root, index, archives, execute=True)
    assert not list(root.iterdir())


def test_existing_modified_history_is_not_overwritten(tmp_path):
    root, archives, index, _, payloads = fixture_archive(tmp_path)
    name = next(iter(payloads))
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"different original must not be silently accepted")
    with pytest.raises(audit.RestoreError, match="existing_audit_SHA_mismatch_preserve_file"):
        audit.restore(root, index, archives, execute=True)
    assert path.read_bytes() == b"different original must not be silently accepted"


@pytest.mark.parametrize("path", ["../escape.json", "/escape.json", "audit/../escape.json",
    "audit/.env", "audit/.aws/config", "p3work/change_algorithm.py", "audit/raw.edf",
    "audit/cache.npy", "audit/source.npz"])
def test_unsafe_or_scientific_payload_cannot_be_a_history_descriptor(tmp_path, path):
    rows = [{"path": path, "sha256": "a" * 64, "bytes": 1, "archive": "main"}]
    root, archives, index, _, _ = fixture_archive(tmp_path, rows=rows)
    with pytest.raises(audit.RestoreError):
        audit.restore(root, index, archives, execute=True)
    assert not list(root.iterdir())


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_archive_link_is_never_followed(tmp_path, kind):
    root, archives, index, _, _ = fixture_archive(tmp_path, kind=kind)
    with pytest.raises(audit.RestoreError, match="exact_regular_audit_member_required"):
        audit.restore(root, index, archives, execute=True)
    assert not list(root.iterdir())


def test_wrong_member_hash_does_not_commit_or_leave_partial(tmp_path):
    root, archives, index, _, _ = fixture_archive(tmp_path)
    manifest = json.loads(index.read_bytes())
    manifest["files"][0]["sha256"] = "f" * 64
    index.write_text(json.dumps(manifest))
    with pytest.raises(audit.RestoreError, match="audit_member_SHA_or_length_mismatch"):
        audit.restore(root, index, archives, execute=True)
    assert not [path for path in root.rglob("*") if path.is_file()]


def test_symlink_root_and_destination_cannot_be_used(tmp_path):
    root, archives, index, _, _ = fixture_archive(tmp_path)
    target = tmp_path / "actual"
    target.mkdir()
    (root / "audit").symlink_to(target, target_is_directory=True)
    with pytest.raises(audit.RestoreError, match="symlink_path_forbidden"):
        audit.restore(root, index, archives, execute=True)
    assert not list(target.iterdir())
