"""Delivery regression fixtures: no real EEG, R2, training, or server actions.

Receipt JSON below is an explicit software test double, not an actual backup
claim. Protocol tests copy the already imported design documents; fake wheel
and cache payloads are clearly marked bytes and are never installed or fitted.
"""
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from tools import build_delivery as delivery
from tools.build_delivery import create_archive, safe_member, build


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _fixture_receipt(root, path, artifacts):
    """Construct a receipt TEST DOUBLE that the structural verifier can inspect."""
    prefix = "paper3/delivery-unit-fixture/"
    rows = []
    for artifact in artifacts:
        descriptor = _descriptor(root, artifact)
        nbytes = artifact.stat().st_size
        rows.append({**descriptor, "bytes": nbytes, "role": "archive" if artifact.suffix == ".whl" else "audit",
                     "dataset": "SoftwareFixtureOnly", "key": prefix + descriptor["path"], "status": "verified",
                     "put": {"status": "created", "http_status": 200, "request_body_bytes": nbytes,
                             "request_body_sha256": descriptor["sha256"]},
                     "get": {"status": "verified", "http_status": 200, "complete_body_read": True,
                             "bytes_read": nbytes, "sha256": descriptor["sha256"], "declared_content_length": nbytes}})
    _write_json(path, {"schema_version": 1, "operation": "paper3_explicit_manifest_backup",
                     "status": "verified", "execute": True, "verification": "full_get_body_sha256",
                     "prefix": prefix, "objects": rows, "scope": "SOFTWARE_TEST_DOUBLE_NOT_R2_EVIDENCE"})
    return path


def _provider_evidence_fixture(root):
    """Writer-shaped 22 descriptors; these bytes are not real provider evidence."""
    files = []
    names = ([f"adapter-copy-{i}.py" for i in range(6)] + [f"paper-copy-{i}.xml" for i in range(4)]
             + [f"provider-copy-{i}.txt" for i in range(12)])
    for name in names:
        path = root / "audit/source_schema/evidence" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("SOFTWARE_TEST_DOUBLE_NOT_PROVIDER_EVIDENCE:" + name).encode())
        files.append(path)
    manifest = root / "audit/source_schema/evidence_manifest.json"
    _write_json(manifest, {"files": [_descriptor(root, p) for p in files],
                          "scope": "SOFTWARE_TEST_DOUBLE_NOT_LICENSE_OR_PROVIDER_EVIDENCE"})
    return manifest, files


@pytest.fixture
def selection_fixture(tmp_path, monkeypatch):
    """Nothing is git tracked; original 52 protocol documents must still enter."""
    root = tmp_path / "paper3"
    root.mkdir()
    protocol = root / "protocol"
    protocol.mkdir()
    imported = delivery.ROOT / "protocol"
    manifest = (imported / "PACKAGE_CONTENTS.sha256").read_bytes()
    (protocol / "PACKAGE_CONTENTS.sha256").write_bytes(manifest)
    for line in manifest.decode().splitlines():
        _, relative = line.split(maxsplit=1)
        target = protocol / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(imported / relative, target)
    wheel_dir = root / "data/dependency_wheels/cpu"
    wheel_dir.mkdir(parents=True)
    wheels = []
    for prefix in delivery.BOOTSTRAP_NAMES:
        p = wheel_dir / (prefix + "0.0-py3-none-any.whl")
        p.write_bytes(("SOFTWARE_TEST_DOUBLE_NOT_INSTALLABLE_WHEEL:" + prefix).encode())
        wheels.append(p)
    _fixture_receipt(root, root / "audit/r2_dependencies_receipt.json", wheels)
    def no_tracked_files(args, **kwargs):
        assert args == ["git", "ls-files", "-z"]
        return b""
    monkeypatch.setattr(delivery.subprocess, "check_output", no_tracked_files)
    return root, wheels


@pytest.fixture
def cache_backup_fixture(tmp_path):
    root = tmp_path
    evidence_manifest, evidence_files = _provider_evidence_fixture(root)
    artifacts = [evidence_manifest, *evidence_files]
    def artifact(relative):
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(("SOFTWARE_TEST_DOUBLE_NOT_EEG:" + relative).encode())
        artifacts.append(p)
        return _descriptor(root, p)
    inventory = root / "manifests/raw_inventory.json"
    _write_json(inventory, {"provider_terms_evidence": _descriptor(root, evidence_manifest),
                            "scope": "SOFTWARE_TEST_DOUBLE_NOT_RAW_INVENTORY"})
    artifacts.append(inventory)
    manifest = {"scope": "SOFTWARE_FIXTURE_NOT_A_SOURCE_CACHE",
                "raw_inventory": _descriptor(root, inventory),
                "shards": [{key: artifact("cache/software_fixture/" + name)
                            for key, name in (("X", "X.npy"), ("y", "y.npy"), ("events", "events.json"),
                                              ("exclusions", "exclusions.json"))}],
                "participants_without_usable_trials": [{"exclusions": artifact("cache/software_fixture/no_usable_exclusions.json")}]
                }
    path = root / "manifests/source_cache.json"
    _write_json(path, manifest)
    artifacts.append(path)
    cache = SimpleNamespace(manifest_path=path.relative_to(root).as_posix(),
                            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    receipt = _fixture_receipt(root, root / "audit/cache_receipt.json", artifacts)
    return root, cache, artifacts, receipt


def test_secret_or_escape_paths_rejected():
    for path in ["../x", "/tmp/x", ".env", ".aws/config", "x/private.key", "x/.env.prod", "x/a.partial", "audit/raw.mat", "cache/x.npy",
                 "audit/credentials.json", "audit/secrets.json", "audit/raw.h5", "audit/private.p12",
                 "audit/raw.zip", "audit/.ENV.production", "audit/.AWS/config", "audit/model.pt"]:
        assert not safe_member(path)
    assert safe_member("protocol/METHOD_SPEC.md")


def test_deterministic_tar_integrity(tmp_path):
    entries = {"status/example.json": b'{"training_started":false}', "tools/example.py": b"pass\n"}
    a, b = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    assert create_archive(a, entries) == 3
    create_archive(b, entries)
    assert a.read_bytes() == b.read_bytes()
    with tarfile.open(a) as archive:
        assert archive.extractfile("paper3/status/example.json").read() == entries["status/example.json"]


def test_default_delivery_dry_run_has_no_stage_or_network_side_effect(tmp_path):
    result = build(tmp_path)
    assert result["status"] == "PLANNED" and not result["execute"]
    assert not list(tmp_path.iterdir())


def test_untracked_original_52_protocol_files_are_packaged(selection_fixture):
    root, _ = selection_fixture
    entries = delivery.selected_files(root)
    manifest = (root / "protocol/PACKAGE_CONTENTS.sha256").read_text().splitlines()
    assert len(manifest) == 52
    for line in manifest:
        expected, relative = line.split(maxsplit=1)
        assert hashlib.sha256(entries["protocol/" + relative]).hexdigest() == expected
    assert "protocol/PACKAGE_CONTENTS.sha256" in entries
    assert "protocol/EXPERIMENT_PROTOCOL.md" in entries


def test_accidentally_tracked_prediction_metadata_cannot_enter_execution_bundle(selection_fixture, monkeypatch):
    root, _ = selection_fixture
    target = root / "predictions/software_fixture/receipt.json"
    _write_json(target, {"scope": "SOFTWARE_TEST_DOUBLE_NOT_A_PREDICTION"})
    monkeypatch.setattr(delivery.subprocess, "check_output",
                        lambda *args, **kwargs: b"predictions/software_fixture/receipt.json\0")
    with pytest.raises(ValueError, match="Large data/weights must use explicit R2 manifests"):
        delivery.selected_files(root)


def test_changed_protocol_bytes_stop_packaging(selection_fixture):
    root, _ = selection_fixture
    with (root / "protocol/EXPERIMENT_PROTOCOL.md").open("ab") as stream:
        stream.write(b"\nSOFTWARE_TEST_CHANGE\n")
    with pytest.raises(ValueError, match="protocol bytes changed"):
        delivery.selected_files(root)


@pytest.mark.parametrize("change", ["missing_entry", "duplicate_entry"])
def test_missing_or_duplicate_protocol_manifest_entry_rejected(selection_fixture, change):
    root, _ = selection_fixture
    path = root / "protocol/PACKAGE_CONTENTS.sha256"
    lines = path.read_text().splitlines()
    if change == "missing_entry":
        lines.pop()
    else:
        lines[-1] = lines[0]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="52 original"):
        delivery.selected_files(root)


def test_bootstrap_symlink_outside_root_is_rejected(selection_fixture):
    root, wheels = selection_fixture
    external = root.parent / "outside-fixture.whl"
    external.write_bytes(b"OUTSIDE_SOFTWARE_FIXTURE_NOT_A_SECRET")
    wheels[0].unlink()
    wheels[0].symlink_to(external)
    with pytest.raises(Exception, match="symlink_forbidden"):
        delivery.selected_files(root)


def test_bootstrap_bytes_must_match_verified_dependency_receipt(selection_fixture):
    root, wheels = selection_fixture
    wheels[0].write_bytes(b"CHANGED_SOFTWARE_WHEEL_FIXTURE")
    with pytest.raises(ValueError, match="Bootstrap wheel differs"):
        delivery.selected_files(root)


def test_complete_cache_receipt_covers_every_required_artifact(cache_backup_fixture):
    root, cache, _, receipt = cache_backup_fixture
    delivery.validate_cache_backup(root, cache, receipt)


def test_exact_manifest_authorizes_provider_python_XML_copies_and_packages_all_22(selection_fixture):
    root, _ = selection_fixture
    evidence_manifest, evidence_files = _provider_evidence_fixture(root)
    unrelated = root / "audit/source_schema/evidence/not-in-manifest.py"
    unrelated.write_bytes(b"SOFTWARE_TEST_DOUBLE_UNAUTHORIZED_COPY")
    entries = delivery.selected_files(root)
    assert len(evidence_files) == 22
    assert any(p.suffix == ".py" for p in evidence_files) and any(p.suffix == ".xml" for p in evidence_files)
    for path in [evidence_manifest, *evidence_files]:
        assert entries[path.relative_to(root).as_posix()] == path.read_bytes()
    assert unrelated.relative_to(root).as_posix() not in entries


@pytest.mark.parametrize("suffix", [".py", ".xml"])
def test_unlisted_provider_copy_cannot_enter_cache_restore_receipt(cache_backup_fixture, suffix):
    root, cache, artifacts, receipt = cache_backup_fixture
    path = root / ("audit/source_schema/evidence/unlisted-copy" + suffix)
    path.write_bytes(b"SOFTWARE_TEST_DOUBLE_UNAUTHORIZED_EVIDENCE_COPY")
    _fixture_receipt(root, receipt, [*artifacts, path])
    with pytest.raises(ValueError, match="explicit metadata"):
        delivery.validate_cache_backup(root, cache, receipt)


@pytest.mark.parametrize("suffix", [".py", ".xml"])
def test_changed_evidence_SHA_rejected_even_with_resealed_R2_receipt(cache_backup_fixture, suffix):
    root, cache, artifacts, receipt = cache_backup_fixture
    path = next(p for p in artifacts if p.suffix == suffix)
    path.write_bytes(b"SOFTWARE_TEST_DOUBLE_CHANGED_BYTES")
    _fixture_receipt(root, receipt, artifacts)
    with pytest.raises(ValueError, match="full SHA descriptor"):
        delivery.validate_cache_backup(root, cache, receipt)


def test_packaging_rechecks_all_provider_evidence_bytes(selection_fixture):
    root, _ = selection_fixture
    _, files = _provider_evidence_fixture(root)
    files[0].write_bytes(b"SOFTWARE_TEST_DOUBLE_CHANGED_AFTER_MANIFEST")
    with pytest.raises(ValueError, match="full SHA descriptor"):
        delivery.selected_files(root)


def test_cache_receipt_must_cover_exact_provider_evidence(cache_backup_fixture):
    root, cache, artifacts, receipt = cache_backup_fixture
    missing = next(p for p in artifacts if p.suffix == ".py")
    _fixture_receipt(root, receipt, [p for p in artifacts if p != missing])
    with pytest.raises(ValueError, match="does not cover"):
        delivery.validate_cache_backup(root, cache, receipt)


@pytest.mark.parametrize("missing", ["manifest", "raw_inventory", "X", "events", "exclusions", "no_usable_exclusions"])
def test_cache_receipt_missing_any_required_artifact_cannot_be_ready(cache_backup_fixture, missing):
    root, cache, artifacts, receipt = cache_backup_fixture
    match = {"manifest": "manifests/source_cache.json", "raw_inventory": "manifests/raw_inventory.json",
             "X": "cache/software_fixture/X.npy", "events": "cache/software_fixture/events.json",
             "exclusions": "cache/software_fixture/exclusions.json",
             "no_usable_exclusions": "cache/software_fixture/no_usable_exclusions.json"}[missing]
    _fixture_receipt(root, receipt, [p for p in artifacts if p.relative_to(root).as_posix() != match])
    with pytest.raises(ValueError, match="does not cover"):
        delivery.validate_cache_backup(root, cache, receipt)


def test_unrelated_verified_cache_receipt_never_reaches_packaging(cache_backup_fixture, monkeypatch):
    root, cache, _, _ = cache_backup_fixture
    unrelated = root / "manifests/unrelated.json"
    _write_json(unrelated, {"scope": "SOFTWARE_FIXTURE_UNRELATED_TO_SOURCE_CACHE"})
    receipt = _fixture_receipt(root, root / "audit/unrelated_receipt.json", [unrelated])
    for name in ("r2_original_archive_receipt.json", "r2_schema_raw_receipt.json", "r2_dependencies_receipt.json"):
        _fixture_receipt(root, root / "audit" / name, [unrelated])
    _write_json(root / "status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {
        "status": "COMPLETE", "cache_r2_receipt": receipt.relative_to(root).as_posix()}})
    import tools.run_controlled_pilot as pilot
    # Stub only the earlier source-input gate so this test isolates receipt binding.
    # It does NOT claim the fake byte artifacts are valid EEG or W2 products.
    monkeypatch.setattr(pilot, "verify_inputs", lambda _root: cache)
    monkeypatch.setattr(delivery, "validate_dependency_backups", lambda _root: None)
    def must_not_package(*args, **kwargs):
        pytest.fail("An unrelated verified receipt reached package selection or R2")
    monkeypatch.setattr(delivery, "selected_files", must_not_package)
    monkeypatch.setattr(delivery, "run_backup", must_not_package)
    with pytest.raises(ValueError, match="does not cover"):
        delivery.build(root, execute=True)
    assert not (root / "dist").exists()


def test_cache_receipt_requires_complete_get_proof(cache_backup_fixture):
    root, cache, _, receipt = cache_backup_fixture
    value = json.loads(receipt.read_text())
    value["objects"][0]["get"]["complete_body_read"] = False
    _write_json(receipt, value)
    with pytest.raises(Exception, match="backup_object_not_fully_verified"):
        delivery.validate_cache_backup(root, cache, receipt)


@pytest.mark.parametrize("suffix", [".h5", ".dat", ".bin", ".npy"])
def test_complete_cache_receipt_rejects_unrelated_binary_payloads(cache_backup_fixture, suffix):
    root, cache, artifacts, receipt = cache_backup_fixture
    extra = root / ("data/unrelated_raw_fixture" + suffix)
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"SOFTWARE_FIXTURE_NOT_RAW_EEG")
    _fixture_receipt(root, receipt, artifacts + [extra])
    with pytest.raises(ValueError):
        delivery.validate_cache_backup(root, cache, receipt)


def test_startup_restores_both_dependency_receipts_and_cache_before_readiness():
    receipt = "audit/source_full/cache_backup_receipt.json"
    text = delivery.startup_text(checkpoint=False, cache_receipt=receipt)
    bootstrap = text.index("--find-links bootstrap_wheels")
    dependencies = text.index("--receipt audit/r2_dependencies_receipt.json")
    extras = text.index("--receipt audit/r2_dependency_extras_receipt.json")
    install_gpu = text.index("-r requirements-gpu.lock")
    source_cache = text.index("--receipt " + receipt)
    gate = text.index("python tools/check_pilot_readiness.py --require-ready")
    assert bootstrap < dependencies < extras < install_gpu < source_cache < gate
    assert text.index("--execute --user-started-gpu") > gate
    assert "GPU compatibility and timings remain unmeasured" in text


@pytest.mark.parametrize("checkpoint", [False, True])
def test_startup_result_backup_remains_reachable_after_failed_pilot(tmp_path, checkpoint):
    """Execute shell test doubles only: the failed pilot must not skip backup."""
    text = delivery.startup_text(checkpoint=checkpoint, cache_receipt="audit/cache_receipt.json")
    blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)
    pilot = next(block for block in blocks if block.startswith("python tools/run_controlled_pilot.py "))
    backup = next(block for block in blocks if block.startswith("python tools/r2_backup.py "))
    pilot_args, backup_args = shlex.split(pilot), shlex.split(backup)
    output = Path(pilot_args[pilot_args.index("--output") + 1])
    assert output.as_posix() == "runs/controlled_source_pilot_seed20261018"
    assert "--execute" in pilot_args and "--user-started-gpu" in pilot_args
    assert Path(backup_args[backup_args.index("--manifest") + 1]) == output / "engineering_backup_manifest.json"
    assert backup_args[backup_args.index("--root") + 1] == "."
    assert backup_args[backup_args.index("--prefix") + 1] == "paper3/source-engineering-pilot/seed20261018/"
    assert backup_args[backup_args.index("--receipt") + 1] == "audit/r2_source_engineering_pilot_receipt.json"
    assert "--execute" in backup_args
    assert "&&" not in pilot + backup
    assert "runs/controlled_pilot_job_claim.json" in text and "one controlled pilot job" in text
    assert "total200 updates" in text and "No automatic GPU command is executed" in text
    assert "even when the pilot exits nonzero" in text
    assert "A failed pilot keeps its failure status" in text
    if checkpoint:
        assert "CHECKPOINT: W1/W2 pending; no GPU pilot is permitted." in text

    log = tmp_path / "invocations.txt"
    python_double = tmp_path / "python-test-double"
    python_double.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$1\" >> {shlex.quote(str(log))}\n"
        "if [ \"$1\" = tools/run_controlled_pilot.py ]; then exit 3; fi\n"
        "exit 0\n"
    )
    python_double.chmod(0o700)
    command = "\n".join(
        shlex.quote(str(python_double)) + block[len("python"):]
        for block in (pilot, backup)
    )
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert log.read_text().splitlines() == ["tools/run_controlled_pilot.py", "tools/r2_backup.py"]
