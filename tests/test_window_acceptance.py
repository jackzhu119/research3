"""SOFTWARE_TEST_DOUBLE metadata only; no EEG, HTTP, training or real W1 claim."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from p3work.contracts import CHANNELS, file_digest
from p3work.splits import _hash
from p3work.window import choose_window
from p3work.window_acceptance import WindowAcceptanceError, audit_duration_batch, exact_window_counts, verify_frozen_window
from tools import stieger_metadata_batch as base
from tools import verify_window_gate as cli


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def row(person, session, source, sha, index, label, seconds=3):
    return {"SOFTWARE_TEST_DOUBLE": True, "dataset": "Stieger2021", "dataset_version": 1,
        "participant_id": person, "session_id": str(session), "source_record_id": source, "source_file_sha256": sha,
        "tasknumber": 1, "label": label, "trial_uid": f"software:{person}:{session}:{index}",
        "feedback_end_verified": True, "available_feedback_seconds": float(seconds),
        "analysis_anchor_sample": 4000, "feedback_stop_exclusive_sample": 4000 + seconds * 1000,
        "resultind_matlab_one_based_inclusive": 4000 + seconds * 1000,
        "triallength_reported_seconds": float(seconds), "native_trial_samples_including_padding": 11000}


def math_fixture(retained):
    people = [f"S{i}" for i in range(1, 41)]
    split = {"status": "FROZEN_METADATA_ONLY", "partitions": {"meta_train": people}, "split_sha256": "a" * 64}
    rows, sessions = [], []
    for i, person in enumerate(people):
        sha, source = digest(person), f"{person}_Session_1.mat"
        for label in (0, 1):
            rows += [row(person, 1, source, sha, label * 100 + j, label,
                3 if j < retained[i][label] else 2) for j in range(100)]
        sessions.append({"participant_id": person, "session_id": "1", "source_record_id": source,
            "source_file_sha256": sha, "original_lr_trials": 200,
            "trial_metadata_sha256": "b" * 64, "adapter_audit_sha256": "c" * 64})
    manifest = {"split_sha256": split["split_sha256"], "selection": "all_official_sessions_for_frozen_meta_train",
        "inventory_sha256": "d" * 64, "sessions": sessions}
    manifest["manifest_sha256"] = _hash(manifest)
    return rows, split, manifest


@pytest.mark.parametrize("counts,retention,gap", [
    ([(83, 93)] * 40, Fraction(22, 25), Fraction(1, 10)),
    ([(61, 61)] * 13 + [(84, 84)] * 13 + [(95, 95)] * 13 + [(80, 80)], Fraction(4, 5), Fraction(0)),
])
def test_fixed_window_agrees_with_independent_exact_counter_at_previous_float_failures(counts, retention, gap):
    rows, split, manifest = math_fixture(counts)
    result = exact_window_counts(rows, split, expected_session_manifest=manifest)
    assert result["floating_selected"] == "long"
    assert result["exact_selected"] == "long" and result["selection_agrees"] is True
    for field, value in (("participant_mean_retention", retention),
                         ("participant_mean_absolute_class_exclusion_gap", gap)):
        assert result["candidates"]["3.0"][field] == {"numerator": value.numerator, "denominator": value.denominator}
        assert result["floating_coverage"]["candidates"]["3.0"][field] == float(value)
    # Keep actual persisted JSON numbers compatible with the frozen contract
    # validator, including its inclusive float threshold comparisons.
    roundtrip = json.loads(json.dumps(result["floating_coverage"], allow_nan=False))
    summary = roundtrip["candidates"]["3.0"]
    assert type(summary["participant_mean_retention"]) is float
    assert type(summary["participant_mean_absolute_class_exclusion_gap"]) is float
    assert summary["participant_mean_retention"] >= .8
    assert summary["participant_mean_absolute_class_exclusion_gap"] <= .1


@pytest.mark.parametrize("counts,expected", [
    ([(80, 80)] * 40, "long"),
    ([(79, 79)] + [(80, 80)] * 39, "short"),
    ([(81, 81)] + [(80, 80)] * 39, "long"),
    ([(83, 93)] * 40, "long"),
    ([(82, 94)] + [(83, 93)] * 39, "short"),
    ([(84, 92)] + [(83, 93)] * 39, "long"),
])
def test_window_inclusive_exact_threshold_and_adjacent_integer_counts(counts, expected):
    rows, split, manifest = math_fixture(counts)
    selected = choose_window(rows, split, expected_session_manifest=manifest)
    independent = exact_window_counts(rows, split, expected_session_manifest=manifest)
    assert selected["selected"] == independent["exact_selected"] == expected
    assert independent["selection_agrees"]


def test_existing_per_trial_duration_tolerance_and_short_failure_are_unchanged():
    rows, split, manifest = math_fixture([(100, 100)] * 40)
    # This is deliberately a choose_window-only input: no native re-interpretation
    # of provider durations is introduced by the count arithmetic correction.
    for r in rows:
        r["available_feedback_seconds"] = 3 - 5e-10
    assert choose_window(rows, split, expected_session_manifest=manifest)["selected"] == "long"
    for r in rows:
        r["available_feedback_seconds"] = 3 - 2e-9
    assert choose_window(rows, split, expected_session_manifest=manifest)["selected"] == "short"
    for r in rows:
        r["available_feedback_seconds"] = 2 - 2e-9
    result = choose_window(rows, split, expected_session_manifest=manifest)
    assert result["selected"] is None and result["status"] == "BLOCKED_SHORT_WINDOW_COVERAGE"


def test_equal_person_weights_pool_all_sessions_before_class_fractions():
    rows, split, manifest = math_fixture([(80, 80)] * 40)
    # Duplicate only one person's trials under a second registered session.
    extra = [dict(r, session_id="2", trial_uid=r["trial_uid"] + ":extra") for r in rows if r["participant_id"] == "S1"]
    manifest["sessions"].append(dict(manifest["sessions"][0], session_id="2"))
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    manifest["manifest_sha256"] = _hash(body)
    result = exact_window_counts(rows + extra, split, expected_session_manifest=manifest)
    assert result["exact_selected"] == "long" and result["selection_agrees"]
    assert result["candidates"]["3.0"]["participant_mean_retention"] == {"numerator": 4, "denominator": 5}
    assert result["candidates"]["3.0"]["participants"]["S1"]["original_lr_trials"] == 400


@pytest.mark.parametrize("mutation", ["partial_session", "duplicate_uid", "unverified_endpoint", "noninteger_stop", "padded_endpoint", "fabricated_available"])
def test_exact_counter_rejects_incomplete_or_unverified_native_proof(mutation):
    rows, split, manifest = math_fixture([(80, 80)] * 40)
    if mutation == "partial_session":
        rows.pop()
    elif mutation == "duplicate_uid":
        rows[1]["trial_uid"] = rows[0]["trial_uid"]
    elif mutation == "unverified_endpoint":
        rows[0]["feedback_end_verified"] = False
    elif mutation == "noninteger_stop":
        rows[0]["feedback_stop_exclusive_sample"] = 7000.0
    elif mutation == "padded_endpoint":
        rows[0]["feedback_stop_exclusive_sample"] = 11000
    else:
        rows[0]["available_feedback_seconds"] = 3.1
    with pytest.raises(WindowAcceptanceError):
        exact_window_counts(rows, split, expected_session_manifest=manifest)


def build_complete_software_batch(root):
    """598 official-shaped object identities, 396 metadata-only meta-train ones.

    Provider identity, receipts and SHA below are explicit software doubles.
    No .mat file is created, opened, downloaded, or represented as real evidence.
    """
    files = []
    for person in range(1, 63):
        sessions = 10 if person <= 36 or person >= 59 else 9
        for session in range(1, sessions + 1):
            ident = len(files) + 1
            files.append({"id": ident, "name": f"S{person}_Session_{session}.mat", "provider_subject": f"S{person}",
                "session_ordinal": session, "download_url": f"https://ndownloader.figshare.com/files/{ident}",
                "size": 123, "computed_md5": hashlib.md5(f"software:{ident}".encode()).hexdigest()})
    inventory_path = root / "data_catalog/stieger2021/official_inventory.json"
    write(inventory_path, {"SOFTWARE_TEST_DOUBLE": True, "article_id": 13123148, "version": 1,
        "license": {"name": "CC BY 4.0"}, "files": files})
    qualification = base.build_plan(inventory_path)
    write(root / "plans/stieger_qualification.json", qualification)
    split = {"schema_version": "1", "dataset": "Stieger2021", "status": "FROZEN_METADATA_ONLY",
        "counts": {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11},
        "partitions": {"meta_train": [f"S{i}" for i in range(1, 41)], "meta_validation": [f"S{i}" for i in range(41, 52)],
            "internal_holdout": [f"S{i}" for i in range(52, 63)]}, "no_post_window_reassignment": True}
    split["split_sha256"] = _hash(split)
    split_path = root / "audit/stieger_w1/batches" / qualification["plan_sha256"] / "participant_split.json"
    write(split_path, split)
    plan = base.build_plan(inventory_path, stage="meta-duration", split_path=split_path)
    write(root / "plans/stieger_meta_duration_selection.json", plan)
    batch = root / "audit/stieger_w1/batches" / plan["plan_sha256"]
    for entry in plan["files"]:
        obj = batch / "objects" / str(entry["id"])
        raw_sha = digest(f"SOFTWARE_TEST_DOUBLE:{entry['id']}")
        rows = [row(entry["provider_subject"], entry["session_ordinal"], entry["name"], raw_sha, c, c) for c in (0, 1)]
        write(obj / "trial_metadata.json", rows)
        write(obj / "adapter_audit.json", {"SOFTWARE_TEST_DOUBLE": True, "dataset": "Stieger2021", "source_version": 1, "role": "development",
            "participant_id": entry["provider_subject"], "session_id": str(entry["session_ordinal"]),
            "source_file_sha256": raw_sha, "source_file_bytes": entry["size"], "native_sfreq": 1000.0,
            "endpoint_errors": 0, "pure_lr_trials": 2, "pure_lr_label_counts": {"0": 1, "1": 1}})
        descriptor = lambda p: {"path": p.relative_to(root).as_posix(), "sha256": file_digest(p),
            "bytes": p.stat().st_size, "role": "audit", "dataset": "Stieger2021"}
        raw = {"path": f"data/stieger2021/w1-batch-temp/{entry['id']}/{entry['name']}", "sha256": raw_sha,
            "bytes": entry["size"], "role": "source_development_raw", "dataset": "Stieger2021"}
        artifacts = [raw, descriptor(obj / "adapter_audit.json"), descriptor(obj / "trial_metadata.json")]
        write(obj / "backup_manifest.json", {"schema_version": 1, "files": artifacts})
        write(obj / "backup_receipt.json", {"SOFTWARE_TEST_DOUBLE": True, "status": "verified", "execute": True,
            "objects": [{**a, "status": "verified", "get": {"status": "verified", "complete_body_read": True,
                "http_status": 200, "declared_content_length": a["bytes"],
                "bytes_read": a["bytes"], "sha256": a["sha256"]}} for a in artifacts]})
        write(obj / "state.json", {"file_id": entry["id"], "name": entry["name"], "plan_sha256": plan["plan_sha256"],
            "status": "COMPLETE", "schema_blocked": False,
            "raw_path": raw["path"],
            "raw_integrity": {"bytes": entry["size"], "sha256": raw_sha, "md5": entry["computed_md5"]},
            "backup_receipt": (obj / "backup_receipt.json").relative_to(root).as_posix(),
            "retained_metadata_sha256": {(obj / name).relative_to(root).as_posix(): file_digest(obj / name)
                for name in ("adapter_audit.json", "trial_metadata.json", "backup_manifest.json", "backup_receipt.json")}})
    base.finish_durations(plan, root, batch, split_path)
    coverage = json.loads((batch / "window_coverage.json").read_text())
    manifest = json.loads((batch / "window_input_session_manifest.json").read_text())
    receipt = {"decision_pool_role": "meta_train", "split_sha256": split["split_sha256"],
        "raw_metadata_manifest_sha256": manifest["manifest_sha256"], "base_selection_plan_sha256": plan["plan_sha256"]}
    for name in ("window_coverage", "window_input_session_manifest"):
        p = batch / f"{name}.json"
        receipt[name] = {"path": p.relative_to(root).as_posix(), "sha256": file_digest(p)}
    for profile, key in (("long", "3.0"), ("short", "2.0")):
        c = coverage["candidates"][key]
        receipt[profile] = {"participant_mean_coverage": c["participant_mean_retention"],
            "participant_mean_abs_class_exclusion_gap": c["participant_mean_absolute_class_exclusion_gap"]}
    write(root / "contracts/input_contract.json", {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
        "selected_profile": "long", "context_s": [0, 3], "crop_s": [.5, 2.5], "n_times": 256, "window_audit_receipt": receipt})
    write(root / "status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {"status": "PENDING"}})
    return plan, batch, split_path


@pytest.fixture(scope="module")
def complete_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("SOFTWARE_TEST_DOUBLE_complete_metadata")
    build_complete_software_batch(root)
    return root


@pytest.fixture
def complete_batch(complete_template, tmp_path):
    shutil.copytree(complete_template, tmp_path, dirs_exist_ok=True)
    plan = json.loads((tmp_path / "plans/stieger_meta_duration_selection.json").read_text())
    batch = tmp_path / "audit/stieger_w1/batches" / plan["plan_sha256"]
    return tmp_path, plan, batch


def test_complete_actual_writer_shape_accepts_read_only_without_raw_or_other_sessions(complete_batch, monkeypatch):
    root, plan, batch = complete_batch
    original_open = Path.open
    def constrained_open(path, *args, **kwargs):
        assert path.suffix.lower() not in {".mat", ".edf"}, "guard must never read raw EEG"
        assert not any(f"S{i}_Session" in str(path) for i in range(41, 63)), "other partition session read"
        if path.exists():
            assert all(letter not in (args[0] if args else kwargs.get("mode", "r")) for letter in "wax+")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", constrained_open)
    monkeypatch.setattr(base.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network forbidden"))
    result = verify_frozen_window(root)
    assert result["status"] == "EXACT_FROZEN_WINDOW_VERIFIED" and result["cache_authorized"]
    assert result["complete_objects"] == 396 and result["participants"] == 40
    sha = result.pop("artifact_sha256")
    assert sha == _hash(result)
    assert result["exact_selected"] == result["floating_selected"] == "long"


def test_partial_metadata_never_calls_either_window_selector(complete_batch, monkeypatch):
    root, plan, batch = complete_batch
    (batch / "objects" / str(plan["files"][-1]["id"]) / "state.json").unlink()
    monkeypatch.setattr("p3work.window_acceptance.choose_window", lambda *a, **k: pytest.fail("partial selection forbidden"))
    with pytest.raises(WindowAcceptanceError, match="metadata_incomplete") as error:
        audit_duration_batch(root)
    assert error.value.details == {"complete_objects": 395, "required_objects": 396}


@pytest.mark.parametrize("corruption", ["metadata_sha", "raw_get", "state_identity", "state_allowlist", "audit_count", "manifest", "coverage", "contract_summary", "contract_descriptor", "split"])
def test_complete_metadata_or_frozen_reference_corruption_blocks(complete_batch, corruption):
    root, plan, batch = complete_batch
    obj = batch / "objects" / str(plan["files"][0]["id"])
    path = obj / "state.json"
    state = json.loads(path.read_text())
    if corruption == "metadata_sha":
        (obj / "trial_metadata.json").write_text("[]")
    elif corruption == "raw_get":
        receipt = json.loads((obj / "backup_receipt.json").read_text())
        receipt["objects"][0]["get"]["complete_body_read"] = False
        write(obj / "backup_receipt.json", receipt)
        state["retained_metadata_sha256"][(obj / "backup_receipt.json").relative_to(root).as_posix()] = file_digest(obj / "backup_receipt.json")
        write(path, state)
    elif corruption == "state_identity":
        state["file_id"] += 1; write(path, state)
    elif corruption == "state_allowlist":
        state["retained_metadata_sha256"] = {"data/must-not-open.mat": "a" * 64}; write(path, state)
    elif corruption == "audit_count":
        audit = json.loads((obj / "adapter_audit.json").read_text()); audit["pure_lr_trials"] = 3
        write(obj / "adapter_audit.json", audit)
        state["retained_metadata_sha256"][(obj / "adapter_audit.json").relative_to(root).as_posix()] = file_digest(obj / "adapter_audit.json")
        write(path, state)
    elif corruption in ("manifest", "coverage"):
        path = batch / ("window_input_session_manifest.json" if corruption == "manifest" else "window_coverage.json")
        content = json.loads(path.read_text()); content["unexpected"] = True; write(path, content)
    elif corruption.startswith("contract"):
        path = root / "contracts/input_contract.json"; content = json.loads(path.read_text())
        if corruption == "contract_summary":
            content["window_audit_receipt"]["long"]["participant_mean_coverage"] = .9
        else:
            content["window_audit_receipt"]["window_coverage"]["sha256"] = "f" * 64
        write(path, content)
    else:
        qualification = json.loads((root / "plans/stieger_qualification.json").read_text())
        split_path = root / "audit/stieger_w1/batches" / qualification["plan_sha256"] / "participant_split.json"
        split = json.loads(split_path.read_text()); split["partitions"]["meta_train"][0] = "S41"; write(split_path, split)
    with pytest.raises(WindowAcceptanceError):
        verify_frozen_window(root)


def test_complete_audit_without_w1_or_published_decision_never_authorizes_cache(complete_batch):
    root, plan, batch = complete_batch
    write(root / "status/tasks.json", {"W1": {"status": "IN_PROGRESS"}})
    (batch / "window_coverage.json").unlink()
    (batch / "window_input_session_manifest.json").unlink()
    result = audit_duration_batch(root)
    assert result["exact_selected"] == "long" and result["cache_authorized"] is False
    assert result["window_coverage"] is None and result["window_input_session_manifest"] is None
    with pytest.raises(WindowAcceptanceError, match="w1_complete"):
        verify_frozen_window(root)


def test_formal_gate_requires_published_coverage_even_when_counts_agree(complete_batch):
    root, plan, batch = complete_batch
    (batch / "window_coverage.json").unlink()
    with pytest.raises(WindowAcceptanceError, match="full_evidence_identity"):
        verify_frozen_window(root)


@pytest.mark.parametrize("corruption", ["missing_metadata", "role", "dataset", "path", "sha256", "bytes", "get_failed", "get_incomplete", "get_length", "manifest_extra"])
def test_all_three_backup_objects_need_exact_manifest_and_complete_get(complete_batch, corruption):
    root, plan, batch = complete_batch
    obj = batch / "objects" / str(plan["files"][0]["id"])
    state_path = obj / "state.json"
    state = json.loads(state_path.read_text())
    path = obj / ("backup_manifest.json" if corruption == "manifest_extra" else "backup_receipt.json")
    value = json.loads(path.read_text())
    if corruption == "manifest_extra":
        value["files"].append(dict(value["files"][-1]))
    else:
        item = value["objects"][-1]
        if corruption == "missing_metadata":
            value["objects"].pop()
        elif corruption in {"role", "dataset", "path", "sha256", "bytes"}:
            item[corruption] = {"role": "result", "dataset": "HSStroke2026_v6", "path": "data/must-not-read.mat",
                "sha256": "f" * 64, "bytes": item["bytes"] + 1}[corruption]
        elif corruption == "get_failed":
            item["get"]["status"] = "failed"
        elif corruption == "get_incomplete":
            item["get"]["complete_body_read"] = False
        else:
            item["get"]["declared_content_length"] += 1
    write(path, value)
    # Reproduce the vulnerability: even updating the trusted local receipt SHA
    # cannot make incomplete metadata backup evidence acceptable.
    state["retained_metadata_sha256"][path.relative_to(root).as_posix()] = file_digest(path)
    write(state_path, state)
    with pytest.raises(WindowAcceptanceError, match="backup"):
        verify_frozen_window(root)


def make_qualification_reuse(root, plan, batch):
    """Mirror actual reuse_qualification_object's copy/identity semantics."""
    qualification = json.loads((root / "plans/stieger_qualification.json").read_text())
    entry = plan["files"][0]
    obj = batch / "objects" / str(entry["id"])
    prior = root / "audit/stieger_w1/batches" / qualification["plan_sha256"] / "objects" / str(entry["id"])
    shutil.copytree(obj, prior)
    current_prefix, prior_prefix = obj.relative_to(root).as_posix(), prior.relative_to(root).as_posix()
    for name, field in (("backup_manifest.json", "files"), ("backup_receipt.json", "objects")):
        value = json.loads((prior / name).read_text())
        for item in value[field]:
            if item["role"] == "audit":
                item["path"] = item["path"].replace(current_prefix, prior_prefix)
        write(prior / name, value)
        shutil.copyfile(prior / name, obj / name)
    state = json.loads((obj / "state.json").read_text())
    prior_state = deepcopy(state)
    prior_state.update(plan_sha256=qualification["plan_sha256"],
        backup_receipt=(prior / "backup_receipt.json").relative_to(root).as_posix(),
        retained_metadata_sha256={(prior / name).relative_to(root).as_posix(): file_digest(prior / name)
            for name in ("adapter_audit.json", "trial_metadata.json", "backup_manifest.json", "backup_receipt.json")})
    write(prior / "state.json", prior_state)
    state.update(reused_qualification_object=True, reused_qualification_state=(prior / "state.json").relative_to(root).as_posix(),
        retained_metadata_sha256={(obj / name).relative_to(root).as_posix(): file_digest(obj / name)
            for name in ("adapter_audit.json", "trial_metadata.json", "backup_manifest.json", "backup_receipt.json")})
    write(obj / "state.json", state)
    return obj, prior


def test_legitimate_qualification_copy_verifies_original_three_object_receipt(complete_batch):
    root, plan, batch = complete_batch
    make_qualification_reuse(root, plan, batch)
    result = verify_frozen_window(root)
    assert result["status"] == "EXACT_FROZEN_WINDOW_VERIFIED" and result["cache_authorized"]
    assert result["complete_objects"] == 396


@pytest.mark.parametrize("corruption", ["foreign_path", "prior_identity", "prior_plan", "prior_raw", "copy_sha", "prior_receipt_metadata_get"])
def test_reused_qualification_requires_exact_original_identity_and_matching_copy(complete_batch, corruption):
    root, plan, batch = complete_batch
    obj, prior = make_qualification_reuse(root, plan, batch)
    if corruption == "foreign_path":
        path = obj / "state.json"; state = json.loads(path.read_text())
        state["reused_qualification_state"] = "audit/must-not-read-other-partition/state.json"
        write(path, state)
    elif corruption in {"prior_identity", "prior_plan", "prior_raw"}:
        path = prior / "state.json"; state = json.loads(path.read_text())
        if corruption == "prior_identity":
            state["file_id"] += 1
        elif corruption == "prior_plan":
            state["plan_sha256"] = "f" * 64
        else:
            state["raw_integrity"]["sha256"] = "f" * 64
        write(path, state)
    elif corruption == "copy_sha":
        # Both are valid JSON but not byte-identical copies of the archived
        # provider-native audit. The current local hash cannot replace lineage.
        path = obj / "adapter_audit.json"; value = json.loads(path.read_text())
        value["copy_extra"] = True; write(path, value)
        state = json.loads((obj / "state.json").read_text())
        state["retained_metadata_sha256"][path.relative_to(root).as_posix()] = file_digest(path)
        write(obj / "state.json", state)
    else:
        path = prior / "backup_receipt.json"; value = json.loads(path.read_text())
        value["objects"][-1]["get"]["complete_body_read"] = False; write(path, value)
        for directory in (obj, prior):
            if directory == obj:
                shutil.copyfile(path, obj / path.name)
            state_path = directory / "state.json"; state = json.loads(state_path.read_text())
            state["retained_metadata_sha256"][(directory / path.name).relative_to(root).as_posix()] = file_digest(directory / path.name)
            write(state_path, state)
    with pytest.raises(WindowAcceptanceError):
        verify_frozen_window(root)


def test_exact_mismatch_blocks_formal_even_if_floating_contract_is_valid(complete_batch, monkeypatch):
    root, plan, batch = complete_batch
    # The integer boundary counterexamples above exercise actual counting. This
    # separate software seam tests the formal guard's fail-closed comparison.
    proof = audit_duration_batch(root)
    proof["exact_selected"] = "short"; proof["selection_agrees"] = False
    monkeypatch.setattr("p3work.window_acceptance.audit_duration_batch", lambda root: proof)
    before = (root / "contracts/input_contract.json").read_bytes()
    with pytest.raises(WindowAcceptanceError, match="exact_window_selection_disagrees"):
        verify_frozen_window(root)
    assert (root / "contracts/input_contract.json").read_bytes() == before


def test_state_symlink_and_metadata_path_traversal_cannot_read_outside_root(complete_batch, tmp_path):
    root, plan, batch = complete_batch
    state = batch / "objects" / str(plan["files"][0]["id"]) / "state.json"
    state.unlink(); state.symlink_to("/must-not-read-outside-root")
    with pytest.raises(WindowAcceptanceError, match="symlink"):
        audit_duration_batch(root)
    with pytest.raises(WindowAcceptanceError, match="path_invalid"):
        audit_duration_batch(root, selection_plan="../outside.json")


def test_cli_audit_stays_nonzero_and_formal_requires_canonical_paths(complete_batch, capsys):
    root, plan, batch = complete_batch
    assert cli.main(["--root", str(root), "--audit-duration-batch"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["cache_authorized"] is False and "participants" not in result["candidates"]["3.0"]
    assert result.pop("artifact_sha256") == _hash(result)
    assert len(result["full_artifact_sha256"]) == 64
    assert cli.main(["--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "EXACT_FROZEN_WINDOW_VERIFIED"
    assert cli.main(["--root", str(root), "--selection-plan", "other.json"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "formal_gate_uses_canonical_frozen_evidence_only"


def test_standalone_and_wrapper_source_execution_block_before_any_worker(tmp_path, monkeypatch):
    from tools import prepare_all_sources as wrapper
    from tools import source_preparation_batch as source
    write(tmp_path / "status/tasks.json", {"W1": {"status": "COMPLETE"}})
    write(tmp_path / "contracts/input_contract.json", {"SOFTWARE_TEST_DOUBLE": "only ordinary contract validator mocked"})
    monkeypatch.setattr(source, "validate_signal_contract", lambda c: "a" * 64)
    # The guard itself has its own unmocked contract validation and fails before
    # any raw HEAD, progress creation or injected worker can run.
    monkeypatch.setattr(source.urllib.request, "urlopen", lambda *a, **k: pytest.fail("raw HEAD before exact gate"))
    with pytest.raises(source.BatchError, match="frozen_signal_contract_invalid"):
        source.execute_plan(source.build_plan(), root=tmp_path)
    with pytest.raises(source.BatchError, match="frozen_signal_contract_invalid"):
        wrapper.prepare_all(tmp_path, execute=True, raw_fn=lambda *a, **k: pytest.fail("raw worker before exact gate"))
    assert not (tmp_path / "audit").exists()


def test_formal_cache_gate_blocks_before_raw_or_resource_journal(tmp_path, monkeypatch):
    from tools import prepare_source_cache as cache
    write(tmp_path / "status/tasks.json", {"W1": {"status": "COMPLETE"}})
    write(tmp_path / "contracts/input_contract.json", {"SOFTWARE_TEST_DOUBLE": "ordinary validator mocked"})
    monkeypatch.setattr(cache, "validate_signal_contract", lambda c: "a" * 64)
    monkeypatch.setattr(cache, "iter_native_contexts", lambda *a, **k: pytest.fail("raw accessed before exact gate"))
    with pytest.raises(cache.SourceCacheGateError, match="frozen_signal_contract_invalid"):
        cache.prepare_cache(tmp_path, execute=True)
    assert not (tmp_path / "audit").exists() and not (tmp_path / "cache").exists()
