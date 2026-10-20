"""Temporary software fixtures; never run real preparation or data admission."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tools import build_data_plan_deploy_manifest as old
from tools import build_data_plan_deploy_manifest_v2 as new
from p3work import source_extension_io as extension


REPO = Path(__file__).resolve().parents[1]


def write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    body = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode()
    target.write_bytes(body)
    return old.descriptor(root, path)


@pytest.fixture
def software(tmp_path, monkeypatch):
    """Synthetic 189-row history and new acceptance, entirely in tmp_path."""
    root = tmp_path
    for name in (new.ORIGINAL_ENTRY, new.ENTRY, new.FINALIZE_HELPER, new.OLD_BRIDGE_SNAPSHOT):
        write(root, name, (REPO / name).read_bytes())
    bridge = write(root, new.BRIDGE, b"# synthetic new runtime, never executed\n")
    monkeypatch.setattr(new, "CURRENT_BRIDGE_SHA", bridge["sha256"])
    monkeypatch.setattr(new, "CURRENT_BRIDGE_BYTES", bridge["bytes"])
    before = new._fixed(new.BRIDGE, new.OLD_BRIDGE_SHA, new.OLD_BRIDGE_BYTES)
    rows = [write(root, f"p3work/software_fixture/item_{i:03d}.py", f"# fixture {i}\n".encode()) for i in range(188)]
    historical_inventory = write(root, new.HISTORICAL_INVENTORY,
        {"schema_version": "paper3-full-software-input-inventory-v1", "files": [before, *rows]})
    monkeypatch.setattr(new, "HISTORICAL_INVENTORY_SHA", historical_inventory["sha256"])
    monkeypatch.setattr(new, "HISTORICAL_INVENTORY_BYTES", historical_inventory["bytes"])
    historical_result = write(root, new.HISTORICAL_RESULT,
        {"status": "PASS", "actual_exit_code": 0, "passed_tests": 3350, "input_files": 189,
         "all_189_SHA_and_bytes_rechecked_after_actual_exit": True, "input_inventory": historical_inventory})
    monkeypatch.setattr(new, "HISTORICAL_RESULT_SHA", historical_result["sha256"])
    monkeypatch.setattr(new, "HISTORICAL_RESULT_BYTES", historical_result["bytes"])
    adapter = old.descriptor(root, new.ENTRY)
    helper = old.descriptor(root, new.FINALIZE_HELPER)
    inventory_path = "audit/software_fixture/new_input_inventory.json"
    tested = write(root, inventory_path, {"files": [bridge, adapter, helper, *rows]})
    acceptance = {"schema_version": new.ACCEPTANCE_SCHEMA, "status": "PASS",
        "current_bridge": bridge, "deployment_adapter": adapter, "current_finalize_helper": helper,
        "actual_exit_code": 0, "passed_tests": 3370,
        "all_input_SHA_and_bytes_rechecked_after_actual_exit": True, "input_inventory": tested,
        "scientific_fits": 0, "GPU_actions": 0, "RunPod_actions": 0, "network_operations": 0}
    acceptance_path = "audit/software_fixture/new_acceptance.json"
    acceptance_ref = write(root, acceptance_path, acceptance)
    contract = {"schema_version": new.FIX_SCHEMA, "status": "SOFTWARE_ACCEPTED_DATA_GATES_REMAIN",
        "original_builder": old.descriptor(root, new.ORIGINAL_ENTRY),
        "previous_bridge": before, "previous_bridge_snapshot": old.descriptor(root, new.OLD_BRIDGE_SNAPSHOT),
        "current_bridge": bridge, "current_deploy_adapter": adapter, "current_finalize_helper": helper,
        "software_acceptance": acceptance_ref, "historical_3350_acceptance": historical_result,
        "historical_189_input_inventory": historical_inventory,
        "historical_tests_do_not_validate_new_bridge": True, "scientific_kernel_changed": False,
        "frozen_plan_changed": False, "raw_payloads_selected": 0, "network_operations_by_adapter": 0}
    write(root, new.FIX_CONTRACT, contract)
    return {"root": root, "bridge": bridge, "old": before, "rows": rows,
            "acceptance": acceptance, "acceptance_path": acceptance_path,
            "inventory_path": inventory_path, "contract": contract}


def update_acceptance(fixture, value):
    fixture["contract"]["software_acceptance"] = write(fixture["root"], fixture["acceptance_path"], value)
    write(fixture["root"], new.FIX_CONTRACT, fixture["contract"])


@pytest.fixture
def episode_software(software, monkeypatch):
    """Synthetic software-only lineage; immutable historical source bodies."""
    root = software["root"]
    for path in (new.OLD_EPISODE_SNAPSHOT, new.OLD_PREDICTION_SNAPSHOT, new.OLD_W3_TEST_SNAPSHOT):
        write(root, path, (REPO / path).read_bytes())
    current_episode = write(root, new.EPISODE, b"# synthetic current episode consumer\n")
    current_prediction = write(root, new.PREDICTION, b"# synthetic current prediction consumer\n")
    monkeypatch.setattr(new, "CURRENT_EPISODE_SHA", current_episode["sha256"])
    monkeypatch.setattr(new, "CURRENT_EPISODE_BYTES", current_episode["bytes"])
    monkeypatch.setattr(new, "CURRENT_PREDICTION_SHA", current_prediction["sha256"])
    monkeypatch.setattr(new, "CURRENT_PREDICTION_BYTES", current_prediction["bytes"])
    old_episode = new._fixed(new.EPISODE, new.OLD_EPISODE_SHA, new.OLD_EPISODE_BYTES)
    old_prediction = new._fixed(new.PREDICTION, new.OLD_PREDICTION_SHA, new.OLD_PREDICTION_BYTES)
    old_test = new._fixed(new.LEGACY_W3_TEST, new.OLD_W3_TEST_SHA, new.OLD_W3_TEST_BYTES)
    current_test = write(root, new.LEGACY_W3_TEST, b"# synthetic historical-auditor fixture restoration\n")
    original_manifest = write(root, "audit/software_code_manifest.json",
        {"files": [old_episode, old_prediction, *software["rows"][:115]]})
    monkeypatch.setattr(new, "ORIGINAL_117_SHA", original_manifest["sha256"])
    old_inventory = write(root, new.HISTORICAL_INVENTORY,
        {"files": [software["old"], old_episode, old_prediction, old_test, *software["rows"][:185]]})
    monkeypatch.setattr(new, "HISTORICAL_INVENTORY_SHA", old_inventory["sha256"])
    monkeypatch.setattr(new, "HISTORICAL_INVENTORY_BYTES", old_inventory["bytes"])
    old_result = write(root, new.HISTORICAL_RESULT,
        {"status": "PASS", "actual_exit_code": 0, "passed_tests": 3350, "input_files": 189,
         "all_189_SHA_and_bytes_rechecked_after_actual_exit": True, "input_inventory": old_inventory})
    monkeypatch.setattr(new, "HISTORICAL_RESULT_SHA", old_result["sha256"])
    monkeypatch.setattr(new, "HISTORICAL_RESULT_BYTES", old_result["bytes"])
    descriptors = {"current_episode": current_episode, "current_prediction_store": current_prediction,
        "current_w3_auditor_legacy_tests": current_test}
    for name, path in (("current_w3_auditor", new.W3_AUDITOR), ("current_w3_auditor_tests", new.W3_AUDITOR_TESTS),
                       ("current_episode_tests", new.EPISODE_TESTS), ("current_prediction_tests", new.PREDICTION_TESTS)):
        descriptors[name] = write(root, path, ("# synthetic tested " + name + "\n").encode())
    tested = write(root, software["inventory_path"], {"files": [software["bridge"],
        old.descriptor(root, new.ENTRY), old.descriptor(root, new.FINALIZE_HELPER),
        *software["rows"], *descriptors.values()]})
    software["acceptance"].update(descriptors, input_inventory=tested)
    software["contract"].update(historical_3350_acceptance=old_result, historical_189_input_inventory=old_inventory)
    update_acceptance(software, software["acceptance"])
    contract = {"schema_version": new.EPISODE_FIX_SCHEMA, "status": "SOFTWARE_ACCEPTED_DATA_GATES_REMAIN",
        "original_builder": old.descriptor(root, new.ORIGINAL_ENTRY),
        "previous_episode": old_episode, "previous_episode_snapshot": old.descriptor(root, new.OLD_EPISODE_SNAPSHOT),
        "current_episode": current_episode, "previous_prediction_store": old_prediction,
        "previous_prediction_store_snapshot": old.descriptor(root, new.OLD_PREDICTION_SNAPSHOT),
        "current_prediction_store": current_prediction, "current_deploy_adapter": old.descriptor(root, new.ENTRY),
        "original_117_manifest": original_manifest, "historical_189_input_inventory": old_inventory,
        "historical_3350_acceptance": old_result, "historical_tests_do_not_validate_new_runtime": True,
        "scientific_kernel_changed": False, "frozen_plan_changed": False, "raw_payloads_selected": 0,
        "network_operations_by_adapter": 0, "software_acceptance": software["contract"]["software_acceptance"],
        "test_fixture_history": {"previous_test": old_test,
            "previous_test_snapshot": old.descriptor(root, new.OLD_W3_TEST_SNAPSHOT), "current_test": current_test},
        **{k: v for k, v in descriptors.items() if k not in
           {"current_episode", "current_prediction_store", "current_w3_auditor_legacy_tests"}}}
    write(root, new.EPISODE_CONTRACT, contract)
    software.update(episode_contract=contract, episode_descriptors=descriptors)
    return software


def update_episode_acceptance(fixture, value):
    update_acceptance(fixture, value)
    fixture["episode_contract"]["software_acceptance"] = fixture["contract"]["software_acceptance"]
    write(fixture["root"], new.EPISODE_CONTRACT, fixture["episode_contract"])


def test_episode_history_deferred_register_has_three_exact_aliases_and_no_walk(episode_software, monkeypatch):
    closure = new.Closure(episode_software["root"])
    monkeypatch.setattr(closure, "visit", lambda *a, **kw: pytest.fail("deferred software registration must not traverse data"))
    evidence = closure.register_episode_history()
    assert closure.files == {}
    assert len(closure.code_aliases) == 3
    assert evidence["current_implementation"] == episode_software["episode_descriptors"]["current_episode"]
    assert evidence["prediction_store_history"]["archive_snapshot"]["path"] == new.OLD_PREDICTION_SNAPSHOT
    assert evidence["test_fixture_history"]["archive_snapshot"]["path"] == new.OLD_W3_TEST_SNAPSHOT
    assert not (episode_software["root"] / "data").exists()


def test_episode_then_bridge_then_walk_preserves_all_original_json_history(episode_software):
    closure = new.Closure(episode_software["root"])
    bodies = {p: (episode_software["root"] / p).read_bytes() for p in
              (new.HISTORICAL_RESULT, new.HISTORICAL_INVENTORY, "audit/software_code_manifest.json")}
    closure.register_episode_history(defer_walk=True)
    closure.register_bridge_history()
    closure.walk_episode_history()
    for p in (new.OLD_EPISODE_SNAPSHOT, new.OLD_PREDICTION_SNAPSHOT, new.OLD_W3_TEST_SNAPSHOT):
        assert p in closure.files
    for current in episode_software["episode_descriptors"].values():
        assert closure.files[current["path"]]["sha256"] == current["sha256"]
    assert all((episode_software["root"] / p).read_bytes() == b for p, b in bodies.items())
    assert len(closure.bridge_software_history) == len(closure.episode_artifact_compatibility_history) == 1


@pytest.mark.parametrize("source,sha,size", [(new.EPISODE, new.OLD_EPISODE_SHA, new.OLD_EPISODE_BYTES),
    (new.PREDICTION, new.OLD_PREDICTION_SHA, new.OLD_PREDICTION_BYTES),
    (new.LEGACY_W3_TEST, new.OLD_W3_TEST_SHA, new.OLD_W3_TEST_BYTES)])
def test_each_episode_alias_is_explicit_and_rejects_wrong_bytes(episode_software, source, sha, size):
    closure = new.Closure(episode_software["root"])
    with pytest.raises(new.DeploymentError, match="explicit_episode_software_history_required"):
        closure.visit(new._fixed(source, sha, size))
    closure.register_episode_history()
    for wrong in (size + 1, True):
        with pytest.raises(new.DeploymentError, match="historical_episode_descriptor_bytes_mismatch"):
            closure.visit(new._fixed(source, sha, wrong))
    with pytest.raises(new.DeploymentError, match="dependency_file_SHA_or_bytes_mismatch"):
        closure.visit(new._fixed(source, "f" * 64, size))


@pytest.mark.parametrize("path", [new.OLD_EPISODE_SNAPSHOT, new.OLD_PREDICTION_SNAPSHOT,
    new.OLD_W3_TEST_SNAPSHOT, new.EPISODE, new.PREDICTION, new.LEGACY_W3_TEST, new.W3_AUDITOR])
def test_episode_history_rejects_changed_current_or_archived_software(episode_software, path):
    (episode_software["root"] / path).write_bytes(b"# wrong software identity\n")
    closure = new.Closure(episode_software["root"])
    with pytest.raises(new.DeploymentError): closure.register_episode_history()
    assert closure.code_aliases == {}
    assert closure.episode_artifact_compatibility_history == []


@pytest.mark.parametrize("field", ["previous_episode", "previous_prediction_store", "test_fixture_history"])
def test_episode_history_requires_exact_declared_old_identity(episode_software, field):
    value = episode_software["episode_contract"]
    target = value[field] if field != "test_fixture_history" else value[field]["previous_test"]
    target["sha256"] = "0" * 64
    write(episode_software["root"], new.EPISODE_CONTRACT, value)
    with pytest.raises(new.DeploymentError): new.Closure(episode_software["root"]).register_episode_history()


@pytest.mark.parametrize("change", ["old_suite", "GPU", "missing_prediction", "missing_episode_tests", "not_rehashed", "future_contract"])
def test_episode_history_requires_actual_complete_new_software_inventory(episode_software, change):
    value = copy.deepcopy(episode_software["acceptance"])
    rows = json.loads((episode_software["root"] / episode_software["inventory_path"]).read_bytes())["files"]
    if change == "old_suite": value["passed_tests"] = 100
    if change == "GPU": value["GPU_actions"] = 1
    if change == "not_rehashed": value["all_input_SHA_and_bytes_rechecked_after_actual_exit"] = False
    if change.startswith("missing_"):
        missing = new.PREDICTION if change == "missing_prediction" else new.EPISODE_TESTS
        rows = [r for r in rows if r["path"] != missing]
    if change == "future_contract": rows.append(old.descriptor(episode_software["root"], new.EPISODE_CONTRACT))
    value["input_inventory"] = write(episode_software["root"], episode_software["inventory_path"], {"files": rows})
    update_episode_acceptance(episode_software, value)
    with pytest.raises(new.DeploymentError): new.Closure(episode_software["root"]).register_episode_history()


def test_episode_history_missing_contract_or_unfrozen_consumers_fail_closed(episode_software, monkeypatch):
    monkeypatch.setattr(new, "CURRENT_EPISODE_SHA", None)
    with pytest.raises(new.DeploymentError, match="descriptor_consumers_not_yet_frozen"):
        new.Closure(episode_software["root"]).register_episode_history()


def test_failed_episode_dependency_walk_revokes_all_three_new_aliases(episode_software):
    closure = new.Closure(episode_software["root"])
    closure.register_episode_history()
    # The original old bridge is not yet authorized: never skip its history.
    with pytest.raises(new.DeploymentError, match="explicit_bridge_software_history_required"):
        closure.walk_episode_history()
    assert closure.code_aliases == {}
    assert closure.episode_artifact_compatibility_history == []


def test_individually_valid_but_different_software_acceptance_is_rejected(episode_software):
    closure = new.Closure(episode_software["root"])
    closure.register_episode_history()
    contract = episode_software["contract"]
    contract["software_acceptance"] = write(episode_software["root"], "audit/software_fixture/other_actual_acceptance.json",
        episode_software["acceptance"])
    write(episode_software["root"], new.FIX_CONTRACT, contract)
    with pytest.raises(new.DeploymentError, match="one_actual_full_suite"):
        closure.register_bridge_history()
    assert (new.BRIDGE, new.OLD_BRIDGE_SHA) not in closure.code_aliases


def test_episode_history_still_traverses_unknown_native_metadata_dependencies(episode_software):
    value = episode_software["episode_contract"]
    value["extra_native_evidence"] = {"path": "audit/source_extensions_20261019/Dreyer2023/missing/native_metadata.json",
        "sha256": new.OLD_EPISODE_SHA, "bytes": new.OLD_EPISODE_BYTES}
    write(episode_software["root"], new.EPISODE_CONTRACT, value)
    closure = new.Closure(episode_software["root"])
    closure.register_episode_history()
    closure.register_bridge_history()
    with pytest.raises((new.DeploymentError, FileNotFoundError)):
        closure.walk_episode_history()
    assert (new.BRIDGE, new.OLD_BRIDGE_SHA) in closure.code_aliases
    assert all((source, sha) not in closure.code_aliases for source, sha in
        ((new.EPISODE, new.OLD_EPISODE_SHA), (new.PREDICTION, new.OLD_PREDICTION_SHA), (new.LEGACY_W3_TEST, new.OLD_W3_TEST_SHA)))


def test_current_descriptor_consumers_and_archived_sources_are_exactly_pinned():
    for path, sha, size in ((new.EPISODE, new.CURRENT_EPISODE_SHA, new.CURRENT_EPISODE_BYTES),
        (new.PREDICTION, new.CURRENT_PREDICTION_SHA, new.CURRENT_PREDICTION_BYTES),
        (new.OLD_EPISODE_SNAPSHOT, new.OLD_EPISODE_SHA, new.OLD_EPISODE_BYTES),
        (new.OLD_PREDICTION_SNAPSHOT, new.OLD_PREDICTION_SHA, new.OLD_PREDICTION_BYTES),
        (new.OLD_W3_TEST_SNAPSHOT, new.OLD_W3_TEST_SHA, new.OLD_W3_TEST_BYTES)):
        body = (REPO / path).read_bytes()
        assert (hashlib.sha256(body).hexdigest(), len(body)) == (sha, size)


def registered(fixture):
    closure = new.Closure(fixture["root"])
    closure.register_bridge_history()
    return closure


def test_exact_old_alias_and_new_runtime_are_both_bound(software):
    closure = registered(software)
    assert closure.code_aliases[(new.BRIDGE, new.OLD_BRIDGE_SHA)]["path"] == new.OLD_BRIDGE_SNAPSHOT
    assert closure.files[new.BRIDGE]["sha256"] == software["bridge"]["sha256"]
    assert closure.files[new.OLD_BRIDGE_SNAPSHOT]["sha256"] == new.OLD_BRIDGE_SHA
    assert closure.adapter_evidence["no_original_module_globals_replaced"] is True
    assert closure.bridge_software_history[0]["old_suite_does_not_validate_new_runtime"] is True


def test_all_five_old_contract_paths_remain_original_json_and_resolve_archive(software):
    closure = registered(software)
    paths = ["preparation_software_validation_20261020.json", "cpu_hard_interruption_recovery_20261020.json",
        "software_regression_189_20261020.json", "yang_range_transport_software_20261020.json",
        "yang_range_transport_actual_stage_20261020.json"]
    bodies = {}
    for name in paths:
        item = write(software["root"], "contracts/" + name, {"historical_exact_code": software["old"]})
        bodies[item["path"]] = (software["root"] / item["path"]).read_bytes()
        closure.visit(item)
    assert all((software["root"] / p).read_bytes() == body for p, body in bodies.items())
    assert closure.files[new.BRIDGE]["sha256"] == software["bridge"]["sha256"]
    assert new.OLD_BRIDGE_SNAPSHOT in closure.files


def test_undeclared_alias_is_rejected_even_if_bytes_exist(software):
    with pytest.raises(new.DeploymentError, match="explicit_bridge_software_history_required"):
        new.Closure(software["root"]).visit(software["old"])


@pytest.mark.parametrize("key", ["sha256", "bytes"])
def test_wrong_declared_old_identity_rejected(software, key):
    software["contract"]["previous_bridge"][key] = "0" * 64 if key == "sha256" else 7
    write(software["root"], new.FIX_CONTRACT, software["contract"])
    with pytest.raises(new.DeploymentError, match="exact_declared_bridge_software_fix_required"):
        registered(software)


@pytest.mark.parametrize("path", [new.OLD_BRIDGE_SNAPSHOT, new.BRIDGE, new.ORIGINAL_ENTRY, new.ENTRY])
def test_wrong_archive_current_runtime_or_builder_rejected(software, path):
    (software["root"] / path).write_bytes(b"# tampered\n")
    with pytest.raises(new.DeploymentError):
        registered(software)


def test_wrong_bytes_on_nested_old_descriptor_cannot_hide_behind_alias(software):
    closure = registered(software)
    with pytest.raises(new.DeploymentError, match="historical_bridge_descriptor_bytes_mismatch"):
        closure.visit(dict(software["old"], bytes=1))
    with pytest.raises(new.DeploymentError, match="historical_bridge_descriptor_bytes_mismatch"):
        closure.visit(dict(software["old"], path=str(software["root"] / new.BRIDGE), bytes=1))


def test_other_historical_hash_is_not_aliased(software):
    closure = registered(software)
    with pytest.raises(new.DeploymentError, match="dependency_file_SHA_or_bytes_mismatch"):
        closure.visit(dict(software["old"], sha256="0" * 64))


@pytest.mark.parametrize("change", ["targeted_only", "exit_failure", "not_rehashed", "GPU", "missing_bridge_input"])
def test_acceptance_must_be_actual_new_full_suite(software, change):
    acceptance = copy.deepcopy(software["acceptance"])
    if change == "targeted_only": acceptance["passed_tests"] = 98
    if change == "exit_failure": acceptance["actual_exit_code"] = 1
    if change == "not_rehashed": acceptance["all_input_SHA_and_bytes_rechecked_after_actual_exit"] = False
    if change == "GPU": acceptance["GPU_actions"] = 1
    if change == "missing_bridge_input":
        acceptance["input_inventory"] = write(software["root"], software["inventory_path"],
            {"files": [old.descriptor(software["root"], new.ENTRY),
                       old.descriptor(software["root"], new.FINALIZE_HELPER), *software["rows"]]})
    update_acceptance(software, acceptance)
    with pytest.raises(new.DeploymentError): registered(software)


def test_contract_and_acceptance_cannot_form_tested_input_cycle(software):
    acceptance = copy.deepcopy(software["acceptance"])
    rows = json.loads((software["root"] / software["inventory_path"]).read_bytes())["files"]
    rows.append(old.descriptor(software["root"], new.FIX_CONTRACT))
    acceptance["input_inventory"] = write(software["root"], software["inventory_path"], {"files": rows})
    update_acceptance(software, acceptance)
    with pytest.raises(new.DeploymentError, match="future_contract_or_acceptance_cannot_be_software_test_input"):
        registered(software)


def test_changed_nested_tested_input_rolls_back_alias(software):
    (software["root"] / software["rows"][0]["path"]).write_bytes(b"changed input")
    closure = new.Closure(software["root"])
    with pytest.raises(new.DeploymentError): closure.register_bridge_history()
    assert (new.BRIDGE, new.OLD_BRIDGE_SHA) not in closure.code_aliases
    assert closure.adapter_evidence is None
    assert closure.bridge_software_history == []


@pytest.mark.parametrize("opaque", ["acceptance", "inventory"])
def test_opaque_acceptance_or_input_inventory_is_rejected(software, opaque):
    acceptance = copy.deepcopy(software["acceptance"])
    if opaque == "inventory":
        body = (software["root"] / software["inventory_path"]).read_bytes()
        acceptance["input_inventory"] = write(software["root"], software["inventory_path"] + ".txt", body)
        update_acceptance(software, acceptance)
    else:
        software["contract"]["software_acceptance"] = write(software["root"], software["acceptance_path"] + ".txt", acceptance)
        write(software["root"], new.FIX_CONTRACT, software["contract"])
    with pytest.raises(new.DeploymentError, match="must_be_structured_JSON"):
        registered(software)


@pytest.mark.parametrize("changed", [False, True])
def test_new_only_software_input_is_hashed_and_selected(software, changed):
    new_only = write(software["root"], "tests/new_only_fixture.py", b"# only the new suite tested this\n")
    value = json.loads((software["root"] / software["inventory_path"]).read_bytes())
    value["files"].append(new_only)
    acceptance = copy.deepcopy(software["acceptance"])
    acceptance["input_inventory"] = write(software["root"], software["inventory_path"], value)
    update_acceptance(software, acceptance)
    if changed:
        (software["root"] / new_only["path"]).write_bytes(b"# changed after the claimed test\n")
        with pytest.raises(new.DeploymentError, match="software_history_full_SHA_or_bytes_mismatch"):
            registered(software)
    else:
        closure = registered(software)
        assert closure.files[new_only["path"]]["sha256"] == new_only["sha256"]
        assert acceptance["input_inventory"]["path"] in closure.files


def test_new_software_inventory_cannot_request_raw_payload_hashing(software):
    raw = {"path": "data/source_extensions_20261019/Yang2025/unopened.bdf", "sha256": "a" * 64, "bytes": 999}
    value = json.loads((software["root"] / software["inventory_path"]).read_bytes())
    value["files"].append(raw)
    acceptance = copy.deepcopy(software["acceptance"])
    acceptance["input_inventory"] = write(software["root"], software["inventory_path"], value)
    update_acceptance(software, acceptance)
    with pytest.raises(new.DeploymentError, match="software_input_must_be_code_or_build_configuration"):
        registered(software)


def test_missing_native_metadata_is_never_aliased(software):
    closure = registered(software)
    native = {"path": "audit/source_extensions_20261019/Dreyer2023/frozen/native_metadata.json",
              "sha256": new.OLD_BRIDGE_SHA, "bytes": new.OLD_BRIDGE_BYTES}
    with pytest.raises((new.DeploymentError, FileNotFoundError)):
        closure.visit(native)


def test_missing_real_contract_and_unfrozen_runtime_fail_closed(software, monkeypatch):
    (software["root"] / new.FIX_CONTRACT).unlink()
    with pytest.raises(new.DeploymentError, match="dependency_file_missing"): registered(software)
    monkeypatch.setattr(new, "CURRENT_BRIDGE_SHA", None)
    with pytest.raises(new.DeploymentError, match="new_bridge_implementation_not_yet_frozen"): registered(software)


def test_original_52_protocol_runtime_receipt_and_budget_gates_are_inherited():
    for name in ("original_protocol", "portable_runtime", "_receipt", "load_history",
                 "load_runtime_recovery", "register_precision_fix", "check_stable_snapshot"):
        assert getattr(new.Closure, name) is getattr(old.Closure, name)
    assert new.COMPONENTS == old.COMPONENTS
    assert new.SCHEMA == old.SCHEMA


def test_inspect_orchestration_preserves_all_original_gates_exactly():
    original_tree = ast.parse((REPO / new.ORIGINAL_ENTRY).read_bytes())
    adapter_tree = ast.parse((REPO / new.ENTRY).read_bytes())
    before = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == "inspect")
    after = next(n for n in adapter_tree.body if isinstance(n, ast.FunctionDef) and n.name == "inspect")
    added = [n for n in after.body if isinstance(n, ast.If) and any(
        isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "register_bridge_history"
        for x in ast.walk(n))]
    assert len(added) == 1
    after.body.remove(added[0])
    for node in ast.walk(after):
        if isinstance(node, ast.Dict):
            kept = [(k, v) for k, v in zip(node.keys, node.values) if not
                    (isinstance(k, ast.Constant) and k.value in
                     {"deployment_adapter", "historical_software_code_aliases", "historical_provider_reference_contexts",
                      "episode_artifact_compatibility_history"})]
            node.keys = [k for k, _ in kept]
            node.values = [v for _, v in kept]
    assert ast.dump(before) == ast.dump(after)


def test_inspect_does_not_replace_original_module_globals(tmp_path):
    original_class = old.Closure
    report = new.inspect(tmp_path, manifests={name: None for name in new.COMPONENTS},
                         include_software=False, include_wheels=False, history=[])
    assert old.Closure is original_class
    assert report["all_required_data_COMPLETE"] is False
    assert report["raw_payloads_selected"] == 0
    assert report["GPU_actions"] == 0
    assert report["deployment_adapter"] is None
    with pytest.raises(new.DeploymentError): new.final_manifest(report)


def backup_receipt(root, name, rows, *, source_extension=False):
    prefix = "paper3/software-fixture/"
    objects = [{**row, "key": prefix + row["path"], "status": "verified",
        "put": {"status": "created", "http_status": 200, "request_body_bytes": row["bytes"],
                "request_body_sha256": row["sha256"]},
        "get": {"status": "verified", "http_status": 200, "complete_body_read": True,
                "bytes_read": row["bytes"], "sha256": row["sha256"], "declared_content_length": row["bytes"]}}
        for row in rows]
    receipt = {"schema_version": 1, "operation": "paper3_explicit_manifest_backup", "status": "verified",
        "execute": True, "verification": "full_get_body_sha256", "prefix": prefix, "objects": objects}
    if source_extension: receipt["source_extension_profile"] = extension.profile()
    return write(root, name, receipt)


@pytest.mark.parametrize("dataset", ["BNCI2014_001", "Yang2025", "Dreyer2023"])
def test_baseline_and_two_extension_dataset_receipts_keep_exact_profiles_and_raw_zero(software, dataset):
    closure = registered(software)
    source = dataset != "BNCI2014_001"
    prefix = f"audit/source_extensions_20261019/{dataset}/fixture" if source else "audit/software_fixture"
    metadata = write(software["root"], prefix + "/native.json", {"fixture": True})
    raw = {"path": f"data/source_extensions_20261019/{dataset}/native.bdf", "sha256": "a" * 64,
        "bytes": 10**10, "role": "source_development_raw", "dataset": dataset}
    item = backup_receipt(software["root"], prefix + "/receipt.json",
        [dict(metadata, role="audit", dataset=dataset), raw], source_extension=source)
    closure.visit(item)
    group = closure.receipts[item["path"]]
    assert group["io_profile"] == ("source_extension" if source else "baseline")
    assert group["paths"] == [metadata["path"]]
    assert group["excluded_raw_paths"] == [raw["path"]]
    assert raw["path"] not in closure.files


def test_offline_cpu_and_gpu_wheels_use_original_runtime_policy(software):
    closure = registered(software)
    for profile, dataset in (("cpu", "PythonDependenciesCPU"), ("gpu", "PythonDependenciesGPU")):
        item = write(software["root"], f"data/dependency_wheels/{profile}/fixture-1-py3-none-any.whl", b"wheel fixture")
        closure.visit(item)
        assert closure.files[item["path"]]["dataset"] == dataset
    raw = {"path": "data/source_extensions_20261019/Yang2025/never-local.bdf", "sha256": "a" * 64, "bytes": 999}
    closure.visit(raw)
    assert raw["path"] not in closure.files


def test_versioned_finalizer_keeps_old_functions_and_explicit_import_only():
    directory = REPO / "audit/data_plan_20261020/deployment_finalize_review"
    old_body = (directory / "prepare_finalize_plan.py").read_bytes()
    assert hashlib.sha256(old_body).hexdigest() == "6d40cb72f60470ed0fbd3401dbe711628d4820c72d5cb1255a222c9a2c0314fa"
    before = ast.parse(old_body)
    after = ast.parse((directory / "prepare_finalize_plan_v2.py").read_bytes())
    functions = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert functions(before) == functions(after)
    imports = [n for n in after.body if isinstance(n, ast.ImportFrom)]
    assert any(n.module == "tools" and [(a.name, a.asname) for a in n.names] ==
               [("build_data_plan_deploy_manifest_v2", "deployment")] for n in imports)
    spec = importlib.util.spec_from_file_location("_fixture_finalize_v2", directory / "prepare_finalize_plan_v2.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.ROOT == REPO
    assert helper.deployment is new


def test_versioned_final_manifest_retains_adapter_evidence_without_changing_io_schema(episode_software):
    closure = new.Closure(episode_software["root"])
    closure.register_episode_history()
    closure.register_bridge_history()
    closure.walk_episode_history()
    report = {"status": "CACHE_CLOSURE_READY_FOR_BACKUP", "all_required_data_COMPLETE": True,
        "gaps": [], "raw_payloads_selected": 0, "files": list(closure.files.values()),
        "verified_restore_groups": [], "deployment_adapter": closure.adapter_evidence,
        "historical_software_code_aliases": closure.bridge_software_history,
        "episode_artifact_compatibility_history": closure.episode_artifact_compatibility_history}
    report["closure_sha256"] = old.digest(report)
    final = new.final_manifest(report)
    assert final["schema_version"] == old.SCHEMA
    assert set(final["backup_manifests"]) == {"baseline", "source_extension"}
    assert final["raw_payloads_selected"] == 0
    assert final["deployment_adapter"]["version"] == 2
    assert final["historical_software_code_aliases"][0]["archive_snapshot"]["sha256"] == new.OLD_BRIDGE_SHA
    assert final["episode_artifact_compatibility_history"][0]["prediction_store_history"]["archive_snapshot"]["sha256"] == new.OLD_PREDICTION_SHA
    report["episode_artifact_compatibility_history"][0]["actual_new_software_acceptance"] = {"path": "different.json"}
    report["closure_sha256"] = old.digest({k: v for k, v in report.items() if k != "closure_sha256"})
    with pytest.raises(new.DeploymentError, match="one_actual_full_suite"):
        new.final_manifest(report)


def test_original_builder_bytes_remain_frozen():
    body = (REPO / new.ORIGINAL_ENTRY).read_bytes()
    assert (hashlib.sha256(body).hexdigest(), len(body)) == (new.ORIGINAL_SHA, new.ORIGINAL_BYTES)


def copy_policy_metadata(root, start):
    """Copy immutable policy/code only; never a provider file or EEG array."""
    seen = set()
    aliases = {
        (new.MA_CODE, new.MA_OLD_SHA): "audit/data_plan_20261019/other_datasets/ma_native_v1_failure/ma2022.py",
        ("tools/prepare_ma2022_development.py", "7809ae3b2b27b23c6406f88404d0d8b5ebd67bc270479f78af75f329e3144e33"):
            "audit/data_plan_20261019/other_datasets/ma_native_v1_failure/prepare_ma2022_development.py",
    }
    def visit(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= set(value):
                path = old.relative(REPO, value["path"])
                if old.classify(path) == "raw_reference_only" or Path(path).suffix == ".npy": return
                path = aliases.get((path, value["sha256"]), path)
                if path in seen: return
                seen.add(path)
                data = (REPO / path).read_bytes()
                assert hashlib.sha256(data).hexdigest() == value["sha256"]
                write(root, path, data)
                if Path(path).suffix == ".json": visit(json.loads(data))
                return
            for child in value.values():
                if isinstance(child, (dict, list)): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(start)


@pytest.fixture
def yang_history(tmp_path):
    root = tmp_path
    plan_ref = old.descriptor(REPO, new.YANG_PLAN)
    copy_policy_metadata(root, plan_ref)
    write(root, new.BRIDGE, (REPO / new.BRIDGE).read_bytes())
    plan = json.loads((root / new.YANG_PLAN).read_bytes())
    shards = [{"record_id": r["record_id"], "subject_id": r["subject_id"], "session_id": r["session_id"],
               "raw_files": r["raw_files"]} for r in plan["records"]]
    first = plan["records"][0]
    raw_rows = [dict({k: raw[k] for k in ("path", "sha256", "bytes")}, dataset="Yang2025", role="source_development_raw")
                for raw in first["raw_files"]]
    shards[0]["raw_backup_receipt"] = backup_receipt(root,
        "audit/source_extensions_20261019/Yang2025/fixture/raw_receipt.json", raw_rows, source_extension=True)
    write(root, f"cache/source_extensions_20261019/Yang2025/{new.YANG_PLAN_SHA}/source_cache.json",
        {"status": "SOURCE_EXTENSION_CACHE_COMPLETE", "dataset": "Yang2025", "plan_sha256": new.YANG_PLAN_SHA,
         "record_count": 153, "shards": shards})
    return {"root": root, "item": old.descriptor(root, new.YANG_HISTORY), "raw": first["raw_files"]}


def test_pinned_Yang_history_two_raw_references_are_portable_without_original_payload(yang_history, monkeypatch):
    original_open = Path.open
    def no_origin(path, *args, **kwargs):
        if str(path).startswith("/workspace/research3/data/"):
            raise AssertionError("producer raw must never be read")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_origin)
    closure = new.Closure(yang_history["root"])
    closure.visit(yang_history["item"])
    assert all((raw["path"], raw["sha256"]) in closure.raw for raw in yang_history["raw"])
    assert all(not (yang_history["root"] / raw["path"]).exists() for raw in yang_history["raw"])
    assert all(raw["path"] not in closure.files for raw in yang_history["raw"])
    assert closure.provider_reference_contexts[0]["local_original_payload_read"] is False


def test_Yang_raw_descriptor_without_pinned_receipt_context_is_rejected(yang_history):
    value = json.loads((yang_history["root"] / new.YANG_HISTORY).read_bytes())["objects"][0]
    with pytest.raises(new.DeploymentError, match="absolute_dependency_outside_current_root"):
        new.Closure(yang_history["root"]).visit(value)


def test_same_Yang_history_body_at_other_path_is_not_a_raw_exception(yang_history):
    item = write(yang_history["root"], "audit/unregistered_history.json",
                 (yang_history["root"] / new.YANG_HISTORY).read_bytes())
    with pytest.raises(new.DeploymentError, match="absolute_dependency_outside_current_root"):
        new.Closure(yang_history["root"]).visit(item)


@pytest.mark.parametrize("field,value", [("sha256", "0"*64), ("bytes", 1), ("path", "/elsewhere/data.bdf")])
def test_changed_Yang_historical_identity_cannot_change_pinned_whole_receipt(yang_history, field, value):
    data = json.loads((yang_history["root"] / new.YANG_HISTORY).read_bytes())
    data["objects"][0][field] = value
    item = write(yang_history["root"], new.YANG_HISTORY, data)
    with pytest.raises(new.DeploymentError, match="software_history_full_SHA_or_bytes_mismatch"):
        new.Closure(yang_history["root"]).visit(item)


def test_Yang_original_raw_full_GET_proof_remains_mandatory(yang_history):
    path = yang_history["root"] / "audit/source_extensions_20261019/Yang2025/fixture/raw_receipt.json"
    data = json.loads(path.read_bytes()); data["objects"][0]["get"]["complete_body_read"] = False
    path.write_bytes(body := (json.dumps(data) + "\n").encode())
    with pytest.raises(Exception): new.Closure(yang_history["root"]).visit(yang_history["item"])


def ma_fixture(root, legacy=False):
    namespace = "development_car3" if legacy else "development_car3_native_v2"
    plan_path = new.MA_PLANS[namespace][0]
    copy_policy_metadata(root, old.descriptor(REPO, plan_path))
    write(root, new.MA_CODE, (REPO / new.MA_CODE).read_bytes())
    if legacy:
        history = old.descriptor(REPO, new.MA_HISTORY)
        copy_policy_metadata(root, history)
        # Unchanged current code is needed by the original authenticated alias
        # registrar. Old code bodies remain only at their exact snapshot paths.
        plan = json.loads((root / plan_path).read_bytes())
        for name in plan["code"]:
            write(root, name, (REPO / name).read_bytes())
    plan = json.loads((root / plan_path).read_bytes())
    record = plan["records"][0]
    path = record["output"] + "/manifest.json"
    metadata = json.loads((REPO / path).read_bytes())
    # The test covers metadata portability, with tiny synthetic cached payloads;
    # no real EEG array is read or copied into this temporary fixture.
    for name in metadata["artifacts"]:
        artifact = write(root, record["output"] + "/" + name, [] if name.endswith(".json") else b"synthetic array fixture")
        metadata["artifacts"][name] = dict(artifact, path="/workspace/research3/" + artifact["path"])
    item = write(root, path, metadata)
    state_path = f"audit/source_extensions_20261019/Ma2022/{plan['plan_sha256']}/{record['record_id']}/state.json"
    write(root, state_path, {"status": "COMPLETE", "plan_sha256": plan["plan_sha256"],
        "record_id": record["record_id"], "artifacts": [item]})
    closure = new.Closure(root)
    if legacy: closure.load_history(new.MA_HISTORY, defer_walk=True)
    return {"root": root, "item": item, "state_path": state_path, "closure": closure,
            "code_sha": new.MA_OLD_SHA if legacy else new.MA_CURRENT_SHA}


@pytest.mark.parametrize("legacy", [False, True])
def test_Ma_plan_bound_adapter_identity_is_portable_and_preserves_authenticated_history(tmp_path, legacy):
    sample = ma_fixture(tmp_path, legacy)
    sample["closure"].visit(sample["item"])
    code_path = "audit/data_plan_20261019/other_datasets/ma_native_v1_failure/ma2022.py" if legacy else new.MA_CODE
    assert sample["closure"].files[code_path]["sha256"] == sample["code_sha"]
    assert sample["closure"].provider_reference_contexts[-1]["kind"] == "Ma_recorded_adapter_identity"


@pytest.mark.parametrize("change", ["missing_alias", "unregistered_history", "wrong_archive_SHA"])
def test_legacy_Ma_absolute_code_requires_existing_authenticated_alias(tmp_path, change):
    sample = ma_fixture(tmp_path, True)
    closure = sample["closure"]
    if change == "missing_alias": closure.code_aliases.pop((new.MA_CODE, new.MA_OLD_SHA))
    if change == "unregistered_history": closure.history.clear()
    if change == "wrong_archive_SHA": closure.code_aliases[(new.MA_CODE, new.MA_OLD_SHA)] = dict(
        closure.code_aliases[(new.MA_CODE, new.MA_OLD_SHA)], sha256="0"*64)
    with pytest.raises(new.DeploymentError): closure.visit(sample["item"])


def update_ma_metadata(sample, transform):
    data = json.loads((sample["root"] / sample["item"]["path"]).read_bytes())
    transform(data)
    sample["item"] = write(sample["root"], sample["item"]["path"], data)
    state = json.loads((sample["root"] / sample["state_path"]).read_bytes())
    state["artifacts"] = [sample["item"]]
    write(sample["root"], sample["state_path"], state)


@pytest.mark.parametrize("change", ["wrong_schema", "wrong_origin", "wrong_code_SHA", "wrong_code_bytes", "wrong_role"])
def test_Ma_field_schema_origin_and_exact_code_identity_cannot_expand(tmp_path, change):
    sample = ma_fixture(tmp_path)
    def mutate(data):
        if change == "wrong_schema": data["schema_version"] = "unknown"
        if change == "wrong_origin": data["adapter_code"]["path"] = str(tmp_path / new.MA_CODE)
        if change == "wrong_code_SHA": data["adapter_code"]["sha256"] = "0"*64
        if change == "wrong_code_bytes": data["adapter_code"]["bytes"] = 1
        if change == "wrong_role": data["role"] = "source_train"
    update_ma_metadata(sample, mutate)
    with pytest.raises(new.DeploymentError): sample["closure"].visit(sample["item"])


def test_Ma_context_keeps_every_other_nested_dependency_mandatory(tmp_path):
    sample = ma_fixture(tmp_path)
    update_ma_metadata(sample, lambda data: data.update(other_dependency={"path": "audit/required_missing.json",
        "sha256": "0"*64, "bytes": 10}))
    with pytest.raises((FileNotFoundError, new.DeploymentError)): sample["closure"].visit(sample["item"])


def test_absolute_code_outside_authenticated_Ma_field_context_still_fails(tmp_path):
    sample = ma_fixture(tmp_path)
    code = {"path": "/workspace/research3/" + new.MA_CODE, "sha256": new.MA_CURRENT_SHA, "bytes": 14577}
    with pytest.raises(new.DeploymentError, match="absolute_dependency_outside_current_root"):
        sample["closure"].visit(code)


@pytest.fixture
def stieger_component_metadata(tmp_path):
    root=tmp_path;path='cache/stieger_v2/stieger_cache.json';sha='1'*64
    data={'schema_version':'paper3-stieger-cache-v2','status':'STIEGER_CACHE_COMPLETE',
        'session_count':598,'sessions':[{'synthetic_metadata_only':True} for _ in range(598)],
        'participant_count':62,'plan_sha256':sha}
    manifest=write(root,path,data)
    amendment=write(root,'audit/recovery/amendment.json',{'effective_Stieger_CPU_wall_cap_seconds':120})
    completion=write(root,'audit/recovery/completion.json',{'schema_version':'paper3-recovered-Stieger-CPU-completion-v1',
        'amendment':amendment,'cache_manifest':manifest,'scientific_plan_sha256':sha,
        'original_body_caps_unchanged':True,'effective_CPU_wall_cap_seconds':120})
    receipt=write(root,'audit/recovery/receipt.json',{'operation':'paper3_explicit_manifest_backup','status':'not_verified'})
    task={'status':'COMPLETE','cache_manifest':manifest,'cpu_operational_recovery':amendment,
        'cpu_operational_completion':completion,'cpu_operational_R2_receipt':{k:receipt[k] for k in ('path','sha256')},
        'effective_CPU_wall_cap_seconds':120}
    progress={'status':'W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY','plan_sha256':sha,'completed_sessions':598,
        'explicit_runtime_reconciliation':amendment,'active_wall_elapsed_s':100}
    write(root,'status/tasks.json',{'W3':task});write(root,'cache/stieger_v2/progress.json',progress)
    return root,path,data,task,progress,receipt


def commit_stieger_component(stieger_component_metadata):
    root,path,data,task,progress,receipt=stieger_component_metadata
    write(root,'status/tasks.json',{'W3':task});write(root,'cache/stieger_v2/progress.json',progress)


def test_legal_pair_is_exactly_the_original_boundary_gap(stieger_component_metadata):
    root,path,data,task,progress,receipt=stieger_component_metadata
    with pytest.raises(new.DeploymentError,match='Stieger_completed_CPU_recovery_artifacts_required'):
        new.original.complete_component(root,'Stieger2021',path)
    actual,receipts=new.complete_component(root,'Stieger2021',path)
    assert actual==data and receipt['path'] in receipts
    assert task['cpu_operational_R2_receipt']=={k:receipt[k] for k in ('path','sha256')}


def test_existing_triplet_still_passes_unchanged(stieger_component_metadata):
    root,path,data,task,progress,receipt=stieger_component_metadata
    task['cpu_operational_R2_receipt']=receipt;commit_stieger_component(stieger_component_metadata)
    assert new.complete_component(root,'Stieger2021',path)==new.original.complete_component(root,'Stieger2021',path)


@pytest.mark.parametrize('change',['wrong_sha','wrong_bytes','negative_bytes','float_bytes','bool_bytes','unknown_key','missing_path','tampered_body'])
def test_pair_or_triplet_never_weakens_full_identity(stieger_component_metadata,change):
    root,path,data,task,progress,receipt=stieger_component_metadata
    item=task['cpu_operational_R2_receipt']
    if change=='wrong_sha':item['sha256']='0'*64
    if change=='wrong_bytes':item['bytes']=receipt['bytes']+1
    if change=='negative_bytes':item['bytes']=-1
    if change=='float_bytes':item['bytes']=float(receipt['bytes'])
    if change=='bool_bytes':item['bytes']=False
    if change=='unknown_key':item['role']='audit'
    if change=='missing_path':del item['path']
    if change=='tampered_body':write(root,receipt['path'],{'different':'body'})
    commit_stieger_component(stieger_component_metadata)
    with pytest.raises(new.DeploymentError,match='Stieger_completed_CPU_recovery_artifacts_required'):
        new.complete_component(root,'Stieger2021',path)


@pytest.mark.parametrize('change',['open_wall','not_complete','597','wrong_plan','wrong_cache','wrong_cap','over_cap','negative_wall','bool_wall','nonfinite_wall'])
def test_all_original_global_and_budget_gates_remain_closed(stieger_component_metadata,change):
    root,path,data,task,progress,receipt=stieger_component_metadata
    if change=='open_wall':progress['wall_intent']=None
    if change=='not_complete':task['status']='RUNNING'
    if change=='597':progress['completed_sessions']=597
    if change=='wrong_plan':progress['plan_sha256']='0'*64
    if change=='wrong_cache':task['cache_manifest']['bytes']+=1
    if change=='wrong_cap':task['effective_CPU_wall_cap_seconds']=60
    if change=='over_cap':progress['active_wall_elapsed_s']=121
    if change=='negative_wall':progress['active_wall_elapsed_s']=-1
    if change=='bool_wall':progress['active_wall_elapsed_s']=True
    if change=='nonfinite_wall':progress['active_wall_elapsed_s']=float('nan')
    commit_stieger_component(stieger_component_metadata)
    with pytest.raises(new.DeploymentError):new.complete_component(root,'Stieger2021',path)


def test_pair_full_SHA_never_substitutes_for_downstream_full_GET_verification(stieger_component_metadata):
    root,path,data,task,progress,receipt=stieger_component_metadata
    new.complete_component(root,'Stieger2021',path)
    with pytest.raises(Exception):new.original.Closure(root).path(receipt['path'])


@pytest.mark.parametrize('name',['source4','Yang2025','Dreyer2023','Ma2022'])
def test_all_other_components_delegate_without_any_local_reads(tmp_path,monkeypatch,name):
    calls=[];sentinel=({'delegate':name},['original'])
    def delegated(root,actual_name,path):calls.append((root,actual_name,path));return sentinel
    monkeypatch.setattr(new.original,'complete_component',delegated)
    assert new.complete_component(tmp_path,name,'unopened.json') is sentinel
    assert calls==[(tmp_path,name,'unopened.json')]
    assert list(tmp_path.iterdir())==[]


def test_entire_stieger_branch_AST_is_original_except_one_descriptor_check():
    original=ast.parse(Path(new.original.__file__).read_bytes())
    before=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='complete_component')
    branch=before.body[1].orelse[0]
    assert ast.unparse(branch.test)=="name == 'Stieger2021'"
    expected=copy.deepcopy(branch.body)
    loops=[n for s in expected for n in ast.walk(s) if isinstance(n,ast.For)
           and isinstance(n.iter,ast.Tuple) and len(n.iter.elts)==3
           and all(isinstance(x,ast.Constant) for x in n.iter.elts)
           and n.iter.elts[-1].value=='cpu_operational_R2_receipt']
    assert len(loops)==1 and isinstance(loops[0].body[1],ast.If)
    loops[0].body[1]=ast.parse('_completed_CPU_reference(root,item)').body[0]
    candidate=ast.parse((REPO/new.ENTRY).read_bytes())
    after=next(n for n in candidate.body if isinstance(n,ast.FunctionDef) and n.name=='complete_component')
    assert ast.dump(ast.Module(body=expected,type_ignores=[]))==ast.dump(ast.Module(body=after.body[2:-1],type_ignores=[]))
    assert new.descriptor is new.original.descriptor and new.EXPECTED is new.original.EXPECTED
