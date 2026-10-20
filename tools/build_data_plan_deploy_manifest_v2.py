#!/usr/bin/env python3
"""Versioned, local-only cache closure with explicit software history aliases.

The original bf8e builder stays byte-identical. Its Closure and all scientific,
receipt, profile, protocol and raw-exclusion gates are reused explicitly. This
adapter adds only a declared software-history binding; it never patches the
original module's globals or treats a missing artifact as optional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENTRY = "tools/build_data_plan_deploy_manifest_v2.py"
ORIGINAL_ENTRY = "tools/build_data_plan_deploy_manifest.py"
ORIGINAL_SHA = "bf8e5664b445d2108db1586b655b70a57d04985e9601f62321ac1d69b64e588a"
ORIGINAL_BYTES = 56983
# Check the original source before importing it, as well as the selected root
# during every inspection. No code is loaded from a contract-controlled path.
_original_body = (ROOT / ORIGINAL_ENTRY).read_bytes()
if (hashlib.sha256(_original_body).hexdigest(), len(_original_body)) != (ORIGINAL_SHA, ORIGINAL_BYTES):
    raise ValueError("frozen_original_deployment_builder_changed")
from tools import build_data_plan_deploy_manifest as original

SCHEMA = original.SCHEMA
COMPONENTS = original.COMPONENTS
EXPECTED = original.EXPECTED
PRECISION_AUDIT = original.PRECISION_AUDIT
RUNTIME_CONTRACT = original.RUNTIME_CONTRACT
DEFAULT_WHEELS = original.DEFAULT_WHEELS
DeploymentError = original.DeploymentError
descriptor = original.descriptor
relative = original.relative
digest = original.digest
default_manifests = original.default_manifests
write_once = original.write_once

FIX_CONTRACT = "contracts/yang_zip_member_reference_fix_20261020.json"
FIX_SCHEMA = "paper3-yang-zip-member-reference-compatibility-fix-v1"
ACCEPTANCE_SCHEMA = "paper3-yang-zip-member-reference-software-acceptance-v1"
BRIDGE = "p3work/source_extension_bridge.py"
OLD_BRIDGE_SHA = "4590359dae8b61051f9eaebe30b45495eca5dbbc71ac9400b7a5df7ccb154da0"
OLD_BRIDGE_BYTES = 48735
OLD_BRIDGE_SNAPSHOT = "audit/data_plan_20261020/yang_zip_member_reference_fix/source_extension_bridge_4590359.py.txt"
CURRENT_BRIDGE_SHA = "54b1d2068ffd6d49817109661bd6bab580b30a7b69ad720a073eb12f0ac71420"
CURRENT_BRIDGE_BYTES = 61719
FINALIZE_HELPER = "audit/data_plan_20261020/deployment_finalize_review/prepare_finalize_plan_v2.py"
HISTORICAL_RESULT = "audit/data_plan_20261020/software_regression/full_suite_hard_interruption_actual_result.json"
HISTORICAL_RESULT_SHA = "fe2b7c508a9447fe55ec18accb15372c399625067f55fe61b8c319022b9f2de0"
HISTORICAL_RESULT_BYTES = 1026
HISTORICAL_INVENTORY = "audit/data_plan_20261020/software_regression/full_suite_hard_interruption_input_inventory.json"
HISTORICAL_INVENTORY_SHA = "fe0be310eedfcecc5cd2c64582cb7d680e9363bf4fa3197b35bfd084b5bd62d0"
HISTORICAL_INVENTORY_BYTES = 32151
YANG_HISTORY = "audit/data_plan_20261019/yang/single_session_download_receipt.json"
YANG_HISTORY_SHA = "2c90692b775985ab5fb68d2e92ea31513ab9e88b3318dfc9f6daa8b1edd08c9f"
YANG_HISTORY_BYTES = 8929
YANG_PLAN_SHA = "ab9a075ff1fa84548bfe958d3c4b6d6897d6b9dbc29d0ed49d807ec3d4b2b609"
YANG_PLAN = f"audit/source_extensions_20261019/Yang2025/{YANG_PLAN_SHA}/frozen_plan.json"
YANG_PLAN_FILE_SHA = "55c7a36d339e2923d0c3fec45072d660127ff84f9a65dfbc3805f9259d3b3fe6"
MA_CODE = "p3work/source_extensions/ma2022.py"
MA_CURRENT_SHA = "72c55ec6c392063ab5fb077bd8e14a7ddce8d1ab946e0b354400bc118a901e06"
MA_OLD_SHA = "e1578d33cd8e68a5d49bf5883757ea829066b2b6af6a060e3d6aa73bd04b42eb"
MA_HISTORY = "audit/data_plan_20261019/other_datasets/ma_native_v1_failure/budget_conservation_amendment.json"
MA_HISTORY_SHA = "bfab31355405608e61db272e2ae9ce936cffbaac60ea65575e6b3c82196240c2"
EPISODE_CONTRACT = "contracts/episode_descriptor_compatibility_20261020.json"
EPISODE_FIX_SCHEMA = "paper3-episode-artifact-descriptor-compatibility-fix-v1"
EPISODE = "p3work/episodes.py"
OLD_EPISODE_SHA = "348604283251b8d50a34c5b0545a895afb7bbaa024ff82d7c62be11e30504a65"
OLD_EPISODE_BYTES = 43688
OLD_EPISODE_SNAPSHOT = "audit/data_plan_20261020/episode_descriptor_compatibility_fix/original_episodes_348604.py.txt"
CURRENT_EPISODE_SHA = "6ee44b0adafdf080885be943a45a3df84cb7407c602a64bf8dcd549746f092cb"
CURRENT_EPISODE_BYTES = 44001
PREDICTION = "p3work/prediction_store.py"
OLD_PREDICTION_SHA = "e1af634b84bc234bc71394c6fff222d4cd67b2d4ce7f88bc2da5a40b4abde5ab"
OLD_PREDICTION_BYTES = 32085
OLD_PREDICTION_SNAPSHOT = "audit/data_plan_20261020/episode_descriptor_compatibility_fix/original_prediction_store_e1af63.py.txt"
CURRENT_PREDICTION_SHA = "788ec15bb27ea015b729e7e6856982427990938f64dc28aea5416bb8ffea6f86"
CURRENT_PREDICTION_BYTES = 32664
LEGACY_W3_TEST = "tests/test_audit_completed_stieger_w3.py"
OLD_W3_TEST_SHA = "5393535aa0582aeebd7bbda940bf594ebf67d8183adab8db23b15826d07f0df7"
OLD_W3_TEST_BYTES = 17545
OLD_W3_TEST_SNAPSHOT = "audit/data_plan_20261020/episode_descriptor_compatibility_fix/original_test_audit_completed_stieger_w3.py.txt"
W3_AUDITOR = "tools/audit_completed_stieger_w3_v2.py"
W3_AUDITOR_TESTS = "tests/test_audit_completed_stieger_w3_v2.py"
EPISODE_TESTS = "tests/test_episode_artifact_descriptor_compatibility.py"
PREDICTION_TESTS = "tests/test_prediction_artifact_descriptor_compatibility.py"
ORIGINAL_117_SHA = original.ORIGINAL_SOFTWARE_MANIFEST_SHA
MA_PLANS = {
    "development_car3_native_v2": ("catalog/source_extensions_20261019/Ma2022/development_plan_native_v2.json",
        "1663a158f2fde4d3a4f44909785155322b75978ccb9b6090bfba68528e826bef", 126154,
        "46e6abb9458925d7827bafa5d9feaf12cb075ecb9c1f3b980d804f9daa4b6f8b", MA_CURRENT_SHA, 14577),
    "development_car3": ("catalog/source_extensions_20261019/Ma2022/development_plan.json",
        "add18ace4969f07d4a9c711724c1994153854c34b03129655d5102b0646d5824", 124815,
        "b027657dcfb234b093d806a23a69ebb9bfbc611f7cb4181631c330e1899f08eb", MA_OLD_SHA, 12504),
}


def _exact(root, item, expected=None):
    if (not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}
            or not isinstance(item["path"], str) or type(item["bytes"]) is not int
            or item["bytes"] < 0 or not isinstance(item["sha256"], str)):
        raise DeploymentError("exact_software_history_descriptor_required")
    actual = descriptor(root, relative(root, item["path"]))
    if actual != item or expected is not None and actual != expected:
        raise DeploymentError("software_history_full_SHA_or_bytes_mismatch", item["path"])
    return actual


def _fixed(path, sha, size):
    return {"path": path, "sha256": sha, "bytes": size}


def _completed_CPU_reference(root, item):
    """A producer's SHA-bound pair may optionally carry exact integer bytes."""
    if (not isinstance(item,dict) or set(item) not in
            ({'path','sha256'},{'path','sha256','bytes'})
            or not isinstance(item['path'],str)
            or not isinstance(item['sha256'],str)
            or re.fullmatch('[a-f0-9]{64}',item['sha256']) is None
            or 'bytes' in item and (type(item['bytes']) is not int or item['bytes']<0)):
        raise DeploymentError('Stieger_completed_CPU_recovery_artifacts_required')
    actual=descriptor(root,item['path'])
    if any(actual[key]!=value for key,value in item.items()):
        raise DeploymentError('Stieger_completed_CPU_recovery_artifacts_required')
    return actual


def complete_component(root: Path, name: str, path: str) -> tuple[dict,list[str]]:
    if name != 'Stieger2021':
        return original.complete_component(root,name,path)
    data=json.loads((root/path).read_bytes())
    if (data.get("schema_version") != "paper3-stieger-cache-v2" or data.get("status") != "STIEGER_CACHE_COMPLETE"
            or data.get("session_count") != 598 or len(data.get("sessions", [])) != EXPECTED[name]
            or data.get("participant_count") != 62):
        raise DeploymentError("all598_stieger_COMPLETE_manifest_required")
    task=json.loads((root/"status/tasks.json").read_bytes()).get("W3",{})
    progress_path=str(Path(path).parent/"progress.json")
    progress=json.loads((root/progress_path).read_bytes())
    if (task.get("status") != "COMPLETE" or task.get("cache_manifest") != descriptor(root,path)
            or progress.get("status") != "W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY"
            or progress.get("plan_sha256") != data.get("plan_sha256")
            or progress.get("completed_sessions") != 598 or "wall_intent" in progress):
        raise DeploymentError("Stieger_actual_closed_global_completion_required")
    receipts = [str(Path(path).parent/"manifest_backup_receipt.json"),progress_path]
    if progress.get("explicit_runtime_reconciliation") is not None:
        if task.get("cpu_operational_recovery") != progress["explicit_runtime_reconciliation"]:
            raise DeploymentError("Stieger_explicit_CPU_recovery_must_match_final_status")
        for key in ("cpu_operational_recovery","cpu_operational_completion","cpu_operational_R2_receipt"):
            item=task.get(key)
            _completed_CPU_reference(root,item)
            receipts.append(item["path"])
        completion=json.loads((root/task["cpu_operational_completion"]["path"]).read_bytes())
        amendment=json.loads((root/task["cpu_operational_recovery"]["path"]).read_bytes())
        cap=amendment.get("effective_Stieger_CPU_wall_cap_seconds")
        if (completion.get("schema_version") != "paper3-recovered-Stieger-CPU-completion-v1"
                or completion.get("amendment") != task["cpu_operational_recovery"]
                or completion.get("cache_manifest") != descriptor(root,path)
                or completion.get("scientific_plan_sha256") != data["plan_sha256"]
                or completion.get("original_body_caps_unchanged") is not True
                or completion.get("effective_CPU_wall_cap_seconds") != cap
                or task.get("effective_CPU_wall_cap_seconds") != cap
                or type(progress.get("active_wall_elapsed_s")) not in (int,float)
                or not math.isfinite(progress["active_wall_elapsed_s"])
                or not 0 <= progress["active_wall_elapsed_s"] <= cap):
            raise DeploymentError("Stieger_completed_CPU_recovery_scope_or_budget_changed")
    return data,receipts


class Closure(original.Closure):
    """Original closure plus a single, explicitly authorized software alias."""

    def __init__(self, root):
        super().__init__(root)
        self.bridge_software_history = []
        self.adapter_evidence = None
        self.provider_reference_contexts = []
        self._metadata_context = None
        self._metadata_plan_cache = {}
        self.episode_artifact_compatibility_history = []
        self._pending_episode_history = None

    def visit(self, item, **kwargs):
        path = relative(self.root, item["path"]) if isinstance(item, dict) and isinstance(item.get("path"), str) else None
        if path == BRIDGE and item.get("sha256") == OLD_BRIDGE_SHA:
            if (BRIDGE, OLD_BRIDGE_SHA) not in self.code_aliases:
                raise DeploymentError("explicit_bridge_software_history_required", BRIDGE)
            if "bytes" in item and item["bytes"] != OLD_BRIDGE_BYTES:
                raise DeploymentError("historical_bridge_descriptor_bytes_mismatch", BRIDGE)
        for source, sha, size in ((EPISODE, OLD_EPISODE_SHA, OLD_EPISODE_BYTES),
                                  (PREDICTION, OLD_PREDICTION_SHA, OLD_PREDICTION_BYTES),
                                  (LEGACY_W3_TEST, OLD_W3_TEST_SHA, OLD_W3_TEST_BYTES)):
            if path == source and item.get("sha256") == sha:
                if (source, sha) not in self.code_aliases:
                    raise DeploymentError("explicit_episode_software_history_required", source)
                if "bytes" in item and (type(item["bytes"]) is not int or item["bytes"] != size):
                    raise DeploymentError("historical_episode_descriptor_bytes_mismatch", source)
        if path in self.visited:
            return super().visit(item, **kwargs)
        context = None
        if path == YANG_HISTORY:
            context = self._yang_history_context(item)
        elif (path is not None and original.re.fullmatch(
                r"cache/source_extensions_20261019/Ma2022/(development_car3|development_car3_native_v2)/sub-\d{3}/ses-\d{2}/manifest\.json", path)):
            context = self._ma_adapter_context(item)
        if context is None:
            return super().visit(item, **kwargs)
        previous = self._metadata_context
        self._metadata_context = context
        try:
            result = super().visit(item, **kwargs)
        finally:
            self._metadata_context = previous
        for dependency in context["dependencies"]:
            self.visit(dependency)
        return result

    def walk(self, value, *, context="Paper3"):
        """Classify only authenticated fields in two known producer schemas."""
        qualified = self._metadata_context
        if qualified is None or value != qualified["body"]:
            return super().walk(value, context=context)
        self._metadata_context = None
        try:
            for name, child in value.items():
                if name == qualified["field"]:
                    if qualified["kind"] == "Yang_historical_raw_identity":
                        for recorded, raw in zip(child, qualified["raw"]):
                            self.raw[(raw["path"], raw["sha256"])] = {
                                **{k: raw[k] for k in ("path", "sha256", "bytes")},
                                "dataset": "Yang2025", "payload_selected": False,
                                "recorded_producer_path": recorded["path"]}
                            for key, nested in recorded.items():
                                if key not in {"path", "sha256", "bytes"} and isinstance(nested, (dict, list)):
                                    self.walk(nested, context=context)
                    else:
                        self.visit(qualified["canonical_code"])
                        for key, nested in child.items():
                            if key not in {"path", "sha256", "bytes"} and isinstance(nested, (dict, list)):
                                self.walk(nested, context=context)
                elif isinstance(child, (dict, list)):
                    self.walk(child, context=context)
            self.provider_reference_contexts.append({"kind": qualified["kind"],
                "metadata": qualified["metadata"], "field": qualified["field"],
                "local_original_payload_read": False, "raw_payload_selected": False})
        finally:
            self._metadata_context = qualified

    def _pinned_plan(self, path, sha, size=None):
        item = self.actual_descriptor(path)
        if item["sha256"] != sha or size is not None and item["bytes"] != size:
            raise DeploymentError("context_frozen_plan_SHA_or_bytes_changed", path)
        if path not in self._metadata_plan_cache:
            self._metadata_plan_cache[path] = json.loads((self.root / path).read_bytes())
        return item, self._metadata_plan_cache[path]

    def _yang_history_context(self, item):
        """The original pinned receipt's two identities; never open its origin."""
        metadata = _exact(self.root, {k: item[k] for k in ("path", "sha256", "bytes")},
            _fixed(YANG_HISTORY, YANG_HISTORY_SHA, YANG_HISTORY_BYTES))
        _exact(self.root, _fixed(BRIDGE, CURRENT_BRIDGE_SHA, CURRENT_BRIDGE_BYTES))
        from p3work import source_extension_bridge as bridge
        if Path(bridge.__file__).read_bytes() != (self.root / BRIDGE).read_bytes():
            raise DeploymentError("executed_historical_reference_verifier_differs_from_root")
        plan_ref, plan = self._pinned_plan(YANG_PLAN, YANG_PLAN_FILE_SHA)
        if (plan.get("plan_sha256") != YANG_PLAN_SHA
                or digest({k:v for k,v in plan.items() if k != "plan_sha256"}) != YANG_PLAN_SHA
                or plan.get("dataset") != "Yang2025" or plan.get("role") != "source_train"
                or plan.get("source_only") is not True or plan.get("task_subset") != "2C"
                or plan.get("record_count") != 153 or len(plan.get("records", [])) != 153
                or plan.get("historical_get_evidence") != [metadata]):
            raise DeploymentError("exact_Yang_historical_reference_plan_required")
        cache_path = f"cache/source_extensions_20261019/Yang2025/{YANG_PLAN_SHA}/source_cache.json"
        cache_ref = self.actual_descriptor(cache_path)
        manifest = json.loads((self.root / cache_path).read_bytes())
        if (manifest.get("status") != "SOURCE_EXTENSION_CACHE_COMPLETE" or manifest.get("dataset") != "Yang2025"
                or manifest.get("plan_sha256") != YANG_PLAN_SHA or manifest.get("record_count") != 153
                or len(manifest.get("shards", [])) != 153):
            raise DeploymentError("completed_Yang_reference_manifest_required")
        record = next(r for r in plan["records"] if r["record_id"] == "sub-001_ses-01")
        shards = [s for s in manifest["shards"] if s.get("record_id") == record["record_id"]]
        if (len(shards) != 1 or shards[0].get("raw_files") != record["raw_files"]
                or shards[0].get("subject_id") != record["subject_id"]
                or shards[0].get("session_id") != record["session_id"]):
            raise DeploymentError("exact_Yang_historical_reference_record_required")
        data = json.loads((self.root / metadata["path"]).read_bytes())
        native_path = f"audit/source_extensions_20261019/Yang2025/{YANG_PLAN_SHA}/sub-001_ses-01/native_metadata.json"
        catalog = {"historical": metadata, "records": {native_path: (record, shards[0])}}
        verified = {}
        bridge._yang_historical_members(self.root, metadata, data, catalog, verified)
        dependencies = [plan_ref, cache_ref] + [self.actual_descriptor(p) for p in verified]
        return {"kind": "Yang_historical_raw_identity", "body": data, "field": "objects",
            "metadata": metadata, "raw": record["raw_files"], "dependencies": dependencies}

    def _ma_adapter_context(self, item):
        metadata = _exact(self.root, {k: item[k] for k in ("path", "sha256", "bytes")})
        namespace = Path(item["path"]).parts[3]
        plan_path, file_sha, size, semantic, code_sha, code_size = MA_PLANS[namespace]
        plan_ref, plan = self._pinned_plan(plan_path, file_sha, size)
        canonical = _fixed(MA_CODE, code_sha, code_size)
        if (plan.get("plan_sha256") != semantic
                or digest({k:v for k,v in plan.items() if k != "plan_sha256"}) != semantic
                or plan.get("dataset") != "Ma2022" or plan.get("role") != "independent_development_unassigned"
                or plan.get("code", {}).get(MA_CODE) != canonical):
            raise DeploymentError("exact_Ma_record_plan_and_adapter_required")
        matches = [r for r in plan["records"] if r["output"] + "/manifest.json" == item["path"]]
        if len(matches) != 1:
            raise DeploymentError("Ma_metadata_must_match_exact_frozen_record_output")
        record = matches[0]
        state_path = f"audit/source_extensions_20261019/Ma2022/{semantic}/{record['record_id']}/state.json"
        state_ref = self.actual_descriptor(state_path)
        state = json.loads((self.root / state_path).read_bytes())
        if (state.get("status") != "COMPLETE" or state.get("plan_sha256") != semantic
                or state.get("record_id") != record["record_id"] or metadata not in state.get("artifacts", [])):
            raise DeploymentError("Ma_adapter_reference_requires_actual_complete_record_binding")
        if namespace == "development_car3":
            snapshot = self.code_aliases.get((MA_CODE, MA_OLD_SHA))
            expected_history = _fixed(MA_HISTORY, MA_HISTORY_SHA, 1925)
            if (snapshot is None or not any(h.get("amendment") == expected_history for h in self.history)
                    or snapshot.get("sha256") != MA_OLD_SHA or snapshot.get("bytes") != 12504):
                raise DeploymentError("authenticated_existing_Ma_historical_code_alias_required")
            _exact(self.root, snapshot)
        else:
            _exact(self.root, canonical)
        data = json.loads((self.root / metadata["path"]).read_bytes())
        expected_code = dict(canonical, path="/workspace/research3/" + MA_CODE)
        if (data.get("schema_version") != "paper3-Ma2022-native-development-audit-v1"
                or data.get("dataset") != "Ma2022" or type(data.get("provider_version")) is not int
                or data["provider_version"] != 1
                or data.get("role") != "independent_development" or data.get("adapter_code") != expected_code
                or any(data.get("identity", {}).get(kind, {}).get("path") != "/workspace/research3/" + raw["path"]
                    for kind, raw in zip(("edf", "events"), record["raw_files"]))):
            raise DeploymentError("exact_Ma_native_metadata_adapter_field_context_required")
        return {"kind": "Ma_recorded_adapter_identity", "body": data, "field": "adapter_code",
            "metadata": metadata, "canonical_code": canonical, "dependencies": [plan_ref, state_ref]}

    def register_episode_history(self, *, defer_walk=True):
        """Authenticate exact descriptor-consumer history, without data gates.

        Deferred registration deliberately does not recursively walk the final
        software inventory: its historical bridge descriptor needs the bridge
        alias to be registered first. Every current software input is still
        independently hashed here. No ledger, worker or scientific fit is run.
        """
        if self.episode_artifact_compatibility_history:
            raise DeploymentError("episode_software_history_already_registered")
        root = self.root
        base = _exact(root, _fixed(ORIGINAL_ENTRY, ORIGINAL_SHA, ORIGINAL_BYTES))
        adapter = _exact(root, descriptor(root, ENTRY))
        if ((root / ORIGINAL_ENTRY).read_bytes() != Path(original.__file__).read_bytes()
                or (root / ENTRY).read_bytes() != Path(__file__).read_bytes()):
            raise DeploymentError("executed_episode_history_builder_differs_from_root")
        if None in (CURRENT_EPISODE_SHA, CURRENT_EPISODE_BYTES,
                    CURRENT_PREDICTION_SHA, CURRENT_PREDICTION_BYTES):
            raise DeploymentError("descriptor_consumers_not_yet_frozen")
        old_episode = _fixed(EPISODE, OLD_EPISODE_SHA, OLD_EPISODE_BYTES)
        old_prediction = _fixed(PREDICTION, OLD_PREDICTION_SHA, OLD_PREDICTION_BYTES)
        old_test = _fixed(LEGACY_W3_TEST, OLD_W3_TEST_SHA, OLD_W3_TEST_BYTES)
        episode_snapshot = _exact(root, _fixed(OLD_EPISODE_SNAPSHOT, OLD_EPISODE_SHA, OLD_EPISODE_BYTES))
        prediction_snapshot = _exact(root, _fixed(OLD_PREDICTION_SNAPSHOT, OLD_PREDICTION_SHA, OLD_PREDICTION_BYTES))
        test_snapshot = _exact(root, _fixed(OLD_W3_TEST_SNAPSHOT, OLD_W3_TEST_SHA, OLD_W3_TEST_BYTES))
        current_episode = _exact(root, _fixed(EPISODE, CURRENT_EPISODE_SHA, CURRENT_EPISODE_BYTES))
        current_prediction = _exact(root, _fixed(PREDICTION, CURRENT_PREDICTION_SHA, CURRENT_PREDICTION_BYTES))
        original_manifest = _exact(root, descriptor(root, "audit/software_code_manifest.json"))
        rows117 = json.loads((root / original_manifest["path"]).read_bytes()).get("files", [])
        if (original_manifest["sha256"] != ORIGINAL_117_SHA
                or len(rows117) != 117 or len({r.get("path") for r in rows117}) != 117
                or [r for r in rows117 if r.get("path") == EPISODE] != [old_episode]
                or [r for r in rows117 if r.get("path") == PREDICTION] != [old_prediction]):
            raise DeploymentError("exact_117_historical_descriptor_consumer_rows_required")
        historical_inventory = _exact(root, _fixed(HISTORICAL_INVENTORY, HISTORICAL_INVENTORY_SHA, HISTORICAL_INVENTORY_BYTES))
        historical_result = _exact(root, _fixed(HISTORICAL_RESULT, HISTORICAL_RESULT_SHA, HISTORICAL_RESULT_BYTES))
        prior = json.loads((root / HISTORICAL_RESULT).read_bytes())
        prior_rows = json.loads((root / HISTORICAL_INVENTORY).read_bytes()).get("files", [])
        if (prior.get("status") != "PASS" or prior.get("actual_exit_code") != 0
                or prior.get("passed_tests") != 3350 or prior.get("input_files") != 189
                or prior.get("all_189_SHA_and_bytes_rechecked_after_actual_exit") is not True
                or prior.get("input_inventory") != historical_inventory
                or len(prior_rows) != 189 or len({r.get("path") for r in prior_rows}) != 189
                or any([r for r in prior_rows if r.get("path") == old["path"]] != [old]
                       for old in (old_episode, old_prediction, old_test))):
            raise DeploymentError("exact_189_historical_descriptor_consumer_and_fixture_rows_required")
        contract_ref = descriptor(root, EPISODE_CONTRACT)
        value = json.loads((root / EPISODE_CONTRACT).read_bytes())
        expected = {"schema_version": EPISODE_FIX_SCHEMA, "status": "SOFTWARE_ACCEPTED_DATA_GATES_REMAIN",
            "original_builder": base, "previous_episode": old_episode,
            "previous_episode_snapshot": episode_snapshot, "current_episode": current_episode,
            "previous_prediction_store": old_prediction, "previous_prediction_store_snapshot": prediction_snapshot,
            "current_prediction_store": current_prediction, "current_deploy_adapter": adapter,
            "original_117_manifest": original_manifest, "historical_189_input_inventory": historical_inventory,
            "historical_3350_acceptance": historical_result, "historical_tests_do_not_validate_new_runtime": True,
            "scientific_kernel_changed": False, "frozen_plan_changed": False,
            "raw_payloads_selected": 0, "network_operations_by_adapter": 0}
        if any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in expected.items()):
            raise DeploymentError("exact_declared_episode_software_fix_required")
        fixture = value.get("test_fixture_history")
        if (not isinstance(fixture, dict) or set(fixture) != {"previous_test", "previous_test_snapshot", "current_test"}
                or fixture.get("previous_test") != old_test or fixture.get("previous_test_snapshot") != test_snapshot):
            raise DeploymentError("exact_old_W3_software_fixture_history_required")
        current_test = _exact(root, fixture.get("current_test"))
        if current_test["path"] != LEGACY_W3_TEST:
            raise DeploymentError("exact_current_W3_fixture_path_required")
        required = {"current_episode": current_episode, "current_prediction_store": current_prediction,
            "deployment_adapter": adapter, "current_w3_auditor_legacy_tests": current_test}
        for key, path in (("current_w3_auditor", W3_AUDITOR), ("current_w3_auditor_tests", W3_AUDITOR_TESTS),
                          ("current_episode_tests", EPISODE_TESTS), ("current_prediction_tests", PREDICTION_TESTS)):
            required[key] = _exact(root, value.get(key))
            if required[key]["path"] != path:
                raise DeploymentError("exact_descriptor_consumer_verifier_and_test_paths_required")
        acceptance_ref = _exact(root, value.get("software_acceptance"))
        if Path(acceptance_ref["path"]).suffix != ".json":
            raise DeploymentError("software_acceptance_must_be_structured_JSON")
        acceptance = json.loads((root / acceptance_ref["path"]).read_bytes())
        proof = {"schema_version": ACCEPTANCE_SCHEMA, "status": "PASS", "actual_exit_code": 0,
            "all_input_SHA_and_bytes_rechecked_after_actual_exit": True,
            "scientific_fits": 0, "GPU_actions": 0, "RunPod_actions": 0, "network_operations": 0, **required}
        if (any(type(acceptance.get(k)) is not type(v) or acceptance.get(k) != v for k, v in proof.items())
                or type(acceptance.get("passed_tests")) is not int or acceptance["passed_tests"] < 3350):
            raise DeploymentError("actual_descriptor_consumer_full_software_acceptance_required")
        tested_ref = _exact(root, acceptance.get("input_inventory"))
        if Path(tested_ref["path"]).suffix != ".json":
            raise DeploymentError("software_input_inventory_must_be_structured_JSON")
        tested = json.loads((root / tested_ref["path"]).read_bytes()).get("files")
        if (not isinstance(tested, list) or len(tested) < 189
                or any(not isinstance(r, dict) or set(r) != {"path", "sha256", "bytes"} for r in tested)
                or len({r["path"] for r in tested}) != len(tested)
                or any(item not in tested for item in required.values())):
            raise DeploymentError("all_current_descriptor_consumers_and_tests_must_be_actual_suite_inputs")
        if {EPISODE_CONTRACT, FIX_CONTRACT, acceptance_ref["path"], tested_ref["path"]} & {r["path"] for r in tested}:
            raise DeploymentError("future_contract_or_acceptance_cannot_be_software_test_input")
        for row in tested:
            path = relative(root, row["path"])
            if (Path(path).suffix not in {".py", ".toml", ".lock"}
                    or original.classify(path) in {"raw_reference_only", "cache", "wheel", "software_runtime"}):
                raise DeploymentError("software_input_must_be_code_or_build_configuration", path)
            _exact(root, row)
        def history(old, snapshot, current):
            return {"historical_source_path": old["path"], "historical_source_sha256": old["sha256"],
                "historical_source_bytes": old["bytes"], "archive_snapshot": snapshot, "current_implementation": current}
        evidence = {**history(old_episode, episode_snapshot, current_episode),
            "prediction_store_history": history(old_prediction, prediction_snapshot, current_prediction),
            "test_fixture_history": history(old_test, test_snapshot, current_test),
            "software_fix_contract": contract_ref, "actual_new_software_acceptance": acceptance_ref,
            "original_117_manifest": original_manifest, "original_189_input_inventory": historical_inventory,
            "old_suite_does_not_validate_new_runtime": True,
            **{k: required[k] for k in ("current_w3_auditor", "current_w3_auditor_tests", "current_episode_tests", "current_prediction_tests")}}
        for old, snapshot in ((old_episode, episode_snapshot), (old_prediction, prediction_snapshot), (old_test, test_snapshot)):
            self.code_aliases[(old["path"], old["sha256"])] = snapshot
        self.episode_artifact_compatibility_history.append(evidence)
        self._pending_episode_history = [contract_ref, acceptance_ref, tested_ref, *tested]
        if not defer_walk:
            self.walk_episode_history()
        return evidence

    def walk_episode_history(self):
        if not self.episode_artifact_compatibility_history or self._pending_episode_history is None:
            raise DeploymentError("episode_software_history_must_be_registered_before_walk")
        try:
            for item in self._pending_episode_history:
                self.visit(item)
        except Exception:
            for source, sha in ((EPISODE, OLD_EPISODE_SHA), (PREDICTION, OLD_PREDICTION_SHA),
                                (LEGACY_W3_TEST, OLD_W3_TEST_SHA)):
                self.code_aliases.pop((source, sha), None)
            self.episode_artifact_compatibility_history.clear()
            self._pending_episode_history = None
            raise

    def register_bridge_history(self):
        """Register only the exact old bridge after real new-software acceptance.

        Old input inventories remain original JSON, not rewritten evidence.
        Their 4590 descriptor resolves to the byte-identical historical body;
        the current canonical bridge is independently selected at its new SHA.
        """
        if self.bridge_software_history:
            raise DeploymentError("bridge_software_history_already_registered")
        root = self.root
        base = _exact(root, _fixed(ORIGINAL_ENTRY, ORIGINAL_SHA, ORIGINAL_BYTES))
        if (root / ORIGINAL_ENTRY).read_bytes() != Path(original.__file__).read_bytes():
            raise DeploymentError("executed_original_builder_differs_from_root")
        adapter = descriptor(root, ENTRY)
        if (root / ENTRY).read_bytes() != Path(__file__).read_bytes():
            raise DeploymentError("executed_deploy_adapter_differs_from_root")
        if CURRENT_BRIDGE_SHA is None or CURRENT_BRIDGE_BYTES is None:
            raise DeploymentError("new_bridge_implementation_not_yet_frozen")
        current = _exact(root, _fixed(BRIDGE, CURRENT_BRIDGE_SHA, CURRENT_BRIDGE_BYTES))
        old = _fixed(BRIDGE, OLD_BRIDGE_SHA, OLD_BRIDGE_BYTES)
        snapshot = _exact(root, _fixed(OLD_BRIDGE_SNAPSHOT, OLD_BRIDGE_SHA, OLD_BRIDGE_BYTES))
        historical_result = _exact(root, _fixed(HISTORICAL_RESULT, HISTORICAL_RESULT_SHA, HISTORICAL_RESULT_BYTES))
        historical_inventory = _exact(root, _fixed(HISTORICAL_INVENTORY, HISTORICAL_INVENTORY_SHA, HISTORICAL_INVENTORY_BYTES))
        prior = json.loads((root / HISTORICAL_RESULT).read_bytes())
        inventory = json.loads((root / HISTORICAL_INVENTORY).read_bytes())
        prior_rows = inventory.get("files", [])
        if (prior.get("status") != "PASS" or prior.get("actual_exit_code") != 0
                or prior.get("passed_tests") != 3350 or prior.get("input_files") != 189
                or prior.get("all_189_SHA_and_bytes_rechecked_after_actual_exit") is not True
                or prior.get("input_inventory") != historical_inventory
                or len(prior_rows) != 189
                or [row for row in prior_rows if row.get("path") == BRIDGE] != [old]):
            raise DeploymentError("exact_historical_3350_and_189_software_proof_required")
        contract_ref = descriptor(root, FIX_CONTRACT)
        contract = json.loads((root / FIX_CONTRACT).read_bytes())
        expected = {"schema_version": FIX_SCHEMA, "status": "SOFTWARE_ACCEPTED_DATA_GATES_REMAIN",
            "original_builder": base, "previous_bridge": old, "previous_bridge_snapshot": snapshot,
            "current_bridge": current, "current_deploy_adapter": adapter,
            "historical_3350_acceptance": historical_result,
            "historical_189_input_inventory": historical_inventory,
            "historical_tests_do_not_validate_new_bridge": True,
            "scientific_kernel_changed": False, "frozen_plan_changed": False,
            "raw_payloads_selected": 0, "network_operations_by_adapter": 0}
        if any(type(contract.get(key)) is not type(value) or contract.get(key) != value
               for key, value in expected.items()):
            raise DeploymentError("exact_declared_bridge_software_fix_required")
        helper = _exact(root, contract.get("current_finalize_helper"))
        if helper["path"] != FINALIZE_HELPER:
            raise DeploymentError("versioned_local_finalize_helper_required")
        acceptance_ref = _exact(root, contract.get("software_acceptance"))
        if Path(acceptance_ref["path"]).suffix != ".json":
            raise DeploymentError("software_acceptance_must_be_structured_JSON")
        acceptance = json.loads((root / acceptance_ref["path"]).read_bytes())
        proof_expected = {"schema_version": ACCEPTANCE_SCHEMA, "status": "PASS",
            "current_bridge": current, "deployment_adapter": adapter,
            "current_finalize_helper": helper, "actual_exit_code": 0,
            "all_input_SHA_and_bytes_rechecked_after_actual_exit": True,
            "scientific_fits": 0, "GPU_actions": 0, "RunPod_actions": 0,
            "network_operations": 0}
        if (any(type(acceptance.get(key)) is not type(value) or acceptance.get(key) != value
                for key, value in proof_expected.items())
                or type(acceptance.get("passed_tests")) is not int or acceptance["passed_tests"] < 3350):
            raise DeploymentError("actual_new_full_software_suite_acceptance_required")
        tested_ref = _exact(root, acceptance.get("input_inventory"))
        if Path(tested_ref["path"]).suffix != ".json":
            raise DeploymentError("software_input_inventory_must_be_structured_JSON")
        tested = json.loads((root / tested_ref["path"]).read_bytes()).get("files")
        if (not isinstance(tested, list) or len(tested) < 189
                or any(not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"} for row in tested)
                or len({row["path"] for row in tested}) != len(tested)
                or any(item not in tested for item in (current, adapter, helper))):
            raise DeploymentError("actual_new_runtime_and_adapter_tested_input_inventory_required")
        if {FIX_CONTRACT, acceptance_ref["path"], tested_ref["path"]} & {row["path"] for row in tested}:
            raise DeploymentError("future_contract_or_acceptance_cannot_be_software_test_input")
        for row in tested:
            path = relative(root, row["path"])
            if (Path(path).suffix not in {".py", ".toml", ".lock"}
                    or original.classify(path) in {"raw_reference_only", "cache", "wheel", "software_runtime"}):
                raise DeploymentError("software_input_must_be_code_or_build_configuration", path)
            _exact(root, row)
        if (self.episode_artifact_compatibility_history
                and self.episode_artifact_compatibility_history[0]["actual_new_software_acceptance"] != acceptance_ref):
            raise DeploymentError("one_actual_full_suite_must_bind_both_software_history_gates")
        self.code_aliases[(BRIDGE, OLD_BRIDGE_SHA)] = snapshot
        evidence = {"historical_source_path": BRIDGE, "historical_source_sha256": OLD_BRIDGE_SHA,
            "historical_source_bytes": OLD_BRIDGE_BYTES, "archive_snapshot": snapshot,
            "current_implementation": current, "software_fix_contract": contract_ref,
            "actual_new_software_acceptance": acceptance_ref,
            "original_3350_acceptance": historical_result, "original_189_input_inventory": historical_inventory,
            "old_suite_does_not_validate_new_runtime": True}
        self.bridge_software_history.append(evidence)
        self.adapter_evidence = {"implementation": adapter, "original_builder": base,
            "version": 2, "new_software_history_registration": True,
            "original_builder_source_unchanged": True, "original_data_gates_reused": True,
            "no_original_module_globals_replaced": True}
        # Every proof, full tested input list, helper and old alias body enters
        # ordinary exact-SHA traversal after the alias is fully authorized.
        try:
            self.visit(contract_ref)
            self.visit(acceptance_ref)
            self.visit(tested_ref)
            for row in tested:
                self.visit(row)
            self.visit(adapter, role="code")
        except Exception:
            self.code_aliases.pop((BRIDGE, OLD_BRIDGE_SHA), None)
            self.bridge_software_history.clear()
            self.adapter_evidence = None
            raise
        return evidence


def final_manifest(report):
    evidence = report.get("deployment_adapter")
    aliases = report.get("historical_software_code_aliases")
    body = Path(__file__).read_bytes()
    expected_adapter = {"implementation": _fixed(ENTRY, hashlib.sha256(body).hexdigest(), len(body)),
        "original_builder": _fixed(ORIGINAL_ENTRY, ORIGINAL_SHA, ORIGINAL_BYTES), "version": 2,
        "new_software_history_registration": True, "original_builder_source_unchanged": True,
        "original_data_gates_reused": True, "no_original_module_globals_replaced": True}
    expected_alias = {"historical_source_path": BRIDGE, "historical_source_sha256": OLD_BRIDGE_SHA,
        "historical_source_bytes": OLD_BRIDGE_BYTES,
        "archive_snapshot": _fixed(OLD_BRIDGE_SNAPSHOT, OLD_BRIDGE_SHA, OLD_BRIDGE_BYTES),
        "current_implementation": _fixed(BRIDGE, CURRENT_BRIDGE_SHA, CURRENT_BRIDGE_BYTES),
        "old_suite_does_not_validate_new_runtime": True}
    if (CURRENT_BRIDGE_SHA is None or CURRENT_BRIDGE_BYTES is None
            or evidence != expected_adapter or not isinstance(aliases, list) or len(aliases) != 1
            or not isinstance(aliases[0], dict)
            or any(aliases[0].get(key) != value for key, value in expected_alias.items())
            or aliases[0].get("software_fix_contract", {}).get("path") != FIX_CONTRACT):
        raise DeploymentError("versioned_software_alias_evidence_required")
    episode_history = report.get("episode_artifact_compatibility_history")
    if (not isinstance(episode_history, list) or len(episode_history) != 1
            or not isinstance(episode_history[0], dict)):
        raise DeploymentError("descriptor_consumer_compatibility_evidence_required")
    history = episode_history[0]
    expected_histories = ((history, EPISODE, OLD_EPISODE_SHA, OLD_EPISODE_BYTES,
        OLD_EPISODE_SNAPSHOT, CURRENT_EPISODE_SHA, CURRENT_EPISODE_BYTES),
        (history.get("prediction_store_history", {}), PREDICTION, OLD_PREDICTION_SHA, OLD_PREDICTION_BYTES,
         OLD_PREDICTION_SNAPSHOT, CURRENT_PREDICTION_SHA, CURRENT_PREDICTION_BYTES),
        (history.get("test_fixture_history", {}), LEGACY_W3_TEST, OLD_W3_TEST_SHA, OLD_W3_TEST_BYTES,
         OLD_W3_TEST_SNAPSHOT, None, None))
    for item, source, sha, size, snapshot, current_sha, current_size in expected_histories:
        expected = {"historical_source_path": source, "historical_source_sha256": sha,
            "historical_source_bytes": size, "archive_snapshot": _fixed(snapshot, sha, size)}
        if (not isinstance(item, dict) or any(item.get(k) != v for k, v in expected.items())
                or item.get("current_implementation", {}).get("path") != source
                or source != LEGACY_W3_TEST and (current_sha is None or current_size is None
                    or item.get("current_implementation") != _fixed(source, current_sha, current_size))):
            raise DeploymentError("descriptor_consumer_compatibility_identity_mismatch")
    if (history.get("software_fix_contract", {}).get("path") != EPISODE_CONTRACT
            or history.get("old_suite_does_not_validate_new_runtime") is not True
            or history.get("actual_new_software_acceptance") != aliases[0].get("actual_new_software_acceptance")):
        raise DeploymentError("one_actual_full_suite_must_bind_both_software_history_gates")
    result = original.final_manifest(report)
    result["deployment_adapter"] = evidence
    result["historical_software_code_aliases"] = aliases
    result["historical_provider_reference_contexts"] = report.get("historical_provider_reference_contexts", [])
    result["episode_artifact_compatibility_history"] = episode_history
    return result



def inspect(root: Path, *, manifests=None, include_software=True, include_wheels=True, history=None) -> dict:
    root = Path(os.path.abspath(root))
    closure = Closure(root)
    if include_software:
        try:
            closure.register_precision_fix()
        except Exception as exc:
            closure.error(str(exc) if isinstance(exc,DeploymentError) else "classical_precision_history_unverified",
                          getattr(exc,"path",None) or PRECISION_AUDIT,"software")
    if include_software:
        try:
            closure.register_episode_history(defer_walk=True)
        except Exception as exc:
            closure.error(str(exc) if isinstance(exc, DeploymentError) else "episode_software_history_unverified",
                          getattr(exc, "path", None) or EPISODE_CONTRACT, "software")
        try:
            closure.register_bridge_history()
        except Exception as exc:
            closure.error(str(exc) if isinstance(exc, DeploymentError) else "bridge_software_history_unverified",
                          getattr(exc, "path", None) or FIX_CONTRACT, "software")
        if closure.episode_artifact_compatibility_history:
            try:
                closure.walk_episode_history()
            except Exception as exc:
                closure.error(str(exc) if isinstance(exc, DeploymentError) else "episode_software_history_dependency_unverified",
                              getattr(exc, "path", None) or EPISODE_CONTRACT, "software")
    history = ([p.relative_to(root).as_posix() for p in sorted(
        p for base in ("audit/data_plan_20261019","audit/data_plan_20261020")
        for p in (root/base).rglob("budget_conservation_amendment.json"))]
        if history is None else history)
    for path in history:
        try:
            closure.load_history(path, defer_walk=True)
        except Exception as exc:
            closure.error(str(exc) if isinstance(exc, DeploymentError) else "historical_budget_or_code_snapshot_unverified",
                          getattr(exc, "path", None) or path, "history")
    # A repaired plan can itself become a stopped parent. Register every
    # (original path, historical SHA) binding before traversing any chain;
    # file-name ordering must not choose the active parser for an old receipt.
    for item, old in closure.pending_history:
        try:
            closure.walk_history(item, old)
        except Exception as exc:
            closure.error(str(exc) if isinstance(exc, DeploymentError) else "historical_dependency_unverified",
                          getattr(exc, "path", None) or item["path"], "history")
    if include_software:
        for transaction in sorted((root/"audit/data_plan_20261020/runtime_recovery").glob("*/transaction.json")):
            try:
                closure.load_runtime_recovery(transaction.relative_to(root).as_posix())
            except Exception as exc:
                closure.error(str(exc) if isinstance(exc,DeploymentError) else "CPU_runtime_recovery_evidence_unverified",
                              getattr(exc,"path",None) or transaction.relative_to(root).as_posix(),"runtime_recovery")
        for history_item in closure.precision_history:
            try:
                closure.walk(history_item)
            except Exception as exc:
                closure.error(str(exc) if isinstance(exc,DeploymentError) else "classical_precision_dependency_unverified",
                              getattr(exc,"path",None) or PRECISION_AUDIT,"software")
    manifests = default_manifests(root) if manifests is None else manifests
    if set(manifests) != set(COMPONENTS):
        raise DeploymentError("exact_five_required_dataset_components")
    components = {}
    for name in COMPONENTS:
        path = manifests[name]
        try:
            if path is None:
                raise DeploymentError("unique_COMPLETE_manifest_not_yet_available")
            path = relative(root, path)
            if not (root/path).is_file():
                raise DeploymentError("COMPLETE_manifest_not_yet_available")
            _, receipts = complete_component(root, name, path)
            closure.path(path, context=name)
            for receipt in receipts:
                closure.path(receipt, context=name)
            components[name] = {"status": "COMPLETE_LOCAL_SHA_AND_R2_EVIDENCE_VERIFIED", "manifest": descriptor(root, path)}
        except Exception as exc:
            code = str(exc) if isinstance(exc, DeploymentError) else "dependency_missing_or_unverified_details_suppressed"
            components[name] = {"status": "IN_PROGRESS_OR_BLOCKED", "manifest_path": path, "gap": code}
            closure.error(code, getattr(exc, "path", None) or path, name)
    if include_software:
        names = [p.relative_to(root).as_posix() for tree in ("p3work", "tools") for p in (root/tree).rglob("*.py")]
        names += ["requirements-cpu.lock", "requirements-gpu.lock", "pyproject.toml", "contracts/input_contract.json",
                  "contracts/work_contract.json", "status/tasks.json", "plans/controlled_pilot.json"]
        names += [p.relative_to(root).as_posix() for p in (root/"contracts").glob("*.json")]
        try:
            closure.original_protocol()
            tasks = json.loads((root/"status/tasks.json").read_bytes())
            for key in ("W0", "W1", "W2", "W3"):
                names += tasks.get(key, {}).get("evidence", [])
            w2 = tasks.get("W2", {})
            catalog = w2.get("raw_r2_receipts_catalog")
            if catalog:
                closure.visit({"path": catalog, "sha256": w2["raw_r2_receipts_catalog_sha256"]})
        except Exception as exc:
            closure.error(str(exc) if isinstance(exc, DeploymentError) else "original_protocol_or_runtime_gate_evidence_unverified", context="software")
        for path in sorted(set(names)):
            try:
                closure.path(path)
            except Exception as exc:
                closure.error(str(exc) if isinstance(exc, DeploymentError) else "runtime_dependency_unverified", path, "software")
        for path in ("manifests/geometry_constants/manifest.json",
                     "audit/external_expansion_20261019/round2/car3_preparation/R2_receipt.json"):
            try:
                closure.path(path)
                if path.endswith("geometry_constants/manifest.json"):
                    geometry = json.loads((root/path).read_bytes())
                    for item in geometry["files"]:
                        closure.visit({"path": item["path"], "sha256": item["file_sha256"]}, role="protocol")
            except Exception as exc:
                closure.error(str(exc) if isinstance(exc, DeploymentError) else "fixed_geometry_or_CAR3_cache_evidence_unverified",
                              getattr(exc, "path", None) or path, "software")
        try:
            portable = closure.portable_runtime()
        except Exception as exc:
            portable = {"status":"PENDING_PORTABLE_RUNTIME_EVIDENCE","GPU_compatibility_verified":False}
            closure.error(str(exc) if isinstance(exc, DeploymentError) else "portable_runtime_evidence_unverified",
                          getattr(exc,"path",None) or RUNTIME_CONTRACT,"software")
    else:
        portable = {"status":"NOT_SELECTED","GPU_compatibility_verified":False}
    if include_wheels:
        for path in DEFAULT_WHEELS:
            try:
                closure.path(path, receipt_profile="baseline")
            except Exception as exc:
                closure.error(str(exc) if isinstance(exc, DeploymentError) else "offline_wheel_receipt_or_payload_unverified", path, "wheels")
        try:
            from tools.build_delivery import validate_dependency_backups
            validate_dependency_backups(root)
        except Exception:
            closure.error("exact_offline_CPU_GPU_requirement_pins_not_fully_receipt_covered", context="wheels")
    # Every selected path remains an individually hashed file. Raw receipt rows
    # are retained for provenance but never become an implicit all-approved GET.
    closure.check_stable_snapshot()
    files = sorted(closure.files.values(), key=lambda r: r["path"])
    covered = sum((r["path"], r["sha256"], r["bytes"]) in closure.r2_rows for r in files)
    ready = not closure.errors and all(c["status"] == "COMPLETE_LOCAL_SHA_AND_R2_EVIDENCE_VERIFIED" for c in components.values())
    result = {"schema_version": SCHEMA, "status": "CACHE_CLOSURE_READY_FOR_BACKUP" if ready else "IN_PROGRESS_NO_FINAL_DEPLOY_MANIFEST",
        "all_required_data_COMPLETE": ready, "components": components, "gaps": closure.errors,
        "files": files, "file_count": len(files), "bytes": sum(r["bytes"] for r in files),
        "verified_restore_groups": sorted(closure.receipts.values(), key=lambda r: r["receipt"]["path"]),
        "deployment_adapter": closure.adapter_evidence,
        "historical_software_code_aliases": closure.bridge_software_history,
        "historical_provider_reference_contexts": closure.provider_reference_contexts,
        "episode_artifact_compatibility_history": closure.episode_artifact_compatibility_history,
        "historical_budget_amendments": closure.history,
        "classical_precision_history": closure.precision_history,
        "runtime_recovery_history": closure.runtime_recovery_history,
        "historical_nonruntime_comparisons": closure.historical_nonruntime,
        "portable_runtime": portable,
        "files_already_covered_by_individual_R2_GET_receipts": covered,
        "raw_proof_objects": len(closure.raw_get_backed), "raw_reference_objects": len(closure.raw), "raw_payloads_selected": 0,
        "original_protocol_manifest_members_verified": len(closure.protocol_members),
        "raw_reference_bytes_not_selected": sum(r["bytes"] for r in closure.raw.values() if type(r.get("bytes")) is int),
        "network_operations": 0, "backup_executed": False, "fresh_restore_executed": False,
        "training_started": False, "GPU_actions": 0, "external_query_opened": False,
        "Ma2022_scientific_training_ready": False, "user_manual_GPU_boot_required": True}
    result["closure_sha256"] = digest(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifests", type=Path, help="Optional exact five-name mapping to root-relative COMPLETE manifests")
    parser.add_argument("--report", type=Path, help="Optional immutable local readiness snapshot")
    parser.add_argument("--history", action="append", help="Explicit historical budget-conservation amendment; otherwise discover only the known repair folders")
    parser.add_argument("--output-dir", type=Path, help="Write final local backup manifests only after all data gates pass")
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.manifests.read_bytes()) if args.manifests else None
        report = inspect(args.root, manifests=mapping, history=args.history)
        if args.report:
            write_once(args.report, report)
        if args.output_dir:
            final = final_manifest(report)
            write_once(args.output_dir/"deploy_manifest.json", final)
            for name, batch in final["backup_manifests"].items():
                write_once(args.output_dir/(name+"_backup_manifest.json"), batch)
        print(json.dumps({k: report[k] for k in ("status", "closure_sha256", "components", "gaps", "file_count", "bytes",
            "raw_proof_objects", "raw_payloads_selected", "network_operations", "training_started", "Ma2022_scientific_training_ready")},
            indent=2, ensure_ascii=False))
        return 0 if report["all_required_data_COMPLETE"] else 2
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(exc).__name__,
            "error_code": str(exc) if isinstance(exc, DeploymentError) else "local_closure_validation_failed_details_suppressed",
            "network_operations": 0, "training_started": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
