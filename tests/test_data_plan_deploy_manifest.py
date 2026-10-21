"""Cache-only closure, profile separation and false-readiness prevention tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from p3work import source_extension_io as extension
from tools import build_data_plan_deploy_manifest as deploy


class DeploymentManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, value):
        target = self.root/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(json.dumps(value).encode() if not isinstance(value, bytes) else value)
        return deploy.descriptor(self.root, path)

    def entry(self, item, *, dataset="BNCI2014_001", role="result"):
        return {**item, "dataset": dataset, "role": role}

    def receipt(self, path, rows, *, new_profile=False):
        prefix = "paper3/test-deployment/"
        objects = []
        for row in rows:
            objects.append({**row, "key": prefix+row["path"], "status": "verified",
                "put": {"status": "created", "http_status": 200,
                        "request_body_bytes": row["bytes"], "request_body_sha256": row["sha256"]},
                "get": {"status": "verified", "http_status": 200, "complete_body_read": True,
                        "bytes_read": row["bytes"], "sha256": row["sha256"], "declared_content_length": row["bytes"]}})
        value = {"schema_version": 1, "operation": "paper3_explicit_manifest_backup", "status": "verified",
                 "execute": True, "verification": "full_get_body_sha256", "prefix": prefix, "objects": objects}
        if new_profile:
            value["source_extension_profile"] = extension.profile()
        return self.write(path, value)

    def test_hash_bound_recursive_closure_includes_native_code_and_contract(self):
        code = self.write("p3work/native.py", b"# frozen code\n")
        contract = self.write("contracts/example.json", {"reference": "CAR"})
        native = self.write("audit/native.json", {"adapter_code": code, "contract": contract})
        manifest = self.write("cache/source_v2/source_cache.json", {"native_metadata": native})
        closure = deploy.Closure(self.root)
        closure.visit(manifest)
        self.assertEqual(set(closure.files), {r["path"] for r in (code, contract, native, manifest)})

    def test_explicit_test_code_in_existing_preparation_receipt_is_preserved(self):
        code = self.write("tests/test_car3_cache.py", b"# historical mathematical verification\n")
        item = self.receipt("audit/preparation_receipt.json", [self.entry(code, dataset="Paper3", role="code")])
        closure = deploy.Closure(self.root)
        closure.visit(item)
        self.assertEqual(closure.files[code["path"]]["role"], "code")

    def test_changed_nested_evidence_is_not_accepted(self):
        item = self.write("audit/native.json", {"label": 0})
        manifest = self.write("cache/source_v2/source_cache.json", {"native": item})
        self.write("audit/native.json", {"label": 1})
        with self.assertRaisesRegex(deploy.DeploymentError, "dependency_file_SHA_or_bytes_mismatch"):
            deploy.Closure(self.root).visit(manifest)

    def test_late_file_change_invalidates_snapshot_after_it_was_hashed(self):
        item = self.write("audit/native.json", {"label": 0})
        closure = deploy.Closure(self.root)
        closure.visit(item)
        self.write("audit/native.json", {"label": 1})
        closure.check_stable_snapshot()
        self.assertEqual(closure.errors[0]["code"], "dependency_changed_during_closure_snapshot")

    def test_raw_descriptors_do_not_require_or_read_local_raw(self):
        raw = {"path": "data/source_preparation/staging/native.mat", "sha256": "a"*64, "bytes": 10**12}
        manifest = self.write("cache/source_v2/source_cache.json", {"original": raw})
        closure = deploy.Closure(self.root)
        closure.visit(manifest)
        self.assertEqual(len(closure.raw), 1)
        self.assertEqual(len(closure.raw_get_backed), 0)
        self.assertNotIn(raw["path"], closure.files)

    def test_mixed_raw_receipt_generates_only_exact_cache_metadata_paths(self):
        native = self.write("audit/native_metadata.json", {"native_original_trials": 40})
        raw = {"path": "data/source_preparation/staging/large.mat", "sha256": "b"*64, "bytes": 100000000000}
        receipt = self.receipt("audit/backup_receipt.json", [self.entry(raw, role="source_development_raw"),
                                                             self.entry(native, role="audit")])
        closure = deploy.Closure(self.root)
        closure.visit(receipt)
        group = closure.receipts[receipt["path"]]
        self.assertEqual(group["io_profile"], "baseline")
        self.assertEqual(group["paths"], [native["path"]])
        self.assertEqual(group["excluded_raw_paths"], [raw["path"]])
        self.assertNotIn(raw["path"], closure.files)

    def test_raw_cannot_be_laundered_by_archive_receipt_role(self):
        raw = {"path": "data/unknown/sealed.edf", "sha256": "b"*64, "bytes": 999999999}
        receipt = self.receipt("audit/receipt.json", [self.entry(raw, dataset="Paper3", role="archive")])
        closure = deploy.Closure(self.root)
        closure.visit(receipt)
        self.assertEqual(closure.receipts[receipt["path"]]["paths"], [])
        self.assertEqual(len(closure.raw), 1)
        self.assertFalse(closure.receipts[receipt["path"]]["restore_required"])

    def test_source_extension_profile_is_kept_separate_and_validated(self):
        cache = self.write("cache/source_extensions_20261019/Dreyer2023/abc/record/X.npy", b"array bytes")
        receipt = self.receipt("audit/source_extensions_20261019/Dreyer2023/abc/cache_receipt.json",
            [self.entry(cache, dataset="Dreyer2023")], new_profile=True)
        closure = deploy.Closure(self.root)
        closure.visit(receipt)
        self.assertEqual(closure.receipts[receipt["path"]]["io_profile"], "source_extension")
        self.assertEqual(closure.files[cache["path"]]["dataset"], "Dreyer2023")
        self.assertEqual(closure.files[cache["path"]]["io_profile"], "source_extension")

    def test_extension_audit_directory_cannot_admit_baseline_receipt(self):
        cache = self.write("cache/source_extensions_20261019/Yang2025/abc/record/X.npy", b"cache")
        receipt = self.receipt("audit/source_extensions_20261019/Yang2025/abc/cache_receipt.json",
            [self.entry(cache, dataset="Yang2025")], new_profile=False)
        with self.assertRaisesRegex(deploy.DeploymentError, "backup_receipt_uses_wrong_IO_profile"):
            deploy.Closure(self.root).visit(receipt)

    def test_tampered_doc_profile_or_incomplete_get_cannot_enter_closure(self):
        cache = self.write("cache/source_extensions_20261019/Ma2022/native_v2/X.npy", b"cache")
        path = "audit/source_extensions_20261019/Ma2022/a/cache_receipt.json"
        item = self.receipt(path, [self.entry(cache, dataset="Ma2022")], new_profile=True)
        value = json.loads((self.root/path).read_bytes())
        for broken in ("profile", "GET"):
            changed = copy.deepcopy(value)
            if broken == "profile":
                changed["source_extension_profile"]["document"]["sha256"] = "0"*64
            else:
                changed["objects"][0]["get"]["complete_body_read"] = False
            with self.subTest(broken=broken):
                updated = self.write(path, changed)
                with self.assertRaises(Exception):
                    deploy.Closure(self.root).visit(updated)

    def test_raw_roots_legacy_npz_predictions_and_foreign_configs_rejected(self):
        for path in ("cache/source_v2/legacy.npz", "predictions/query.json", "paper1/config.toml",
                     "paper2/config.json", "data/dependency_wheels/raw.npy", "cache/external/query.npy"):
            with self.subTest(path=path):
                with self.assertRaises(deploy.DeploymentError):
                    deploy.classify(path)
        with self.assertRaisesRegex(deploy.DeploymentError, "raw_payload_cannot_be_a_deployment_root"):
            deploy.Closure(self.root).path("data/raw/source.edf")

    def test_credentials_symlinks_and_traversal_cannot_be_dependencies(self):
        for path in ("../outside.json", "audit//a.json", "audit/./a.json", "audit/token.json", ".aws/credentials"):
            with self.subTest(path=path):
                with self.assertRaises(Exception):
                    deploy.relative(self.root, path)
        (self.root/"audit").mkdir()
        (self.root/"link").symlink_to(self.root/"audit", target_is_directory=True)
        with self.assertRaises(Exception):
            deploy.relative(self.root, "link/a.json")

    def test_absolute_retained_metadata_paths_are_normalized_inside_current_root_only(self):
        item = self.write("audit/native.json", {"retained_trials": 99})
        absolute = dict(item, path=str(self.root/item["path"]))
        closure = deploy.Closure(self.root)
        closure.visit(absolute)
        self.assertEqual(list(closure.files), [item["path"]])
        with self.assertRaisesRegex(deploy.DeploymentError, "absolute_dependency_outside_current_root"):
            deploy.relative(self.root, "/another-host/audit/file.json")

    def test_Ma_producer_paths_relocate_without_reading_original_host_or_raw(self):
        target = "cache/source_extensions_20261019/Ma2022/development_car3_native_v2/sub-001/ses-01/X.npy"
        item = self.write(target, b"restored Ma array bytes")
        absolute = dict(item, path="/workspace/research3/"+target)
        raw = {"path":"/workspace/research3/data/source_extensions_20261019/Ma2022/native.edf",
               "sha256":"a"*64,"bytes":9000000}
        manifest = self.write("cache/source_extensions_20261019/Ma2022/development_car3_native_v2/sub-001/ses-01/manifest.json",
                              {"artifacts":{"X.npy":absolute},"identity":{"edf":raw}})
        closure = deploy.Closure(self.root)
        closure.visit(manifest, context="Ma2022")
        self.assertIn(target, closure.files)
        self.assertEqual(len(closure.raw), 1)
        self.assertEqual(next(iter(closure.raw.values()))["path"], raw["path"].removeprefix("/workspace/research3/"))
        for bad in ("/workspace/research3/audit/foreign.json",
                    "/workspace/research3/cache/source_extensions_20261019/Dreyer2023/X.npy",
                    "/workspace/research3/cache/source_extensions_20261019/Ma2022/../Yang2025/X.npy"):
            with self.subTest(path=bad), self.assertRaises(deploy.DeploymentError):
                deploy.relative(self.root,bad)

    def test_exact_CAR3_prior_comparisons_are_preserved_without_old_archive_payloads(self):
        array = self.write("cache/source_c3czc4_v1/BNCI2014_001/1/X.npy", b"CAR3")
        prior = [{"path":p,"sha256":s,"bytes":n,"sidecar_matched":True}
                 for p,(s,n) in deploy.PRIOR_ARCHIVES.items()]
        item = self.write(deploy.CAR3_REREAD,{"schema_version":"paper3-car3-independent-output-reread-v1",
                                            "prior_archives_unchanged":prior,"current_array":array})
        closure = deploy.Closure(self.root)
        closure.visit(item)
        self.assertIn(array["path"],closure.files)
        self.assertEqual(len(closure.historical_nonruntime),2)
        self.assertFalse(any(p.startswith("/workspace/shared") for p in closure.files))
        changed=copy.deepcopy(prior);changed[0]["sha256"]="f"*64
        item=self.write(deploy.CAR3_REREAD,{"schema_version":"paper3-car3-independent-output-reread-v1",
                                         "prior_archives_unchanged":changed})
        with self.assertRaisesRegex(deploy.DeploymentError,"historical_delivery_comparison_identity_changed"):
            deploy.Closure(self.root).visit(item)

    def test_historical_comparison_exception_cannot_hide_arbitrary_absolute_dependency(self):
        prior=[{"path":p,"sha256":s,"bytes":n,"sidecar_matched":True}
               for p,(s,n) in deploy.PRIOR_ARCHIVES.items()]
        item=self.write(deploy.CAR3_REREAD,{"schema_version":"paper3-car3-independent-output-reread-v1",
            "prior_archives_unchanged":prior,"current_array":{"path":"/untrusted/array.npy","sha256":"b"*64}})
        with self.assertRaisesRegex(deploy.DeploymentError,"absolute_dependency_outside_current_root"):
            deploy.Closure(self.root).visit(item)
        other=self.write("audit/other.json",{"prior_archives_unchanged":prior})
        with self.assertRaisesRegex(deploy.DeploymentError,"absolute_dependency_outside_current_root"):
            deploy.Closure(self.root).visit(other)

    def test_runtime_allowlist_is_exact_and_checks_archive_bytes(self):
        self.assertEqual(deploy.classify(deploy.RUNTIME_PATH),"software_runtime")
        for other in ("runtime/native.edf","runtime/other.tar.gz","runtime/python/bin/python3.12"):
            with self.subTest(path=other),self.assertRaisesRegex(deploy.DeploymentError,"only_exact_pinned"):
                deploy.classify(other)
        fake=self.write(deploy.RUNTIME_PATH,b"forged runtime")
        with self.assertRaisesRegex(deploy.DeploymentError,"portable_python_archive_full_SHA_or_bytes_mismatch"):
            deploy.Closure(self.root).visit(fake)

    def portable_runtime_fixture(self):
        archive=self.write(deploy.RUNTIME_PATH,b"SYNTHETIC SOFTWARE CONTRACT FIXTURE")
        requirements=self.write("requirements-gpu.lock",b"synthetic-fixed-profile==1\n")
        provider=self.write("audit/runtime_provider.json",{**archive,
            "provider_GitHub_asset_digest":"sha256:"+archive["sha256"],
            "full_provider_GET_and_SHA_verified":True,"archive_path_validation_passed":True,
            "license_files_preserved_inside_archive":["python/LICENSE"]})
        acceptance_value={"schema_version":"paper3-portable-python-cpu-acceptance-v1",
            "status":"PORTABLE_PYTHON_FULL_GPU_PROFILE_OFFLINE_INSTALL_AND_CPU_CHECK_COMPLETE",
            "archive":archive,"interpreter_version":"3.12.15","requirements":requirements,
            "exact_requirement_pins":100,"installed_requested_packages":100,"available_fixed_wheels":110,
            "all_requirement_versions_match":True,"all_install_sources_local_file":True,
            "GPU_checked":False,"GPU_compatibility_verified":False,"actual_scientific_fits":0,
            "network_operations":0,"R2_operations":0,"temporary_venv_and_extracted_runtime_removed":True,
            "commands":[{"returncode":0}]}
        acceptance=self.write("audit/runtime_CPU_acceptance.json",acceptance_value)
        contract={"schema_version":"paper3-portable-python-312-v1","runtime_archive":archive,
            "implementation":"CPython","python_version":"3.12.15","platform":"x86_64-unknown-linux-gnu",
            "GPU_checked":False,"GPU_compatibility_verified":False,"driver_compatibility_verified":False,
            "actual_scientific_fits":0,"requires_user_manual_GPU_boot":True,
            "provider_download_receipt":provider,"offline_CPU_acceptance":acceptance,"dependency_profile":requirements}
        self.write(deploy.RUNTIME_CONTRACT,contract)
        return archive,contract,acceptance_value

    def test_portable_runtime_requires_bound_offline_profile_and_keeps_GPU_pending(self):
        archive,_,_=self.portable_runtime_fixture()
        with patch.multiple(deploy,RUNTIME_SHA=archive["sha256"],RUNTIME_BYTES=archive["bytes"]):
            closure=deploy.Closure(self.root)
            report=closure.portable_runtime()
            self.assertFalse(report["GPU_compatibility_verified"])
            self.assertEqual(closure.files[deploy.RUNTIME_PATH]["role"],"archive")
            self.assertEqual(closure.files[deploy.RUNTIME_PATH]["dataset"],"PythonRuntime312")

    def test_portable_runtime_cannot_claim_GPU_or_fit_from_CPU_install(self):
        for key,bad in (("GPU_checked",True),("GPU_compatibility_verified",True),
                        ("actual_scientific_fits",1),("all_install_sources_local_file",False)):
            archive,contract,value=self.portable_runtime_fixture()
            value[key]=bad
            contract["offline_CPU_acceptance"]=self.write(contract["offline_CPU_acceptance"]["path"],value)
            self.write(deploy.RUNTIME_CONTRACT,contract)
            with self.subTest(key=key),patch.multiple(deploy,RUNTIME_SHA=archive["sha256"],RUNTIME_BYTES=archive["bytes"]):
                with self.assertRaisesRegex(deploy.DeploymentError,"full_offline_CPU_acceptance_required"):
                    deploy.Closure(self.root).portable_runtime()

    def test_portable_runtime_install_failure_is_not_hidden_by_nominal_status(self):
        archive,contract,value=self.portable_runtime_fixture()
        value["commands"][0]["returncode"]=1
        contract["offline_CPU_acceptance"]=self.write(contract["offline_CPU_acceptance"]["path"],value)
        self.write(deploy.RUNTIME_CONTRACT,contract)
        with patch.multiple(deploy,RUNTIME_SHA=archive["sha256"],RUNTIME_BYTES=archive["bytes"]):
            with self.assertRaisesRegex(deploy.DeploymentError,"full_offline_CPU_acceptance_required"):
                deploy.Closure(self.root).portable_runtime()

    def test_portable_runtime_profile_change_after_acceptance_fails_SHA(self):
        archive,_,_=self.portable_runtime_fixture()
        self.write("requirements-gpu.lock",b"different-profile==2\n")
        with patch.multiple(deploy,RUNTIME_SHA=archive["sha256"],RUNTIME_BYTES=archive["bytes"]):
            with self.assertRaisesRegex(deploy.DeploymentError,"actual_evidence_SHA_required"):
                deploy.Closure(self.root).portable_runtime()

    def test_missing_in_progress_data_produces_report_but_no_final_manifest(self):
        mapping = dict.fromkeys(deploy.COMPONENTS, None)
        report = deploy.inspect(self.root, manifests=mapping, include_software=False, include_wheels=False, history=[])
        self.assertEqual(report["status"], "IN_PROGRESS_NO_FINAL_DEPLOY_MANIFEST")
        self.assertFalse(report["all_required_data_COMPLETE"])
        self.assertEqual(report["network_operations"], 0)
        self.assertEqual(len(report["gaps"]), 5)
        with self.assertRaisesRegex(deploy.DeploymentError, "all_actual_COMPLETE_data_and_verified_closure_required"):
            deploy.final_manifest(report)

    def test_final_manifest_cannot_silently_change_report(self):
        report = {"status": "CACHE_CLOSURE_READY_FOR_BACKUP", "all_required_data_COMPLETE": True,
                  "gaps": [], "raw_payloads_selected": 0, "files": [], "verified_restore_groups": []}
        report["closure_sha256"] = deploy.digest(report)
        report["files"] = [{"path": "data/source/file.edf"}]
        with self.assertRaisesRegex(deploy.DeploymentError, "readiness_report_SHA_mismatch"):
            deploy.final_manifest(report)

    def test_final_batches_preserve_exact_dataset_and_IO_profile(self):
        baseline = self.write("cache/source_v2/BNCI2014_001/1/X.npy", b"oldsource")
        new = self.write("cache/source_extensions_20261019/Yang2025/a/X.npy", b"newsource")
        rows = [{**self.entry(baseline), "io_profile": "baseline"},
                {**self.entry(new, dataset="Yang2025"), "io_profile": "source_extension"}]
        report = {"status": "CACHE_CLOSURE_READY_FOR_BACKUP", "all_required_data_COMPLETE": True,
                  "gaps": [], "raw_payloads_selected": 0, "files": rows, "verified_restore_groups": []}
        report["closure_sha256"] = deploy.digest(report)
        final = deploy.final_manifest(report)
        self.assertEqual(final["backup_manifests"]["baseline"]["files"][0]["dataset"], "BNCI2014_001")
        self.assertEqual(final["backup_manifests"]["source_extension"]["files"][0]["dataset"], "Yang2025")
        self.assertFalse(final["fresh_restore_executed"])
        self.assertFalse(final["Ma2022_scientific_training_ready"])

    def test_any_raw_row_cannot_enter_final_even_with_newly_recomputed_report_hash(self):
        report = {"status": "CACHE_CLOSURE_READY_FOR_BACKUP", "all_required_data_COMPLETE": True,
            "gaps": [], "raw_payloads_selected": 0, "verified_restore_groups": [],
            "files": [{"path": "data/source/large.mat", "sha256": "a"*64, "bytes": 10**12,
                       "role": "archive", "dataset": "Paper3", "io_profile": "baseline"}]}
        report["closure_sha256"] = deploy.digest(report)
        with self.assertRaisesRegex(deploy.DeploymentError, "raw_payload_cannot_enter_final_deployment_manifest"):
            deploy.final_manifest(report)

    def test_open_ledger_prevents_complete_selection(self):
        path = "audit/source_extensions_20261019/Yang2025/a/budget_ledger.json"
        self.write(path, {"plan_sha256": "a", "active_transfer": {}})
        with self.assertRaisesRegex(deploy.DeploymentError, "final_coordinator_ledger_must_be_closed"):
            deploy.closed_ledger(self.root, path, "a")

    def fake_complete_yang(self, sha, *, ledger_open=False):
        path = f"cache/source_extensions_20261019/Yang2025/{sha}/source_cache.json"
        item = self.write(path, {"schema_version": "paper3-source-extension-cache-v1", "dataset": "Yang2025",
            "status": "SOURCE_EXTENSION_CACHE_COMPLETE", "role": "source_train", "source_only": True,
            "unique_role_within_branch": True, "record_count": 153, "shards": [{}]*153, "plan_sha256": sha})
        base = f"audit/source_extensions_20261019/Yang2025/{sha}"
        receipt = self.receipt(base+"/final_backup_receipt.json", [self.entry(item, dataset="Yang2025")], new_profile=True)
        self.write(base+"/coordinator_status.json", {"status": "SOURCE_EXTENSION_CACHE_COMPLETE", "record_count": 153,
            "manifest": item, "backup_receipt": receipt})
        self.write(base+"/budget_ledger.json", {"plan_sha256": sha, **({"wall_intent": {}} if ledger_open else {})})
        return path

    def test_multiple_plans_choose_only_unique_actual_complete_closed_commit(self):
        self.fake_complete_yang("a"*64, ledger_open=True)
        second = self.fake_complete_yang("b"*64)
        self.assertEqual(deploy.default_manifests(self.root)["Yang2025"], second)
        self.fake_complete_yang("c"*64)
        self.assertIsNone(deploy.default_manifests(self.root)["Yang2025"])

    def test_nominal_complete_status_without_actual_full_get_receipt_not_selected(self):
        path = self.fake_complete_yang("d"*64)
        receipt = self.root/"audit/source_extensions_20261019/Yang2025"/("d"*64)/"final_backup_receipt.json"
        value = json.loads(receipt.read_bytes())
        value["status"] = "failed"
        receipt.write_text(json.dumps(value))
        self.assertIsNone(deploy.default_manifests(self.root)["Yang2025"])

    def history_fixture(self):
        base = "audit/data_plan_20261019/dreyer/coordinator_v1_failure"
        source = "p3work/parser.py"
        old_code = self.write(source, b"# old parser\n")
        self.write(base+"/parser.py", b"# old parser\n")
        current = self.write(source, b"# repaired parser\n")
        caps = {"provider_body_bytes": 100, "r2_get_body_bytes": 200, "wall_seconds": 10}
        remaining = {"provider_body_bytes": 80, "r2_get_body_bytes": 150, "wall_seconds": 8}
        old = {"code": {source: old_code}, "budgets": caps}
        old["plan_sha256"] = deploy.digest(old)
        new = {"code": {source: current}, "budgets": remaining}
        new["plan_sha256"] = deploy.digest(new)
        old_item = self.write(base+"/old_plan.json", old)
        new_item = self.write("audit/source_extensions_20261019/Dreyer2023/plan_new.json", new)
        ledger = {"plan_sha256": old["plan_sha256"], "charged": {"provider_body_bytes": 20, "r2_get_body_bytes": 50},
                  "elapsed_seconds": 1.5}
        ledger_item = self.write(base+"/old_ledger.json", ledger)
        amendment = {"schema_version": "paper3-source-preparation-parser-repair-budget-conservation-v1",
            "old_plan": old_item, "new_plan": new_item, "old_closed_ledger": ledger_item,
            "old_usage": {"charged": ledger["charged"], "elapsed_seconds": 1.5},
            "new_budgets": remaining, "aggregate_original_caps": caps}
        self.write(base+"/budget_conservation_amendment.json", amendment)
        return base+"/budget_conservation_amendment.json", amendment

    def test_historical_code_resolves_only_to_identical_archived_SHA(self):
        path, amendment = self.history_fixture()
        closure = deploy.Closure(self.root)
        closure.load_history(path)
        binding = closure.history[0]["historical_code_bindings"][0]
        self.assertEqual(binding["historical_source_path"], "p3work/parser.py")
        self.assertIn("coordinator_v1_failure", binding["archive_snapshot"]["path"])
        self.assertIn("p3work/parser.py", closure.files)
        self.assertIn(binding["archive_snapshot"]["path"], closure.files)

    def test_wrong_archived_code_SHA_fails_closed(self):
        path, amendment = self.history_fixture()
        self.write(str(Path(path).parent/"parser.py"), b"# not the old parser")
        with self.assertRaisesRegex(deploy.DeploymentError, "historical_code_snapshot_must_match_old_code_SHA"):
            deploy.Closure(self.root).load_history(path)

    def test_historical_chain_registers_all_SHA_aliases_before_intermediate_plan_walk(self):
        source="p3work/parser.py"
        bases=["audit/data_plan_20261019/dreyer/coordinator_v1_failure",
               "audit/data_plan_20261019/dreyer/online_suffix_v2_failure"]
        code=[]
        for i in range(3):code.append(self.write(source,f"# parser version {i+1}\n".encode()))
        plans=[]
        for i,c in enumerate(code):
            payload={"dataset":"Dreyer2023","code":{source:c},
                     "budgets":{"provider_body_bytes":100-20*i,"r2_get_body_bytes":200-50*i,"wall_seconds":10-2*i}}
            payload["plan_sha256"]=deploy.digest(payload)
            plans.append(self.write(f"audit/source_extensions_20261019/Dreyer2023/plan{i+1}.json",payload))
        paths=[]
        for i,base in enumerate(bases):
            self.write(base+"/parser.py",f"# parser version {i+1}\n".encode())
            old=json.loads((self.root/plans[i]["path"]).read_bytes())
            new=json.loads((self.root/plans[i+1]["path"]).read_bytes())
            ledger={"plan_sha256":old["plan_sha256"],"charged":{"provider_body_bytes":20,"r2_get_body_bytes":50},"elapsed_seconds":1.5}
            entry=self.write(base+"/old_ledger.json",ledger)
            value={"schema_version":"paper3-source-preparation-parser-repair-budget-conservation-v1",
                "old_plan":plans[i],"new_plan":plans[i+1],"old_closed_ledger":entry,
                "old_usage":{"charged":ledger["charged"],"elapsed_seconds":1.5},
                "new_budgets":new["budgets"],"aggregate_original_caps":old["budgets"]}
            paths.append(self.write(base+"/budget_conservation_amendment.json",value)["path"])
        closure=deploy.Closure(self.root)
        for path in paths:closure.load_history(path,defer_walk=True)
        for item,old in closure.pending_history:closure.walk_history(item,old)
        self.assertEqual(set(closure.code_aliases),{(source,code[i]["sha256"]) for i in (0,1)})
        self.assertEqual(closure.files[source]["sha256"],code[2]["sha256"])
        self.assertTrue(all(base+"/parser.py" in closure.files for base in bases))

    def test_failed_parent_budget_cannot_be_reset_in_child_plan(self):
        path, amendment = self.history_fixture()
        amendment["aggregate_original_caps"]["provider_body_bytes"] = 99
        self.write(path, amendment)
        with self.assertRaisesRegex(deploy.DeploymentError, "historical_body_budget_not_conserved"):
            deploy.Closure(self.root).load_history(path)

    def test_write_once_preserves_existing_fixed_snapshot(self):
        path = self.root/"audit/deploy.json"
        deploy.write_once(path, {"sha": "old"})
        with self.assertRaisesRegex(deploy.DeploymentError, "existing_deploy_artifact_not_overwritten"):
            deploy.write_once(path, {"sha": "new"})
        self.assertEqual(json.loads(path.read_bytes()), {"sha": "old"})

    def precision_fixture(self):
        old=self.write(deploy.PRECISION_AUDIT+"/original_classical.py",b"# original precision\n")
        current=self.write(deploy.PRECISION_SOURCE,b"# explicit float64 promotion\n")
        inventory={"files":[{"path":deploy.PRECISION_SOURCE,"sha256":old["sha256"],"bytes":old["bytes"]}]+
                   [{"path":f"p3work/retained_{i}.py","sha256":"a"*64,"bytes":0} for i in range(116)]}
        manifest=self.write(deploy.PRECISION_AUDIT+"/original_software_code_manifest.json",inventory)
        self.write(deploy.PRECISION_AUDIT+"/native_regression.json",{
            "schema_version":"paper3-classical-cache-precision-native-regression-v1",
            "current_Paper3_changed_file":deploy.PRECISION_SOURCE,"native_dtype":"float32","promoted_dtype":"float64",
            "covariances_bitwise_equal":True,"no_signal_modification":True,"actual_fits":0,
            "GPU_actions":0,"network_operations":0,"P1_P2_code_changed":False})
        return old,current,manifest

    def test_precision_history_keeps_original_manifest_SHA_and_current_code_separate(self):
        old,current,manifest=self.precision_fixture()
        with patch.multiple(deploy,PRECISION_OLD_SHA=old["sha256"],PRECISION_CURRENT_SHA=current["sha256"],
                            ORIGINAL_SOFTWARE_MANIFEST_SHA=manifest["sha256"]):
            closure=deploy.Closure(self.root)
            closure.register_precision_fix()
            closure.visit({"path":deploy.PRECISION_SOURCE,"sha256":old["sha256"],"bytes":old["bytes"]})
            closure.visit(current)
            self.assertEqual(closure.files[deploy.PRECISION_SOURCE]["sha256"],current["sha256"])
            self.assertIn(old["path"],closure.files)
            self.assertTrue(closure.precision_history[0]["original_inventory_is_historical"])

    def test_precision_alias_rejects_wrong_snapshot_even_under_expected_name(self):
        old,current,manifest=self.precision_fixture()
        self.write(old["path"],b"wrong historical implementation")
        with patch.multiple(deploy,PRECISION_OLD_SHA=old["sha256"],PRECISION_CURRENT_SHA=current["sha256"],
                            ORIGINAL_SOFTWARE_MANIFEST_SHA=manifest["sha256"]):
            with self.assertRaisesRegex(deploy.DeploymentError,"exact_classical_precision_fix"):
                deploy.Closure(self.root).register_precision_fix()

    def runtime_recovery_fixture(self):
        from tools import recover_cpu_preparation_20261020 as recovery
        tool=self.write(deploy.RECOVERY_TOOL,Path(recovery.__file__).read_bytes())
        original_boot="00000000-0000-0000-0000-000000000001"
        now_boot="00000000-0000-0000-0000-000000000002"
        clock={"utc":"2026-10-19T12:00:00+00:00","boot_id":original_boot,"monotonic_ns":1000000000}
        new_clock={"utc":"2026-10-19T12:01:00+00:00","boot_id":now_boot,"monotonic_ns":0}
        before_p={"plan_sha256":recovery.STIEGER_SHA,"active_wall_elapsed_s":12.,"wall_intent":clock}
        before_y={"schema_version":"paper3-source-extension-budget-ledger-v1","plan_sha256":recovery.YANG_SHA,
            "elapsed_seconds":4.,"wall_intent":clock,"charged":{"provider_body_bytes":10,"r2_get_body_bytes":20},
            "history":[{"kind":"provider_body_bytes","reserved_bytes":10,"verified_bytes":10,"status":"SUCCESS_FULL_BODY_VERIFIED"}],
            "active_transfer":{"kind":"r2_get_body_bytes","reserved_bytes":20,"identity":"interrupted provider proof"}}
        after_p,after_y,accounting=recovery.reconcile_values(before_p,before_y,new_clock)
        base="audit/data_plan_20261020/runtime_recovery/"+now_boot
        originals=[]
        for path,value in [(recovery.PROGRESS,before_p),(recovery.YANG_LEDGER,before_y)]:
            item=self.write(base+"/originals/"+path,value)
            originals.append({"original_path":path,"snapshot":item,"original_sha256":item["sha256"]})
        stieger=self.write("audit/stieger_plan.json",{"max_wall_seconds":86400})
        yang=self.write("audit/yang_plan.json",{"budgets":{"wall_seconds":172800}})
        value={"schema_version":"paper3-explicit-CPU-runtime-reconciliation-v1","implementation":tool,
            "body_caps_unchanged":True,"scientific_fits":0,"GPU_actions":0,"remote_server_actions":False,
            "external_data_access":False,"original_records":originals,"new_runtime":new_clock,
            "proposed_runtime_records":{recovery.PROGRESS:after_p,recovery.YANG_LEDGER:after_y},
            "conservative_accounting":accounting,"Stieger_plan":stieger,"Yang_plan":yang,
            "effective_Stieger_CPU_wall_cap_seconds":172800,"original_Stieger_plan_wall_cap_seconds":86400,
            "Yang_plan_wall_cap_seconds_unchanged":172800}
        amendment=self.write(base+"/runtime_reconciliation_amendment.json",value)
        transaction=self.write(base+"/transaction.json",{"schema_version":"paper3-runtime-recovery-transaction-v1",
            "status":"RECONCILIATION_COMMITTED","amendment":amendment})
        return tool,transaction,value

    def test_runtime_recovery_rechecks_complete_gap_and_preserves_interrupted_charge(self):
        tool,transaction,_=self.runtime_recovery_fixture()
        with patch.object(deploy,"RECOVERY_TOOL_SHA",tool["sha256"]):
            closure=deploy.Closure(self.root)
            closure.load_runtime_recovery(transaction["path"])
            row=closure.runtime_recovery_history[0]
            self.assertEqual(row["entire_UTC_gap_billed_seconds"],{"Stieger2021":60,"Yang2025":60})
            self.assertEqual(row["body_reservation_refunded_bytes"],0)
            self.assertEqual(row["immutable_original_records_verified"],2)

    def test_runtime_recovery_does_not_accept_rehashed_lower_consumption(self):
        from tools import recover_cpu_preparation_20261020 as recovery
        tool,transaction,value=self.runtime_recovery_fixture()
        value["proposed_runtime_records"][recovery.YANG_LEDGER]["charged"]["r2_get_body_bytes"]-=1
        transaction_value=json.loads((self.root/transaction["path"]).read_bytes())
        transaction_value["amendment"]=self.write(transaction_value["amendment"]["path"],value)
        self.write(transaction["path"],transaction_value)
        with patch.object(deploy,"RECOVERY_TOOL_SHA",tool["sha256"]):
            with self.assertRaisesRegex(deploy.DeploymentError,"preserve_all_charges_and_bill_entire_gap"):
                deploy.Closure(self.root).load_runtime_recovery(transaction["path"])

    def test_uncommitted_runtime_transaction_is_never_a_completed_recovery(self):
        _,transaction,_=self.runtime_recovery_fixture()
        value=json.loads((self.root/transaction["path"]).read_bytes());value["status"]="RECONCILIATION_PREPARED"
        self.write(transaction["path"],value)
        with self.assertRaisesRegex(deploy.DeploymentError,"committed_CPU_runtime_reconciliation_required"):
            deploy.Closure(self.root).load_runtime_recovery(transaction["path"])

    def test_Stieger_status_requires_closed_final_progress(self):
        path="cache/stieger_v2/stieger_cache.json"
        manifest=self.write(path,{"schema_version":"paper3-stieger-cache-v2","status":"STIEGER_CACHE_COMPLETE",
            "session_count":598,"sessions":[{}]*598,"participant_count":62,"plan_sha256":"a"*64})
        task={"W3":{"status":"COMPLETE","cache_manifest":manifest}}
        self.write("status/tasks.json",task)
        progress={"status":"W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY","plan_sha256":"a"*64,"completed_sessions":598}
        self.write("cache/stieger_v2/progress.json",{**progress,"wall_intent":{}})
        with self.assertRaisesRegex(deploy.DeploymentError,"Stieger_actual_closed_global_completion_required"):
            deploy.complete_component(self.root,"Stieger2021",path)
        self.write("cache/stieger_v2/progress.json",progress)
        _,receipts=deploy.complete_component(self.root,"Stieger2021",path)
        self.assertIn("cache/stieger_v2/progress.json",receipts)


if __name__ == "__main__":
    unittest.main()
