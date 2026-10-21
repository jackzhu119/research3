#!/usr/bin/env python3
"""Inspect a cache-only GPU deployment closure; no network or training operations.

The optional final output contains explicit local backup manifests, separated
by original and source-extension IO profiles, and exact selected paths from
already verified receipts. Raw objects are evidence references only and can
never enter a restore selection. A generated manifest is not an upload receipt
or a claim that a fresh GPU restoration or scientific fit has happened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work import source_extension_io as extension
from tools import r2_backup, r2_restore

SCHEMA = "paper3-data-plan-cache-only-deploy-v1"
COMPONENTS = ("source4", "Stieger2021", "Yang2025", "Dreyer2023", "Ma2022")
SOURCE4_SHA = "bfc204577dfe51c1fc798c1241391ab63509fc2ad9ebfb267ba09254e01ae644"
SOURCE_IDS = {"BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI"}
EXPECTED = {"source4": 224, "Stieger2021": 598, "Yang2025": 153, "Dreyer2023": 520, "Ma2022": 125}
RAW_SUFFIXES = {".mat", ".edf", ".bdf", ".gdf", ".set", ".fdt", ".cnt", ".dat", ".eeg", ".fif", ".h5", ".hdf5"}
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".log", ".sha256", ".tsv", ".csv", ".html", ".xml", ".pdf", ".py", ".toml", ".lock"}
DEFAULT_WHEELS = ("audit/r2_dependencies_receipt.json", "audit/r2_dependency_extras_receipt.json")
PRODUCER_ROOT = Path("/workspace/research3")
MA_RELOCATABLE_PREFIXES = ("cache/source_extensions_20261019/Ma2022/",
                          "data/source_extensions_20261019/Ma2022/")
RUNTIME_CONTRACT = "contracts/portable_python_312.json"
RUNTIME_PATH = "runtime/cpython-3.12.15+20261013-x86_64-unknown-linux-gnu-install_only.tar.gz"
RUNTIME_SHA = "5d546fc004fc2435fa142eb8f42ed8de5eb4725963dc22f7e6671014a80eb560"
RUNTIME_BYTES = 66983838
CAR3_REREAD = "audit/external_expansion_20261019/round2/car3_preparation/independent_output_reread.json"
PRIOR_ARCHIVES = {
    "/workspace/shared/paper3-controlled-pilot-ready-b5fb2b424951eb16.tar.gz":
        ("b5fb2b424951eb16e7ccf7fa807b642384d48148665db388f39a7194ea6003c1", 32795394),
    "/workspace/shared/paper3-external-qualification-addon-20261019-b06ee6c193eba755.tar.gz":
        ("b06ee6c193eba755d57fd4fc549abdc07da5032355ed18c3f5170c41a9efe872", 6233908),
}
PRECISION_SOURCE = "p3work/methods/classical.py"
PRECISION_OLD_SHA = "6dc2adc5cbf29e26e766ae92ea52e42f7e5c7b25c3c81800e26ae800ad6d91a6"
PRECISION_CURRENT_SHA = "0422c73d00b74c659f744657539a142aecff0977aac22552dcafe84818fdfe2e"
ORIGINAL_SOFTWARE_MANIFEST_SHA = "87e2cd22f37f8d96aebd020daa6f57872526f60b6b904fefbc3ce47d373ce3a0"
PRECISION_AUDIT = "audit/data_plan_20261020/classical_precision_fix"
RECOVERY_TOOL = "tools/recover_cpu_preparation_20261020.py"
RECOVERY_TOOL_SHA = "1f4fd5b3c25115228fbc4c8e0d726223c02729a3a624b2880b7bdfad85efb406"


class DeploymentError(ValueError):
    """Only fixed, non-sensitive error codes are public."""

    def __init__(self, code, path=None):
        self.path = path
        super().__init__(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def relative(root: Path, value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or any(ord(c) < 32 for c in value):
        raise DeploymentError("unsafe_dependency_path")
    p = Path(value)
    if p.is_absolute():
        try:
            value = p.relative_to(root).as_posix()
        except ValueError:
            # The native Ma producer recorded its actual creation paths. Only
            # these two data namespaces have a schema-bound relocation map;
            # never read the original absolute path on a restoration host.
            try:
                candidate = p.relative_to(PRODUCER_ROOT).as_posix()
            except ValueError:
                raise DeploymentError("absolute_dependency_outside_current_root") from None
            if not candidate.startswith(MA_RELOCATABLE_PREFIXES):
                raise DeploymentError("absolute_dependency_outside_current_root")
            value = candidate
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise DeploymentError("unsafe_dependency_path")
    extension._credential_components(Path(value))
    r2_restore.no_symlink_components(root/value)
    if not (root/value).resolve().is_relative_to(root.resolve()):
        raise DeploymentError("dependency_outside_root")
    return value


def classify(path: str) -> str:
    p = Path(path)
    parts, suffix = p.parts, p.suffix.lower()
    if not parts:
        raise DeploymentError("unsafe_dependency_path")
    if any(part.lower() in {".git", "__pycache__", "paper1", "paper2"} for part in parts):
        raise DeploymentError("foreign_project_or_private_tree_forbidden")
    if suffix in {".npz", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".partial", ".tmp"}:
        raise DeploymentError("legacy_cache_weights_or_partial_forbidden")
    if parts[0] in {"runs", "predictions", "checkpoints"}:
        raise DeploymentError("training_or_query_outputs_forbidden")
    if parts[0] == "runtime":
        if path != RUNTIME_PATH:
            raise DeploymentError("only_exact_pinned_software_runtime_archive_allowed")
        return "software_runtime"
    if parts[:2] == ("data", "dependency_wheels"):
        if len(parts) != 4 or parts[2] not in {"cpu", "gpu"} or suffix != ".whl":
            raise DeploymentError("only_exact_offline_wheels_allowed_under_data")
        return "wheel"
    if parts[0] == "data" or suffix in RAW_SUFFIXES:
        return "raw_reference_only"
    if parts[0] == "cache":
        allowed = {"source_v2", "stieger_v2", "source_extensions_20261019", "source_c3czc4_v1"}
        if len(parts) < 3 or parts[1] not in allowed or suffix not in TEXT_SUFFIXES | {".npy"}:
            raise DeploymentError("cache_outside_approved_preparation_scope")
        return "cache"
    if suffix == ".npy":
        if parts[:2] != ("manifests", "geometry_constants"):
            raise DeploymentError("array_outside_approved_cache_or_geometry")
        return "contract"
    if parts[0] not in {"audit", "catalog", "data_catalog", "manifests", "contracts", "plans", "protocol", "status", "provenance", "p3work", "tools", "tests"}:
        if path not in {"requirements-cpu.lock", "requirements-gpu.lock", "pyproject.toml", "README.md"}:
            raise DeploymentError("dependency_outside_paper3_runtime_allowlist")
    if suffix not in TEXT_SUFFIXES:
        raise DeploymentError("unsupported_runtime_dependency_type")
    # P1 audit evidence may establish native identity; P1/P2 configuration and
    # any foreign runtime trees are never included in a GPU preparation bundle.
    if (suffix in {".toml", ".lock"} and re.search(r"(?:^|[/_])p[12](?:[/_.]|$)", path.lower())):
        raise DeploymentError("prior_paper_configuration_forbidden")
    if parts[0] in {"p3work", "tools", "tests"}:
        return "code"
    return "metadata"


def descriptor(root: Path, path: str) -> dict:
    path = relative(root, path)
    if not (root/path).is_file():
        raise DeploymentError("dependency_file_missing", path)
    sha, size = r2_backup.sha256_file(root/path)
    return {"path": path, "sha256": sha, "bytes": size}


class Closure:
    def __init__(self, root: Path):
        self.root = Path(os.path.abspath(root))
        self.files, self.receipts, self.raw, self.errors = {}, {}, {}, []
        self.visited, self.r2_rows = set(), {}
        self.code_aliases, self.history = {}, []
        self.pending_history, self.historical_nonruntime = [], []
        self.precision_history, self.runtime_recovery_history = [], []
        self.protocol_members, self.raw_get_backed = set(), set()
        self.integrity_cache = {}

    def error(self, code, path=None, context=None):
        value = {"code": code}
        if path is not None:
            value["path"] = path
        if context is not None:
            value["context"] = context
        if value not in self.errors:
            self.errors.append(value)

    def actual_descriptor(self, path):
        stat = (self.root/path).stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        previous = self.integrity_cache.get(path)
        if previous is not None and previous[0] == identity:
            return previous[1]
        actual = descriptor(self.root, path)
        after = (self.root/path).stat()
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise DeploymentError("dependency_changed_during_full_SHA_read", path)
        self.integrity_cache[path] = (identity, actual)
        return actual

    def check_stable_snapshot(self):
        for path, (expected, _) in self.integrity_cache.items():
            try:
                stat = (self.root/path).stat()
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            except OSError:
                identity = None
            if identity != expected:
                self.error("dependency_changed_during_closure_snapshot", path)

    def visit(self, item: dict, *, context="Paper3", role=None, receipt_profile=None):
        if (not isinstance(item, dict) or not {"path", "sha256"} <= set(item)
                or not re.fullmatch("[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise DeploymentError("explicit_SHA_dependency_descriptor_required")
        path = relative(self.root, item["path"])
        alias = self.code_aliases.get((path, item["sha256"]))
        if alias is not None:
            self.visit(alias, context=context, role="audit")
            return
        kind = "original_protocol" if path in self.protocol_members else classify(path)
        if kind == "raw_reference_only":
            self.raw[(path, item["sha256"])] = {"path": path, "sha256": item["sha256"],
                "bytes": item.get("bytes"), "dataset": item.get("dataset", context), "payload_selected": False}
            return
        actual = self.actual_descriptor(path)
        if actual["sha256"] != item["sha256"] or ("bytes" in item and actual["bytes"] != item["bytes"]):
            raise DeploymentError("dependency_file_SHA_or_bytes_mismatch", path)
        if path in self.files and self.files[path]["sha256"] != actual["sha256"]:
            raise DeploymentError("conflicting_dependency_SHA_for_same_path")
        profile = "source_extension" if path.startswith((extension.RAW_ROOT+"/", extension.RESULT_ROOT+"/",
            "audit/source_extensions_20261019/", "catalog/source_extensions_20261019/")) else "baseline"
        dataset = context if context in SOURCE_IDS | {"Stieger2021", "Yang2025", "Dreyer2023", "Ma2022"} else "Paper3"
        if profile == "source_extension":
            dataset = Path(path).parts[2]
            if dataset not in extension.REGISTERED_ROLES:
                raise DeploymentError("unregistered_source_extension_dependency")
        if kind == "software_runtime":
            if actual["sha256"] != RUNTIME_SHA or actual["bytes"] != RUNTIME_BYTES:
                raise DeploymentError("portable_python_archive_full_SHA_or_bytes_mismatch", path)
            role, dataset = "archive", "PythonRuntime312"
        elif kind == "wheel":
            role, dataset = "archive", "PythonDependenciesGPU" if "/gpu/" in path else "PythonDependenciesCPU"
        elif kind == "cache":
            role = role or "result"
            if Path(path).parts[1] in {"source_v2", "source_c3czc4_v1"} and len(Path(path).parts) > 3:
                component = Path(path).parts[2]
                if component in SOURCE_IDS:
                    dataset = component
            if path.startswith("cache/stieger_v2/"):
                dataset = "Stieger2021"
        else:
            role = role or ("code" if kind == "code" else "protocol" if kind in {"contract", "original_protocol"} else "audit")
        row = {**actual, "role": role, "dataset": dataset, "io_profile": profile}
        if kind == "original_protocol":
            row["original_protocol_manifest_member"] = True
        if profile == "source_extension":
            extension.validate_entry({key: row[key] for key in extension.ENTRY_FIELDS})
        else:
            r2_backup.validate_entry({key: row[key] for key in extension.ENTRY_FIELDS})
        self.files[path] = row
        if path in self.visited:
            if receipt_profile is not None and path in self.receipts and self.receipts[path]["io_profile"] != receipt_profile:
                raise DeploymentError("backup_receipt_uses_wrong_IO_profile")
            return
        self.visited.add(path)
        if kind == "original_protocol":
            return  # Original document content is authority/evidence, not new runtime instructions.
        if path in {"provenance/files.json", "provenance/origin.json"}:
            return  # Import receipts name members in the original ZIP namespace, already independently checked.
        suffix = Path(path).suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            return
        raw = (self.root/path).read_bytes()
        if suffix == ".jsonl":
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            data = json.loads(raw)
        if isinstance(data, dict) and data.get("operation") == "paper3_explicit_manifest_backup":
            self._receipt(actual, data, expected_profile=receipt_profile or
                          ("source_extension" if profile == "source_extension" else None))
        else:
            if path == CAR3_REREAD:
                data = self._car3_historical_archives(data, actual)
            self.walk(data, context=context)

    def _car3_historical_archives(self, data, source):
        """Retain the exact old comparison evidence without restoring old ZIPs.

        Other fields, including CAR3 arrays and conversion receipts, still
        enter the ordinary dependency walker. This exception is tied to one
        producer schema and two immutable published delivery identities.
        """
        if (not isinstance(data, dict)
                or data.get("schema_version") != "paper3-car3-independent-output-reread-v1"):
            raise DeploymentError("known_CAR3_reread_schema_required")
        rows = data.get("prior_archives_unchanged")
        if not isinstance(rows, list) or len(rows) != len(PRIOR_ARCHIVES):
            raise DeploymentError("exact_historical_delivery_comparisons_required")
        seen = set()
        for row in rows:
            path = row.get("path") if isinstance(row, dict) else None
            if (path not in PRIOR_ARCHIVES or path in seen
                    or (row.get("sha256"), row.get("bytes")) != PRIOR_ARCHIVES[path]
                    or row.get("sidecar_matched") is not True):
                raise DeploymentError("historical_delivery_comparison_identity_changed")
            seen.add(path)
            self.historical_nonruntime.append({"evidence": source, "comparison": dict(row),
                "runtime_dependency": False, "payload_selected": False})
        return {k: v for k, v in data.items() if k != "prior_archives_unchanged"}

    def original_protocol(self):
        archive = self.root/"archives/Paper3_Work_Handoff_20261018.zip"
        if r2_backup.sha256_file(archive) != (
                "0398664faf34682a07fb38de4e46df9829f0e5ad050f657f905d0373dbfaf175", 529763):
            raise DeploymentError("original_user_protocol_archive_identity_mismatch")
        with zipfile.ZipFile(archive) as package:
            manifest = package.read("paper3_work_handoff_20261018/PACKAGE_CONTENTS.sha256")
        if (self.root/"protocol/PACKAGE_CONTENTS.sha256").read_bytes() != manifest:
            raise DeploymentError("original_52_protocol_manifest_bytes_changed")
        items = []
        for line in manifest.decode().splitlines():
            sha, name = line.split(maxsplit=1)
            path = relative(self.root, "protocol/"+name.removeprefix("*"))
            if Path(path).suffix.lower() in RAW_SUFFIXES | {".npz", ".pt", ".pth"}:
                raise DeploymentError("raw_or_weight_cannot_enter_original_protocol_transport")
            items.append({"path": path, "sha256": sha})
        if len(items) != 52 or len({r["path"] for r in items}) != 52:
            raise DeploymentError("all_original_52_unique_protocol_files_required")
        self.protocol_members.update(r["path"] for r in items)
        for item in items:
            self.visit(item)
        self.path("protocol/PACKAGE_CONTENTS.sha256", role="protocol")

    def portable_runtime(self):
        """One pinned software archive, backed by actual CPU/offline evidence."""
        contract_item = descriptor(self.root, RUNTIME_CONTRACT)
        value = json.loads((self.root/RUNTIME_CONTRACT).read_bytes())
        expected_archive = {"path":RUNTIME_PATH,"sha256":RUNTIME_SHA,"bytes":RUNTIME_BYTES}
        if (value.get("schema_version") != "paper3-portable-python-312-v1"
                or value.get("runtime_archive") != expected_archive
                or value.get("implementation") != "CPython" or value.get("python_version") != "3.12.15"
                or value.get("platform") != "x86_64-unknown-linux-gnu"
                or value.get("GPU_checked") is not False or value.get("GPU_compatibility_verified") is not False
                or value.get("driver_compatibility_verified") is not False
                or value.get("actual_scientific_fits") != 0 or value.get("requires_user_manual_GPU_boot") is not True):
            raise DeploymentError("exact_portable_python_contract_and_GPU_pending_scope_required")
        receipts = {}
        for name in ("provider_download_receipt", "offline_CPU_acceptance", "dependency_profile"):
            item = value.get(name)
            if (not isinstance(item,dict) or not {"path","sha256","bytes"} <= set(item)
                    or descriptor(self.root,item["path"]) != item):
                raise DeploymentError("portable_python_actual_evidence_SHA_required")
            if name != "dependency_profile":
                receipts[name] = json.loads((self.root/item["path"]).read_bytes())
        if value["dependency_profile"]["path"] != "requirements-gpu.lock":
            raise DeploymentError("portable_python_exact_GPU_dependency_profile_required")
        provider = receipts["provider_download_receipt"]
        acceptance = receipts["offline_CPU_acceptance"]
        if (any(provider.get(k) != v for k,v in expected_archive.items())
                or provider.get("provider_GitHub_asset_digest") != "sha256:"+RUNTIME_SHA
                or provider.get("full_provider_GET_and_SHA_verified") is not True
                or provider.get("archive_path_validation_passed") is not True
                or not provider.get("license_files_preserved_inside_archive")):
            raise DeploymentError("portable_python_actual_provider_GET_and_license_evidence_required")
        if (acceptance.get("schema_version") != "paper3-portable-python-cpu-acceptance-v1"
                or acceptance.get("status") != "PORTABLE_PYTHON_FULL_GPU_PROFILE_OFFLINE_INSTALL_AND_CPU_CHECK_COMPLETE"
                or acceptance.get("archive") != expected_archive or acceptance.get("interpreter_version") != "3.12.15"
                or acceptance.get("requirements") != value["dependency_profile"]
                or acceptance.get("exact_requirement_pins") != 100
                or acceptance.get("installed_requested_packages") != 100
                or acceptance.get("available_fixed_wheels") != 110
                or acceptance.get("all_requirement_versions_match") is not True
                or acceptance.get("all_install_sources_local_file") is not True
                or acceptance.get("GPU_checked") is not False
                or acceptance.get("GPU_compatibility_verified") is not False
                or acceptance.get("actual_scientific_fits") != 0
                or acceptance.get("network_operations") != 0 or acceptance.get("R2_operations") != 0
                or acceptance.get("temporary_venv_and_extracted_runtime_removed") is not True
                or not acceptance.get("commands")
                or any(row.get("returncode") != 0 for row in acceptance["commands"])):
            raise DeploymentError("portable_python_full_offline_CPU_acceptance_required_GPU_still_pending")
        self.path(RUNTIME_CONTRACT, role="protocol")
        return {"status":"PINNED_ARCHIVE_AND_FULL_OFFLINE_CPU_ACCEPTANCE_VERIFIED",
                "contract":contract_item,"archive":expected_archive,
                "GPU_compatibility_verified":False,"actual_scientific_fits":0}

    def register_precision_fix(self):
        """The original 117-file inventory stays historical after this one fix."""
        source = descriptor(self.root, PRECISION_SOURCE)
        if source["sha256"] == PRECISION_OLD_SHA:
            return
        archived = descriptor(self.root, PRECISION_AUDIT+"/original_classical.py")
        original_manifest = descriptor(self.root, PRECISION_AUDIT+"/original_software_code_manifest.json")
        regression = descriptor(self.root, PRECISION_AUDIT+"/native_regression.json")
        old = json.loads((self.root/original_manifest["path"]).read_bytes())
        record = [row for row in old.get("files",[]) if row.get("path")==PRECISION_SOURCE]
        proof = json.loads((self.root/regression["path"]).read_bytes())
        if (source["sha256"] != PRECISION_CURRENT_SHA or archived["sha256"] != PRECISION_OLD_SHA
                or original_manifest["sha256"] != ORIGINAL_SOFTWARE_MANIFEST_SHA
                or len(old.get("files",[])) != 117
                or len(record) != 1 or record[0].get("sha256") != PRECISION_OLD_SHA
                or record[0].get("bytes") != archived["bytes"]
                or proof.get("schema_version") != "paper3-classical-cache-precision-native-regression-v1"
                or proof.get("current_Paper3_changed_file") != PRECISION_SOURCE
                or proof.get("native_dtype") != "float32" or proof.get("promoted_dtype") != "float64"
                or proof.get("covariances_bitwise_equal") is not True or proof.get("no_signal_modification") is not True
                or proof.get("actual_fits") != 0 or proof.get("GPU_actions") != 0
                or proof.get("network_operations") != 0 or proof.get("P1_P2_code_changed") is not False):
            raise DeploymentError("exact_classical_precision_fix_and_historical_snapshot_required")
        self.code_aliases[(PRECISION_SOURCE,PRECISION_OLD_SHA)] = archived
        self.precision_history.append({"historical_source_path":PRECISION_SOURCE,
            "historical_source_sha256":PRECISION_OLD_SHA,"archive_snapshot":archived,
            "current_implementation":source,"original_117_manifest":original_manifest,
            "native_numerical_regression":regression,"original_inventory_is_historical":True})

    def load_runtime_recovery(self, path):
        """Verify a committed operational transaction without resuming a worker."""
        transaction_item = descriptor(self.root,path)
        transaction = json.loads((self.root/path).read_bytes())
        if (transaction.get("schema_version") != "paper3-runtime-recovery-transaction-v1"
                or transaction.get("status") != "RECONCILIATION_COMMITTED"):
            raise DeploymentError("committed_CPU_runtime_reconciliation_required")
        item = transaction.get("amendment")
        if not isinstance(item,dict) or descriptor(self.root,item["path"]) != item:
            raise DeploymentError("CPU_runtime_amendment_full_SHA_required")
        amendment = json.loads((self.root/item["path"]).read_bytes())
        tool = amendment.get("implementation")
        if (amendment.get("schema_version") != "paper3-explicit-CPU-runtime-reconciliation-v1"
                or not isinstance(tool,dict) or tool.get("path") != RECOVERY_TOOL
                or tool.get("sha256") != RECOVERY_TOOL_SHA or descriptor(self.root,RECOVERY_TOOL) != tool
                or amendment.get("body_caps_unchanged") is not True
                or amendment.get("scientific_fits") != 0 or amendment.get("GPU_actions") != 0
                or amendment.get("remote_server_actions") is not False or amendment.get("external_data_access") is not False):
            raise DeploymentError("exact_CPU_only_runtime_reconciliation_scope_required")
        originals={}
        for row in amendment.get("original_records",[]):
            key = relative(self.root,row["original_path"])
            snapshot = row["snapshot"]
            if (key in originals or descriptor(self.root,snapshot["path"]) != snapshot
                    or snapshot["sha256"] != row.get("original_sha256")):
                raise DeploymentError("immutable_pre_reconciliation_snapshots_required")
            originals[key] = json.loads((self.root/snapshot["path"]).read_bytes())
        specification=importlib.util.spec_from_file_location("_paper3_deploy_runtime_recovery",self.root/RECOVERY_TOOL)
        module=importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        before_p,before_y=originals.get(module.PROGRESS),originals.get(module.YANG_LEDGER)
        if not isinstance(before_p,dict) or not isinstance(before_y,dict):
            raise DeploymentError("both_original_runtime_records_required")
        # This is the verified tool's pure calculation only: no locks, writes,
        # SDK, CUDA, worker launch or allowance change occurs in an inspection.
        expected_p,expected_y,accounting=module.reconcile_values(before_p,before_y,amendment["new_runtime"])
        expected={module.PROGRESS:expected_p,module.YANG_LEDGER:expected_y}
        if (amendment.get("proposed_runtime_records") != expected
                or amendment.get("conservative_accounting") != accounting
                or accounting.get("body_reservation_refunded_bytes") != 0):
            raise DeploymentError("runtime_reconciliation_must_preserve_all_charges_and_bill_entire_gap")
        plans={}
        for name in ("Stieger_plan","Yang_plan"):
            ref=amendment[name]
            if descriptor(self.root,ref["path"]) != ref:
                raise DeploymentError("runtime_reconciliation_original_plan_SHA_required")
            plans[name]=json.loads((self.root/ref["path"]).read_bytes())
        cap=amendment.get("effective_Stieger_CPU_wall_cap_seconds")
        if (type(cap) is not int or not 172800 <= cap <= 259200
                or amendment.get("original_Stieger_plan_wall_cap_seconds") != plans["Stieger_plan"].get("max_wall_seconds")
                or amendment.get("Yang_plan_wall_cap_seconds_unchanged") != plans["Yang_plan"].get("budgets",{}).get("wall_seconds")
                or expected_p["active_wall_elapsed_s"] >= cap
                or expected_y["elapsed_seconds"] >= plans["Yang_plan"]["budgets"]["wall_seconds"]):
            raise DeploymentError("CPU_runtime_cap_must_keep_actual_prior_consumption")
        self.runtime_recovery_history.append({"transaction":transaction_item,"amendment":item,
            "entire_UTC_gap_billed_seconds":accounting["entire_UTC_gap_billed_seconds"],
            "new_wall_high_water_seconds":accounting["new_wall_high_water_seconds"],
            "body_reservation_refunded_bytes":0,"immutable_original_records_verified":len(originals)})
        self.visit(transaction_item)

    def load_history(self, path: str, *, defer_walk=False):
        """Validate archived code and remaining budgets before walking an old plan.

        A historical source path is resolved to an archived byte-identical
        snapshot only when its full SHA matches. Active new code keeps its own
        path and SHA; an old descriptor never authorizes using new bytes.
        """
        item = descriptor(self.root, path)
        amendment = json.loads((self.root/path).read_bytes())
        if amendment.get("schema_version") != "paper3-source-preparation-parser-repair-budget-conservation-v1":
            raise DeploymentError("unsupported_historical_budget_amendment_schema")
        plans = {}
        for name in ("old_plan", "new_plan", "old_closed_ledger"):
            dep = amendment[name]
            p = relative(self.root, dep["path"])
            actual = descriptor(self.root, p)
            if actual["sha256"] != dep["sha256"] or ("bytes" in dep and actual["bytes"] != dep["bytes"]):
                raise DeploymentError("historical_plan_or_ledger_SHA_mismatch")
            plans[name] = json.loads((self.root/p).read_bytes())
        old, new, ledger = plans["old_plan"], plans["new_plan"], plans["old_closed_ledger"]
        if "active_transfer" in ledger or "wall_intent" in ledger or ledger.get("plan_sha256") != old.get("plan_sha256"):
            raise DeploymentError("historical_budget_ledger_must_be_closed_and_plan_bound")
        for plan in (old, new):
            if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256"):
                raise DeploymentError("historical_plan_semantic_SHA_mismatch")
        usage, caps, remaining = amendment["old_usage"], amendment["aggregate_original_caps"], amendment["new_budgets"]
        if (usage.get("charged") != ledger.get("charged") or usage.get("elapsed_seconds") != ledger.get("elapsed_seconds")
                or new.get("budgets") != remaining):
            raise DeploymentError("historical_usage_and_remaining_budget_disagree")
        for key in ("provider_body_bytes", "r2_get_body_bytes"):
            if (type(usage["charged"].get(key)) is not int or usage["charged"][key] < 0
                    or type(remaining.get(key)) is not int or remaining[key] <= 0
                    or type(caps.get(key)) is not int or caps[key] <= 0
                    or remaining[key]+usage["charged"][key] > caps[key]):
                raise DeploymentError("historical_body_budget_not_conserved")
        if (type(usage.get("elapsed_seconds")) not in {int, float} or not math.isfinite(usage["elapsed_seconds"])
                or usage["elapsed_seconds"] < 0 or remaining["wall_seconds"]+usage["elapsed_seconds"] > caps["wall_seconds"]):
            raise DeploymentError("historical_elapsed_budget_not_conserved")
        snapshots = []
        for old_code in old.get("code", {}).values():
            source = relative(self.root, old_code["path"])
            current = self.root/source
            if current.is_file() and descriptor(self.root, source)["sha256"] == old_code["sha256"]:
                continue
            archived = Path(path).parent/Path(source).name
            snapshot = descriptor(self.root, archived.as_posix())
            if snapshot["sha256"] != old_code["sha256"]:
                raise DeploymentError("historical_code_snapshot_must_match_old_code_SHA")
            self.code_aliases[(source, old_code["sha256"])] = snapshot
            snapshots.append({"historical_source_path": source, "historical_source_sha256": old_code["sha256"],
                              "archive_snapshot": snapshot})
        self.history.append({"amendment": item, "historical_code_bindings": snapshots,
                             "budget_conservation_verified": True})
        self.pending_history.append((item, old))
        if not defer_walk:
            self.walk_history(item, old)

    def walk_history(self, item, old):
        self.visit(item)
        # Stopped parent states retain their original raw/cache proofs. These
        # are historical provenance; only a new final COMPLETE manifest can
        # authorize the corresponding current source/development component.
        old_scope = self.root/"audit/source_extensions_20261019"/old.get("dataset", "")/old["plan_sha256"]
        for state_path in sorted(old_scope.glob("*/state.json")):
            state = json.loads(state_path.read_bytes())
            if state.get("plan_sha256") != old["plan_sha256"]:
                raise DeploymentError("historical_parent_state_plan_mismatch")
            self.path(state_path.relative_to(self.root).as_posix(), context=old["dataset"])

    def path(self, path: str, *, context="Paper3", role=None, receipt_profile=None):
        path = relative(self.root, path)
        if classify(path) == "raw_reference_only":
            raise DeploymentError("raw_payload_cannot_be_a_deployment_root")
        self.visit(self.actual_descriptor(path), context=context, role=role, receipt_profile=receipt_profile)

    def walk(self, value, *, context="Paper3"):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.visit(value, context=context)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    self.walk(child, context=context)
        elif isinstance(value, list):
            for item in value:
                self.walk(item, context=context)

    def _receipt(self, item, data, *, expected_profile=None):
        profile = "source_extension" if "source_extension_profile" in data else "baseline"
        if expected_profile is not None and profile != expected_profile:
            raise DeploymentError("backup_receipt_uses_wrong_IO_profile")
        if profile == "source_extension":
            _, loader = extension._modules()
        else:
            loader = r2_restore
        rows, sha = loader.load_verified_receipt(self.root/item["path"])
        if sha != item["sha256"]:
            raise DeploymentError("receipt_changed_during_closure_validation")
        selected, excluded = [], []
        self.receipts[item["path"]] = {"io_profile": profile, "receipt": item, "paths": selected,
                                     "excluded_raw_paths": excluded, "full_GET_receipt_verified": True}
        for row in rows:
            path = relative(self.root, row["path"])
            if row["role"] == "source_development_raw" or classify(path) == "raw_reference_only":
                excluded.append(path)
                self.raw[(path, row["sha256"])] = {**{k: row[k] for k in extension.ENTRY_FIELDS}, "payload_selected": False}
                if row["role"] == "source_development_raw":
                    self.raw_get_backed.add((path, row["sha256"]))
                continue
            self.r2_rows.setdefault((path, row["sha256"], row["bytes"]), []).append(item["path"])
            selected.append(path)
            self.visit(row, context=row["dataset"], role=row["role"])
        self.receipts[item["path"]]["restore_required"] = bool(selected)


def default_manifests(root: Path) -> dict:
    result = {"source4": "cache/source_v2/source_cache.json", "Stieger2021": "cache/stieger_v2/stieger_cache.json",
              "Ma2022": "cache/source_extensions_20261019/Ma2022/development_cache_native_v2.json"}
    for dataset in ("Yang2025", "Dreyer2023"):
        matches = sorted((root/extension.RESULT_ROOT/dataset).glob("*/source_cache.json"))
        valid = []
        for path in matches:
            try:
                complete_component(root, dataset, path.relative_to(root).as_posix())
                valid.append(path)
            except Exception:
                continue
        result[dataset] = valid[0].relative_to(root).as_posix() if len(valid) == 1 else None
    return result


def closed_ledger(root: Path, path: str, sha: str):
    data = json.loads((root/path).read_bytes())
    if data.get("plan_sha256") != sha or "active_transfer" in data or "wall_intent" in data:
        raise DeploymentError("final_coordinator_ledger_must_be_closed_and_plan_bound")
    return path


def verified_final_commit(root: Path, manifest: dict, receipt: dict, dataset: str):
    if descriptor(root, receipt["path"]) != receipt:
        raise DeploymentError("final_backup_receipt_descriptor_mismatch")
    _, restore = extension._modules()
    rows, _ = restore.load_verified_receipt(root/receipt["path"])
    matches = [r for r in rows if r["path"] == manifest["path"] and r["sha256"] == manifest["sha256"]
               and r["bytes"] == manifest["bytes"] and r["dataset"] == dataset and r["role"] == "result"]
    if len(matches) != 1:
        raise DeploymentError("final_manifest_must_have_exact_complete_R2_GET_receipt")


def complete_component(root: Path, name: str, path: str) -> tuple[dict, list[str]]:
    data = json.loads((root/path).read_bytes())
    if name == "source4":
        if (descriptor(root, path)["sha256"] != SOURCE4_SHA or data.get("status") != "SOURCE_CACHE_COMPLETE"
                or len(data.get("shards", [])) != EXPECTED[name]
                or {s.get("dataset") for s in data["shards"]} != SOURCE_IDS):
            raise DeploymentError("exact_frozen_four_source_COMPLETE_manifest_required")
        receipts = ["audit/source_full/cache_backup_receipt.json"]
    elif name == "Stieger2021":
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
                if not isinstance(item,dict) or descriptor(root,item["path"]) != item:
                    raise DeploymentError("Stieger_completed_CPU_recovery_artifacts_required")
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
    elif name in {"Yang2025", "Dreyer2023"}:
        if (data.get("schema_version") != "paper3-source-extension-cache-v1"
                or data.get("status") != "SOURCE_EXTENSION_CACHE_COMPLETE" or data.get("dataset") != name
                or data.get("role") != "source_train" or data.get("source_only") is not True
                or data.get("unique_role_within_branch") is not True
                or data.get("record_count") != EXPECTED[name] or len(data.get("shards", [])) != EXPECTED[name]):
            raise DeploymentError("all_frozen_source_extension_records_COMPLETE_required")
        sha = data.get("plan_sha256")
        if not re.fullmatch("[0-9a-f]{64}", str(sha)):
            raise DeploymentError("source_extension_plan_SHA_required")
        status_path = f"audit/source_extensions_20261019/{name}/{sha}/coordinator_status.json"
        state = json.loads((root/status_path).read_bytes())
        if state.get("status") != "SOURCE_EXTENSION_CACHE_COMPLETE" or state.get("record_count") != EXPECTED[name]:
            raise DeploymentError("actual_source_extension_final_R2_commit_required")
        if state.get("manifest") != descriptor(root, path):
            raise DeploymentError("source_extension_final_manifest_descriptor_mismatch")
        verified_final_commit(root, state["manifest"], state["backup_receipt"], name)
        ledger = closed_ledger(root, f"audit/source_extensions_20261019/{name}/{sha}/budget_ledger.json", sha)
        receipts = [state["backup_receipt"]["path"], ledger, status_path]
    else:
        if (data.get("schema_version") != "paper3-Ma2022-development-cache-v1"
                or data.get("status") != "ALL125_DEVELOPMENT_PAIRS_VERIFIED" or data.get("sessions") != EXPECTED[name]
                or len(data.get("shards", [])) != EXPECTED[name] or data.get("role") != "independent_development_unassigned"
                or any(data.get(k) is not False for k in ("training_ready", "source_training_allowed",
                    "source3_training_pool_member", "primary11_compatible", "original_B20_prefix_recoverable",
                    "scientific_pool_assignment_frozen"))):
            raise DeploymentError("all125_Ma_development_COMPLETE_with_training_gates_required")
        sha = data.get("plan_sha256")
        if not re.fullmatch("[0-9a-f]{64}", str(sha)):
            raise DeploymentError("Ma_development_plan_SHA_required")
        status_path = f"audit/source_extensions_20261019/Ma2022/{sha}/coordinator_status.json"
        state = json.loads((root/status_path).read_bytes())
        if (state.get("status") != "MA2022_DEVELOPMENT_CACHE_COMPLETE" or state.get("verified_records") != 125
                or state.get("manifest") != descriptor(root, path)):
            raise DeploymentError("actual_Ma_final_R2_commit_required")
        verified_final_commit(root, state["manifest"], state["backup_receipt"], name)
        ledger = closed_ledger(root, f"audit/source_extensions_20261019/Ma2022/{sha}/budget_ledger.json", sha)
        receipts = [state["backup_receipt"]["path"], ledger, status_path]
    return data, receipts


def inspect(root: Path, *, manifests=None, include_software=True, include_wheels=True, history=None) -> dict:
    root = Path(os.path.abspath(root))
    closure = Closure(root)
    if include_software:
        try:
            closure.register_precision_fix()
        except Exception as exc:
            closure.error(str(exc) if isinstance(exc,DeploymentError) else "classical_precision_history_unverified",
                          getattr(exc,"path",None) or PRECISION_AUDIT,"software")
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


def final_manifest(report: dict) -> dict:
    if (report.get("status") != "CACHE_CLOSURE_READY_FOR_BACKUP" or report.get("all_required_data_COMPLETE") is not True
            or report.get("gaps") or report.get("raw_payloads_selected") != 0):
        raise DeploymentError("all_actual_COMPLETE_data_and_verified_closure_required_before_final_manifest")
    if digest({k: v for k, v in report.items() if k != "closure_sha256"}) != report.get("closure_sha256"):
        raise DeploymentError("readiness_report_SHA_mismatch")
    batches = {name: {"schema_version": 1, "files": []} for name in ("baseline", "source_extension")}
    for row in report["files"]:
        path = row["path"]
        if Path(path).suffix.lower() in RAW_SUFFIXES | {".npz", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}:
            raise DeploymentError("raw_payload_cannot_enter_final_deployment_manifest")
        protocol_member = row.get("original_protocol_manifest_member") is True and path.startswith("protocol/")
        if (not protocol_member and classify(path) == "raw_reference_only") or row["role"] == "source_development_raw":
            raise DeploymentError("raw_payload_cannot_enter_final_deployment_manifest")
        entry = {key: row[key] for key in extension.ENTRY_FIELDS}
        (extension.validate_entry if row["io_profile"] == "source_extension" else r2_backup.validate_entry)(entry)
        batches[row["io_profile"]]["files"].append(entry)
    return {"schema_version": SCHEMA, "status": "FIXED_LOCAL_CACHE_ONLY_BACKUP_MANIFEST_NOT_YET_UPLOADED",
        "closure_sha256": report["closure_sha256"], "backup_manifests": batches,
        "verified_existing_restore_groups": [group for group in report["verified_restore_groups"] if group["paths"]],
        "raw_payloads_selected": 0, "network_operations": 0, "fresh_restore_executed": False,
        "restore_policy": "Use only each listed exact paths selection; never --all-approved on a mixed raw receipt. Source-extension receipts use the document-bound source-extension restore API; baseline receipts use the original restore API.",
        "Ma2022_scientific_training_ready": False, "training_started": False, "GPU_actions": 0}


def write_once(path: Path, value):
    r2_restore.no_symlink_components(path)
    extension._credential_components(path)
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise DeploymentError("existing_deploy_artifact_not_overwritten")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".deploy-", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
