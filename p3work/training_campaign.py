"""Local W4--W6 compiler and resumable dispatcher, with no server/network actions.

Planning reads metadata only. Execution calls the existing production gates;
this file does not confer a data capability, construct a model, or implement a
fit. Missing dependencies remain pending, and an uncertain existing attempt is
left for its original kernel's recovery rules. Optional source extensions and
external evaluations deliberately have no dispatch here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import multiprocessing
import signal
import threading
import time

from .contracts import SOURCE_IDS, digest, file_digest, validate_signal_contract
from . import method_registry as registry
from .development_evaluation import DevelopmentJob, SEEDS
from .meta_training import MetaJob
from .recovery_budget import PersistentWallBudget
from .training_state import atomic_json

SCHEMA = "paper3-mainline-training-campaign-v1"
OUTPUT = "runs/training_campaign_v1"
CLAIM = "runs/training_campaign_v1_claim.json"
LOCK = "runs/.training_campaign_v1.lock"
PILOT = "runs/controlled_source_pilot_seed20261018"
SOURCE = "runs/source_training_v2"
CLASSICAL = "runs/source_classical_v2"
LEARNED = tuple(m for m in registry.METHOD_IDS if m in registry.LEARNED_METHODS)
BOOTSTRAP = {"pilot_wall_seconds": 1800, "classical_wall_seconds": 1800,
             "meta_engineering_wall_seconds": 1800, "profile_calibration_seconds": 120}
CODE = ("p3work/training_campaign.py", "tools/run_training_campaign.py")
ACTION_STATUSES = {"pilot": "ENGINEERING_PILOT_COMPLETE", "source": "SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING",
    "classical": "SOURCE_CLASSICAL_CORE_COMPLETE", "reference": "SOURCE_REFERENCE_COMPLETE",
    "meta_engineering": "META_ENGINEERING_COMPLETE", "meta_screening": "META_JOB_COMPLETE",
    "meta_confirmation": "META_JOB_COMPLETE", "meta_selection": "META_CONFIGURATION_FROZEN",
    "calibration_profile": "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE", "development": "DEVELOPMENT_EVALUATION_COMPLETE",
    "development_selection": "DEVELOPMENT_CONFIGURATION_FROZEN", "strongest_control": "STRONGEST_DEVELOPMENT_CONTROL_FROZEN",
    "freeze": "IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE", "internal": "ONE_TIME_INTERNAL_CHECK_COMPLETE"}


class CampaignError(ValueError):
    pass


def local(root, relative, *, exists=True):
    if not isinstance(relative, str) or not relative:
        raise CampaignError("explicit_root_relative_path_required")
    path = Path(relative)
    if (path.is_absolute() or ".." in path.parts or "legacy_baseline_v1" in path.parts
            or path.suffix.lower() in {".npz", ".pkl", ".pickle"}):
        raise CampaignError("campaign_path_escapes_new_research_root")
    current = Path(root)
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise CampaignError("campaign_artifact_symlink_forbidden")
    result = (Path(root) / path).resolve(strict=exists)
    if not result.is_relative_to(Path(root).resolve()):
        raise CampaignError("campaign_path_escapes_new_research_root")
    return result


def read(path):
    return json.loads(Path(path).read_text())


def descriptor(root, path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(Path(root).resolve()):
        raise CampaignError("artifact_outside_campaign_root")
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def checked(root, ref):
    if not isinstance(ref, dict) or not {"path", "sha256"} <= set(ref):
        raise CampaignError("actual_SHA_bound_artifact_required")
    path = local(root, ref.get("path"))
    if not path.is_file() or file_digest(path) != ref.get("sha256"):
        raise CampaignError("artifact_SHA_mismatch")
    return path


def immutable(root, relative, value):
    path = local(root, relative, exists=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise CampaignError("immutable_campaign_artifact_changed:" + relative)
    else:
        # The global lock serializes publication; atomic_json also fsyncs data.
        atomic_json(path, value)
    return descriptor(root, path)


def _existing(root, relative, status=None):
    path = local(root, relative, exists=False)
    if not path.is_file():
        return None
    value = read(path)
    if status is not None and value.get("status") != status:
        return None
    return descriptor(root, path)


def meta_job(method, configuration, seed=SEEDS[0]):
    return MetaJob(method, configuration.get("dimension", configuration.get("rank")),
        configuration["radius"], seed, "screening" if seed == SEEDS[0] else "confirmation")


def meta_receipt(job):
    return "runs/meta/" + job.identity + "/receipt.json"


def source_receipt(seed):
    return SOURCE + f"/f0_receipt_seed-{seed}.json"


def reference_receipt(seed):
    return OUTPUT + f"/references/seed-{seed}/receipt.json"


def selection_path(method):
    return OUTPUT + "/selections/" + method + ".json"


def evaluation_path(job):
    return OUTPUT + "/development/" + job.identity + "/receipt.json"


def profile_path(method):
    return OUTPUT + "/profiles/" + method + "/receipt.json"


def scope_identity(root):
    """Stable authorization; W3 may finish after planning, before execution."""
    root = Path(root).resolve(strict=True)
    tasks = read(root / "status/tasks.json")
    contract = read(root / "contracts/input_contract.json")
    signal_sha = validate_signal_contract(contract)
    cache = tasks.get("W2", {}).get("cache_manifest")
    if cache is not None:
        checked(root, cache)
    executed = Path(__file__).resolve().parents[1]
    for relative in CODE:
        if local(root, relative).read_bytes() != (executed / relative).read_bytes():
            raise CampaignError("recorded_campaign_code_differs_from_executed_package")
    return {"schema_version": SCHEMA, "scope": "base4_Stieger40_11_11_W4_through_one_W6_internal_check",
        "source_datasets": sorted(SOURCE_IDS), "seeds": list(SEEDS),
        "input_contract_sha256": signal_sha, "input_contract": descriptor(root, root / "contracts/input_contract.json"),
        "source_cache": cache,
        "development_cache": tasks.get("W3", {}).get("cache_manifest") if tasks.get("W3", {}).get("status") == "COMPLETE" else None,
        "split_sha256": contract["window_audit_receipt"]["split_sha256"],
        "code": {p: file_digest(local(root, p)) for p in CODE},
        "source_extensions_enabled": False, "external_query_permission": False,
        "stop_after": "ONE_TIME_INTERNAL_CHECK_COMPLETE"}


def inventory(root):
    """Only actual completed receipt files, never fabricated future artifacts.

    This inventory is a metadata index, not a production-verification receipt.
    Each production consumer deeply checks its dependencies before execution.
    """
    root = Path(root).resolve(strict=True)
    seeds, learned, banks = {}, [], {"B4": {}, "B5": {}}
    for seed in SEEDS:
        source = _existing(root, source_receipt(seed), "SOURCE_F0_COMPLETE")
        reference = _existing(root, reference_receipt(seed), "SOURCE_REFERENCE_COMPLETE")
        if source and reference:
            seeds[str(seed)] = {"source_f0": source, "source_reference": reference}
    for method in LEARNED:
        configs = registry.registered_grid(method)
        for seed in SEEDS:
            if seed != SEEDS[0]:
                selection = _existing(root, selection_path(method), "DEVELOPMENT_CONFIGURATION_FROZEN")
                if not selection:
                    continue
                configs = (read(checked(root, selection))["configuration"],)
            for config in configs:
                job = meta_job(method, config, seed)
                receipt = _existing(root, meta_receipt(job), "META_JOB_COMPLETE")
                if receipt:
                    value = read(checked(root, receipt))
                    if value.get("job") != asdict(job) or value.get("synthetic") is not False:
                        raise CampaignError("meta_inventory_receipt_identity_mismatch")
                    learned.append({"method": method, "seed": seed, "configuration": dict(config), "receipt": receipt})
    for method, key, values in (("B4", "m", (1, 2, 3)), ("B5", "shrink", (.01, .05, .1))):
        for value in values:
            relative = CLASSICAL + f"/bank/ALL_SOURCE/{method}/{key}-{value}/fit_receipt.json"
            ref = _existing(root, relative, "SOURCE_CLASSICAL_FIT_COMPLETE")
            if ref:
                banks[method][str(value)] = ref
    return {"schema_version": "paper3-development-artifact-catalog-v1", "seeds": seeds,
            "learned_artifacts": learned, "source_banks": banks}


def publish_catalog(root):
    value = inventory(root)
    return immutable(root, OUTPUT + "/catalogs/" + digest(value) + ".json", value)


def compile_campaign(root):
    """Read-only readiness and exact full-grid counts. No CUDA or payload read."""
    root = Path(root).resolve(strict=True)
    identity = scope_identity(root)
    tasks = read(root / "status/tasks.json")
    gaps = [s + "_COMPLETE" for s in ("W1", "W2", "W3") if tasks.get(s, {}).get("status") != "COMPLETE"]
    stage_bindings = {"W2_cache": identity["source_cache"], "W3_cache": None, "participant_split": None}
    if identity["source_cache"] is None:
        gaps.append("W2_actual_cache_manifest")
    if tasks.get("W3", {}).get("status") == "COMPLETE":
        ref = tasks["W3"].get("cache_manifest")
        if ref is None:
            gaps.append("W3_actual_cache_manifest")
        else:
            manifest = read(checked(root, ref))
            split_ref = manifest.get("participant_split")
            split = read(checked(root, split_ref))
            if (manifest.get("schema_version") != "paper3-stieger-cache-v2"
                    or manifest.get("status") != "STIEGER_CACHE_COMPLETE"
                    or manifest.get("synthetic") is not False or manifest.get("session_count") != 598
                    or manifest.get("participant_count") != 62
                    or manifest.get("input_contract_sha256") != identity["input_contract_sha256"]
                    or manifest.get("split_sha256") != identity["split_sha256"]
                    or split.get("split_sha256") != identity["split_sha256"]
                    or {k: len(v) for k, v in split.get("partitions", {}).items()} != {
                        "meta_train": 40, "meta_validation": 11, "internal_holdout": 11}):
                raise CampaignError("W3_metadata_must_bind_real_base4_Stieger598_40_11_11_input_split")
            stage_bindings.update(W3_cache=ref, participant_split=split_ref)
    catalog = inventory(root)
    plan = {"schema_version": SCHEMA, "status": "PENDING_DATA_GATES" if gaps else "READY_FOR_USER_STARTED_GPU",
        "identity": identity, "minimum_gaps": gaps, "execute": False,
        "stage_metadata_bindings": stage_bindings,
        "planned_counts": {"source_engineering_updates": 200, "neural_source_fits": 15,
            "classical_core_fits": 10, "classical_additional_bank_fits": 6,
            "meta_engineering_jobs": len(LEARNED), "meta_engineering_outer_updates_per_job": 100,
            "meta_screening_fits": 45, "meta_confirmation_additional_fits": 10,
            "development_calibration_profile_settings": 118, "development_screening_jobs": 118,
            "development_confirmation_additional_jobs": 28, "one_time_internal_checks": 1},
        "registered_grids": {m: list(registry.registered_grid(m)) for m in registry.METHOD_IDS},
        "actual_artifact_inventory": catalog, "inventory_is_production_verification": False,
        "selection": "complete_registered_grids_then_BA_BCE_checkpoint_grid_ties; Bstar_all11_controls_3seeds_0.005BA_then_same_device_time_then_id",
        "bootstrap_caps_are_limits_not_measured_estimates": dict(BOOTSTRAP),
        "derived_budget_rule": "measured evidence * exact registered work * safety_factor + fixed serialization allowance; all bounds capped by campaign authorization",
        "default_max_cumulative_wall_seconds": 43200, "global_claim": CLAIM,
        "resume": "same_fixed_job_and_latest_verified_checkpoint; completed_artifacts_reverified_without_refit",
        "internal_budget_basis": "frozen W3 registry metadata upper bound only, never open internal catalog to plan",
        "optional_research_extensions": {"source_extensions": "independent_f0_preparation_only_not_ready_for_fair_downstream_comparison",
            "C3_montage": "separate_campaign_not_enabled_here", "new_source_into_old_gate": False},
        "backup_manifest_is_upload_receipt": False, "GPU_checked": False,
        "network_operations": 0, "server_actions": False, "fits_dispatched": 0,
        "W7_freeze_created": False, "external_query_accessed": False}
    return plan


@contextmanager
def campaign_lock(root, identity):
    """Root-wide nonblocking lock and permanent same-scope claim."""
    root = Path(root).resolve(strict=True)
    lock_path = local(root, LOCK, exists=False)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CampaignError("another_mainline_campaign_owns_the_global_lock") from None
        try:
            prior_claim = local(root, CLAIM, exists=False).exists()
            immutable(root, CLAIM, {"schema_version": SCHEMA, "identity": identity,
                "identity_sha256": digest(identity), "fixed_output": OUTPUT,
                "claim_must_not_be_deleted_or_renamed_to_retry": True})
            yield prior_claim
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def measured_meta_budget(proof, job):
    """No timing fabrication: includes both validation and all profile overhead."""
    if (proof.get("status") != "META_ENGINEERING_COMPLETE" or proof.get("synthetic") is not False
            or proof.get("outer_updates_completed") != 100 or proof.get("validation_checkpoints") != 2
            or proof.get("job", {}).get("method") != job.method):
        raise CampaignError("actual_same_method_100_update_meta_profile_required")
    seconds = proof.get("active_wall_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise CampaignError("positive_measured_meta_profile_wall_required")
    return {"formula": "ceil(4 * engineering_total_wall * (registered_outer_updates / 100) + 120)",
        "measured_engineering_wall_seconds": seconds, "safety_factor": 4,
        "derived_wall_seconds": math.ceil(4 * seconds * (job.updates / 100) + 120)}


def measured_evaluation_budget(maximum, sessions):
    if (type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0
            or type(sessions) is not int or sessions < 1):
        raise CampaignError("positive_real_calibration_and_catalog_session_count_required")
    per = math.ceil(4 * maximum + 5)
    return {"formula": "per_calibration=ceil(4*measured_max+5); wall=ceil(sessions*(per_calibration+5)+120)",
        "measured_maximum_calibration_seconds": maximum, "sessions": sessions,
        "max_calibration_seconds": per, "derived_wall_seconds": math.ceil(sessions * (per + 5) + 120)}


def verify_budget_dependencies(root, value):
    for ref in value.get("evidence", []):
        checked(root, ref)


def _budget(root, action, body, *, evidence=(), derivation=None, campaign_limit):
    """Immutable bounds: reusing a job never silently replaces its budget."""
    value = {**body, "evidence": list(evidence), "campaign_identity_sha256": digest(scope_identity(root)),
        "derivation": derivation or {"basis": "explicit_unmeasured_bootstrap_cap_not_prediction"},
        "campaign_cumulative_limit_seconds_at_creation": campaign_limit}
    relative = OUTPUT + "/budgets/" + action + ".json"
    existing = local(root, relative, exists=False)
    if existing.exists():
        prior = read(existing)
        # A larger top-level authorization is not a child budget amendment.
        # Preserve the original child bounds, even after global limit changes.
        def stable(item):
            return {k: v for k, v in item.items() if k not in {
                "max_wall_seconds", "max_gpu_seconds", "campaign_cumulative_limit_seconds_at_creation"}}
        if stable(prior) != stable(value):
            raise CampaignError("persisted_job_budget_differs_preserve_original:" + action)
        verify_budget_dependencies(root, prior)
        return descriptor(root, existing)
    verify_budget_dependencies(root, value)
    return immutable(root, relative, value)


def backup_manifest(root):
    """Finite result-only manifest; never credentials/configs, never an upload."""
    root = Path(root).resolve(strict=True)
    folders = (OUTPUT, PILOT, SOURCE, CLASSICAL, "runs/meta", "runs/internal_validation_once")
    files = {}
    for folder in folders:
        base = local(root, folder, exists=False)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.name.endswith(".lock"):
                continue
            if path.name in {"backup_manifest.json", "campaign_progress.json"}:
                continue
            if path.suffix not in {".json", ".npy", ".pt"}:
                continue
            ref = descriptor(root, path)
            files[ref["path"]] = {**ref, "bytes": path.stat().st_size, "role": "result", "dataset": "Paper3_mainline"}
    for relative in (CLAIM, "runs/controlled_pilot_job_claim.json", "runs/internal_validation_once_claim.json"):
        path = local(root, relative, exists=False)
        if path.exists():
            ref = descriptor(root, path)
            files[relative] = {**ref, "bytes": path.stat().st_size, "role": "result", "dataset": "Paper3_mainline"}
    value = {"schema_version": 1, "files": [files[p] for p in sorted(files)],
        "upload_performed": False, "full_GET_verified": False, "contains_credentials": False}
    path = local(root, OUTPUT + "/backup_manifest.json", exists=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    return descriptor(root, path)


def _profile_methods(root):
    return [m for m in registry.METHOD_IDS if _existing(root, profile_path(m), "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE")]


class Production:
    """Thin, explicit adapters to the original real production interfaces."""
    def __init__(self, root, maximum):
        self.root, self.maximum = Path(root), maximum
        self.cache = None

    def _source_context(self):
        if self.cache is None:
            from tools.run_controlled_pilot import verify_inputs
            self.cache = verify_inputs(self.root)
        return self.cache

    def budget(self, action, body, evidence=(), derivation=None):
        return _budget(self.root, action, body, evidence=evidence, derivation=derivation, campaign_limit=self.maximum)

    def perform(self, action):
        """Returns an actual receipt descriptor; no guessed output status."""
        kind, method = action["kind"], action.get("method")
        root = self.root
        guard_child_recovery(root, action)
        from . import development_evaluation as dev
        if kind == "pilot":
            from tools.run_controlled_pilot import execute_pilot
            from .source_training import verify_engineering_pilot
            cache = self._source_context()
            path = _existing(root, PILOT + "/engineering_receipt.json", "ENGINEERING_PILOT_COMPLETE")
            if path:
                verify_engineering_pilot(cache, path["path"])
            else:
                execute_pilot(cache, PILOT, user_started_gpu=True)
            verify_engineering_pilot(cache, PILOT + "/engineering_receipt.json")
            return descriptor(root, root / PILOT / "engineering_receipt.json")
        if kind == "source":
            from .source_training import build_training_plan, execute_plan, verify_engineering_pilot
            from tools.train_source import cuda_runtime
            cache = self._source_context()
            plan = build_training_plan(cache, profile="development")
            runtime = cuda_runtime()
            pilot_ref = descriptor(root, root / PILOT / "engineering_receipt.json")
            pilot = read(checked(root, pilot_ref))
            evidence = verify_engineering_pilot(cache, pilot_ref["path"], runtime=runtime)
            measured = pilot["total_monotonic_wall_seconds"]
            wall = min(self.maximum, math.ceil(4 * measured / 200 * plan["max_neural_optimizer_updates"] + 600))
            budget = self.budget("source", {"stage": "source_development", "max_wall_seconds": wall,
                "max_neural_optimizer_updates": plan["max_neural_optimizer_updates"], "max_jobs": 15}, [pilot_ref],
                {"formula": "min(campaign_limit,ceil(4*pilot_total_wall/200*registered_max_updates+600))",
                 "pilot_total_wall_seconds": measured, "training_plan_sha256": plan["plan_sha256"]})
            result = execute_plan(cache, plan, root / SOURCE, runtime=runtime, pilot_evidence=evidence,
                max_wall_seconds=read(checked(root, budget))["max_wall_seconds"])
            if result["status"] != "NEURAL_SOURCE_COMPLETE_CLASSICAL_PENDING":
                return None
            return descriptor(root, root / SOURCE / "source_profile_development.json")
        if kind == "classical":
            from .classical_training import prepare_classical_sources
            plan = prepare_classical_sources(root, include_source_bank=True)["plan"]
            budget = self.budget("classical", {"stage": "source_classical_cpu", "max_jobs": 16,
                "core_jobs": 10, "additional_bank_jobs": 6, "max_fit_attempts": 16,
                "max_wall_seconds": min(self.maximum, BOOTSTRAP["classical_wall_seconds"]),
                "max_fit_input_bytes": max(j["input_bytes"] for j in plan["jobs"]),
                "max_gpu_hours": 0, "max_download_bytes": 0}, [plan["cache_manifest"]],
                {"basis": "explicit_CPU_fit_timeout; input_byte_limit_from_actual_full_fixed_plan", "plan_sha256": plan["plan_sha256"]})
            result = prepare_classical_sources(root, execute=True, include_source_bank=True,
                budget_path=budget["path"], output=CLASSICAL)
            if result.get("status") != "SOURCE_CLASSICAL_CORE_COMPLETE":
                return None
            return descriptor(root, root / CLASSICAL / "receipt.json")
        if kind == "reference":
            from .source_reference_io import prepare_source_reference, load_verified_reference
            seed = action["seed"]
            source = source_receipt(seed)
            output = str(Path(reference_receipt(seed)).parent)
            contract = read(root / "contracts/input_contract.json")
            if not (root / reference_receipt(seed)).exists():
                prepare_source_reference(root, source, contract, execute=True, output=output)
            load_verified_reference(root, reference_receipt(seed), source, contract)
            return descriptor(root, root / reference_receipt(seed))
        if kind in {"meta_engineering", "meta_screening", "meta_confirmation"}:
            from .meta_training import execute_meta_job, verify_meta_inputs
            from .meta_engineering import verify_meta_engineering
            job = MetaJob(**action["job"])
            source, reference = source_receipt(job.seed), reference_receipt(job.seed)
            split = _split_descriptor(root)["path"]
            engineering = MetaJob(method, max((1, 2, 4) if method in {"B10", "B11"} else (2, 4, 8)), .35,
                                  SEEDS[0], "engineering")
            engineering_ref = _existing(root, meta_receipt(engineering), "META_ENGINEERING_COMPLETE")
            evidence = [descriptor(root, root / source), descriptor(root, root / reference), _split_descriptor(root)]
            extra = {}
            if job.phase == "engineering":
                wall, derivation = min(self.maximum, BOOTSTRAP["meta_engineering_wall_seconds"]), None
            else:
                if not engineering_ref:
                    raise CampaignError("measured_meta_engineering_receipt_missing")
                selection = action.get("meta_selection")
                _, _, _, _, inputs = verify_meta_inputs(root, source_receipt=source, reference_receipt=reference,
                    split_path=split, job=job, selection_receipt=selection)
                proof = verify_meta_engineering(root, engineering_ref, job=job, inputs=inputs)
                derivation = measured_meta_budget(proof, job)
                wall = min(self.maximum, derivation["derived_wall_seconds"])
                evidence.append(engineering_ref)
                extra["measured_meta_engineering_receipt"] = engineering_ref
            budget = self.budget("meta-"+job.identity, {"stage": "meta_training", "job_sha256": job.identity,
                "max_outer_updates": job.updates, "max_jobs": 1, "max_wall_seconds": wall, **extra}, evidence, derivation)
            latest = root / "runs/meta" / job.identity / "latest_checkpoint.json"
            resume = None
            if latest.exists():
                checkpoint = read(latest)
                if checkpoint.get("relative_to_job_output") is not True or Path(checkpoint["path"]).name != checkpoint["path"]:
                    raise CampaignError("invalid_same_job_latest_checkpoint")
                resume_ref = {"path": (latest.parent / checkpoint["path"]).relative_to(root).as_posix(), "sha256": checkpoint["sha256"]}
                resume = checked(root, resume_ref).relative_to(root).as_posix()
            result = execute_meta_job(root, job=job, source_receipt=source, reference_receipt=reference,
                split_path=split, budget_path=budget["path"], user_started_gpu=True,
                resume=resume, selection_receipt=action.get("meta_selection"))
            if result.get("status") not in {"META_JOB_COMPLETE", "META_ENGINEERING_COMPLETE"}:
                return None
            return descriptor(root, root / meta_receipt(job))
        if kind == "meta_selection":
            from .meta_artifacts import freeze_development_configuration
            catalog = inventory(root)
            rows = [r["receipt"]["path"] for r in catalog["learned_artifacts"] if r["method"] == method and r["seed"] == SEEDS[0]]
            source = catalog["seeds"][str(SEEDS[0])]
            output = OUTPUT + "/meta_selections/" + method + ".json"
            local(root, output, exists=False).parent.mkdir(parents=True, exist_ok=True)
            freeze_development_configuration(root, rows, method=method, output=output,
                source_receipt=source["source_f0"], reference_receipt=source["source_reference"],
                contract=read(root / "contracts/input_contract.json"), split_sha256=scope_identity(root)["split_sha256"], execute=True)
            return descriptor(root, root / output)
        if kind == "calibration_profile":
            catalog = publish_catalog(root)
            plan = dev.build_calibration_profile_plan((method,))
            limit = BOOTSTRAP["profile_calibration_seconds"]
            wall = min(self.maximum, len(plan["jobs"])*(limit+10)+120)
            budget = self.budget("profile-"+method, {"stage": "development_calibration_profile",
                "plan_sha256": plan["plan_sha256"], "max_jobs": 1, "max_download_bytes": 0,
                "max_fit_attempts": len(plan["jobs"]), "max_calibration_seconds": limit,
                "max_wall_seconds": wall, "max_gpu_seconds": wall}, [catalog])
            result = dev.execute_calibration_profile(root, artifact_catalog=catalog, budget=budget,
                methods=(method,), user_started_gpu=True, output=str(Path(profile_path(method)).parent))
            return result.get("receipt")
        if kind == "development":
            job = DevelopmentJob(**action["job"])
            catalog = publish_catalog(root)
            profile_ref = descriptor(root, root / profile_path(job.method))
            profile = dev.verify_calibration_profile(root, profile_ref)
            proof = read(checked(root, profile_ref))
            sessions = sum(p["query_eligible"] for p in proof["catalog"]["plans"])
            derivation = measured_evaluation_budget(profile.maximum_calibration_seconds[job.method], sessions)
            wall = min(self.maximum, derivation["derived_wall_seconds"])
            budget = self.budget("development-"+job.identity, {"stage": "development_evaluation",
                "job_sha256": job.identity, "max_jobs": 1, "max_download_bytes": 0,
                "max_fit_attempts": sessions, "max_calibration_seconds": derivation["max_calibration_seconds"],
                "max_wall_seconds": wall, "max_gpu_seconds": wall, "measured_timing_profile": profile_ref},
                [catalog, profile_ref], derivation)
            result = dev.execute_validation_job(root, job=job, artifact_catalog=catalog, budget=budget,
                user_started_gpu=True, configuration_receipt=action.get("configuration_receipt"),
                output=str(Path(evaluation_path(job)).parent))
            return result.get("receipt")
        if kind == "development_selection":
            refs = [descriptor(root, root / evaluation_path(DevelopmentJob(method, i)))
                    for i in range(len(registry.registered_grid(method)))]
            dev.freeze_screening_configuration(root, method=method, screening_receipts=refs,
                output=selection_path(method), execute=True)
            return descriptor(root, root / selection_path(method))
        if kind == "strongest_control":
            configs, confirmations = _selected_development(root)
            output = OUTPUT + "/strongest_control.json"
            dev.select_strongest_control(root, configuration_receipts={m: configs[m] for m in dev.CONTROLS},
                confirmation_receipts=[r for r in confirmations if read(checked(root, r))["method"] in dev.CONTROLS],
                output=output, execute=True)
            return descriptor(root, root / output)
        if kind == "freeze":
            from .internal_validation import freeze_development
            configs, confirmations = _selected_development(root)
            spec = {"schema_version": "paper3-internal-freeze-spec-v1", "configuration_receipts": configs,
                "confirmation_receipts": confirmations,
                "strongest_control_receipt": descriptor(root, root / OUTPUT / "strongest_control.json"),
                "artifact_catalog": publish_catalog(root)}
            immutable(root, OUTPUT+"/internal_freeze_spec.json", spec)
            freeze_development(root, spec, output=OUTPUT+"/pre_internal_freeze.json", execute=True)
            return descriptor(root, root / OUTPUT / "pre_internal_freeze.json")
        if kind == "internal":
            from .internal_validation import execute_internal, verify_freeze
            freeze = descriptor(root, root / OUTPUT / "pre_internal_freeze.json")
            frozen = verify_freeze(root, freeze)
            profiles = [descriptor(root, root / profile_path(m)) for m in registry.METHOD_IDS]
            maxima = [max(dev.verify_calibration_profile(root, r).maximum_calibration_seconds.values()) for r in profiles]
            # Public frozen metadata supplies a conservative session cap. This
            # does not materialize an internal episode or inspect query labels.
            count = internal_session_upper_bound(root)
            derivation = measured_evaluation_budget(max(maxima), count*len(registry.METHOD_IDS)*len(SEEDS))
            derivation["session_count_is_metadata_upper_bound_not_actual_fits"] = True
            wall = min(self.maximum, derivation["derived_wall_seconds"])
            budget = self.budget("internal", {"stage": "internal_development_check",
                "freeze_sha256": frozen["freeze_sha256"], "max_jobs": 1, "max_download_bytes": 0,
                "max_fit_attempts": count*len(registry.METHOD_IDS)*len(SEEDS),
                "max_calibration_seconds": derivation["max_calibration_seconds"],
                "max_wall_seconds": wall, "max_gpu_seconds": wall}, [freeze, *profiles], derivation)
            result = execute_internal(root, freeze=freeze, budget=budget, user_started_gpu=True)
            if result.get("status") != "ONE_TIME_INTERNAL_CHECK_COMPLETE":
                return None
            return descriptor(root, root / "runs/internal_validation_once/receipt.json")
        raise CampaignError("unregistered_campaign_action")


def _split_descriptor(root):
    tasks = read(Path(root) / "status/tasks.json")
    manifest = read(checked(root, tasks["W3"]["cache_manifest"]))
    ref = manifest["participant_split"]
    checked(root, ref)
    return ref


def guard_child_recovery(root, action, *, prior_dispatch=False):
    """Do not let an older kernel reconstruct lost accounting/fit identity.

    This is stricter than a mere status check. In particular, classical's old
    executor permits another numerical attempt after FIT_RUNNING; the campaign
    refuses that uncertain fit when no atomic model was actually committed.
    No claim is deleted, budget reset, or replacement job name created.
    """
    root = Path(root)
    kind = action["kind"]
    directory = None
    if kind == "source":
        directory = local(root, SOURCE, exists=False)
        observed = directory.exists() and any(directory.rglob("*.pt"))
        observed = observed or (directory.exists() and any(directory.rglob("status.json")))
        if observed:
            for filename in ("training_plan.json", "operational_budget.json"):
                if not (directory / filename).is_file():
                    raise CampaignError("missing_source_recovery_ledger_or_plan_preserve_attempt")
            for progress in directory.rglob("execution_progress.json"):
                if not (progress.parent / "identity.json").is_file() or not (progress.parent / "last_state.pt").is_file():
                    raise CampaignError("missing_source_fit_identity_or_checkpoint_preserve_attempt")
            for checkpoint in directory.rglob("last_state.pt"):
                if not (checkpoint.parent / "identity.json").is_file():
                    raise CampaignError("missing_source_fit_identity_preserve_attempt")
                status = checkpoint.parent / "status.json"
                if status.exists() and not (checkpoint.parent / "execution_progress.json").is_file():
                    raise CampaignError("missing_source_optimizer_call_ledger_preserve_attempt")
    elif kind == "classical":
        directory = local(root, CLASSICAL, exists=False)
        observed = directory.exists() and (any(directory.rglob("status.json")) or any(directory.rglob("model/receipt.json")))
        if observed:
            for filename in ("identity.json", "progress.json", "plan.json"):
                if not (directory / filename).is_file():
                    raise CampaignError("missing_classical_recovery_ledger_or_identity_preserve_attempt")
            checked(root, read(directory / "identity.json")["budget"])
            for state_path in directory.rglob("status.json"):
                state = read(state_path)
                if (state.get("status") == "FIT_RUNNING" and not (state_path.parent / "model/receipt.json").is_file()):
                    raise CampaignError("uncertain_classical_FIT_RUNNING_no_committed_model_refit_forbidden")
    elif kind in {"meta_engineering", "meta_screening", "meta_confirmation"}:
        job = MetaJob(**action["job"])
        directory = local(root, str(Path(meta_receipt(job)).parent), exists=False)
        observed = directory.exists() and any((directory / name).exists()
            for name in ("latest_checkpoint.json", "progress.json", "receipt.json", "wall_budget.json"))
        if observed:
            for filename in ("identity.json", "latest_checkpoint.json", "progress.json", "wall_budget.json"):
                if not (directory / filename).is_file():
                    raise CampaignError("missing_meta_recovery_ledger_identity_or_checkpoint_preserve_attempt")
    elif kind in {"calibration_profile", "development"}:
        directory = local(root, str(Path(action["expected_receipt"]).parent), exists=False)
        observed = directory.exists() and any((directory / name).exists()
            for name in ("progress.json", "fits", "sessions", "receipt.json", "wall_budget.json"))
        if observed:
            for filename in ("identity.json", "progress.json", "wall_budget.json", "initial_budget.json"):
                if not (directory / filename).is_file():
                    raise CampaignError("missing_development_recovery_ledger_or_identity_preserve_attempt")
    elif kind == "internal":
        directory = local(root, "runs/internal_validation_once", exists=False)
        # The public internal executor owns its stronger root claim/access gate.
    elif kind == "pilot":
        directory = local(root, PILOT, exists=False)
    if prior_dispatch and directory is not None and not directory.exists():
        raise CampaignError("dispatched_child_output_missing_do_not_reconstruct_or_refit")


def internal_session_upper_bound(root):
    """Read existing W3 registry only, without entering an internal capability."""
    tasks = read(Path(root) / "status/tasks.json")
    manifest = read(checked(root, tasks["W3"]["cache_manifest"]))
    rows = manifest.get("sessions", [])
    count = sum(row.get("partition") == "internal_holdout" for row in rows)
    if count < 1:
        # Qualification metadata is already frozen before scientific training.
        # No heuristic fallback to an invented participant/session count.
        raise CampaignError("W3_registry_internal_session_upper_bound_missing")
    return count


def _selected_development(root):
    configs, confirmations = {}, []
    for method in registry.METHOD_IDS:
        ref = descriptor(root, Path(root) / selection_path(method))
        configs[method] = ref
        frozen = read(checked(root, ref))
        confirmations.append(frozen["selected_screening_receipt"])
        for seed in SEEDS[1:]:
            job = DevelopmentJob(method, frozen["configuration_index"], seed, "confirmation")
            confirmations.append(descriptor(root, Path(root) / evaluation_path(job)))
    return configs, confirmations


def selected_meta_matches_development(meta_selected, development_selected):
    config = development_selected["configuration"]
    return (meta_selected["dimension"] == config.get("dimension", config.get("rank"))
            and meta_selected["radius"] == config["radius"])


def next_action(root):
    """Compile the next missing real stage, never a command with fake artifacts."""
    root = Path(root)
    def action(kind, key, receipt, **kwargs):
        return {"kind": kind, "key": key, "expected_receipt": receipt, **kwargs}
    if not _existing(root, PILOT+"/engineering_receipt.json", "ENGINEERING_PILOT_COMPLETE"):
        return action("pilot", "pilot", PILOT+"/engineering_receipt.json")
    if not _existing(root, SOURCE+"/source_profile_development.json", "SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING"):
        return action("source", "source", SOURCE+"/source_profile_development.json")
    if not _existing(root, CLASSICAL+"/receipt.json", "SOURCE_CLASSICAL_CORE_COMPLETE"):
        return action("classical", "classical", CLASSICAL+"/receipt.json")
    for seed in SEEDS:
        if not _existing(root, reference_receipt(seed), "SOURCE_REFERENCE_COMPLETE"):
            return action("reference", "reference-"+str(seed), reference_receipt(seed), seed=seed)
    for method in LEARNED:
        job = MetaJob(method, 4 if method in {"B10", "B11"} else 8, .35, SEEDS[0], "engineering")
        if not _existing(root, meta_receipt(job), "META_ENGINEERING_COMPLETE"):
            return action("meta_engineering", "meta-"+job.identity, meta_receipt(job), method=method, job=asdict(job))
        for config in registry.registered_grid(method):
            job = meta_job(method, config)
            if not _existing(root, meta_receipt(job), "META_JOB_COMPLETE"):
                return action("meta_screening", "meta-"+job.identity, meta_receipt(job), method=method, job=asdict(job))
        relative = OUTPUT+"/meta_selections/"+method+".json"
        if not _existing(root, relative, "META_CONFIGURATION_FROZEN"):
            return action("meta_selection", "meta-selection-"+method, relative, method=method)
    for method in registry.METHOD_IDS:
        if not _existing(root, profile_path(method), "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE"):
            return action("calibration_profile", "profile-"+method, profile_path(method), method=method)
        for index in range(len(registry.registered_grid(method))):
            job = DevelopmentJob(method, index)
            if not _existing(root, evaluation_path(job), "DEVELOPMENT_EVALUATION_COMPLETE"):
                return action("development", "development-"+job.identity, evaluation_path(job), method=method, job=asdict(job))
        if not _existing(root, selection_path(method), "DEVELOPMENT_CONFIGURATION_FROZEN"):
            return action("development_selection", "selection-"+method, selection_path(method), method=method)
    for method in registry.METHOD_IDS:
        selected_ref = descriptor(root, root / selection_path(method))
        selected = read(checked(root, selected_ref))
        meta_selection = OUTPUT+"/meta_selections/"+method+".json"
        if method in LEARNED:
            meta_selected = read(local(root, meta_selection))
            if not selected_meta_matches_development(meta_selected, selected):
                raise CampaignError("BLOCKED_SELECTION_DISAGREEMENT:"+method+":meta_checkpoint_grid_vs_development_grid")
        for seed in SEEDS[1:]:
            if method in LEARNED:
                job = meta_job(method, selected["configuration"], seed)
                if not _existing(root, meta_receipt(job), "META_JOB_COMPLETE"):
                    return action("meta_confirmation", "meta-"+job.identity, meta_receipt(job),
                        method=method, job=asdict(job), meta_selection=meta_selection)
            job = DevelopmentJob(method, selected["configuration_index"], seed, "confirmation")
            if not _existing(root, evaluation_path(job), "DEVELOPMENT_EVALUATION_COMPLETE"):
                return action("development", "development-"+job.identity, evaluation_path(job),
                    method=method, job=asdict(job), configuration_receipt=selected_ref)
    if not _existing(root, OUTPUT+"/strongest_control.json", "STRONGEST_DEVELOPMENT_CONTROL_FROZEN"):
        return action("strongest_control", "strongest-control", OUTPUT+"/strongest_control.json")
    if not _existing(root, OUTPUT+"/pre_internal_freeze.json", "IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE"):
        return action("freeze", "freeze", OUTPUT+"/pre_internal_freeze.json")
    if not _existing(root, "runs/internal_validation_once/receipt.json", "ONE_TIME_INTERNAL_CHECK_COMPLETE"):
        return action("internal", "internal", "runs/internal_validation_once/receipt.json")
    return None


def watch_owned_stage(stop, owner_pid, pgid, absolute_deadline):
    """Pure CPU watchdog: lost parent or deadline kills only this child group."""
    while not stop.is_set():
        remaining = absolute_deadline-time.monotonic()
        if os.getppid() != owner_pid or remaining <= 0:
            os.killpg(pgid, signal.SIGKILL)
            return
        stop.wait(min(.1, remaining))


def start_stage_watchdog(owner_pid, pgid, absolute_deadline):
    # Only an independently setsid child may create this signal authority.
    # In particular, calling it in the campaign's own process is refused.
    if (pgid != os.getpid() or pgid != os.getpgrp() or owner_pid == pgid
            or not math.isfinite(absolute_deadline)):
        raise CampaignError("watchdog_requires_independent_owned_child_process_group")
    stop = threading.Event()
    thread = threading.Thread(target=watch_owned_stage,
        args=(stop, owner_pid, pgid, absolute_deadline), daemon=True)
    thread.start()
    return stop, thread


def _stage_worker(root, maximum, action, connection, owner_pid, absolute_deadline):
    """Spawned group owns one real stage, including any CPU fit subprocess."""
    os.setsid()
    pgid = os.getpgrp()
    stop, watchdog = start_stage_watchdog(owner_pid, pgid, absolute_deadline)
    try:
        # The child may not dispatch a kernel until its parent knows the owned
        # PGID. This closes the setsid/import/timeout race without a fit bypass.
        connection.send({"type": "group_ready", "pid": os.getpid(), "pgid": pgid})
        if connection.recv() != "dispatch_owned_stage":
            return
        result = Production(root, maximum).perform(action)
        connection.send({"type": "result", "ok": True, "receipt": result})
    except BaseException as error:
        connection.send({"type": "result", "ok": False, "error_type": type(error).__name__, "reason": str(error)})
    finally:
        # Keep the owned leader alive until parent cleanup, also on success.
        # The parent kills the entire group after receiving the durable result,
        # so an unnoticed surviving classical worker cannot outlive this stage.
        try:
            connection.recv()
        except (EOFError, OSError):
            pass
        # A normal returned stage can cancel its watchdog only when no live
        # descendants remain. Otherwise keep the watchdog/leader until parent
        # group cleanup (or owner loss/deadline) has closed that live work.
        if any(pid != os.getpid() for pid in live_group_members(pgid)):
            while True:
                stop.wait(.1)
        stop.set()
        watchdog.join(.2)
        connection.close()


def live_group_members(pgid):
    """Linux live members only: zombies cannot fit and may await init reaping."""
    if not Path("/proc/self/stat").is_file():
        raise CampaignError("Linux_proc_required_to_confirm_owned_stage_group_cleanup")
    live = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm may contain spaces or parentheses; fields after its final
            # ')' begin state, ppid, pgrp, session. Never print command text.
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X", "x"}:
                live.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Stage descendants inherit this UID and are readable. Foreign
            # inaccessible processes are not members of the owned session.
            continue
    return tuple(sorted(live))


def stop_owned_stage(worker, pgid):
    """Do not return/release the campaign lock while owned live work remains.

    Before the handshake, only the Process handle is authorized for signals.
    After it, killpg includes descendants even if the leader exited or was OOM
    killed. A slow uninterruptible process keeps this cleanup and the global
    lock pending; it never authorizes a replacement fit.
    """
    started = time.monotonic()
    while True:
        force = time.monotonic()-started >= 1.
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
        if worker.is_alive():
            # Required even without a PGID: spawn imports can time out before
            # _stage_worker has executed setsid. These are our own handles.
            if force:
                worker.kill()
            else:
                worker.terminate()
        worker.join(.1)
        live = live_group_members(pgid) if pgid is not None else ()
        if not worker.is_alive() and not live:
            return


def dispatch_bounded(root, maximum, action, *, remaining_seconds, on_group_ready=None, on_group_stopped=None):
    """Independent hard parent deadline; child limits remain cumulative too.

    A killed stage has exactly the same kernel output, claim and checkpoint on
    its next invocation. Uncertain fits are blocked by that kernel, not retried
    under a new job. setsid allows the deadline to include classical workers.
    """
    if not math.isfinite(remaining_seconds) or remaining_seconds <= 10:
        return None
    # Fix the absolute deadline BEFORE spawn startup/import overhead.
    deadline = time.monotonic()+remaining_seconds-5
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    worker = context.Process(target=_stage_worker,
        args=(str(root), maximum, action, child, os.getpid(), deadline))
    started, pgid = False, None
    try:
        worker.start()
        started = True
        child.close()
        while time.monotonic() < deadline:
            if parent.poll(min(.1, max(0., deadline-time.monotonic()))):
                try:
                    message = parent.recv()
                except EOFError:
                    raise CampaignError("stage_process_exited_without_receipt_preserve_same_job") from None
                if message.get("type") == "group_ready":
                    if pgid is not None or message.get("pid") != worker.pid or message.get("pgid") != worker.pid:
                        raise CampaignError("invalid_owned_stage_group_handshake")
                    pgid = worker.pid
                    if time.monotonic() >= deadline:
                        return None
                    if on_group_ready is not None:
                        # The real parent persists ownership BEFORE authorizing
                        # any kernel. A failed publication sends no dispatch ACK.
                        on_group_ready(worker.pid, pgid, deadline)
                    parent.send("dispatch_owned_stage")
                elif message.get("type") == "result" and pgid is not None:
                    if time.monotonic() >= deadline:
                        return None  # Never accept a late COMPLETE transport.
                    if not message["ok"]:
                        raise CampaignError("stage_blocked:"+message["error_type"]+":"+message["reason"])
                    parent.send("stage_result_received")
                    return message["receipt"]
                else:
                    raise CampaignError("stage_result_without_owned_group_authority")
            if not worker.is_alive():
                raise CampaignError("stage_process_exited_without_receipt_preserve_same_job")
        return None
    finally:
        # Cleanup also runs after successful receipt delivery, parent errors,
        # and a dead/OOM leader; never condition group cleanup on leader.alive.
        if started:
            stop_owned_stage(worker, pgid)
            if pgid is not None and on_group_stopped is not None:
                on_group_stopped()
        child.close()
        parent.close()


def published_wall_floor(root, state, identity_sha):
    """Retain every published high-water mark across an older ledger restore."""
    root = Path(root)
    snapshots = local(root, OUTPUT+"/progress_snapshots", exists=False)
    records = [state]
    if snapshots.exists():
        for path in sorted(snapshots.glob("*.json")):
            value = read(local(root, path.relative_to(root).as_posix()))
            if path.stem != digest(value):
                raise CampaignError("immutable_progress_snapshot_content_address_mismatch")
            records.append(value)
    floor = 0.
    for record in records:
        if record.get("schema_version") != SCHEMA or record.get("identity_sha256") != identity_sha:
            raise CampaignError("published_progress_or_snapshot_identity_changed")
        budget = record.get("wall_budget")
        if budget is None:
            continue  # Fresh pre-dispatch state has no published budget yet.
        if not isinstance(budget, dict) or budget.get("identity_sha256") != identity_sha:
            raise CampaignError("published_wall_budget_identity_changed")
        if budget.get("blocked") is True or budget.get("clock_block") is not None:
            raise CampaignError("published_clock_block_cannot_be_erased_by_ledger_rollback")
        for key in ("wall_seconds", "minimum_charged_seconds", "checkpoint_wall_high_water_seconds"):
            value = budget.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise CampaignError("invalid_published_wall_high_water")
            floor = max(floor, value)
    return floor


def stage_group_record(pid, pgid, deadline, identity_sha, action_key):
    """Durable ownership record written by the parent before its dispatch ACK."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if (pid != pgid or int(fields[1]) != os.getpid() or int(fields[2]) != pgid or int(fields[3]) != pgid):
        raise CampaignError("owned_stage_group_or_parent_identity_mismatch")
    return {"schema_version": "paper3-campaign-owned-stage-group-v1", "campaign_identity_sha256": identity_sha,
        "action_key": action_key, "owner_pid": os.getpid(), "pid": pid, "pgid": pgid,
        "pid_start_time_ticks": int(fields[19]), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "absolute_deadline_monotonic": deadline, "cleanup_verified": False}


def guard_previous_stage_group(state, identity_sha):
    """A new campaign may not dispatch across an unknown/live old intent."""
    if state.get("active") is None:
        return
    record = state.get("active_process")
    if (not isinstance(record, dict) or record.get("schema_version") != "paper3-campaign-owned-stage-group-v1"
            or record.get("campaign_identity_sha256") != identity_sha
            or record.get("action_key") != state["active"]["key"]
            or type(record.get("pid")) is not int or record["pid"] < 2 or record.get("pgid") != record["pid"]
            or type(record.get("owner_pid")) is not int or record["owner_pid"] < 1 or record["owner_pid"] == record["pid"]
            or type(record.get("pid_start_time_ticks")) is not int or record["pid_start_time_ticks"] < 1
            or type(record.get("absolute_deadline_monotonic")) not in (int, float)
            or not math.isfinite(record["absolute_deadline_monotonic"])
            or not isinstance(record.get("boot_id"), str) or not record["boot_id"]
            or type(record.get("cleanup_verified")) is not bool):
        raise CampaignError("unknown_previous_stage_dispatch_identity_no_new_job_authorized")
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if record.get("boot_id") != current_boot and record.get("cleanup_verified") is not True:
        raise CampaignError("previous_stage_boot_changed_without_cleanup_proof")
    if record.get("boot_id") == current_boot and live_group_members(record["pgid"]):
        raise CampaignError("previous_campaign_stage_group_still_alive_no_new_job_authorized")


def verify_source_prefix(root):
    """Before skipping W4, deeply verify its source/classical outputs, no fit."""
    root = Path(root)
    source_profile = _existing(root, SOURCE+"/source_profile_development.json", "SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING")
    if source_profile:
        from .f0_bridge import load_verified_source
        contract = read(root / "contracts/input_contract.json")
        for seed in SEEDS:
            source, info = load_verified_source(root, source_receipt(seed), contract)
            if info.get("seed") != seed or info.get("pipeline_seed") != seed:
                raise CampaignError("existing_source_profile_seed_mismatch")
            source.assert_unchanged()
    if _existing(root, CLASSICAL+"/receipt.json", "SOURCE_CLASSICAL_CORE_COMPLETE"):
        from .classical_training import verified_context, build_classical_plan, verify_complete_job
        cache, _ = verified_context(root)
        plan = build_classical_plan(cache, include_source_bank=True)
        identity = read(root / CLASSICAL / "identity.json")
        if identity.get("plan_sha256") != plan["plan_sha256"] or identity.get("scope") != "REAL_SOURCE_CLASSICAL_CPU":
            raise CampaignError("existing_classical_profile_plan_mismatch")
        checked(root, identity["budget"])
        for job in plan["jobs"]:
            state, model = verify_complete_job(root, root / CLASSICAL / job["job_id"], digest(identity), job)
            if state.get("status") != "SOURCE_CLASSICAL_FIT_COMPLETE" or model.receipt.get("synthetic") is not False:
                raise CampaignError("existing_classical_profile_not_real")


def execute_campaign(root, *, user_started_gpu=False, max_wall_seconds=43200,
                     amendment_note=None, max_actions=None):
    """Real local-only progression. Called only after explicit execution flag.

    No callback/fixture bypass exists in this production entry. The original
    kernels own fit claims, checkpoints, CUDA checks and resume decisions.
    """
    if user_started_gpu is not True:
        raise CampaignError("explicit_user_started_GPU_attestation_required")
    if type(max_wall_seconds) is not int or max_wall_seconds < 1:
        raise CampaignError("positive_integer_cumulative_wall_limit_required")
    if max_actions is not None and (type(max_actions) is not int or max_actions < 1):
        raise CampaignError("positive_optional_dispatch_count_required")
    plan = compile_campaign(root)
    if plan["minimum_gaps"]:
        return {**plan, "status": "PENDING_DATA_GATES", "execute": True}
    import torch
    if not torch.cuda.is_available():
        raise CampaignError("CUDA_required_no_CPU_fit_fallback")
    root = Path(root).resolve(strict=True)
    identity = plan["identity"]
    with campaign_lock(root, identity) as prior_claim:
        state_path = local(root, OUTPUT+"/campaign_progress.json", exists=False)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        wall_path = root / OUTPUT / "wall_budget.json"
        if prior_claim and (not state_path.is_file() or not wall_path.is_file()):
            raise CampaignError("claimed_campaign_missing_progress_or_wall_ledger_no_reset")
        state = read(state_path) if state_path.exists() else {"schema_version": SCHEMA,
            "identity_sha256": digest(identity), "actions": [], "active": None}
        if state.get("identity_sha256") != digest(identity):
            raise CampaignError("campaign_progress_identity_changed")
        guard_previous_stage_group(state, digest(identity))
        floor = published_wall_floor(root, state, digest(identity))
        journal = PersistentWallBudget(wall_path, digest(identity), max_wall_seconds,
            minimum_charged_seconds=floor, amendment_note=amendment_note)
        state["published_wall_floor_seconds_at_resume"] = floor
        dispatched = 0
        try:
            journal.start_interval()
            verify_source_prefix(root)
            # Recheck all published stage dependencies before skipping them.
            for row in state["actions"]:
                checked(root, row["receipt"])
            while True:
                if journal.exhausted or journal.remaining_seconds <= 10:
                    state["status"] = "PAUSED_CAMPAIGN_WALL_LIMIT"
                    break
                action = next_action(root)
                if action is None:
                    from .internal_validation import verify_internal_completion
                    verify_internal_completion(root, freeze=descriptor(root, root / OUTPUT / "pre_internal_freeze.json"))
                    state["status"] = "MAINLINE_THROUGH_ONE_INTERNAL_CHECK_COMPLETE"
                    state["active"] = state["active_process"] = None
                    break
                if state.get("active") and state["active"] != action:
                    # A previous child can complete after its parent was killed.
                    prior = state["active"]
                    previous = _existing(root, prior["expected_receipt"], ACTION_STATUSES[prior["kind"]])
                    if previous is None:
                        raise CampaignError("active_campaign_job_changed_without_actual_completion")
                    state["actions"].append({"key": prior["key"], "kind": prior["kind"], "receipt": previous, "recovered_after_parent_interruption": True})
                guard_child_recovery(root, action, prior_dispatch=state.get("active") == action)
                if max_actions is not None and dispatched >= max_actions:
                    state["status"] = "PAUSED_REQUESTED_ACTION_BOUND"
                    state["active"] = state["active_process"] = None
                    break
                state["active"] = action
                state["active_process"] = None
                state["status"] = "RUNNING_LOCAL_USER_STARTED_GPU"
                atomic_json(state_path, state)  # Durable intent before child call.
                def group_ready(pid, pgid, deadline):
                    state["active_process"] = stage_group_record(pid, pgid, deadline, digest(identity), action["key"])
                    atomic_json(state_path, state)
                def group_stopped():
                    if state.get("active_process") is not None:
                        state["active_process"]["cleanup_verified"] = True
                        atomic_json(state_path, state)
                ref = dispatch_bounded(root, max_wall_seconds, action, remaining_seconds=journal.remaining_seconds,
                    on_group_ready=group_ready, on_group_stopped=group_stopped)
                dispatched += 1
                journal.checkpoint()
                if ref is None:
                    state["status"] = "PAUSED_SAME_JOB_RECOVERY_REQUIRED"
                    break
                checked(root, ref)
                if ref["path"] != action["expected_receipt"]:
                    raise CampaignError("production_receipt_outside_fixed_job")
                if read(checked(root, ref)).get("status") != ACTION_STATUSES[action["kind"]]:
                    raise CampaignError("production_stage_has_no_actual_complete_receipt")
                state["actions"].append({"key": action["key"], "kind": action["kind"], "receipt": ref})
                state["active"] = state["active_process"] = None
                atomic_json(state_path, state)
        except BaseException as error:
            state["status"] = "INTERRUPTED_PRESERVE_SAME_JOB"
            state["error_type"] = type(error).__name__
            atomic_json(state_path, state)
            raise
        finally:
            journal.graceful_close()
            state["wall_budget"] = journal.report()
            journal.close()
            snapshot = {k: v for k, v in state.items() if k != "backup_manifest"}
            immutable(root, OUTPUT+"/progress_snapshots/"+digest(snapshot)+".json", snapshot)
            state["backup_manifest"] = backup_manifest(root)
            state["backup_is_uploaded_or_GET_verified"] = False
            state["server_actions"] = False
            state["external_query_accessed"] = False
            atomic_json(state_path, state)
        return state
