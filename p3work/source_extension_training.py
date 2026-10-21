"""Source-only enhanced arms, with the real baseline epoch held fixed.

The immutable primary source kernel supplies EEGNet, sampling, normalization,
checkpoints, RNG recovery and optimizer-call journals. No module globals, old
SOURCE_IDS, primary capabilities, or primary receipts are modified. Extension
receipts have their own schema and cannot authorize the old meta pipeline.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import time

import torch

from . import source_training as core
from .contracts import SOURCE_IDS, digest, file_digest
from .source_extension_bridge import (ARMS, BASE_MANIFEST, BASE_SHA, INPUT_SHA, SEEDS,
    ExtensionCache, ExtensionGateError, artifact, baseline_epoch, descriptor,
    load_extension_cache, local, read_json)

CODE_FILES = ("p3work/source_extension_training.py", "p3work/source_extension_bridge.py",
              "tools/train_source_extension.py")
SCHEMA = "paper3-source-extension-fit-v1"


def code_evidence(root):
    result = {p: descriptor(root, local(root, p)) for p in CODE_FILES}
    if Path(__file__).read_bytes() != (root / CODE_FILES[0]).read_bytes():
        raise ExtensionGateError("executed_extension_training_code_differs_from_root")
    from . import source_extension_bridge, source_training, f0_bridge, spatial_geometry
    for module in (source_extension_bridge, source_training, f0_bridge, spatial_geometry):
        relative = "p3work/" + Path(module.__file__).name
        if Path(module.__file__).read_bytes() != local(root, relative).read_bytes():
            raise ExtensionGateError("executed_extension_dependency_differs_from_root")
        result[relative] = descriptor(root, local(root, relative))
    return result


def pending_plan(arm, seed):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ExtensionGateError("registered_source_extension_arm_and_seed_required")
    return {"schema_version": SCHEMA, "status": "PENDING_REAL_CACHE_AND_BASELINE_EPOCH",
        "arm": arm, "seed": seed, "source_datasets": sorted(set(SOURCE_IDS) | {ARMS[arm]}),
        "baseline_manifest_sha256": BASE_SHA, "input_contract_sha256": INPUT_SHA,
        "additional_LODO_search": False, "planned_new_fits_per_arm_seed": 1,
        "epochs": None, "optimizer_updates": None, "actual_fits": 0,
        "training_started": False, "primary_H1_changed": False,
        "old_meta_entry_compatible": False, "requires_manual_GPU_boot": True,
        "minimum_gaps": ["complete SHA-bound extension source-only native cache",
            "same-seed actual completed baseline-four final f0 and four LODO epoch proofs",
            "actual original controlled source pilot", "positive run-specific resource budget"]}


def build_plan(root, *, arm, seed, extension_manifest, extension_sha256,
               baseline_receipt, output=None):
    root = Path(root).resolve(strict=True)
    pending_plan(arm, seed)
    output = output or f"runs/source_extensions/{arm}/seed-{seed}"
    if not output.startswith(f"runs/source_extensions/{arm}/"):
        raise ExtensionGateError("owned_source_extension_output_required")
    local(root, output + "/ownership.json", exists=False)
    cache = load_extension_cache(root, arm=arm, manifest_path=extension_manifest,
        manifest_sha256=extension_sha256, output=output)
    epoch = baseline_epoch(root, baseline_receipt, seed)
    job = {"job_id": f"{arm}/final/seed-{seed}", "phase": "final", "heldout_dataset": None,
        "seed": seed, "arm": arm, "source_extension_schema": SCHEMA,
        "train_indices": list(range(len(cache.rows))), "validation_indices": [],
        "train_uids_sha256": digest([r["event_uid"] for r in cache.rows]),
        "validation_uids_sha256": digest([]), "train_dataset_ids": sorted(set(SOURCE_IDS) | {ARMS[arm]}),
        "baseline_epoch_evidence": epoch, "implementation": code_evidence(root)}
    plan = {"schema_version": SCHEMA, "arm": arm, "seed": seed, "output": output,
        "baseline_manifest": descriptor(root, local(root, BASE_MANIFEST)),
        "extension_manifest": cache.extension_manifest, "combined_manifest": {
            "path": cache.manifest_path, "sha256": cache.manifest_sha256},
        "input_contract_sha256": INPUT_SHA, "job": job,
        "epochs": epoch["epochs"], "configuration": dict(core.SOURCE_CONFIG),
        "optimizer_updates": epoch["epochs"] * 64,
        "sampled_examples": epoch["epochs"] * 64 * 64,
        "sampling": "dataset_equal/subject_equal/observed_class_equal/trial_equal_with_replacement",
        "epoch_policy": "actual same-seed baseline-four final epoch; no new source LODO search",
        "planned_new_fits_per_arm_seed": 1, "primary_H1_changed": False,
        "all_compared_methods_must_share_this_arm_f0_and_own_Rref": True,
        "old_meta_entry_compatible": False,
        "implementation": code_evidence(root)}
    return cache, {**plan, "plan_sha256": digest(plan)}


def _read_budget(root, path, plan):
    budget_file = local(root, path)
    budget = read_json(budget_file)
    target = plan["optimizer_updates"]
    keys = {"stage", "arm", "seed", "plan_sha256", "max_jobs", "max_download_bytes",
            "max_wall_seconds", "max_gpu_seconds", "max_fit_invocations", "max_observed_optimizer_calls"}
    if (set(budget) != keys or budget.get("stage") != "source_extension_training" or budget.get("arm") != plan["arm"]
            or budget.get("seed") != plan["seed"] or budget.get("plan_sha256") != plan["plan_sha256"]
            or type(budget.get("max_jobs")) is not int or budget["max_jobs"] != 1
            or budget.get("max_download_bytes") != 0
            or type(budget.get("max_wall_seconds")) is not int or budget["max_wall_seconds"] < 1
            or type(budget.get("max_gpu_seconds")) is not int
            or budget["max_gpu_seconds"] < budget["max_wall_seconds"]
            or type(budget.get("max_fit_invocations")) is not int or budget["max_fit_invocations"] < 1
            or type(budget.get("max_observed_optimizer_calls")) is not int
            or budget["max_observed_optimizer_calls"] < target):
        raise ExtensionGateError("exact_positive_extension_job_wall_GPU_call_budget_required")
    return budget, descriptor(root, budget_file)


def _publish_plan(cache, plan):
    folder = cache.root / plan["output"]
    core.immutable_json(folder / "source_extension_plan.json", plan)
    core.immutable_json(folder / "combined_source_manifest.json", cache.manifest_payload)
    if file_digest(folder / "combined_source_manifest.json") != cache.manifest_sha256:
        raise ExtensionGateError("composite_source_manifest_bytes_changed")


def _claim_state(cache, plan, budget_descriptor, runtime, output):
    """One durable trajectory per registered arm/seed, independent of output name."""
    claim = cache.root / "runs/source_extensions/claims" / f"{plan['arm']}-seed-{plan['seed']}.json"
    identity = {"schema_version": "paper3-source-extension-single-fit-claim-v1",
        "arm": plan["arm"], "seed": plan["seed"], "plan_sha256": plan["plan_sha256"],
        "output": output, "budget": budget_descriptor, "runtime": runtime,
        "one_new_fit_per_arm_seed": True}
    if claim.exists():
        state = read_json(claim)
        if state.get("identity") != identity:
            raise ExtensionGateError("source_extension_claim_identity_changed_no_duplicate_fit")
        if state.get("active_invocation") is True:
            # The source kernel journals every observed optimizer call; a kill
            # between the call and that journal can hide at most one more call.
            state["uncertain_optimizer_calls"] += 1
    else:
        folder = cache.root / output
        if folder.exists() and any(folder.iterdir()):
            raise ExtensionGateError("missing_claim_for_existing_extension_fit_no_restart")
        state = {"identity": identity, "fit_invocations": 0, "uncertain_optimizer_calls": 0,
                 "observed_optimizer_calls_highwater": 0,
                 "observed_wall_seconds_highwater": 0.0,
                 "active_invocation": False, "status": "CLAIMED_NOT_STARTED"}
    if (type(state.get("fit_invocations")) is not int or state["fit_invocations"] < 0
            or type(state.get("uncertain_optimizer_calls")) is not int
            or state["uncertain_optimizer_calls"] < 0 or type(state.get("active_invocation")) is not bool):
        raise ExtensionGateError("invalid_extension_fit_claim_journal")
    if (type(state.get("observed_optimizer_calls_highwater")) is not int
            or state["observed_optimizer_calls_highwater"] < 0
            or not core._wall_number(state.get("observed_wall_seconds_highwater"))
            or state["active_invocation"] and state["fit_invocations"] < 1):
        raise ExtensionGateError("invalid_extension_observed_call_highwater")
    if state["fit_invocations"] > 0:
        _resume_evidence(cache, plan, runtime, output, state)
    core.atomic_json(claim, state)
    return claim, state


def _resume_evidence(cache, plan, runtime, output, claim):
    """A dispatched trajectory never recreates absent clocks, fit or journals."""
    folder = cache.root / output
    if not (folder / "operational_budget.json").is_file():
        raise ExtensionGateError("dispatched_extension_wall_ledger_missing_do_not_restart")
    wall = read_json(folder / "operational_budget.json")
    limits = read_json(artifact(cache.root, claim["identity"]["budget"]))
    consumed = wall.get("consumed_active_seconds")
    clock = wall.get("clock_state", {})
    highwater = clock.get("wall_high_water_seconds") if isinstance(clock, dict) else None
    if (not core._wall_number(consumed) or not core._wall_number(highwater)
            or max(consumed, highwater) < claim["observed_wall_seconds_highwater"]
            or wall.get("initial_seconds") != limits["max_wall_seconds"]
            or wall.get("authorized_total_seconds") != limits["max_wall_seconds"]
            or wall.get("amendments") != [] or wall.get("revision") != 1):
        raise ExtensionGateError("extension_resume_wall_budget_or_highwater_rollback")
    fit = folder / "fit"
    required = [fit/"identity.json", fit/"last_state.pt", core.checkpoint_sidecar(fit/"last_state.pt"),
                fit/"execution_progress.json"]
    if any(not path.is_file() for path in required):
        raise ExtensionGateError("dispatched_extension_fit_recovery_artifact_missing_do_not_restart")
    expected = core.fit_identity(cache, plan["job"],
        {**core.SOURCE_CONFIG, "effective_epochs": plan["epochs"]}, runtime)
    saved = core.verified_checkpoint(fit/"last_state.pt")
    progress = read_json(fit/"execution_progress.json")
    observed, committed = saved.get("observed_optimizer_calls_total"), saved.get("completed_updates")
    calls = progress.get("observed_optimizer_calls_total")
    if (read_json(fit/"identity.json") != expected or saved.get("identity") != expected
            or saved.get("identity_sha256") != digest(expected)
            or progress.get("identity_sha256") != digest(expected)
            or type(observed) is not int or type(committed) is not int
            or not 0 <= committed <= plan["optimizer_updates"] or observed < committed
            or type(calls) is not int or calls < max(observed, claim["observed_optimizer_calls_highwater"])):
        raise ExtensionGateError("extension_resume_identity_or_call_journal_rollback")
    claim["observed_optimizer_calls_highwater"] = calls


def _observed(folder):
    path = folder / "execution_progress.json"
    if not path.exists():
        return 0
    value = read_json(path).get("observed_optimizer_calls_total")
    if type(value) is not int or value < 0:
        raise ExtensionGateError("invalid_extension_optimizer_call_journal")
    return value


def execute_extension(root, *, arm, seed, extension_manifest, extension_sha256,
                      baseline_receipt, budget_path, user_started_gpu=False,
                      pilot_receipt="runs/controlled_source_pilot_seed20261018/engineering_receipt.json",
                      output=None, max_new_updates=None):
    """Explicit local CUDA only; no source/meta role changes, network or R2."""
    if not user_started_gpu:
        raise ExtensionGateError("later_manual_GPU_boot_and_explicit_flag_required")
    if max_new_updates is not None and (type(max_new_updates) is not int or max_new_updates < 1):
        raise ExtensionGateError("positive_extension_update_pause_required")
    cache, plan = build_plan(root, arm=arm, seed=seed, extension_manifest=extension_manifest,
        extension_sha256=extension_sha256, baseline_receipt=baseline_receipt, output=output)
    limits, budget_descriptor = _read_budget(cache.root, budget_path, plan)
    # Verify actual old pilot before initializing CUDA. Runtime matching follows
    # using the unchanged primary helper, not a forged extension pilot receipt.
    core.verify_engineering_pilot(cache.baseline, pilot_receipt)
    from tools.train_source import cuda_runtime
    runtime = cuda_runtime()
    pilot = core.verify_engineering_pilot(cache.baseline, pilot_receipt, runtime=runtime)
    runtime = {**runtime, "source_extension_implementation": plan["implementation"],
               "source_extension_scope": "independent_source_only_not_primary_H1"}
    output = plan["output"]
    claim_dir = cache.root / "runs/source_extensions/claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    lock_path = claim_dir / f"{arm}-seed-{seed}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExtensionGateError("same_extension_arm_seed_already_running") from None
        claim_path, claim = _claim_state(cache, plan, budget_descriptor, runtime, output)
        _publish_plan(cache, plan)
        cache.assert_fresh(composite_required=True)
        folder = cache.root / output
        receipt_path = folder / "f0_receipt.json"
        if receipt_path.exists():
            verified = verify_completed_extension(cache.root, receipt_path.relative_to(cache.root).as_posix())
            return {"status": verified["status"], "receipt": descriptor(cache.root, receipt_path),
                    "actual_new_fits_this_invocation": 0, "training_started_this_invocation": False}
        budget_file, wall = core.operational_budget(folder, initial_seconds=limits["max_wall_seconds"])
        if claim["fit_invocations"] >= limits["max_fit_invocations"]:
            raise ExtensionGateError("source_extension_fit_invocation_budget_exhausted")
        calls = _observed(folder / "fit") + claim["uncertain_optimizer_calls"]
        remaining_calls = limits["max_observed_optimizer_calls"] - calls
        remaining_wall = wall["authorized_total_seconds"] - wall["consumed_active_seconds"]
        if remaining_calls < 1 or remaining_wall <= 0:
            raise ExtensionGateError("source_extension_call_or_wall_budget_exhausted")
        allowed = min(remaining_calls, max_new_updates) if max_new_updates is not None else remaining_calls
        clock = core._source_wall_clock()
        wall["clock_state"]["interval"] = {"base_wall_seconds": wall["consumed_active_seconds"],
            "start_utc_seconds": clock["utc_seconds"], "start_monotonic_seconds": clock["monotonic_seconds"],
            "last_observed_utc_seconds": clock["utc_seconds"], "last_observed_monotonic_seconds": clock["monotonic_seconds"],
            "boot_id": clock["boot_id"], "intent": "persisted_before_source_extension_fit"}
        wall["active_invocation"] = True
        core.atomic_json(budget_file, wall)
        claim.update(active_invocation=True, fit_invocations=claim["fit_invocations"]+1, status="RUNNING")
        core.atomic_json(claim_path, claim)
        def heartbeat():
            observed = _observed(folder / "fit")
            if observed < claim["observed_optimizer_calls_highwater"]:
                raise ExtensionGateError("extension_observed_optimizer_journal_rolled_back")
            try:
                core._observe_source_wall(budget_file, wall)
            finally:
                claim["observed_optimizer_calls_highwater"] = observed
                claim["observed_wall_seconds_highwater"] = max(
                    claim["observed_wall_seconds_highwater"], wall["consumed_active_seconds"])
                core.atomic_json(claim_path, claim)
        try:
            summary = core.run_fit(cache, plan["job"], folder / "fit", runtime=runtime,
                epochs=plan["epochs"], max_new_updates=allowed,
                deadline=clock["monotonic_seconds"] + remaining_wall, heartbeat=heartbeat)
        except BaseException as error:
            claim["uncertain_optimizer_calls"] += 1
            claim["status"] = "INTERRUPTED_NOT_COMPLETE"
            raise
        finally:
            try:
                heartbeat()
            finally:
                wall["active_invocation"] = False
                wall["clock_state"]["interval"] = None
                core.atomic_json(budget_file, wall)
                claim["active_invocation"] = False
                core.atomic_json(claim_path, claim)
        cache.assert_fresh(composite_required=True)
        if wall["consumed_active_seconds"] > wall["authorized_total_seconds"]:
            claim["status"] = "RESOURCE_OVERRUN_NOT_EXTENSION_COMPLETE"
            core.atomic_json(claim_path, claim)
            raise ExtensionGateError("measured_extension_wall_overrun_no_completion_receipt")
        if _observed(folder / "fit") + claim["uncertain_optimizer_calls"] > limits["max_observed_optimizer_calls"]:
            raise ExtensionGateError("measured_extension_optimizer_call_budget_exceeded")
        if not summary["complete"]:
            claim["status"] = "PAUSED_AT_COMMITTED_UPDATE"
            core.atomic_json(claim_path, claim)
            result = {"status": claim["status"], "plan_sha256": plan["plan_sha256"],
                "committed_optimizer_updates": summary["committed_optimizer_updates"],
                "actual_scientific_fits_complete": 0, "training_started": summary["training_started"],
                "call_journal": _observed(folder / "fit"),
                "uncertain_optimizer_calls": claim["uncertain_optimizer_calls"], "wall_budget": wall}
            core.atomic_json(folder / "progress.json", result)
            return result
        checked = core.verify_complete_fit(cache, plan["job"], folder / "fit")
        if (checked["epochs_completed"] != plan["epochs"]
                or checked["committed_optimizer_updates"] != plan["optimizer_updates"]
                or checked["selected_epoch"] != plan["epochs"]):
            raise ExtensionGateError("extension_fit_did_not_reach_exact_baseline_epoch_budget")
        receipt = {"schema_version": SCHEMA, "status": "SOURCE_EXTENSION_F0_COMPLETE",
            "arm": arm, "seed": seed, "pipeline_seed": seed, "output": output,
            "synthetic": False, "training_started": True, "actual_scientific_fits_complete": 1,
            "primary_H1_changed": False, "old_meta_entry_compatible": False,
            "downstream_meta_ready": False, "reference_status": "PENDING_OWN_SOURCE_EXTENSION_REFERENCE",
            "architecture": "independent_source_extension_compact_eegnet_clinic11",
            "input_contract_sha256": INPUT_SHA, "training_dataset_ids": plan["job"]["train_dataset_ids"],
            "epochs": plan["epochs"], "optimizer_updates": plan["optimizer_updates"],
            "sampled_examples": plan["sampled_examples"], "baseline_epoch_evidence": plan["job"]["baseline_epoch_evidence"],
            "plan": descriptor(cache.root, folder / "source_extension_plan.json"),
            "combined_manifest": descriptor(cache.root, folder / "combined_source_manifest.json"),
            "extension_manifest": cache.extension_manifest, "baseline_manifest": plan["baseline_manifest"],
            "implementation": plan["implementation"], "budget": budget_descriptor,
            "claim": claim_path.relative_to(cache.root).as_posix(), "runtime": runtime, "engineering_pilot": pilot,
            "checkpoint": descriptor(cache.root, folder / "fit/selected.pt"),
            "training_uid_manifest": descriptor(cache.root, folder / "fit/training_uids.json"),
            "fit_status": descriptor(cache.root, folder / "fit/status.json"),
            "fit_artifact_manifest": descriptor(cache.root, folder / "fit/artifact_manifest.json"),
            "observed_optimizer_calls": _observed(folder / "fit"),
            "uncertain_optimizer_calls": claim["uncertain_optimizer_calls"],
            "operational_wall_budget": descriptor(cache.root, budget_file),
            "no_external_or_meta_training_events": True, "server_actions": False, "R2_uploaded": False}
        core.immutable_json(receipt_path, receipt)
        claim["status"] = receipt["status"]
        core.atomic_json(claim_path, claim)
        files = [descriptor(cache.root, path) for path in sorted(folder.rglob("*"))
                 if path.is_file() and not path.name.startswith(".")]
        files.extend((descriptor(cache.root, claim_path), budget_descriptor))
        core.immutable_json(folder / "backup_manifest.json", {"schema_version": 1,
            "files": [{**item, "role": "result", "dataset": "Paper3SourceExtension"} for item in files]})
        return {"status": receipt["status"], "receipt": descriptor(cache.root, receipt_path),
                "actual_new_fits_this_invocation": 1, "training_started_this_invocation": True,
                "downstream_meta_ready": False, "R2_uploaded": False}


def verify_completed_extension(root, receipt_path):
    """Full CPU-only verification before an independent consumer loads weights."""
    root = Path(root).resolve(strict=True)
    path = local(root, receipt_path)
    receipt = read_json(path)
    if (receipt.get("schema_version") != SCHEMA or receipt.get("status") != "SOURCE_EXTENSION_F0_COMPLETE"
            or receipt.get("synthetic") is not False or receipt.get("actual_scientific_fits_complete") != 1
            or receipt.get("primary_H1_changed") is not False or receipt.get("old_meta_entry_compatible") is not False
            or receipt.get("downstream_meta_ready") is not False):
        raise ExtensionGateError("actual_independent_extension_f0_receipt_required")
    plan = read_json(artifact(root, receipt["plan"]))
    cache, rebuilt = build_plan(root, arm=receipt["arm"], seed=receipt["seed"],
        extension_manifest=receipt["extension_manifest"]["path"],
        extension_sha256=receipt["extension_manifest"]["sha256"],
        baseline_receipt=receipt["baseline_epoch_evidence"]["baseline_f0"]["path"], output=receipt["output"])
    if plan != rebuilt or receipt.get("implementation") != plan["implementation"]:
        raise ExtensionGateError("completed_extension_plan_or_code_changed")
    if path != root / plan["output"] / "f0_receipt.json":
        raise ExtensionGateError("extension_receipt_not_in_owned_output")
    cache.assert_fresh(composite_required=True)
    folder = root / plan["output"]
    summary = core.verify_complete_fit(cache, plan["job"], folder / "fit")
    identity = read_json(folder / "fit/identity.json")
    artifact(root, receipt["budget"])
    limits, budget_desc = _read_budget(root, receipt["budget"]["path"], plan)
    if budget_desc != receipt["budget"]:
        raise ExtensionGateError("completed_extension_budget_changed")
    expected_claim = f"runs/source_extensions/claims/{receipt['arm']}-seed-{receipt['seed']}.json"
    if receipt.get("claim") != expected_claim:
        raise ExtensionGateError("completed_extension_requires_global_arm_seed_claim")
    claim = read_json(local(root, expected_claim))
    expected_identity = {"schema_version": "paper3-source-extension-single-fit-claim-v1",
        "arm": plan["arm"], "seed": plan["seed"], "plan_sha256": plan["plan_sha256"],
        "output": plan["output"], "budget": receipt["budget"], "runtime": receipt["runtime"],
        "one_new_fit_per_arm_seed": True}
    if (claim.get("status") != "SOURCE_EXTENSION_F0_COMPLETE" or claim.get("active_invocation") is not False
            or claim.get("identity") != expected_identity
            or type(claim.get("fit_invocations")) is not int
            or not 1 <= claim["fit_invocations"] <= limits["max_fit_invocations"]
            or type(claim.get("uncertain_optimizer_calls")) is not int
            or claim["uncertain_optimizer_calls"] < 0
            or type(claim.get("observed_optimizer_calls_highwater")) is not int
            or not core._wall_number(claim.get("observed_wall_seconds_highwater"))):
        raise ExtensionGateError("completed_extension_claim_mismatch")
    wall = read_json(artifact(root, receipt["operational_wall_budget"]))
    if (wall.get("active_invocation") is not False or wall.get("clock_state", {}).get("clock_block") is not None
            or wall.get("clock_state", {}).get("interval") is not None
            or wall.get("authorized_total_seconds") != limits["max_wall_seconds"]
            or not core._wall_number(wall.get("consumed_active_seconds"))
            or wall["consumed_active_seconds"] > limits["max_wall_seconds"]
            or wall["consumed_active_seconds"] != claim["observed_wall_seconds_highwater"]
            or summary["committed_optimizer_updates"] != plan["optimizer_updates"]
            or summary["epochs_completed"] != plan["epochs"]
            or identity.get("runtime") != receipt.get("runtime")
            or receipt.get("epochs") != plan["epochs"] or receipt.get("optimizer_updates") != plan["optimizer_updates"]
            or receipt.get("training_dataset_ids") != plan["job"]["train_dataset_ids"]
            or receipt.get("baseline_epoch_evidence") != plan["job"]["baseline_epoch_evidence"]
            or receipt.get("input_contract_sha256") != INPUT_SHA):
        raise ExtensionGateError("completed_extension_actual_epoch_runtime_resource_mismatch")
    actual_pilot = core.verify_engineering_pilot(cache.baseline, receipt["engineering_pilot"]["receipt"]["path"],
                                               runtime=receipt["runtime"])
    if actual_pilot != receipt["engineering_pilot"]:
        raise ExtensionGateError("completed_extension_actual_pilot_evidence_changed")
    for key, expected in (("checkpoint", folder / "fit/selected.pt"),
                          ("training_uid_manifest", folder / "fit/training_uids.json"),
                          ("fit_status", folder / "fit/status.json"),
                          ("fit_artifact_manifest", folder / "fit/artifact_manifest.json"),
                          ("combined_manifest", folder / "combined_source_manifest.json")):
        if artifact(root, receipt[key]) != expected:
            raise ExtensionGateError("completed_extension_artifact_location_mismatch")
    if (_observed(folder / "fit") != receipt["observed_optimizer_calls"]
            or claim["observed_optimizer_calls_highwater"] != receipt["observed_optimizer_calls"]
            or receipt["uncertain_optimizer_calls"] != claim["uncertain_optimizer_calls"]
            or receipt["observed_optimizer_calls"] + receipt["uncertain_optimizer_calls"] > limits["max_observed_optimizer_calls"]):
        raise ExtensionGateError("completed_extension_observed_call_budget_mismatch")
    return receipt
