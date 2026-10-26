"""Audited cross-node clock recovery for verified R1 meta checkpoints only.

An unclean exit can leave a wall-accounting interval on GPU node A. Node B has
a different kernel boot/monotonic clock, so the original operational journal
blocks even after its model/RNG checkpoint was verified. This overlay retains
that whole clock block and every charged second, explicitly marks the interval
as uncertain, and lets a new real clock interval start. It changes no model,
identity, update cap, checkpoint, data acceptance or GPU observation.
"""
from __future__ import annotations

import copy
from contextvars import ContextVar
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import time

SCHEMA = "paper3-r1-authorized-cross-node-clock-recovery-v1"
NOTE = ("User authorized continuous R1 checkpoint resume across school GPU nodes on "
        "2026-10-26; preserve the old uncertain interval, original clock observations, "
        "all cumulative charged floors and scientific work caps. No exact elapsed "
        "time is inferred for a clock history that could not be joined.")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _emit(event, **fields):
    try:
        print(json.dumps({"event": event,
                          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          **fields}, sort_keys=True, allow_nan=False), flush=True)
    except Exception:
        pass


def _read(path):
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            return None
        value = json.loads(data)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _registered_resume(meta, root, values):
    """Qualify one native loop; the native checkpoint loader still must pass."""
    identity, resume = values.get("identity"), values.get("resume")
    if not isinstance(identity, dict) or resume is None:
        return None
    try:
        job = meta.MetaJob(**identity["job"])
        output = Path(values["output"]).resolve()
        expected = root / "runs/meta" / job.identity
        requested = Path(resume).resolve()
        if (output != expected or requested.parent != output
                or values.get("total_updates", 1000) != job.updates
                or _read(output / "identity.json") != identity):
            return None
        latest = _read(output / "latest_checkpoint.json")
        if (not latest or latest.get("relative_to_job_output") is not True
                or Path(str(latest.get("path", ""))).name != latest.get("path")
                or requested != output / latest["path"]
                or not re.fullmatch(r"[0-9a-f]{64}", str(latest.get("sha256", "")))):
            return None
        sidecar = _read(requested.with_suffix(requested.suffix + ".sha256.json"))
        if (not sidecar or sidecar.get("sha256") != latest["sha256"]
                or sidecar.get("identity_sha256") != _digest(identity)
                or sidecar.get("bytes") != requested.stat().st_size):
            return None
        return {"root": root, "output": output, "resume": requested,
                "identity_sha256": _digest(identity), "job": copy.deepcopy(identity["job"]),
                "latest": copy.deepcopy(latest), "checkpoint_verified": None,
                "resolutions": []}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _recover_cross_node_clock(budget, proof):
    """Called while the original journal's exclusive OS lock is still held."""
    state = budget._state
    block, interval = state.get("clock_block"), state.get("interval")
    verified = proof.get("checkpoint_verified")
    if (not verified or not isinstance(block, dict) or not isinstance(interval, dict)
            or block.get("reasons") != ["boot_changed_or_unknown"]
            or block.get("interval") != interval
            or state.get("identity_sha256") != proof["identity_sha256"]
            or interval.get("identity_sha256") != proof["identity_sha256"]
            or budget.path != proof["output"] / "wall_budget.json"):
        return False
    old_boot = interval.get("boot_id")
    observed = block.get("clock")
    new_boot = observed.get("boot_id") if isinstance(observed, dict) else None
    if (not isinstance(old_boot, str) or not old_boot.strip()
            or not isinstance(new_boot, str) or not new_boot.strip() or new_boot == old_boot):
        return False
    try:
        current = budget._clock()  # Real current clock; no property or UUID spoofing.
    except Exception:
        return False
    if not isinstance(current.get("boot_id"), str) or not current["boot_id"].strip():
        return False
    if (current["utc_seconds"] < max(interval["start_utc_seconds"],
                                    interval.get("last_observed_utc_seconds", interval["start_utc_seconds"]),
                                    observed["utc_seconds"])
            or (current["boot_id"] == new_boot
                and current["monotonic_seconds"] < observed["monotonic_seconds"])
            or (current["boot_id"] == old_boot
                and current["monotonic_seconds"] < max(
                    interval["start_monotonic_seconds"],
                    interval.get("last_observed_monotonic_seconds", interval["start_monotonic_seconds"])))):
        return False
    durable = state.get("durable_wall_seconds")
    floor = state.get("checkpoint_wall_high_water_seconds")
    if not all(_number(value) for value in (durable, floor, verified["active_wall_seconds"])):
        return False
    preserved_floor = max(durable, floor, verified["active_wall_seconds"])
    qualification = state.get("clock_history_qualification", {})
    if not isinstance(qualification, dict):
        return False
    prior_unknowns = qualification.get("unknown_interval_count", 0)
    if type(prior_unknowns) is not int or prior_unknowns < 0:
        return False
    resolutions = state.get("authorized_clock_recoveries", [])
    if not isinstance(resolutions, list):
        return False
    event = {"schema_version": SCHEMA, "authorization": NOTE,
             "scientific_identity_sha256": proof["identity_sha256"],
             "job": proof["job"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "verified_checkpoint": copy.deepcopy(verified),
             "original_clock_block": copy.deepcopy(block),
             "original_unsettled_interval": copy.deepcopy(interval),
             "recovery_observed_clock": current,
             "prior_durable_charged_seconds": durable,
             "prior_checkpoint_charged_floor_seconds": floor,
             "preserved_charged_floor_seconds": preserved_floor,
             "unknown_additional_elapsed_seconds": None,
             "elapsed_time_inferred_or_reset": False,
             "scientific_identity_changed": False,
             "checkpoint_bytes_changed": False,
             "optimizer_sampler_or_update_counters_reset": False,
             "policy": "retire_cross_boot_interval_keep_uncertainty_start_new_real_interval"}
    event["event_sha256"] = _digest(event)
    state["authorized_clock_recoveries"] = [*resolutions, event]
    state["clock_history_qualification"] = {
        **qualification, "exact_cumulative_elapsed_claimed": False,
        "charged_seconds_are_recorded_operational_accounting": True,
        "unknown_interval_count": max(prior_unknowns, len(resolutions)) + 1,
        "unknown_elapsed_seconds": None,
        "note": "Unjoinable clock intervals remain preserved, not converted to invented elapsed time."}
    state["durable_wall_seconds"] = preserved_floor
    state["checkpoint_wall_high_water_seconds"] = max(floor, verified["active_wall_seconds"])
    # Do not count an uncertain interval as an ordinary fully settled interval.
    state["interval"] = None
    state["clock_block"] = None
    budget._save()
    proof["resolutions"].append(copy.deepcopy(event))
    _emit("R1_CROSS_NODE_CLOCK_CHECKPOINT_RESUME_AUTHORIZED",
          event_sha256=event["event_sha256"], identity_sha256=proof["identity_sha256"],
          historical_boot_id=old_boot, blocked_observed_boot_id=new_boot,
          actual_resume_boot_id=current["boot_id"],
          charged_floor_preserved_seconds=preserved_floor,
          unknown_elapsed_seconds=None, exact_cumulative_elapsed_claimed=False,
          checkpoint_bytes_changed=False)
    return True


def _qualify_result(proof, result):
    """Carry honest operational timing provenance into the next native receipt."""
    if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict)):
        return result
    state = _read(proof["output"] / "wall_budget.json")
    if not state or state.get("identity_sha256") != proof["identity_sha256"]:
        return result
    qualification = state.get("clock_history_qualification")
    if not isinstance(qualification, dict) or qualification.get("exact_cumulative_elapsed_claimed") is not False:
        return result
    metadata = copy.deepcopy(qualification)
    metadata["audit_path"] = str((proof["output"] / "wall_budget.json").relative_to(proof["root"]))
    metadata["scientific_identity_sha256"] = proof["identity_sha256"]
    # Existing completed receipts are never opened or rewritten. The original
    # execute_meta_job caller publishes this extra provenance with its result.
    return result[0], {**result[1], "operational_clock_history_qualification": metadata}


def install(root):
    """Install only in R1 action/owned-child processes, before native meta work."""
    root = Path(root).resolve()
    meta = importlib.import_module("p3work.meta_training")
    budget_module = importlib.import_module("p3work.recovery_budget")
    existing = getattr(meta, "_paper3_r1_cross_node_clock_recovery", None)
    if existing is not None:
        if existing["root"] != str(root):
            raise ValueError("R1 clock recovery is already bound to a different project root")
        return {"status": "ALREADY_INSTALLED", "root": existing["root"]}
    loop, load, base = meta.run_outer_loop, meta.load_checkpoint, meta.PersistentWallBudget
    context = ContextVar("paper3_r1_verified_meta_clock_resume", default=None)

    @functools.wraps(load)
    def verified_load(path, *, identity, cuda=False):
        result = load(path, identity=identity, cuda=cuda)
        proof = context.get()
        if (proof is not None and Path(path).resolve() == proof["resume"]
                and _digest(identity) == proof["identity_sha256"]
                and isinstance(result, dict) and _number(result.get("active_wall_seconds"))
                and type(result.get("outer_cursor")) is int):
            proof["checkpoint_verified"] = {
                "path": str(proof["resume"].relative_to(root)),
                "sha256": proof["latest"]["sha256"],
                "outer_cursor": result["outer_cursor"],
                "active_wall_seconds": result["active_wall_seconds"],
                "verification": "original_full_SHA_identity_and_RNG_checkpoint_loader_returned"}
        return result

    class ResumeClockBudget(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            proof = context.get()
            if proof is not None and self.blocked:
                prior = copy.deepcopy(self._state)
                try:
                    _recover_cross_node_clock(self, proof)
                except Exception as error:
                    # A failed bookkeeping amendment cannot retain an open OS
                    # lock or quietly turn into permission to start work.
                    self._state = prior
                    try:
                        self._save()
                    except Exception:
                        pass
                    self._lock.close()
                    self._closed = True
                    _emit("R1_CROSS_NODE_CLOCK_RECOVERY_NOT_APPLIED", error_type=type(error).__name__)
                    raise budget_module.WallBudgetBlocked(
                        "Original operational clock block retained; clock recovery could not be recorded") from error

        def report(self):
            value = super().report()
            qualification = self._state.get("clock_history_qualification")
            if qualification:
                value["clock_history_qualification"] = copy.deepcopy(qualification)
            return value

    @functools.wraps(loop)
    def resume_loop(**values):
        # A valid Slurm allocation plus the existing native GPU execution path
        # owns work authorization. The helper supplies no GPU/data capability.
        eligible = bool(re.fullmatch(r"[0-9]{1,20}", os.environ.get("SLURM_JOB_ID", "")))
        proof = _registered_resume(meta, root, values) if eligible else None
        token = context.set(proof)
        try:
            result = loop(**values)
            if proof is not None and proof["resolutions"]:
                _emit("R1_CROSS_NODE_CLOCK_RECOVERY_FINISHED",
                      identity_sha256=proof["identity_sha256"],
                      resolutions=len(proof["resolutions"]),
                      exact_cumulative_elapsed_claimed=False,
                      original_clock_evidence_preserved=True)
            return _qualify_result(proof, result) if proof is not None else result
        finally:
            context.reset(token)

    meta.run_outer_loop = resume_loop
    meta.load_checkpoint = verified_load
    meta.PersistentWallBudget = ResumeClockBudget
    meta._paper3_r1_cross_node_clock_recovery = {"root": str(root), "schema": SCHEMA}
    return {"status": "INSTALLED", "root": str(root), "scope": "verified_same_identity_R1_meta_checkpoint_resume_only"}
