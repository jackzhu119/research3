"""Explicit, bounded CPU source-classical fits and fit-level checkpoint resume.

No execution happens on import or dry run. Core ten fits independently retain
the immutable v1 source recipes, including its float32 intermediate/log power.
They are baseline models, never the new Paper3 algorithm. New B4/B5 source-bank
models use their distinct current formulas and require six additional fits.
An interrupted numerical solve is restarted, counted as another fit attempt;
committed numerical checkpoints and completed jobs are verified and reused.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import tempfile
import time
import warnings

import numpy as np
import scipy
from scipy.linalg import eigh, helmert
from scipy.special import expit
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from .contracts import SOURCE_IDS, digest, file_digest, validate_signal_contract
from .methods import classical as kernels


SCHEMA = "paper3-source-classical-training-v1"
CORE_CSP = "source_csp_lda_fixed_v1"
CORE_TS = "source_ts_lr_fixed_v1"
CORE_METHODS = (CORE_CSP, CORE_TS)
CODE_FILES = ("p3work/classical_training.py", "p3work/methods/classical.py",
              "p3work/contracts.py", "p3work/adapt.py", "p3work/spatial_geometry.py",
              "tools/train_classical_source.py", "tools/run_controlled_pilot.py")
POLICY_FILES = ("protocol/EXPERIMENT_PROTOCOL.md", "protocol/BASELINE_SPEC.md", "protocol/METHOD_SPEC.md",
                "protocol/WORK_TASKS.md", "protocol/legacy_baseline_v1/configs/default.json",
                "protocol/legacy_baseline_v1/p3/core.py", "protocol/legacy_baseline_v1/p3/models.py")
RECIPE_DIFFERENCES = {
    "legacy_CSP": "joint_RMS_float64_then_float32_then_float64; scipy_Helmert; shrunk_trial_cov; low2_high2_ascending; legacy_log_power_with_float_tiny_floor",
    "B4_bank": "float64_trace_cov; fixed_main_Q; hierarchical_class_means_then_shrink; deterministic_sign_and_ties; log_normalized_variance_without_floor",
    "legacy_TS": "joint_RMS_float64_then_float32_then_float64; scipy_Helmert; covariance_div_Tminus1; shrink_0.1; weighted_log_Euclidean_reference_scaler_C1_LR",
    "B5_bank": "float64_trace_cov_directly; fixed_main_Q; matching_shrink_log_Euclidean_reference_scaler_C1_LR",
    "reuse": "NONE: differing numerical paths do not justify shared core/bank models; six bank fits are separately counted",
}


class ClassicalTrainingError(ValueError):
    pass


class FitWallBudgetExpired(TimeoutError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def local(root, relative, *, exists=True):
    from tools.r2_restore import no_symlink_components
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts or "legacy_baseline_v1" in Path(relative).parts
            or Path(relative).suffix.lower() in {".npz", ".pickle", ".pkl", ".pt", ".pth"}):
        raise ClassicalTrainingError("New root-relative non-pickle artifact required")
    path = root / relative
    no_symlink_components(path)
    result = path.resolve(strict=exists)
    if not result.is_relative_to(root):
        raise ClassicalTrainingError("Artifact escaped the independent Paper3 root")
    return result


def artifact(root, path):
    path = Path(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}


def checked_artifact(root, item):
    if not isinstance(item, dict):
        raise ClassicalTrainingError("Actual artifact descriptor required")
    path = local(root, item.get("path"))
    if file_digest(path) != item.get("sha256") or ("bytes" in item and path.stat().st_size != item["bytes"]):
        raise ClassicalTrainingError("Classical artifact SHA256/byte count mismatch")
    return path


@contextmanager
def preserve_rng():
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)


def _code_policy(root):
    loaded_root = Path(__file__).resolve().parents[1]
    code = {name: file_digest(root / name) for name in CODE_FILES}
    if any(value != file_digest(loaded_root / name) for name, value in code.items()):
        raise ClassicalTrainingError("Packaged and executing classical implementations differ")
    return {"code": code, "policy": {name: file_digest(root / name) for name in POLICY_FILES}}


def verified_context(root):
    """Actual W1/W2/cache/input/inventory gates, before any output or fitting."""
    from tools.run_controlled_pilot import verify_inputs
    from tools.prepare_source_cache import expected_source_objects
    root = Path(root).resolve(strict=True)
    cache = verify_inputs(root)
    contract = json.loads((root / "contracts/input_contract.json").read_text())
    if validate_signal_contract(contract) != cache.input_contract_sha256:
        raise ClassicalTrainingError("Frozen signal/cache identities differ")
    manifest = json.loads(local(root, cache.manifest_path).read_text())
    if (manifest.get("synthetic") is not False or manifest.get("full_source_inventory_verified") is not True
            or manifest.get("normalization_applied") is not False
            or any(manifest.get(key) for key in ("test_double_not_research_evidence", "fixture_scope", "test_double"))):
        raise ClassicalTrainingError("Complete real source cache required; software fixtures cannot complete scientific fits")
    inventory_file = checked_artifact(root, manifest.get("raw_inventory"))
    inventory = json.loads(inventory_file.read_text())
    expected = expected_source_objects()
    files = inventory.get("files", [])
    keys = [(row.get("dataset"), str(row.get("provider_subject")), str(row.get("session_id")), row.get("file")) for row in files]
    if (inventory.get("status") != "SOURCE_RAW_INVENTORY_COMPLETE" or inventory.get("provider_roster_verified") is not True
            or inventory.get("synthetic") is not False or inventory.get("old_npz_reused") is not False
            or inventory.get("role") != "source_train"
            or any(inventory.get(key) for key in ("test_double_not_research_evidence", "fixture_scope", "test_double"))
            or len(keys) != len(expected) or set(keys) != set(expected) or len(set(keys)) != len(keys)):
        raise ClassicalTrainingError("All 505 official native source objects must be inventory verified")
    checked_artifact(root, inventory.get("provider_terms_evidence"))
    raw_hashes = set()
    for row, key in zip(files, keys):
        if (row.get("url") != expected[key] or row.get("native_schema_audited") is not True
                or row.get("complete_download") is not True or row.get("http_status") != 200):
            raise ClassicalTrainingError("Native source inventory lacks actual download/schema evidence")
        kernels._sha(row.get("sha256"), "native source object")
        raw_hashes.add((row["dataset"], row["sha256"]))
    if any((row["dataset"], row.get("raw_sha256")) not in raw_hashes for row in cache.rows):
        raise ClassicalTrainingError("Source-cache events do not belong to the verified native inventory")
    return cache, contract


def build_classical_plan(cache, *, include_source_bank=False):
    if type(include_source_bank) is not bool:
        raise ClassicalTrainingError("Source-bank opt-in must be explicit boolean")
    if {row.get("dataset") for row in cache.rows} != SOURCE_IDS or any(row.get("role") != "source_train" for row in cache.rows):
        raise ClassicalTrainingError("All and only four source_train datasets required")
    uids, fingerprints = set(), defaultdict(set)
    for row in cache.rows:
        uid = row.get("event_uid")
        if not isinstance(uid, str) or not uid or uid in uids or not str(row.get("subject_id", "")):
            raise ClassicalTrainingError("Unique native event and subject identities required")
        uids.add(uid)
        value = row.get("processed_signal_sha256")
        kernels._sha(value, "processed source signal")
        fingerprints[value].add(row["dataset"])
    if any(len(datasets) > 1 for datasets in fingerprints.values()):
        raise ClassicalTrainingError("Processed-signal duplicates cross source LODO; resolve the audit first")
    jobs = []
    folds = [("lodo", dataset) for dataset in sorted(SOURCE_IDS)] + [("final", None)]
    for phase, heldout in folds:
        for method in CORE_METHODS:
            jobs.append({"job_id": f"{phase}/{heldout or 'ALL_SOURCE'}/{method}", "group": "core",
                         "phase": phase, "held_out_dataset": heldout, "method": method,
                         "configuration": {"m": 2} if method == CORE_CSP else {"shrink": .1},
                         "feature_mode": "legacy_log_power" if method == CORE_CSP else "legacy_log_Euclidean_TS"})
    if include_source_bank:
        for method, key, values in (("B4", "m", (1, 2, 3)), ("B5", "shrink", (.01, .05, .1))):
            for value in values:
                jobs.append({"job_id": f"bank/ALL_SOURCE/{method}/{key}-{value}", "group": "bank",
                             "phase": "bank", "held_out_dataset": None, "method": method,
                             "configuration": {key: value},
                             "feature_mode": "log_normalized_variance" if method == "B4" else "log_Euclidean_TS_float64"})
    for job in jobs:
        heldout = job["held_out_dataset"]
        train = [index for index, row in enumerate(cache.rows) if heldout is None or row["dataset"] != heldout]
        validation = [index for index, row in enumerate(cache.rows) if row["dataset"] == heldout] if heldout else []
        train_people = {(cache.rows[i]["dataset"], str(cache.rows[i]["subject_id"])) for i in train}
        validation_people = {(cache.rows[i]["dataset"], str(cache.rows[i]["subject_id"])) for i in validation}
        if train_people & validation_people or set(np.asarray(cache.labels)[train]) != {0, 1} or (validation and set(np.asarray(cache.labels)[validation]) != {0, 1}):
            raise ClassicalTrainingError("Source LODO people/classes are not separated")
        job.update(train_indices=train, validation_indices=validation,
                   input_contract_sha256=cache.input_contract_sha256,
                   source_cache_manifest_sha256=cache.manifest_sha256,
                   train_uid_sha256=digest([cache.rows[i]["event_uid"] for i in train]),
                   validation_uid_sha256=digest([cache.rows[i]["event_uid"] for i in validation]),
                   train_datasets=sorted({cache.rows[i]["dataset"] for i in train}),
                   input_bytes=len(train) * 11 * cache.n_times * max(array.dtype.itemsize for array in cache.arrays))
        job["job_sha256"] = digest(job)
    value = {"schema_version": SCHEMA, "input_contract_sha256": cache.input_contract_sha256,
             "cache_manifest": {"path": cache.manifest_path, "sha256": cache.manifest_sha256},
             "source_artifact_sha256": {name: value for name, value in cache.artifact_sha256.items()
                                        if name != "status/tasks.json"}, "code_policy": _code_policy(cache.root),
             "jobs": jobs, "planned_core_fits": 10, "planned_additional_bank_fits": 6 if include_source_bank else 0,
             "planned_total_fits": len(jobs), "include_source_bank": include_source_bank,
             "recipe_differences": copy.deepcopy(RECIPE_DIFFERENCES), "core_to_bank_models_reused": 0,
             "source_validation": "held_out_source_dataset_participant_equal_metrics_only",
             "cpu_blas_threads": 1, "fit_resume_granularity": "atomic_numerical_fit_checkpoint_then_validation",
             "training_started": False, "gpu_execution": False, "external_access": False}
    return {**value, "plan_sha256": digest(value)}


def validate_plan(cache, plan):
    if plan != build_classical_plan(cache, include_source_bank=plan.get("include_source_bank")):
        raise ClassicalTrainingError("Complete fixed classical plan/code/cache identity required")


def legacy_standardized_trials(x):
    # Independently copied mathematical path, including its intentional float32
    # round trip. Never import, modify, or execute the immutable legacy package.
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(x * x, axis=(-2, -1), keepdims=True))
    if (rms <= 0).any() or not np.isfinite(rms).all():
        raise ClassicalTrainingError("Invalid legacy source trial RMS")
    return (x / rms).astype(np.float32)


def legacy_covariance(x):
    basis = helmert(x.shape[1], full=False)
    z = np.einsum("dc,nct->ndt", basis, legacy_standardized_trials(x).astype(np.float64))
    cov = z @ z.transpose(0, 2, 1) / (z.shape[-1] - 1)
    trace = np.trace(cov, axis1=1, axis2=2)
    if (trace <= 0).any() or not np.isfinite(cov).all():
        raise ClassicalTrainingError("Invalid legacy source covariance")
    cov /= trace[:, None, None]
    return .9 * cov + .1 * np.eye(10)[None] / 10, basis


def legacy_matrix_function(matrix, function):
    values, vectors = np.linalg.eigh(matrix)
    if not np.isfinite(values).all() or (function in (np.log, np.sqrt) and (values <= 0).any()):
        raise ClassicalTrainingError("Invalid legacy source covariance spectrum")
    return (vectors * function(values)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def legacy_csp_features(x, basis, filters):
    z = np.einsum("dc,nct->ndt", basis, legacy_standardized_trials(x).astype(np.float64))
    filtered = np.einsum("kd,ndt->nkt", filters, z)
    return np.log(np.maximum(np.mean(filtered * filtered, axis=-1), np.finfo(float).tiny))


def legacy_tangent_features(covariance, inverse_root):
    logs = legacy_matrix_function(inverse_root @ covariance @ inverse_root, np.log)
    i, j = np.triu_indices(10)
    return logs[:, i, j] * np.where(i == j, 1.0, np.sqrt(2.0))


def fit_source_math(inputs, job):
    """One CPU numerical fit; genuine execution uses only gated source inputs."""
    started = time.monotonic()
    x, y, signal_sha = inputs.validate()
    if (inputs.held_out_dataset != job["held_out_dataset"] or signal_sha != job["input_contract_sha256"]
            or inputs.source_manifest_sha256 != job["source_cache_manifest_sha256"]
            or digest([row["event_uid"] for row in inputs.rows]) != job["train_uid_sha256"]):
        raise ClassicalTrainingError("Numerical fit inputs differ from the fixed source job")
    with preserve_rng(), threadpool_limits(limits=1):
        if job["group"] == "bank":
            if job["method"] not in {"B4", "B5"}:
                raise ClassicalTrainingError("Source-bank fit must be an explicit B4/B5 recipe")
            model = (kernels.fit_b4_source(inputs, **job["configuration"]) if job["method"] == "B4"
                     else kernels.fit_b5_source(inputs, **job["configuration"]))
            if model.receipt.get("status") != "SOURCE_CLASSICAL_COMPLETE":
                raise ClassicalTrainingError("Source-bank numerical fit did not converge")
            return kernels._model(model.method, model.arrays, {**model.receipt, "feature_mode": job["feature_mode"],
                "source_plan_job_sha256": job["job_sha256"], "core_legacy_model_reused": False,
                "bank_training_uid_set_sha256": hashlib.sha256("\n".join(sorted(row["event_uid"] for row in inputs.rows)).encode()).hexdigest()})
        if ((job["method"] == CORE_CSP and job["configuration"] != {"m": 2})
                or (job["method"] == CORE_TS and job["configuration"] != {"shrink": .1})):
            raise ClassicalTrainingError("Core source recipes retain fixed v1 m2/shrink0.1")
        covariance, basis = legacy_covariance(x)
        weights = kernels.hierarchical_weights(inputs.rows, y)
        if job["method"] == CORE_CSP:
            means = [np.average(covariance[y == c], axis=0, weights=weights[y == c]) for c in (0, 1)]
            _, vectors = eigh(means[1], means[0] + means[1])
            filters = vectors[:, [0, 1, 8, 9]].T
            features = legacy_csp_features(x, basis, filters)
            classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5, .5]).fit(features, y)
            arrays = {"basis": basis, "filters": filters, "coef": classifier.coef_, "intercept": classifier.intercept_}
            updates = 0
        elif job["method"] == CORE_TS:
            meanlog = np.average(legacy_matrix_function(covariance, np.log), axis=0, weights=weights)
            reference = legacy_matrix_function(meanlog, np.exp)
            eigenvalues, vectors = np.linalg.eigh(reference)
            inverse = (vectors * (eigenvalues ** -.5)) @ vectors.T
            features = legacy_tangent_features(covariance, inverse)
            scaler = StandardScaler().fit(features, sample_weight=weights)
            classifier = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, random_state=0)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(scaler.transform(features), y, sample_weight=weights * len(y))
            if np.any(classifier.n_iter_ >= 2000) or any(issubclass(item.category, ConvergenceWarning) for item in recorded):
                raise ClassicalTrainingError("Legacy source LR did not converge")
            arrays = {"basis": basis, "inverse_root": inverse, "scaler_mean": scaler.mean_,
                      "scaler_scale": scaler.scale_, "coef": classifier.coef_, "intercept": classifier.intercept_}
            updates = int(classifier.n_iter_[0])
        else:
            raise ClassicalTrainingError("Unknown fixed source-classical recipe")
    receipt = {"status": "SOURCE_CLASSICAL_COMPLETE", "kind": "source_model", "synthetic": inputs.synthetic,
               "method": job["method"], "feature_mode": job["feature_mode"], "configuration": job["configuration"],
               "signal_contract_sha256": signal_sha, "source_manifest_sha256": inputs.source_manifest_sha256,
               "held_out_dataset": inputs.held_out_dataset, "source_datasets": sorted({row["dataset"] for row in inputs.rows}),
               "source_event_uids_sha256": digest([row["event_uid"] for row in inputs.rows]),
               "source_rows_sha256": digest(inputs.rows), "source_x_sha256": kernels.array_sha256(x),
               "source_y_sha256": kernels.array_sha256(y), "source_trials": len(x), "n_times": x.shape[2],
               "source_plan_job_sha256": job["job_sha256"], "implementation_sha256": file_digest(__file__),
               "source_classifier_weighting": "unweighted_LDA" if job["method"] == CORE_CSP else "hierarchical_weights_times_N_LR_C1",
               "source_statistics_weighting": "dataset/subject/observed_class/trial_equal",
               "legacy_recipe_independently_reimplemented": True, "legacy_code_executed": False,
               "optimizer_updates": updates, "wall_seconds": time.monotonic() - started, "query_accessed": False}
    return kernels._model(job["method"], arrays, receipt)


def _fit_worker(inputs, job, connection):
    try:
        inputs.validate()
        connection.send(("FIT_STARTED", None))
        model = fit_source_math(inputs, job)
        connection.send(("MODEL", model))
    except BaseException as error:
        connection.send(("ERROR", type(error).__name__))
    finally:
        connection.close()


def fit_with_deadline(inputs, job, *, deadline, on_started, monotonic=time.monotonic):
    """Spawn one CPU worker and terminate it at the remaining explicit budget."""
    if monotonic() >= deadline:
        raise FitWallBudgetExpired("CPU fit wall budget is already exhausted")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_fit_worker, args=(inputs, job, sender), daemon=True)
    process.start()
    sender.close()
    try:
        while monotonic() < deadline:
            if receiver.poll(min(.2, max(0, deadline - monotonic()))):
                try:
                    tag, value = receiver.recv()
                except EOFError:
                    raise ClassicalTrainingError("CPU source-classical worker closed without a model") from None
                if tag == "FIT_STARTED":
                    on_started()
                elif tag == "MODEL":
                    value.assert_unchanged()
                    return value
                else:
                    raise ClassicalTrainingError("CPU numerical fit failed: " + value)
            if not process.is_alive() and not receiver.poll():
                raise ClassicalTrainingError("CPU source-classical worker exited without a model")
        raise FitWallBudgetExpired("CPU fit reached the explicit wall budget; no fit completion claimed")
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


def predict_source_math(model, x):
    model.assert_unchanged()
    with threadpool_limits(limits=1):
        if model.method == CORE_CSP:
            features = legacy_csp_features(x, model.arrays["basis"], model.arrays["filters"])
        elif model.method == CORE_TS:
            covariance, _ = legacy_covariance(x)
            features = (legacy_tangent_features(covariance, model.arrays["inverse_root"])
                        - model.arrays["scaler_mean"]) / model.arrays["scaler_scale"]
        else:
            raise ClassicalTrainingError("Only core LODO models have held-out-source prediction jobs")
        margin = (features @ model.arrays["coef"].T + model.arrays["intercept"]).reshape(-1)
        probability = expit(margin)
        return np.stack((1 - probability, probability), axis=1), np.stack((np.zeros_like(margin), margin), axis=1)


def source_validation_metrics(rows, labels, probabilities):
    """Participant-equal metrics within the single held-out source; no fit."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if (not rows or probabilities.shape != (len(rows), 2) or labels.shape != (len(rows),)
            or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1)
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ClassicalTrainingError("Invalid source-LODO validation values")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("dataset") not in SOURCE_IDS or row.get("role") != "source_train":
            raise PermissionError("Only held-out source validation may be evaluated")
        groups[(row["dataset"], str(row["subject_id"]))].append(index)
    people = []
    prediction = probabilities.argmax(1)
    for (dataset, subject), indices in sorted(groups.items()):
        indices = np.array(indices)
        y = labels[indices]
        observed = sorted(set(y.tolist()))
        loss = -np.log(np.clip(probabilities[indices, y], 1e-12, 1))
        people.append({"dataset": dataset, "subject_id": subject, "trials": len(indices),
                       "cross_entropy": float(loss.mean()), "accuracy": float((prediction[indices] == y).mean()),
                       "balanced_accuracy": float(np.mean([(prediction[indices][y == c] == c).mean() for c in observed]))
                       if observed == [0, 1] else None, "observed_classes": observed})
    if len({person["dataset"] for person in people}) != 1:
        raise ClassicalTrainingError("LODO validation must be exactly one held-out source")
    return {"participant_mean_cross_entropy": float(np.mean([person["cross_entropy"] for person in people])),
            "participant_mean_accuracy": float(np.mean([person["accuracy"] for person in people])),
            "participant_mean_balanced_accuracy": float(np.mean([person["balanced_accuracy"] for person in people]))
                if all(person["balanced_accuracy"] is not None for person in people) else None,
            "participants": people, "aggregation": "participant_equal/trial_equal_within_participant",
            "fit_or_selection_on_validation": False}


def read_budget(root, path, plan):
    if path is None:
        raise ClassicalTrainingError("Execution requires an explicit root-relative CPU budget JSON")
    budget_path = local(root, path)
    value = json.loads(budget_path.read_text())
    if (value.get("stage") != "source_classical_cpu" or value.get("max_jobs") != len(plan["jobs"])
            or value.get("core_jobs") != 10 or value.get("additional_bank_jobs") != plan["planned_additional_bank_fits"]
            or type(value.get("max_fit_attempts")) is not int or value["max_fit_attempts"] < len(plan["jobs"])
            or type(value.get("max_wall_seconds")) not in (int, float) or not math.isfinite(value["max_wall_seconds"])
            or value["max_wall_seconds"] <= 0 or value.get("max_gpu_hours") != 0 or value.get("max_download_bytes") != 0
            or type(value.get("max_fit_input_bytes")) is not int or value["max_fit_input_bytes"] <= 0):
        raise ClassicalTrainingError("Explicit exact job/attempt/wall/input-byte budgets and zero GPU/download required")
    if max(job["input_bytes"] for job in plan["jobs"]) > value["max_fit_input_bytes"]:
        raise ClassicalTrainingError("Source-classical input exceeds the explicit per-fit input-byte budget")
    return value, artifact(root, budget_path)


def _manifest_files(root, folder):
    result = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ClassicalTrainingError("Symlinks cannot enter a classical fit artifact manifest")
        if not path.is_file() or path.name in {"status.json", "artifact_manifest.json"}:
            continue
        if path.suffix not in {".npy", ".json"}:
            raise ClassicalTrainingError("Unexpected non-numerical file in owned fit artifact directory")
        local(root, path.relative_to(root).as_posix())
        result.append(artifact(root, path))
    return result


def _verify_checkpoint(root, folder, state, job):
    receipt = checked_artifact(root, state["model_receipt"])
    model = kernels.load_classical_model(receipt.parent, expected_receipt_sha256=state["model_receipt"]["sha256"])
    if (model.receipt.get("source_plan_job_sha256") != job["job_sha256"] or model.method != job["method"]
            or model.receipt.get("source_event_uids_sha256") != job["train_uid_sha256"]
            or model.receipt.get("signal_contract_sha256") != job["input_contract_sha256"]
            or model.receipt.get("source_manifest_sha256") != job["source_cache_manifest_sha256"]
            or model.receipt.get("source_datasets") != job["train_datasets"]
            or model.receipt.get("synthetic") is not state["synthetic"]):
        raise ClassicalTrainingError("Source numerical checkpoint job identity differs")
    return model


def verify_complete_job(root, folder, run_identity, job):
    state = json.loads((folder / "status.json").read_text())
    if state.get("run_identity_sha256") != run_identity or state.get("job_sha256") != job["job_sha256"]:
        raise ClassicalTrainingError("Completed source job has the wrong identity")
    if state.get("status") not in {"SOURCE_CLASSICAL_FIT_COMPLETE", "SOFTWARE_TEST_DOUBLE_FIT_COMPLETE"}:
        raise ClassicalTrainingError("Source-classical job is not complete")
    manifest_path = checked_artifact(root, state["artifact_manifest"])
    manifest = json.loads(manifest_path.read_text())
    actual = _manifest_files(root, folder)
    if manifest.get("files") != actual or manifest.get("run_identity_sha256") != run_identity:
        raise ClassicalTrainingError("Classical full artifact manifest changed")
    model = _verify_checkpoint(root, folder, state, job)
    for item in manifest["files"]:
        checked_artifact(root, item)
    return state, model


def _runtime():
    return {"device": "cpu", "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": str(torch.__version__),
            "blas_threads": 1, "gpu_checked": False, "gpu_execution": False}


def execute_classical_plan(cache, contract, plan, output, *, budget, budget_descriptor, max_new_fits=None):
    """Internal gated-context state machine; software doubles retain their label."""
    validate_plan(cache, plan)
    if validate_signal_contract(contract) != cache.input_contract_sha256:
        raise ClassicalTrainingError("Execution contract and source plan differ")
    if max_new_fits is not None and (type(max_new_fits) is not int or max_new_fits < 0):
        raise ClassicalTrainingError("Nonnegative bounded new-fit count required")
    root, output = cache.root, Path(output)
    if not output.is_relative_to(root):
        raise ClassicalTrainingError("Source-classical output must be inside the independent root")
    # Internal tests explicitly label a mocked cache capability. Public entry
    # obtains only the real verified cache and rejects fixture markers.
    synthetic = bool(getattr(cache, "software_test_double_scope", None))
    identity = {"plan_sha256": plan["plan_sha256"], "budget": budget_descriptor, "runtime": _runtime(),
                "scope": "SOFTWARE_TEST_DOUBLE" if synthetic else "REAL_SOURCE_CLASSICAL_CPU"}
    identity_sha = digest(identity)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ClassicalTrainingError("An existing CPU classical coordinator owns this output") from None
    started = time.monotonic()
    progress_path = output / "progress.json"
    run_path = output / "identity.json"
    try:
        if run_path.exists():
            if json.loads(run_path.read_text()) != identity:
                raise ClassicalTrainingError("Existing classical run code/data/budget/runtime identity differs")
        else:
            if any(path.name != "run.lock" for path in output.iterdir()):
                raise ClassicalTrainingError("Unowned existing output is preserved; use a new run directory")
            atomic_json(run_path, identity)
            atomic_json(output / "plan.json", plan)
        previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        elapsed_before = previous.get("active_wall_seconds", 0.)
        fit_attempts = previous.get("actual_fit_attempts_started", 0)
        if type(fit_attempts) is not int or fit_attempts < 0 or not isinstance(elapsed_before, (int, float)) or elapsed_before < 0:
            raise ClassicalTrainingError("Invalid persisted CPU classical budget ledger")
        for job in plan["jobs"]:
            path = output / job["job_id"] / "status.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get("run_identity_sha256") != identity_sha or saved.get("job_sha256") != job["job_sha256"]:
                    raise ClassicalTrainingError("Existing source fit checkpoint identity differs")
                fit_attempts = max(fit_attempts, saved.get("fit_attempts_started", 0))
        deadline = started + max(0, budget["max_wall_seconds"] - elapsed_before)
        complete, new_fits, reused = [], 0, 0

        def progress(status):
            core = sum(item["group"] == "core" for item in complete)
            bank = sum(item["group"] == "bank" for item in complete)
            value = {"schema_version": SCHEMA, "status": status, "run_identity_sha256": identity_sha,
                     "synthetic": synthetic, "scope": identity["scope"],
                     "planned_core_fits": 10, "planned_additional_bank_fits": plan["planned_additional_bank_fits"],
                     "completed_core_fit_tasks": core, "completed_additional_bank_fit_tasks": bank,
                     "actual_scientific_fits_complete": 0 if synthetic else len(complete),
                     "actual_fit_attempts_started": fit_attempts, "active_wall_seconds": elapsed_before + time.monotonic() - started,
                     "new_fit_attempts_this_invocation": new_fits, "completed_jobs_reused_this_invocation": reused,
                     "training_started": fit_attempts > 0, "overall_W4_complete": False,
                     "gpu_execution": False, "external_access": False, "completed_jobs": complete}
            atomic_json(progress_path, value)
            return value

        with preserve_rng():
            for job in plan["jobs"]:
                folder = output / job["job_id"]
                state_path = folder / "status.json"
                state = json.loads(state_path.read_text()) if state_path.exists() else None
                if state:
                    if state.get("run_identity_sha256") != identity_sha or state.get("job_sha256") != job["job_sha256"]:
                        raise ClassicalTrainingError("Existing source fit checkpoint identity differs")
                    if state.get("status") in {"SOURCE_CLASSICAL_FIT_COMPLETE", "SOFTWARE_TEST_DOUBLE_FIT_COMPLETE"}:
                        checked, _ = verify_complete_job(root, folder, identity_sha, job)
                        complete.append({"job_id": job["job_id"], "group": job["group"],
                                         "receipt": checked["fit_receipt"], "artifact_manifest": checked["artifact_manifest"],
                                         "model_receipt": checked["model_receipt"]})
                        reused += 1
                        continue
                model = None
                if state and state.get("model_receipt"):
                    model = _verify_checkpoint(root, folder, state, job)
                elif state and (folder / "model/receipt.json").is_file():
                    # Atomic directory rename may succeed immediately before an
                    # interruption of the status write. Recover only a model
                    # that matches this already-owned run/job/input/UID state.
                    candidate = {**state, "model_receipt": artifact(root, folder / "model/receipt.json")}
                    model = _verify_checkpoint(root, folder, candidate, job)
                    state = {**candidate, "status": "MODEL_CHECKPOINTED"}
                    atomic_json(state_path, state)
                if time.monotonic() >= deadline:
                    return progress("PAUSED_WALL_BUDGET")
                if model is None:
                    if max_new_fits is not None and new_fits >= max_new_fits:
                        return progress("PAUSED_NEW_FIT_LIMIT")
                    if fit_attempts >= budget["max_fit_attempts"]:
                        return progress("PAUSED_FIT_ATTEMPT_BUDGET")
                    folder.mkdir(parents=True, exist_ok=True)
                    state = {"run_identity_sha256": identity_sha, "job_sha256": job["job_sha256"], "status": "FIT_PENDING",
                             "synthetic": synthetic, "model_receipt": None}
                    atomic_json(state_path, state)
                    indices = np.array(job["train_indices"], dtype=np.int64)
                    inputs = kernels.SourceTrainingInputs(cache.batch(indices), np.asarray(cache.labels)[indices],
                        tuple(cache.rows[i] for i in indices), contract, cache.manifest_sha256,
                        held_out_dataset=job["held_out_dataset"], synthetic=synthetic)

                    def on_started():
                        nonlocal fit_attempts, new_fits
                        fit_attempts += 1
                        new_fits += 1
                        state.update(status="FIT_RUNNING", fit_attempts_started=fit_attempts)
                        atomic_json(state_path, state)
                        progress("CPU_SOURCE_CLASSICAL_RUNNING")

                    model = fit_with_deadline(inputs, job, deadline=deadline, on_started=on_started)
                    del inputs
                    temporary = folder / f"model.pending-attempt-{fit_attempts}"
                    if temporary.exists():
                        raise ClassicalTrainingError("Uncommitted model output preserved for inspection")
                    kernels.save_classical_model(model, temporary)
                    destination = folder / "model"
                    if destination.exists():
                        raise ClassicalTrainingError("Uncheckpointed model destination exists; refusing overwrite")
                    temporary.rename(destination)
                    state.update(status="MODEL_CHECKPOINTED", model_receipt=artifact(root, destination / "receipt.json"))
                    atomic_json(state_path, state)
                validation = job["validation_indices"]
                predictions, logits = [], []
                for start in range(0, len(validation), 128):
                    if time.monotonic() >= deadline:
                        return progress("PAUSED_WALL_BUDGET")
                    probabilities, batch_logits = predict_source_math(model, cache.batch(np.array(validation[start:start + 128], dtype=np.int64)))
                    predictions.append(probabilities)
                    logits.append(batch_logits)
                metric = None
                if validation:
                    probabilities, all_logits = np.concatenate(predictions), np.concatenate(logits)
                    metric = source_validation_metrics([cache.rows[i] for i in validation], np.asarray(cache.labels)[validation], probabilities)
                    np.save(folder / "validation_probabilities.npy", probabilities, allow_pickle=False)
                    np.save(folder / "validation_logits.npy", all_logits, allow_pickle=False)
                uid_manifest = {"role": "source_train", "held_out_dataset": job["held_out_dataset"],
                    "train_event_uids": [cache.rows[i]["event_uid"] for i in job["train_indices"]],
                    "validation_event_uids": [cache.rows[i]["event_uid"] for i in validation],
                    "training_uid_sha256": job["train_uid_sha256"], "validation_uid_sha256": job["validation_uid_sha256"],
                    "synthetic": synthetic, "scope": identity["scope"]}
                atomic_json(folder / "event_uids.json", uid_manifest)
                receipt = {"status": "SOFTWARE_TEST_DOUBLE_FIT_COMPLETE" if synthetic else "SOURCE_CLASSICAL_FIT_COMPLETE",
                    "schema_version": SCHEMA, "synthetic": synthetic, "scope": identity["scope"],
                    "run_identity_sha256": identity_sha, "job": job, "model_receipt": state["model_receipt"],
                    "source_validation": metric, "actual_fit_tasks_complete": 1, "cpu_only": True,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "external_access": False,
                    "recipe_differences": copy.deepcopy(RECIPE_DIFFERENCES)}
                atomic_json(folder / "fit_receipt.json", receipt)
                atomic_json(folder / "artifact_manifest.json", {"run_identity_sha256": identity_sha, "files": _manifest_files(root, folder)})
                state.update(status=receipt["status"], fit_receipt=artifact(root, folder / "fit_receipt.json"),
                             artifact_manifest=artifact(root, folder / "artifact_manifest.json"))
                atomic_json(state_path, state)
                complete.append({"job_id": job["job_id"], "group": job["group"], "receipt": state["fit_receipt"],
                                 "artifact_manifest": state["artifact_manifest"], "model_receipt": state["model_receipt"]})
                progress("CPU_SOURCE_CLASSICAL_RUNNING")
        result = progress("SOFTWARE_TEST_DOUBLE_COMPLETED" if synthetic else "SOURCE_CLASSICAL_CORE_COMPLETE")
        atomic_json(output / "receipt.json", result)
        return result
    except FitWallBudgetExpired:
        if "progress" in locals():
            return progress("PAUSED_WALL_BUDGET")
        raise
    except BaseException:
        if "progress" in locals():
            progress("INTERRUPTED_NOT_COMPLETE")
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def prepare_classical_sources(root, *, execute=False, include_source_bank=False,
                              budget_path=None, output="runs/source_classical_v2", max_new_fits=None):
    cache, contract = verified_context(root)
    plan = build_classical_plan(cache, include_source_bank=include_source_bank)
    if not execute:
        return {"status": "SOURCE_CLASSICAL_DRY_RUN", "schema_version": SCHEMA, "plan": plan,
                "training_started": False, "actual_scientific_fits_complete": 0, "gpu_checked": False,
                "required_budget": {"stage": "source_classical_cpu", "max_jobs": len(plan["jobs"]),
                    "core_jobs": 10, "additional_bank_jobs": plan["planned_additional_bank_fits"],
                    "max_fit_attempts": len(plan["jobs"]), "max_wall_seconds": "EXPLICIT_POSITIVE_LIMIT_REQUIRED",
                    "max_fit_input_bytes": "EXPLICIT_POSITIVE_LIMIT_REQUIRED", "max_gpu_hours": 0, "max_download_bytes": 0}}
    budget, descriptor = read_budget(cache.root, budget_path, plan)
    return execute_classical_plan(cache, contract, plan, local(cache.root, output, exists=False),
                                  budget=budget, budget_descriptor=descriptor, max_new_fits=max_new_fits)


def load_verified_source_bank(root, receipt_path, *, method, expected_m=None, expected_shrink=None,
                              input_sha, cache_sha, expected_receipt_sha256=None):
    """Formal B4/B5 bank loader, never a legacy alias or raw array capability.

    receipt_path is an independently SHA-bound descriptor or a root-relative
    fit_receipt.json path accompanied by expected_receipt_sha256. All real W1/W2
    gates, the fixed 16-job plan, the ten core receipts, and the selected bank
    numerical artifacts are rechecked. Portable loading does not require the
    inference host to have the source CPU worker's PyTorch build string.
    """
    if method not in {"B4", "B5"}:
        raise ClassicalTrainingError("A legacy core model cannot serve as the B4/B5 source bank")
    if ((method == "B4" and (expected_m not in (1, 2, 3) or expected_shrink is not None))
            or (method == "B5" and (expected_shrink not in (.01, .05, .1) or expected_m is not None))):
        raise ClassicalTrainingError("An exact matching B4 m or B5 shrink is required")
    kernels._sha(input_sha, "expected bank signal input")
    kernels._sha(cache_sha, "expected bank source cache")
    root = Path(root).resolve(strict=True)
    if isinstance(receipt_path, dict):
        descriptor = receipt_path
    else:
        kernels._sha(expected_receipt_sha256, "expected source-bank receipt")
        descriptor = {"path": receipt_path, "sha256": expected_receipt_sha256}
    receipt_file = checked_artifact(root, descriptor)
    receipt = json.loads(receipt_file.read_text())
    if (receipt_file.name != "fit_receipt.json" or receipt.get("status") != "SOURCE_CLASSICAL_FIT_COMPLETE"
            or receipt.get("synthetic") is not False or receipt.get("scope") != "REAL_SOURCE_CLASSICAL_CPU"
            or receipt.get("job", {}).get("method") != method or receipt["job"].get("group") != "bank"
            or receipt["job"].get("phase") != "bank"):
        raise ClassicalTrainingError("A completed real independently fitted B4/B5 bank receipt is required")
    cache, contract = verified_context(root)
    if getattr(cache, "software_test_double_scope", None):
        raise ClassicalTrainingError("Software-test-double caches cannot issue a scientific source-bank capability")
    if input_sha != cache.input_contract_sha256 or cache_sha != cache.manifest_sha256:
        raise ClassicalTrainingError("Source bank and caller input/cache identities differ")
    plan = build_classical_plan(cache, include_source_bank=True)
    expected_configuration = {"m": expected_m} if method == "B4" else {"shrink": expected_shrink}
    job = next(job for job in plan["jobs"] if job["group"] == "bank" and job["method"] == method
               and job["configuration"] == expected_configuration)
    if receipt["job"] != job:
        raise ClassicalTrainingError("Source-bank receipt differs from the fixed matching configuration")
    folder = receipt_file.parent
    depth = len(Path(job["job_id"]).parts)
    output = folder.parents[depth - 1]
    if output / job["job_id"] != folder:
        raise ClassicalTrainingError("Source-bank receipt is outside its registered run/job path")
    identity_file = local(root, (output / "identity.json").relative_to(root).as_posix())
    identity = json.loads(identity_file.read_text())
    if (identity.get("scope") != "REAL_SOURCE_CLASSICAL_CPU" or identity.get("plan_sha256") != plan["plan_sha256"]
            or identity.get("runtime", {}).get("device") != "cpu" or identity["runtime"].get("gpu_execution") is not False
            or receipt["run_identity_sha256"] != digest(identity)):
        raise ClassicalTrainingError("Source-bank fixed run/code/runtime identity differs")
    budget_file = checked_artifact(root, identity.get("budget"))
    read_budget(root, budget_file.relative_to(root).as_posix(), plan)
    plan_file = local(root, (output / "plan.json").relative_to(root).as_posix())
    if json.loads(plan_file.read_text()) != plan:
        raise ClassicalTrainingError("Saved source-bank plan was changed")
    for core_job in (item for item in plan["jobs"] if item["group"] == "core"):
        core_state, core_model = verify_complete_job(root, output / core_job["job_id"], digest(identity), core_job)
        if core_state["status"] != "SOURCE_CLASSICAL_FIT_COMPLETE" or core_model.receipt.get("synthetic") is not False:
            raise ClassicalTrainingError("The ten core fits must have genuine complete source receipts")
    state, model = verify_complete_job(root, folder, digest(identity), job)
    if state["fit_receipt"] != artifact(root, receipt_file):
        raise ClassicalTrainingError("Selected source-bank receipt differs from its complete job status")
    expected_mode = "log_normalized_variance" if method == "B4" else "log_Euclidean_TS_float64"
    expected_uids = hashlib.sha256("\n".join(sorted(row["event_uid"] for row in cache.rows)).encode()).hexdigest()
    if (model.method != method or model.receipt.get("synthetic") is not False
            or model.receipt.get("held_out_dataset") is not None or model.receipt.get("source_datasets") != sorted(SOURCE_IDS)
            or model.receipt.get("feature_mode") != expected_mode or model.receipt.get("core_legacy_model_reused") is not False
            or model.receipt.get("bank_training_uid_set_sha256") != expected_uids
            or any(model.receipt.get(key) != value for key, value in expected_configuration.items())):
        raise ClassicalTrainingError("Source-bank numerical model is a mismatched or legacy prior")
    return model
