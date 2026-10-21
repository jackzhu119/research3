"""SOFTWARE_TEST_DOUBLE: CPU math/transaction tests, never actual source fits.

Mock cache capabilities carry an explicit software scope; resulting run receipts
must retain synthetic=True and zero scientific fits. The public real-data gate
is tested separately and never accepts these fixture manifest declarations.
"""
import ast
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import eigh, helmert
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from test_classical_controls import synthetic_sources
from p3work import classical_training as training
from p3work.contracts import SOURCE_IDS, digest, file_digest
from p3work.methods.classical import SourceTrainingInputs, fit_b4_target, load_classical_model
from tools import train_classical_source as cli


ROOT = Path(__file__).resolve().parents[1]
SCOPE = "SOFTWARE_TEST_DOUBLE_NOT_SCIENTIFIC_SOURCE_TRAINING"
ACTUAL_FIT_WITH_DEADLINE = training.fit_with_deadline


@pytest.fixture
def fixture_cache(tmp_path, monkeypatch):
    root = tmp_path / SCOPE
    inputs = synthetic_sources()
    for relative in training.CODE_FILES + training.POLICY_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    training.atomic_json(root / "contracts/input_contract.json", inputs.signal_contract)
    training.atomic_json(root / "cache/source_fixture.json", {"scope": SCOPE, "synthetic": True})
    rows = [{**row, "subject_id": row["subject"], "label": int(inputs.y[index]),
             "processed_signal_sha256": training.kernels.array_sha256(inputs.x[index])}
            for index, row in enumerate(inputs.rows)]
    cache = SimpleNamespace(root=root, n_times=128, input_contract_sha256=digest(inputs.signal_contract),
        manifest_path="cache/source_fixture.json", manifest_sha256=file_digest(root / "cache/source_fixture.json"),
        arrays=[inputs.x], labels=inputs.y, rows=rows,
        locations=[(0, index) for index in range(len(inputs.y))],
        artifact_sha256={"cache/source_fixture.json": file_digest(root / "cache/source_fixture.json"),
                         "status/tasks.json": "a" * 64},
        software_test_double_scope=SCOPE,
        batch=lambda indices: inputs.x[np.asarray(indices, dtype=np.int64)].copy())
    calls = []
    def software_only_inline(inputs, job, *, deadline, on_started):
        assert inputs.synthetic is True
        assert training.time.monotonic() < deadline
        on_started()
        calls.append(job["job_id"])
        return training.fit_source_math(inputs, job)
    monkeypatch.setattr(training, "fit_with_deadline", software_only_inline)
    monkeypatch.setattr(training, "verified_context", lambda requested: (cache, inputs.signal_contract))
    return SimpleNamespace(cache=cache, contract=inputs.signal_contract, calls=calls)


def budget_for(fixture, plan, **updates):
    value = {"stage": "source_classical_cpu", "max_jobs": len(plan["jobs"]), "core_jobs": 10,
             "additional_bank_jobs": plan["planned_additional_bank_fits"],
             "max_fit_attempts": len(plan["jobs"]) * 2, "max_wall_seconds": 10000,
             "max_fit_input_bytes": 100_000_000, "max_gpu_hours": 0, "max_download_bytes": 0, **updates}
    path = fixture.cache.root / "plans/SOFTWARE_TEST_DOUBLE_budget.json"
    training.atomic_json(path, value)
    return training.read_budget(fixture.cache.root, path.relative_to(fixture.cache.root).as_posix(), plan)


def run(fixture, *, bank=False, limit=None, output="runs/SOFTWARE_TEST_DOUBLE"):
    plan = training.build_classical_plan(fixture.cache, include_source_bank=bank)
    budget, descriptor = budget_for(fixture, plan)
    result = training.execute_classical_plan(fixture.cache, fixture.contract, plan,
        fixture.cache.root / output, budget=budget, budget_descriptor=descriptor, max_new_fits=limit)
    return result, plan


def immutable_legacy_math_reference():
    """Compile only read-only original mathematical definitions in memory.

    No legacy package is imported, no pycache is written under protocol/, and
    every call uses explicit synthetic software inputs, never EEG data.
    """
    namespace = {"np": np, "eigh": eigh, "helmert": helmert, "expit": expit,
                 "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
                 "LogisticRegression": LogisticRegression, "StandardScaler": StandardScaler}
    for filename, names in (("core.py", {"standardized_trials"}),
                            ("models.py", {"covariance", "matrix_function", "tangent_features", "balanced_weights",
                                           "csp_features", "fit_classical", "predict_classical"})):
        path = ROOT / "protocol/legacy_baseline_v1/p3" / filename
        tree = ast.parse(path.read_text())
        selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
                              type_ignores=[])
        exec(compile(ast.fix_missing_locations(selected), f"SOFTWARE_TEST_DOUBLE_REFERENCE/{filename}", "exec"), namespace)
    return namespace


def test_registered_core_ten_and_opt_in_six_are_separate_not_completed(fixture_cache):
    cache = fixture_cache.cache
    core = training.build_classical_plan(cache)
    extended = training.build_classical_plan(cache, include_source_bank=True)
    assert len(core["jobs"]) == core["planned_core_fits"] == 10
    assert extended["planned_core_fits"] == 10 and extended["planned_additional_bank_fits"] == 6
    assert extended["planned_total_fits"] == 16 and extended["core_to_bank_models_reused"] == 0
    assert core["training_started"] is False and core["gpu_execution"] is False
    assert [job for job in extended["jobs"] if job["group"] == "core"] == core["jobs"]
    assert {(job["method"], tuple(job["configuration"].items())) for job in extended["jobs"] if job["group"] == "bank"} == {
        ("B4", (("m", 1),)), ("B4", (("m", 2),)), ("B4", (("m", 3),)),
        ("B5", (("shrink", .01),)), ("B5", (("shrink", .05),)), ("B5", (("shrink", .1),))}
    for job in core["jobs"]:
        training_rows = [cache.rows[index] for index in job["train_indices"]]
        validation_rows = [cache.rows[index] for index in job["validation_indices"]]
        if job["phase"] == "lodo":
            assert {row["dataset"] for row in validation_rows} == {job["held_out_dataset"]}
            assert len({row["dataset"] for row in training_rows}) == 3
        else:
            assert not validation_rows and len({row["dataset"] for row in training_rows}) == 4
        assert not ({(row["dataset"], row["subject_id"]) for row in training_rows}
                    & {(row["dataset"], row["subject_id"]) for row in validation_rows})


@pytest.mark.parametrize("method,kind", [(training.CORE_CSP, "csp_lda"), (training.CORE_TS, "ts_lr")])
def test_exact_core_recipe_matches_immutable_legacy_math_including_float32_roundtrip(fixture_cache, method, kind):
    fixture = fixture_cache
    plan = training.build_classical_plan(fixture.cache)
    job = next(job for job in plan["jobs"] if job["phase"] == "final" and job["method"] == method)
    x, y, rows = fixture.cache.batch(np.array(job["train_indices"])), fixture.cache.labels, tuple(fixture.cache.rows)
    inputs = SourceTrainingInputs(x, y, rows, fixture.contract, fixture.cache.manifest_sha256, synthetic=True)
    model = training.fit_source_math(inputs, job)
    reference = immutable_legacy_math_reference()
    with training.threadpool_limits(limits=1):
        expected = reference["fit_classical"](kind, x, y, rows)
        expected_probability, expected_logits = reference["predict_classical"](kind, expected, x[:9])
    assert model.method == method and model.receipt["synthetic"] is True
    for name, value in expected.items():
        np.testing.assert_array_equal(model.arrays[name], value)
    probability, logits = training.predict_source_math(model, x[:9])
    np.testing.assert_array_equal(probability, expected_probability)
    np.testing.assert_array_equal(logits, expected_logits)
    assert model.receipt["legacy_code_executed"] is False  # Production fit used independent new code.
    if method == training.CORE_CSP:
        assert model.receipt["feature_mode"] == "legacy_log_power"
        with pytest.raises(ValueError, match="incorrect source model"):
            fit_b4_target(model, None)


def test_default_dryrun_never_fits_writes_or_checks_cuda(fixture_cache, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("SOFTWARE_TEST_DOUBLE dry run attempted execution/CUDA")
    monkeypatch.setattr(training, "fit_with_deadline", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    before = {path.relative_to(fixture_cache.cache.root) for path in fixture_cache.cache.root.rglob("*")}
    result = training.prepare_classical_sources(fixture_cache.cache.root)
    after = {path.relative_to(fixture_cache.cache.root) for path in fixture_cache.cache.root.rglob("*")}
    assert before == after
    assert result["status"] == "SOURCE_CLASSICAL_DRY_RUN"
    assert result["actual_scientific_fits_complete"] == 0 and result["training_started"] is False
    assert result["required_budget"]["max_jobs"] == 10


def test_checkpoint_resume_reuses_completed_fits_and_never_calls_them_again(fixture_cache):
    first, plan = run(fixture_cache, limit=3)
    assert first["status"] == "PAUSED_NEW_FIT_LIMIT" and len(fixture_cache.calls) == 3
    assert first["completed_core_fit_tasks"] == 3
    second, _ = run(fixture_cache)
    assert second["status"] == "SOFTWARE_TEST_DOUBLE_COMPLETED"
    assert second["completed_core_fit_tasks"] == 10 and len(fixture_cache.calls) == 10
    assert second["actual_scientific_fits_complete"] == 0 and second["synthetic"] is True
    assert second["actual_fit_attempts_started"] == 10
    third, _ = run(fixture_cache)
    assert len(fixture_cache.calls) == 10
    assert third["new_fit_attempts_this_invocation"] == 0
    assert third["completed_jobs_reused_this_invocation"] == 10
    root = fixture_cache.cache.root
    folder = root / "runs/SOFTWARE_TEST_DOUBLE" / plan["jobs"][0]["job_id"]
    state = json.loads((folder / "status.json").read_text())
    manifest = json.loads(training.checked_artifact(root, state["artifact_manifest"]).read_text())
    assert all(training.checked_artifact(root, item).is_file() for item in manifest["files"])
    assert all(Path(item["path"]).suffix in {".npy", ".json"} for item in manifest["files"])


def test_optional_bank_counts_six_actual_additional_tasks(fixture_cache):
    result, _ = run(fixture_cache, bank=True)
    assert result["completed_core_fit_tasks"] == 10
    assert result["completed_additional_bank_fit_tasks"] == 6
    assert len(fixture_cache.calls) == result["actual_fit_attempts_started"] == 16
    assert result["actual_scientific_fits_complete"] == 0
    methods = []
    for item in result["completed_jobs"]:
        if item["group"] == "bank":
            path = training.checked_artifact(fixture_cache.cache.root, item["model_receipt"])
            model = load_classical_model(path.parent, expected_receipt_sha256=item["model_receipt"]["sha256"])
            assert model.receipt["core_legacy_model_reused"] is False
            methods.append(model.method)
    assert methods.count("B4") == methods.count("B5") == 3


def test_numerical_checkpoint_survives_interruption_during_validation_without_refit(fixture_cache, monkeypatch):
    original = training.predict_source_math
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("SOFTWARE_TEST_DOUBLE validation interruption")
    monkeypatch.setattr(training, "predict_source_math", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(fixture_cache)
    assert len(fixture_cache.calls) == 1
    monkeypatch.setattr(training, "predict_source_math", original)
    result, _ = run(fixture_cache, limit=0)
    assert len(fixture_cache.calls) == 1 and result["completed_core_fit_tasks"] == 1
    assert result["status"] == "PAUSED_NEW_FIT_LIMIT"


def test_changed_plan_runtime_or_checkpoint_artifact_fail_closed(fixture_cache, monkeypatch):
    result, plan = run(fixture_cache, limit=1)
    changed = copy.deepcopy(plan)
    changed["jobs"] = changed["jobs"][:1]
    changed["plan_sha256"] = digest({key: value for key, value in changed.items() if key != "plan_sha256"})
    budget, descriptor = budget_for(fixture_cache, plan)
    with pytest.raises(training.ClassicalTrainingError, match="Complete fixed"):
        training.execute_classical_plan(fixture_cache.cache, fixture_cache.contract, changed,
            fixture_cache.cache.root / "should_not_exist", budget=budget, budget_descriptor=descriptor)
    assert not (fixture_cache.cache.root / "should_not_exist").exists()
    runtime = training._runtime
    monkeypatch.setattr(training, "_runtime", lambda: {**runtime(), "numpy": "different"})
    with pytest.raises(training.ClassicalTrainingError, match="runtime identity"):
        run(fixture_cache)
    monkeypatch.setattr(training, "_runtime", runtime)
    path = training.checked_artifact(fixture_cache.cache.root, result["completed_jobs"][0]["model_receipt"])
    payload = path.parent / "coef.npy"
    payload.write_bytes(payload.read_bytes() + b"SOFTWARE_TEST_DOUBLE_CORRUPTION")
    with pytest.raises((training.ClassicalTrainingError, ValueError), match="manifest|SHA256|hash/size"):
        run(fixture_cache)
    assert len(fixture_cache.calls) == 1


def test_rng_is_restored_even_when_numerical_attempt_fails(fixture_cache, monkeypatch):
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def failure(inputs, job, *, deadline, on_started):
        on_started()
        random.random()
        np.random.random(7)
        torch.rand(7)
        raise RuntimeError("SOFTWARE_TEST_DOUBLE numerical failure")
    monkeypatch.setattr(training, "fit_with_deadline", failure)
    with pytest.raises(RuntimeError):
        run(fixture_cache)
    assert random.getstate() == before[0]
    assert np.random.get_state()[0] == before[1][0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert np.random.get_state()[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    ledger = json.loads((fixture_cache.cache.root / "runs/SOFTWARE_TEST_DOUBLE/progress.json").read_text())
    assert ledger["status"] == "INTERRUPTED_NOT_COMPLETE"
    assert ledger["actual_fit_attempts_started"] == 1 and ledger["actual_scientific_fits_complete"] == 0


@pytest.mark.parametrize("key,value", [("max_jobs", 9), ("core_jobs", 9), ("max_fit_attempts", 9),
                                       ("max_wall_seconds", 0), ("max_wall_seconds", float("inf")),
                                       ("max_gpu_hours", 1), ("max_download_bytes", 1), ("max_fit_input_bytes", 1)])
def test_explicit_budget_is_required_and_invalid_budgets_never_fit(fixture_cache, key, value):
    plan = training.build_classical_plan(fixture_cache.cache)
    if isinstance(value, float) and not np.isfinite(value):
        # Invalid JSON nonfinite values are rejected before a budget is written.
        with pytest.raises(ValueError):
            budget_for(fixture_cache, plan, **{key: value})
    else:
        with pytest.raises(training.ClassicalTrainingError):
            budget_for(fixture_cache, plan, **{key: value})
    assert not fixture_cache.calls


def test_participant_equal_source_validation_does_not_weight_large_people():
    rows = [{"dataset": "Cho2017", "subject_id": "large", "role": "source_train"}] * 3
    rows += [{"dataset": "Cho2017", "subject_id": "small", "role": "source_train"}]
    y = np.zeros(4, dtype=np.int64)
    p = np.array([[.9, .1]] * 3 + [[.1, .9]])
    metric = training.source_validation_metrics(rows, y, p)
    assert metric["participant_mean_cross_entropy"] == pytest.approx((-np.log(.9) - np.log(.1)) / 2)
    assert metric["participant_mean_accuracy"] == .5
    assert metric["participant_mean_balanced_accuracy"] is None
    with pytest.raises(PermissionError):
        training.source_validation_metrics([dict(row, dataset="Stieger2021") for row in rows], y, p)


def test_status_file_unrelated_stages_do_not_change_scientific_source_plan(fixture_cache):
    before = training.build_classical_plan(fixture_cache.cache)
    fixture_cache.cache.artifact_sha256["status/tasks.json"] = "b" * 64
    assert training.build_classical_plan(fixture_cache.cache) == before


def test_real_data_entry_rejects_incomplete_w1_before_any_fit_or_output(tmp_path, monkeypatch, capsys):
    root = tmp_path / SCOPE
    training.atomic_json(root / "status/tasks.json", {"W1": {"status": "IN_PROGRESS"}, "W2": {"status": "COMPLETE"}})
    def forbidden(*args, **kwargs):
        pytest.fail("Unfinished W1 attempted a source fit or CUDA access")
    monkeypatch.setattr(training, "fit_with_deadline", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert cli.main(["--root", str(root), "--execute"]) == 2
    assert "W1 must be COMPLETE" in capsys.readouterr().out
    assert not (root / "runs").exists()


@pytest.mark.parametrize("synthetic,extra", [(True, {}), (False, {"test_double_not_research_evidence": True})])
def test_real_entry_rejects_synthetic_cache_despite_claimed_complete_status(tmp_path, monkeypatch, synthetic, extra):
    root = tmp_path / SCOPE
    inputs = synthetic_sources()
    training.atomic_json(root / "contracts/input_contract.json", inputs.signal_contract)
    training.atomic_json(root / "manifest.json", {"synthetic": synthetic, "full_source_inventory_verified": True,
                                                  "normalization_applied": False, **extra})
    fake = SimpleNamespace(input_contract_sha256=digest(inputs.signal_contract), manifest_path="manifest.json")
    monkeypatch.setattr("tools.run_controlled_pilot.verify_inputs", lambda requested: fake)
    with pytest.raises(training.ClassicalTrainingError, match="software fixtures"):
        training.prepare_classical_sources(root, execute=True)
    assert not (root / "runs").exists()


def test_cpu_worker_deadline_is_bounded_without_cuda_or_completion(fixture_cache):
    inputs = synthetic_sources()
    plan = training.build_classical_plan(fixture_cache.cache)
    started = []
    # This reaches the deadline before a spawned Python worker can fit. The
    # worker is terminated and no model/COMPLETE is returned or persisted.
    with pytest.raises(training.FitWallBudgetExpired):
        ACTUAL_FIT_WITH_DEADLINE(inputs, plan["jobs"][0], deadline=training.time.monotonic() + .01,
                                 on_started=lambda: started.append(True))
    assert not started


def test_actual_spawn_worker_returns_only_a_synthetic_cpu_numerical_model(fixture_cache):
    fixture = fixture_cache
    plan = training.build_classical_plan(fixture.cache)
    job = next(item for item in plan["jobs"] if item["phase"] == "final" and item["method"] == training.CORE_CSP)
    inputs = SourceTrainingInputs(fixture.cache.batch(np.array(job["train_indices"])), fixture.cache.labels,
        tuple(fixture.cache.rows), fixture.contract, fixture.cache.manifest_sha256, synthetic=True)
    starts = []
    model = ACTUAL_FIT_WITH_DEADLINE(inputs, job, deadline=training.time.monotonic() + 20,
                                    on_started=lambda: starts.append(SCOPE))
    assert starts == [SCOPE]
    assert model.method == training.CORE_CSP and model.receipt["synthetic"] is True
    assert model.receipt["source_plan_job_sha256"] == job["job_sha256"]
    assert not (fixture.cache.root / "runs").exists()


def test_source_bank_loader_rejects_core_alias_fixture_completion_and_unbound_receipt(fixture_cache):
    fixture = fixture_cache
    result, _ = run(fixture, bank=True)
    core = next(item for item in result["completed_jobs"] if item["group"] == "core")
    bank = next(item for item in result["completed_jobs"] if item["group"] == "bank")
    arguments = {"method": "B4", "expected_m": 1,
                 "input_sha": fixture.cache.input_contract_sha256, "cache_sha": fixture.cache.manifest_sha256}
    for item in (core, bank):
        with pytest.raises(training.ClassicalTrainingError, match="completed real"):
            training.load_verified_source_bank(fixture.cache.root, item["receipt"], **arguments)
    with pytest.raises(ValueError, match="concrete SHA256"):
        training.load_verified_source_bank(fixture.cache.root, bank["receipt"]["path"], **arguments)
    with pytest.raises(training.ClassicalTrainingError, match="legacy core"):
        training.load_verified_source_bank(fixture.cache.root, core["receipt"],
            **{**arguments, "method": training.CORE_CSP})
    with pytest.raises(training.ClassicalTrainingError, match="exact matching"):
        training.load_verified_source_bank(fixture.cache.root, bank["receipt"], **{**arguments, "expected_m": 4})
    descriptor = {**bank["receipt"], "sha256": "f" * 64}
    with pytest.raises(training.ClassicalTrainingError, match="SHA256"):
        training.load_verified_source_bank(fixture.cache.root, descriptor, **arguments)


def test_wrong_source_math_job_identity_is_rejected_before_any_solver(fixture_cache, monkeypatch):
    fixture = fixture_cache
    plan = training.build_classical_plan(fixture.cache)
    lodo_job = plan["jobs"][0]
    inputs = SourceTrainingInputs(fixture.cache.batch(np.arange(len(fixture.cache.rows))), fixture.cache.labels,
        tuple(fixture.cache.rows), fixture.contract, fixture.cache.manifest_sha256, synthetic=True)
    monkeypatch.setattr(training, "legacy_covariance", lambda *args: pytest.fail("mismatched job attempted fit"))
    with pytest.raises(training.ClassicalTrainingError, match="differ.*fixed source job"):
        training.fit_source_math(inputs, lodo_job)


def test_interrupted_atomic_model_commit_recovers_without_duplicate_fit(fixture_cache, monkeypatch):
    original = training.atomic_json
    interrupted = []
    def simulate_status_interruption(path, value):
        if Path(path).name == "status.json" and value.get("status") == "MODEL_CHECKPOINTED" and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt("SOFTWARE_TEST_DOUBLE immediately after atomic numerical-directory commit")
        return original(path, value)
    monkeypatch.setattr(training, "atomic_json", simulate_status_interruption)
    with pytest.raises(KeyboardInterrupt):
        run(fixture_cache)
    assert len(fixture_cache.calls) == 1
    monkeypatch.setattr(training, "atomic_json", original)
    result, _ = run(fixture_cache, limit=0)
    assert result["completed_core_fit_tasks"] == 1 and len(fixture_cache.calls) == 1
