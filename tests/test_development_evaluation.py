"""W6 software fixtures only; no real EEG fit, GPU, query job or network.

Episode/store fixtures exercise real transaction and label-capability code.
Selection math tests replace the deliberately blocked scientific verifier with
explicit metadata doubles; their returned values are never research evidence.
"""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from p3work import development_evaluation as development
from p3work.adapt import PredictionResult
from p3work.contracts import digest, file_digest
from p3work.episodes import EvaluationLabels
from p3work.prediction_store import commit_predictions, verify_prediction_receipt
from tests.test_prediction_store import fixture as store_fixture
from tests.test_episodes import write, descriptor

torch.set_num_threads(1)
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path):
    native, _, _, references, cost, _ = store_fixture(tmp_path)
    config_path = tmp_path / references["configuration"]["path"]
    write(config_path, {"method": "P", "configuration": development.DevelopmentJob("P").configuration,
        "scope": "SYNTHETIC_FIXTURE_NOT_RESEARCH_EVIDENCE"})
    references["configuration"] = descriptor(tmp_path, config_path)
    catalog = native.load(role="meta_validation")
    calls = []
    def predict(plan, output):
        calls.append((plan.participant_id, plan.session_id))
        values = torch.linspace(-1, 1, len(plan.query_event_uids), dtype=torch.float64)
        logits = torch.stack((-values, values), 1)
        prediction = PredictionResult(logits, logits.softmax(1), logits.argmax(1),
            plan.query_event_uids, plan.input_contract_sha256)
        measured = {**cost, "valid_support_trials": len(plan.support_event_uids),
            "accessible_labels": len(plan.support_event_uids), "label_access_event_uids": list(plan.support_event_uids)}
        return commit_predictions(tmp_path, output+"/predictions", prediction, plan,
            method="P", pipeline_seed=20261018, references=references, cost=measured,
            purpose="development_evaluation")["receipt"]
    def run(**kwargs):
        return development.run_synthetic_validation(tmp_path, job=development.DevelopmentJob("P"),
            catalog=catalog, output="runs/development_fixture", predict_one=kwargs.pop("predict_one", predict),
            max_fit_attempts=kwargs.pop("max_fit_attempts", 20), max_wall_seconds=600, **kwargs)
    return native, catalog, calls, predict, run


def test_exact_fixed_plan_default_is_read_only_and_has_no_scientific_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("planning must not inspect CUDA"))
    plan = development.build_development_plan(REPO)
    assert plan["screening_jobs"] == 118
    assert plan["additional_confirmation_jobs"] == 28
    assert plan["confirmation_seeds"] == [20261018, 20261019, 20261020]
    assert plan["B_star_candidates"] == [f"B{i}" for i in range(1, 12)]
    assert plan["B_star_BA_tolerance"] == .005
    assert plan["production_execution_status"] == "BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR"
    assert plan["training_started"] is False and plan["external_access"] is False
    assert not list(tmp_path.iterdir())
    assert len(plan["methods"]["B0"]) == 1
    assert all(len(v) == 9 for k, v in plan["methods"].items() if k != "B0")
    body = dict(plan); sha = body.pop("plan_sha256")
    assert digest(body) == sha
    plan["methods"]["B6"][0]["configuration"]["rank"] = 999
    assert development.DevelopmentJob("B6").configuration == {"rank": 1, "radius": .05}


@pytest.mark.parametrize("kwargs", [dict(method="legacy_baseline_v1"), dict(method="B1", configuration_index=9),
    dict(method="B1", configuration_index=True), dict(method="B1", seed=20261019),
    dict(method="B1", phase="confirmation", seed=20261018), dict(method="B1", phase="confirmation", seed=20261021),
    dict(method="B1", phase="internal_holdout")])
def test_nonregistered_job_or_seed_cannot_expand_search(kwargs):
    with pytest.raises(development.DevelopmentGateError):
        development.DevelopmentJob(**kwargs)


def test_real_incomplete_stages_stop_before_artifact_or_device_reads(tmp_path, monkeypatch):
    write(tmp_path / "status/tasks.json", {"W1": {"status": "IN_PROGRESS"},
        "W2": {"status": "COMPLETE"}, "W3": {"status": "COMPLETE"}})
    monkeypatch.setattr(development.registry.Context, "from_verified", lambda *a: pytest.fail("no source load"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("no CUDA check"))
    with pytest.raises(development.DevelopmentGateError, match="W1_W2_W3"):
        development.verify_development_inputs(tmp_path, job=development.DevelopmentJob("B0"), artifact_catalog={})
    with pytest.raises(PermissionError, match="user_GPU_boot"):
        development.execute_validation_job(tmp_path, job=development.DevelopmentJob("B0"), artifact_catalog={}, budget={})


def test_status_runtime_JSON_cannot_authorize_a_measured_production_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(development, "verify_development_inputs", lambda *a, **k: (None, None, {}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("missing validator must stop before CUDA"))
    job = development.DevelopmentJob("B1")
    target = tmp_path / "runs/budget.json"
    write(target, {"stage": "development_evaluation", "job_sha256": job.identity,
        "max_jobs": 1, "max_download_bytes": 0, "max_fit_attempts": 11,
        "max_wall_seconds": 600, "max_gpu_seconds": 600, "max_calibration_seconds": 20,
        "measured_timing_profile": {"status": "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE", "runtime": "forged"}})
    with pytest.raises(development.DevelopmentGateError, match="BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR"):
        development.execute_validation_job(tmp_path, job=job, artifact_catalog={},
            budget=descriptor(tmp_path, target), user_started_gpu=True)
    assert not (tmp_path / "runs/development").exists()


def test_invalid_operational_budget_cannot_be_assumed_from_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(development, "verify_development_inputs", lambda *a, **k: (None, None, {}))
    path = tmp_path / "runs/budget.json"
    write(path, {"stage": "development_evaluation", "max_wall_seconds": 600})
    with pytest.raises(development.DevelopmentGateError, match="explicit_exact_job"):
        development.execute_validation_job(tmp_path, job=development.DevelopmentJob("B1"), artifact_catalog={},
            budget=descriptor(tmp_path, path), user_started_gpu=True)


@pytest.mark.parametrize("mismatch", [None, "seed", "split", "W2"])
def test_read_only_input_binding_uses_same_seed_f0_Rref_full_catalog_and_W2(fixture, monkeypatch, mismatch):
    native, catalog, _, _, _ = fixture
    source = native.root / "runs/verified_inputs/source.json"
    reference = native.root / "runs/verified_inputs/reference.json"
    source_cache = native.root / "cache/source/source_cache.json"
    write(source, {"scope": "READ_ONLY_ARTIFACT_LOADER_TEST_DOUBLE"})
    write(reference, {"scope": "READ_ONLY_ARTIFACT_LOADER_TEST_DOUBLE"})
    write(source_cache, {"scope": "READ_ONLY_ARTIFACT_LOADER_TEST_DOUBLE"})
    catalog_path = native.root / "runs/verified_inputs/catalog.json"
    write(catalog_path, {"schema_version": "paper3-development-artifact-catalog-v1", "seeds": {
        "20261018": {"source_f0": descriptor(native.root, source), "source_reference": descriptor(native.root, reference)}}})
    native.tasks["W2"] = {"status": "COMPLETE", "cache_manifest": descriptor(native.root, source_cache)}
    write(native.root / "status/tasks.json", native.tasks)
    context = SimpleNamespace(pipeline_seed=20261019 if mismatch == "seed" else 20261018,
        signal_contract_sha256=catalog.input_contract_sha256,
        split_sha256="e"*64 if mismatch == "split" else catalog.split_sha256,
        source_cache_sha256="f"*64 if mismatch == "W2" else file_digest(source_cache), identity_sha256="c"*64)
    calls = []
    def load(root, source_path, reference_path, contract):
        calls.append((source_path, reference_path))
        return context
    monkeypatch.setattr(development.registry.Context, "from_verified", load)
    monkeypatch.setattr(development, "load_episode_catalog", lambda *a, **k: catalog)
    monkeypatch.setattr(development, "build_development_plan", lambda *a: {"code": {}, "policy": {}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("read-only gates do not inspect CUDA"))
    kwargs = {"job": development.DevelopmentJob("B0"), "artifact_catalog": descriptor(native.root, catalog_path)}
    if mismatch:
        with pytest.raises(development.DevelopmentGateError):
            development.verify_development_inputs(native.root, **kwargs)
    else:
        loaded, selected, inputs = development.verify_development_inputs(native.root, **kwargs)
        assert loaded is context and selected is catalog
        assert inputs["dependencies"]["source_f0"] == descriptor(native.root, source)
        assert inputs["catalog"]["participants"] == sorted(catalog.partition_people)
    assert calls == [(source.relative_to(native.root).as_posix(), reference.relative_to(native.root).as_posix())]


def test_prediction_commit_full_SHA_always_precedes_label_capability_and_equal_people(fixture, monkeypatch):
    native, catalog, calls, _, run = fixture
    previous = EvaluationLabels.labels
    accessed = []
    def labels(self, **kwargs):
        receipt = kwargs.get("prediction_receipt")
        assert receipt is not None
        value = verify_prediction_receipt(native.root, receipt)
        assert value["episode_plan_sha256"] == self.plan_sha256
        accessed.append(self.plan_sha256)
        return previous(self, **kwargs)
    monkeypatch.setattr(EvaluationLabels, "labels", labels)
    result = run()
    assert result["status"] == "SYNTHETIC_VALIDATION_FIXTURE_COMPLETE"
    assert result["synthetic"] is True and result["scientific_fit_completed"] is False
    assert result["measured"] is False
    assert len(calls) == len(catalog.plans) == 11
    assert len(accessed) >= 11
    assert len(result["participants"]) == 11
    assert result["participant_mean_ba"] == pytest.approx(sum(p["ba"] for p in result["participants"])/11)
    assert result["calibration_median_seconds"] == .125
    before = list(calls)
    assert run()["receipt"] == result["receipt"]
    assert calls == before
    with pytest.raises(development.DevelopmentGateError, match="no_fixture"):
        development.verify_development_receipt(native.root, result["receipt"])


def test_prediction_committed_then_hard_style_interrupt_resumes_without_refit(fixture):
    native, catalog, calls, predict, run = fixture
    class Interrupted(BaseException):
        pass
    def interrupted(plan, output):
        predict(plan, output)
        raise Interrupted("synthetic interruption after immutable prediction transaction")
    with pytest.raises(Interrupted):
        run(predict_one=interrupted)
    progress = json.loads((native.root / "runs/development_fixture/progress.json").read_text())
    assert progress["fit_attempts_dispatched"] == 1 and not progress["sessions"]
    assert progress["active"]["attempt"] == 1
    result = run()
    assert len(calls) == len(catalog.plans) == 11
    assert result["fit_attempts_dispatched"] == 11


def test_uncommitted_attempt_is_retained_and_new_attempt_counted(fixture):
    native, _, calls, predict, run = fixture
    def interrupted(plan, output):
        target = native.root / output / "partial_diagnostic.json"
        write(target, {"scope": "SYNTHETIC_PARTIAL", "not_a_fit": True})
        raise RuntimeError("synthetic interrupted before prediction commit")
    with pytest.raises(RuntimeError):
        run(predict_one=interrupted)
    result = run()
    assert result["fit_attempts_dispatched"] == 12 and len(calls) == 11
    partials = list((native.root / "runs/development_fixture").rglob("partial_diagnostic.json"))
    assert len(partials) == 1


def test_budget_pause_never_claims_complete_and_preserves_one_session(fixture):
    native, _, calls, _, run = fixture
    result = run(max_fit_attempts=1)
    assert result["status"] == "PAUSED_OPERATIONAL_BUDGET"
    assert result["sessions_completed"] == len(calls) == 1
    assert result["scientific_fit_completed"] is False
    assert not (native.root / "runs/development_fixture/receipt.json").exists()


def test_corrupted_committed_array_stops_resume_and_never_refits(fixture):
    native, _, calls, _, run = fixture
    run(max_fit_attempts=1)
    array = next((native.root / "runs/development_fixture").rglob("logits.npy"))
    before = array.read_bytes(); array.write_bytes(before+b"tamper")
    with pytest.raises(ValueError, match="sha256"):
        run()
    assert len(calls) == 1 and array.read_bytes().endswith(b"tamper")


def test_missing_progress_after_real_store_transactions_cannot_trigger_silent_refit(fixture):
    native, _, calls, _, run = fixture
    run(max_fit_attempts=1)
    (native.root / "runs/development_fixture/progress.json").unlink()
    with pytest.raises(development.DevelopmentGateError, match="missing_progress"):
        run()
    assert len(calls) == 1


def test_active_attempt_cannot_redirect_to_other_existing_prediction_directory(fixture):
    native, _, calls, predict, run = fixture
    def interrupted(plan, output):
        predict(plan, output)
        raise RuntimeError("synthetic stop after commit")
    with pytest.raises(RuntimeError):
        run(predict_one=interrupted)
    target = native.root / "runs/development_fixture/progress.json"
    state = json.loads(target.read_text())
    state["active"]["output"] = "runs/different_job"
    write(target, state)
    with pytest.raises(development.DevelopmentGateError, match="invalid_active"):
        run()
    assert len(calls) == 1


def test_synthetic_callback_cannot_return_wrong_method_or_seed_transaction(fixture):
    native, _, _, predict, run = fixture
    def wrong(plan, output):
        saved = predict(plan, output)
        # The JSON proof stays valid; change the *requested* job instead.
        return saved
    with pytest.raises(development.DevelopmentGateError, match="method_seed_or_configuration"):
        development.run_synthetic_validation(native.root, job=development.DevelopmentJob("B1"),
            catalog=fixture[1], output="runs/wrong_method_fixture", predict_one=wrong,
            max_fit_attempts=11, max_wall_seconds=600)


def test_unclean_operational_interval_charges_downtime_before_next_dispatch(fixture, monkeypatch):
    native, _, calls, predict, run = fixture
    clock = [100.]
    original_close = development.PersistentWallBudget.close
    def leave_open(self, *args, **kwargs):
        # Model SIGKILL ownership loss: retain the open interval on disk and
        # release only the file lock. No real process, sleep or fit is used.
        self._lock.close()
        self._closed = True
    def interrupted(plan, output):
        predict(plan, output)
        clock[0] += 2
        raise RuntimeError("synthetic unclean interval")
    monkeypatch.setattr(development.PersistentWallBudget, "close", leave_open)
    with pytest.raises(RuntimeError):
        run(predict_one=interrupted, monotonic=lambda: clock[0], utc_now=lambda: 1000000000+clock[0], boot_id="SYNTHETIC_BOOT")
    monkeypatch.setattr(development.PersistentWallBudget, "close", original_close)
    clock[0] += 700
    with pytest.raises(ValueError, match="wall budget is exhausted"):
        run(monotonic=lambda: clock[0], utc_now=lambda: 1000000000+clock[0], boot_id="SYNTHETIC_BOOT")
    journal = json.loads((native.root / "runs/development_fixture/wall_budget.json").read_text())
    assert journal["durable_wall_seconds"] == 702
    assert len(journal["recovery_events"]) == 1 and len(calls) == 1


def test_frozen_catalog_cannot_drop_participant_or_use_internal_partition(fixture):
    _, catalog, _, _, _ = fixture
    catalog.plans = catalog.plans[:-1]
    with pytest.raises(development.DevelopmentGateError, match="every_frozen"):
        development._catalog_binding(catalog)
    catalog.role = "internal_holdout"
    with pytest.raises(development.DevelopmentGateError, match="meta_validation"):
        development._catalog_binding(catalog)


def test_pending_progress_identity_cannot_be_reused_for_changed_configuration(fixture):
    native, catalog, _, predict, run = fixture
    run(max_fit_attempts=1)
    with pytest.raises(ValueError):
        development.run_synthetic_validation(native.root, job=development.DevelopmentJob("P", 1), catalog=catalog,
            output="runs/development_fixture", predict_one=predict, max_fit_attempts=20, max_wall_seconds=600)


def selection_doubles(monkeypatch):
    """Only fixed metadata math; not an accepted scientific receipt chain."""
    evidence, frozen = {}, {}
    def row(method, index, seed=20261018):
        job = development.DevelopmentJob(method, index, seed, "screening" if seed == 20261018 else "confirmation")
        return {"job": {"method": method, "configuration_index": index, "seed": seed, "phase": job.phase},
            "method": method, "seed": seed, "configuration": job.configuration,
            "participant_mean_ba": .7, "participant_mean_bce": .6,
            "calibration_median_seconds": 1., "catalog": {"test_double": "SAME_VALIDATION"},
            "timing_protocol": development.TIMING_PROTOCOL, "inputs": {"code": {"scope": "TEST_DOUBLE"},
                "device_identity_sha256": "d"*64, "selected_meta_outer_update": None,
                "configuration_receipt": None,
                "dependencies": {"source_f0": {"path": f"runs/fixture/source{seed}.json", "sha256": str(seed)[-1]*64}}}}
    monkeypatch.setattr(development, "verify_development_receipt", lambda root, d: deepcopy(evidence[d["path"]]))
    monkeypatch.setattr(development, "verify_configuration_receipt", lambda root, d: deepcopy(frozen[d["path"]]))
    return row, evidence, frozen


def test_screen_selection_requires_all_nine_and_predeclared_tie_order(tmp_path, monkeypatch):
    row, evidence, _ = selection_doubles(monkeypatch)
    refs = [{"path": f"runs/screen/{i}.json", "sha256": "a"*64} for i in range(9)]
    evidence.update({d["path"]: row("B1", i) for i, d in enumerate(refs)})
    result = development.freeze_screening_configuration(tmp_path, method="B1", screening_receipts=refs)
    assert result["configuration_index"] == 0
    assert result["configuration_tie_rule"] == development.CONFIGURATION_TIE_RULE
    with pytest.raises(development.DevelopmentGateError, match="all_registered"):
        development.freeze_screening_configuration(tmp_path, method="B1", screening_receipts=refs[:-1])
    with pytest.raises(development.DevelopmentGateError, match="complete_same"):
        development.freeze_screening_configuration(tmp_path, method="B1", screening_receipts=refs[:-1]+[refs[0]])
    evidence[refs[1]["path"]]["participant_mean_ba"] += .01
    assert development.freeze_screening_configuration(tmp_path, method="B1", screening_receipts=refs)["configuration_index"] == 1


def confirmation_doubles(monkeypatch):
    row, evidence, frozen = selection_doubles(monkeypatch)
    selections, refs = {}, []
    for method in development.CONTROLS:
        selected = {"path": f"runs/select/{method}.json", "sha256": "b"*64}
        selections[method] = selected
        screen = {"path": f"runs/screen/{method}.json", "sha256": "a"*64}
        frozen[selected["path"]] = {"method": method, "configuration_index": 0,
            "configuration": development.registry.registered_grid(method)[0], "selected_screening_receipt": screen}
        for seed in development.SEEDS:
            d = screen if seed == 20261018 else {"path": f"runs/confirm/{method}/{seed}.json", "sha256": "a"*64}
            value = row(method, 0, seed)
            if seed != 20261018:
                value["inputs"]["configuration_receipt"] = selected
            evidence[d["path"]] = value; refs.append(d)
    return selections, refs, evidence


def test_Bstar_all_eleven_three_seeds_BA_tolerance_same_device_median_ID(tmp_path, monkeypatch):
    selections, refs, evidence = confirmation_doubles(monkeypatch)
    for value in evidence.values():
        value["participant_mean_ba"] = .65
    for d in refs:
        value = evidence[d["path"]]
        if value["method"] == "B1":
            value["participant_mean_ba"] = .8
            value["calibration_median_seconds"] = 3.
        if value["method"] in {"B2", "B3"}:
            value["participant_mean_ba"] = .795
            value["calibration_median_seconds"] = 1.
    result = development.select_strongest_control(tmp_path, configuration_receipts=selections, confirmation_receipts=refs)
    assert result["selected"]["method"] == "B2"
    assert result["timing_statistic"] == "median_of_three_pipeline_session_medians"
    assert result["does_not_complete_internal_check_or_W6"] is True
    assert result["primary_method"] == "P" and result["PCT_primary_weight"] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_seed_source", "different_device", "changed_configuration", "refit_screen_seed", "cross_method_selection"])
def test_Bstar_cannot_ignore_failed_control_or_change_seed_source_query_or_config(tmp_path, monkeypatch, kind):
    selections, refs, evidence = confirmation_doubles(monkeypatch)
    if kind == "missing":
        refs.pop()
    elif kind == "duplicate":
        refs[-1] = refs[0]
    elif kind == "wrong_seed_source":
        evidence[refs[0]["path"]]["inputs"]["dependencies"]["source_f0"]["sha256"] = "f"*64
    elif kind == "different_device":
        evidence[refs[0]["path"]]["inputs"]["device_identity_sha256"] = "e"*64
    elif kind == "changed_configuration":
        evidence[refs[1]["path"]]["configuration"] = development.registry.registered_grid("B1")[1]
    elif kind == "cross_method_selection":
        previous = development.verify_configuration_receipt
        def swapped(root, item):
            value = previous(root, item)
            if item == selections["B6"]:
                value["method"] = "B7"
            return value
        monkeypatch.setattr(development, "verify_configuration_receipt", swapped)
    else:
        value = evidence.pop(refs[0]["path"])
        refs[0] = {"path": "runs/refit/screen_seed.json", "sha256": "c"*64}
        evidence[refs[0]["path"]] = value
    with pytest.raises(development.DevelopmentGateError):
        development.select_strongest_control(tmp_path, configuration_receipts=selections, confirmation_receipts=refs)


def test_fake_real_status_receipt_stays_blocked_and_synthetic_core_cannot_issue_real_completion(tmp_path):
    path = tmp_path / "runs/forged.json"
    write(path, {"schema_version": development.SCHEMA, "status": "DEVELOPMENT_EVALUATION_COMPLETE",
        "synthetic": False, "scientific_fit_completed": True, "measured": True, "role": "meta_validation"})
    with pytest.raises(development.DevelopmentGateError, match="BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR"):
        development.verify_development_receipt(tmp_path, descriptor(tmp_path, path))
    with pytest.raises(development.DevelopmentGateError, match="BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR"):
        development._run_core(tmp_path, job=None, catalog=None, inputs={}, output="runs/forged",
            predict_one=lambda *a: pytest.fail("no production callback"), max_fit_attempts=1,
            max_wall_seconds=1, synthetic=False)


def test_CLI_default_no_training_no_device_check_and_nonzero_uncompleted_exit(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("evaluate_development_cli", REPO / "tools/evaluate_development.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("dryrun no GPU inspection"))
    assert module.main([]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLANNED_NOT_EXECUTED" and result["stages_verified"] is False
    assert result["training_started"] is False


def test_CLI_verify_input_path_becomes_explicit_SHA_descriptor_without_execution(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("evaluate_development_cli_verify", REPO / "tools/evaluate_development.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    path = tmp_path / "runs/artifacts.json"
    write(path, {"schema_version": "SYNTHETIC_ARTIFACT_CATALOG_FIXTURE"})
    calls = []
    def verify(root, **kwargs):
        calls.append(kwargs)
        return None, None, {"scope": "SYNTHETIC_READ_ONLY_LOADER_TEST_DOUBLE"}
    monkeypatch.setattr(module, "verify_development_inputs", verify)
    monkeypatch.setattr(module, "execute_validation_job", lambda *a, **k: pytest.fail("read-only does not execute"))
    assert module.main(["--root", str(tmp_path), "--verify-inputs", "--artifact-catalog", "runs/artifacts.json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "INPUTS_VERIFIED_NOT_EXECUTED"
    assert calls[0]["artifact_catalog"] == descriptor(tmp_path, path)
