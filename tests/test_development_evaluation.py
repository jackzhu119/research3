"""W6 software fixtures only; no real EEG fit, GPU, query job or network.

Episode/store fixtures exercise real transaction and label-capability code.
Selection math tests replace the deliberately blocked scientific verifier with
explicit metadata doubles; their returned values are never research evidence.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import shutil
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
from tests.test_method_registry import fixture as registry_fixture

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
    assert plan["production_execution_status"] == "GATED_NOT_EXECUTED_NO_REAL_PROFILE_OR_EVALUATION_CLAIM"
    assert plan["profile_plan"]["registered_calibrations"] == 118
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
    with pytest.raises(development.DevelopmentGateError, match="complete_measured_profile_artifact_descriptor"):
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
    with pytest.raises(AttributeError):
        catalog.plans = catalog.plans[:-1]
    malformed = SimpleNamespace(**{name: getattr(catalog, name) for name in (
        "role", "purpose", "plans", "partition_people", "manifest_sha256", "split_sha256",
        "input_contract_sha256", "identity_sha256")})
    malformed.plans = malformed.plans[:-1]
    with pytest.raises(development.DevelopmentGateError, match="every_frozen"):
        development._catalog_binding(malformed)
    malformed.role = "internal_holdout"
    with pytest.raises(development.DevelopmentGateError, match="meta_validation"):
        development._catalog_binding(malformed)


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
    with pytest.raises(development.DevelopmentGateError, match="complete_production_parameter_prediction_and_budget_proof"):
        development.verify_development_receipt(tmp_path, descriptor(tmp_path, path))
    with pytest.raises(development.DevelopmentGateError, match="BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR"):
        development._run_core(tmp_path, job=None, catalog=None, inputs={}, output="runs/forged",
            predict_one=lambda *a: pytest.fail("no production callback"), max_fit_attempts=1,
            max_wall_seconds=1, synthetic=False)


def test_CLI_default_no_training_no_device_check_and_successful_plan_exit(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("evaluate_development_cli", REPO / "tools/evaluate_development.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("dryrun no GPU inspection"))
    assert module.main([]) == 0
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
    assert module.main(["--root", str(tmp_path), "--verify-inputs", "--artifact-catalog", "runs/artifacts.json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "INPUTS_VERIFIED_NOT_EXECUTED"
    assert calls[0]["artifact_catalog"] == descriptor(tmp_path, path)


def synthetic_warmup(root, directory, context, plan, candidate, runtime):
    """Explicit CPU-fixture metadata; never a measured GPU warm-up claim."""
    write(root / directory / "warmup.json", {
        "status": "SUPPORT_SOURCE_FORWARD_WARMUP_COMPLETE", "synthetic": True,
        "candidate_identity_sha256": digest(candidate), "runtime_profile": runtime,
        "source_context_identity_sha256": context.identity_sha256,
        "signal_contract_sha256": context.signal_contract_sha256,
        "support_event_uids": list(plan.support_event_uids), "acquired_event_uids": list(plan.acquired_event_uids),
        "source_forward_calls": development.WARMUP_FORWARDS, "target_optimizer_updates": 0,
        "wall_seconds": .01, "query_labels_accessed": False, "final_protocol_freeze_sha256": None})


def synthetic_profile(registry_fixture, monkeypatch, *, method="B1"):
    """Actual tiny CPU kernel states, explicitly synthetic on every artifact.

    Only the expensive production source/corpus proof loader is replaced. The
    fit artifact serializer/loader, numeric checkpoint SHA, candidate timing,
    complete-grid validator and backup manifest remain real implementations.
    """
    from p3work import fit_artifacts
    from p3work.episodes import EpisodePlan
    root = registry_fixture["root"]
    for name in fit_artifacts.CODE_FILES:
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    context = registry_fixture["build"]()
    support = registry_fixture["support"]("meta_validation_support")
    if len(support.acquired_event_uids) != 20:
        support = replace(support, acquired_event_uids=tuple(f"s{i}" for i in range(20)))
    plans = tuple(EpisodePlan(f"S{i}", "1", "meta_validation", support.acquired_event_uids, support.event_uids,
        tuple(f"q{i}-{j}" for j in range(20)), (), (), (), True, None, "SYNTHETIC_METADATA_BLOCK",
        "d"*64, context.split_sha256, context.signal_contract_sha256,
        digest({"synthetic_participant": i}), json.dumps({"synthetic_participant": i})) for i in range(1, 12))
    catalog = SimpleNamespace(role="meta_validation", purpose="model_selection", plans=plans,
        partition_people=tuple(p.participant_id for p in plans), manifest_sha256="d"*64,
        split_sha256=context.split_sha256, input_contract_sha256=context.signal_contract_sha256,
        identity_sha256="e"*64,
        materialize_support=lambda p, **kwargs: support,
        materialize=lambda *a, **k: pytest.fail("profile must not materialize query/EvaluationLabels"))
    plan = development.build_calibration_profile_plan((method,))
    jobs = [development.DevelopmentJob(**r["job"]) for r in plan["jobs"]]
    runtime = {"scope": "SYNTHETIC_CPU_NOT_GPU_TIMING", "timing_protocol": development.TIMING_PROTOCOL,
        "device_total_memory_bytes": 1024}
    dependencies = {j.identity: {"source_f0": {"path": "source.json", "sha256": context.source_receipt_sha256},
        "source_reference": {"path": "reference.json", "sha256": context.reference_receipt_sha256},
        "learned": None, "source_bank": None} for j in jobs}
    identity = {"schema_version": development.PROFILE_SCHEMA, "plan": plan,
        "dependencies": dependencies, "runtime_profile": runtime, "fixture_scope": "SYNTHETIC_ONLY"}
    identity_sha = digest(identity)
    output = "runs/profile_fixture"
    base = root / output
    write(base / "identity.json", identity)
    initial = {"stage": "development_calibration_profile", "plan_sha256": plan["plan_sha256"],
        "max_jobs": 1, "max_download_bytes": 0, "max_fit_attempts": len(jobs), "max_wall_seconds": 600,
        "max_gpu_seconds": 600, "max_calibration_seconds": 20}
    write(base / "initial_budget.json", initial)
    initial_ref = descriptor(root, base / "initial_budget.json")
    records = []
    for attempt, job in enumerate(jobs, 1):
        candidate = development.candidate_identity(context, job, plans[0], purpose="calibration_engineering_profile",
            execution_identity_sha256=identity_sha, initial_budget=initial_ref, attempt=attempt)
        candidate["dependencies"] = dependencies[job.identity]
        directory = output+f"/fits/{job.identity}/attempt-{attempt:06d}"
        fitted = context.fit_development(method, support, job.configuration)
        measured = .1+attempt*.01
        synthetic_warmup(root, directory, context, plans[0], candidate, runtime)
        timing = {"schema_version": development.SCHEMA, "status": "CALIBRATION_MEASURED", "synthetic": True,
            "candidate_identity_sha256": digest(candidate), "runtime_profile": runtime,
            "timing_protocol": development.TIMING_PROTOCOL, "started_at_utc": "2026-10-18T00:00:00+00:00",
            "finished_at_utc": "2026-10-18T00:00:00.500000+00:00", "calibration_wall_seconds": measured,
            "measurement_samples": 1, "p50_calibration_wall_seconds": measured, "p90_calibration_wall_seconds": measured,
            "calibration_deadline_seconds": 20., "cuda_event_elapsed_seconds": 0., "peak_GPU_allocated_bytes": 0,
            "peak_GPU_reserved_bytes": 0,
            "GPU_memory_scope": development.GPU_MEMORY_SCOPE,
            "query_labels_accessed": False, "final_protocol_freeze_sha256": None}
        write(root / directory / "timing.json", timing)
        write(root / directory / "cost.json", {**development._fit_statistics(fitted, plans[0]), "calibration_wall_seconds": measured})
        fit_artifacts.save_registered_fit(context, fitted, directory+"/fitted", candidate_identity=candidate,
            software_test_double=True)
        _, saved, _, _ = development._candidate_load(root, directory, context, job, plans[0], candidate, runtime,
            software_test_double=True, allow_pointer_recovery=True)
        records.append({"job": asdict(job), "attempt": attempt, "candidate_state": saved})
    clock = [100.]
    journal = development.PersistentWallBudget(base / "wall_budget.json", identity_sha, 600,
        monotonic=lambda: clock[0], utc_now=lambda: 1000000000+clock[0], boot_id="SYNTHETIC_BOOT")
    journal.start_interval(); clock[0] += 10
    counter = {"identity_sha256": identity_sha, "fit_attempts_dispatched": len(jobs),
        "initial_fit_attempt_limit": len(jobs), "fit_attempt_limit": len(jobs), "fit_limit_amendments": [],
        "initial_gpu_limit_seconds": 600, "gpu_limit_seconds": 600, "gpu_limit_amendments": []}
    completion = development._completion_record(root, output, journal, counter, initial, initial_ref, schema=development.PROFILE_SCHEMA)
    journal.close()
    write(base / "completion_budget.json", completion)
    result = {"schema_version": development.PROFILE_SCHEMA, "status": "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE",
        "synthetic": True, "identity_sha256": identity_sha, "identity": descriptor(root, base / "identity.json"),
        "initial_budget": initial_ref, "catalog": development._catalog_binding(catalog),
        "profile_session": json.loads(json.dumps(plans[0].to_dict())), "input_summary": development._input_summary(context),
        "runtime_profile": runtime, "methods": [method], "records": records,
        "actual_calibrations_completed": len(jobs), "fit_attempts_dispatched": len(jobs),
        "query_predictions_computed": False, "query_labels_accessed": False, "scientific_selection_allowed": False,
        "timing_protocol": development.TIMING_PROTOCOL, "final_protocol_freeze_sha256": None,
        "completion_budget": descriptor(root, base / "completion_budget.json"), "publication_wall_seconds": 11.}
    write(base / "receipt.json", result)
    reference = descriptor(root, base / "receipt.json")
    refs = [result["identity"], initial_ref, result["completion_budget"], reference, completion["wall_journal_snapshot"]]
    for row in records:
        refs.extend(development._candidate_artifacts(root, row["candidate_state"]))
    write(base / "artifact_manifest.json", development._artifact_manifest(root, output, refs))
    monkeypatch.setattr(development, "_verified_profile_context", lambda *a, **k: (context, catalog, jobs, plans[0], support))
    return root, context, catalog, result, reference


def test_complete_profile_proof_checks_every_actual_saved_parameter_without_refit_or_labels(registry_fixture, monkeypatch):
    root, context, catalog, result, reference = synthetic_profile(registry_fixture, monkeypatch)
    monkeypatch.setattr(development.registry.Context, "fit_development", lambda *a, **k: pytest.fail("profile verification cannot refit"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("read-only proof verification cannot inspect CUDA"))
    proof = development.verify_calibration_profile(root, reference, software_test_double=True)
    assert proof.synthetic is True and proof.methods == ("B1",)
    assert proof.maximum_calibration_seconds["B1"] == pytest.approx(.19)
    assert result["actual_calibrations_completed"] == 9
    assert result["query_labels_accessed"] is False
    with pytest.raises(development.DevelopmentGateError, match="complete_typed"):
        development.verify_calibration_profile(root, reference)


@pytest.mark.parametrize("kind", ["missing_parameter", "changed_parameter", "short_grid", "changed_cost", "changed_timing", "changed_warmup", "changed_prefix", "over_budget"])
def test_profile_JSON_flags_never_substitute_for_full_parameter_time_cost_and_grid_proof(registry_fixture, monkeypatch, kind):
    root, _, _, result, reference = synthetic_profile(registry_fixture, monkeypatch)
    first = result["records"][0]
    candidate_path = root / first["candidate_state"]["path"]
    candidate_state = json.loads(candidate_path.read_text())
    model = json.loads((root / candidate_state["parameters"]["path"]).read_text())
    if kind == "missing_parameter":
        (root / model["parameters"]["path"]).unlink()
    elif kind == "changed_parameter":
        path = root / model["parameters"]["path"]
        path.write_bytes(path.read_bytes()+b"tamper")
    elif kind == "short_grid":
        result["records"].pop()
        write(root / reference["path"], result); reference = descriptor(root, root / reference["path"])
    elif kind in {"changed_cost", "changed_timing", "changed_warmup"}:
        key = {"changed_cost": "cost", "changed_timing": "timing", "changed_warmup": "warmup"}[kind]
        path = root / candidate_state[key]["path"]
        body = json.loads(path.read_text())
        body[{"cost": "optimizer_updates", "timing": "calibration_wall_seconds", "warmup": "source_forward_calls"}[key]] = 0
        write(path, body)
    elif kind == "changed_prefix":
        candidate_state["candidate_identity"]["acquired_event_uids"][-1] = "unpaid-replacement"
        write(candidate_path, candidate_state)
    else:
        result["publication_wall_seconds"] = 601.
        write(root / reference["path"], result); reference = descriptor(root, root / reference["path"])
    with pytest.raises((ValueError, FileNotFoundError)):
        development.verify_calibration_profile(root, reference, software_test_double=True)


def test_profile_plan_is_fixed_whole_grids_and_resource_measurement_never_scientific_selection():
    assert development.build_calibration_profile_plan()["registered_calibrations"] == 118
    assert development.build_calibration_profile_plan(("B0",))["registered_calibrations"] == 1
    assert development.build_calibration_profile_plan(("B1",))["registered_calibrations"] == 9
    assert development.build_calibration_profile_plan(("B1",))["scientific_selection_allowed"] is False
    for methods in (("B1", "B1"), ("B2", "B1"), ("unknown",), ()):
        with pytest.raises(development.DevelopmentGateError):
            development.build_calibration_profile_plan(methods)


def test_parameter_checkpoint_then_interruption_restores_same_candidate_without_fit_or_GPU(registry_fixture, monkeypatch):
    root, context, catalog, result, _ = synthetic_profile(registry_fixture, monkeypatch, method="B0")
    record = result["records"][0]
    state_path = root / record["candidate_state"]["path"]
    state = json.loads(state_path.read_text())
    candidate = state["candidate_identity"]
    directory = state_path.parent.relative_to(root).as_posix()
    # Simulate the atomic parameter directory being published just before a
    # process kill: its pointer was never committed. Cost/timing predate it.
    state_path.unlink()
    monkeypatch.setattr(development.registry.Context, "fit_development", lambda *a, **k: pytest.fail("checkpoint resume must not fit"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("checkpoint resume must not initialize CUDA"))
    restored, saved, timing, cost = development._measure_candidate(root, directory, context,
        development.DevelopmentJob("B0"), catalog, catalog.plans[0], candidate, result["runtime_profile"],
        deadline_seconds=20, wall_budget=None, software_test_double=True)
    assert saved == record["candidate_state"]
    assert cost["optimizer_updates"] == cost["accessible_labels"] == 0
    assert restored.method == "B0" and state_path.is_file()


def test_saved_parameters_with_missing_cost_never_trigger_unconditional_refit(registry_fixture, monkeypatch):
    root, context, catalog, result, _ = synthetic_profile(registry_fixture, monkeypatch, method="B0")
    path = root / result["records"][0]["candidate_state"]["path"]
    state = json.loads(path.read_text())
    (root / state["cost"]["path"]).unlink()
    monkeypatch.setattr(development.registry.Context, "fit_development", lambda *a, **k: pytest.fail("incomplete evidence is preserved, not refit"))
    with pytest.raises(FileNotFoundError):
        development._measure_candidate(root, path.parent.relative_to(root).as_posix(), context,
            development.DevelopmentJob("B0"), catalog, catalog.plans[0], state["candidate_identity"], result["runtime_profile"],
            deadline_seconds=20, wall_budget=None, software_test_double=True)
    assert path.exists()


def test_completion_publication_checks_budget_after_payload_hashing_and_can_resume_uncommitted_manifest(tmp_path):
    directory = tmp_path / "runs/publication_fixture"
    write(directory / "payload.json", {"synthetic": True})
    refs = [descriptor(tmp_path, directory / "payload.json")]
    class Clock:
        def __init__(self, stop): self.calls = 0; self.stop = stop
        def checkpoint(self): pass
        @property
        def wall_seconds(self): return 2.
        @property
        def exhausted(self):
            self.calls += 1
            return self.stop and self.calls >= 2
    result = {"status": "SYNTHETIC_PUBLICATION_FIXTURE_ONLY", "synthetic": True}
    with pytest.raises(development.DevelopmentGateError, match="after_manifest"):
        development._publish_completion(tmp_path, "runs/publication_fixture", result, refs, Clock(True))
    assert not (directory / "receipt.json").exists()
    assert len(list((directory / ".publication").glob("*.json"))) == 1
    saved = development._publish_completion(tmp_path, "runs/publication_fixture", result, refs, Clock(False))
    assert saved == descriptor(tmp_path, directory / "receipt.json")
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    assert {Path(r["path"]).name for r in manifest["files"]} == {"payload.json", "receipt.json"}


def test_CLI_exception_after_entering_execution_reports_unknown_started_state(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("evaluate_development_cli_execution", REPO / "tools/evaluate_development.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    write(tmp_path / "runs/catalog.json", {"scope": "SYNTHETIC"})
    write(tmp_path / "runs/budget.json", {"scope": "SYNTHETIC"})
    def interrupted(*args, **kwargs):
        raise TimeoutError("SYNTHETIC: dispatch may already have happened")
    monkeypatch.setattr(module, "execute_validation_job", interrupted)
    assert module.main(["--root", str(tmp_path), "--artifact-catalog", "runs/catalog.json", "--budget", "runs/budget.json", "--execute"]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["execution_attempted"] is True and error["training_started"] is None


def registered_evaluation_fixture(registry_fixture, monkeypatch):
    """CPU-only registered kernel/store transaction, never production evidence."""
    from torch import nn
    from p3work.adapt import module_state_sha256
    from p3work.contracts import SOURCE_IDS
    from p3work import fit_artifacts
    root = registry_fixture["root"]
    native, *_ = store_fixture(root)
    catalog = native.load(role="meta_validation")
    for name in development.CODE_FILES + development.POLICY_FILES:
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    source_cache = root / "cache/source/source_cache.json"
    write(source_cache, {"scope": "SYNTHETIC_SOURCE_CACHE_LOADER_DOUBLE"})
    native.tasks["W2"] = {"status": "COMPLETE", "cache_manifest": descriptor(root, source_cache)}
    write(root / "status/tasks.json", native.tasks)
    contract = registry_fixture["contract"]
    contract.clear(); contract.update(deepcopy(native.contract))
    source = registry_fixture["source"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(661)
        source.encoder = nn.Sequential(nn.Flatten(), nn.Linear(11*256, 3).double(), nn.Tanh())
    for parameter in source.parameters(): parameter.requires_grad_(False)
    source.eval(); source.frozen_state_sha256 = module_state_sha256(source)
    source_path = root / "runs/source_fixture/source_f0_receipt.json"
    reference_path = root / "runs/source_fixture/reference.json"
    write(source_path, {"status": "SOURCE_F0_COMPLETE", "synthetic": False,
        "pipeline_seed": 20261018, "training_dataset_ids": sorted(SOURCE_IDS),
        "signal_contract_sha256": native.contract_sha, "scope": "SYNTHETIC_SOURCE_LOADER_DOUBLE_NOT_RESEARCH_EVIDENCE"})
    write(reference_path, {"scope": "SYNTHETIC_REFERENCE_LOADER_DOUBLE_NOT_RESEARCH_EVIDENCE"})
    registry_fixture["source_info"].update(signal_contract_sha256=native.contract_sha,
        source_cache_manifest_sha256=file_digest(source_cache))
    registry_fixture["reference_info"].update(signal_contract_sha256=native.contract_sha,
        source_f0_receipt_sha256=file_digest(source_path), source_cache_manifest={"sha256": file_digest(source_cache)})
    context = development.registry.Context.from_verified(root, source_path.relative_to(root).as_posix(),
        reference_path.relative_to(root).as_posix(), native.contract)
    registry_fixture["build"] = lambda: context
    registry_fixture["support"] = lambda *a, **k: catalog.materialize_support(catalog.plans[0])
    _, _, _, profile_result, profile_ref = synthetic_profile(registry_fixture, monkeypatch)
    proof = development.verify_calibration_profile(root, profile_ref, software_test_double=True)
    catalog_file = root / "runs/fixture_artifacts.json"
    write(catalog_file, {"schema_version": "paper3-development-artifact-catalog-v1", "seeds": {"20261018": {
        "source_f0": descriptor(root, source_path), "source_reference": descriptor(root, reference_path)}}})
    job = development.DevelopmentJob("B1")
    limits = {"stage": "development_evaluation", "job_sha256": job.identity, "max_jobs": 1,
        "max_download_bytes": 0, "max_fit_attempts": 12, "max_wall_seconds": 600, "max_gpu_seconds": 600,
        "max_calibration_seconds": 20, "measured_timing_profile": profile_ref}
    calls = []
    def make_candidate(root, directory, ctx, requested, cat, plan, candidate, runtime):
        if development._candidate_exists(root, directory):
            return development._candidate_load(root, directory, ctx, requested, plan, candidate, runtime, software_test_double=True)
        calls.append((plan.participant_id, plan.session_id))
        fitted = ctx.fit_development(requested.method, cat.materialize_support(plan), requested.configuration)
        synthetic_warmup(root, directory, ctx, plan, candidate, runtime)
        timing = {"schema_version": development.SCHEMA, "status": "CALIBRATION_MEASURED", "synthetic": True,
            "candidate_identity_sha256": digest(candidate), "runtime_profile": runtime,
            "timing_protocol": development.TIMING_PROTOCOL, "started_at_utc": "2026-10-18T00:00:00+00:00",
            "finished_at_utc": "2026-10-18T00:00:00.125000+00:00", "calibration_wall_seconds": .125,
            "measurement_samples": 1, "p50_calibration_wall_seconds": .125, "p90_calibration_wall_seconds": .125,
            "calibration_deadline_seconds": 20., "cuda_event_elapsed_seconds": 0., "peak_GPU_allocated_bytes": 0,
            "peak_GPU_reserved_bytes": 0, "query_labels_accessed": False, "final_protocol_freeze_sha256": None}
        timing["GPU_memory_scope"] = development.GPU_MEMORY_SCOPE
        write(root / directory / "timing.json", timing)
        write(root / directory / "cost.json", {**development._fit_statistics(fitted, plan), "calibration_wall_seconds": .125})
        fit_artifacts.save_registered_fit(ctx, fitted, directory+"/fitted", candidate_identity=candidate, software_test_double=True)
        return development._candidate_load(root, directory, ctx, requested, plan, candidate, runtime,
            software_test_double=True, allow_pointer_recovery=True)
    def run(**kwargs):
        return development._execute_validation_production(root, job=job, artifact_catalog=descriptor(root, catalog_file),
            limits=limits, profile=proof, runtime=profile_result["runtime_profile"], configuration_receipt=None,
            output="runs/registered_evaluation_fixture", software_test_double=True,
            candidate_producer=make_candidate, **kwargs)
    return native, catalog, context, calls, run


def test_registered_full_candidate_to_prediction_resume_and_receipt_reverification_stays_synthetic(registry_fixture, monkeypatch):
    native, catalog, context, calls, run = registered_evaluation_fixture(registry_fixture, monkeypatch)
    class Interrupted(BaseException): pass
    def interrupt_after_parameter_commit(*args, **kwargs):
        raise Interrupted("SYNTHETIC: fitted parameters committed, prediction not yet saved")
    with pytest.raises(Interrupted):
        run(prediction_producer=interrupt_after_parameter_commit)
    progress_path = native.root / "runs/registered_evaluation_fixture/progress.json"
    progress = json.loads(progress_path.read_text())
    assert progress["fit_attempts_dispatched"] == 1 and progress["active"]["phase"] == "FIT_PARAMETERS_COMMITTED"
    assert len(calls) == 1
    result = run()
    assert result["status"] == "SYNTHETIC_REGISTERED_VALIDATION_FIXTURE_COMPLETE"
    assert result["synthetic"] is True and result["scientific_fit_completed"] is False and result["measured"] is False
    assert len(calls) == result["fit_attempts_dispatched"] == len(catalog.plans) == 11
    monkeypatch.setattr(development.registry.Context, "fit_development", lambda *a, **k: pytest.fail("receipt verification must never refit"))
    verified = development.verify_synthetic_development_receipt(native.root, result["receipt"])
    assert verified["participant_mean_ba"] == result["participant_mean_ba"]
    assert run()["receipt"] == result["receipt"]
    with pytest.raises(development.DevelopmentGateError, match="no_fixture"):
        development.verify_development_receipt(native.root, result["receipt"])


def test_production_loop_cannot_accept_caller_fit_or_prediction_callback(tmp_path):
    with pytest.raises(development.DevelopmentGateError, match="cannot_accept_caller"):
        development._execute_validation_production(tmp_path, job=None, artifact_catalog=None, limits=None,
            profile=None, runtime=None, configuration_receipt=None, output=None,
            candidate_producer=lambda *a: pytest.fail("not authorized"))


def test_profile_memory_proof_rejects_consistently_rehashed_impossible_reserved_peak(registry_fixture, monkeypatch):
    root, _, _, result, reference = synthetic_profile(registry_fixture, monkeypatch, method="B0")
    record = result["records"][0]
    state_path = root / record["candidate_state"]["path"]
    state = json.loads(state_path.read_text())
    timing_path = root / state["timing"]["path"]
    timing = json.loads(timing_path.read_text())
    timing["peak_GPU_reserved_bytes"] = 1025
    write(timing_path, timing)
    state["timing"] = descriptor(root, timing_path)
    write(state_path, state)
    record["candidate_state"] = descriptor(root, state_path)
    write(root / reference["path"], result)
    reference = descriptor(root, root / reference["path"])
    with pytest.raises(development.DevelopmentGateError, match="timing_proof_invalid"):
        development.verify_calibration_profile(root, reference, software_test_double=True)


def test_missing_wall_ledger_after_parameter_commit_blocks_resume_without_reset_or_refit(registry_fixture, monkeypatch):
    native, _, _, calls, run = registered_evaluation_fixture(registry_fixture, monkeypatch)
    def interrupted(*a, **k): raise RuntimeError("SYNTHETIC after parameter commit")
    with pytest.raises(RuntimeError): run(prediction_producer=interrupted)
    (native.root / "runs/registered_evaluation_fixture/wall_budget.json").unlink()
    with pytest.raises(development.DevelopmentGateError, match="missing_wall_ledger"):
        run()
    assert len(calls) == 1
    assert not (native.root / "runs/registered_evaluation_fixture/wall_budget.json").exists()


@pytest.mark.parametrize("evidence", ["initialized_marker", "warmup_receipt"])
def test_missing_wall_ledger_blocks_after_source_warmup_without_target_fit(tmp_path, evidence):
    output = "runs/warmup_metadata_only"
    state = {"fit_attempts_dispatched": 0, "active": None, "completed": {}}
    if evidence == "initialized_marker":
        state["wall_ledger_initialized"] = True
    else:
        write(tmp_path / output / "fits/job/attempt-000001/warmup.json", {
            "synthetic": True, "source_forward_calls": 2, "target_optimizer_updates": 0})
    with pytest.raises(development.DevelopmentGateError, match="missing_wall_ledger"):
        development._resource_history_gate(tmp_path, output, state, work_directory="fits")
    assert not (tmp_path / output / "wall_budget.json").exists()


def test_profile_budget_cannot_expand_self_reported_wall_or_GPU_cap_without_actual_journal_amendments(registry_fixture, monkeypatch):
    root, _, _, result, reference = synthetic_profile(registry_fixture, monkeypatch, method="B0")
    path = root / result["completion_budget"]["path"]
    completion = json.loads(path.read_text())
    completion["wall_budget"].update(wall_seconds=1000., max_wall_seconds=6000.)
    completion["max_gpu_seconds"] = 6000.
    write(path, completion)
    result["completion_budget"] = descriptor(root, path)
    result["publication_wall_seconds"] = 1001.
    write(root / reference["path"], result)
    reference = descriptor(root, root / reference["path"])
    with pytest.raises(development.DevelopmentGateError, match="durable_amendment_chain"):
        development.verify_calibration_profile(root, reference, software_test_double=True)


def test_full_prediction_store_with_other_same_seed_source_is_rejected_before_label_release(registry_fixture, monkeypatch):
    from p3work.prediction_store import COST_OPTIONAL, COST_REQUIRED
    native, _, _, calls, run = registered_evaluation_fixture(registry_fixture, monkeypatch)
    class Interrupted(BaseException): pass
    def different_source(root, directory, context, job, catalog, plan, fitted, cost, configuration):
        foreign = root / "runs/foreign_source/source_f0_receipt.json"
        body = json.loads((root / context.source_receipt_path).read_text())
        body["source_branch"] = "OTHER_SYNTHETIC_SAME_SEED_BRANCH"
        write(foreign, body)
        episode = catalog.materialize(plan)
        with torch.no_grad(): prediction = context.predict(fitted, episode.prediction_inputs)
        manifest = native.manifest
        refs = {"source_f0": descriptor(root, foreign), "model": descriptor(root, root / directory / "fitted/receipt.json"),
            "method_code": descriptor(root, root / "p3work/method_registry.py"), "configuration": configuration,
            "input_contract": manifest["input_contract"], "participant_split": manifest["participant_split"],
            "episode_manifest": descriptor(root, native.manifest_path)}
        measured = {k: v for k, v in cost.items() if k in COST_REQUIRED | COST_OPTIONAL}
        commit_predictions(root, directory+"/predictions", prediction, plan, method=job.method,
            pipeline_seed=job.seed, references=refs, cost=measured, purpose="development_evaluation")
        raise Interrupted("SYNTHETIC completed wrong-source prediction; now exercise resume verification")
    with pytest.raises(Interrupted): run(prediction_producer=different_source)
    monkeypatch.setattr(EvaluationLabels, "labels", lambda *a, **k: pytest.fail("wrong source must stop before query label release"))
    with pytest.raises(development.DevelopmentGateError, match="dependency_mismatch"):
        run()
    assert len(calls) == 1
    assert not (native.root / "runs/registered_evaluation_fixture/receipt.json").exists()
