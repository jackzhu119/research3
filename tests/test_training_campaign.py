"""Campaign metadata/control-plane tests: no real fit, CUDA, network, or EEG."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from p3work import training_campaign as campaign
from p3work import method_registry as registry
from p3work.contracts import digest
from p3work.meta_training import MetaJob

REPO = Path(__file__).resolve().parents[1]


def put(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return campaign.descriptor(root, path)


@pytest.fixture
def metadata_root(tmp_path):
    """Explicit control-plane fixture, never accepted as scientific evidence."""
    for relative in ("contracts/input_contract.json", *campaign.CODE):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    cache = put(tmp_path, "cache/source.json", {"scope": "SYNTHETIC_METADATA_FIXTURE_ONLY"})
    put(tmp_path, "status/tasks.json", {"W1": {"status": "COMPLETE"},
        "W2": {"status": "COMPLETE", "cache_manifest": cache},
        "W3": {"status": "IN_PROGRESS_LOCAL_CPU"}})
    return tmp_path


def test_readiness_is_read_only_and_retains_all_registered_work(metadata_root, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("planning must not inspect CUDA"))
    before = sorted(str(p.relative_to(metadata_root)) for p in metadata_root.rglob("*") if p.is_file())
    plan = campaign.compile_campaign(metadata_root)
    after = sorted(str(p.relative_to(metadata_root)) for p in metadata_root.rglob("*") if p.is_file())
    assert before == after
    assert plan["minimum_gaps"] == ["W3_COMPLETE"]
    assert plan["planned_counts"] == {"source_engineering_updates": 200,
        "neural_source_fits": 15, "classical_core_fits": 10, "classical_additional_bank_fits": 6,
        "meta_engineering_jobs": 5, "meta_engineering_outer_updates_per_job": 100,
        "meta_screening_fits": 45, "meta_confirmation_additional_fits": 10,
        "development_calibration_profile_settings": 118, "development_screening_jobs": 118,
        "development_confirmation_additional_jobs": 28, "one_time_internal_checks": 1}
    assert sum(map(len, plan["registered_grids"].values())) == 118
    assert plan["actual_artifact_inventory"]["learned_artifacts"] == []
    assert plan["actual_artifact_inventory"]["seeds"] == {}
    assert plan["identity"]["external_query_permission"] is False
    assert plan["identity"]["source_extensions_enabled"] is False


def test_no_execution_without_attestation_or_complete_data(metadata_root, monkeypatch):
    with pytest.raises(campaign.CampaignError, match="attestation"):
        campaign.execute_campaign(metadata_root)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("data gate must precede CUDA"))
    result = campaign.execute_campaign(metadata_root, user_started_gpu=True)
    assert result["status"] == "PENDING_DATA_GATES"
    assert not (metadata_root / campaign.CLAIM).exists()


def test_single_global_claim_and_nonblocking_lock(metadata_root):
    identity = campaign.scope_identity(metadata_root)
    with campaign.campaign_lock(metadata_root, identity):
        with pytest.raises(campaign.CampaignError, match="global_lock"):
            with campaign.campaign_lock(metadata_root, identity):
                pass
    # Same claim can resume; a new output identity cannot dodge it.
    with campaign.campaign_lock(metadata_root, identity):
        pass
    with pytest.raises(campaign.CampaignError, match="immutable"):
        with campaign.campaign_lock(metadata_root, {**identity, "seeds": [1]}):
            pass


@pytest.mark.parametrize("relative", ["../outside.json", "/tmp/outside.json",
    "legacy_baseline_v1/model.json", "cache/old.npz", "cache/model.pkl"])
def test_path_guard(metadata_root, relative):
    with pytest.raises(campaign.CampaignError):
        campaign.local(metadata_root, relative, exists=False)


def test_symlink_guard_even_inside_root(metadata_root):
    target = metadata_root / "target.json"
    target.write_text("{}")
    (metadata_root / "alias.json").symlink_to(target)
    with pytest.raises(campaign.CampaignError, match="symlink"):
        campaign.local(metadata_root, "alias.json")


def test_actual_SHA_closure_no_status_only_trust(metadata_root):
    ref = put(metadata_root, "runs/example.json", {"status": "SYNTHETIC_NOT_A_FIT"})
    assert campaign.checked(metadata_root, ref).is_file()
    (metadata_root / ref["path"]).write_text("changed")
    with pytest.raises(campaign.CampaignError, match="SHA"):
        campaign.checked(metadata_root, ref)
    with pytest.raises(campaign.CampaignError, match="SHA_bound"):
        campaign.checked(metadata_root, {"path": "runs/example.json", "other": "value"})


def test_next_stage_never_creates_future_receipts(metadata_root):
    action = campaign.next_action(metadata_root)
    assert action["kind"] == "pilot"
    assert not (metadata_root / action["expected_receipt"]).exists()
    # These are control-plane selector fixtures only, not production inputs.
    put(metadata_root, action["expected_receipt"], {"status": "ENGINEERING_PILOT_COMPLETE", "synthetic": True})
    assert campaign.next_action(metadata_root)["kind"] == "source"
    put(metadata_root, campaign.SOURCE+"/source_profile_development.json",
        {"status": "SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING", "synthetic": True})
    assert campaign.next_action(metadata_root)["kind"] == "classical"


def test_partial_failed_meta_is_not_registered_as_completed(metadata_root):
    job = campaign.meta_job("P", registry.registered_grid("P")[0])
    put(metadata_root, campaign.meta_receipt(job), {"status": "INTERRUPTED_NOT_COMPLETE",
        "job": asdict(job), "synthetic": False})
    assert campaign.inventory(metadata_root)["learned_artifacts"] == []


def test_measured_cost_formulas_and_rejection_of_unmeasured():
    job = MetaJob("B8", 8, .35)
    proof = {"status": "META_ENGINEERING_COMPLETE", "synthetic": False,
        "job": {"method": "B8"}, "outer_updates_completed": 100,
        "validation_checkpoints": 2, "active_wall_seconds": 17.25}
    assert campaign.measured_meta_budget(proof, job)["derived_wall_seconds"] == 810
    with pytest.raises(campaign.CampaignError, match="actual_same_method"):
        campaign.measured_meta_budget({**proof, "synthetic": True}, job)
    with pytest.raises(campaign.CampaignError, match="positive_measured"):
        campaign.measured_meta_budget({**proof, "active_wall_seconds": float("nan")}, job)
    value = campaign.measured_evaluation_budget(1.25, 105)
    assert value["max_calibration_seconds"] == 10
    assert value["derived_wall_seconds"] == 1695
    for maximum, count in ((0, 1), (float("inf"), 1), (1, 0), (1, True)):
        with pytest.raises(campaign.CampaignError):
            campaign.measured_evaluation_budget(maximum, count)


def test_budget_is_immutable_and_global_amendment_cannot_increase_child(metadata_root):
    evidence = put(metadata_root, "runs/profile.json", {"scope": "SYNTHETIC_COST_FIXTURE"})
    original = campaign._budget(metadata_root, "test", {"stage": "fixture", "max_wall_seconds": 20,
        "max_jobs": 1}, evidence=[evidence], campaign_limit=100)
    reused = campaign._budget(metadata_root, "test", {"stage": "fixture", "max_wall_seconds": 50,
        "max_jobs": 1}, evidence=[evidence], campaign_limit=200)
    assert reused == original
    assert campaign.read(campaign.checked(metadata_root, reused))["max_wall_seconds"] == 20
    with pytest.raises(campaign.CampaignError, match="persisted_job_budget"):
        campaign._budget(metadata_root, "test", {"stage": "fixture", "max_wall_seconds": 20,
            "max_jobs": 2}, evidence=[evidence], campaign_limit=100)


@pytest.mark.parametrize("method", campaign.LEARNED)
def test_meta_and_development_sorting_equivalent_for_same_real_scores(method):
    """Registered grid ordering is dimension/rank then radius for both APIs."""
    grid = registry.registered_grid(method)
    rows = [{"index": i, "config": config, "ba": .7 + (i % 3)*.01,
             "bce": .4 - (i % 2)*.01, "outer": 50 + (i % 2)*50}
            for i, config in enumerate(grid)]
    def meta_key(row):
        config = row["config"]
        return (-row["ba"], row["bce"], row["outer"],
                config.get("dimension", config.get("rank")), config["radius"])
    def dev_key(row):
        return (-row["ba"], row["bce"], row["outer"], row["index"])
    assert sorted(rows, key=meta_key) == sorted(rows, key=dev_key)
    # Exhaust every pair of exact scientific-score ties, including every grid
    # boundary. This tests tie order, not a hand-coded unique winner.
    tied = [{**r, "ba": .7, "bce": .4, "outer": 100} for r in rows]
    assert sorted(tied, key=meta_key) == sorted(tied, key=dev_key)
    best = min(rows, key=meta_key)["config"]
    meta = {"dimension": best.get("dimension", best.get("rank")), "radius": best["radius"]}
    assert campaign.selected_meta_matches_development(meta, {"configuration": best})
    assert not campaign.selected_meta_matches_development({**meta, "radius": .111}, {"configuration": best})


def test_internal_budget_reads_public_registry_only(metadata_root):
    # There are no event/EEG files in this fixture; only frozen registry rows.
    manifest = put(metadata_root, "cache/stieger.json", {"sessions": [
        {"partition": "internal_holdout", "events": {"path": "FORBIDDEN_QUERY.json"}},
        {"partition": "meta_validation"}, {"partition": "internal_holdout"}]})
    tasks = campaign.read(metadata_root / "status/tasks.json")
    tasks["W3"] = {"status": "COMPLETE", "cache_manifest": manifest}
    put(metadata_root, "status/tasks.json", tasks)
    assert campaign.internal_session_upper_bound(metadata_root) == 2


def test_backup_manifest_excludes_config_credentials_and_reports_no_upload(metadata_root):
    put(metadata_root, campaign.OUTPUT+"/result.json", {"scope": "SYNTHETIC_RESULT"})
    put(metadata_root, "configs/private.json", {"fixture": True})
    (metadata_root / campaign.OUTPUT / "secret.env").write_text("fixture")
    ref = campaign.backup_manifest(metadata_root)
    value = campaign.read(campaign.checked(metadata_root, ref))
    assert [row["path"] for row in value["files"]] == [campaign.OUTPUT+"/result.json"]
    assert value["upload_performed"] is False
    assert value["full_GET_verified"] is False


def test_deadline_refuses_dispatch_without_remaining_authorization(metadata_root, monkeypatch):
    monkeypatch.setattr(campaign.multiprocessing, "get_context", lambda *_: pytest.fail("no process permitted"))
    assert campaign.dispatch_bounded(metadata_root, 100, {"kind": "pilot"}, remaining_seconds=5) is None


@pytest.mark.parametrize("kind", ["source", "classical", "meta_engineering", "development", "calibration_profile"])
def test_prior_dispatched_child_output_cannot_be_recreated(metadata_root, kind):
    action = {"kind": kind, "expected_receipt": campaign.OUTPUT+"/fixture/receipt.json"}
    if kind == "meta_engineering":
        action["job"] = asdict(MetaJob("P", 8, .35, phase="engineering"))
    with pytest.raises(campaign.CampaignError, match="dispatched_child_output_missing"):
        campaign.guard_child_recovery(metadata_root, action, prior_dispatch=True)


@pytest.mark.parametrize("missing", ["training_plan.json", "operational_budget.json", "identity.json", "execution_progress.json"])
def test_source_missing_ledger_or_identity_blocks_before_kernel(metadata_root, missing):
    put(metadata_root, campaign.SOURCE+"/training_plan.json", {})
    put(metadata_root, campaign.SOURCE+"/operational_budget.json", {})
    folder = campaign.SOURCE+"/lodo/DATA/eegnet/seed-20261018"
    put(metadata_root, folder+"/identity.json", {})
    put(metadata_root, folder+"/execution_progress.json", {"observed_optimizer_calls_total": 1})
    put(metadata_root, folder+"/status.json", {"status": "RUNNING", "training_started": True})
    (metadata_root / folder / "last_state.pt").write_bytes(b"synthetic_not_a_model")
    target = campaign.SOURCE+"/"+missing if missing in {"training_plan.json", "operational_budget.json"} else folder+"/"+missing
    (metadata_root / target).unlink()
    with pytest.raises(campaign.CampaignError, match="missing_source"):
        campaign.guard_child_recovery(metadata_root, {"kind": "source"})


def test_unknown_classical_numerical_fit_cannot_be_dispatched_again(metadata_root):
    budget = put(metadata_root, "runs/budget.json", {"scope": "SYNTHETIC_BUDGET_FIXTURE"})
    put(metadata_root, campaign.CLASSICAL+"/identity.json", {"budget": budget})
    put(metadata_root, campaign.CLASSICAL+"/progress.json", {"actual_fit_attempts_started": 1})
    put(metadata_root, campaign.CLASSICAL+"/plan.json", {})
    put(metadata_root, campaign.CLASSICAL+"/bank/ALL_SOURCE/B4/m-1/status.json", {"status": "FIT_RUNNING"})
    with pytest.raises(campaign.CampaignError, match="refit_forbidden"):
        campaign.guard_child_recovery(metadata_root, {"kind": "classical"})


@pytest.mark.parametrize("missing", ["identity.json", "latest_checkpoint.json", "progress.json", "wall_budget.json"])
def test_meta_missing_recovery_evidence_is_not_reconstructed(metadata_root, missing):
    job = MetaJob("B8", 8, .35)
    folder = str(Path(campaign.meta_receipt(job)).parent)
    for filename in ("identity.json", "latest_checkpoint.json", "progress.json", "wall_budget.json", "receipt.json"):
        put(metadata_root, folder+"/"+filename, {"scope": "SYNTHETIC_LEDGER_FIXTURE"})
    (metadata_root / folder / missing).unlink()
    with pytest.raises(campaign.CampaignError, match="missing_meta"):
        campaign.guard_child_recovery(metadata_root, {"kind": "meta_screening", "job": asdict(job)})


def test_recorded_code_must_match_executed_campaign(metadata_root):
    (metadata_root / campaign.CODE[0]).write_text("# different package")
    with pytest.raises(campaign.CampaignError, match="executed_package"):
        campaign.compile_campaign(metadata_root)


def test_claimed_campaign_missing_global_ledger_blocks_before_any_dispatch(metadata_root, monkeypatch):
    import torch
    tasks = campaign.read(metadata_root / "status/tasks.json")
    identity = campaign.scope_identity(metadata_root)
    split = put(metadata_root, "audit/split.json", {"split_sha256": identity["split_sha256"], "partitions": {
        "meta_train": list(range(40)), "meta_validation": list(range(40, 51)), "internal_holdout": list(range(51, 62))}})
    manifest = put(metadata_root, "cache/stieger.json", {"schema_version": "paper3-stieger-cache-v2",
        "status": "STIEGER_CACHE_COMPLETE", "synthetic": False, "scope": "SYNTHETIC_METADATA_FIXTURE_ONLY",
        "participant_split": split, "session_count": 598, "participant_count": 62,
        "split_sha256": identity["split_sha256"], "input_contract_sha256": identity["input_contract_sha256"]})
    tasks["W3"] = {"status": "COMPLETE", "cache_manifest": manifest}
    put(metadata_root, "status/tasks.json", tasks)
    identity = campaign.scope_identity(metadata_root)
    with campaign.campaign_lock(metadata_root, identity):
        pass
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(campaign, "dispatch_bounded", lambda *a, **k: pytest.fail("dispatch after missing ledger"))
    with pytest.raises(campaign.CampaignError, match="missing_progress_or_wall_ledger"):
        campaign.execute_campaign(metadata_root, user_started_gpu=True)


def test_status_only_W3_is_not_training_readiness(metadata_root):
    tasks = campaign.read(metadata_root / "status/tasks.json")
    tasks["W3"] = {"status": "COMPLETE"}
    put(metadata_root, "status/tasks.json", tasks)
    assert campaign.compile_campaign(metadata_root)["minimum_gaps"] == ["W3_actual_cache_manifest"]


def test_real_spawn_transport_refuses_unregistered_action_without_any_fit(metadata_root):
    # Reject the action before the real spawn transport, so interpreter/import
    # overhead cannot swallow this programming error as an ordinary wall pause.
    with pytest.raises(campaign.CampaignError, match="unregistered_campaign_action"):
        campaign.dispatch_bounded(metadata_root, 30, {"kind": "UNREGISTERED_SOFTWARE_FIXTURE"}, remaining_seconds=30)
    assert not (metadata_root / "runs").exists()


def wall_report(identity, seconds):
    return {"identity_sha256": identity, "wall_seconds": seconds,
        "minimum_charged_seconds": seconds, "checkpoint_wall_high_water_seconds": seconds,
        "blocked": False, "clock_block": None}


def test_restored_older_ledger_cannot_erase_state_or_snapshot_high_water(metadata_root):
    identity = "a"*64
    state = {"schema_version": campaign.SCHEMA, "identity_sha256": identity,
             "wall_budget": wall_report(identity, 40)}
    newer = {**state, "wall_budget": wall_report(identity, 80)}
    campaign.immutable(metadata_root, campaign.OUTPUT+"/progress_snapshots/"+digest(newer)+".json", newer)
    floor = campaign.published_wall_floor(metadata_root, state, identity)
    assert floor == 80  # A newer immutable snapshot also survives old state.
    path = metadata_root / campaign.OUTPUT / "wall_budget.json"
    original = campaign.PersistentWallBudget(path, identity, 100)
    original.close()  # Internally valid old ledger, zero recorded elapsed time.
    restored = campaign.PersistentWallBudget(path, identity, 100, minimum_charged_seconds=floor)
    try:
        assert restored.report()["wall_seconds"] >= 80
        assert restored.remaining_seconds <= 20
    finally:
        restored.close()


def test_snapshot_tamper_and_published_clock_block_cannot_be_rolled_back(metadata_root):
    identity = "b"*64
    state = {"schema_version": campaign.SCHEMA, "identity_sha256": identity,
             "wall_budget": wall_report(identity, 20)}
    blocked = {**state, "wall_budget": {**state["wall_budget"], "blocked": True}}
    target = campaign.OUTPUT+"/progress_snapshots/"+digest(blocked)+".json"
    campaign.immutable(metadata_root, target, blocked)
    with pytest.raises(campaign.CampaignError, match="published_clock_block"):
        campaign.published_wall_floor(metadata_root, state, identity)
    (metadata_root / target).write_text(json.dumps(state))
    with pytest.raises(campaign.CampaignError, match="content_address_mismatch"):
        campaign.published_wall_floor(metadata_root, state, identity)


@pytest.mark.parametrize("seconds", [float("nan"), -1, True])
def test_invalid_published_high_water_never_becomes_free_time(metadata_root, seconds):
    identity = "c"*64
    state = {"schema_version": campaign.SCHEMA, "identity_sha256": identity,
             "wall_budget": wall_report(identity, seconds)}
    with pytest.raises(campaign.CampaignError, match="invalid_published_wall_high_water"):
        campaign.published_wall_floor(metadata_root, state, identity)


def test_execute_passes_snapshot_high_water_to_original_budget_constructor(metadata_root, monkeypatch):
    """Stop at budget construction: no child, source loader, GPU, or fit runs."""
    import torch
    identity = campaign.scope_identity(metadata_root)
    split = put(metadata_root, "audit/split.json", {"split_sha256": identity["split_sha256"], "partitions": {
        "meta_train": list(range(40)), "meta_validation": list(range(40, 51)), "internal_holdout": list(range(51, 62))}})
    manifest = put(metadata_root, "cache/stieger.json", {"schema_version": "paper3-stieger-cache-v2",
        "status": "STIEGER_CACHE_COMPLETE", "synthetic": False, "scope": "SYNTHETIC_METADATA_FIXTURE_ONLY",
        "participant_split": split, "session_count": 598, "participant_count": 62,
        "split_sha256": identity["split_sha256"], "input_contract_sha256": identity["input_contract_sha256"]})
    tasks = campaign.read(metadata_root / "status/tasks.json")
    tasks["W3"] = {"status": "COMPLETE", "cache_manifest": manifest}
    put(metadata_root, "status/tasks.json", tasks)
    identity = campaign.scope_identity(metadata_root)
    identity_sha = digest(identity)
    with campaign.campaign_lock(metadata_root, identity):
        pass
    state = {"schema_version": campaign.SCHEMA, "identity_sha256": identity_sha,
             "actions": [], "active": None, "wall_budget": wall_report(identity_sha, 40)}
    put(metadata_root, campaign.OUTPUT+"/campaign_progress.json", state)
    newer = {**state, "wall_budget": wall_report(identity_sha, 75)}
    campaign.immutable(metadata_root, campaign.OUTPUT+"/progress_snapshots/"+digest(newer)+".json", newer)
    original = campaign.PersistentWallBudget(metadata_root / campaign.OUTPUT / "wall_budget.json", identity_sha, 100)
    original.close()  # Restore an older, self-consistent ledger (zero seconds).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    class StopAtConstructor(RuntimeError):
        pass
    calls = []
    def constructor(*args, **kwargs):
        calls.append(kwargs)
        raise StopAtConstructor
    monkeypatch.setattr(campaign, "PersistentWallBudget", constructor)
    monkeypatch.setattr(campaign, "dispatch_bounded", lambda *a, **k: pytest.fail("no dispatch permitted"))
    with pytest.raises(StopAtConstructor):
        campaign.execute_campaign(metadata_root, user_started_gpu=True, max_wall_seconds=100)
    assert calls[0]["minimum_charged_seconds"] == 75


def fake_transport(monkeypatch, *, messages, start_overhead=0, ignore_term=False, on_poll=None, on_recv=None):
    """Own-process control-plane double, with no OS process or production fit."""
    clock = {"now": 0.}
    monkeypatch.setattr(campaign.time, "monotonic", lambda: clock["now"])
    class Process:
        pid = 987654321
        def __init__(self):
            self.alive, self.terms, self.kills = False, 0, 0
        def start(self):
            self.alive = True
            clock["now"] += start_overhead
        def is_alive(self):
            return self.alive
        def join(self, seconds):
            clock["now"] += seconds
        def terminate(self):
            self.terms += 1
            if not ignore_term:
                self.alive = False
        def kill(self):
            self.kills += 1
            self.alive = False
    worker = Process()
    class Connection:
        def __init__(self):
            self.messages, self.sent = list(messages), []
        def poll(self, seconds):
            if on_poll:
                on_poll(self, worker, clock)
            if not self.messages:
                clock["now"] += seconds
            return bool(self.messages)
        def recv(self):
            message = self.messages.pop(0)
            if on_recv:
                on_recv(message, clock)
            return message
        def send(self, message):
            self.sent.append(message)
        def close(self):
            pass
    parent, child = Connection(), Connection()
    class Context:
        def Pipe(self, *, duplex):
            assert duplex is True
            return parent, child
        def Process(self, **kwargs):
            return worker
    monkeypatch.setattr(campaign.multiprocessing, "get_context", lambda kind: Context())
    return worker, parent, clock


def group_ready():
    return {"type": "group_ready", "pid": 987654321, "pgid": 987654321}


def test_spawn_import_timeout_uses_owned_process_before_any_setsid(metadata_root, monkeypatch):
    worker, parent, _ = fake_transport(monkeypatch, messages=[], start_overhead=30)
    monkeypatch.setattr(campaign.os, "killpg", lambda *a: pytest.fail("no PGID authorized before handshake"))
    monkeypatch.setattr(campaign, "live_group_members", lambda *a: pytest.fail("no group exists"))
    assert campaign.dispatch_bounded(metadata_root, 30, {"kind": "pilot"}, remaining_seconds=30) is None
    assert not worker.alive and worker.terms == 1 and parent.sent == []


def test_missing_PGID_still_terminates_owned_handle_and_ignored_TERM_escalates(metadata_root, monkeypatch):
    def expired_after_ack(connection, worker, clock):
        if connection.sent:
            clock["now"] = max(clock["now"], 31.)
    worker, parent, _ = fake_transport(monkeypatch, messages=[group_ready()],
        ignore_term=True, on_poll=expired_after_ack)
    calls = []
    def missing_group(*args):
        calls.append(args)
        raise ProcessLookupError
    monkeypatch.setattr(campaign.os, "killpg", missing_group)
    monkeypatch.setattr(campaign, "live_group_members", lambda pgid: ())
    assert campaign.dispatch_bounded(metadata_root, 30, {"kind": "pilot"}, remaining_seconds=30) is None
    assert calls and worker.terms > 0 and worker.kills == 1 and not worker.alive
    assert parent.sent == ["dispatch_owned_stage"]


def test_dead_leader_does_not_leave_its_descendants_or_release_early(metadata_root, monkeypatch):
    def dead_after_ack(connection, worker, clock):
        if connection.sent:
            worker.alive = False
    worker, _, _ = fake_transport(monkeypatch, messages=[group_ready()], on_poll=dead_after_ack)
    calls = []
    monkeypatch.setattr(campaign.os, "killpg", lambda *a: calls.append(a))
    # The first cleanup signal has not yet stopped the descendant. The function
    # must signal and confirm again despite the leader already being dead.
    monkeypatch.setattr(campaign, "live_group_members", lambda pgid: (777,) if len(calls) < 2 else ())
    with pytest.raises(campaign.CampaignError, match="exited_without_receipt"):
        campaign.dispatch_bounded(metadata_root, 30, {"kind": "pilot"}, remaining_seconds=30)
    assert len(calls) >= 2 and worker.terms == 0 and worker.kills == 0


def test_late_complete_is_rejected_and_success_also_cleans_group(metadata_root, monkeypatch):
    complete = {"type": "result", "ok": True, "receipt": {"path": "SYNTHETIC_NO_FIT.json", "sha256": "d"*64}}
    def late(message, clock):
        if message.get("type") == "result":
            clock["now"] += 30
    worker, _, _ = fake_transport(monkeypatch, messages=[group_ready(), complete], on_recv=late)
    calls = []
    monkeypatch.setattr(campaign.os, "killpg", lambda *a: calls.append(a))
    monkeypatch.setattr(campaign, "live_group_members", lambda pgid: ())
    assert campaign.dispatch_bounded(metadata_root, 30, {"kind": "pilot"}, remaining_seconds=30) is None
    assert calls and not worker.alive
    worker, _, _ = fake_transport(monkeypatch, messages=[group_ready(), complete])
    assert campaign.dispatch_bounded(metadata_root, 30, {"kind": "pilot"}, remaining_seconds=30) == complete["receipt"]
    assert len(calls) >= 2 and not worker.alive


@pytest.mark.parametrize("owner_lost", [True, False])
def test_watchdog_closes_group_on_owner_loss_or_original_absolute_deadline(monkeypatch, owner_lost):
    clock = {"now": 0.}
    class Stop:
        def is_set(self):
            return False
        def wait(self, seconds):
            clock["now"] += seconds
    monkeypatch.setattr(campaign.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(campaign.os, "getppid", lambda: 999 if owner_lost else 123)
    signals = []
    monkeypatch.setattr(campaign.os, "killpg", lambda *args: signals.append(args))
    campaign.watch_owned_stage(Stop(), 123, 456, .25)
    assert signals == [(456, campaign.signal.SIGKILL)]
    assert clock["now"] == 0 if owner_lost else clock["now"] >= .25


def test_cancelled_watchdog_and_campaign_main_group_cannot_send_signals(monkeypatch):
    stop = campaign.threading.Event()
    stop.set()
    monkeypatch.setattr(campaign.os, "killpg", lambda *a: pytest.fail("cancelled watchdog signalled"))
    campaign.watch_owned_stage(stop, 123, 456, 0.)
    monkeypatch.setattr(campaign.os, "getpid", lambda: 123)
    monkeypatch.setattr(campaign.os, "getpgrp", lambda: 456)
    with pytest.raises(campaign.CampaignError, match="independent_owned_child"):
        campaign.start_stage_watchdog(999, 456, 100.)


def test_group_ownership_is_persisted_before_any_dispatch_ACK(metadata_root, monkeypatch):
    complete = {"type": "result", "ok": True, "receipt": None}
    worker, parent, _ = fake_transport(monkeypatch, messages=[group_ready(), complete])
    monkeypatch.setattr(campaign.os, "killpg", lambda *a: None)
    monkeypatch.setattr(campaign, "live_group_members", lambda pgid: ())
    publications = []
    def ready(pid, pgid, deadline):
        assert parent.sent == []
        publications.append((pid, pgid, deadline))
    stopped = []
    campaign.dispatch_bounded(metadata_root, 30, {"kind": "pilot"}, remaining_seconds=30,
        on_group_ready=ready, on_group_stopped=lambda: stopped.append(True))
    assert publications == [(worker.pid, worker.pid, 25.)]
    assert stopped == [True]


def test_unknown_or_still_live_previous_owner_group_cannot_authorize_a_new_job(monkeypatch):
    identity = "d"*64
    state = {"active": {"key": "fixture"}, "active_process": None}
    with pytest.raises(campaign.CampaignError, match="unknown_previous_stage"):
        campaign.guard_previous_stage_group(state, identity)
    state["active_process"] = {"schema_version": "paper3-campaign-owned-stage-group-v1",
        "campaign_identity_sha256": identity, "action_key": "fixture", "pid": 456, "pgid": 456,
        "owner_pid": 123, "pid_start_time_ticks": 1, "absolute_deadline_monotonic": 100.,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "cleanup_verified": False}
    monkeypatch.setattr(campaign, "live_group_members", lambda pgid: (456, 457))
    with pytest.raises(campaign.CampaignError, match="still_alive"):
        campaign.guard_previous_stage_group(state, identity)
    monkeypatch.setattr(campaign, "live_group_members", lambda pgid: ())
    campaign.guard_previous_stage_group(state, identity)  # Same owned group is now verified stopped.
