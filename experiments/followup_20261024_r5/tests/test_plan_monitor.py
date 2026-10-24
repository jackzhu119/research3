"""Frozen matrix and monitor contracts, using synthetic files and fake APIs."""
import base64
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import urllib.error

import pytest

from paper3r5 import monitor, plan as scientific_plan, report
from paper3r5.io import put, read


def test_internal_plan_has_105_tasks_and_a_closed_topological_dependency_dag(tmp_path):
    plan = scientific_plan.make_plan(tmp_path / "project", tmp_path / "run")
    assert len(plan["tasks"]) == 105
    assert Counter(task["family"] for task in plan["tasks"]) == {
        "basis": 12, "core": 48, "sensitivity": 39, "reference": 6}
    assert plan["counts"] == {"basis": 12, "core": 48, "sensitivity": 39, "reference": 6}
    assert plan["seeds"] == [20261018, 20261019, 20261020]
    assert set(plan["budgets"]) == {4, 8, 12, 16, 20}
    assert len(plan["comparisons"]) == len(set(map(tuple, plan["comparisons"]))) == 8
    assert plan["external_data_required_for_core"] is False
    assert all("external_manifest" not in task for task in plan["tasks"])
    assert "already used" in plan["interpretation"]
    by_key, visited = {task["key"]: task for task in plan["tasks"]}, set()
    assert len(by_key) == len(plan["tasks"])
    for task in plan["tasks"]:
        # Producers precede consumers; this also proves a cycle-free closed DAG.
        assert set(task["requires"]) <= visited
        for dependency in task["requires"]:
            producer = by_key[dependency]
            assert producer["family"] == "basis"
            assert producer["seed"] == task["seed"]
            assert producer["config"]["rank"] == task["config"]["rank"]
        if task["family"] in {"core", "sensitivity"}:
            assert task["requires"] == [task["basis_key"]]
        else:
            assert task["requires"] == []
        visited.add(task["key"])
    bases = [task for task in plan["tasks"] if task["family"] == "basis"]
    assert Counter(task["config"]["rank"] for task in bases) == {4: 12}
    assert all(task["updates"] == 1000 for task in bases)
    assert plan["episodes_per_update"] == 4 and plan["validate_every"] == 50


def test_geometry_comparisons_match_parameter_budget_and_probes_are_separate(tmp_path):
    plan=scientific_plan.make_plan(tmp_path,tmp_path/'run')
    by_key={t['key']:t for t in plan['tasks']}
    groups=defaultdict(dict)
    for task in plan['tasks']:
        if task['family']=='core':
            assert task['seed'] not in groups[task['group']]
            groups[task['group']][task['seed']]=task
    for candidate,baseline in plan['comparisons']:
        assert set(groups[candidate])==set(groups[baseline])==set(plan['seeds'])
        for seed in plan['seeds']:
            a,b=groups[candidate][seed],groups[baseline][seed]
            for key in ('variant','head_l2','theta_l2','rank','radius','damping','inner_steps','initializer'):
                assert a['config'][key]==b['config'][key]
            assert a['config']['variant']=='geometry_refined'
            assert a['config']['inner_steps']==5 and a['config']['head_l2']==.1
            assert by_key[a['basis_key']]['seed']==by_key[b['basis_key']]['seed']==seed
    probes=[t for t in plan['tasks'] if t.get('mechanism_probe')]
    assert len(probes)==15 and all(t['config']['variant']=='geometry_tangent' for t in probes)
    assert all(t['group'] not in pair for t in probes for pair in plan['comparisons'])
    assert {t['config']['geometry'] for t in probes}==set(scientific_plan.PROBE_CHARTS)
    assert sum(t['family']=='sensitivity' and not t.get('mechanism_probe') for t in plan['tasks'])==24
    assert sum(len(t.get('scenarios',[]))-1 for t in plan['tasks'] if len(t.get('scenarios',[]))>1)==72
    assert plan['task_breakdown']=={'basis':12,'core':48,'radius_sensitivity':24,'shared_theta_probe':15,'reference':6}


def test_external_matrix_is_separate_21_task_descriptive_plan_and_binds_inputs(tmp_path):
    basis = tmp_path / "core-run"
    core = scientific_plan.make_plan(tmp_path, basis)
    put(basis / "plan.json", core)
    manifest = tmp_path / "external.json"
    put(manifest, {"fixture": "synthetic-published-order-manifest"})
    old_bytes = (basis / "plan.json").read_bytes()
    external = scientific_plan.make_external_plan(tmp_path, tmp_path / "external-run", manifest, basis)
    assert len(external["tasks"]) == 21
    assert {task["family"] for task in external["tasks"]} == {"transfer"}
    assert Counter(task['group'] for task in external['tasks'])=={
        'single:single':3,'single:left_rotation':3,'single:right_rotation':3,'single:symmetric':3,
        'left_rotation:left_rotation':3,'right_rotation:right_rotation':3,'symmetric:symmetric':3}
    assert external["comparisons"] == []
    assert external["basis_recipe_sha256"] == core["recipe_sha256"]
    assert external["external_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert external["output"] != core["output"]
    assert (basis / "plan.json").read_bytes() == old_bytes
    assert "physical acquisition clock unverified" in external["interpretation"]
    for task in external["tasks"]:
        assert task["requires"] == []
        assert task["basis_output"] == str(basis.resolve())
        assert task["external_manifest"] == str(manifest.resolve())
        assert task["partition"] == "weibo_published_order"
        assert task["scenarios"] == ["clean"]
        assert set(task["budgets"]) == {4, 8, 12, 16, 20}
        assert task["config"]["head_l2"] == .1
        assert task["basis_key"] in {candidate["key"] for candidate in core["tasks"]
                                      if candidate["family"] == "basis"}


def test_aggregate_only_removes_nested_individual_arrays_and_preserves_aggregate_counts():
    raw = {"n_participants": 11, "participant_mean_ba": .7, "sessions": 22,
           "participants": [{"participant_id": "S1", "ba": .8}],
           "participant_ids": ["S1"], "session_ids": ["session1"],
           "expected_sessions": [{"participant_id": "S1", "session_id": "session1"}],
           "tables": [{"delta_ba_pp": 2., "ci95_pp": [-1., 5.], "p_holm": .1,
                       "participant_deltas_ba": {"S1": .2}, "participant_seed_counts": {"S1": 3},
                       "nested": {"participant_id": "S1", "session_id": "session1", "labels": [0, 1],
                                  "event_uids": ["S1-1"], "probabilities": [[.2, .8]],
                                  "predictions": [1], "f0_predictions": [0],
                                  "f0_probabilities": [[.8, .2]], "n_participants": 10}}]}
    before = copy.deepcopy(raw)
    cleaned = monitor.aggregate_only(raw)
    assert cleaned == {"n_participants": 11, "participant_mean_ba": .7, "sessions": 22,
                       "tables": [{"delta_ba_pp": 2., "ci95_pp": [-1., 5.], "p_holm": .1,
                                   "nested": {"n_participants": 10}}]}
    assert raw == before  # Sanitizing a publication cannot mutate local evidence.


def monitoring_plan(tmp_path):
    return {"output": str(tmp_path), "recipe_sha256": "synthetic-monitor-recipe",
            "interpretation": "exploratory internal replication",
            "tasks": [{"key": "basis", "family": "basis", "seed": 20261018},
                      {"key": "evaluate", "family": "core", "seed": 20261018},
                      {"key": "completed", "family": "core", "seed": 20261018}]}


def test_collect_matches_runtime_and_worker_schemas_and_complete_result_wins(tmp_path):
    plan = monitoring_plan(tmp_path)
    put(tmp_path / "runtime/state.json", {"controller_status": "WAITING_FOR_CAPACITY", "tasks": {
        "0": {"phase": "ACTIVE"}, "1": {"phase": "WAITING_INPUT"}, "2": {"phase": "ACTIVE"}}})
    launch = {"partition": "hpc_gpu1", "gres": "gpu:3", "cpus": 12, "memory_total": "164G", "slots": 3}
    put(tmp_path / "runtime/launch.json", launch)
    progress = {"task": "basis", "status": "TRAINING", "stage": "meta_train", "outer_update": 123,
                "total_updates": 1000, "updated_utc": "2026-10-24T12:00:00Z"}
    put(tmp_path / "basis/progress.json", progress)
    evaluation = {"status": "EVALUATING", "stage": "holdout", "session": 2, "sessions_total": 22,
                  "budget": 20, "scenario": "clean", "updated_utc": "2026-10-24T12:00:00Z",
                  "participant_id": "S1", "labels": [0, 1]}
    put(tmp_path / "evaluate/progress.json", evaluation)
    put(tmp_path / "completed/progress.json", {"status": "EVALUATING", "stage": "diagnostics"})
    put(tmp_path / "completed/result.json", {"status": "COMPLETE", "outer_updates": 0})
    snapshot = monitor.collect(plan)
    assert snapshot["controller_status"] == "WAITING_FOR_CAPACITY"
    assert snapshot["requested_resources"]["partition"] == "hpc_gpu1"
    assert snapshot["requested_resources"]["cpus_per_gpu"] == 12
    assert snapshot["requested_resources"]["cpus_per_job"] == 36
    assert snapshot["requested_resources"]["memory_per_job"] == "164G"
    assert snapshot["complete_tasks"] == 1 and snapshot["total_tasks"] == 3
    assert snapshot["scientific_completion_claimed"] is False
    basis, evaluation_row, completed = snapshot["tasks"]
    assert basis["status"] == "TRAINING" and basis["outer_update"] == 123 and basis["total_updates"] == 1000
    assert basis["stage"] == "meta_train" and basis["progress_age_seconds"] >= 0
    assert evaluation_row["session"] == 2 and evaluation_row["sessions_total"] == 22
    assert evaluation_row["budget"] == 20 and evaluation_row["scenario"] == "clean"
    assert "participant_id" not in evaluation_row and "labels" not in evaluation_row
    assert completed["status"] == "COMPLETE"


@pytest.mark.parametrize("failure_stage", ["ensure_branch", "publish"])
def test_github_failure_preserves_local_status_and_retries_without_stopping(tmp_path, monkeypatch, failure_stage):
    plan = monitoring_plan(tmp_path)
    plan["tasks"] = plan["tasks"][:1]
    put(tmp_path / "basis/result.json", {"status": "COMPLETE"})
    built = {"status": "COMPLETE", "primary_family_complete": True,
             "tables": {"primary_comparisons": [{"n_participants": 11, "delta_ba_pp": 2.,
                 "participant_deltas_ba": {"S1": .1}, "labels": [0, 1]}]}}
    calls, events = [], []
    monkeypatch.setattr(report, "build_report", lambda supplied: calls.append(supplied) or built)
    monkeypatch.setattr(monitor, "emit", lambda event, **values: events.append(event))
    class FailingOnce:
        def __init__(self):
            self.failed = False
            self.published = []
        def maybe_fail(self, stage):
            if failure_stage == stage and not self.failed:
                self.failed = True
                raise OSError("simulated GitHub outage")
        def ensure_branch(self):
            self.maybe_fail("ensure_branch")
        def publish(self, snapshot):
            self.maybe_fail("publish")
            self.published.append(snapshot)
    client, cache = FailingOnce(), {}
    returned, ready, finished = monitor.cycle(plan, None, client=client, cache=cache)
    assert returned is client and not finished
    local = read(tmp_path / "status.json")
    assert local["complete_tasks"] == 1 and local["report_status"] == "COMPLETE"
    assert local["tables"]["primary_comparisons"] == [{"n_participants": 11, "delta_ba_pp": 2.}]
    assert "GITHUB_SYNC_RETRY" in events
    _, ready, finished = monitor.cycle(plan, None, client=client, branch_ready=ready, cache=cache)
    assert ready and finished and len(client.published) == 1
    assert len(calls) == 1  # A GitHub retry does not rerun valid numerical reporting.


def test_error_report_is_retried_when_completed_task_count_is_unchanged(tmp_path, monkeypatch):
    plan = monitoring_plan(tmp_path)
    plan["tasks"] = []
    reports = iter([{"status": "ERROR", "tables": {}}, {"status": "COMPLETE", "tables": {}}])
    monkeypatch.setattr(report, "build_report", lambda supplied: next(reports))
    monkeypatch.setattr(monitor, "emit", lambda *args, **kwargs: None)
    class Client:
        def ensure_branch(self):
            pass
        def publish(self, snapshot):
            pass
    client, cache = Client(), {}
    _, ready, finished = monitor.cycle(plan, None, client=client, cache=cache)
    assert not finished and read(tmp_path / "status.json")["report_status"] == "ERROR"
    _, _, finished = monitor.cycle(plan, None, client=client, branch_ready=ready, cache=cache)
    assert finished and read(tmp_path / "status.json")["report_status"] == "COMPLETE"


def github_error(code):
    return urllib.error.HTTPError("https://api.github.com/synthetic", code, "simulated", None, None)


def github_client():
    client = monitor.GitHub.__new__(monitor.GitHub)
    client.config = {"status_path": monitor.STATUS_PATH}
    return client


@pytest.mark.parametrize("conflict", [409, 422])
def test_github_publish_conflict_refreshes_sha_and_retries_same_snapshot(monkeypatch, conflict):
    client = github_client()
    reads, writes = [], []
    def api(path, body=None, method=None):
        if method != "PUT":
            reads.append(path)
            return {"sha": "sha-" + str(len(reads))}
        writes.append((path, body))
        if len(writes) == 1:
            raise github_error(conflict)
        return {"content": {"sha": "new-sha"}}
    monkeypatch.setattr(client, "api", api)
    snapshot = {"complete_tasks": 3, "tables": {"primary": [{"n_participants": 11}]}}
    assert client.publish(snapshot) == {"content": {"sha": "new-sha"}}
    assert len(reads) == len(writes) == 2
    assert writes[0][1]["sha"] == "sha-1" and writes[1][1]["sha"] == "sha-2"
    assert all("repos/" + monitor.REPOSITORY + "/contents/" + monitor.STATUS_PATH in path
               for path, _ in writes)
    assert all(body["branch"] == monitor.BRANCH for _, body in writes)
    assert all(json.loads(base64.b64decode(body["content"])) == snapshot for _, body in writes)


def test_github_create_conflict_refetches_newly_created_file_sha(monkeypatch):
    client = github_client()
    reads, writes = [], []
    def api(path, body=None, method=None):
        if method != "PUT":
            reads.append(path)
            if len(reads) == 1:
                raise github_error(404)
            return {"sha": "created-by-another-monitor"}
        writes.append(body)
        if len(writes) == 1:
            raise github_error(422)
        return {"ok": True}
    monkeypatch.setattr(client, "api", api)
    assert client.publish({"complete_tasks": 0}) == {"ok": True}
    assert "sha" not in writes[0] and writes[1]["sha"] == "created-by-another-monitor"


@pytest.mark.parametrize("status,expected_attempts", [(409, 3), (422, 3), (403, 1)])
def test_github_publish_retry_budget_is_bounded_and_does_not_retry_authorization(monkeypatch, status, expected_attempts):
    client = github_client()
    writes = []
    def api(path, body=None, method=None):
        if method != "PUT":
            return {"sha": "current-sha"}
        writes.append(body)
        raise github_error(status)
    monkeypatch.setattr(client, "api", api)
    with pytest.raises(urllib.error.HTTPError) as error:
        client.publish({"complete_tasks": 0})
    assert error.value.code == status and len(writes) == expected_attempts


@pytest.mark.parametrize("external", ["version", "manifest", None])
def test_cycle_routes_weibo_to_separate_status_path_and_ignores_old_private_branch(tmp_path, monkeypatch, external):
    plan = monitoring_plan(tmp_path)
    plan["tasks"] = []
    plan["version"] = "paper3-round5-profiled-information-v1"
    if external == "version":
        plan["version"] = "paper3-round5-weibo-published-order-v1"
    if external == "manifest":
        plan["external_manifest"] = str(tmp_path / "weibo.json")
    credentials = tmp_path / "private-github.json"
    put(credentials, {"token": "synthetic", "repo": "old/repository", "branch": "old-private-branch",
                      "status_path": "monitoring/old/status.json"})
    monkeypatch.setattr(report, "build_report", lambda supplied: {"status": "COMPLETE", "tables": {}})
    monkeypatch.setattr(monitor, "emit", lambda *args, **kwargs: None)
    # Fake only the HTTP boundary, preserving real client config and routing.
    calls = []
    def api(self, path, body=None, method=None):
        calls.append((path, body, method))
        return {"sha": "synthetic-sha"}
    monkeypatch.setattr(monitor.GitHub, "api", api)
    client, ready, finished = monitor.cycle(plan, credentials)
    expected = monitor.WEIBO_STATUS_PATH if external else monitor.STATUS_PATH
    assert ready and finished
    assert client.config["status_path"] == expected
    assert client.config["repo"] == monitor.REPOSITORY and client.config["branch"] == monitor.BRANCH
    published = [(path, body) for path, body, method in calls if method == "PUT"]
    assert len(published) == 1
    assert published[0][0] == "repos/" + monitor.REPOSITORY + "/contents/" + expected
    assert published[0][1]["branch"] == monitor.BRANCH
