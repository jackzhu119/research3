"""Read-only compact evidence capture for completed Paper3 rounds.

Only declared completion receipts and condition summaries are opened. Large
shallow report files are streamed verbatim; no trial/session scan, model import,
report rebuild, averaging, scheduler command or network operation is performed.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time


OUTPUT_NAMES = {4: "followup_20261024_r4", 5: "followup_20261024_r5",
                6: "followup_20261024_r6", 8: "followup_20261025_r8"}
REPORT_PARSE_LIMIT = 16 * 1024 * 1024
REPORT_SUFFIXES = {".csv", ".json", ".jsonl", ".md"}
ROOT_REPORT_NAMES = ("report.json", "report.md", "report.csv", "full-report.json",
                     "report-cache.json")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _signature(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _source_bytes(path, attempts=2):
    """One stable byte snapshot, with a single retry after atomic replacement."""
    path = Path(path)
    for _ in range(attempts):
        with path.open("rb") as stream:
            before = _signature(os.fstat(stream.fileno()))
            data = stream.read()
            after = _signature(os.fstat(stream.fileno()))
        if before == after == _signature(path.stat()):
            return data
    raise OSError("source_changed_during_capture: " + str(path))


def _capture_file(source, destination, attempts=2):
    """Stream even very large reports, retaining every byte and a whole SHA."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        fd, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", dir=destination.parent)
        try:
            digest, count = hashlib.sha256(), 0
            with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
                before = _signature(os.fstat(inp.fileno()))
                for block in iter(lambda: inp.read(1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
                    out.write(block)
                after = _signature(os.fstat(inp.fileno()))
                out.flush()
                os.fsync(out.fileno())
            if before == after == _signature(source.stat()):
                os.replace(temporary, destination)
                return {"bytes": count, "sha256": digest.hexdigest(),
                        "source_signature": list(before)}
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    raise OSError("source_changed_during_capture: " + str(source))


def _safe_task(task):
    key = task.get("key")
    return isinstance(key, str) and key not in ("", ".", "..") and Path(key).name == key


def _coordinates(plan, task):
    """Coordinates mirror the worker's explicit budget/scenario loop."""
    if task.get("family") in ("basis", "train", "class_reference"):
        return []
    partitions = task.get("partitions", [task.get("partition", "holdout")])
    budgets = task.get("budgets", plan.get("budgets", [20, 4, 8, 12, 16]))
    result = []
    for partition in partitions:
        if not isinstance(partition, str) or Path(partition).name != partition:
            raise ValueError("Invalid declared evaluation partition")
        for budget in budgets:
            if isinstance(budget, bool) or int(budget) != budget or int(budget) <= 0:
                raise ValueError("Invalid declared support budget")
            for scenario in task.get("scenarios", ["clean"]):
                if not isinstance(scenario, str) or Path(scenario).name != scenario:
                    raise ValueError("Invalid declared evaluation scenario")
                # All four current workers execute diagnostics/nulls only B20.
                if scenario != "clean" and int(budget) != 20:
                    continue
                result.append((partition, int(budget), scenario))
    return list(dict.fromkeys(result))


def _metric_warning(summary):
    participants = summary.get("participants")
    if not isinstance(participants, list) or not participants:
        return "summary_has_no_participant_metrics"
    for person in participants:
        if not isinstance(person, dict) or "participant_id" not in person or "ba" not in person or not any(
                key in person for key in ("bce", "nll_balanced", "nll")):
            return "summary_participant_BA_or_NLL_missing"
    return None


def report_candidates(output):
    """Exact shallow report inventory, shared with upload retry freshness checks."""
    output = Path(output)
    candidates = []
    report_dir = output / "reports"
    if report_dir.is_dir():
        candidates.extend(path for path in report_dir.iterdir()
                          if path.suffix.lower() in REPORT_SUFFIXES and path.is_file())
    candidates.extend(output / name for name in ROOT_REPORT_NAMES if (output / name).is_file())
    candidates.extend(path for path in output.iterdir()
                      if path.suffix.lower() == ".csv" and path.is_file())
    correction_path = output / "runtime/runtime-corrections.json"
    if correction_path.is_file():
        candidates.append(correction_path)
    return sorted(set(candidates))


def collect_round(root, round_number, work_dir, emit=None):
    """Capture the requested round into a local, immutable-style staging folder.

    `all_complete` describes observed plan/receipt agreement. Existing legacy
    receipts without an explicit recipe field are exported with an explicit
    provenance label, rather than imposing a new experimental acceptance gate.
    Missing/corrupt evidence is reported as warnings; other evidence is retained.
    """
    if round_number not in OUTPUT_NAMES:
        raise ValueError("Supported rounds are 4, 5, 6 and 8")
    root, work_dir = Path(root).expanduser().resolve(), Path(work_dir).expanduser().resolve()
    output = root / "runs" / OUTPUT_NAMES[round_number]
    for name in OUTPUT_NAMES.values():
        if work_dir.is_relative_to((root / "runs" / name).resolve()):
            raise ValueError("Evidence staging must be outside scientific output directories")
    stage = work_dir / ("round-%02d" % round_number) / "paper-evidence"
    stage.mkdir(parents=True, exist_ok=True)
    paths, sources, warnings, reports = [], [], [], []
    notify = emit or (lambda *args, **kwargs: None)

    def warn(code, **values):
        row = dict(code=code, **values)
        warnings.append(row)
        notify("COLLECT_WARNING", round=round_number, **row)

    def relative(path):
        try:
            return Path(path).relative_to(root).as_posix()
        except ValueError:
            return str(path)

    def generated(name, value):
        _atomic_bytes(stage / name, _json_bytes(value))
        paths.append(name)

    plan_path = output / "plan.json"
    plan_raw = _source_bytes(plan_path)
    plan = json.loads(plan_raw)
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        raise ValueError("Existing plan lacks task declarations")
    if not isinstance(plan.get("recipe_sha256"), str):
        raise ValueError("Existing plan lacks recipe identity")
    if plan.get("output") and Path(plan["output"]).resolve() != output.resolve():
        warn("plan_output_path_differs", declared=plan["output"], observed=str(output))
    _atomic_bytes(stage / "plan.json", plan_raw)
    paths.append("plan.json")
    sources.append({"source_path": relative(plan_path), "path": "plan.json",
                    "bytes": len(plan_raw), "sha256": _sha(plan_raw)})
    task_lines, condition_lines, task_index, optional_lines = [], [], [], []
    complete_count = expected_count = captured_count = metric_count = 0
    recipe_modes = {}
    for index, task in enumerate(plan["tasks"]):
        if not isinstance(task, dict) or not _safe_task(task):
            warn("unsafe_task_key", task_index=index)
            task_index.append({"round": round_number, "task_index": index, "complete": False,
                               "status": "INVALID_TASK_DECLARATION"})
            continue
        key = task["key"]
        receipt_path = output / key / "result.json"
        receipt, receipt_sha, receipt_bytes, issue = None, None, None, None
        try:
            receipt_bytes = _source_bytes(receipt_path)
            receipt_sha = _sha(receipt_bytes)
            receipt = json.loads(receipt_bytes)
            if not isinstance(receipt, dict):
                raise ValueError("receipt_not_object")
        except (OSError, ValueError) as error:
            issue = type(error).__name__ + ":" + str(error)[:400]
            warn("receipt_unreadable", task=key, source_path=relative(receipt_path), reason=issue)
        complete = bool(receipt and receipt.get("status") == "COMPLETE")
        recipe_mode = "missing_receipt"
        if receipt is not None:
            if receipt.get("task") is not None and receipt["task"] != task:
                complete = False
                warn("receipt_task_mismatch", task=key)
            if "recipe_sha256" in receipt:
                recipe_mode = "explicit_recipe_hash"
                if receipt["recipe_sha256"] != plan["recipe_sha256"]:
                    complete = False
                    warn("receipt_recipe_mismatch", task=key)
            else:
                recipe_mode = "legacy_receipt_no_explicit_recipe_hash; plan/task/frozenidentity exported"
            recipe_modes[recipe_mode] = recipe_modes.get(recipe_mode, 0) + 1
            task_lines.append({"round": round_number, "task_index": index, "task_key": key,
                "source_path": relative(receipt_path), "source_sha256": receipt_sha,
                "source_bytes": len(receipt_bytes), "plan_recipe_sha256": plan["recipe_sha256"],
                "recipe_verification": recipe_mode, "plan_task_agrees": receipt.get("task", task) == task,
                "complete_receipt_verified": complete, "receipt": receipt})
            sources.append({"source_path": relative(receipt_path), "bundle": "analysis/task-results.jsonl",
                            "record_task_index": index, "bytes": len(receipt_bytes), "sha256": receipt_sha})
        if complete:
            complete_count += 1
        elif receipt is not None and receipt.get("status") != "COMPLETE":
            warn("task_not_complete", task=key, status=receipt.get("status"))
        conditions = _coordinates(plan, task)
        expected_count += len(conditions)
        found = 0
        for partition, budget, scenario in conditions:
            source = output / key / "evaluations" / partition / ("B%d" % budget) / scenario / "summary.json"
            try:
                raw = _source_bytes(source)
                summary = json.loads(raw)
                if not isinstance(summary, dict):
                    raise ValueError("summary_not_object")
            except (OSError, ValueError) as error:
                warn("condition_summary_unreadable", task=key, partition=partition, budget=budget,
                     scenario=scenario, source_path=relative(source), reason=type(error).__name__)
                continue
            found += 1
            captured_count += 1
            row = {"round": round_number, "task_index": index, "task_key": key,
                   "family": task.get("family"), "group": task.get("group"), "seed": task.get("seed"),
                   "partition": partition, "budget": budget, "scenario": scenario,
                   "task_identity": receipt.get("identity") if receipt else None,
                   "source_path": relative(source), "source_sha256": _sha(raw), "source_bytes": len(raw),
                   "plan_recipe_sha256": plan["recipe_sha256"], "summary": summary}
            condition_lines.append(row)
            sources.append({"source_path": relative(source), "bundle": "analysis/condition-summaries.jsonl",
                "record_condition": [key, partition, budget, scenario], "bytes": len(raw), "sha256": _sha(raw)})
            metric_issue = _metric_warning(summary)
            if metric_issue:
                warn(metric_issue, task=key, partition=partition, budget=budget, scenario=scenario)
            else:
                metric_count += 1
        # One known final validation summary per dictionary learner is optional.
        # Intermediate checkpoint trajectories remain in the full archive channel.
        if task.get("family") == "basis":
            source = output / key / "evaluations/meta_validation/B20/clean/summary.json"
            if source.is_file():
                try:
                    raw = _source_bytes(source)
                    optional_lines.append({"round": round_number, "task_key": key,
                        "kind": "selected_basis_meta_validation", "source_path": relative(source),
                        "source_sha256": _sha(raw), "summary": json.loads(raw)})
                    sources.append({"source_path": relative(source),
                        "bundle": "analysis/selected-basis-validation.jsonl", "record_task_key": key,
                        "bytes": len(raw), "sha256": _sha(raw)})
                except (OSError, ValueError) as error:
                    warn("optional_basis_summary_unreadable", task=key, reason=type(error).__name__)
        correction = receipt.get("runtime_corrections", {}) if receipt else {}
        task_index.append({"round": round_number, "task_index": index, "task_key": key,
            "family": task.get("family"), "group": task.get("group"), "seed": task.get("seed"),
            "status": receipt.get("status") if receipt else "UNREADABLE", "complete": complete,
            "recipe_verification": recipe_mode, "identity": receipt.get("identity") if receipt else None,
            "receipt_path": relative(receipt_path), "receipt_sha256": receipt_sha,
            "runtime_hook_sha256": correction.get("runtime_hook_sha256") if isinstance(correction, dict) else None,
            "expected_conditions": len(conditions), "captured_conditions": found, "read_issue": issue})
        if (index + 1) % 100 == 0 or index + 1 == len(plan["tasks"]):
            notify("COLLECT_TASK_PROGRESS", round=round_number, checked=index + 1,
                   total=len(plan["tasks"]), complete=complete_count, summaries=captured_count)

    def jsonl(name, rows):
        _atomic_bytes(stage / name, b"".join(_json_bytes(row) for row in rows))
        paths.append(name)

    jsonl("analysis/task-results.jsonl", task_lines)
    jsonl("analysis/condition-summaries.jsonl", condition_lines)
    jsonl("analysis/selected-basis-validation.jsonl", optional_lines)
    fields = ["round", "task_index", "task_key", "family", "group", "seed", "status", "complete",
              "recipe_verification", "identity", "receipt_path", "receipt_sha256", "runtime_hook_sha256",
              "expected_conditions", "captured_conditions", "read_issue"]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    writer.writerows(task_index)
    _atomic_bytes(stage / "analysis/task-index.csv", text.getvalue().encode())
    paths.append("analysis/task-index.csv")

    candidates = report_candidates(output)
    for source in candidates:
        if not source.resolve().is_relative_to(output.resolve()):
            warn("report_path_outside_round_output", source_path=relative(source))
            continue
        target = source.relative_to(output).as_posix()
        try:
            info = _capture_file(source, stage / target)
        except OSError as error:
            warn("report_capture_unavailable", source_path=relative(source), reason=str(error)[:400])
            continue
        paths.append(target)
        row = {"source_path": relative(source), "path": target, **info}
        if source.suffix.lower() == ".json":
            if info["bytes"] <= REPORT_PARSE_LIMIT:
                try:
                    value = json.loads((stage / target).read_bytes())
                    if isinstance(value, dict):
                        row["reported_metadata"] = {k: value[k] for k in
                            ("status", "complete_tasks", "total_tasks", "core_complete", "recipe_sha256", "round")
                            if k in value}
                except ValueError:
                    row["reported_metadata"] = {"parse_status": "UNREADABLE_JSON_CAPTURED_EXACTLY"}
            else:
                row["reported_metadata"] = {"parse_status": "LARGE_REPORT_CAPTURED_NOT_PARSED",
                                            "size_parse_limit_bytes": REPORT_PARSE_LIMIT}
        reports.append(row)
        sources.append(row)
        notify("COLLECT_REPORT_CAPTURED", round=round_number, path=target, bytes=info["bytes"])
    all_complete = bool(plan["tasks"]) and complete_count == len(plan["tasks"])
    paper_complete = all_complete and metric_count == expected_count
    primary_reports = [row for row in reports if Path(row["path"]).name in
                       ("report.json", "full-report.json", "report-full.json", "summary.json")]
    reports_stale_or_unknown = not primary_reports or any(
        row.get("reported_metadata", {}).get("status") not in ("COMPLETE", "COMPLETED", "FINISHED")
        or row.get("reported_metadata", {}).get("complete_tasks", complete_count) != complete_count
        or row.get("reported_metadata", {}).get("total_tasks", len(plan["tasks"])) != len(plan["tasks"])
        or row.get("reported_metadata", {}).get("recipe_sha256", plan["recipe_sha256"]) != plan["recipe_sha256"]
        for row in primary_reports)
    critical_codes = {"receipt_unreadable", "receipt_task_mismatch", "receipt_recipe_mismatch", "report_capture_unavailable",
                      "task_not_complete", "unsafe_task_key", "condition_summary_unreadable",
                      "summary_has_no_participant_metrics", "summary_participant_BA_or_NLL_missing"}
    required_missing = [row for row in warnings if row["code"] in critical_codes]
    metadata = {"schema_version": "paper3-completed-round-compact-evidence-v1", "round": round_number,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_output": str(output),
        "plan_source_path": relative(plan_path), "plan_sha256": _sha(plan_raw),
        "recipe_sha256": plan["recipe_sha256"], "total_tasks": len(plan["tasks"]),
        "complete_tasks": complete_count, "all_complete": all_complete,
        "recipe_verification_modes": recipe_modes, "task_receipts_captured": len(task_lines),
        "expected_condition_summaries": expected_count, "captured_condition_summaries": captured_count,
        "condition_summaries_with_participant_BA_NLL": metric_count,
        "all_condition_summaries_captured": captured_count == expected_count,
        "paper_metric_evidence_complete": paper_complete,
        "collection_status": "COMPLETE" if paper_complete and not required_missing else "PARTIAL",
        "required_missing": required_missing,
        "reports_stale_or_unknown": reports_stale_or_unknown,
        "optional_basis_validation_summaries": len(optional_lines), "reports": reports,
        "report_inventory": sorted(relative(path) for path in candidates),
        "warnings": warnings, "source_versions": {"plan_sha256": _sha(plan_raw),
            "recipe_sha256": plan["recipe_sha256"], "code_sha256": plan.get("code_sha256")},
        "source_outputs_modified": False, "native_reports_recomputed": False,
        "statistics_recomputed": False, "numpy_or_torch_imported": False,
        "per_session_or_trial_files_scanned": False, "large_reports_skipped": False,
        "evidence_scope": "Stored participant BA/NLL summaries and exact report/receipt evidence; validation/trial/checkpoint payloads use the full archive channel"}
    generated("analysis/collection-metadata.json", metadata)
    generated("analysis/source-index.json", {"schema_version": "paper3-evidence-source-index-v1",
              "round": round_number, "sources": sources})
    notify("COLLECT_ROUND_READY", round=round_number, complete=complete_count, total=len(plan["tasks"]),
           condition_summaries=captured_count, expected_condition_summaries=expected_count,
           reports=len(reports), warnings=len(warnings))
    return {"source_base": stage, "paths": sorted(set(paths)), "metadata": metadata}
