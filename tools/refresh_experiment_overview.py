#!/usr/bin/env python3
"""Refresh repository navigation from committed observations; never access Slurm."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = "jackzhu119/research3"
START = "<!-- PAPER3_LIVE_STATUS_START -->"
END = "<!-- PAPER3_LIVE_STATUS_END -->"
MAX_BYTES = 32 * 1024 * 1024
SOURCES = {
    1: ("paper3-r1-progress", "monitoring/paper3-round1/status.json"),
    2: ("paper3-v2-progress", "monitoring/paper3-followup-v2/status.json"),
    **{n: (f"paper3-r{n}-progress", f"monitoring/paper3-round{n}/status.json") for n in (3, 4, 7)},
    **{n: (f"paper3-r{n}-progress", f"monitoring/paper3-round{n}/three-jobs-status.json") for n in (5, 6, 8)},
}
R1_PHASES = {
    "EXISTING_PREREQUISITES_PENDING": "既有前序产物待齐备",
    "META_SCREENING": "元学习参数筛选",
    "DEVELOPMENT_SCREENING": "开发集方法筛选",
    "META_CONFIRMATION": "额外种子元学习确认",
    "DEVELOPMENT_CONFIRMATION": "开发集确认",
    "MAINLINE_FINALIZATION": "主线冻结与内部评价",
    "EXISTING_SOURCE_EXTENSIONS": "既有源扩展实验",
    "ALL_ORIGINAL_WORK_COMPLETE": "全部原计划完成",
    "SCIENTIFIC_SELECTION_REQUIRES_REVIEW": "选择结果待核对",
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def timestamp(value):
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def local_time(value):
    stamp = timestamp(value)
    return dt.datetime.fromtimestamp(stamp, dt.timezone(dt.timedelta(hours=8))).strftime("%m-%d %H:%M:%S") if stamp else "未提供"


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


class Client:
    def __init__(self, repository, token=None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid repository")
        self.repository = repository
        self.token = token
        self.refs = {}

    def request(self, url, *, api=False):
        headers = {"User-Agent": "paper3-repository-overview", "Accept": "application/vnd.github+json"}
        if api and self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(MAX_BYTES + 1)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise RuntimeError("GitHubHTTP" + str(error.code)) from None
        if len(data) > MAX_BYTES:
            raise ValueError("Snapshot exceeds 32 MiB")
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("Expected JSON object")
        return value

    def head(self, branch):
        if branch not in self.refs:
            value = self.request("https://api.github.com/repos/" + self.repository + "/git/ref/heads/" + urllib.parse.quote(branch, safe="/"), api=True)
            self.refs[branch] = value["object"]["sha"] if value else None
        return self.refs[branch]

    def file(self, branch, path):
        sha = self.head(branch)
        if sha is None:
            return None, None
        value = self.request("https://raw.githubusercontent.com/" + self.repository + "/" + sha + "/" + urllib.parse.quote(path, safe="/"))
        return value, sha


def queue_observation(snapshot):
    queue = snapshot.get("live_queue", {})
    rows = queue.get("jobs", []) if isinstance(queue, dict) else []
    observed = isinstance(queue, dict) and queue.get("status") == "OBSERVED"
    if not observed:
        jobs = snapshot.get("jobs", [])
        jobs = list(jobs.values()) if isinstance(jobs, dict) else jobs
        rows = [j for j in jobs if isinstance(j, dict) and j.get("live_state")]
        observed = bool(rows)
    if not observed:
        # Legacy multi-GPU monitors do not perform a live queue query. Keep it unknown.
        return {"observed": False, "running": None, "pending": None, "jobs": []}
    jobs = []
    for row in rows:
        state = row.get("state", row.get("live_state", "UNKNOWN"))
        jobs.append({"job_id": str(row.get("job_id", "")), "state": state,
                     "reason": row.get("reason_or_node", row.get("live_reason_or_node")),
                     "partition": row.get("partition")})
    return {"observed": True, "running": sum(j["state"] == "RUNNING" for j in jobs),
            "pending": sum(j["state"] == "PENDING" for j in jobs), "jobs": jobs,
            "observed_utc": queue.get("observed_utc", snapshot.get("updated_utc"))}


def normalize(number, snapshot, known=None):
    known = known or {}
    observed_done = snapshot.get("complete_tasks")
    observed_total = snapshot.get("total_tasks")
    completed = (isinstance(observed_done, int) and observed_done == observed_total and observed_done > 0)
    complete_known = known.get("completed") is True
    queue = queue_observation(snapshot)
    if number == 1:
        phase = snapshot.get("coordinator", {}).get("phase", snapshot.get("phase"))
        if snapshot.get("all_original_complete") is True or snapshot.get("all_complete") is True or phase == "ALL_ORIGINAL_WORK_COMPLETE":
            status = "全部原计划完成"
        elif snapshot.get("mainline_complete") is True or snapshot.get("mainline", {}).get("coordinator_declares_complete") is True:
            status = "主线完成；既有扩展进度另计"
        else:
            status = "阶段记录：" + R1_PHASES.get(phase, str(phase or "尚无阶段记录"))
        done, total = None, None
    elif completed or complete_known:
        status = "任务完成"
        done = known.get("complete_tasks") if complete_known else observed_done
        total = known.get("total_tasks") if complete_known else observed_total
    else:
        done, total = observed_done, observed_total
        if queue["observed"] and queue["running"]:
            status = "作业运行；优化进度另见任务计数"
        elif queue["observed"] and queue["pending"]:
            status = "作业排队"
        elif snapshot.get("report_status") == "COMPLETE":
            status = "报告完成；任务计数口径或时间不同"
        else:
            status = "最近回传状态；未确认当前调度"
    row = {"round": number, "status": status, "complete_tasks": done, "total_tasks": total,
           "snapshot_complete_tasks": observed_done, "snapshot_total_tasks": observed_total,
           "updated_utc": snapshot.get("updated_utc"), "queue": queue,
           "report_status": snapshot.get("report_status"), "controller_status": snapshot.get("status", snapshot.get("controller_status")),
           "completion_evidence": known if complete_known else None,
           "counts_conflict": complete_known and (observed_done, observed_total) != (done, total)}
    if number == 1:
        row["completed_by_kind"] = snapshot.get("completed_by_kind", {})
        row["phase"] = snapshot.get("coordinator", {}).get("phase")
        row["active_actions"] = [a for a in snapshot.get("mainline", {}).get("active_actions", []) if a.get("status") == "RUNNING"]
        row["old_interruption_observations"] = [
            {"action_key": action.get("key"), "age_seconds": observation.get("age_seconds"),
             "snapshot_observed_utc": snapshot.get("updated_utc"), "values": observation.get("values")}
            for action in row["active_actions"] for observation in action.get("observations", [])
            if observation.get("values", {}).get("status") == "INTERRUPTED_NOT_COMPLETE"]
        if row["old_interruption_observations"] and not (snapshot.get("all_original_complete") is True or snapshot.get("mainline_complete") is True):
            row["status"] += "；当前动作仍展示旧中断断点"
    return row


def collect(client, previous, known, now):
    old_rounds = {r["round"]: r for r in previous.get("rounds", [])}
    rows = []
    for number, (branch, path) in SOURCES.items():
        old = old_rounds.get(number, {})
        row = dict(old)
        completion = known.get("rounds", {}).get(str(number), {})
        try:
            value, commit = client.file(branch, path)
            if value is None:
                raise ValueError("ProgressNotPublished")
            row = normalize(number, value, completion)
            row.update(source_branch=branch, source_path=path, source_commit=commit,
                       source_url=f"https://github.com/{client.repository}/blob/{branch}/{path}")
            row.pop("source_error", None)
        except Exception as error:
            if completion.get("completed") is True:
                row.update(status="任务完成", complete_tasks=completion.get("complete_tasks"),
                           total_tasks=completion.get("total_tasks"), completion_evidence=completion)
            row.update(round=number, source_branch=branch, source_path=path,
                       source_error=type(error).__name__, status=row.get("status", "尚无可读回传"))
        age = timestamp(now)
        updated = timestamp(row.get("updated_utc"))
        row["snapshot_stale"] = updated is None or age - updated > 900
        if number == 1:
            row["last_manual_resume_observation"] = known.get("round1_resume_observation")
        for filename, field in (("LATEST.json", "full_archive"),
                                ("AUTO_SUMMARY_LATEST.json", "automatic_summary"),
                                ("PAPER_DATA_LATEST.json", "paper_reports")):
            try:
                receipt, commit = client.file("paper3-results", f"results/paper3/round-{number:02d}/{filename}")
                verified = bool(receipt and receipt.get("round") == number)
                if field == "full_archive":
                    verified = verified and receipt.get("status") == "COMPLETE_OUTPUT_UPLOADED"
                elif field == "automatic_summary":
                    verified = verified and receipt.get("status") == "AUTOMATIC_EXISTING_RESULT_SUMMARY_UPLOADED"
                else:
                    if receipt and (receipt.get("round") != number or receipt.get("status") not in
                                    {"PAPER_REPORT_FILES_UPLOADED", "PAPER_FILES_PARTIAL"}):
                        raise ValueError("UnrecognizedPaperReportReceipt")
                    verified = verified and receipt.get("status") in {"PAPER_REPORT_FILES_UPLOADED", "PAPER_FILES_PARTIAL"}
                row[field] = {"published": verified, "source_commit": commit,
                              "receipt_path": f"results/paper3/round-{number:02d}/{filename}"}
                if field == "paper_reports":
                    row[field]["complete"] = verified and receipt.get("status") == "PAPER_REPORT_FILES_UPLOADED"
                if receipt:
                    row[field]["receipt"] = receipt
            except Exception as error:
                retained = dict(old.get(field, {"published": False}))
                retained["read_error"] = type(error).__name__
                row[field] = retained
        rows.append(row)
    return {"schema_version": "paper3-live-overview-v1", "repository": client.repository,
            "refreshed_utc": now, "scope": "Committed remote observations, not a direct school scheduler query",
            "rounds": sorted(rows, key=lambda r: r["round"]),
            "last_school_upload_observation": known.get("upload_observation"),
            "last_manual_queue_observation": known.get("queue_observation"),
            "freshness_policy": "Progress sources older than 15 minutes are marked stale; completed-task evidence remains valid."}


def compact_table(data):
    lines = ["| 轮次 | 最近确认状态 | 任务完成 | 队列观测 | 来源时间（北京） | 论文报告 | 完整结果包 |",
             "|---|---|---|---|---|---|---|"]
    for row in data["rounds"]:
        number = row["round"]
        count = f'{row["complete_tasks"]}/{row["total_tasks"]}' if row.get("complete_tasks") is not None else "分阶段，不合并百分比"
        queue = row.get("queue", {})
        state = f'{queue["running"]}运行 / {queue["pending"]}排队' if queue.get("observed") else "本源未查询队列"
        tag = "；快照滞后" if row.get("snapshot_stale") else ""
        if row.get("source_error"):
            tag += "；读取失败，保留上次记录"
        full = "上传完成回执已发布" if row.get("full_archive", {}).get("published") else "未见完整上传完成回执"
        if row.get("full_archive", {}).get("read_error"):
            full = "远端读取失败；完整上传状态待核实"
        paper = row.get("paper_reports", {})
        paper_text = "尚未发布论文报告回执"
        if paper.get("published"):
            paper_label = "论文报告已上传" if paper.get("complete") else "部分论文文件已上传"
            paper_url = f"https://github.com/{data['repository']}/blob/paper3-results/results/paper3/round-{number:02d}/PAPER_DATA_ZH.md"
            paper_text = f"[{paper_label}]({paper_url})"
        if paper.get("read_error"):
            paper_text += "；读取失败，保留上次记录" if paper.get("published") else "；读取失败，状态待核实"
        status = row.get("status", "未知") + tag
        url = row.get("source_url", f"https://github.com/{data['repository']}/tree/{row['source_branch']}")
        lines.append(f'| [第{number}轮]({url}) | {status} | {count} | {state} | {local_time(row.get("updated_utc"))} | {paper_text} | {full} |')
    return "\n".join(lines)


def render(data):
    lines = ["# 八轮最新回传状态", "", f'最近汇集：**{local_time(data["refreshed_utc"])} 北京时间**。源文件各有独立时间；刷新页面不会刷新学校本地日志。', "",
             compact_table(data), "", "任务完成、报告完成、Slurm作业运行和完整文件上传分别记录。控制器的 RUNNING 不代表 GPU 已开始优化。任务数量不能换算成剩余时间。",
             "", "第一轮按阶段保存断点；进程心跳不等于模型更新。第七轮旧监控计数与已完成证据冲突时，保留两个口径，不让旧快照抹去已确认完成项。",
             "", "论文报告以 PAPER_DATA_LATEST.json 回执为准，包含该回执声明的既有报告、CSV和任务元数据；部分回执不表示已传齐所需文件。论文报告与 AUTO_SUMMARY_LATEST.json 分析摘要都不表示完整预测或模型断点已备份。完整包以 paper3-results 各轮 LATEST.json 的上传器回执为准。回执是已发布声明，本页面不重新下载或核验大型归档。"]
    upload = data.get("last_school_upload_observation")
    if upload:
        lines += ["", "## 学校完整上传器：最后人工观测", "", "```json", json.dumps(upload, ensure_ascii=False, indent=2), "```", "", "这段学校本地打包进度不是实时远端推送；后续完整上传完成后，以上各轮归档回执会自动刷新。"]
    lines += ["", "## 自动更新", "", "默认分支 GitHub Actions 每30分钟尝试汇集一次，可手动触发 Refresh experiment overview。GitHub 定时任务可能延迟；仓库 Actions 被禁用、默认令牌写权限受限或网络失败时，页面不会按期更新，学校训练不受影响。工作流运行结果可在仓库 Actions 查看；部署文件存在不等于首次自动运行已验证。学校已有监控仍独立推送各自分支。", "", "[机器可读状态](LIVE_STATUS.json) · [分支分类](BRANCHES_ZH.md) · [实验总索引](experiments/README_ZH.md)", ""]
    return "\n".join(lines)


def semantic(value):
    return {k: v for k, v in value.items() if k != "refreshed_utc"}


def update(root, data):
    status = root / "docs/LIVE_STATUS.json"
    previous = read_json(status, {})
    if semantic(previous) == semantic(data):
        return False
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (root / "docs/LIVE_STATUS_ZH.md").write_text(render(data))
    readme = root / "README.md"
    if readme.is_file():
        content = readme.read_text()
        if START not in content or END not in content:
            raise ValueError("README live-status markers missing")
        begin, rest = content.split(START, 1)
        _, end = rest.split(END, 1)
        replacement = "\n\n" + compact_table(data) + "\n\n[完整状态、观测时间与上传口径](docs/LIVE_STATUS_ZH.md)\n\n"
        readme.write_text(begin + START + replacement + END + end)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", REPOSITORY))
    args = parser.parse_args()
    known = read_json(args.root / "docs/records/KNOWN_COMPLETIONS_20261025.json", {})
    previous = read_json(args.root / "docs/LIVE_STATUS.json", {})
    client = Client(args.repository, os.environ.get("GITHUB_TOKEN"))
    value = collect(client, previous, known, utc_now())
    changed = update(args.root, value)
    print(json.dumps({"status": "UPDATED" if changed else "UNCHANGED", "rounds": len(value["rounds"]),
        "source_read_errors": sum(bool(r.get("source_error")) for r in value["rounds"]),
        "archive_read_errors": sum(bool(r.get("full_archive", {}).get("read_error")) for r in value["rounds"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
