# 各轮最新回传状态

最近汇集：**10-28 00:03:41 北京时间**。源文件各有独立时间；刷新页面不会刷新学校本地日志。

| 轮次 | 最近确认状态 | 任务完成 | 队列观测 | 来源时间（北京） | 论文报告 | 完整结果包 |
|---|---|---|---|---|---|---|
| [第1轮](https://github.com/jackzhu119/research3/blob/paper3-r1-progress/monitoring/paper3-round1/status.json) | 阶段记录：开发集方法筛选 | 分阶段，不合并百分比 | 3运行 / 0排队 | 10-27 23:57:39 | 尚未发布论文报告回执 | 未见完整上传完成回执 |
| [第2轮](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json) | 任务完成；快照滞后 | 66/66 | 本源未查询队列 | 10-25 10:16:55 | [论文报告已上传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-02/PAPER_DATA_ZH.md) | 上传完成回执已发布 |
| [第3轮](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/monitoring/paper3-round3/status.json) | 任务完成；快照滞后 | 174/174 | 本源未查询队列 | 10-25 08:09:10 | [论文报告已上传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-03/PAPER_DATA_ZH.md) | 上传完成回执已发布 |
| [第4轮](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/monitoring/paper3-round4/status.json) | 任务完成；快照滞后 | 204/204 | 本源未查询队列 | 10-25 16:46:54 | 尚未发布论文报告回执；读取失败，状态待核实 | 上传完成回执已发布 |
| [第5轮](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/three-jobs-status.json) | 任务完成 | 105/105 | 0运行 / 0排队 | 10-27 23:55:35 | 尚未发布论文报告回执；读取失败，状态待核实 | 上传完成回执已发布 |
| [第6轮](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/monitoring/paper3-round6/three-jobs-status.json) | 任务完成 | 198/198 | 0运行 / 0排队 | 10-27 23:55:41 | 尚未发布论文报告回执；读取失败，状态待核实 | 上传完成回执已发布 |
| [第7轮](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/monitoring/paper3-round7/status.json) | 任务完成；快照滞后 | 273/273 | 本源未查询队列 | 10-25 09:03:06 | [论文报告已上传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-07/PAPER_DATA_ZH.md)；[聚合JSON已补传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-07/REPORT_JSON_ZH.md) | 上传完成回执已发布 |
| [第8轮](https://github.com/jackzhu119/research3/blob/paper3-r8-progress/monitoring/paper3-round8/three-jobs-status.json) | 任务完成 | 2835/2835 | 0运行 / 0排队 | 10-27 23:55:30 | 尚未发布论文报告回执；读取失败，状态待核实 | 未见完整上传完成回执 |
| [第9轮](https://github.com/jackzhu119/research3/blob/paper3-r9-progress/monitoring/paper3-round9/three-jobs-status.json) | 任务完成 | 1548/1548 | 0运行 / 0排队 | 10-27 23:56:30 | [论文报告已上传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-09/PAPER_DATA_ZH.md) | 上传完成回执已发布 |

任务完成、报告完成、Slurm作业运行和完整文件上传分别记录。控制器的 RUNNING 不代表 GPU 已开始优化。任务数量不能换算成剩余时间。

第一轮按阶段保存断点；进程心跳不等于模型更新。第七轮旧监控计数与已完成证据冲突时，保留两个口径，不让旧快照抹去已确认完成项。

论文报告以 PAPER_DATA_LATEST.json 回执为准，包含该回执声明的既有报告、CSV和任务元数据；部分回执不表示已传齐所需文件。第七轮大聚合JSON另以 REPORT_JSON_LATEST.json 补传回执为准，CSV回执不等于该JSON已经补齐。论文报告与 AUTO_SUMMARY_LATEST.json 分析摘要都不表示完整预测或模型断点已备份。完整包以 paper3-results 各轮 LATEST.json 的上传器回执为准。回执是已发布声明，本页面不重新下载或核验大型归档。

## 学校完整上传器：最后人工观测

```json
{
  "observed_at": "2026-10-25T15:44:38+08:00",
  "round": 2,
  "event": "PACK_PROGRESS",
  "files": 92377,
  "total_files": 152481,
  "source_bytes": 2577783849,
  "archive_bytes": 502931153,
  "process_pid": 645884,
  "process_present": true,
  "rounds_pending_in_order": [
    2,
    3,
    7
  ],
  "scope": "local packing before network upload; latest user-provided manual snapshot, not live",
  "source": "user_pasted_school_readonly_diagnostic_20261025_1544"
}
```

这段学校本地打包进度不是实时远端推送；后续完整上传完成后，以上各轮归档回执会自动刷新。

## 自动更新

默认分支 GitHub Actions 每30分钟尝试汇集一次，可手动触发 Refresh experiment overview。GitHub 定时任务可能延迟；仓库 Actions 被禁用、默认令牌写权限受限或网络失败时，页面不会按期更新，学校训练不受影响。工作流运行结果可在仓库 Actions 查看；部署文件存在不等于首次自动运行已验证。学校已有监控仍独立推送各自分支。

[机器可读状态](LIVE_STATUS.json) · [分支分类](BRANCHES_ZH.md) · [实验总索引](experiments/README_ZH.md)
