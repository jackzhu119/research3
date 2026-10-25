[第一轮SXM4-40GB拆分修复与独立续训入口](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/operations/round1_repair_20261025/LAUNCH_ZH.md)

本次仅更新第一轮；第六轮迁移已取消。第5/6轮的旧运行配置不变。

# 论文3 · 第1、5、6轮三作业并行运行方案

本分支保存2026年10月25日的资源交接封装。每轮三个独立单卡 Slurm 作业，复用原有完成项、断点与输出目录；本分支用于运行工具和交付记录，实时进度由各轮独立分支承载。

| 轮次 | 当前部署配置 | 状态入口 |
|---|---|---|
| 第1轮 | `hpc_gpu` A100 PCIe 40GB；各12CPU，内存43G/43G/42G；合计36CPU/128G | [第1轮状态](https://github.com/jackzhu119/research3/blob/paper3-r1-progress/monitoring/paper3-round1/status.json) |
| 第5轮 | `hpc_gpu1` A100 SXM4 80GB；各12CPU/64G；合计36CPU/192G | [第5轮三作业状态](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/three-jobs-status.json) |
| 第6轮 | `hpc_gpu1` A100 SXM4 80GB；各12CPU/64G；合计36CPU/192G | [第6轮三作业状态](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/monitoring/paper3-round6/three-jobs-status.json) |

- [运行与交接源码](operations/parallel_20261025/)。
- [已发布启动与恢复说明](operations/parallel_20261025/online/LAUNCH_ZH.md)。
- [按轮分类结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3)。

`hpc_gpu`、`hpc_gpu1`是分区；`gpu1`、`gpu2`等是节点名。状态文件的更新时间、实际队列和模型更新计数共同用于判断进展。启动说明保留已发布固定提交，已有作业无需因首页整理重复提交。

## 仓库导航

[仓库主页](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0) · [八轮实验总索引](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/experiments/README_ZH.md) · [最新进展](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/LIVE_STATUS_ZH.md) · [后续安排](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/ROADMAP_ZH.md) · [全部分支分类](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/BRANCHES_ZH.md)

## 历史首页

[整理前的 README 原文](docs/records/README_BEFORE_ORGANIZE_20261025.md)按原 Git blob 保留。
