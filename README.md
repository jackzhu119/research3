# 论文3 · 第5轮状态与源码

本分支承载第5轮的异步 GitHub 回传，并保留本轮实验源码与设计。旋转与相对增益的几何组合、有序/对称 split、图精化和共同参数机制探针。

## 先看当前状态

- [第5轮最新回传](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/three-jobs-status.json)：`monitoring/paper3-round5/three-jobs-status.json`。
- [本轮设计与实现](https://github.com/jackzhu119/research3/tree/paper3-r5-progress/experiments/followup_20261024_r5)。
- [第5轮分类结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-05)。

[旧单作业状态](monitoring/paper3-round5/status.json)保留历史语境；当前三作业交接和排队以本页主状态入口为准。

运行、排队、实际模型更新、报告生成和完整文件上传分别记录。请查看状态文件内的源更新时间；心跳或 `RUNNING` 字样本身不能证明模型更新次数正在增长。旧快照不覆盖更新后的学校本地回执。

## 学校资源与结果保存

最新部署：`hpc_gpu1`，A100 SXM4 80GB；三个独立单卡作业，各12CPU / 64G主机内存，总36CPU / 192G。

学校共享项目：`/public/home/zhuziyuan/projects/paper3`。模型断点、逐会话预测和本轮输出先保存在学校共享目录。进度回传及分析摘要不等于全部模型或逐试次文件已完成异地归档；完整归档以 `paper3-results` 对应轮次的 `LATEST.json` 及其引用 manifest/分片确认。

源码、启动说明和固定提交下载地址供部署与恢复查阅。已有作业的实际运行状态通过上述回传查看，不凭本分支首页重新提交。

## 仓库导航

[仓库主页](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0) · [八轮实验总索引](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/experiments/README_ZH.md) · [最新进展](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/LIVE_STATUS_ZH.md) · [后续安排](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/ROADMAP_ZH.md) · [全部分支分类](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/BRANCHES_ZH.md)

## 历史记录

本次整理前的根 README 按原 Git blob 保存在 [旧首页](docs/records/README_BEFORE_ORGANIZE_20261025.md)。旧说明保留原发布日期，当前状态通过本页上方入口读取。
