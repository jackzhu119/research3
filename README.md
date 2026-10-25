# 论文3 · 第7轮状态与源码

本分支承载第7轮的异步 GitHub 回传，并保留本轮实验源码与设计。有限窗时空 Lie、头容量可辨识性、源相对风险与成分消融。

## 先看当前状态

- [第7轮最新回传](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/monitoring/paper3-round7/status.json)：`monitoring/paper3-round7/status.json`。
- [本轮设计与实现](https://github.com/jackzhu119/research3/tree/paper3-r7-progress/experiments/followup_20261024_r7)。
- [第7轮分类结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-07)。

本轮旧队列计数曾滞后于完成报告；请同时核对报告完整性、学校本地任务回执和源更新时间。

运行、排队、实际模型更新、报告生成和完整文件上传分别记录。请查看状态文件内的源更新时间；心跳或 `RUNNING` 字样本身不能证明模型更新次数正在增长。旧快照不覆盖更新后的学校本地回执。

## 学校资源与结果保存

`hpc_gpu`，A100 PCIe 40GB；一个节点、一个作业内三卡并行，总36CPU / 128G主机内存。

学校共享项目：`/public/home/zhuziyuan/projects/paper3`。模型断点、逐会话预测和本轮输出先保存在学校共享目录。进度回传及分析摘要不等于全部模型或逐试次文件已完成异地归档；完整归档以 `paper3-results` 对应轮次的 `LATEST.json` 及其引用 manifest/分片确认。

源码、启动说明和固定提交下载地址供部署与恢复查阅。已有作业的实际运行状态通过上述回传查看，不凭本分支首页重新提交。

## 仓库导航

[仓库主页](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0) · [八轮实验总索引](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/experiments/README_ZH.md) · [最新进展](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/LIVE_STATUS_ZH.md) · [后续安排](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/ROADMAP_ZH.md) · [全部分支分类](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/BRANCHES_ZH.md)

## 历史记录

本次整理前的根 README 按原 Git blob 保存在 [旧首页](docs/records/README_BEFORE_ORGANIZE_20261025.md)。旧说明保留原发布日期，当前状态通过本页上方入口读取。
