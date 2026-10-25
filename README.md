# 论文3 · 第八轮实验发布

本分支保存第八轮的独立实验实现、数学设计、有限实验矩阵、数据缓存和在线安装包。研究方向为部分协方差输运、Bures/源度量几何，以及保持协方差的类别方向校正；数学性质和软件验证不预先证明识别性能提升。

## 本轮入口

- [数学与实验设计](experiments/followup_20261025_r8/DESIGN_ZH.md) · [前轮证据](experiments/followup_20261025_r8/EVIDENCE_ZH.md) · [文献](experiments/followup_20261025_r8/LITERATURE.md)。
- [源码与运行手册](experiments/followup_20261025_r8/README.md) · [发布的单行启动说明](experiments/followup_20261025_r8/LAUNCH_ZH.md)。
- [当前三作业状态](https://github.com/jackzhu119/research3/blob/paper3-r8-progress/monitoring/paper3-round8/three-jobs-status.json)。
- [第八轮分类结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-08)。

实验计划包含2,835项固定任务，其中24项元训练；Stieger、Pan、Weibo分别分析。学校配置为 `hpc_gpu` A100 PCIe 40GB，三个独立单卡作业，各16CPU / 64G主机内存，总48CPU / 192G。源码发布、学校排队、实际优化和结果归档是不同状态，具体进展查看独立状态分支。

学校输出：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261025_r8`。训练断点和逐会话结果在本地保存；自动分析摘要发往 `paper3-results:results/paper3/round-08/automatic-summary/`，摘要不能替代完整模型和预测文件备份。公开启动说明供部署或恢复查阅；已有作业应先查看当前队列与回执。

## 仓库导航

[仓库主页](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0) · [八轮实验总索引](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/experiments/README_ZH.md) · [最新进展](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/LIVE_STATUS_ZH.md) · [后续安排](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/ROADMAP_ZH.md) · [全部分支分类](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/BRANCHES_ZH.md)

## 历史首页

[整理前的 README 原文](docs/records/README_BEFORE_ORGANIZE_20261025.md)按原 Git blob 保留。固定提交启动包和全部实验源码保持原路径。
