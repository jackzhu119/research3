# 论文3 · 第4至6轮早期单节点三卡迁移交付

本分支保存2026年10月24日第4、5、6轮迁往 `hpc_gpu` 的独立操作包、固定启动入口和验证证据。当时的统一目标为 A100 PCIe 40GB、一个节点一个作业内三卡、总36CPU / 128G主机内存。

**最新部署有后续变更：第4轮继续使用单作业内三卡方案；第5、6轮已登记为 `hpc_gpu1` 三个独立单卡作业，各12CPU / 64G。** 第5、6轮后续封装请看 [10月25日三作业交接分支](https://github.com/jackzhu119/research3/tree/paper3-parallel-migrations-20261025)。

## 代码与当前状态

- [本次交付源码](operations/hpc_gpu_migrations_20261024/) · [历史启动说明](operations/hpc_gpu_migrations_20261024/online/LAUNCH_ZH.md)。
- [第4轮当前状态](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/monitoring/paper3-round4/status.json)。
- [第5轮三作业状态](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/three-jobs-status.json)。
- [第6轮三作业状态](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/monitoring/paper3-round6/three-jobs-status.json)。
- [各轮分类结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3)。

这是带日期的运行交付记录。已发布的包、数学实现和固定提交下载地址保留；判断正在排队或运行的任务，请读取各轮状态的新鲜队列观测。

## 仓库导航

[仓库主页](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0) · [八轮实验总索引](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/experiments/README_ZH.md) · [最新进展](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/LIVE_STATUS_ZH.md) · [后续安排](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/ROADMAP_ZH.md) · [全部分支分类](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/BRANCHES_ZH.md)

## 历史首页

[整理前的 README 原文](docs/records/README_BEFORE_ORGANIZE_20261025.md)按原 Git blob 保留。
