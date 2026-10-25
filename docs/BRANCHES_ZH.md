# 论文3分支分类目录

分支审计：2026-10-25 13:41:53（北京时间）；共 **26** 个分支。默认分支为 `research/paper3-protocol-v2-w0`。分支名称保留，已发布固定提交下载、运行监控和实验输出通过统一导航衔接。

## 默认主页与核心协议（1）

默认分支保留核心算法、数据合同、协议与总导航。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [research/paper3-protocol-v2-w0](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0) | 默认主页、核心 p3work/ 与 protocol/、八轮实验总索引。 | [`c961ceba6c96`](https://github.com/jackzhu119/research3/commit/c961ceba6c96cd803c09bb608ffec61d7ee6f659) |

## 独立实验发布（1）

第八轮单独发布；第二至七轮的代码和设计位于各轮进度分支的 experiments/ 子目录。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [paper3-round8-20261025](https://github.com/jackzhu119/research3/tree/paper3-round8-20261025) | 第八轮部分协方差输运、Bures 几何与保持协方差的类别方向校正；experiments/followup_20261025_r8/。 | [`f1d5f6b543ee`](https://github.com/jackzhu119/research3/commit/f1d5f6b543ee90d4b7371da4331e20da0cb39bf9) |

## 按轮实时回传与对应实验源码（8）

原监控持续写入这些分支；不要合并、重命名或删除。状态文件是异步快照，报告与任务计数可能不同步。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [paper3-r1-progress](https://github.com/jackzhu119/research3/tree/paper3-r1-progress) | 第1轮并行续跑状态；monitoring/paper3-round1/status.json。 | [`0799bf9cfc73`](https://github.com/jackzhu119/research3/commit/0799bf9cfc732908f1b709b4dc4cb05b53bddf97) |
| [paper3-r3-progress](https://github.com/jackzhu119/research3/tree/paper3-r3-progress) | 第3轮代码 experiments/followup_20261024_r3/；状态 monitoring/paper3-round3/status.json。 | [`b56484bff67f`](https://github.com/jackzhu119/research3/commit/b56484bff67fd593b70e972a9e4c62d9d44d98aa) |
| [paper3-r4-progress](https://github.com/jackzhu119/research3/tree/paper3-r4-progress) | 第4轮代码 experiments/followup_20261024_r4/；状态与资源交接记录。 | [`7bcf9580e6f8`](https://github.com/jackzhu119/research3/commit/7bcf9580e6f83b5e037837b722519987f1f2e409) |
| [paper3-r5-progress](https://github.com/jackzhu119/research3/tree/paper3-r5-progress) | 第5轮代码 experiments/followup_20261024_r5/；状态与三作业交接记录。 | [`322bdee29070`](https://github.com/jackzhu119/research3/commit/322bdee29070c7c59cb1502497717303f13be2d7) |
| [paper3-r6-progress](https://github.com/jackzhu119/research3/tree/paper3-r6-progress) | 第6轮代码及 Pan 缓存 experiments/followup_20261024_r6/；状态与三作业交接记录。 | [`e46b70333449`](https://github.com/jackzhu119/research3/commit/e46b70333449b8d812144920218ba45ed5eb2f3f) |
| [paper3-r7-progress](https://github.com/jackzhu119/research3/tree/paper3-r7-progress) | 第7轮时空与可辨识性实验源码、缓存与异步摘要；experiments/followup_20261024_r7/。 | [`f6804b9a72ff`](https://github.com/jackzhu119/research3/commit/f6804b9a72ff118b732643e6b8e05b9f0f6c3cc6) |
| [paper3-r8-progress](https://github.com/jackzhu119/research3/tree/paper3-r8-progress) | 第8轮三个单卡作业的轻量运行状态；实验实现从独立第8轮发布分支读取。 | [`347928bb8e62`](https://github.com/jackzhu119/research3/commit/347928bb8e626560bac579cb5cb4660439fccf05) |
| [paper3-v2-progress](https://github.com/jackzhu119/research3/tree/paper3-v2-progress) | 第2轮代码 experiments/followup_20261023/；状态 monitoring/paper3-followup-v2/status.json。 | [`4706017ef85f`](https://github.com/jackzhu119/research3/commit/4706017ef85ff2fcf702e55da9258fd101fdbbf0) |

## 按轮结果分类与归档（1）

摘要、历史快照、完整结果归档分别标识；完整归档以 LATEST.json 及对应 manifest/Release 分片为准。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [paper3-results](https://github.com/jackzhu119/research3/tree/paper3-results) | 统一结果分区 results/paper3/round-01/ 至 round-08/；原始回传、自动摘要和完整归档按实际存在分开。 | [`919be7973d31`](https://github.com/jackzhu119/research3/commit/919be7973d312d940f82bc58a28266cf5ac51742) |

## 运行与资源迁移（2）

封装、调度和迁移入口；固定提交下载地址被历史命令引用，应保留。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [paper3-hpc-gpu-migrations-20261024](https://github.com/jackzhu119/research3/tree/paper3-hpc-gpu-migrations-20261024) | 第4至6轮迁往 hpc_gpu 的早期单节点三卡交付；第5、6轮最新部署已由10月25日方案取代。 | [`00d337fb2ba8`](https://github.com/jackzhu119/research3/commit/00d337fb2ba8368e6eae2317db6522a7586c02f7) |
| [paper3-parallel-migrations-20261025](https://github.com/jackzhu119/research3/tree/paper3-parallel-migrations-20261025) | 第1轮 hpc_gpu 与第5、6轮 hpc_gpu1 三个独立单卡作业方案；operations/parallel_20261025/。 | [`ecf4313561ba`](https://github.com/jackzhu119/research3/commit/ecf4313561bacdc35b96f2f976a4bf25e0ac9858) |

## 历史启动与学校部署版本（3）

保留历史启动问题、修复和交付证据；当前运行方案通过默认主页和各轮手册查找。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [research/paper3-oneclick-20261021-9ff3e9](https://github.com/jackzhu119/research3/tree/research/paper3-oneclick-20261021-9ff3e9) | 10月21日一键部署历史版本与启动验证记录。 | [`b8ae36286b25`](https://github.com/jackzhu119/research3/commit/b8ae36286b2548519fa5c9ab64e206d17becdf3e) |
| [research/paper3-online-20261021](https://github.com/jackzhu119/research3/tree/research/paper3-online-20261021) | 10月21日在线部署历史版本。 | [`c7ad797b072b`](https://github.com/jackzhu119/research3/commit/c7ad797b072b3d075911bb8f32224af473fc112a) |
| [research/paper3-slurm-hpc-20261021](https://github.com/jackzhu119/research3/tree/research/paper3-slurm-hpc-20261021) | 10月21日学校 Slurm 部署历史版本。 | [`5752600ea467`](https://github.com/jackzhu119/research3/commit/5752600ea4676fec2196142ce6a27f776cef21d2) |

## 历史离线分发包（10）

按原始提交和包标识保留的离线交付记录，不作为新实验默认入口。分支名中的 completed 仅是交付版本名称，不表示全部科学实验完成。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [distribution/paper3-hpc-completed-4d43325fc960](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-completed-4d43325fc960) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`5c1b904dd84e`](https://github.com/jackzhu119/research3/commit/5c1b904dd84e344f4529e060a5b0d9a541b3b4da) |
| [distribution/paper3-hpc-offline-7ae65bf](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-7ae65bf) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`fd03f3c715ee`](https://github.com/jackzhu119/research3/commit/fd03f3c715ee892050f0a1cf7aea7ac596bc207e) |
| [distribution/paper3-hpc-offline-9e42181](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-9e42181) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`4e1aa0962e98`](https://github.com/jackzhu119/research3/commit/4e1aa0962e983bd79a1ec2173a90b78f3b71500e) |
| [distribution/paper3-hpc-offline-27b7eca](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-27b7eca) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`87d004fe99e0`](https://github.com/jackzhu119/research3/commit/87d004fe99e02df7d1c74bfdd7e4db3a9a37272b) |
| [distribution/paper3-hpc-offline-57cf4a4](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-57cf4a4) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`5eab3a5dbbce`](https://github.com/jackzhu119/research3/commit/5eab3a5dbbce1b3a6a3eddc5221d955bbfa3741a) |
| [distribution/paper3-hpc-offline-063f509](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-063f509) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`0675f025311d`](https://github.com/jackzhu119/research3/commit/0675f025311de660cfbed20060e08c096551d036) |
| [distribution/paper3-hpc-offline-71cec2d1de75-curl](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-71cec2d1de75-curl) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`19749d9feff9`](https://github.com/jackzhu119/research3/commit/19749d9feff943558f2b128d1a695d394b4208cc) |
| [distribution/paper3-hpc-offline-71cec2d1de75](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-71cec2d1de75) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`7703948216b2`](https://github.com/jackzhu119/research3/commit/7703948216b2842f0c490cfc068447c674c0055d) |
| [distribution/paper3-hpc-offline-7371f0c](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-7371f0c) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`81fd44418f17`](https://github.com/jackzhu119/research3/commit/81fd44418f1777b6be76157120f73501c5e72388) |
| [distribution/paper3-hpc-offline-e35f055](https://github.com/jackzhu119/research3/tree/distribution/paper3-hpc-offline-e35f055) | 学校 HPC 离线分发快照；按分支名中的包标识和原提交追溯。 | [`6444d6f3265f`](https://github.com/jackzhu119/research3/commit/6444d6f3265f4a672f05ad9f017bc0135cc0efd4) |

## 源码与结果怎么找

- 核心原始主线：默认分支 `p3work/`、`protocol/`；第1轮并行封装在 `paper3-parallel-migrations-20261025`。
- 第2至7轮代码：各轮进度分支的 `experiments/` 对应目录；这些分支同时承载运行摘要，不代表应把实验混成一个目录或总计数。
- 第8轮代码：`paper3-round8-20261025/experiments/followup_20261025_r8/`；轻量状态由 `paper3-r8-progress` 回传。
- 统一结果：`paper3-results/results/paper3/round-NN/`；`previous/` 是历史摘要，`automatic-summary/` 是独立分析摘要，`final/` 与 `LATEST.json` 为完整归档确认。目录是否存在应按本轮远端证据判断。

## 保留与入口维护

这些分支被运行中的监控、固定提交下载地址和历史恢复命令引用。此次整理通过分组目录、准确状态和清晰入口实现；不改分支名，不合并监控，不移动正在使用的源码或结果路径。后续新增轮次沿用独立源码、独立回传、独立结果分区，并在默认主页索引登记。

