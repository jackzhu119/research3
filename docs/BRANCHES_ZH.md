# 论文3分支分类目录

分支审计：2026-10-25 16:28（北京时间）；共 **27** 个分支。默认分支为 `research/paper3-protocol-v2-w0`。这是审计时快照，各进度分支随后仍会更新。原有分支和固定下载提交保持可访问。

## 默认主页与核心协议（1）

默认分支保留核心算法、数据合同、协议与总导航。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [research/paper3-protocol-v2-w0](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0) | 默认主页、核心 p3work/ 与 protocol/、八轮实验总索引。 | [`1c5d2bc01eaf`](https://github.com/jackzhu119/research3/commit/1c5d2bc01eaf5e2ad381d08b7949cb9be1b590e4) |

## 独立实验发布（2）

第八、九轮各自独立发布；第二至七轮代码和设计位于各轮进度分支 experiments/ 子目录。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [paper3-round8-20261025](https://github.com/jackzhu119/research3/tree/paper3-round8-20261025) | 第八轮部分协方差输运、Bures 几何与保持协方差的类别方向校正；experiments/followup_20261025_r8/。 | [`8f922f765c0c`](https://github.com/jackzhu119/research3/commit/8f922f765c0c8f87a4aba783cfd58c700ad73af2) |
| [paper3-round9-20261025](https://github.com/jackzhu119/research3/tree/paper3-round9-20261025) | 第九轮源残差GP、支持信息局部锚定；1548项设计与开发验证，未提交学校GPU作业。 | [`a6563a9e20b5`](https://github.com/jackzhu119/research3/commit/a6563a9e20b5f3cbf3249cd06714d3a179c45311) |

## 按轮实时回传与对应实验源码（8）

原监控持续写入这些分支；不要合并、重命名或删除。状态文件是异步快照，报告与任务计数可能不同步。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [paper3-r1-progress](https://github.com/jackzhu119/research3/tree/paper3-r1-progress) | 第1轮并行续跑状态；monitoring/paper3-round1/status.json。 | [`e6ce0aeb76b9`](https://github.com/jackzhu119/research3/commit/e6ce0aeb76b9f0ae3fe794f069aa795e63534e18) |
| [paper3-r3-progress](https://github.com/jackzhu119/research3/tree/paper3-r3-progress) | 第3轮代码 experiments/followup_20261024_r3/；状态 monitoring/paper3-round3/status.json。 | [`c79a092eb243`](https://github.com/jackzhu119/research3/commit/c79a092eb243b9a8eefe7a7fdc14990fe1ffe0dd) |
| [paper3-r4-progress](https://github.com/jackzhu119/research3/tree/paper3-r4-progress) | 第4轮代码 experiments/followup_20261024_r4/；状态与资源交接记录。 | [`dcde34685dfa`](https://github.com/jackzhu119/research3/commit/dcde34685dfa36166e6f2a2d1380557c29253537) |
| [paper3-r5-progress](https://github.com/jackzhu119/research3/tree/paper3-r5-progress) | 第5轮代码 experiments/followup_20261024_r5/；状态与三作业交接记录。 | [`8b64443cc2c3`](https://github.com/jackzhu119/research3/commit/8b64443cc2c3b597352064cd4580a6275758dae2) |
| [paper3-r6-progress](https://github.com/jackzhu119/research3/tree/paper3-r6-progress) | 第6轮代码及 Pan 缓存 experiments/followup_20261024_r6/；状态与三作业交接记录。 | [`aa9a3d5cea3e`](https://github.com/jackzhu119/research3/commit/aa9a3d5cea3ed525f3682c2d07fdf1097a7e7ab8) |
| [paper3-r7-progress](https://github.com/jackzhu119/research3/tree/paper3-r7-progress) | 第7轮时空与可辨识性实验源码、缓存与异步摘要；experiments/followup_20261024_r7/。 | [`8b0f721b70f2`](https://github.com/jackzhu119/research3/commit/8b0f721b70f27903289c084d5877bad716930347) |
| [paper3-r8-progress](https://github.com/jackzhu119/research3/tree/paper3-r8-progress) | 第8轮三个单卡作业的轻量运行状态；实验实现从独立第8轮发布分支读取。 | [`e32d75b5ddee`](https://github.com/jackzhu119/research3/commit/e32d75b5ddeebdbd3bd703e42c1a5086e04f32b9) |
| [paper3-v2-progress](https://github.com/jackzhu119/research3/tree/paper3-v2-progress) | 第2轮代码 experiments/followup_20261023/；状态 monitoring/paper3-followup-v2/status.json。 | [`988e13150c80`](https://github.com/jackzhu119/research3/commit/988e13150c80f3e96c1d58ffd884cb2061c26b1c) |

## 按轮结果分类与归档（1）

摘要、历史快照、完整结果归档分别标识；完整归档以 LATEST.json 及对应 manifest/Release 分片为准。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [paper3-results](https://github.com/jackzhu119/research3/tree/paper3-results) | 统一结果分区 results/paper3/round-01/ 至 round-08/；原始回传、自动摘要和完整归档按实际存在分开。 | [`1aa6851bb0fa`](https://github.com/jackzhu119/research3/commit/1aa6851bb0fad2b920c299e3f9155c0c71cb81dd) |

## 运行与资源迁移（2）

封装、调度和迁移入口；固定提交下载地址被历史命令引用，应保留。

| 分支 | 用途 | 审计时提交 |
|---|---|---|
| [paper3-hpc-gpu-migrations-20261024](https://github.com/jackzhu119/research3/tree/paper3-hpc-gpu-migrations-20261024) | 第4至6轮迁往 hpc_gpu 的早期单节点三卡交付；第5、6轮最新部署已由10月25日方案取代。 | [`cef9fc877338`](https://github.com/jackzhu119/research3/commit/cef9fc8773389df29fef015297c1b203f39cdefd) |
| [paper3-parallel-migrations-20261025](https://github.com/jackzhu119/research3/tree/paper3-parallel-migrations-20261025) | 第1轮 hpc_gpu 与第5、6轮 hpc_gpu1 三个独立单卡作业方案；operations/parallel_20261025/。 | [`491c87a249ed`](https://github.com/jackzhu119/research3/commit/491c87a249ed7a61ca0eb925bc3f6d743c986d13) |

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

结果的论文数据、聚合JSON补传、科学完成、完整归档分别记录。第九轮源码发布不等于已提交学校任务；未来运行时才会建立其进度分支。
[主页](../README.md) · [九轮实验索引](experiments/README_ZH.md) · [机器目录](BRANCH_INDEX.json)
