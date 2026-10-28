# 论文3 · 学校HPC实验与结果

围绕**少量新会话前缀试次下的EEG空间/时空校准**，保存原论文主线与九轮算法探索、运行证据和分轮结果。当前计算平台为学校Slurm服务器。

**[最新进度](docs/LIVE_STATUS_ZH.md) · [九轮实验分类](docs/experiments/README_ZH.md) · [全部分支](docs/BRANCHES_ZH.md) · [结果分区](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3) · [后续安排](docs/ROADMAP_ZH.md)**

[论文报告优先上传：不等待完整备份](operations/paper_results_upload/README_ZH.md) · [后续所有轮次双通道上传约定](docs/RESULT_UPLOAD_POLICY_ZH.md)

[第一轮拆分错误修复：实际SXM4-40GB，沿用350断点](operations/round1_repair_20261025/README_ZH.md)

[第七轮完整总JSON定向补传](operations/round7_report_supplement_20261025/README_ZH.md) · [第九轮：实测依据、数学与代码](docs/experiments/ROUND9_ZH.md)（已设计，未提交学校任务）

## 各轮最新回传

<!-- PAPER3_LIVE_STATUS_START -->

| 轮次 | 最近确认状态 | 任务完成 | 队列观测 | 来源时间（北京） | 论文报告 | 完整结果包 |
|---|---|---|---|---|---|---|
| [第1轮](https://github.com/jackzhu119/research3/blob/paper3-r1-progress/monitoring/paper3-round1/status.json) | 阶段记录：主线冻结与内部评价 | 分阶段，不合并百分比 | 3运行 / 0排队 | 10-28 20:58:26 | 尚未发布论文报告回执 | 未见完整上传完成回执 |
| [第2轮](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json) | 任务完成；快照滞后 | 66/66 | 本源未查询队列 | 10-25 10:16:55 | [论文报告已上传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-02/PAPER_DATA_ZH.md) | 上传完成回执已发布 |
| [第3轮](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/monitoring/paper3-round3/status.json) | 任务完成；快照滞后 | 174/174 | 本源未查询队列 | 10-25 08:09:10 | [论文报告已上传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-03/PAPER_DATA_ZH.md) | 上传完成回执已发布 |
| [第4轮](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/monitoring/paper3-round4/status.json) | 任务完成；快照滞后 | 204/204 | 本源未查询队列 | 10-25 16:46:54 | 尚未发布论文报告回执；读取失败，状态待核实 | 上传完成回执已发布 |
| [第5轮](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/three-jobs-status.json) | 任务完成 | 105/105 | 0运行 / 0排队 | 10-28 20:59:16 | 尚未发布论文报告回执；读取失败，状态待核实 | 上传完成回执已发布 |
| [第6轮](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/monitoring/paper3-round6/three-jobs-status.json) | 任务完成 | 198/198 | 0运行 / 0排队 | 10-28 20:59:33 | 尚未发布论文报告回执；读取失败，状态待核实 | 上传完成回执已发布 |
| [第7轮](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/monitoring/paper3-round7/status.json) | 任务完成；快照滞后 | 273/273 | 本源未查询队列 | 10-25 09:03:06 | [论文报告已上传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-07/PAPER_DATA_ZH.md)；[聚合JSON已补传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-07/REPORT_JSON_ZH.md) | 上传完成回执已发布 |
| [第8轮](https://github.com/jackzhu119/research3/blob/paper3-r8-progress/monitoring/paper3-round8/three-jobs-status.json) | 任务完成 | 2835/2835 | 0运行 / 0排队 | 10-28 20:59:12 | 尚未发布论文报告回执；读取失败，状态待核实 | 未见完整上传完成回执 |
| [第9轮](https://github.com/jackzhu119/research3/blob/paper3-r9-progress/monitoring/paper3-round9/three-jobs-status.json) | 任务完成 | 1548/1548 | 0运行 / 0排队 | 10-28 21:00:03 | [论文报告已上传](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-09/PAPER_DATA_ZH.md) | 上传完成回执已发布 |

[完整状态、观测时间与上传口径](docs/LIVE_STATUS_ZH.md)

<!-- PAPER3_LIVE_STATUS_END -->

学校监控约180秒尝试更新各轮原分支；主页每30分钟由GitHub Actions尝试汇总。来源快照、任务完成、报告完成和完整包上传分开显示；[刷新任务与运行记录](https://github.com/jackzhu119/research3/actions/workflows/refresh-experiment-overview.yml)。

## 九轮分别做什么

| 轮次 | 方向 | 计划规模 | 入口 |
|---|---|---|---|
| 1 | 原论文主线：Lie空间校准P/PCT与完整基线 | 分阶段完整主线 | [第1轮](docs/experiments/ROUND1_ZH.md) |
| 2 | 空间成分门控与可靠性保护 | 66 | [第2轮](docs/experiments/ROUND2_ZH.md) |
| 3 | 信息几何、解析切线与Schur消头 | 174 | [第3轮](docs/experiments/ROUND3_ZH.md) |
| 4 | 支持内交叉拟合、收缩与稳健堆叠 | 204 | [第4轮](docs/experiments/ROUND4_ZH.md) |
| 5 | 旋转与增益的几何耦合 | 105 | [第5轮](docs/experiments/ROUND5_ZH.md) |
| 6 | 曲率形状预测平均与Pan外部评价 | 198 | [第6轮](docs/experiments/ROUND6_ZH.md) |
| 7 | 有限窗时空Lie、头容量与源相对风险 | 273 | [第7轮](docs/experiments/ROUND7_ZH.md) |
| 8 | 部分协方差输运与保持协方差的类别方向旋转 | 2835，其中24元训练 | [第8轮](docs/experiments/ROUND8_ZH.md) |
| 9 | 支持信息约束的源残差GP与局部预测锚定 | 1548，复用冻结源模型；尚未提交 | [第9轮](docs/experiments/ROUND9_ZH.md) |

任务规模包含不同成本的训练、评价、参考与敏感性工作，不能按数量推算训练时间；各轮原始矩阵和可选外测范围见详细页。第八轮研究协方差输运与保持协方差的类别旋转，第九轮研究冻结特征中的源残差GP与支持信息锚定；新候选均无效果提升结论。

## 代码、运行、结果在哪里

| 要找的内容 | 入口 |
|---|---|
| 核心方法与原始协议 | [p3work/](p3work/) · [protocol/](protocol/) · [输入合同](contracts/) |
| 第2至7轮代码与设计 | [各轮源码导航](docs/experiments/README_ZH.md) |
| 第9轮源码、公式和在线包 | [第9轮独立发布](https://github.com/jackzhu119/research3/tree/paper3-round9-20261025/experiments/followup_20261025_r9) |
| 第8轮源码、公式和在线包 | [第8轮独立发布](https://github.com/jackzhu119/research3/tree/paper3-round8-20261025/experiments/followup_20261025_r8) |
| 当前各轮资源和输出目录 | [九轮配置表](docs/experiments/README_ZH.md#当前有效资源与结果入口) |
| 全部27个分支及历史版本用途 | [分支分类目录](docs/BRANCHES_ZH.md) |
| 已同步摘要与完整结果归档 | [paper3-results分轮目录](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3) |
| 数据、环境和历史故障 | [共同运行约定](docs/experiments/COMMON_DATA_RUNTIME_ZH.md) · [学校环境](docs/SCHOOL_HPC_ENVIRONMENT_ZH.md) |
| 后续任务与结果分析 | [路线与待办](docs/ROADMAP_ZH.md) |
| 整理前页面和来源证据 | [历史原文](docs/records/20261025-before-reorganization/README_ZH.md) · [本次核对记录](docs/records/STATUS_REVIEW_20261025.json) |

## 当前资源口径

`hpc_gpu`为A100 40GB资源组（第一轮实际回执为SXM4-40GB，以作业观测为准），`hpc_gpu1`为A100 SXM4 80GB资源组；节点`gpu1`与分区`hpc_gpu1`不同。第一轮为hpc_gpu三个独立单卡作业，总36CPU/128G；第四轮为hpc_gpu单作业内三卡，总36CPU/128G；第五、六轮为hpc_gpu1各三个独立单卡作业，每作业12CPU/64G；第八、九轮均为hpc_gpu三个独立单卡作业，每作业16CPU/64G。实际可同时运行数量以学校配额与队列为准。

结果先保存在学校共享目录`/public/home/zhuziyuan/projects/paper3/runs/`。完成项与断点继续复用；各轮监控和上传网络故障独立于训练。**摘要已回传不等于完整预测或checkpoint已备份到GitHub**；完整归档以各轮上传器发布的LATEST.json及引用manifest为准。

研究结论以参与者配对比较、效应区间和预设多重比较为依据；探索性分析与后续独立确认分开。
