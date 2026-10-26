# 论文3 · 九轮实验分类总索引

[仓库主页](../../README.md) · [最新进度与上传状态](../LIVE_STATUS_ZH.md) · [全部分支](../BRANCHES_ZH.md) · [后续安排](../ROADMAP_ZH.md)

共同主线：冻结已有EEGNet，使用新会话少量已获取前缀试次，研究空间/时空校准、参数可辨识性与有限支持下的泛化。所有实验使用学校Slurm服务器。

## 研究方向与规模

| 轮次 | 研究内容 | 计划 | 详情 |
|---|---|---|---|
| 1 | 原论文主线：Lie空间校准P/PCT与完整基线 | 分阶段计划：15源神经fit、45元筛选、10额外元确认、开发/确认与扩展 | [第1轮](ROUND1_ZH.md) |
| 2 | 空间成分门控与可靠性保护 | 66：36核心＋24敏感性＋6参考 | [第2轮](ROUND2_ZH.md) |
| 3 | 信息几何、解析切线与Schur消头 | 174：21字典＋78核心＋72敏感性＋3参考；可选外测另18 | [第3轮](ROUND3_ZH.md) |
| 4 | 支持内交叉拟合、收缩与稳健堆叠 | 204：21字典＋90核心＋90敏感性＋3参考；可选外测另21 | [第4轮](ROUND4_ZH.md) |
| 5 | 旋转与增益的几何耦合 | 105：12字典＋48交叉评价＋24半径＋15探针＋6参考；可选外测另21 | [第5轮](ROUND5_ZH.md) |
| 6 | 曲率形状预测平均与Pan外部评价 | 198：6字典＋60内部核心＋48敏感性＋12参考＋72Pan；Weibo另36 | [第6轮](ROUND6_ZH.md) |
| 7 | 有限窗时空Lie、头容量与源相对风险 | 273：15字典＋72内部核心＋72敏感性＋21参考＋93重复Pan；Weibo另60 | [第7轮](ROUND7_ZH.md) |
| 8 | 部分协方差输运与保持协方差的类别方向旋转 | 2835：3类参考＋24元训练＋207核心＋792压力＋27标签空效应＋1566敏感性＋216旋转敏感性 | [第8轮](ROUND8_ZH.md) |

第2、3、7轮主计划已有本地完成证据；第1、4轮仍有运行工作；第5、6、8轮已提交并在等待配额。该句为本次整理快照，[最新汇总](../LIVE_STATUS_ZH.md)会继续跟随回传更新。

## 当前有效资源与结果入口

| 轮次 | 分区 | 提交方式与资源 | 代码 | 回传 | 结果 |
|---|---|---|---|---|---|
| 1 | `hpc_gpu` | 3个独立单卡作业；各12CPU，内存43G/43G/42G，总36CPU/128G | [代码](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0/p3work) | [状态](https://github.com/jackzhu119/research3/blob/paper3-r1-progress/monitoring/paper3-round1/status.json) | [结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-01) |
| 2 | `hpc_gpu` | 最多3个独立单卡作业；各12CPU/64G | [代码](https://github.com/jackzhu119/research3/tree/paper3-v2-progress/experiments/followup_20261023) | [状态](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json) | [结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-02) |
| 3 | `hpc_gpu` | 最多3个独立单卡作业；各12CPU/64G | [代码](https://github.com/jackzhu119/research3/tree/paper3-r3-progress/experiments/followup_20261024_r3) | [状态](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/monitoring/paper3-round3/status.json) | [结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-03) |
| 4 | `hpc_gpu` | 1节点1作业3GPU；总36CPU/128G共享 | [代码](https://github.com/jackzhu119/research3/tree/paper3-r4-progress/experiments/followup_20261024_r4) | [状态](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/monitoring/paper3-round4/status.json) | [结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-04) |
| 5 | `hpc_gpu1` | 3个独立单卡作业；各12CPU/64G，总36CPU/192G | [代码](https://github.com/jackzhu119/research3/tree/paper3-r5-progress/experiments/followup_20261024_r5) | [状态](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/three-jobs-status.json) | [结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-05) |
| 6 | `hpc_gpu1` | 3个独立单卡作业；各12CPU/64G，总36CPU/192G | [代码](https://github.com/jackzhu119/research3/tree/paper3-r6-progress/experiments/followup_20261024_r6) | [状态](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/monitoring/paper3-round6/three-jobs-status.json) | [结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-06) |
| 7 | `hpc_gpu` | 1节点1作业3GPU；总36CPU/128G共享 | [代码](https://github.com/jackzhu119/research3/tree/paper3-r7-progress/experiments/followup_20261024_r7) | [状态](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/monitoring/paper3-round7/status.json) | [结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-07) |
| 8 | `hpc_gpu` | 3个独立单卡作业；各16CPU/64G，总48CPU/192G | [代码](https://github.com/jackzhu119/research3/tree/paper3-round8-20261025/experiments/followup_20261025_r8) | [状态](https://github.com/jackzhu119/research3/blob/paper3-r8-progress/monitoring/paper3-round8/three-jobs-status.json) | [结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-08) |

资源表表示每轮配置；排队时不代表这些GPU已分配。`AssocGrpGRES`是关联组GPU配额限制，增加提交数量不会自动获得更多同时运行的GPU。

## 数据、复现和分类口径

- [共同数据、环境与历史故障](COMMON_DATA_RUNTIME_ZH.md) · [学校环境](../SCHOOL_HPC_ENVIRONMENT_ZH.md)。
- 原学校根目录：`/public/home/zhuziyuan/projects/paper3`；既有离线环境、源模型和数据缓存继续复用。
- Stieger内部、Pan重复外部与Weibo发布顺序稳健性分开报告。Pan/Weibo发布顺序不能直接宣称真实物理采集顺序；多轮探索不视为新的独立盲测。
- 不把任务数当同成本完整EEGNet重训次数，不把控制器心跳当模型更新计数，不把软件测试当算法效果证明。
- [2026-10-24状态留档](STATUS_SNAPSHOT_20261024.json) · [整理前页面原文](../records/20261025-before-reorganization/README_ZH.md)。

## 第九轮：从已有结果继续优化

[设计与执行包](ROUND9_ZH.md)：1548项有限任务，源预测残差GP、信息局部锚定、强基线及机制消融；3个独立单卡hpc_gpu作业，每作业16CPU/64G。当前仅设计及开发环境验证，未提交学校GPU。


## 第十轮：证据保留的查询函数校准（2026-10-26新增）

[完整入口](ROUND10_ZH.md)：原始CSV/预测档案与文献核查、数学推导、18方法消融、15→165→57分阶段计划、四作业代码与离线启动。每作业1GPU/16CPU/68G；CPU验证，不代表真实EEG效果或GPU实验已完成。该新增记录不改写上面原轮次的历史状态。
