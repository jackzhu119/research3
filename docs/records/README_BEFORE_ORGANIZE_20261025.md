# 论文3 · 学校HPC七轮实验记录

<!-- PAPER3_RESULTS_ENTRY_20261025 -->
## 2026-10-25：已完成结果与分类归档入口

**[进入按轮次分类的结果总归档](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/README_ZH.md)**。

用户在2026-10-25贴回的学校本地只读诊断已确认：**第二轮66/66、第三轮174/174、第七轮273/273任务全部完成**。这项本地证据更新了完成状态；下方2026-10-24的计数、迁移准备与未提交说明继续作为带时间的历史记录保留，不应当作当前调度状态。

| 轮次 | 分类结果分区 | 完整上传确认入口 |
|---|---|---|
| 第二轮 | [第二轮结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-02) | [LATEST.json](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-02/LATEST.json) |
| 第三轮 | [第三轮结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-03) | [LATEST.json](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-03/LATEST.json) |
| 第七轮 | [第七轮结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-07) | [LATEST.json](https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-07/LATEST.json) |

`previous/<source_commit>/`保留此前已经回传的监控快照及来源校验；`final/<snapshot_id>/`保存学校完成后上传的结果。**本地完成不等于GitHub完整上传：以各轮上传器校验后发布的`LATEST.json`及其引用清单为准；入口尚不存在时，表示完整上传尚未确认。** 旧监控的任务计数与耗时汇总可能不同步，不能用延迟快照覆盖本地完成证据。

本次分类不更改算法配方、学校资源配置或运行作业；原自动进度分支和历史提交保留。
<!-- /PAPER3_RESULTS_ENTRY_20261025 -->


七轮的研究问题、算法公式、实验数量、服务器配置、断点与GitHub回传已分类归档。全部使用学校Slurm云服务器，不再使用RunPod。第七轮526项本地CPU软件测试、真实线上下载与禁网安装/273项prepare/140会话预算视图读取通过；尚未提交学校作业。前六轮状态保持各自原快照。

**[进入七轮完整分类总档案](docs/experiments/README_ZH.md)** · [数据/环境/历史故障与连续运行](docs/experiments/COMMON_DATA_RUNTIME_ZH.md) · [前六轮源证据快照](docs/experiments/STATUS_SNAPSHOT_20261024.json)

| 轮次 | 研究内容 | 计划规模 | 学校配置与迁移目标 | 详细记录 |
|---|---|---|---|---|
| 第一轮 | 原Lie空间校准P/PCT及完整基线比较 | 15源神经fit；45元筛选＋10额外元确认；开发/内部检查分阶段 | hpc_gpu1，原单卡作业24CPU/128G | [第一轮](docs/experiments/ROUND1_ZH.md) |
| 第二轮V2 | 空间成分门控、可靠性保护、未来run稳定性 | **66**：36核心＋24敏感性＋6参考 | hpc_gpu，最多3独立单卡作业，各12CPU/64G | [第二轮](docs/experiments/ROUND2_ZH.md) |
| 第三轮 | 解析切线、Schur消头、信息半径、支持KL保护 | **174**：21字典＋78核心＋72敏感性＋3EA；外测另18 | hpc_gpu，最多3独立单卡作业，各12CPU/64G | [第三轮](docs/experiments/ROUND3_ZH.md) |
| 第四轮 | OOF收缩、凸堆叠、one-SE选择、GCE稳健校准 | **204**：21字典＋90核心＋90敏感性＋3EA；外测另21 | **迁移目标hpc_gpu，1节点1作业3GPU，总36CPU/128G**；原hpc_gpu1为48CPU/192G | [第四轮](docs/experiments/ROUND4_ZH.md) |
| 第五轮 | 旋转—相对增益耦合、真实图精化、共同θ机制探针 | **105**：12字典＋48交叉评价＋24半径＋15探针＋6参考；外测另21 | **迁移目标hpc_gpu，1节点1作业3GPU，总36CPU/128G**；原hpc_gpu1为36CPU/164G | [第五轮](docs/experiments/ROUND5_ZH.md) |
| 第六轮 | 曲率引导空间预测平均、同中心/同边际机制、新Pan外部验证 | **198**：6字典＋60内部核心＋48敏感性＋12参考＋72Pan；Weibo另36 | **迁移目标hpc_gpu，1节点1作业3GPU，总36CPU/128G**；原hpc_gpu1为36CPU/164G | [第六轮](docs/experiments/ROUND6_ZH.md) |
| 第七轮 | 有限窗时空Lie、头容量可辨识性、源相对风险与成分消融 | **273**：15字典＋72内部核心＋72敏感性＋21参考＋93重复Pan；Weibo另60 | **hpc_gpu，A100-PCIe-40GB，1节点1作业3GPU，总36CPU/128G共享** | [第七轮](docs/experiments/ROUND7_ZH.md) |

第四、五、六轮已追加统一资源迁移计划：`hpc_gpu`、A100-PCIe-40GB、单节点单作业3GPU、总36CPU和128G共享主机内存。三包已真实发布，raw206续传/API完整下载、小bootstrap及禁网安装/prepare验证通过；**尚未取消或提交学校作业**。原配置和旧状态为历史快照；用户执行对应`MIGRATE --submit`后才处理本轮同标记旧排队，保留结果及GitHub分轮回传。见[三轮迁移启动说明](https://github.com/jackzhu119/research3/blob/c7c903fb126e3e60a7e69556bea9f7ac8ce7e489/operations/hpc_gpu_migrations_20261024/online/LAUNCH_ZH.md)。

各轮任务成本不同；174/204/105不是从头重训EEGNet的次数。候选算法尚需完整配对实验检验，不能由数学性质或软件测试宣称显著提升。

## 已核实的进度记录

归档日期：2026-10-24。以下为**各自更新时间的异步快照**，不是实时调度页面；查看最新状态请点对应自动回传。

| 轮次 | 本次读取证据（北京时间） | 最新可确认情况 | 回传/记录 |
|---|---|---|---|
| 第一轮 | 10月24日12:40:59训练日志，另有后续队列归档 | 15源神经fit完成；B11/r2/ρ.05/seed18 outer199；整轮未确认完成 | [人工证据](docs/records/PAPER3_SHARED_CHAT_EVIDENCE_20261024.json) |
| 第二轮 | 10月24日18:17:15 | **23/66 COMPLETE**，PARTIAL | [自动进度](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json) |
| 第三轮 | 10月24日18:55:27 | **6/174 COMPLETE**，控制器RUNNING、PARTIAL | [自动进度](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/monitoring/paper3-round3/status.json) |
| 第四轮 | 10月24日19:22:43 | **771366已提交**，PENDING / AssocGrpGRES，0/204 | [自动进度](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/monitoring/paper3-round4/status.json) |
| 第五轮 | 10月24日19:21:41 | **771408已有提交记录**，AWAITING_ACCOUNTING，0/105 | [自动进度](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/status.json) |
| 第六轮 | 10月24日20:36交付核验 | 设计、数据和线上包就绪；**未提交学校作业**，无效果结果 | [固定启动指令](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/online/LAUNCH_ZH.md) |
| 第七轮 | 10月24日21:48交付核验 | **526项本地CPU软件测试、raw续传/API下载、bootstrap及禁网273项prepare/140视图逐位读取通过**；未提交学校作业，无效果结果 | [设计与交付状态](docs/experiments/ROUND7_ZH.md) |

“已提交”不等于GPU已开始优化；实际outer/session计数增长才说明计算推进。第四、五轮交付时旧页面的“尚未提交”早于这些后续回传，应按时间解释。

## 结果、代码与续跑入口

- [七轮总索引：各轮源码、输出目录、精确矩阵与统计口径](docs/experiments/README_ZH.md)
- [学校环境、数据复用、连续运行与历史故障处理](docs/experiments/COMMON_DATA_RUNTIME_ZH.md)
- [原分享聊天配置交接](docs/PAPER3_CONFIGURATION_HANDOFF_20261024_ZH.md) · [原学校环境记录](docs/SCHOOL_HPC_ENVIRONMENT_ZH.md)
- [机器可读七轮概览](docs/EXPERIMENT_STATUS.json) · [旧两轮首页留档](docs/records/PAPER3_TWO_ROUND_README_20261024_ZH.md)

结果先保存学校共享项目`/public/home/zhuziyuan/projects/paper3/runs/`。第二至七轮设计每180秒尝试独立GitHub汇总同步；第七轮尚未启动监控，网络失败不停止训练。完整checkpoint、逐会话预测和CSV仍保存在学校本地，进度同步不等于模型完整异地备份。

各轮保留独立输出、完成项复用与断点；学校启动不反复重跑全量test/pilot/数据树扫描，没有累计12小时总上限。集群单次作业时限仍存在，由各轮保存/续交机制处理。

默认分支本次仅更新分类文档与人工概览；第七轮程序和复用Pan缓存已在独立分支发布，固定源码/下载证据及启动说明见第七轮页。既有状态快照、训练程序、科学配方与学校运行作业保持原状，不覆盖其自动回传。R4/R5/R6迁移脚本已在`paper3-hpc-gpu-migrations-20261024`分支发布并验证，资源在学校尚未实际切换，运行成果仍待用户提交后观测。
