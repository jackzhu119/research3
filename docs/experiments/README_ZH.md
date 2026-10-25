# 论文3八轮实验分类总档案

<!-- PAPER3_ROUND8_20261025 -->
## 第八轮：已发布，等待用户提交

第八轮基于第二、三、七轮已回传摘要，研究部分协方差输运、Bures/源度量几何和保持协方差的类别方向校正。固定计划 **2835项任务，其中24项元训练**；Stieger内部、Pan与Weibo分开分析，探索性重复验证，不预先声称提升。

- 资源：`hpc_gpu` A100 PCIe40GB，三个独立单卡作业，各16CPU/64G；合计48CPU/192G。
- [设计与公式](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/DESIGN_ZH.md) · [前轮依据](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/EVIDENCE_ZH.md) · [文献](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/LITERATURE.md) · [单行启动](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/LAUNCH_ZH.md)
- 无计算节点下载、全量test/pilot或累计12小时限制；保存双代训练断点与逐会话结果，独立GitHub回传。
- 进度分支`paper3-r8-progress`；摘要归档`paper3-results:results/paper3/round-08`。完整预测和模型仍在学校本地。
- 已通过学校核心软件版本的本地CPU验证，覆盖118项独立测试；线上raw/API完整包下载与安装核对通过。学校GPU现场尚未提交验证。

此前轮次的历史资源/计数记录保留，实际队列请看各轮最新观测。
<!-- /PAPER3_ROUND8_20261025 -->


<!-- PAPER3_PARALLEL_TARGETS_20261025 -->
## 2026-10-25：第一、五、六轮的最新并行迁移目标

本次目标为**每轮三个独立单卡Slurm作业**；不再沿用这些轮次之前的“一个作业内三卡”配置。学校现场尚未执行这次迁移，不能把下面的目标写成已运行资源。

| 轮次 | 新分区与GPU | 每作业CPU / 主机内存 | 本轮总资源 | 原作业 |
|---|---|---|---|---|
| 第一轮 | `hpc_gpu`，A100 PCIe 40GB | 各12CPU，内存43G / 43G / 42G | 3GPU、36CPU、128G | `770043`，原`hpc_gpu1`单卡 |
| 第五轮 | `hpc_gpu1`，A100 SXM4 80GB | 各12CPU、64G | 3GPU、36CPU、192G | `772012`，原`hpc_gpu` |
| 第六轮 | `hpc_gpu1`，A100 SXM4 80GB | 各12CPU、64G | 3GPU、36CPU、192G | `772013`，原`hpc_gpu` |

先确认三份带hold的新作业确实进入队列，再处理本轮旧监控、断点与旧作业，最后放行新作业；完成项和断点沿用原目录。**[完整交接与GitHub回传说明](../PARALLEL_MIGRATION_20261025_ZH.md)**。第四轮及第二、三、七轮的运行/结果归档不属于这次资源修改。后文旧资源表和异步进度快照保留为历史，不应覆盖本段最新目标。
<!-- /PAPER3_PARALLEL_TARGETS_20261025 -->


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


归档日期：2026-10-24（北京时间）。所有轮次均使用学校Slurm服务器，不再使用RunPod。本目录只整理文档和观测证据，不启动、取消或修改学校作业。

**共同主线：冻结EEGNet，使用新会话前20个已获取位置，在很少的空间校准参数下改善跨会话识别。** 已补入第七轮有限窗时空扩展与机制拆解。下面按研究问题、已实现算法、实验规模、资源与回传位置逐轮区分；已发布方案、软件测试、学校提交、实际完成与科学效果是不同状态。

## 1. 每轮研究了什么

| 轮次 | 主要研究问题 | 本轮算法/实验增量 | 详细档案 |
|---|---|---|---|
| 第一轮：原主线 | 小预算Lie空间校准能否优于已有适应方法？ | 无迹生成元指数映射P及PCT；与冻结、调头、全微调、协方差/切空间、加性basis、随机指数、LoRA系列比较 | [第一轮](ROUND1_ZH.md) |
| 第二轮：扩展V2 | 当前前缀证据支持多大、哪种空间调整？ | 固定双门、条件单门/双门、旋转与对称增益成分消融、PCT软化与正则控制；预算、误标和空间漂移诊断 | [第二轮](ROUND2_ZH.md) |
| 第三轮：信息与求解器 | 分类头竞争后，还有多少可辨识的空间信息？ | 解析Lie身份切线、Schur消头、信息有效维数及收缩半径、真实支持KL保护、匹配学习器字典 | [第三轮](ROUND3_ZH.md) |
| 第四轮：支持内交叉拟合 | 适应是否改善支持内没拟合过的位置预测？ | OOF源锚收缩、源/头/空间凸堆叠、one-SE候选选择、GCE稳健校准及消融 | [第四轮](ROUND4_ZH.md) |
| 第五轮：几何耦合 | 同样参数数目，旋转与相对增益怎样组合更合适？ | 联合指数/两种有序split/对称split；真实图精化、训练图×评价图交叉、共同θ与交换子探针 | [第五轮](ROUND5_ZH.md) |
| 第六轮：曲率预测平均 | 单点校准是否丢掉可用的空间—头不确定性？ | Schur形状9点平均、条件头/GH、同中心/同边际/硬标签空效应控制；新增Pan双会话外部评价 | [第六轮](ROUND6_ZH.md) |
| 第七轮：时空与可辨识性 | 静态图是否遗漏时间结构，头容量与风险目标怎样影响小支持适应？ | 有限窗守恒时空Lie、预算head与残差秩、source-relative entropic regret；rotation-only/temporal-only消融 | [第七轮](ROUND7_ZH.md) |

研究问题逐轮递进，不代表后一轮已被证明优于前一轮。新轮分别运行，前轮结果和作业独立保留。

## 2. 精确实验规模与统计口径

| 轮次 | 正式主矩阵 | 可选外部评价 | 如何理解数量 |
|---|---|---:|---|
| 第一轮 | 15神经源fit、10经典corefit及6附加bankfit；45元筛选＋10额外元确认；118开发配置＋28额外确认，1次内部检查 | 不按后续Weibo矩阵计数 | 包含不同阶段，不能把各阶段简单相加当成同成本任务；另5×100工程profile非正式科学fit |
| 第二轮 | 36核心元训练＋24敏感性元训练＋6参考＝**66** | 不在本66项内 | 60项元训练/复用；源EEGNet继续复用 |
| 第三轮 | 21字典＋78核心评价＋72敏感性＋3EA＝**174** | 另18 | 只有21项涉及字典；预算及108个压力条件嵌于评价任务 |
| 第四轮 | 21字典＋90核心评价＋90敏感性＋3EA＝**204** | 另21 | 只有21项涉及字典；108个压力条件嵌于评价任务 |
| 第五轮 | 12字典＋48交叉评价＋24半径＋15探针＋6参考＝**105** | 另21 | 只有12项训练字典；72个耦合压力条件嵌于12项核心评价 |
| 第六轮 | 6字典＋60内部核心＋48敏感性＋12参考＋72新Pan＝**198** | Weibo另36 | 126内部＋72预先固定的新外部；仅6项训练字典，Pan缓存已随包交付 |
| 第七轮 | 15字典＋72内部核心＋72敏感性＋21参考＋93重复Pan＝**273** | Weibo另60 | 180内部＋93重复外部；仅15项训练字典，108个人工条件嵌入评价 |

第三至七轮的174/204/105/198/273项均不是相应次数从头训练EEGNet。任务完成比例也不是剩余GPU时间比例。不能把所有轮数的任务加在一起当一条训练进度。

第二至六轮的核心预算曲线使用B4/8/12/16/20、同一个B20之后未来查询。主要比较以参与者为单位，种子/会话不当独立样本；第二至五轮固定Holm比较族分别为10/10/12/8；第六轮新Pan采用8比较×NLL/BA合计16项统一Holm。既有内部队列的继续探索不是全新前瞻确认，跨轮挑最好结果也不自动消除多重探索影响。

第七轮保持相同预算/未来查询合同，预定12个方法对×NLL/BA合计24项统一Holm。Pan沿用第六轮已准备的同一14人28会话，是重复外部探索，不是新独立盲队列；单轮校正不消除多轮选择影响。

## 3. 本次实际读取到的状态快照

前五轮归档于北京时间19:30:49，第六轮补充交付记录至20:36；下表每行都有自己的源更新时间，**不是同一时刻Slurm全景，也不表示之后永远保持该状态**。机器可读证据含源blob SHA：[STATUS_SNAPSHOT_20261024.json](STATUS_SNAPSHOT_20261024.json)。

| 轮次 | 源证据时间（北京，2026-10-24） | 可确认进度 | 解释 |
|---|---|---|---|
| 第一轮 | 12:40:59训练日志；另有稍后队列人工归档 | 15神经源fit已完成；B11/r2/ρ0.05/seed20261018 outer199 | 原轮已实际训练，整轮完成未确认；未确认持续自动回传 |
| 第二轮 | 18:17:15自动快照 | **23/66 COMPLETE**，报告PARTIAL，core_complete=false | 有完成结果，整轮未完成；具体正在运行任务须看更新后的回传 |
| 第三轮 | 18:55:27自动快照 | **6/174 COMPLETE**，控制器RUNNING，报告PARTIAL | ACTIVE可合并排队/运行，不凭该字段断言GPU正在优化 |
| 第四轮 | 19:22:43自动快照 | **0/204**；作业**771366**，PENDING，`(AssocGrpGRES)` | 已提交、因关联组GPU配额等待，不是“未提交”或算法报错 |
| 第五轮 | 19:21:41自动快照 | **0/105**；作业**771408**，AWAITING_ACCOUNTING | 已有提交记录、等待调度/记账确认，尚无实际训练证据 |
| 第六轮 | 20:36交付核验 | 349项软件测试、Pan已备齐、线上下载/离线安装读取通过 | 未提交学校作业，无第六轮效果数据 |
| 第七轮 | 21:48交付核验 | 526项本地CPU测试通过；源码183个Git blob逐一回读、raw206/API下载、bootstrap与禁网273项prepare/140视图逐位读取通过 | 未提交学校作业，无第七轮效果数据 |

本次未从这些部分快照得出任何“算法已显著提升”结论。各设计页交付时的“尚未提交”保留历史语境，以本表带时间证据了解后续部署。

## 4. 各轮服务器配置

| 轮次 | 分区与拓扑 | 每GPU CPU规划 | 主机内存 |
|---|---|---:|---|
| 第一轮 | `hpc_gpu1`，单GPU原作业770043，历史节点gpu4 | 24核 | 作业128G |
| 第二轮当前部署记录 | `hpc_gpu`，最多3个独立单GPU作业，可跨节点 | 12核 | 每作业64G，合计最多192G |
| 第三轮 | `hpc_gpu`，最多3个独立单GPU作业，可跨节点 | 12核 | 每作业64G，合计最多192G |
| 第四轮原提交配置 | `hpc_gpu1`，1节点1作业3GPU | 16核，总48核 | 作业共享192G（历史64G×3规划） |
| 第四轮迁移目标（包已核验，学校未执行） | **`hpc_gpu`，A100-PCIe-40GB，1节点1作业3GPU** | **12核，总36核** | **作业共享128G总量** |
| 第五轮原提交配置 | `hpc_gpu1`，1节点1作业3GPU | 12核，总36核 | 作业共享164G（历史配置） |
| 第五轮迁移目标（包已核验，学校未执行） | **`hpc_gpu`，A100-PCIe-40GB，1节点1作业3GPU** | **12核，总36核** | **作业共享128G总量** |
| 第六轮原发布配置 | `hpc_gpu1`，1节点1作业3GPU | 12核，总36核 | 作业共享164G（历史配置） |
| 第六轮迁移目标（包已核验，学校未执行） | **`hpc_gpu`，A100-PCIe-40GB，1节点1作业3GPU** | **12核，总36核** | **作业共享128G总量** |
| 第七轮 | **`hpc_gpu`，A100-PCIe-40GB，1节点1作业3GPU** | 12核，总36核 | **作业共享128G总量** |

第二轮早期24核/128G、16核方案、四卡方案和66元素整批提交是历史尝试，不能替换最新实际部署记录。第四轮原入口每槽`--memory 64G`，第五轮原入口整作业`--memory-total 164G`，均保留为历史语义；迁移入口统一请求总128G，不能照搬旧参数。`hpc_gpu1`是分区，`gpu1`是另一个节点名概念。

R4/R5/R6迁移后的每轮各占一个三卡提交作业，三轮不会合并为同一作业或混合GitHub分支，仍受GPU配额与同节点资源限制。发布/下载核验完成后，用户执行各自`MIGRATE --submit`才处理该轮同标记旧排队；当前记录不执行取消、提交或学校文件写入。

## 5. 源码、学校输出、自动回传分类

学校共享项目根目录：`/public/home/zhuziyuan/projects/paper3`。下表学校目录均相对此根目录。

| 轮次 | 代码/设计分支 | 学校输出 | GitHub汇总路径 |
|---|---|---|---|
| 第一轮 | 默认分支及原直接训练发布记录 | `runs/source_training_v2/`、`runs/meta/`、`runs/training_campaign_v1/` | 人工证据档案；未确认持续自动同步 |
| 第二轮 | `paper3-v2-progress` | `runs/followup_20261023_v2/` | `monitoring/paper3-followup-v2/status.json` |
| 第三轮 | `paper3-r3-progress` | `runs/followup_20261024_r3/` | `monitoring/paper3-round3/status.json` |
| 第四轮 | `paper3-r4-progress` | `runs/followup_20261024_r4/` | `monitoring/paper3-round4/status.json` |
| 第五轮 | `paper3-r5-progress` | `runs/followup_20261024_r5/` | `monitoring/paper3-round5/status.json` |
| 第六轮 | `paper3-r6-progress` | `runs/followup_20261024_r6/` | `monitoring/paper3-round6/status.json`（提交监控后生成） |
| 第七轮 | `paper3-r7-progress`（源码及包已发布并实际下载核验） | `runs/followup_20261024_r7/` | `monitoring/paper3-round7/status.json`（提交监控后生成） |

- [第二轮自动进度](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)
- [第三轮自动进度](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/monitoring/paper3-round3/status.json)
- [第四轮自动进度](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/monitoring/paper3-round4/status.json)
- [第五轮自动进度](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/status.json)

第二至六轮尝试每180秒回传汇总；网络失败不终止训练。完整checkpoint、逐会话预测与CSV在学校本地，进度上传不等于模型文件已完整异地备份。R3/R4/R5外测分别使用`weibo-status.json`与独立`_weibo`输出，未在本次核实其启动。

第七轮延续独立180秒汇总同步及本地保存；当前没有启动监控，不将规划中的进度路径当成已回传结果。源码/包/小bootstrap已线上真实下载与禁网安装读取核验；本地CPU和当前网络验证不等于学校GPU或学校网络已现场实测。

### 第四至六轮资源迁移交付（已核验，学校未执行）

独立操作分支为`paper3-hpc-gpu-migrations-20261024`，目录`operations/hpc_gpu_migrations_20261024/round4`、`round5`、`round6`。源码提交`79eabe6c456aa443e18c35c53b6c39a33d27a639`、小bootstrap提交`1d3e4860fc8a138e7be2717dae75ed55a0657e47`已发布；三轮各自raw206续传/API200完整下载、缓存禁网复用及bootstrap实际安装/prepare均通过。未取消学校旧排队，也未提交新作业。迁移只改变资源/提交封装，保留各轮算法、种子、矩阵、完成项、checkpoint和逐会话结果。

本地迁移全套测试R4/R5/R6分别198/266/347项通过。三包均在禁socket/DNS/子进程场景实际安装并prepare，旧计划字节及result/checkpoint/session的字节与mtime保留；R6另读28会话×5预算140个Pan视图。它们是本地交付证据，不是学校GPU/学校网络实测或运行成果。固定执行入口见[三轮迁移启动说明](https://github.com/jackzhu119/research3/blob/c7c903fb126e3e60a7e69556bea9f7ac8ce7e489/operations/hpc_gpu_migrations_20261024/LAUNCH_ZH.md)。启动说明固定提交`c7c903fb126e3e60a7e69556bea9f7ac8ce7e489`，三个RUN-ROUND脚本均已实际下载并完成shell安装/prepare；学校取消/提交仍为0。

第四、五轮继续使用已有项目缓存，不临时获取新原始数据；第六轮约23.4MB Pan缓存随包，计算节点不下载原MAT或安装环境。三轮分别回传`paper3-r4-progress:monitoring/paper3-round4/status.json`、`paper3-r5-progress:monitoring/paper3-round5/status.json`、`paper3-r6-progress:monitoring/paper3-round6/status.json`，不与操作分支合并进度。

## 6. 公共数据与连续运行记录

第三轮补充Weibo2014已公开左右手数据：10人，1580条有效LR，200条B20支持位置，1380条固定查询；原MAT与预处理一致性已核验，缓存和可移植包已发布。第四、五轮可选外测复用同一缓存，未另行增加一个已完成数据采集实验。原始物理采集顺序不可确认，只称“发布顺序LR子任务稳健性”，不称真实采集前20条的前瞻验证。

公共预处理与学校环境、历史故障和续跑机制详见 [COMMON_DATA_RUNTIME_ZH.md](COMMON_DATA_RUNTIME_ZH.md)。软件验证：第三轮131项、第四轮180项、第五轮248项本地测试通过；均不替代学校GPU实跑或科学效果。

无累计12小时总限额；保留Slurm单次作业时限及断点续交。完成项复用、训练双代断点、评价逐会话保存，单任务出错不取消所有独立任务。第一次读输入模型/缓存仍需必要I/O；不重复全量test/pilot/全数据扫描。启动器、checkpoint与调度的具体差异以各轮运行手册为准。

## 7. 归档范围与旧记录

默认分支本次仅修改分类文档与人工概览；第七轮源码、数学设计及复用缓存已在paper3-r7-progress独立交付，固定提交及实际下载证据见第七轮页。不覆盖前六轮自动回传、不改既有科学配方、不提交学校新作业。旧两轮首页与概览已保存到`docs/records/PAPER3_TWO_ROUND_*_20261024*`，共享聊天证据与学校环境页继续保留。

各轮可验证的效果最终来自完整配对报告，而不是设计长度、软件测试数、数学对称性或任务数量。

第六轮新增Pan2023：14人28会话3360条LR，560个B20支持位置、2800个固定查询；28原MAT全量取数及SHA通过，全部预处理逐位比对和禁网读取通过。新Pan是固定方法后的外部发布顺序会话前缀评价，不宣称真实物理采集顺序已验证。见[第六轮详情](ROUND6_ZH.md)。
