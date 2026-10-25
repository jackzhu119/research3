# 第一轮：原论文3主线与完整基线比较

<!-- PAPER3_PARALLEL_TARGET_20261025 -->
## 2026-10-25最新资源目标（现场尚未执行）

本次最新目标为 `hpc_gpu` 的A100 PCIe 40GB：三个独立单卡作业，各12CPU，内存43G/43G/42G，合计36CPU/128G。原`770043`仍在`hpc_gpu1`的SXM4 80GB上运行；本记录不声称已经完成现场切换。

三卡优先并行未完成MetaJob；已完成源模型、经典模型、筛选结果和既有checkpoint沿用原目录，不复制重训。开发校准/计时工作固定slot0；两条原五源扩展`BASE4_PLUS_YANG`、`BASE4_PLUS_DREYER`继续保留，后期分配slot1/2并行。主线完成与全部原工作完成分别回传，不能用一个总百分比代替。

新增第一轮独立回传：`paper3-r1-progress:monitoring/paper3-round1/status.json`。按轮汇总输出进入`paper3-results`，不冒称完整预测/模型已自动备份。

[三作业交接、在线入口与回传配置](../PARALLEL_MIGRATION_20261025_ZH.md)。本段之后的旧资源和进度记录保留历史。
<!-- /PAPER3_PARALLEL_TARGET_20261025 -->


[返回五轮总览](README_ZH.md) · [第二轮](ROUND2_ZH.md)

本页归档原论文3主线，不把最早的24项扩展方案误标为第一轮。
资料整理日期为2026-10-24；进度采用下文注明时间的证据，不能代替当前Slurm查询。

## 1. 本轮要回答什么

在只获取目标新会话前20个真实运动想象试次时，能否预先学到值得调整的少量空间方向，使冻结的EEGNet优于仅调分类头及自由空间适应？
支持集按真实采集顺序定义，后续完整物理运行段用于预测；不改成同一会话随机切分。
无效试次仍消耗前20次采集预算，不免费补到20个好样本。

源模型和参考协方差来自BNCI2014_001、Cho2017、Lee2019_MI、PhysionetMI四个源库。
Stieger2021沿用40名元训练、11名元验证、11名内部留出参与者的冻结划分。
基础输入为11通道、CAR、128Hz、8–30Hz及已冻结的窗口/归一化合同；具体有效名单和窗口以实际manifest为准。

## 2. 主算法与公式

令固定CAR基为 \(Q\in\mathbb R^{10\times11}\)，满足 \(QQ^\top=I\)、\(Q^\top Q=H\)。
源参考为 \(R_{ref}\)，学习的低维生成元字典为 \(G_1,\ldots,G_r\)：

\[
M_\theta=\sum_{k=1}^r\theta_kG_k,\quad
\operatorname{tr}(G_k)=0,\quad
\langle G_k,G_l\rangle_F=\delta_{kl},\quad \|\theta\|_2\le\rho.
\]

\[
A_\theta=Q^\top R_{ref}^{1/2}\exp(M_\theta)R_{ref}^{-1/2}Q.
\]

源EEGNet及其BN统计量冻结。目标会话只用支持集更新r维系数和线性头残差；每个会话重新初始化。
外层利用元训练参与者的未来query学习字典，元验证选择配置和checkpoint。
字典实现使用99维无迹基及Stiefel参数 \(B\in\mathbb R^{99\times r}\)。

在白化后的10维有效坐标，\(\det\exp(M)=1\)、\(\kappa_2(\exp(M))\le e^{2\rho}\)。
11维CAR算子本来秩亏，不能把这两个结论直接写到原始11维电压算子上。
r是系数数目，不是 \(\exp(M)-I\) 的矩阵秩。
这些代数性质不保证准确率提升，也不代表学到了真实电极位移。

数学定义见[原方法规格](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/protocol/METHOD_SPEC.md)。

## 3. 比较方法分类

| 类别 | 方法编号 | 做什么 |
|---|---|---|
| 冻结参考 | B0 | 不适应的源EEGNet |
| 头与全模型适应 | B1、B2 | 仅调头；全量微调 |
| 协方差参考 | B3 | 源协方差对齐后调头 |
| 经典模型 | B4、B5 | CSP源先验LDA；log-Euclidean切线校准 |
| 自由低秩空间适应 | B6、B7 | 自由空间LoRA；加PCT保护 |
| 学习/随机方向参考 | B8、B9 | 元学习加性basis；随机生成元指数 |
| 元学习LoRA | B10、B11 | meta-LoRA；加PCT保护 |
| 主方法 | P、P+PCT | 元学习无迹生成元指数；加PCT保护 |

当前生产注册表含14种方法，而较早协议JSON只列出B0–B11和P；P+PCT已在实现注册表与campaign中单列。
以[方法注册表](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/p3work/method_registry.py)和[主线编排](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/p3work/training_campaign.py)交叉定位，不能只读取早期静态状态字段。

## 4. 精确计划规模

| 环节 | 计划量 | 含义 |
|---|---:|---|
| 源神经模型 | 15 fit | 源模型及预先登记的源训练配置 |
| 经典源模型 | 10 core＋6附加bank fit | 给经典对照提供独立源模型 |
| 源工程pilot | 200次实际优化更新 | 工程与精确恢复检查，不计科学效果 |
| 元工程profile | 5方法×100外层步 | 测量/启动工程工作，不计完整1000步fit |
| 元训练筛选 | 45 fit | 5学习方法×9配置，seed20261018 |
| 元训练确认 | 10新增fit | 各方法选中配置×2新增seed |
| 开发配置评价 | 118设置 | 14方法各自登记网格，不是118次源重训 |
| 开发确认评价 | 28新增项 | 14方法×2新增seed |
| 内部检查 | 1次冻结后的检查 | 主线内部比较，不能替代外部临床验证 |

5种学习方法为P、P+PCT、B8、B10、B11。
P/P+PCT/B8的维度为2/4/8，B10/B11的LoRA秩为1/2/4，均与ρ=.05/.15/.35构成9点网格。
完整元fit是1000外层更新，每步4个episode、每episode10步内层；每50步验证。
三种主线确认种子为20261018、20261019、20261020；选中筛选种子不重训。
上述是计划分环节计数，不能相加后当作等耗时的任务完成百分比。

## 5. 已完成的软件和部署工作

本轮经历依赖审计缺失、pilot结果清单生成失败、pytest快照稳定性检查、工程episode计划身份、启动/共享文件系统及Slurm提交等问题。
后来直接训练入口复用已完成pilot、源模型和既有数据核验，取消反复执行全量pytest、pilot和部署全树扫描的启动链。
已完成200更新pilot及精确恢复比较曾由用户回执确认；结果清单环节单独修复，四份checkpoint被保留。
主线直接运行日志明确记录`full_pytest=false`、`completed_fits_reused=true`、`fixed_12_hour_total_limit=false`。

运行日志增加shell首行`TRAIN_PROCESS_STARTED`、无缓冲Python和实际outer/session计数。
原科学更新数、数据划分和目标前缀预算没有因运行修复而无限增加。
旧源码/早期计划中的43200秒是历史默认值，不能据此判断直接运行版本仍有12小时总限额。
相关历史与最终学校配置见[独立交接记录](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/PAPER3_CONFIGURATION_HANDOFF_20261024_ZH.md)。

## 6. 最新可核验进度与资源

| 项目 | 证据时间/内容 |
|---|---|
| 原始训练计数 | 2026-10-24T04:40:59Z＝北京时间12:40:59 |
| 活跃轨迹 | B11、r=2、ρ=.05、seed20261018，外层186→198→199，内层至7960 |
| 已确认源神经fit | 15项完成 |
| 整轮完成 | 未确认；不能据上述单条轨迹推算总完成率 |
| 最近资源记录 | 作业770043，分区hpc_gpu1，节点gpu4 |
| 每作业申请 | 1GPU、24CPU、128G主机内存；A100 SXM4 80GB来自既有硬件记录 |

上层旧`CampaignError`和`CLASSICAL_PENDING`字段不能单独证明当前轨迹失败或经典模型仍未完成。
节点名与分区名分开解释；hpc_gpu1不是指定名为gpu1的节点。
后续仍按依赖完成剩余元筛选、选模、多seed确认、最强对照固定与内部检查。
外部冻结/临床外测/论文归档属于后续论文阶段，不能提前写为第一轮已完成。

## 7. 保存、续跑与结果位置

学校工程根目录：`/public/home/zhuziyuan/projects/paper3`。
源神经结果在`runs/source_training_v2/`；经典结果在`runs/source_classical_v2/`；元训练在`runs/meta/`。
编排与选择记录在`runs/training_campaign_v1/`；内部检查在`runs/internal_validation_once/`。
原作业日志为`.paper3-hpc/logs/train-770043.out`和`.paper3-hpc/jobs/770043/campaign.log`。
未完成轨迹由原内核读取同身份checkpoint；完整成功产物复用，不因换jobID就重训。
停止后实际可恢复到哪一步，以最后成功写入的checkpoint及恢复记录为准。

## 8. GitHub记录与结论边界

第一轮已有[人工配置和进度归档](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/CURRENT_AND_NEXT_ZH.md)，未确认接入持续自动回传。
`monitoring/paper3-followup/status.json`是旧24项扩展方案，不能冒充第一轮主线监控。
本页记录不会启动、取消、迁移或重启学校作业，也不改变第一轮数值程序。
目前可确认源模型和元训练实际运行；尚无本页所能支持的“整轮完成”“显著提升”“临床有效”结论。
