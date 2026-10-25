# 第四轮：支持内交叉拟合与稳健校准

归档日期：2026-10-24。本文将算法实现、实验计划、软件验证和真实调度状态分别记录；已提交排队不等于已经更新模型。

返回[七轮总目录](README_ZH.md)；相邻记录：[第三轮](ROUND3_ZH.md)、[第五轮](ROUND5_ZH.md)。

## 2026-10-24追加：迁移至hpc_gpu（包已发布核验，学校未执行）

用户最新目标为**hpc_gpu、A100-PCIe-40GB、一个节点一个Slurm作业内部3GPU，总36CPU、128G共享主机内存**，每槽12CPU。原hpc_gpu1、48CPU、192G共享主机内存（每槽16CPU/64G规划）保留为历史发布/提交配置，不代表迁移已在学校生效。新提交应请求`--partition=hpc_gpu --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=36 --mem=128G`；使用迁移入口处理原CLI语义，128G不乘三。

本次只迁移资源与提交封装，不改变本轮算法、数据划分、种子、204项矩阵或原输出目录；完成项和匹配checkpoint/逐会话结果继续复用。使用现有学校项目缓存，不临时补取新原始数据或在GPU节点安装环境。

独立操作分支：`paper3-hpc-gpu-migrations-20261024`，目录`operations/hpc_gpu_migrations_20261024/round4`；[迁移交付目录](https://github.com/jackzhu119/research3/tree/paper3-hpc-gpu-migrations-20261024/operations/hpc_gpu_migrations_20261024/round4)已真实发布。固定源码提交`79eabe6c456aa443e18c35c53b6c39a33d27a639`，小bootstrap提交`1d3e4860fc8a138e7be2717dae75ed55a0657e47`。本轮包323,081bytes，SHA256 `a1ef49e5e888d67477b59f8db0ec8ee798bdba51335b667ae8631cc595c9516a`。

本轮198项本地迁移全套软件测试通过；实际raw保留分片续传收到HTTP206，API完整下载HTTP200及正确缓存禁网复用通过。下载包在禁socket/DNS和子进程场景实际安装、prepare204项，保留原计划字节及result/checkpoint/session的字节和mtime；不下载新增原始数据。线上小bootstrap本身已从raw/API下载核SHA，并实际调用本轮缓存包安装/prepare。以上为本地CPU与交付网络验证，不是学校GPU/学校网络现场实测。

[三轮迁移启动说明](https://github.com/jackzhu119/research3/blob/c7c903fb126e3e60a7e69556bea9f7ac8ce7e489/operations/hpc_gpu_migrations_20261024/online/LAUNCH_ZH.md)交付对应`MIGRATE --submit`；用户执行后才精准处理本轮同标记旧排队，已进入运行的作业不当作排队直接取消。截至本次记录，学校取消/提交均为0，没有迁移后实际GPU训练或结果数据；启动说明固定提交为`c7c903fb126e3e60a7e69556bea9f7ac8ce7e489`；[本轮固定启动shell](https://github.com/jackzhu119/research3/blob/c7c903fb126e3e60a7e69556bea9f7ac8ce7e489/operations/hpc_gpu_migrations_20261024/online/RUN-ROUND4.sh)已实际下载并通过shell安装/prepare，未执行`--submit`。

本轮继续单独回传`paper3-r4-progress:monitoring/paper3-round4/status.json`，不写到其他轮或操作分支。下文旧资源、作业号与状态均保留原时点语境，不能据迁移计划重写为已经切换或已经训练。

## 本轮要回答什么

第三轮研究“前20条数据在给定先验下支持多大空间变化”；第四轮进一步研究“这次适应在支持内未参与拟合的试次上是否有益”。
目标是在源模型、仅头校准和空间校准之间，利用有限前缀内部的证据决定预测组合，同时检验疑似错标时稳健目标的收益与代价。

- 支持内样本外预测能否比同支持重代入更可靠地选择适应强度？
- 收益来自空间调整、头调整，还是多模型概率平均？
- one-SE保守候选选择能否减少支持小样本下的过拟合？
- GCE会降低误标影响，还是连真正有用的困难试次也一起压制？

完整设计：[DESIGN_ZH.md](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/experiments/followup_20261024_r4/DESIGN_ZH.md)。

## 数据与论文主线保持一致

- 冻结已有源EEGNet，复用源参考协方差及原缓存，不更新源BN或主干权重。
- 沿用clinic11、128Hz、8–30Hz、相同时间窗、参与者划分和三个种子20261018/19/20。
- 目标预算为前20个已获取位置，QC失败仍计费，不从后续试次补入。
- B4/8/12/16/20均评价同一B20之后的未来查询，支持变短不扩大查询集。
- 未来查询输入和标签均不进入折分、混合权重、候选选择、阈值或支持拟合。
- 交叉拟合是目标会话内算法计算，不是需要通过才能启动训练的额外流程。

## 数学算法：已实现

**A．OOF源锚收缩。** 将前缀按原付费位置连续分成5个block，各折模型只用其他block拟合，再预测留出block。
记源概率为p0，OOF概率差为d，类别平衡权重为q，解支持内Brier加向源收缩先验：

`α = clip(Σ q·d·(y−p0) / (Σ q·d²+γ), 0, 1)`。

然后用完整有效支持重拟合校准器，未来预测为 `(1−α)p0 + αpS`；这是概率混合，不是参数插值或logit混合。
QC后保留原付费位置对应的折身份；单类训练折按固定规则返回源预测，记录回退但不中断其他会话。

**B．源/头/空间凸堆叠。** 对source、head-only、spatial生成同折OOF概率矩阵P，解三成员单纯形二次问题：

`min_{w≥0, 1ᵀw=1} ||W^(1/2)(Pw−y)||² + (γ/2)||w−e0||²`。

仅3成员，枚举7个非空活跃集求解；完整支持重拟合后沿用w形成未来概率。成员权重是预测组合系数，不直接等于真实生理漂移类型。

**C．one-SE候选选择。** 支持内固定比较头先验.01/.1/1与空间半径.05/.15的六个组合，以真实非线性OOF概率评分。
在最低CV分数加其SE的容许集合中，优先更小半径，再更大头先验；同时保留直接minimum选择消融。
选择规则是启发式，fold共享训练样本且EEG相关，SE不能解释为未来风险证书。

**D．GCE稳健空间拟合。** 主q=.7，使用 `ℓq(py)=(1−py^q)/q`，q=.3/1作为敏感性，q→0回到CE。
每轮工作加权CE/GGN方向用真实GCE加正则目标回溯；不声称求解了全局凸问题或全局最优解。
`xfit_stack_gce`只将空间成员改为GCE，head-only仍为匹配先验的普通逻辑头。

实现：[paper3r4/algorithm.py](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/experiments/followup_20261024_r4/paper3r4/algorithm.py)。

## 精确实验规模与对照

| 分类 | 组成 | 逻辑任务数 |
|---|---|---:|
| 旧字典 | legacy × r2/r4/r8 × 3种子 | 9 |
| 匹配先验字典 | matched_sgd × r4 × 3种子 | 3 |
| 新目标字典 | xfit_shrink / xfit_stack / gce × 3种子 | 9 |
| 核心评价 | 旧字典14组、matched_sgd4组、新字典各4组，共30组×3种子 | 90 |
| 敏感性评价 | 折分、折数、混合先验、基方法、秩、GCE、选择规则、半径 | 90 |
| 支持EA参考 | 3源模型种子 | 3 |
| **核心合计** | **21字典＋90核心评价＋90敏感性＋3参考** | **204** |
| 可选Weibo外部评价 | 固定7方法组 × 3种子 | 另21 |

204不是204次完整源模型训练；至多21项字典fit，完整匹配的旧结果可只读复用。
每个字典最多1000outer更新、每步4episode；每50步验证、每10步保存，按元验证规则选择字典。
核心90项覆盖B4/8/12/16/20；108组压力条件嵌于18个核心任务，仅B20运行。
对照包括源、仅头、仅空间、固定α=.5、均匀堆叠、同支持重代入权重、OOF收缩/堆叠和匹配训练先验的SGD字典。
敏感性含blocked/stratified/LOO、2/10fold、stack先验.001/.1、protected-base、r2/r8、GCE q=.3/1、minimum-CV和ρ=.05/.35。

矩阵证据：[EXPERIMENT_MATRIX.json](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/experiments/followup_20261024_r4/EXPERIMENT_MATRIX.json)。

## 已实施与只提出的内容

已实施的是上述OOF混合、one-SE选择、GCE、必要消融和固定204项矩阵。
**各向异性Ω先验、前向链验证、完整固定α=.25/.5/.75/1网格尚未实施**，不计入已完成实验。
Weibo21项为可选独立评价；缓存复用和准备成功不表示该评价已经完成。
OOF经验目标不劣于源的性质，只针对所优化的支持内经验目标；完整支持重拟合会改变函数，不能转成未来准确率不降定理。
stacking、交叉验证、one-SE、GCE均有既有文献，本轮候选贡献在论文任务下的组合、机制分解与严格对照，不把经典方法改名当新理论。

## 软件与部署验证

- 精确版本环境为Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6；180项本地CPU软件测试通过，0失败。
- 验证OOF排除被留出标签、付费位置折分、闭式收缩与单纯形QP一致、三成员KKT、GCE q=0极限及回溯。
- 验证新字典完整元梯度、合成元训练→三槽分配→评价→报告→复用未变、独立任务失败不取消其他槽。
- 覆盖单节点单作业资源、CPU亲和隔离、单卡可见、Bash4.2信号、提交丢失回执及QoS等待。
- 本地验证没有GPU；它不代替学校三卡实跑、真实EEG提升或统计功效验证。

依据：[VALIDATION.json](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/experiments/followup_20261024_r4/VALIDATION.json)。其中未提交字段记录发布时验证范围，不是此后的实时调度状态。

## 已观察状态：已提交，等待GPU配额

GitHub回传 `updated_utc = 2026-10-24T11:22:43Z`，即北京时间19:22:43：

- 学校已接受单个三卡作业 **771366**。
- Slurm状态 **PENDING**，排队原因 **(AssocGrpGRES)**，控制器QUEUED。
- **0/204 COMPLETE**，204项为QUEUED_ALLOCATION；尚无三槽实际优化进度。
- 报告PARTIAL、主要比较族未齐、`scientific_completion_claimed=false`。

因此这份快照表示“已提交等待关联GPU配额”，不是训练代码报错，也不能写成“已完成”或“尚未提交”。
实验分支README的发布时“未提交”句子早于这份调度回传，应以带时间的作业证据解释当前阶段。
状态证据：[monitoring/paper3-round4/status.json](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/monitoring/paper3-round4/status.json)。链接会继续更新；本文仅记录上述时点。

## 原提交资源与单作业三卡执行（历史配置）

资源组为 **hpc_gpu1**，A100-SXM4-80GB配置；一个节点、一个Slurm作业，内部三个独立单卡worker动态领取DAG就绪任务。
精确请求：`--nodes=1 --ntasks=1 --cpus-per-task=48 --mem=192G --gres=gpu:3 --partition=hpc_gpu1`。
三个槽各16CPU亲和组；64G/槽是共享192G内存的规划预算，不是给进程设置64G虚拟地址硬上限。
`--cpus 16 --memory 64G`按每GPU槽解释，整作业自动乘三；不与第五轮的总内存CLI混用。
不使用数组、额外srun或DDP；hpc_gpu1是分区名，不猜测具名NodeName。
仅GPU空闲三张不代表满足同节点48CPU/192G，CPU不足或关联配额不足均可能等待。

## 续跑、回传和结果路径

源码：`~/paper3-round4/<包hash>/`；结果：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r4`；外部独立后缀 `_r4_weibo`。
主节点日志为 `logs/train-作业号.out/.err`，任务日志为 `logs/task-索引-attempt次数.out/.err`，每30秒报告三槽实际进度。
已完成result直接复用；字典双代checkpoint含随机状态，评价逐会话保存，恢复只补缺失项。
单任务科学失败记录错误，其他独立任务继续；依赖该失败项的任务等待，不冒充完成。
无累计12小时限制；单次默认5天3小时，提前300秒USR1保存，退出75后对账续交一个三卡作业。
学校启动不重跑全量pytest/pilot/数据树扫描，不重装环境；用户取消、TERM或STOP不无限自动复活。
GitHub每180秒尝试同步 `paper3-r4-progress:monitoring/paper3-round4/status.json`；外部为独立 `weibo-status.json`。
回传和报告失败不停止训练；checkpoint、逐会话预测、`reports/report.json`与CSV始终先写学校本地。

运行与固定入口：[RUNBOOK_ZH.md](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/experiments/followup_20261024_r4/RUNBOOK_ZH.md)、[online/ONLINE_LAUNCH.json](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/experiments/followup_20261024_r4/online/ONLINE_LAUNCH.json)。

## 结果解释边界

12项主要比较预先固定，按参与者配对、会话和三种子内聚合后统一Holm；已有内部队列仍是探索性复验。
GCE可能抑制真正困难样本，须同时保留clean、错标压力、纠错和负翻转，不能只选有利结果。
Weibo真实run与物理采集顺序未知，只是发布顺序LR外部稳健性，不能当作新的在线前瞻确认。
目前0/204的排队快照没有第四轮识别提升证据；待所需组完成后才讨论显著性。

统计依据：[STATISTICS.md](https://github.com/jackzhu119/research3/blob/paper3-r4-progress/experiments/followup_20261024_r4/STATISTICS.md)。

