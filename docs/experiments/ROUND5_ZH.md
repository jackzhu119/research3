# 第五轮：旋转与相对增益的耦合方式

[返回五轮总索引](README_ZH.md)。本页记录已实现的设计、程序和可核验运行证据，不把数学性质当成准确率提升。

## 1. 研究问题与论文主线

继续冻结已有EEGNet，只用目标会话前20个已获取位置，拟合少量空间参数与同一个分类头。研究同样的低维参数预算下，旋转与相对增益的组合顺序能否改善跨会话校准，及改善来自目标适配图还是字典元学习。

保持clinic11、CAR后10维、128Hz、8–30Hz、固定时间窗与预RMS缓存；QC失败仍消耗原采集位置。B4/8/12/16/20共用B20之后的未来查询，不用查询样本或标签选择目标适应器。

## 2. 数学算法：已经实现的四种主图

令同一rank-4字典给出的生成元为 $M(\theta)=\operatorname{mat}(EB\theta)$，分解为 $K=(M-M^T)/2$ 与 $S=(M+M^T)/2$。K反对称，S对称无迹。

| 程序名 | 白化10维变换 | 本轮角色 |
|---|---|---|
| `single` | $\exp(K+S)$ | 联合指数参考 |
| `left_rotation` | $\exp(K)\exp(S)$ | 先相对增益、后旋转 |
| `right_rotation` | $\exp(S)\exp(K)$ | 先旋转、后相对增益 |
| `symmetric` | $\exp(K/2)\exp(S)\exp(K/2)$ | 对称耦合候选 |

矩阵作用于列信号，右侧因子先作用。四图使用同一组θ与同一分类头，没有给K、S分别增加一组参数。

四图具有相同身份切线。共同θ时，三个split图彼此正交相似，拥有相同白化奇异值、条件数、旋转极因子与酉不变位移范数；经参考协方差非正交重着色后，这些欧氏幅度匹配不再自动成立。记录白化和CAR10坐标的诊断，不能声称原始EEG扰动完全等幅。

有序图与联合指数的首项差为正负 $[K,S]/2$；对称图的差从三阶开始。联合指数与对称图均有参数取反的逆一致性。有序真实漂移也可能更适合有序图，未预设对称图必胜。

## 3. 目标适配与元训练

主方法 `geometry_refined`：共同身份切线上的10步Newton初始化，加各自真实非线性图上的5步支持损失精化；使用相同先验、投影端点与实际支持目标回溯。外层梯度通过初始化和精化，预测使用真实图及真实变换后的冻结特征。

主点为rank4、半径0.15、空间先验0.01、头先验0.1。每个字典最多1000次outer更新、每次4个episode；每50步验证、每10步保存，种子20261018/19/20。冻结源EEGNet复用，新图对应的字典重新定义，不能把旧字典冒充新图已完成训练。

`geometry_tangent`固定共同初始化θ与头，只换最终预测图，是机制探针；`bch2=exp(M+[K,S]/2)`只在该探针中出现，不列为新增部署主方法。

## 4. 实验任务分类

| 家族 | 精确组成 | 逻辑任务数 |
|---|---|---:|
| 字典训练 | 4种主图×3种子 | 12 |
| 核心交叉评价 | 4种训练图×4种评价图×3种子 | 48 |
| 半径敏感性 | 0.05/0.35×4种图×3种子；共同single字典 | 24 |
| 共同θ机制探针 | 4种主图及BCH2×3种子，B20 | 15 |
| 参考 | 仅头Newton、支持EA×3种子 | 6 |
| **主计划总计** | **只有12项训练字典，其余主要做校准评价** | **105** |
| 可选外部评价 | single字典4图×3，加3种split各自匹配字典×3 | 另21 |

48项核心评价报告五个支持预算。另在single字典的12项核心任务中嵌入72个控制漂移条件，不另计72次字典训练。

控制漂移包含KS、SK、同谱可交换三类，各0.15/0.35强度；同一变换同时作用支持与未来查询，标签、采集位置不变。三类匹配注入的旋转角谱与增益谱，实际位移仍单独记录。注入因子可交换不意味着学到的图或分类准确率必须相同。

## 5. 统计、诊断与可能成果

预先固定8项主比较：同single字典上三个split对joint、对称图对两个有序图，以及三个split各自匹配训练字典对共同single字典。参与者为单位，先在参与者内聚合会话和种子，再报告配对BA百分点、区间及整体Holm；敏感性不升级成事后主比较。

输出交换子范数、旋转/相对增益大小、白化与CAR条件数/位移、逆一致性、支持损失轨迹、回溯比例、概率质量、纠错与负翻转、校准耗时。已用过的内部队列仍属探索性复验。经典Strang/BCH不是本研究新定理；候选创新是特定小预算校准中可验证的耦合归纳偏置。

## 6. 已做的软件与发布工作

匹配Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6的本地环境完成248项软件测试；包括代数恒等式、元梯度方向差分、真实新图合成训练到报告、逐任务恢复、单作业三槽隔离、精确资源参数和在线安装。验证主机无GPU，不能据此称学校三卡实跑或识别效果已通过。

源码固定发布提交：`79ed3a5c2f74ebed1ee264b5ce54f43c0c378e29`；包装脚本SHA256：`5a5aeead706e4346caa50347f2719209129a658fc861fc2f775ba3acad060cb0`。发布时的“学校未提交”是交付当时事实，之后已有回传提交证据，见下一节。

## 7. 运行证据与资源

本次实际读取的自动快照时间为 **2026-10-24 11:21:41 UTC / 北京19:21:41**：作业 **771408**，控制器 `AWAITING_ACCOUNTING`，0/105 COMPLETE，105项NOT_STARTED。说明已有提交记录，尚不能证明节点训练开始；不是全轮失败结论。精简快照及源blob身份已存 [STATUS_SNAPSHOT_20261024.json](STATUS_SNAPSHOT_20261024.json)。

| 项目 | 申请配置 |
|---|---|
| 分区 | `hpc_gpu1`，不是节点名gpu1 |
| 拓扑 | 一个节点、一个普通Slurm作业、三张GPU内部并行 |
| CPU | 总36核，每槽12核亲和组 |
| 主机内存 | **整作业共享164G**，不乘三 |
| Slurm | `--nodes=1 --ntasks=1 --cpus-per-task=36 --mem=164G --gres=gpu:3` |
| 入口参数 | `--partition hpc_gpu1 --gpus 3 --cpus 12 --memory-total 164G --submit` |

164G主机内存不是显存；等分预算约54.67GiB不是进程硬限额。调度器选择同节点资源，不猜固定NodeName。三槽动态领取DAG任务，没有DDP、srun或大数组。

## 8. 保存、续跑、同步与结果位置

源码：`~/paper3-round5/ba5fab6bfa849d9ecaeb/`；输出：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5`。单任务失败记录到error.json，其他独立任务继续；双代checkpoint含训练/RNG状态，评价逐会话保存；相同固定启动器复用完成项及匹配断点。

不跑学校启动全量pytest/pilot/全数据树扫描；无累计12小时上限。单次Slurm默认5天3小时，提前300秒请求保存，控制器处理可续交中断；明确取消保持停止。重新执行入口不能代替修复仍存在的数值错误。

每180秒尝试GitHub汇总同步，网络失败不终止训练。完整checkpoint、预测和CSV保存在学校本地；公开汇总不等于完整模型异地备份。

- [本轮源码、设计和运行手册](https://github.com/jackzhu119/research3/tree/paper3-r5-progress/experiments/followup_20261024_r5)
- [实时回传入口](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/monitoring/paper3-round5/status.json)
- [固定线上启动信息](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/experiments/followup_20261024_r5/online/ONLINE_LAUNCH.json)
- [精确任务矩阵](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/experiments/followup_20261024_r5/EXPERIMENT_MATRIX.json)
- [数学设计](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/experiments/followup_20261024_r5/DESIGN_ZH.md) · [统计](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/experiments/followup_20261024_r5/STATISTICS.md) · [软件验证](https://github.com/jackzhu119/research3/blob/paper3-r5-progress/experiments/followup_20261024_r5/VALIDATION.json)

可选Weibo阶段独立输出`runs/followup_20261024_r5_weibo`，单独回传`monitoring/paper3-round5/weibo-status.json`；本次归档未确认它已启动。
