# 第二轮：空间成分、可靠性保护与未来运行段稳定性

[返回五轮总览](README_ZH.md) · [第一轮](ROUND1_ZH.md) · [第三轮](ROUND3_ZH.md)

本轮目录为`followup_20261023_v2`，合计66项；不是更早24项V1方案。
本页归档2026-10-24可读取证据，状态保留下文时间，不是实时Slurm控制页。

## 1. 本轮要回答什么

继续使用冻结源EEGNet、前20个已采集试次、相同未来完整运行段和少量空间校准参数。
分别检验：空间旋转/相对增益是否需要不同调整强度；不可靠教师保护是否阻碍校准；平均未来损失是否掩盖后续运行段不稳定。
不重新训练大骨干，不将随机会话划分替代真实时间前缀。
源库、40/11/11参与者划分、11通道CAR及128Hz/8–30Hz合同沿用第一轮冻结输入。

## 2. 方向A：条件化空间成分

由同一r维系数和同一字典构造生成元，分解为：

\[
M=\operatorname{mat}(EB\theta),\quad S=(M+M^\top)/2,\quad K=(M-M^\top)/2,
\quad M'=\alpha_sS+\alpha_kK.
\]

\[
(\alpha_s,\alpha_k)=\sigma(W[e;1]),\quad
A=Q^\top R_{ref}^{1/2}\exp(M')R_{ref}^{-1/2}Q.
\]

e含六项支持集证据：协方差偏移、交错分半协方差分歧、源模型平衡CE、源预测熵、类别不平衡、有效/已采集比例。
证据特征detach，门初值.9；双门只增加14个全局元参数，目标会话仍只更新θ和原分类头残差。
`global`、`scalar`、`dual`分别区分固定收缩、单条件权重、成分分别条件化。
`symonly`与`skewonly`为成分消融，同时改变可达方向及能量，不能当作纯因果隔离。
在白化10D空间仍保留无迹和幅度界；它们不保证准确率或无新错误。

## 3. 方向B：支持集可靠性保护

`PCT`使用原硬正确性保护；`softPCT`改为逐试次平滑可靠性权重：

\[
r_i={1+\sum_{j:y_j=y_i}\mathbf1(\hat y_j=y_j)-\mathbf1(\hat y_i=y_i)\over n_{y_i}+1},
\quad w_i=1+5r_i p_0(y_i|x_i)\left(1-{H(p_0(x_i))\over\log2}\right).
\]

\[
L_{in}=L_{balanced\ CE}+L_{reg}+.1\,\operatorname{mean}_i
\left[w_i\|z_i-z_{0i}\|_2^2/2\right].
\]

`softPCT_mean`把同一支持集的w取均值后用于所有试次，以排除只是减弱总约束的解释。
只读取支持输入及标签，不用未来query标签形成权重。
评价同时记录BA、负翻转和纠错；不能只挑有利指标。

## 4. 方向C：未来运行段风险

\[
L_{outer}=.75L_{all\ future}+.25\tau\log\left({1\over J}\sum_j e^{L_{run_j}/\tau}\right),
\quad\tau=.5.
\]

仅使用真实完整物理运行段；不足两个合格双类运行段时回退到原均值损失并记录比例，不中断该episode。
`runmean`用普通等权运行段均值替代倾斜项，隔离尾部强化和运行段重新加权。
`dual_temporal`是预先指定组合；不根据内部留出结果临时拼装新方法。

## 5. 精确实验矩阵

| 类别 | 配置 | 逻辑任务数 |
|---|---|---:|
| 核心 | 12方法×3种子，r4/ρ.15 | 36 |
| 敏感性 | P/dual×4配置×3种子 | 24 |
| 参考 | head_only/support_EA×3种子 | 6 |
| 合计 | 60项元训练＋6项无元训练参考 | 66 |

12方法为`P, global, scalar, dual, PCT, softPCT, temporal, dual_temporal, softPCT_mean, runmean, symonly, skewonly`。
敏感性为(r2,.15)、(r8,.15)、(r4,.05)、(r4,.35)，仅B20评价；不扩成所有参数全笛卡尔积。
种子20261018/20261019/20261020；每元fit1000外层×4episode×10内层，每50步验证。
门参数使用functional Adam(.003)，其步数及两个动量进入checkpoint；空间字典使用原Stiefel更新。
已有P/PCT结果只有在配置、源模型、数据和完整产物匹配时才能复用，不提前计入完成数。

核心预算曲线为B4/8/12/16/20，均保持B20定义的同一future query。
P/dual/PCT/softPCT/softPCT_mean各3种子追加六种诊断：支持标签翻转10%/20%、查询增益及旋转漂移各两档。
六种诊断共90个模型-条件评价，嵌入现有任务，不是90次额外元训练。
查询漂移只作用后续输入，支持保持原样；不能解释为算法已看见漂移并主动跟踪。

## 6. 已做的软件和部署工作

[固定设计](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/experiments/followup_20261023/DESIGN_ZH.md)保存推导、方法、矩阵和文献。
[软件验证](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/experiments/followup_20261023/VALIDATION.json)记录87项CPU测试通过，环境匹配Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6。
测试覆盖P/PCT数值一致、门梯度和Adam状态、两代checkpoint、验证中断恢复、预算/扰动/报告、Bash4.2信号和独立网络同步。
该验证文件是设计交付时的软件证据，其中“尚未提交”已被后续学校实际运行记录更新。
CPU软件通过不等于已实测所有A100、驱动、Slurm权限及共享存储行为。

最初一次提交66数组元素触发`QOSMaxSubmitJobPerUserLimit`；把`%3`改`%2`不能降低提交数量。
后续采用最多三个独立单GPU作业滚动补交，避免整批提交和单个多卡资源碎片问题。
40GB迁移和启动参数修复、最终资源均见[配置交接](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/PAPER3_CONFIGURATION_HANDOFF_20261024_ZH.md)。

## 7. 资源：历史方案与部署记录

| 阶段 | 配置 | 地位 |
|---|---|---|
| 原设计 | hpc_gpu1；每项1GPU/24CPU/128G；数组最多3项 | 历史，不能当当前申请 |
| 中间尝试 | 单作业四卡、3×16CPU及不同内存方案 | 迁移历史，未成为最终配置 |
| 最近确认部署 | hpc_gpu；最多3个独立1GPU作业；每项12CPU/64G | 用户终端已有RUNNING记录 |
| GPU | 已选择A100 PCIe40GB，节点gpu1/gpu2 | 型号来自硬件/迁移记录，节点名不等于分区 |

合计最多3GPU/36CPU/192G，允许跨节点；不是DDP，不是本页把第四/五轮单节点要求套到第二轮。
最新监控仍含旧计划`max_parallel_gpus=1`和`cpus_per_gpu=24`，不能据此推翻终端实际申请。
本次只归档差异，不热修改运行中的控制器或监控程序。

## 8. 最新回传状态

[自动状态](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)此次读取快照：

| 字段 | 观测值 |
|---|---|
| 快照时间 | 2026-10-24T10:17:15Z＝北京时间18:17:15 |
| 完整任务 | 23/66 |
| 报告 | PARTIAL，core_complete=false |
| 活跃示例 | skewonly seed20261019 outer127；P seed20261020已进入meta_train |
| 尚未完整完成 | 43项；不是剩余43次完整源模型训练 |

旧文档12/66是15:45:44北京时间的历史快照，两者反映不同时间，不互相覆盖证据。
23/66是逻辑项完成比例，不能直接解释为墙钟时间完成34.8%。
1000训练更新结束但仍在验证/留出/诊断时，尚不能标为整项COMPLETE。

## 9. 保存、续跑、同步和结果

学校代码包：`/public/home/zhuziyuan/paper3-followup/0e55dba2a78b43464e8c/`。
结果根：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/`。
每任务保存`progress.json`、`result.json`、`selected.json`、`latest.json`、两代checkpoint及逐会话预测。
每10个外层更新保存；验证会话结果单独落盘；同身份完整项跳过，未完成项读取旧断点。
门Adam、字典、随机状态、更新步数及选择历史均在恢复状态中；跨节点恢复不能仅凭调度迁移宣称逐位等价。
不设置累计12小时限制；单个Slurm作业时限仍独立存在，USR1提前保存，集群是否允许requeue取决于学校权限。

每30秒实际进度心跳；独立monitor每180秒尝试GitHub同步，网络失败重试，不停止模型计算。
自动分支为`paper3-v2-progress`；GitHub主要回传进度和部分表格，不代表全部checkpoint已异地备份。
完整模型、概率和CSV先保存在学校持久共享目录。
最终表包括`report.json`、`comparisons.csv`、预算/参考/敏感性/诊断比较表；以同目录实际生成文件为准。

## 10. 统计与创新结论边界

主终点为B20内部留出参与者均值BA；人内聚合会话和三种子后做参与者配对。
固定10项主比较，核心36项齐备后统一Holm；未齐只展示描述结果，不滚动挑显著p值。
预算、敏感性、诊断和概率指标是辅助证据；内部留出沿用既有体系，不能包装为新的外部盲测。
只有dual优于global/scalar，才支持会话条件化或双成分分配的额外价值。
只有softPCT优于同平均权重对照、temporal优于runmean，才更支持各自机制。
本页尚不能认定某个方法最优、显著提升或临床有效；负结果和退步案例同样应保留。
