# R11 实现、四卡并行与学校直接运行

R11 完整实现位于新分支的 `experiments/round11_20261026`，不覆盖 R1–R10 的代码、任务、数据或结果。数学、实验和文献边界分别见 `R11_MATH.md`、`R11_EXPERIMENT_PLAN.md`、`R11_RESEARCH_PROPOSAL_ZH.md`；本次真实来源复核见 `R11_EVIDENCE_AUDIT_ZH.md`。

## 1. 默认会运行什么

默认 `campaign` 是一个确定的 **10,270项任务 DAG**，四个持久单卡作业自动领取工作，不需要人工反复切换阶段：

| 内容 | 任务 |
|---|---:|
| meta_train冻结特征参考 | 3 |
| 550配置×3种子×5预算的离线算子学习 | 8,250 |
| 550配置×3种子的meta_validation评价 | 1,650 |
| 开发集Frozen/EA | 6 |
| 仅开发集选型，主方法及两组对照分别调参 | 1 |
| 选型后九种匹配机制控制的学习 | 135 |
| 25方法×3种子×3旧队列正式探索性评价 | 225 |
| 合计 | **10,270** |

阶段依赖先固定开发网格，再记录选型结果，最后运行三个旧队列；旧holdout/Pan/Weibo仍是探索性。代码不会读取它们的query标签来选配置。独立新数据缺失时不制造确认结果。

这里的任务数量不等于10,270次EEGNet重训。编码器、源分类器及其归一化统计全部冻结；新方法只在meta_train离线学习小矩阵，目标阶段按支持标签计算分数及矩阵乘法。三个已完成源种子直接复用。

550配置为440主方法配置、55无置换惩罚配置、55 exact-logistic ERM配置。默认网格允许扩展：`--grid /path/grid.json`，键为 `feature_ranks`、`noise_strengths`、`ridges`，值为明确有限列表。每次扩展使用新的输出与计划文件，保留全部失败和负结果；增加实验数量不会增加独立参与者数，也不会使旧测试集变成独立确认。

## 2. 实际资源

**4个独立Slurm作业，每个1张GPU、16核CPU、64G内存**，合计4 GPU/64 CPU/256G。分区 `hpc_gpu`，每allocation上限 `5-03:00:00`；到时保存的会话/算子继续复用。实际运行与排队原因单独记录，控制器启动成功不等同于四卡已经分配。

本次开发没有申请GPU、提交新Slurm或取消现有作业。正式学校执行只有显式 `--submit` 才被授权；默认安装和准备不提交。

## 3. 代码接口

| 文件 | 实际行为 |
|---|---|
| `START.py` / `cli.py` | 先设置确定性CuBLAS环境；显式安装/准备/运行/状态/报告入口 |
| `algorithm.py` | NumPy float64 PFST算子；exact fixed-count permutation协方差；矩阵自由CG与约束/ERM控制 |
| `r10_profile.py` / `ablations.py` | 保留R10数学实现作为明确标注的设计基线，不冒称R10已有正面实测 |
| `study.py` / `registry.py` | 锁定550配置、种子、预算、任务身份、数据/源引用；历史基线缺失逐项记录 |
| `data.py` | 原acquired前缀、固定B20未来query、同prefix EA；训练future标签只开放给meta_train |
| `worker.py` | 参考/算子学习/开发选型/目标预测与封印；逐会话断点、真实配置与环境归档 |
| `selection.py` / `report.py` / `statistics.py` | 参与者配对nAUBC one-SE选型、LOPO诊断、完整CSV、固定6K/18K校正家族 |
| `platform/science_daemon.py` | 每allocation长期保留一个科学进程，Torch只导入一次 |
| `platform/runtime.py` / `scheduler.py` | 缓存完成状态、有限重试、依赖终态传播、真实queue/controller元数据 |
| `platform/uploads.py` | 每完成一项只捕获和发布新增证据；独立报告、A、B、monitor后台进程 |
| `build_delivery.py` / `delivery_template.py` | 无需pip/git clone的自包含离线启动包，默认只安装 |

核心接口：`train_operator(pairs, Config(...), reference=...)` → `OperatorState`；`adapt(operator, support_features, support_logits, support_y, support_ids=...)` → `CalibrationState`；`predict(state, query_features, query_logits, query_ids=...)` → 概率与诊断。目标适配和预测接口都没有query标签参数。

训练pairs明确来自meta_train的prefix→固定future，A和Q都使用该A预算拟合的EA。条件置换协方差是人为随机化的精确协方差，不能称为真实EEG标签噪声。类别平衡时它与匹配的中心化特征协方差控制相同，已作逐元素回归验证。

所有R11方法采用共同 `1e-12` 概率下限评分，并保留原生log-probability审计。不能直接把采用不同概率下限的历史NLL并入同一次统计检验。

## 4. 运行时省去什么，保留什么

发布前集中执行测试；正式入口不会自动跑pytest、全量上游验收或GPU预检，不复制整份venv，不依赖历史节点缓存，不要求外部PASS收据或跨包runtime_profile一致。

每allocation的科学子进程只导入一次Torch；source/geometry、只读catalog、特征参考和算子按固定身份复用。catalog命中不会重新读全部events/W3，目标适配状态和标签封印仍逐任务独立。小矩阵BLAS使用单线程避免开销，冻结网络可使用16 CPU线程。

调度使用持久化COMPLETE缓存，启动/恢复只做一次已有回执恢复，之后不在每次领任务时扫描10,270个目录。状态仅返回有限当前/近期行和精确总计；逐任务完整记录仍保存在科学输出里。

必须正确的源模型、实际数据、预算和UID不会被忽略。历史基线缺少artifact时只隔离对应方法/种子，EA/Frozen和新方法继续；报告保留缺失比较，不能用Frozen冒充基线或悄悄缩小多重比较家族。

## 5. 发布前验证与边界

验证环境是 Python **3.12.14**、Torch **2.5.1+cpu**、NumPy **2.2.6**。学校目标是 Python3.12.3/Torch2.5.1+cu121/CUDA12.1：API主版本匹配，实际CUDA构建与学校微版本尚未运行，不能将CPU验证写成GPU实测。

已实现的验证覆盖：数学排列穷举/梯度/全局损失界、非对称反例、平衡退化、有限迭代有效解、native EEGNet和真实文件格式、三个源种子、首20付费位置、固定query、训练/评价标签权限、实际小campaign的训练→选型→目标预测→恢复。平台回放覆盖进程导入失败、Python异常、子进程崩溃、受损CUDA上下文的隔离恢复、SIGUSR1、allocation超时、网络重试、增量上传与10,270任务规模。

`research/FINAL_VERIFICATION.json` 记录最终完整测试数和代码SHA。它不包含R11真实EEG性能结论。可选GPU预检仅在用户已有allocation里显式执行四条合成信号前向；默认启动不等待该收据。

## 6. 学校一条入口

`delivery/SCHOOL_COMMANDS.md` 给出固定提交URL、入口SHA和完整一行下载启动。它会自动完成下载、安装、计划生成和显式提交，不再要求先手动安装另一份文件。

自包含入口的三个明确模式：

```bash
# 只安装，不准备或提交
"$P" -I -S -B -u "$HOME/paper3-r11-downloads/START-PAPER3-R11.py"
# 安装并准备整个campaign，不提交
"$P" -I -S -B -u "$HOME/paper3-r11-downloads/START-PAPER3-R11.py" --start
# 安装、准备并正式启动四作业后台运行
"$P" -I -S -B -u "$HOME/paper3-r11-downloads/START-PAPER3-R11.py" --start --submit
```

这里 `P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python`。入口读取已有学校环境与缓存，不隐式安装依赖或下载数据。准备成功后计划路径为：

`/public/home/zhuziyuan/projects/paper3/runs/round11_preparation/plan-campaign.json`

科学输出默认为 `runs/r11_campaign/`，新包安装到 `~/paper3-round11/<payload>/`。与R10的68G配置、安装目录、进度分支和输出互不覆盖。

需要扩展网格时，通过已安装 `START.py plan --grid GRID --output NEW_R11_OUTPUT --plan-file NEW_PLAN` 创建新计划，再显式 `submit --plan NEW_PLAN --submit`。旧计划内容不可变；常规再次启动直接恢复既有计划，不重复发现全部依赖。

## 7. 后台、断点和错误

显示 `DETACHED_CONTROLLER_STARTED` 后可关闭终端。控制器有自身ready回执；立即退出与尚未应答分别报告，不把单纯spawn PID当成功。真实sbatch job ID、queue原因和最近错误写入runtime并回传。

每个科学任务有独立结果和失败记录。普通Python异常只影响该任务；科学子进程致命退出时仅重建子进程。SIGUSR1在科学单元边界保存，退出75继续；已完成会话与封存预测不重新拟合。连续代码失败的任务隔离并写清原因；其它分支继续。失败候选整组退出选型，selection等待所有开发分支终态而不是永远等失败分支COMPLETE。

没有完整开发候选时使用预先固定默认配置并标记 `DEFAULT_UNVALIDATED`/`NO_VALIDATED_SELECTION`；这是科学有效性标签，不是停止所有训练的性能门槛。报告和全部比较保留该标签，不把未验证选型包装成成功。

```bash
"$P" -I -S -B -u "$HOME/paper3-r11-downloads/START-PAPER3-R11.py" --status --plan /public/home/zhuziyuan/projects/paper3/runs/round11_preparation/plan-campaign.json
"$P" -I -S -B -u "$HOME/paper3-r11-downloads/START-PAPER3-R11.py" --resume --submit --plan /public/home/zhuziyuan/projects/paper3/runs/round11_preparation/plan-campaign.json
```

## 8. 双通道归档与可分析结果

A优先上传完整参与者/会话CSV、比较/消融、配置、诊断及来源；每个sealed task独立发布一次，之后累计索引引用已有task manifest，不反复重传全部历史。变化的reports独立更新。B完整预测、NPZ、训练配对索引及轨迹独立分片与SHA验证；网络失败保留捕获和远端回执，不影响任务计数或触发重训。

R11新的A累计索引schema沿 `tasks[key]` 和 `reports` 各自的manifest追溯。B按已完成任务增量归档。`INDEX.json`仅保存阶段/通道紧凑指针，避免把一万份详细group塞入同一索引。缺失或隔离任务明确留在ledger；完整成功与部分结果分开标识。

结果分支：`paper3-results/results/paper3/round-11/stages/<stage>/<recipe>/`；根阶段索引保留实际路径。进度：`paper3-r11-progress/monitoring/paper3-round11/status.json`，通道会自动创建分支。凭据只读学校已有安全配置，不放入包、日志或聊天。

无网络的本地训练依旧继续。`START.py export --plan PLAN` 可重新启动独立报告/上传，不申请GPU、不重训。R2–R10历史上传器与结果不会被R11控制器接管。
