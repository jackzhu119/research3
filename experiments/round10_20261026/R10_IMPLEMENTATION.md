# R10 实现、学校部署与结果交付

本轮在独立目录与新分支实现，不修改 R1–R9 的科学代码、checkpoint、任务和结果。研究设计、数学定义与统计方案分别见 [研究方案](R10_RESEARCH_PROPOSAL_ZH.md)、[数学](R10_MATH.md) 和 [实验计划](R10_EXPERIMENT_PLAN.md)。本交付完成了软件与合成验证，尚未取得 R10 真实 EEG 效果。

## 实际资源与启动行为

按用户最新要求固定 **4 个独立 Slurm 单卡作业，每个作业 1 GPU、16 CPU、68G 内存**；合计 4 GPU、64 CPU、272G。分区 `hpc_gpu`，作业时间上限 `5-03:00:00`。学校队列决定实际同时运行数量，不承诺绕过资源配额。作业共享有限任务池，领取可运行任务；参考任务先完成，依赖任务再执行，不把每个科学任务拆成一个 sbatch。

安装只解包代码，准备只生成计划和 sbatch 文件。**只有显式 `--submit` 才启动后台控制器并允许其提交 R10 作业。**本次研发没有申请或使用 GPU，没有提交或取消学校作业。

耗时测试在交付前集中执行。运行时不执行全套 pytest、不重新递归扫描原始数据、不每次重复完整上游验收，不把独立 GPU 预检收据设为训练前置条件。提交前检查小型计划和必需模型文件；科学进程首次打开实际缓存时核对数据身份。已有源模型、合法 R9 特征参考和已完成会话直接复用。

## 文件结构与接口

| 文件 | 实际职责 |
|---|---|
| `START.py` / `paper3r10/cli.py` | 安装后命令入口；先设置 CuBLAS 确定性环境，再按需要恢复学校 venv site-packages |
| `algorithm.py` | float64 支持拟合、查询可行集合的保守界、零修正证书、逐查询决策；安全 NPZ 断点 |
| `ablations.py` | 同拟合目标的局部二次 profile 和 R9 式匹配方差门控，明确不提供主算法证书 |
| `baselines.py` / `baseline_registry.json` | 真实 EA、低/全容量 head、历史空间 SGD、R3/R4/R8/R9 适配器；R9 raw采用完整补档支持的RBF，EA采用linear，均沿用原meta_validation配置 |
| `vendor/` | SHA 锁定的原始科学模块；来源与必要运行修复逐项记录，不覆盖旧包 |
| `registry.py` | 仅从已知计划、收据与小型合同定位冻结输入；显式覆盖可补充已存在的路径 |
| `study.py` | 15/165/57 项阶段计划、预算、种子、代码身份、比较家族与任务依赖 |
| `data.py` / `confirmation.py` | 预算前缀与固定未来 query、标签能力隔离；新确认数据的元数据/物理采集顺序接口 |
| `worker.py` / `metrics.py` | 会话断点、预测先封存再读取评价标签、完整指标、模型及实际环境记录 |
| `statistics.py` / `report.py` / `selection.py` | 参与者配对推断、完整 CSV、固定开发分组、选型与冻结 |
| `platform/` | 四作业调度、轻量状态、独立报告进程、双通道上传和完成回执 |
| `build_delivery.py` / `delivery_template.py` | 确定性、自包含、无需 pip 或 git clone 的离线安装包 |

主算法只接受支持标签：`fit(support_features, support_logits, support_y, Config(...), reference=..., support_ids=...)`；`predict(state, query_features, query_logits, query_ids=...)` 没有 query 标签参数。论文分析读取封存后的评价结果。所有 R10 方法统一采用 `1e-12` 概率下限计算指标，同时保留原生未截断 log-probability 审计；不要把这些 NLL 与采用旧下限的历史 NLL 直接合并检验。

## 环境兼容性与已执行验证

学校目标环境：Python 3.12.3、Torch 2.5.1+cu121、CUDA 12.1、NumPy 2.2.6、A100 40/80GB；CuBLAS `:4096:8`，关闭 TF32/autocast，冻结源模型 BN/dropout。不同 A100 UUID、40/80GB 容量和已知驱动版本是实际环境记录，不据此拒绝合法迁移。R9 CUDA cumsum 的确定性兼容修复只作用于 vendored 本轮模块。

本次在 Python 3.12.14 / **Torch 2.5.1+cpu / NumPy 2.2.6** 执行完整 CPU 测试；学校 Python 微版本与实际 CUDA 内核尚未实测。完整验证数量和环境见 `research/FINAL_VERIFICATION.json`。合成测试覆盖数学退化、非零更新、梯度和查询完成界、概率/NLL、真实 baseline 入口、预算身份、预测封存、恢复、统计、冻结、四作业调度与上传故障。安装包还经过禁止 Torch/NumPy/网络/子进程导入的默认安装测试和全部文件 SHA 读回。

GPU 预检已经实现，但本次按要求**未执行**。用户以后可在已有学校 GPU allocation 内显式运行；它只有四个合成信号的冻结网络推理，不申请新卡，不要求登录节点运行 GPU 检查。

## 学校最短操作路径

完整下载校验的一行安装指令随 `delivery/SCHOOL_COMMANDS.md` 发布，固定入口 SHA256，提供主下载与固定提交备用下载。默认不训练。以下命令使用该入口安装后生成的 `CURRENT.json` 定位包，无需手填内容地址：

```bash
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
R=/public/home/zhuziyuan/projects/paper3
E="$HOME/paper3-r10-downloads/START-PAPER3-R10.py"
S=$("$P" -I -S -B -c 'import json,pathlib; print(pathlib.Path(json.loads((pathlib.Path.home()/"paper3-round10/CURRENT.json").read_text())["package"])/"START.py")')
```

可选的一次 CPU 自检，不需要 GPU，训练入口不会自动重复全套测试：

```bash
"$P" -I -S -B -u "$E" --validate
```

直接准备机制实验：命令自动定位当前已完成收据及现成缓存，**不下载数据，不重新训练源模型**。复用参考需要三个固定种子的 raw 与 EA 参考齐备，否则由固定 meta_train 前缀生成两种参考；此选择在开发与正式阶段保持一致。

```bash
"$P" -I -S -B -u "$S" plan --root "$R" --stage mechanism
```

准备成功显示 `PREPARED_NOT_SUBMITTED`、15 项任务、4 作业、每作业 16 CPU/68G。下列命令才正式运行，控制器、报告和上传均脱离终端，显示启动回执后可关闭终端：

```bash
"$P" -I -S -B -u "$S" submit --plan "$R/runs/round10_preparation/plan-mechanism.json" --submit
```

若学校已有收据位于非标准路径，可用 `inputs --root "$R"` 查看具体缺项，按 `registry.py` 描述符格式给 `plan --registry-overrides /path/overrides.json` 指定**已有**模型/计划路径。不要用随机 basis、弱化基线或空收据替代缺失文件。

## 机制实验后的正式实验

第一批只在 meta_train/meta_validation 上执行15任务。后台报告持续输出全部候选及两个预分组的结果，随后显式冻结：

```bash
"$P" -I -S -B -u "$S" report --plan "$R/runs/round10_preparation/plan-mechanism.json"
"$P" -I -S -B -u "$S" freeze --plan "$R/runs/round10_preparation/plan-mechanism.json"
```

只有预先规定的开发条件通过才生成真实 `freeze.json`。STOP 表示机制证据不足，保留全部数据用于分析，不能强行标记为突破。软件测试生成的冻结记录不会授权真实实验。

```bash
"$P" -I -S -B -u "$S" plan --root "$R" --stage formal_exploratory --freeze "$R/runs/round10_preparation/freeze.json"
"$P" -I -S -B -u "$S" submit --plan "$R/runs/round10_preparation/plan-formal_exploratory.json" --submit
```

上述165任务覆盖三旧队列、三种子、18方法和五预算；其性质始终是探索性。新确认数据必须有实际此前未使用的数据、冻结信号合同、参与者清单和采集顺序记录，模板 `configs/independent-confirmation.template.json` 不是可执行数据。符合这些条件后：

```bash
"$P" -I -S -B -u "$S" plan --root "$R" --stage formal_confirmatory --freeze "$R/runs/round10_preparation/freeze.json" --confirmation /absolute/path/new-confirmation.json
"$P" -I -S -B -u "$S" submit --plan "$R/runs/round10_preparation/plan-formal_confirmatory.json" --submit
```

现有仓库没有已登记的新确认队列，因此不能提前提供或声称完成这57任务的独立确认结果。

## 状态、恢复与错误处理

```bash
"$P" -I -S -B -u "$S" status --plan "$R/runs/round10_preparation/plan-mechanism.json"
"$P" -I -S -B -u "$S" resume --plan "$R/runs/round10_preparation/plan-mechanism.json" --submit
```

每阶段输出 `runs/r10_<stage>/`：`runtime/tasks.json`、控制器/worker 日志、当前任务/会话/模型更新、最新数值推进时间、独立心跳时间与错误。不能把有心跳当成训练正在推进。在线状态发布到 `paper3-r10-progress` 的 `monitoring/paper3-round10/<stage>/<recipe>/status.json`，兼容 latest 指针保留阶段优先级。

完成会话有预测、状态和指标回执，恢复直接复用；若仅指标计算中断，读取已封存预测恢复评价，不再拟合。Slurm 的 checkpoint 信号在会话边界暂停，退出75继续领取剩余工作。单个确定性任务连续失败最多3次后隔离，其他无依赖任务继续，不用错误重试填满 GPU；修好具体原因后可显式 `resume ... --retry-blocked --submit`。控制器只管理自己的 R10 标记，不取消旧轮作业。网络失败不会进入科学任务失败计数。

## 双通道归档与完整报告

训练一开始就启动独立后台进程：CPU reporter、通道A、通道B、monitor。CPU reporter 只根据新完成任务的紧凑指标更新报告，避免 R9 “任务计数已推进而 report 仍是旧子集”的问题。

- 通道A优先上传完整参与者/会话 CSV、比较与消融、全部配置、数据与代码身份、证书诊断、实际环境和报告，不截断行数。
- 通道B独立上传完整预测、NPZ checkpoint、轨迹及科学归档，分片 SHA256 与远端读回回执。失败可以续传，A不等待B。
- 凭据只读取学校已有安全配置或已有环境变量；不把凭据复制进包或结果，不写日志。

`paper3-results/results/paper3/round-10/stages/<stage>/<recipe前16位>/` 保存分阶段路径，根 `INDEX.json` 与 `PAPER_DATA_LATEST.json` 是阶段索引。COMPLETE 要求该精确计划全部任务、报告任务键覆盖和 SHA 匹配；机制阶段完成不会冒充整轮正式确认完成。独立确认的统计家族与旧探索性队列分开，缺失对照保留 NOT_READY，不静默缩小家族。

单独重启后台报告/归档可运行 `START.py export --plan PLAN`，不申请 GPU、不重训。不需要保持浏览器或 SSH 终端在线。
