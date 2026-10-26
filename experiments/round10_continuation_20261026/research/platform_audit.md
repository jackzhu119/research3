# R10 工程复用审计与接口建议

审计时间：2026-10-26。本文依据实际 Python 源码读取，不依据 README 推断。当前仅提出接口；本代理没有提交 Slurm 作业、使用 GPU 或修改 R1–R9。

## 可复用的科学输入

1. `paper3r7/data.py:load_source` 读取既有 `runs/source_training_v2/f0_receipt_seed-{seed}.json`，检查 `SOURCE_F0_COMPLETE`、匹配 pipeline seed、检查点 SHA、信号契约与同 seed 源参考收据；加载相同 `CompactEEGNet(11,n_times,dropout)`。`FrozenSource` 冻结全部参数并始终处于 eval，特征提取前采用既有 `joint_trial_rms`。R10 不必重新训练源模型或下载数据。
2. `paper3r7/data.py:Catalog` 通过 `status/tasks.json -> W3.cache_manifest` 读取实际 Stieger 缓存及既有 participant split，拒绝参与者重叠。只按需打开数组；`np.load(...,mmap_mode='r')` 后可有受内存上限控制的本地数组缓存。信号、manifest、split 身份均可纳入 R10 任务身份。
3. `p3work/splits.py:build_prefix_episode` 将真实采集顺序前 20 个位置作为成本，invalid trial 不补抽。B4/8/12/16 应从这 20 个 acquired UID 列表截取，再筛 valid support；所有预算共用原 B20 query UID 集合。带物理 run 的 Stieger 会丢弃 B20 边界 run 的余部，且仅采用后续完整、已验证 run。不能将其简化成原始第 21 条起全部数据。
4. `paper3r7/external.py` 明确 Pan/Weibo 为 published-order exploratory evidence，未验证真实物理采集时序。原始数据准备、下载与 worker catalog 已分离；R10 worker 必须只读取已有合法缓存，不调用 prepare/download 函数。

## 必须收紧的能力边界

现有 `Catalog.materialize` 返回同时含 support 与 `query_y` 的 `SimpleNamespace`。现有 R9 `worker.evaluate` 在 fitter 前读取 query labels 生成 session cache signature。这并不等于现有 fitter 用 query 调参（其 fitter 实参确实只有 support），但不是 R10 应复用的理想接口。

建议 R10 新建如下独立边界：

```text
Catalog/session metadata (worker private)
  -> CalibrationInputs(x,y,valid_support_uids,acquired_prefix_uids,source/data IDs)
  -> fit(... CalibrationInputs only ...) -> fitted state
  -> PredictionInputs(query_x, query_uids, source/data IDs) -> predictions
  -> persist immutable prediction record with SHA
  -> EvaluationLabels capability opened only by evaluator -> session metrics
```

query labels 不应进入 model/selection/cache-feature identity。固定 query UID、信号数组 SHA、预处理配置及 source checkpoint 身份足以构成 label-free prediction identity；评估标签另有 frozen evaluation identity。既有 query eligibility 是原冻结协议的一部分，应保留其生成记录，不能按 R10 效果重新挑选会话。

每个预算的 EA 只用该预算的有效支持信号统计；query 不能更新均值、协方差或 batch normalization。预测函数还应通过 query 顺序与 batch size 不变性测试，避免隐式 transduction。所有有监督 fit API 不接受 query_x/query_y，query 位置相关修正只能在已拟合状态的 `predict` 阶段使用该位置的特征。

## 需原生纳入 R10 的已知环境修复

- 在 import Torch 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。R5 曾因严格确定性与 cuBLAS workspace 配置时序失败。
- balanced weights 必须以 consumer logits dtype 构造；`.5 / mask.sum().to(dtype=weights.dtype)`，避免 CUDA indexed assignment 的 Float/Double 冲突。B4 单类支持必须走预声明 fallback，不能强行补抽标签或修改预算。
- metadata 的浮点累计和/加权分位数放 CPU：例如 R9 的 `torch.cumsum(weights.detach().cpu(),0)`；严格确定性保持开启，不用 `warn_only=True` 掩盖实际非确定算子。
- 数值线性代数使用 float64、对称化、可审计 jitter 梯度、Cholesky/特征值截断与条件数诊断；原始算法状态及退化结果明确区分。OOM 可缩小纯预测 batch 后重试，但不能偷偷改变拟合目标、正则或校准位置。
- 学校目标：Python 3.12.3 / Torch 2.5.1+cu121 / NumPy 2.2.6 / cuDNN 90100 / CUDA build12.1，CentOS7/glibc2.17/Slurm20.11。CPU 验证应记录实际本地版本；本地 Torch2.8 CPU 测试不能冒称学校 CUDA 验证。
- 不复用 R1 的持久 UTC/monotonic 累计时钟 gate。R10 把每次尝试的 operational timing 作为日志，科学状态仅依赖任务身份和完整原子检查点。跨节点的不同 boot ID 不妨碍同一科学检查点恢复，且不能伪造跨 boot 精确墙钟时间。

## 运行平台建议

现有 `runtime/three_jobs.py` 适合借鉴：stdlib controller，flock task lease + native task lock，明确 recipe SHA，原子 result/progress、`USR1` 通知检查点、逐会话复用以及已完成任务不重训。

不能直接复制其身份/轮次限制：现有 `generic_scheduler.py` 仅允许 round9；`finish_task` 对任何非75错误都无限 RETRYING，会重现先前数十次同一异常重试。R10 应新建命名空间、marker、runtime directory 和 durable retry policy：

- 科学完成仅由 `result.json(status=COMPLETE, recipe/data/source/freeze identity exact)` 判定。状态 heartbeat 不等于模型步数前进。
- 网络上传错误只出现在独立上传状态，不进入训练退出码。
- `PAUSED`/allocation timeout 可在同一完整检查点恢复；结构性配置/数据身份错误标为 `BLOCKED_CONFIGURATION`，不占用 GPU 无限循环。
- 可恢复 IO/节点错误有限退避重试；同一错误签名连续到上限后 quarantine 并给出具体 traceback，其他独立任务继续。不能把失败任务错误地写 COMPLETE。
- 新计划只包含对应阶段所需任务；避免整轮一次展开几千任务。阶段2只接受 meta_train/meta_validation 数据；阶段3固定选择、参数与统计；阶段4需要 freeze receipt，旧 holdout/Pan/Weibo 永远标为 exploratory。
- 配置默认 `partition=hpc_gpu, jobs=4, gres=gpu:1, cpus_per_task=16, mem=68G`。每个 allocation 只用逻辑 cuda:0，不能将 worker slot1/2/3 当 GPU ordinal1/2/3。只有显式 `--submit` 才能调用 sbatch；安装、CPU验证、计划生成、监控读取不得提交或取消作业。
- 独立的 GPU 预检入口只在用户自行进入已分配 GPU 节点后执行，限制为真实设备记录、小张量确定性线性代数、一次冻结 EEGNet 前向和所选算法的小样本 fit/predict；不生成正式科学结果、不自动提交 GPU预检作业。

推荐接口（待主算法确定后实现）：

```text
START.py install --root ROOT                 # code/config only, no Torch/Slurm
START.py validate --cpu                      # synthetic deterministic tests
START.py gpu-precheck --device cuda:0        # explicit only, allocation checked
START.py plan --stage mechanism --root ROOT  # finite metadata-only plan
START.py submit --plan PLAN --submit         # only submission boundary
START.py status --plan PLAN                  # readonly fresh timestamps
START.py resume --plan PLAN --submit         # only unfinished exact identities
START.py freeze --mechanism-report REPORT    # no final query selection allowed
```

## 双通道结果与上传

现有 R9 `result_channels.py` 已实现 paper 与 archive 两个 detached 进程、独立 lock、局部 retry，训练 worker 不导入网络 SDK。应保留这个架构，但 R10 不应复用其固定 round9 prefixes。

更可靠的字节传输参考 `paper3-completed-results-upload-20261026/transport/lossless_publish.py`：UTF-8 原字节分片、逐分片/整源 SHA、最终 Git blob 读回验证、CAS non-force publish，只有远端验证完成后发布 latest pointer。旧快速摘要曾因为16MiB限制省略完整 reports，不应复用这些上限/过滤器。

通道A须完整包含 plan、freeze、runtime recipe、source/data identity、逐参与者与会话指标 CSV、校准有效样本数、各预算/机制消融、预声明统计 family、paired effects/CI/multiplicity、fallback 与 negative-transfer、校正/引入错误、时间与参数复杂度。CSV行/参与者 ID 不得净化掉。A 的完成回执应区分科学任务完成、报告覆盖、远端字节验证。

通道B归档 prediction UID/order、预测概率/源概率、评估标签、模型/算法状态、检查点、全部验证/选择轨迹、失败/恢复证据及 source/data manifests；每个文件和有界 tar.gz part 有 SHA256，远端验证后才有完整完成回执。凭据只通过学校既有安全配置按需读取，不进入包、日志、SHA manifest、聊天或子进程 argv。两条通道网络失败均无限后台退避，但不改变训练状态或重训。

## 测试优先项

1. 固定 query UID 跨预算一致、invalid paid prefix 不补抽、单类/空支持 fallback、原始 run 边界与前后会话不交叉。
2. 用会抛错的 evaluation-label capability 运行 fit/select，证明只在 prediction persisted 后评分；query标签置换不改变 fit、选择或预测。
3. source权重及 BN statistics 前后字节不变；predict 对 query permutation/batch size 不变。
4. float32/float64和单类权重、空矩阵/近奇异/重复样本、确定性 metadata cumsum、CUDA workspace 在 import 前设置。
5. resume中断前后相同source/data/code/plan/freeze身份、随机数与采样器/选择轨迹一致；已完成结果不重训；变更身份拒绝复用。
6. Slurm伪 runner 验证 default 不 sbatch/scancel、显式submit只配置4单卡16CPU68G、重复执行幂等、ambiguous sbatch ack 不重复提交、no-GPU ordinal正确。
7. 训练过程模拟 HTTP失败和损坏远端分片不受影响；A/B独立恢复、全字节校验、两者回执不互相冒领完成。

## 已查阅的源文件

- `/workspace/paper3-round9/paper3r7/data.py`
- `/workspace/paper3-round9/paper3r7/external.py`
- `/workspace/paper3-round9/paper3r7/baseline.py`
- `/workspace/paper3-round9/p3work/splits.py`
- `/workspace/paper3-round9/paper3r9/worker.py`
- `/workspace/paper3-round9/paper3r9/plan.py`
- `/workspace/paper3-round9/runtime/{START,backend_entry,coordinator,three_jobs,generic_scheduler,result_channels}.py`
- `/workspace/paper3-r9-cumsum-repair/r9_runtime_hook.py`
- `/workspace/paper3-completed-results-upload-20261026/{upload_driver.py,transport/lossless_publish.py}`

这些本地源是已部署实现的可检索副本；正式 R10 依赖锁定仍必须以根代理本轮获取的 GitHub pinned commit 与实际文件 SHA 为准，不能只凭本机修改时间。

## 本轮已实现的平台与验证（2026-10-26）

`paper3r10/platform/` 已独立实现 R10 的 stdlib 计划、代码、冻结和完成回执约束；不会修改 R1–R9。`load_plan` 重算全部科学计划内容的 canonical recipe，验证声明的小代码文件、冻结的真实开发数据范围、代码 manifest 和 study registry 身份；机制阶段也禁止 `method_config` 隐式覆盖 canonical config。确认阶段在提交前调用只读 metadata validator，检查独立数据审查、采集顺序、物理 run 边界与会话 evidence，不读取整个 EEG array、不导入 Torch。

四槽每槽1卡、16 CPU、68G，使用 `hpc_gpu`。只有显式 `--submit` 能写提交授权并启动 Slurm controller；prepare 仅写文件。状态含每个 task 的当前 attempt、数值进度、最后真实数值前进时间与独立心跳。任务配置错误立即 blocked，运行异常最多3次后 quarantine；分配在进入任务前就失败也按每槽最多3次隔离。代码75代表正常检查点暂停并继续；无效 partition/account/GRES 等提交配置阻断自动提交；明确的调度配额等待按退避继续。修复后必须显式 resume `--submit --retry-blocked`，保留历史与检查点。没有跨节点 monotonic-clock 硬阻断。

A/B、监控和 CPU reporter 共4个独立 detached 服务。reporter 在新的已封存任务出现后运行最多300秒的 CPU report 子进程；生成失败重试，不影响训练、无需凭据。论文数据完成同时要求全部预期 task 身份、精确 recipe、报告 SHA 和完整 task-key SHA，旧摘要即使完成数量相同也不能冒领当前报告完整性。A 保存完整逐参与者与会话 CSV/统计/配置字节，B保存预测、fit state、检查点与轨迹原字节分片，均独立远端 GET/SHA 验证。

各阶段科学回传命名空间为 `results/paper3/round-10/stages/{stage}/{recipe前16位}/`，监控为 `monitoring/paper3-round10/{stage}/{完整recipe}/status.json`。根 `INDEX.json` 与 `PAPER_DATA_LATEST.json` 是显式 stage-index；非强推 CAS 合并保留各阶段各通道，active stage 只能 mechanism → formal_exploratory → formal_confirmatory。较早机制任务的延迟重试不会覆盖正式阶段入口。

隔离的 CPU 测试验证了任务 lease 与孤儿科学进程锁、防重复执行、正常/异常/检查点恢复、旧错误不污染新 attempt、冻结配置与代码/registry 变更拒绝、真实数据范围、采集无泄漏接口、A/B网络错误隔离、精确字节重建、过时报告拒绝、报告失败后恢复、阶段指针 CAS 冲突合并、安装包路径安全与默认零执行。平台、交付与独立审查回归共74项在本地通过。该数字不代表学校 CUDA 预检、正式 EEG 训练或科学性能已验证。

一键安装器为自包含、SHA锁定的 code-only ZIP/Base85 payload，普通 `os.replace` 支持学校 CentOS7/NFS，内容寻址安装且默认仅安装。`--validate` 才执行有限 CPU 检查；`--gpu-precheck` 必须用户明确调用且只在已有分配内有限执行；`--submit` 是独立显式操作。本轮没有提交、取消或分配任何学校作业或 GPU。
