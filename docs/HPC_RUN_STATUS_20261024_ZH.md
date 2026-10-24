# 论文3｜学校HPC两轮训练与第二轮40GB三作业配置快照（2026-10-24）

> **证据快照，不是实时调度器。** 本文汇总用户提供的学校Slurm现场输出及 [research3 第二轮自动同步状态](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)。第二轮状态文件最近一次读取的记录时间为 **2026-10-24 07:25:53 UTC（北京时间15:25:53）**；作业分配来自同日用户提供的 `squeue`、`scontrol` 和训练日志。该时间之后的状态必须重新查询，不能把本快照理解为始终 RUNNING。
>
> 只归档可核实的状态、资源、程序入口和恢复约定；**不修改、不重新部署、不启动/停止训练，也不复制受限制EEG数据或凭据**。

## 1. 当前部署与资源分配（最近现场记录）

两轮在同一学校Slurm集群并行，是**不同实验与不同输出目录**，不是一个分布式四卡模型。第二轮每个独立作业独占1张GPU，最多3个并行；不同计算节点共享持久结果目录。

| 实验／任务 | 最近作业号 | Slurm分区／节点 | 最近状态 | GPU／CPU／主机内存 |
|---|---|---|---|---|
| 第一轮原始训练 | `770043` | `hpc_gpu1` / `gpu4` | `RUNNING` | 1 × A100 SXM4 **80GB**；24 CPU；128G |
| 第二轮任务12：`P-seed20261019` | `771066_12` | `hpc_gpu` / `gpu2` | `RUNNING` | 1 × A100 PCIe **40GB**；12 CPU；64G |
| 第二轮任务13：`global-seed20261019` | `771065_13` | `hpc_gpu` / `gpu2` | `RUNNING` | 1 × A100 PCIe **40GB**；12 CPU；64G |
| 第二轮任务14：`scalar-seed20261019` | `771064_14` | `hpc_gpu` / `gpu1` | `RUNNING` | 1 × A100 PCIe **40GB**；12 CPU；64G |

上表为**资源申请/已分配量**，不等于GPU始终满载。第二轮合计最多3 GPU、36 CPU、192G主机内存；加上第一轮，现场同时分配合计4 GPU、60 CPU、320G主机内存。**40GB显存对剩余所有模型的适配性仍须逐任务确认**；`sacct MaxRSS` 不是GPU显存峰值。

第二轮之所以不再排 `hpc_gpu1`：该分区80GB A100先前因 `Priority` 排队；先前要求单个4卡作业在 `hpc_gpu` 曾显示 `AssocGrpGRES`。经调整为**三个独立单GPU作业**，在 `hpc_gpu` 获得了实际资源。旧作业 `770766_12`、`770767_13`、`770768_14` 的0秒 `COMPLETED` 是迁移/撤销后的作业记录，不是科学训练完成的证据。

## 2. 第一轮原实验 `770043`

- 根目录：`/public/home/zhuziyuan/projects/paper3`。
- 关键位置：`runs/training_campaign_v1/`、`runs/source_training_v2/`、`runs/meta/`。
- 最新**直接提供的日志记录**：`2026-10-24T04:40:59Z`，协调器 `COORDINATOR_RUNNING`，活动任务 `meta_screening`：**B11、dimension=2、radius=0.05、seed=20261018**。
- 该活动任务确认 `outer_updates_completed=199`、`inner_updates_actual=7960`、`outer_attempts_dispatched=200`；该任务还未从这些数值中证明最终完成。
- 源神经拟合进度文件记录 `actual_neural_fits_complete=15`，`pending_neural_jobs=[]`，仍带旧状态文本 `NEURAL_SOURCE_COMPLETE_CLASSICAL_PENDING`；这**不独立证明**当前经典基线状态。
- campaign活动对象还出现 `error_type=CampaignError`，但外层更新在增加。保留异常字段；不据此断言整个第一轮已经失败或已结束。
- 后续按 [主线调度代码](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/p3work/training_campaign.py) 的实际依赖推进：完成元学习配置筛选、冻结方法配置、开发验证、额外seed确认、选择最强对照、冻结配置、一次内部留出检验。W7外测协议冻结、W8临床外测和W9论文统计归档**不是**作业770043已经自动完成的事项。

## 3. 第二轮扩展V2：科学范围与结果

- 实验根：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2`。
- 独立逻辑任务共 **66项**：**36项**核心（12方法 × 3 seeds）、**24项**超参数敏感性（P/dual × 4配置 × 3 seeds）、**6项**不作元训练的参考（head-only / support covariance EA × 3 seeds）。
- 核心变体：`P, global, scalar, dual, PCT, softPCT, temporal, dual_temporal, softPCT_mean, runmean, symonly, skewonly`。
- 三个种子：`20261018`、`20261019`、`20261020`；核心元训练任务预设 **1000 outer updates**，并按固定周期做元验证。
- 数据：40名参与者、396个元训练session；11名参与者、97个元验证session（见现场 `CATALOG_READY`）。固定预算B4/B8/B12/B16/B20、机制对照及人工误标/空间漂移诊断按既定方案执行。
- [GitHub自动同步状态](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json) 在 `2026-10-24T07:25:53Z`：**12/66 COMPLETE；3项 EVALUATING；51项 QUEUED_OR_NOT_STARTED**；报告 `PARTIAL`、`core_complete=false`。
- 12项完成是seed20261018的12种核心方法，并非全部66项；余下**54项尚未完成**。
- 同一状态快照：`P-seed20261019`、`global-seed20261019`、`scalar-seed20261019` 各推进到 `outer_update=150/1000`，在 `meta_validation`，分别 `session=52/97`、`64/97`、`70/97`。这些数值不是当前秒级数据，也不是整项任务完成比例。
- 已生成的部分报告不得视为最终统计结果；3 seeds未全部完成、强对照/内部与临床证据仍未齐全。

**说明一个自动监控字段的局限：** `monitoring/paper3-followup-v2/status.json` 的 `max_parallel_gpus` / `cpus_per_gpu` 在读取时仍含旧计划参数，**不反映本次迁移后Slurm实际上已经分配的3×(1张40GB、12CPU、64G)**；硬件分配应以 `scontrol show job` 的 `Partition` / `TRES` 和当前队列为准，科学进度以原结果文件及监控时间为准。

## 4. 提交控制、续跑及保护

1. 第二轮运行的是**3个独立单卡Slurm作业、滚动补交**，不是一个4卡作业，也不是一次提交66个数组元素；每完成/回收一个名额再补后续未完成编号。
2. 既有66项 `plan.json`、训练代码和结果根路径保持不变。每项先核对 `result.json.status == "COMPLETE"`；完成项跳过。
3. 未完成项复用共享目录中的 `latest.json` / 两代模型状态检查点，以及原程序的保存和随机状态机制。**新seed的0次更新属于它自身的新测试，不代表完成项重新训练。**
4. 新配置调度器位于学校用户目录 `/public/home/zhuziyuan/paper3-submit-three40/`（安装/切换日志 `switch.log`）；第二轮滚动控制日志在 `/public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/_threejobs16/controller.log`，已按40GB配置切换；数值训练日志在该输出目录的 `logs/train-<jobid>_<taskid>.out`。
5. 旧80GB三单卡队列及单个4卡排队方案已由提交迁移流程取代。**不要重复运行**旧的 `START.py --submit`、旧迁移脚本或另一套控制器，也不要为“刷新”状态取消第一轮 `770043`。
6. `RUNNING`仅表明获得资源；须结合 `V2_PROCESS_STARTED`、`TRAIN_PROGRESS outer_update`、`TASK_COMPLETE` 和完成结果确定科学执行。登录节点或监控器中断、资源不足、40GB显存OOM仍可能发生；不能保证所有剩余任务无故障运行。
7. GitHub第二轮状态记录已出现持续更新提交，约180秒一个周期；**上传状态≠checkpoint已完整异地备份**，需单独审计远端存储/哈希和结果汇总状态。不包含任何凭据。

## 5. 只读检查命令（在学校 `login1` 执行）

```bash
squeue -r -u "$USER" -o "%.20i %.12P %.10T %.6C %.26j %.35R"
scontrol show job 770043 | grep -E 'JobState=|Partition=|NumCPUs=|TRES='
tail -n 35 /public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/_threejobs16/controller.log
tail -n 35 /public/home/zhuziyuan/paper3-submit-three40/switch.log
```

对于任务12–14，用最新的 `squeue` 作业号定位 `logs/train-*.out`；不可认为这些作业号永远不变。

### 参考与工作边界

- [自动进度报告（会更新）](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)
- [第二轮方法与代码（GitHub方案）](https://github.com/jackzhu119/research3/tree/paper3-v2-progress/experiments/followup_20261023)
- [第一轮真实阶段调度源码](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/p3work/training_campaign.py)
- [机器可读的本次人工快照](../status/HPC_ALLOCATION_SNAPSHOT_20261024.json)

文档可用于科研交接和资源审计；**不表示新代码已部署、Slurm实时核查已由本次GitHub提交完成，或实验科研结论已验证**。
