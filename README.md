# 论文3 · 学校HPC训练与后续实验

> **2026-10-24 学校HPC最新可核实运行快照（非实时）**：第一轮作业 `770043` 在 `hpc_gpu1` 使用1×A100 80GB/24CPU/128G；第二轮V2**已经实际启动**，切换到 `hpc_gpu` 的3个独立1×A100 PCIe 40GB/12CPU/64G作业，已知 `12/66` 完成、任务12–14正在运行/验证（2026-10-24 15:25北京时间附近的证据）。[完整资源与科研状态交接](docs/HPC_RUN_STATUS_20261024_ZH.md) · [机器可读快照](status/HPC_ALLOCATION_SNAPSHOT_20261024.json) · [第二轮实时更新源](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)。**下方2026-10-23描述是历史快照，不能据此认为第二轮“尚未提交”。**


**当前实验和下一轮实验均使用学校云服务器（Slurm / NVIDIA A100）；不再使用RunPod。**

更新时间：2026-10-23（北京时间）。当前运行状态依据研究者最新反馈；最近收到的具体训练日志截至 **2026-10-23 14:59:30（北京时间）**。本页不是Slurm实时查询结果，未取得新回执时不推算完成率或宣称全部训练完成。

## 当前正在运行，与下一轮计划

| 项目 | 当前原实验 | 下一轮扩展实验 V2 |
|---|---|---|
| 状态 | **学校HPC运行中：研究者最新反馈** | **设计、代码与软件验证完成；尚未提交** |
| 范围 | 原论文3源模型、元学习筛选/比较及原计划后续阶段 | 12方法核心消融＋参数敏感性＋低成本参考 |
| 最近具体运行证据 | 作业770043，gpu4；日志已进入B8元学习筛选，r=2、ρ=.15、seed=20261018 | 66项计划任务；60个元训练任务、6个无元训练参考任务；完成数0 |
| 资源说明 | 按当前学校作业实际分配；不因新规划改变运行任务 | 最多3张A100并行，每项1GPU、24CPU、128GiB内存 |
| 输出位置 | `runs/training_campaign_v1/`、`runs/source_training_v2/`、`runs/meta/` 等原目录 | 新目录 `runs/followup_20261023_v2/` |
| 结果含义 | 已有实际训练进度，最终科学结果尚未汇总确认 | 87项CPU软件测试通过；不是66项实验完成，也没有真实提升结果 |

原日志曾报告15个源神经网络拟合完成、待运行神经任务为空；这不等于整个论文实验完成。B8的某一配置步数也不是整个计划的完成百分比。作业号770043是最近日志的身份，若学校续跑产生新号，应以实际队列为准。

- [当前阶段、下一轮任务与状态来源](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/CURRENT_AND_NEXT_ZH.md)
- [学校服务器环境与运行注意事项](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/SCHOOL_HPC_ENVIRONMENT_ZH.md)
- [下一轮完整设计、公式和文献](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/experiments/followup_20261023/DESIGN_ZH.md)
- [下一轮三卡代码与启动说明](https://github.com/jackzhu119/research3/tree/paper3-v2-progress/experiments/followup_20261023)
- [下一轮V2自动同步状态](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json) — 该页面不代表当前原实验的进度。

## 当前实验如何查看

在学校 `login1` 查看自己的队列：

```bash
squeue -u zhuziyuan -o '%.20i %.24j %.10T %.12M %R'
```

查看最近已知作业；若作业号已变化，把770043替换为当前作业号：

```bash
sacct -j 770043 --format=JobID,State,ExitCode,Elapsed,NodeList
```

```bash
tail -n 60 /public/home/zhuziyuan/projects/paper3/.paper3-hpc/logs/train-770043.out /public/home/zhuziyuan/projects/paper3/.paper3-hpc/jobs/770043/campaign.log
```

以上只查看状态。恢复当前实验时沿用它自己的已部署续跑入口；不要用下一轮安装包替代当前任务，也不要照旧RunPod说明重新安装、重复pilot或重提test链。

## 下一轮准备了什么

36项核心元训练（12方法×3种子）＋24项秩/幅度敏感性元训练＋6项参考任务，共66项。另附5档校准预算、6种预定义误标/未来空间漂移诊断，以及概率质量、负翻转和配对统计。

已有源模型、数据缓存及匹配的完成基线可以复用；独立数组最多三项并行。断点包含优化器状态，完成项跳过，失败项可续跑；本地30秒进度、GitHub约180秒独立同步。没有人为累计12小时限额；Slurm单作业时限和requeue权限仍由学校配置决定。

下一轮在当前实验结束后使用独立输出启动。目前GitHub状态是 `DESIGN_V2_READY_NOT_SUBMITTED`，并非正在训练。此前24项V1方案已由66项V2取代，不是额外一批正在运行的任务。

## 分支与历史资料

仓库默认分支保留论文协议与准备历史；学校部署工具维护入口位于 [HPC分支](https://github.com/jackzhu119/research3/tree/research/paper3-slurm-hpc-20261021)，下一轮独立实现位于 `paper3-v2-progress/experiments/followup_20261023/`。无需为了查看本页而更新学校正在运行的代码。

旧RunPod入口、CUDA12.8锁文件、初始“尚未训练”说明、旧pilot和恢复规则均为历史版本资料，不能继续作为当前学校训练指令。`ONLINE_RELEASE.json`、`ONLINE_VALIDATION.json`绑定各自历史发布提交，不是本页最新训练状态，也不是本次文档更新的验证清单。旧科学协议、数据角色和原始证据保留，不把软件测试当作科研效果。
