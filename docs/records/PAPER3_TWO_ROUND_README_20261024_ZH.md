# 论文3 · 学校HPC两轮并行实验

更新日期：2026-10-24（北京时间）。两轮均使用学校Slurm云服务器，不再使用RunPod。本页保存可追溯的配置与观测快照，不是Slurm实时查询器。

**第一轮原实验和第二轮V2均已实际启动；第二轮不是“尚未提交”的未来方案。** 分享聊天中的配置变更已独立整理到[完整交接记录](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/PAPER3_CONFIGURATION_HANDOFF_20261024_ZH.md)，删除原分享链接后仍可查阅。

| 项目 | 第一轮：原论文3主线 | 第二轮：扩展V2 |
|---|---|---|
| 最近资源记录 | `770043`，`hpc_gpu1`，节点`gpu4` | `771066_12`、`771065_13`、`771064_14`，`hpc_gpu`，节点`gpu1/gpu2` |
| 每作业资源 | 1 GPU、24 CPU、128G主机内存 | **1 GPU、12 CPU、64G主机内存**；最多3个独立作业 |
| GPU配置记录 | A100 SXM4 80GB | 已选A100 PCIe 40GB；终端TRES记录直接证明GPU数量，型号来自既有硬件/迁移记录 |
| 最近科学进度 | 10月24日12:40:59日志：B11、r=2、ρ=.05、seed20261018，199步 | 10月24日15:45:44自动快照：**12/66 COMPLETE**；第二个种子3项在训练/验证/留出评价 |
| 实验范围 | 源模型、45组元筛选、10组元确认、开发评估及主线内部检查 | 36核心元训练＋24敏感性元训练＋6参考；另有预算曲线和诊断评估 |
| 结果目录 | `runs/source_training_v2/`、`runs/meta/`、`runs/training_campaign_v1/` | `runs/followup_20261023_v2/` |
| GitHub回传 | 有人工归档快照；未确认持续自动回传 | [独立进度与部分汇总](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)持续提交，报告仍为`PARTIAL` |

最近现场同时分配合计4 GPU、60 CPU、320G主机内存。节点名`gpu1`不是分区`hpc_gpu1`；第二轮可以分散到不同节点，每个模型仍是独立单卡计算，不是DDP。

## 当前使用的提交方式

第二轮已采用**三个独立单GPU作业滚动补交**，完成一个再接后续未完成任务。一次提交66个数组元素、`%2/%3`整批提交、单个四卡作业及三作业16CPU/128G方案均属于此前尝试。当前已部署的40GB提交控制器负责后续补交，查看本页无需重新运行旧安装器或`START.py --submit`。

服务器安装路径记录：`/public/home/zhuziyuan/paper3-submit-three40/`；切换日志`switch.log`；历史三作业控制日志为`runs/followup_20261023_v2/_threejobs16/controller.log`；最新40GB控制进程是否继续使用此文件尚未独立核实。当前切换定位优先看`paper3-submit-three40/switch.log`，不根据旧目录名推断资源。

## 交接入口

- [完整配置、变更经过、续跑约定与证据](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/PAPER3_CONFIGURATION_HANDOFF_20261024_ZH.md)
- [两轮状态与后续阶段](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/CURRENT_AND_NEXT_ZH.md)
- [机器可读实验配置](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/EXPERIMENT_STATUS.json)
- [学校环境与调度经验](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/SCHOOL_HPC_ENVIRONMENT_ZH.md)
- [此前资源快照（保留原记录）](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/HPC_RUN_STATUS_20261024_ZH.md)
- [第二轮完整算法设计](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/experiments/followup_20261023/DESIGN_ZH.md)
- [第二轮自动同步结果](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)

## 保存与解释

模型、两代checkpoint、逐会话预测和日志先写学校共享目录`/public/home/zhuziyuan/projects/paper3`；GitHub进度上传失败不等于本地计算结果丢失。GitHub目前回传进度与部分汇总，未据此确认全部checkpoint已异地备份。

已完成项以各任务`result.json.status == COMPLETE`为准；换作业号或新种子从0开始不代表前12项重算。未完成项由原worker读取匹配断点，调度/迁移本身不证明每次恢复都逐位等价。已完成1000步而仍在验证/留出评价的任务也还不能计为完整完成。

本次配置归档只修改文档与人工概览JSON，不部署学校代码、不重启或取消作业、不修改数据划分或科学训练参数，也不覆盖自动同步的`monitoring/.../status.json`。旧RunPod入口、最初24任务V1和原80GB整批启动说明保留为历史资料。

