# 学校HPC：当前执行入口说明

当前原实验正在学校运行（依据研究者反馈，具体日志最近截至2026-10-23 14:59:30北京时间）；下一轮66项三卡实验已准备、尚未提交。**当前和后续都不再使用RunPod。**

- [当前原实验与下一轮V2的区别](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/CURRENT_AND_NEXT_ZH.md)
- [学校环境、日志与断点约定](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/SCHOOL_HPC_ENVIRONMENT_ZH.md)
- [下一轮66项实验设计与代码](https://github.com/jackzhu119/research3/tree/paper3-v2-progress/experiments/followup_20261023)

现有学校解释器为`/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python`，Python3.12.3 / torch2.5.1+cu121 / NumPy2.2.6；GPU分区为hpc_gpu1。运行中的原实验沿用已部署入口及原目录；新实验另用`runs/followup_20261023_v2`。

## 查看当前任务

```bash
squeue -u zhuziyuan -o '%.20i %.24j %.10T %.12M %R'
```

最近日志中的作业为770043；新续跑作业号以学校队列为准。结合Slurm状态、实际step和checkpoint变化判断进度，网页心跳不是模型已完成的证明。

## 恢复原则

已完成训练和pilot复用；恢复从断点和已完成评价分片接续。当前使用的直接训练及下一轮独立worker不再以全套test、pilot、全库SHA扫描或旧备份审批作为每次续跑前提。没有人为累计12小时限制；Slurm的单次时限与重排队仍依学校配置。

本次只更新文档，不修改学校作业、不提交新实验、不要求正在运行的目录执行git pull。此分支保留历史恢复脚本；不要因为看到了旧工具就重走初始test→prepare→train链。

## 历史恢复记录

旧HPC部署和故障排查全文见[历史记录](history/DEPLOYMENT_RECOVERY_ZH_20261023.md)。其中针对767487、767530、767958等旧作业的特例、43200秒预算、固定GPU UUID及重新测试要求，只解释当时恢复方案，不作为当前/下一轮通用条件。

仓库中的`ONLINE_RELEASE.json`、`ONLINE_VALIDATION.json`继续绑定其原固定发布提交；没有把旧清单改写为本次页面更新的验收证明。
