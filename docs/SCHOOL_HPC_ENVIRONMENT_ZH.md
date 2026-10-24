# 学校HPC环境、当前分配与已知调度行为

更新日期：2026-10-24（北京时间）。学校Slurm为论文3当前运行平台，不再使用RunPod。下面区分实际申请、历史硬件记录和程序参数；不是实时节点盘点。

| 项目 | 当前记录 |
|---|---|
| 账号/登录 | zhuziyuan / login1 |
| 第一轮 | `hpc_gpu1`、A100 SXM4 80GB记录、1GPU/24CPU/128G |
| 第二轮 | `hpc_gpu`、已选A100 PCIe40GB、每作业1GPU/12CPU/64G，最多3个独立作业滚动补交 |
| 节点与分区 | gpu1/gpu2/gpu4是节点；hpc_gpu/hpc_gpu1是分区，名称不能互换 |
| 实测版本历史 | Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6；Slurm20.11.8；EL7时代Linux3.10、Bash4.2/glibc2.17兼容基线 |
| 解释器 | `/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python` |
| 共享项目根 | `/public/home/zhuziyuan/projects/paper3` |

40GB/80GB是GPU显存容量；64G/128G是Slurm申请的主机内存，不能混为一谈。用户最新TRES输出直接确认卡数和CPU/主机内存，未打印设备型号；所有剩余模型的峰值显存并未逐项核实。`sacct MaxRSS`也是主机指标，不能用于证明40GB显存一定足够。

已有硬件记录：gpu3为76CPU、8张80GB A100、RealMemory1031528MB；聊天中的hpc_gpu节点截图解读为56CPU、5张40GB卡、约251GiB主机内存。它们不是每个节点的统一保证，也不是当前空闲量。同节点所需CPU、GPU、内存必须同时满足；多个独立单卡作业则可以跨节点分配。

## 提交限制与迁移经验

- 原66元素数组即使使用`%2/%3`，仍可能因提交计数触发`QOSMaxSubmitJobPerUserLimit`；当前使用少量单元素作业滚动补交。
- `Priority`、`AssocGrpGRES`、`JobHeldUser`分别表示优先级、关联组资源配额、暂挂原因，不能全归为训练失败。
- 当前“三作业并发”是实验调度策略，不是可据此推断的学校最大作业/显卡配额。聊天曾提10提交/4运行，当前摘录未包含原始数值表，不能当永久政策。
- 单个四卡作业要求单节点同时满足卡数、CPU与内存，且受账号配额影响；曾经失败/排队的四卡方案不应误当成当前部署。
- 第一轮与第二轮分开保存输出。当前提交控制器已经负责第二轮补交，归档文档不需要启动新控制器或重复部署旧入口。

## CPU与环境

每项申请12CPU不意味着数学库自动用12线程。原worker为`torch.set_num_threads(4)`，OMP/MKL4，OpenBLAS1；本次资源迁移未给出这些计算参数改动的证据。24核相对12核的真实速度差异未做受控测量，不能从申请核数承诺线性提速。

`cu121`是PyTorch构建，不证明系统nvcc/驱动版本。原环境继续复用，不在训练时升级依赖或联网安装。`-I -S`隔离入口自行补充site-packages；不能给任意脚本加`-S`就期待自动导入torch。gpu3曾有无PID/日志问题，根因未完全确证，不写成已证实硬件损坏。

## 日志、结果与恢复

共享`/public/home`保存结果和checkpoint；`/tmp`只作可重建缓存。第二轮独立task目录中的`result.json=COMPLETE`是完成依据；未完成项用原checkpoint和随机状态继续，已完成会话分片可复用。作业号更换、新种子从0开始和验证阶段停留不是从头重做整轮的证据。

shell先打印job/task/node/time，Python无缓冲，原worker定期输出实际step/session；USR1/TERM在正常路径中请求保存，requeue取决于学校配置。没有累计12小时的人为限额，但仍有Slurm单次时限与保存间隔；这些机制不保证强杀后的零重放或跨卡逐位相同。

GitHub第二轮持续同步已核实，第一轮仍是人工快照。模型文件主要保存在学校服务器，上传进度不等于完整异地备份。新旧上传路径独立；网络失败不要求重新跑全量pytest、pilot或数据恢复。

详见[完整配置交接](PAPER3_CONFIGURATION_HANDOFF_20261024_ZH.md)及[现有资源快照](HPC_RUN_STATUS_20261024_ZH.md)。本次只更新文档与人工概览，不改变任何运行作业、安装文件或数值训练设置。
