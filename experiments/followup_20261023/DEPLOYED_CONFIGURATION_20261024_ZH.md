# 第二轮V2：学校当前部署记录（2026-10-24）

科学代码与66任务设计仍保留原版本；学校运行中的提交方式已在分享聊天中另行迁移。本文用于避免把仓库原始启动器默认资源误认为现有分配。

- 当前是`hpc_gpu`分区**三个独立单GPU作业滚动补交**，每项12CPU/64G主机内存，已选A100 PCIe40GB，可跨gpu1/gpu2节点。
- 第一轮`770043`继续在`hpc_gpu1`/gpu4使用1GPU/24CPU/128G。最近同时分配4GPU/60CPU/320G，不是一个四卡模型。
- 第二轮任务表与结果目录仍为`/public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/`；原包`/public/home/zhuziyuan/paper3-followup/0e55dba2a78b43464e8c/`。
- 学校当前迁移工具目录`/public/home/zhuziyuan/paper3-submit-three40/`；`switch.log`记录切换，历史三作业控制日志为`_threejobs16/controller.log`，最终40GB控制进程是否沿用未独立核实。目录名不是当前CPU申请量。
- 12项seed20261018核心任务完成；读取的自动快照时间为北京时间10月24日15:45:44，第二seed三项在1000/1000/950步训练或评价，报告PARTIAL。作业号只代表那次队列观测。
- Github同步路径为[monitoring/paper3-followup-v2/status.json](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)；其中1GPU/24CPU资源元数据尚是旧计划字段，不代表当前3×12CPU/64G现场分配。

当前控制器已接管后续补交。原`START.py --submit`或本目录云端旧入口会按原始整批数组方式申请，曾触发`QOSMaxSubmitJobPerUserLimit`；不能把它们当成当前运行控制器的日常重启方式。查看记录不需要再次安装、迁移、提交或停止任务。

完成项按`result.json=COMPLETE`跳过；未完成项使用原身份匹配的断点和评估分片。变更仅涉及提交/资源，没有重编号66任务或改变算法、学习率、数据划分和科学预算的证据。3个任务分别运行，模型自身没有改成DDP或扩大batch。

原24CPU/128G、等待第一轮结束、单四卡以及三作业16CPU的说明作为历史保留。完整且自包含的[变更经过、证据、数据保存与续跑定位](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/PAPER3_CONFIGURATION_HANDOFF_20261024_ZH.md)已存默认分支，不依赖分享聊天仍可打开。

本页是配置记录，未部署新代码，未修改运行中的控制器或动态状态。学校本地模型/checkpoint文件不因GitHub进度已上传就自动构成完整异地备份。
