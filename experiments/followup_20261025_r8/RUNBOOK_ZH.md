# 第八轮运行与恢复

在 login1 执行发布的 `online/RUN-ROUND8.sh` 单行命令。安装阶段采用本地 `.part` 下载续传、固定SHA256、GitHub raw/API备用地址；下载完成后才安装和提交。计算节点已具备整份代码和两个外部裁剪缓存，不依赖后续网络。

固定资源为 `hpc_gpu`，3个独立单卡作业，各16核CPU/64G内存。不是 `hpc_gpu1`；其40GB GPU与80GB组分别记录。总48核CPU/192G主机内存。每卡独立实验，不做跨节点DDP或NCCL通讯。

程序入口 `START.py --root PROJECT --python PYTHON --submit`；省略`--submit`只准备本轮计划。重复启动相同配方会采用现有控制器/作业。不同科学配方使用新的`--output`目录，避免覆盖已有实验。

本轮路径：

```
runs/followup_20261025_r8/
  plan.json
  report.json
  runtime/three_jobs/
    controller.log
    monitor.log
    report.log
    progress.json
    scheduler-submissions/
  TASKKEY/
    progress.json
    runtime.json
    result.json
    latest.json
    state-0.pt
    state-1.pt
    evaluations/PARTITION/B20/clean/PERSON-SESSION.json
```

读取进度的一整行：

```bash
timeout -k 2s 15s tail -n 15 /public/home/zhuziyuan/projects/paper3/runs/followup_20261025_r8/runtime/three_jobs/controller.log /public/home/zhuziyuan/projects/paper3/runs/followup_20261025_r8/runtime/three_jobs/monitor.log
```

学校Slurm的`PENDING/AssocGrpGRES`表示账户GPU配额等待，`Resources/Priority`表示资源或优先级等待；都不能当作训练程序报错。`RUNNING`也不能单独证明完成模型更新。查看`TRAIN_PROGRESS`的`outer_update`或`session`增长。

运行日志遇到Python导入慢会每30秒打印导入心跳。模型/目录读取也有阶段进度。异常会写入任务目录`error.json`及回溯日志。可用任务继续；不会重新执行全量测试或删除先前完成结果。

GitHub沿用学校已配置凭据文件，按现有前轮安装目录发现；可通过`--credentials PATH`明确指定。凭据不进入源码、日志或公共数据包。网络短暂不可用由监控进程重试，不中断GPU进程。

开发端验证包括实际数值、重特征值梯度、断点与优化器恢复、Slurm20提交/未知ACK/重复启动模拟、安装与下载。学校GPU硬件、驱动、共享文件系统和排队由真实作业确认；不能承诺基础设施永不中断。
