# 少样本空间适配失效机制：受控实验 M12

研究问题：**支持集约束弱的空间更新是否参与负迁移，且其影响不能仅由容量或整体预测改变量解释？**

本目录现包含训练实现和可独立安装的学校启动包。资源固定为 **4 个并行作业，每作业 1 GPU、16 CPU、64G 内存**，队列 `hpc_gpu`。所有新结果写入 `runs/r12_controlled_mechanism`，不依赖第一轮完成，不更改 R1/R11 的代码、运行目录或作业。

- [科学设计与预先规定的继续条件](MECHANISM_EXPERIMENT_ZH.md)
- [统计单位、六项主要比较及功效规划](STATISTICS_ZH.md)
- [历史设计配置](EXPERIMENT_CONFIG.json) · [比较清单](PRIMARY_COMPARISONS.csv)
- [代码与运行验证记录](VALIDATION.json) · [交付文件身份](delivery/PACKAGE.json)
- [既有结果证据与原始 SHA](evidence/MANIFEST.json)

## 实验顺序

1. 一次性读取学校现有小型元数据，冻结三个源 seed、已完成 R7 的 static-CE/rank4 字典、原始数据划分和代码身份。之后工作进程缓存源模型及只读数据目录，按需读取 EEG。
2. 32 名开发参与者提供固定 64 事件参考集；另 8 名先导参与者只用支持集进行尺度校准。三个 seed 完成后冻结每预算 τ，才开放先导未来查询的评价。
3. 8 名先导 × 3 seed 的 24 个配对块，均覆盖 B4/8/12/16/20。比较同维 Top/Random 方向、同参考函数 RMS 的普通收缩、真实更新的弱方向/随机方向删除；保留原始与参数范数匹配的次要层。
4. 按预登记规则判断方向假设。即使 STOP，head 竞争和已知人工 shift 的次要解释仍按计划完成。STOP 是有效研究结论，不能误记为训练故障。
5. 方向路线满足条件才进入 11 人开发验证，之后旧 holdout/Pan/Weibo 分别探索复现。准备齐全时共 217 个任务块，实际拟合远多于块数。旧队列不是新独立确认数据；规划中的 90 名新参与者尚未提供，不会生成虚假确认任务。

本轮重新计算 EA、Frozen、完整空间 SGD10/100、head-only 及方向×head 对照。R3/R4/R9/R10 旧方法只作明确标注的历史上下文，不与新试验冒充逐事件配对。

## 避免已有中断的运行设计

- 四个持久工作进程以租约领取独立任务；取得锁后复核完成回执，避免重复执行。已完成 session/预算与 raw-fit 检查点可恢复，不清空旧结果。
- 不设额外进程内存上限；小 JSON 按实际字节数读取。状态和观测写入失败不终止训练。
- 结果使用独立私有暂存目录、唯一文件名及窄范围 I/O 重试。科学结果确实无法落盘时报告故障，不能伪装完成。
- 没有定时 SIGINT 刷新、周期性堆栈转储或全量输入反复预检。Slurm 退出信号请求在持久化单元之间暂停；支持 raw-fit 恢复。
- 单任务错误与重试隔离，其余可运行任务继续；缺失或损坏的科学身份会明确阻断该依赖，不能偷换模型或标签。
- 报告、论文数据上传、完整归档是三个独立后台进程。二进制按 base64 blob 上传，分块 SHA 校验；网络/凭证故障只重试上传，不向训练发送停止信号。
- `sbatch` 确认不明确时等待队列按精确作业名接管，避免重复提交。不会自动取消现有作业；学校配额不足时新作业可能正常排队。

## 安装、启动和恢复

学校启动入口是 [delivery/START-PAPER3-M12.py](delivery/START-PAPER3-M12.py)。只运行文件默认安装代码；显式 `--prepare --submit` 才会冻结计划并提交新作业。准备阶段没有 GPU 拟合。下载命令和固定 SHA 见随交付发布的 `RUN-PAPER3-M12.sh`。

```bash
PY=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
"$PY" -I -S -B -u "$HOME/START-PAPER3-M12.py" --prepare --submit
```

中断连接后可关闭终端，管理与上传进程已独立运行。再次执行同一命令复用原计划、管理进程和完成结果；也可用 `--resume --submit`。只读查看：

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/START-PAPER3-M12.py" --status
```

本地与学校同版本科学栈（Torch2.5.1 CPU版／NumPy2.2.6／SciPy1.15.3／sklearn1.6.1）均通过160项测试。CPU 验证覆盖真实 CompactEEGNet 梯度、算法配对与尺度匹配、四进程领取、退出恢复、写入故障、上传隔离及实际安装包。学校 GPU、文件系统和调度器的实地验证尚不能由本机 CPU 测试替代；不承诺未来所有本地库或节点故障均不会发生。历史设计离线审计仍可运行 `python -I -S -B audit_design.py`，它只审计设计和证据。
