# Paper3 三卡后续实验 V2

> **2026-10-24当前部署更新：第二轮已与第一轮并行运行。** 学校现用`hpc_gpu`三个独立单GPU作业滚动补交，每项12CPU/64G、已选A100 PCIe40GB；当前运行控制器不使用本页原始整批启动命令。[当前部署与交接记录](DEPLOYED_CONFIGURATION_20261024_ZH.md)。下文24CPU/128G、整批数组及“等第一轮结束”的内容保留为原始交付历史，不要据此重复安装或提交。


3×A100，每卡24CPU/128GB，最多3个独立Slurm数组元素并行。完整算法、14篇文献、比较矩阵和结论边界见 [DESIGN_ZH.md](DESIGN_ZH.md)。

**36个核心元训练＋24个参数敏感性元训练＋6个无元训练参考＝66个任务。** 核心12方法×3种子；额外5档校准预算、6种预定义误标/空间漂移诊断与概率指标。源EEGNet、参考协方差和缓存复用；已有匹配P/PCT任务可复用。不重跑全套pytest、pilot或全库SHA，不设置累计12小时限额。当前旧作业不被修改或取消。

## 开始与继续

将新版单文件 `START-PAPER3-FOLLOWUP.py` 上传至 `/public/home/zhuziyuan/`。当前实验完成后在login1执行：

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /public/home/zhuziyuan/START-PAPER3-FOLLOWUP.py --submit
```

同一命令用于恢复：完成项跳过，运行项复用，仅补未完成且未运行的项；补交数组等待旧数组结束，保持全局最多3卡。训练从已存参数和Adam状态继续，验证/评估已完成的会话也复用。只准备而不提交时去掉 `--submit`。

安装文件内含完整代码和已配置的GitHub同步账户，不需联网下载依赖或补丁。学校现有虚拟环境、三个已完成源模型和缓存仍是实际输入。算法或数据选择变更使用新输出目录；日志/提交修复可继续同一数值实验。

默认目录：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2`。`SUBMITTED_ARRAY 作业号`表示提交成功；`TRAIN_PROCESS_STARTED`表示计算节点已开始执行shell。

## 进度

```bash
squeue -u zhuziyuan -o '%.20i %.24j %.10T %.12M %R'
```

```bash
tail -n 60 /public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/logs/train-*.out
```

```bash
cat /public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/status.json
```

```bash
tail -n 30 /public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/monitor.log
```

本地每30秒输出阶段、outer_update或评价session计数。只看到心跳时间增长不等于模型步数增长。导入期间也有独立心跳；完整traceback进入out和任务error.json。单任务失败不会取消其他任务。

GitHub每180秒同步： https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json

监控器独立运行，网络/报告失败自动重试，不阻断训练。完成数变化即更新部分描述报告；核心36项全部完成后给固定10比较的Holm校正。监控器退出时重跑同一启动命令即可恢复同步，不重做完成项。

## 输出与验证

- `report.json`、`comparisons.csv`：B20受试者配对差、95%CI、固定10比较Holm校正。
- `budget_curves.csv`、`budget_comparisons.csv`：B4/8/12/16/20，始终同一未来查询集。
- `reference_comparisons.csv`：冻结f0、只调头、支持协方差对齐参考。
- `sensitivities.csv`：P/dual的秩和变换幅度敏感性。
- `diagnostics.csv`、`diagnostic_comparisons.csv`：标签误标及未来空间漂移，均为人工诊断。
- 各任务目录：两代checkpoint、选模来源、真实逐会话预测概率、门值/梯度及异常记录。

无累计12小时限额；Slurm单次申请5天3小时，提前300秒保存后尝试requeue。集群若不允许requeue，同一命令可手动续跑。管理员强杀或存储故障仍可能发生，最近保存之后的少量步骤可能重做。

验证使用Python3.12.3、torch2.5.1+cu121、NumPy2.2.6真实发行包；覆盖完整合成数据流程、断点、信号、统计、同步。详见 `VALIDATION.json` 和 `compatibility/`。当前主机无GPU：这不是学校A100/驱动/共享存储实测，也不包含虚构EEG效果或“保证一次跑完”的承诺。
