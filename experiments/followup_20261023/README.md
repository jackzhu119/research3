# Paper3 三卡后续实验

3×A100，每卡24CPU/128GB；独立Slurm数组，最多3个元素并行。设计、公式、文献及统计解释见 [DESIGN_ZH.md](DESIGN_ZH.md)。

本轮8种方法×3种子，共24个逻辑任务。现有同配置P/PCT若完成则复用。源EEGNet、参考协方差和现有数据缓存直接复用；不运行全套pytest、不重复pilot、不重扫全部数据；无累计12小时限额。当前旧作业不被修改或取消。

## 开始与继续

将提供的单文件 `START-PAPER3-FOLLOWUP.py` 上传至 `/public/home/zhuziyuan/`。当前实验完成后在login1执行这一整行：

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /public/home/zhuziyuan/START-PAPER3-FOLLOWUP.py --submit
```

同一命令也用于中断后继续：已完成任务跳过，运行中的同组数组复用，未完成任务恢复保存点。不要修改算法后覆盖原实验目录；日志等运行修复可使用同一输出。

只准备而不提交：去掉 `--submit`。安装器不需要联网下载运行代码。学校已有Python/torch环境和已完成源模型必须存在。

程序输出 `SUBMITTED_ARRAY 作业号` 后即进入Slurm调度。默认输出目录：

```
/public/home/zhuziyuan/projects/paper3/runs/followup_20261023
```

## 查看进度

```bash
squeue -u zhuziyuan -o '%.20i %.24j %.10T %.12M %R'
```

```bash
tail -n 60 /public/home/zhuziyuan/projects/paper3/runs/followup_20261023/logs/train-*.out
```

```bash
cat /public/home/zhuziyuan/projects/paper3/runs/followup_20261023/status.json
```

```bash
tail -n 30 /public/home/zhuziyuan/projects/paper3/runs/followup_20261023/monitor.log
```

每30秒本地心跳，每180秒GitHub同步。看 `outer_update` 和验证 `session/sessions_total`，不要把心跳时间本身当成训练进度。`COMPLETE`才代表该任务训练和所有预算评价均完成。某项失败看该任务的`error.json`及out末尾；其他数组元素继续。

GitHub状态： https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup/status.json

监控器使用安装包内已配置的账号授权；与训练独立。网络失败自动重试，监控器退出后再次运行启动命令可恢复监控，不影响已跑完的模型。

## 结果

- `report.json`：受试者配对统计及8个比较的Holm校正。
- `comparisons.csv`：BA百分点差、95%CI、p值、相对误差降低。
- `budget_curves.csv`：B4/8/12/16/20、负翻转、纠错、未来运行段尾部和校准时间。
- `方法-seed种子/`：两代checkpoint、选定结果、逐会话真实预测与错误记录。

本机验证为CPU软件验证，不能代替A100实测和真实EEG效果。若集群禁止requeue，同一启动命令仍可手动恢复。Slurm单次作业时限5天3小时与无累计12小时限额是两件事。
