# 第三轮学校服务器运行说明

## 1. 与前两轮的关系

本轮只读取原工程中的冻结源模型、参考协方差和缓存。新代码安装在 `~/paper3-round3/<包hash>/`，新结果默认写入 `/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r3`，不覆盖第一轮 `training_campaign_v1`、`source_training_v2`、`meta` 或第二轮 `followup_20261023_v2`。符合种子、源模型、数据身份和选择规则的已完成P字典可读取复用；否则独立训练。程序不会为了启动本轮而取消前两轮。

默认3个并行单卡任务，每个12核CPU、64G内存，队列 `hpc_gpu`，GRES为站点通用 `gpu:1`。这是最新学校配额记录，不是原第一轮 `hpc_gpu1` 的80GB配置。如果当前账户仅允许3个已提交任务，旧轮任务占着配额，新轮会显示 `WAITING_FOR_CAPACITY`，有空位再提交。所有PENDING也计入配额。

## 2. 启动与续跑

线上单文件安装器 `START-PAPER3-R3-ONLINE.py` 自带本轮源码，使用现有Python，不需要git pull或覆盖旧工程。默认只准备，指定 `--submit` 才开始滚动提交。最终固定下载地址及SHA在发布后的 `ONLINE_LAUNCH.json` 中记录。

已安装本地包时，在login1执行：

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /绝对路径/本轮安装包/START.py --gpus 3 --cpus 12 --submit
```

资源允许时可改成 `--cpus 24 --memory 128G`。已有活动控制器时不热改其资源；等该控制器退出再用新的调度参数续跑。科学参数不可在同一个结果目录中静默修改；新方法或新先验用新的输出目录，避免把不同实验混成一次。

入口返回 `PREPARED` 表示已生成计划；`DETACHED_PROCESSES_STARTED` 仅表示login节点控制器与监视器启动；控制器日志中的 `SUBMITTED` 才有Slurm提交回执。节点日志出现 `TRAIN_PROCESS_STARTED` 表示批脚本开始，随后实际 `outer_update` 或已完成会话数增长才能判断训练、评价在推进。不能用网页RUNNING或静止心跳推断模型已更新。

网络或电脑断线不会停止已分离的控制器。如果login节点重启，再执行同一条启动命令即可。启动器和控制器锁会避免同一个实验重复提交。已完成任务读 `result.json` 复用，未完成字典读最近有效断点，未完成评价只补缺失会话。

## 3. 一次查看全部进度

```bash
(R=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r3; date; squeue -u zhuziyuan -o '%.20i %.28j %.10T %.12M %.20R'; tail -n 35 "$R/controller.log" "$R/monitor.log"; /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -S -u -c 'import json,pathlib; p=pathlib.Path("/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r3/status.json"); d=json.loads(p.read_text()) if p.is_file() else {}; print(json.dumps({k:d.get(k) for k in ["updated_utc","complete_tasks","total_tasks","controller_status","report_status","primary_family_complete"]},ensure_ascii=False,indent=2)); [print(json.dumps(t,ensure_ascii=False)) for t in d.get("tasks",[]) if t.get("status") not in ["NOT_STARTED","NEW","COMPLETE"]]')
```

节点标准输出和错误分别在 `logs/train-作业号_任务索引.out/.err`，路径在提交前创建。控制器在 `controller.log`，GitHub同步在 `monitor.log`。任务目录有 `progress.json`、`error.json`、`result.json`；数据和公式错误写出原因，不冒充完成。

`COMPLETE/174` 是逻辑任务覆盖比例，不是GPU时间百分比。21个字典任务通常更耗时，其他校准评价成本较低；先完成的旧B求解器比较会较早产出数据。待前几个真实字典与评价完成后，用实测每类耗时估计剩余时间；目前不提供无GPU实测支撑的小时数。

## 4. 保存、故障和限制

字典每10次更新写双代checkpoint及SHA指针，包含B、优化进度与Python/NumPy/Torch/CUDA随机状态；验证过程也分段提交。最近一代损坏时尝试前一代。每50更新及最后一次做元验证；最佳字典按验证集BA、BCE、最早更新决定，holdout不参与选择。

每个支持预算、压力条件、参与者、会话分别原子保存评价；未来查询标签不参与适配。报告或GitHub同步失败不触发重训，也不改变数值配方。没有每次重跑全量test/pilot/21,262文件SHA的启动步骤；读取实际checkpoint和所需缓存仍然是训练必要I/O。

不存在累计12小时墙钟上限。单次Slurm时限默认 `5-03:00:00`，可按学校支持的时限传 `--time`；不能替学校保证无限单次分配。提前300秒的USR1请求保存，退出码75由控制器续交。节点启动失败有有限自动重试；明确科学错误的任务标为FAILED，依赖它的任务等待，其他可运行任务继续。修复数值实现后保留原实验，按新配方建立新输出；仅运行或报告修复不必重训。

显式scancel或STOP被当作用户停止，不反复自动复活。此时需明确清理本轮对应停止标记再恢复。不要删除checkpoint或已完成结果。

## 5. GitHub与最终结果

沿用学校现有私有JSON凭据：`PAPER3_GITHUB_CREDENTIALS`、`~/.config/paper3/github.json` 或旧安装目录 `~/paper3-followup/*/private/github.json` 自动发现，也可传 `--credentials /路径/github.json`。凭据不写入公开源码和实验计划。

监视器每180秒发布 `paper3-r3-progress:monitoring/paper3-round3/status.json`，只包含任务进度、聚合BA/变化量/耗时和统计表。GitHub断连时在本地继续生成status并重试。全量结果始终在学校本地：任务目录中的checkpoint、逐会话预测，以及 `reports/report.json` 与各CSV表。不能用尚未生成p值判断训练被阻塞：p值只在10项主要比较所需各组的3种子及匹配会话齐备后计算，其他结果先描述性回传。

## 6. 可选外部验证

建议核心轮完成后，再启动独立Weibo18任务，避免与核心轮额外争配额。它读取本轮已完成字典，独立输出 `runs/followup_20261024_r3_weibo`，不从外部数据重新学习字典，不改变核心统计计划。公开状态使用独立 `weibo-status.json`。

优先下载已校验的约11MB衍生缓存；其许可、来源、archive SHA与相对路径清单一同发布。缓存包含10人LR子任务的预处理信号和事件；不含4.34GB原MAT。若自行从原数据重建，使用 `prepare_weibo.py --help`，该步骤在训练前单独完成，训练工作者不联网取数。

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /绝对路径/本轮安装包/START.py --external-manifest /绝对路径/便携缓存/manifest.json --basis-run /public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r3 --gpus 3 --cpus 12 --submit
```

Weibo仅支持发布顺序LR前缀预算稳健性，物理run未知；不输出物理run风险指标，不把其1380查询当成1380个独立统计受试者。
