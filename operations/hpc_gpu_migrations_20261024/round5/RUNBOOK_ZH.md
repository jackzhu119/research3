# 第五轮：一个节点、一个作业、三卡并行

## 资源含义

第五轮提交到 `hpc_gpu` 分区，使用图中A100-PCIe-40GB资源组。申请一个节点、一个普通Slurm作业，无数组作业；节点内三个GPU工作进程动态领取实验任务。

| 项目 | 第五轮配置 |
|---|---|
| GPU | 同一节点三张，`--gres=gpu:3` |
| CPU | 整作业36核，三个进程各12核亲和组 |
| 主机内存 | 整作业共享128G，每槽等分规划约42.67GiB |
| 调度参数 | `--nodes=1 --ntasks=1 --cpus-per-task=36 --mem=128G` |

`--cpus 12` 按每个GPU槽解释；**`--memory-total 128G` 按整个作业解释，不乘三，也不舍入成126G或129G**。不设置进程虚拟地址空间限制，避免CUDA映射造成误判。旧轮 `--memory 64G` 是不同的每槽语义，第五轮不接受该旧参数。内存预算与GPU显存为不同资源。

`hpc_gpu` 是分区/资源组名，不是已知主机名。Slurm负责找到同节点三卡和36核CPU、128G内存；分区总空闲三卡不等于当前同一节点已满足全部条件。资源不足时正常排队，不猜 `--nodelist`，不自动改队列。

## 启动和续跑

线上固定版本单文件 `START-PAPER3-R5-ONLINE.py` 包含本轮源码；最终URL及SHA见 `online/ONLINE_LAUNCH.json`。使用学校现有Python环境，不在作业内安装依赖。新安装目录为 `~/paper3-round5/<包hash>/`，默认结果目录为 `/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5`，外测单独使用 `_r5_weibo`。

已安装后的等效命令：

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /绝对路径/第五轮安装包/START.py --partition hpc_gpu --gpus 3 --cpus 12 --memory-total 128G --submit
```

没有 `--submit` 时只准备计划；带该参数时先启动监控，再启动后台单作业控制器。终端/本地电脑断线不停止已分离的login控制器。相同命令会复用已完成结果与断点，并通过锁和提交回执避免重复提交同一份三卡作业。login节点重启后可再次运行相同命令，对仍在运行的作业进行对账采用。

前三轮及第四轮源码、配置、作业和结果均保持独立。冻结源模型、数据和数值身份匹配的已完成字典可作为只读输入；不会为了第五轮取消其他作业或重复训练已验证完成的任务。

## 判断进度

`PREPARED` 只代表计划和资源配置已写好；`DETACHED_PROCESSES_STARTED` 代表login后台进程已启动；回执中的 `job_id` 表示学校接受了一份作业。`QUEUED` 会给出资源/配额等待原因。`TRAIN_PROCESS_STARTED` 表示节点批脚本开始；三槽 `TRAINING`/`EVALUATING` 以及更新步数或会话数增长才表示实际计算在推进。

在login1执行下面整块，一次读取队列、主控和实际三槽进度：

```bash
(
R=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5
date
squeue -u zhuziyuan -o '%.20i %.28j %.12T %.12M %.40R'
tail -n 25 "$R/controller.log" "$R/monitor.log"
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -S -u -c 'import json,pathlib,time; r=pathlib.Path("/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5"); keys={"updated_utc","allocation_status","controller_status","job_id","complete_tasks","total_tasks","failed_tasks","waiting_tasks","slots","gpu_slots","requested_resources","observed_allocation_resources","report_status","primary_family_complete"}; [(print(str(p),"更新距今秒",round(time.time()-p.stat().st_mtime)),print(json.dumps({k:v for k,v in json.loads(p.read_text()).items() if k in keys},ensure_ascii=False,indent=2))) if p.is_file() else print(str(p),"尚未生成") for p in [r/"allocation_progress.json",r/"status.json"]]'
)
```

节点主日志为 `logs/train-作业号.out/.err`；每30秒输出三个GPU槽的任务和实际进度。具体任务日志为 `logs/task-索引-attempt次数.out/.err`。任务目录还保存 `progress.json`、`error.json`、`result.json`。任务完成比例不等于GPU时间比例，字典训练和一次会话评价成本不同。

## 保存、故障和GitHub

保留计划规定的字典双代checkpoint、随机状态和逐会话原子评价保存。最近一代checkpoint损坏时可读取前一代。已完成结果复用，未完成评价补缺失会话。单个科学任务失败记录错误，不取消其他独立任务，也不把失败冒充完成。依赖它的任务保留等待状态。

学校启动不重跑全量pytest、pilot或全树数据扫描，不重新安装环境；读取实际需要的模型和缓存仍是必要I/O。无累计12小时墙钟上限。单次Slurm时限默认 `5-03:00:00`，到期前300秒的USR1请求三个工作进程保存，退出75后控制器对账续交一个作业。明确scancel/TERM或STOP属于主动停止，不自动复活。数值配方更改使用新输出目录，保留旧结果。

GitHub每180秒尝试同步至 `paper3-r5-progress:monitoring/paper3-round5/status.json`；外测使用独立 `weibo-status.json`。已有R2/R3/R4/R5私有JSON凭据可自动发现，也可传 `--credentials /路径/github.json`。同步或报告失败独立重试，训练继续。公开回传汇总数据；checkpoint、逐会话预测、`reports/report.json`与完整CSV仍在学校本地。

程序验证以 `VALIDATION.json` 为准。本地软件、模拟Slurm及真实CPU子进程测试不能代替学校三卡GPU实跑，也不代表识别效果已提升。
