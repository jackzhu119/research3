# 第四轮：单节点、单作业、三卡并行

## 资源与前轮隔离

本轮提交到图中 `hpc_gpu` 分区，申请一个节点、一个Slurm作业，节点内三张A100-PCIe-40GB同时工作。`hpc_gpu` 是队列名，不是已知的物理主机名；不会添加猜测的 `--nodelist`，也不会自动退回其他分区。

| 资源 | 每个GPU工作槽预算 | 整个作业申请 |
|---|---:|---:|
| GPU | 1张 | 3张，`--gres=gpu:3` |
| CPU | 12核 | 36核 |
| 主机内存 | 约42.67GiB显示预算 | 128G共享总量 |
| 节点 / Slurm作业 | 同一节点的三个工作槽 | 1节点、1作业 |

具体Slurm参数为 `--nodes=1 --ntasks=1 --cpus-per-task=36 --mem=128G --gres=gpu:3 --partition=hpc_gpu`。单batch进程获得整个36核CPU亲和集合，再将其分成三个互不重叠的12核组；三个工作进程各只看见分配给自己的GPU，并使用 `cuda:0`。这里的 `ntasks=1` 是批调度实现方式，GPU工作进程仍有三个。使用原Linux3.10、Bash4.2及Slurm20.11可用接口，不依赖新版 `--gpus-per-task` 或额外 `srun`。

128G为作业共享主机内存申请，每槽约42.67GiB仅为显示预算；不对进程设置虚拟地址空间上限，避免CUDA映射被错误当作物理内存耗尽。公开进度记录实际观察到的CPU数，分配请求不冒充现场观测。截图中的瞬时空闲资源若不足36CPU、128G及同节点3GPU，作业会正常排队，GPU空闲三张本身不代表CPU也足够。

新代码安装在 `~/paper3-round4-hpcgpu/<包hash>/`。默认输出 `/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r4`，外部评价单独使用 `_r4_weibo`。读取原冻结源模型与缓存，符合数值身份的旧字典可只读复用；不写第一、二、三轮代码、结果或运行配置，也不取消它们的作业。本轮任务总数以同一版本 `plan.json` / `EXPERIMENT_MATRIX.json` 为准。

## 启动与续跑

固定版本单文件 `START-PAPER3-R4-HPCGPU.py` 自带本轮源码，使用学校已经安装的Python环境。最终发布URL和SHA在 `online/ONLINE_LAUNCH.json` 记录；下载后验证SHA，再执行即可。默认只准备，传 `--submit` 才启动后台监控与单作业控制器。

已安装包的等效命令：

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /绝对路径/本轮安装包/START.py --partition hpc_gpu --gpus 3 --cpus 12 --memory-total 128G --submit
```

`--cpus 12` 按每GPU槽解释，整作业36CPU；`--memory-total 128G` 是整个作业共享主机内存，不乘三，也不舍入为126G或129G。`--gpus 3` 表示同一作业中的三卡，并非三个独立Slurm作业，也没有数组作业。

本包专用于原第四轮资源迁移。线上安装器先进入 `MIGRATE.py`，协调原第四轮控制器和确切同计划作业：运行中的任务先请求保存，不强杀GPU；等待旧作业清理后，归档旧操作状态并调用新 `START.py --submit`。迁移不取消其他轮作业，不改原计划、科学源码、已完成结果或断点身份。旧计划中的CPU字段仍可保留16；实际新申请以 `runtime/launch.json` 的12CPU/槽、128G总量及Slurm回执为准。

若只是同资源文档重打包，新安装路径不会要求中断已有控制器；资源确实不同时由迁移入口完成交接，不直接覆盖一个仍在使用旧配置的控制器。

同一命令可以续跑。重复执行时锁与持久提交回执会采用已有控制器/已有作业，避免重复占三张卡。计算机断网或关闭本地终端不停止已经分离的login控制器。login节点重启后再次执行同一命令即可采用仍在运行的学校作业；若回执尚不能判定是否已提交，先对账等待，不盲目再交一份三卡任务。

启动标志需区分：

- `PREPARED`：已生成计划与资源配置，没有现场训练声明。
- `DETACHED_PROCESSES_STARTED`：login后台控制器和监视器已启动。
- 控制器回执中的 `job_id`：学校接受了一个普通Slurm作业。
- `QUEUED` / `QUEUED_ALLOCATION`：作业在排队，`queue_reason` 显示资源或配额原因。
- `TRAIN_PROCESS_STARTED`：节点批脚本已执行。
- `ALLOCATION_PROGRESS`：同一作业的三槽分配器在调度，查看各槽实际 `worker_progress`。
- `TRAINING` / `EVALUATING` 配合更新步数或完成会话数增长：实际训练或评价在推进。

`ALLOCATION_RUNNING` 只表明Slurm分配在运行；它与已经完成优化更新是两个证据层级。任务完成比例也不是GPU时间百分比，长短任务成本不同。

## 一次查看完整进度

在login1粘贴执行下面整块，只读已有文件：

```bash
(
R=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r4
date
squeue -u zhuziyuan -o '%.20i %.28j %.10T %.12M %.35R'
tail -n 30 "$R/controller.log" "$R/monitor.log"
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -S -u -c 'import json,pathlib,time; r=pathlib.Path("/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r4"); ps=[r/"allocation_progress.json",r/"status.json"]; keys={"updated_utc","allocation_status","controller_status","job_id","complete_tasks","total_tasks","failed_tasks","waiting_tasks","slots","gpu_slots","requested_resources","observed_allocation_resources","report_status","primary_family_complete"}; [(print(str(p),"更新距今秒",round(time.time()-p.stat().st_mtime)),print(json.dumps({k:v for k,v in json.loads(p.read_text()).items() if k in keys},ensure_ascii=False,indent=2))) if p.is_file() else print(str(p),"尚未生成") for p in ps]'
)
```

整个节点作业日志在 `logs/train-作业号.out/.err`，批脚本一开始就打印作业、节点和三槽身份。三个槽运行的具体任务分别写 `logs/task-索引-attempt次数.out/.err`；节点主日志每30秒输出三槽进度。任务目录还有 `progress.json`、`error.json`、`result.json`。控制器在 `controller.log`，GitHub同步在 `monitor.log`。

## 连续运行、保存与错误处理

同一作业中的分配器动态领取依赖已经完成的任务。一张卡完成任务后马上领取下一个可运行任务；三张卡无需等待同一批次一起结束。单个任务失败时记录错误，该槽继续其他独立任务，依赖失败结果的任务保留等待状态；不会取消其他槽，也不将失败伪装为完成。

已完成 `result.json` 直接复用。字典训练按计划间隔保存轮换checkpoint与SHA指针，最近一代损坏可尝试上一代；包括字典、进度与随机状态。评价按支持预算、条件、参与者和会话原子保存，恢复时补缺失项。实际保存粒度以该版本worker实现和计划为准。未来查询标签不参与支持集选择或适配。

学校启动时不重跑全量pytest、pilot或21,262文件扫描，也不在作业中重新安装环境。读取实际需要的模型与缓存仍是训练必要I/O。没有累计12小时墙钟限制；单次Slurm时限默认 `5-03:00:00`，按学校支持时限可传 `--time`。

到达单次时限前300秒，USR1发给整个三槽分配器，再请求各工作进程保存；退出75后控制器对账并续交一个三卡作业，已完成任务和断点保留。明确 `CANCELLED`、用户TERM或STOP作为主动停止，不自动复活。科学失败不通过无限重复整个分配来掩盖；修复后继续保留原结果。运行/报告层修复不改变数值实验，数值配方变化用新输出目录。

QoS拒绝只重试一个三卡作业，逐渐延长等待至最多300秒间隔；不会连续提交大量数组任务。调度器观测暂时不可用时暂停新提交，恢复后对账，已有节点计算照常运行。

## GitHub与本地结果

监视器每180秒尝试同步到 `paper3-r4-progress:monitoring/paper3-round4/status.json`；外部阶段使用 `weibo-status.json`，不覆盖核心或前轮状态。网络、报告或同步失败不使模型训练退出。本地仍保存checkpoint、逐会话预测、`reports/report.json`及CSV汇总，GitHub回传任务进度与聚合表。

沿用已有私有凭据路径，依次支持显式 `--credentials`、`PAPER3_GITHUB_CREDENTIALS`、`~/.config/paper3/github.json`、旧R2 `~/paper3-followup/*/private/github.json` 及R3/R4对应安装目录。仅读取路径指向的配置，不改旧配置内容；新监视器会将分支与状态路径固定为第四轮。

可选外部评价使用明确缓存manifest和本轮basis输出，在核心完成后单独启动一个三卡作业。Weibo缓存是CC0的发布顺序LR子任务离线稳健性，真实物理采集前20条和run时序未知；不把它当作新的前瞻确认。下载校验和缓存准备不进入核心训练启动流程。

本地合成数据、隔离入口、模拟Slurm及Bash4.2测试验证软件契约；不能代替学校三卡GPU实跑，也不是准确率提升证据。最终测试计数和环境版本见 `VALIDATION.json`。
