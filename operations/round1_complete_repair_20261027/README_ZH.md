# 第一轮现有三个作业的恢复修复（2026-10-27）

本包针对第一轮 B4/B5 反复完整验证源依赖、子进程 `-11` 退出及当前阶段只向槽位0派发任务的问题。使用学校已经安装的 Python/Torch 和原始科研实现，保留已完成结果、断点、种子、数据划分、参数更新数量、模型选择规则和计时原始证据。

## 原因与证据

1. 每次 B4/B5 支持集拟合与拟合工件验证都会调用源模型加载器，源模型加载器又执行完整 W1/W2、505个源对象、事件/数组数值及10项核心拟合检查。已有哈希缓存仅避免部分重复哈希；大量 JSON、数组遍历和路径检查仍重复。用户的失败动作累计827900次哈希命中，不能把这些命中或 bytes_reused 当成 GPU 更新。
2. 原控制器明确把 calibration profile、development、selection 固定到槽位0。槽位1、2完成元训练后即等待，Slurm RUNNING 不等于三卡数值工作。原父进程每次从 queue.json 领取工作，故可以只换 CPU 控制器，无须取消或重启现有 allocation。
3. B4于2026-10-27 08:42:17以 `-11` 退出，日志末尾是截断的周期 faulthandler 堆栈输出。现有日志不能定位具体原生库，不能声称已经证明 faulthandler 是根因。本包去掉可选的周期堆栈采样、保留父进程心跳及可用的致命故障报告，同时消除反复验证带来的负载。详细边界见 `proof/CRASH_AUDIT.json`。

## 实现

- `verified_context_cache.py`：一个 action 首次调用原始完整验证，之后复用已验证的冻结源上下文与小型 B4/B5 源模型；每次 fit 使用独立只读模型副本。不开跨 action 缓存，不缓存支持集拟合结果、query结果或可变学习模型，不拷贝大 EEG mmap。命中只检查小型身份锚点；最终完成回执发布前检查捕获的全部依赖。源身份变化不能产生成功完成回执。全局状态仅核对相关 W1/W2，其他阶段更新不会造成误中断。
- `newdiagnostics.py`：新动作的周期堆栈计时器关闭，保留原检查点信号语义与父进程心跳，尽可能保留真正致命错误报告。
- `parallel_runtime.py`：原依赖 DAG 生成合法就绪动作后，只改变独立 calibration/development 的运维路由。现有父进程直接领取工作；claim、任务锁及原完成回执防止重复。选择、冻结、内部验证等串行动作保留原屏障。修复旧阶段遗留的虚假 RUNNING 状态。
- `newnode_tool.py`：在现有 allocation 内观察原生完整运行环境，不训练模型。恢复旧动作时，严格绑定同用户、作业、父子PID启动身份、action文件及claim token，只向旧科学子进程发送原有 SIGINT 断点退出请求，绝不向科学进程发送 SIGKILL，不取消 allocation。已激活新修复的子进程保留，重复命令不会反复中断它。已提交的拟合与断点保留，尚未提交的单次拟合可能需要继续或重做。
- `supervisor.py`：后台管理恢复。只接管身份可确认的原CPU控制器；现有科学worker与其他轮次不取消。先观察原三槽队列，首次观察零提交；显式 `--submit` 只允许原调度器为已结束槽位补位，不能增至六个作业。
- `install_hooks.py`：仅对已有运维 `action_startup.py` 追加带备份的独立标记块，保留此前时钟、驱动、哈希缓存修复。原 `p3work/` 科学源码、检查点及完成结果不改。

## 三卡并行的实际含义

原科学协议要求校准计时来自实际同类硬件/完整运行环境，原生比较器继续执行。后台工具在已分配GPU上做一次运行环境观察，不拟合、不申请新 allocation；同类环境的现有槽位可以同时执行独立动作。不同或尚未观测的环境不会被伪造成等价，不跨硬件混合耗时，也不修改选择规则。

如果探测暂时失败，保留原槽位0路由继续工作，不能因为新增探测把第一轮整体阻断。状态文件明确区分 `eligible_primary_slots`、`routing_primary_slots` 和原槽位0回退；绝不能仅据控制器 READY 声称三卡已在训练。

## 学校运行

完整固定下载链接及一行命令见 `delivery/SCHOOL_COMMANDS.md`。默认仅离线解包，不改运行文件、不启动子进程、不导入Torch、不访问网络。`--check` 只读。必须显式 `--repair --submit` 才安装运维hook、启动后台控制器、请求旧动作按检查点退出以及续跑原作业槽。

后台位置：`~/paper3-r1-full-repair/work/`，重点文件：

- `STARTED.json`：启动确认；`STARTUP_PENDING` 只表示后台尚未确认就绪。
- `controller-ready.json`：PID、启动nonce及成功队列观察确认。
- `controller-status.json`：当前阶段、作业及实际路由槽位。
- `controller.log`：后台管理日志。
- `hardware-observations.json`：按现有job_id绑定的实际运行环境观察。
- `refresh-results/slot-*.json`：旧子进程断点刷新结果或新进程保留结果。

原科学日志仍在 `runs/training_campaign_parallel_v1/logs/`；原动作尝试仍在 `actions/<key_sha>/attempts/<claim>/`。`repair-active.json` 绑定当前动作、PID、claim及helper SHA，日志中应出现 `R1_SOURCE_CONTEXT_VERIFIED_ONCE`、`R1_SOURCE_BANK_VERIFIED_ONCE`、`R1_VERIFIED_SOURCE_SNAPSHOT_PROGRESS` 等记录，原生完成仍以 `receipt_complete: true` 为准。

启动后可关闭终端。管理进程就绪、实际GPU分配、数值进展和科学完成是四种不同状态。学校现场恢复须以新任务完成回执及后续GitHub回传确认，不能用本地测试代替。

## 验证和复现

测试覆盖原生B4/B5合成支持集拟合与未来预测逐元素一致、RNG不变、500次加载仅一次完整验证、深层依赖变化阻止完成、原DAG屏障和三个原父进程领取不同动作、无重复提交、正确的PID/claim刷新、默认离线安装、幂等与回滚。所有预发布测试在CPU执行，没有向学校实际提交或取消作业。计数与SHA见 `proof/FINAL_VERIFICATION.json`、独立复核见 `proof/INDEPENDENT_REVIEW.json`。

`vendor/` 只有固定来源的运维回归测试夹具，不是运行时替代科研源码。需要复跑CPU测试时设置 `PAPER3_R1_NATIVE_ROOT` 为原项目路径并使用已有科学依赖；来源见 `proof/VENDOR_FIXTURES.json`。本包没有建立跨GPU逐位相等证明，也没有把关闭周期诊断描述成已证明消除了所有可能的原生崩溃。
