# R11 学校平台实现与 R10 状态边界

R11 独立使用 `paper3r11`、`round-11` 归档和 `runs/r11_*`，不写入 R10 的代码、检查点或作业。资源为4个单卡作业，每个1GPU、16CPU、64G，共4GPU/64CPU/256G，队列 `hpc_gpu`。只有显式 `--submit` 才授权控制器调用 Slurm；安装与准备不提交作业。

`delivery_template.py --start` 将安装与有限 campaign 计划准备串起来：开发、验证选择与探索性评估是一张有界依赖图，不要求手动切换阶段或再次安装；同时给 `--submit` 才启动后台控制器、报告与上传。计划身份绑定本轮配置、源/数据描述符和代码，不再要求旧轮 GPU UUID、运行时 profile 或 R10 的分阶段冻结收据来允许训练。开发任务仍只能使用 meta_train/meta_validation；未来目标 query 标签不得选型或拟合，确认任务必须明确使用独立参与者。

提交前只核对小型计划/源收据与所需模型，每份重复 baseline 描述符只验证一次。明确标注 `optional_baseline=True` 的历史对照缺文件时，仅该任务被标为不可运行；核心任务继续，报告明确缺失对照，不使用随机模型替代。正常补齐相同 SHA 文件后，下次准备可恢复该对照。科学 worker 首次实际打开数据时仍核对身份。监控/上传读元数据时不重复遍历代码、原始 EEG 或上游完整验收；不复制无关 node runtime cache，不自动安装依赖。

控制器启动后先写本轮 PID/recipe 的轻量 bootstrap ready 回执，再观察 Slurm。启动命令最多等3秒获取回执；子进程直接失败会明确显示 `CONTROLLER_FAILED_TO_START`，尚待回执则显示 ACK_PENDING。ready 仅表示控制器启动，不冒充模型正在训练。状态包含作业 ID、队列原因、节点、调度器错误、控制器错误、实际数值更新与独立心跳；大型计划的常规状态只发布至多64条活动/阻断/最近任务和完整 family 汇总，完整任务覆盖仍由科学回执和归档 manifest 保留；因此 NEW/0 次更新可以区分未提交、排队和控制器失败。

四个槽位领取独立任务。每个 allocation 保留一个科学子进程，NumPy/Torch只导入一次；逐任务 stdout/stderr 和原子响应文件分离，普通 Python 异常隔离该任务，实际子进程崩溃或损坏 CUDA context 才重启子进程，不重启整张 campaign。锁与完成 SHA 防止重复执行；异常最多3次后隔离实际失败任务或分配，其他任务继续。USR1 与退出75保持原生检查点续跑；Slurm TIMEOUT/PREEMPTED 继续替换分配，不计为算法持续失败。错误配置的提交停止重试，真实网络失败只由独立上传进程退避，不终止训练或触发重训。

通道A完整上传参与者/会话 CSV、候选配置、统计、报告和证据，通道B独立上传完整预测/模型/轨迹归档并读回 SHA。CPU reporter 自动根据新的完成回执生成报告，A/B不依赖终端。报告完成需要精确 recipe、全部期望 task-key SHA 和实际报告 SHA，任务计数相同的旧报告不能冒充完整。`paper3-r11-progress` 在第一次本地状态读取前显式确保分支存在；状态读取临时失败也有本地 branch-ready 与错误回执。

现有 R10 远端实际证据：05:43:23Z 的状态为0/15、全部 NEW、attempt0、job_id为空；已上传计划资源正确。远端A只含计划，没有 scheduler/controller 日志或任务结果，不能据此判断是学校排队还是控制器尚未启动。R11提供上述调度元数据解决可观测性不足；本次没有改动旧作业。

CPU回放覆盖：无效继承 CuBLAS 环境在导入科学模块前修正；Python `-I -S` 使用现有 venv site-packages；导入期 SIGUSR1 退出75；Float/Double logits 与整数标签的平衡 BCE；严格确定性参考权重 cumsum 在 CPU；分配隔离/检查点续跑；监控分支预创建和网络隔离；默认安装零执行与显式 `--start --submit` 边界。实际学校 CUDA、GPU与Slurm尚未执行。

大计划与缓存：签名 JSON 计划限64MiB，不是 EEG 数据上限。prepare/resume 只做一次已有完成目录的元数据恢复，完成后将 task identity、result SHA 和 completion SHA 记入持久 tasks state。领取与常规计数只读该已验证缓存，不再对上万个目录重复探测；活动心跳只读取当前任务的 progress。控制器与 allocation 复用已读取的计划，仅在文件 stat 身份变化后才重算计划 SHA。`requires_terminal` 让候选分支失败并完成隔离后仍可进入预声明的验证选择，严格 `requires` 的真实科学依赖继续保留。A/B与reporter也按完成缓存增量捕获，每份新科学输出只首次读取/封存一次。
