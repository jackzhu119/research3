# R11 运行状态 I/O 修复与四槽续跑（2026-10-27）

本包修复已经运行的原 R11 campaign 的运维写入路径，不重新设计实验。保持原科学 recipe、签名计划、数据身份、算法、种子、训练上限和模型结果字节；原科学包源码不改。

## 已证实的问题与证据边界

用户截图中的 775286、775287 在 `os.replace(temporary, path)` 出现 `FileNotFoundError`，目标分别为 `runtime/status.json` 和 `runtime/numerical-observations/*.json`。异常从监控路径冒泡，导致整个 worker 退出。此前的状态写入异常还使两项训练被登记为 `BLOCKED_CONFIGURATION`；其科学子进程后来显示 COMPLETE，但未补封完成回执，继续阻断下游。

原 `common.atomic_bytes` 已使用唯一 `mkstemp`。本地审计没有找到删除这些临时文件的全局清理器，所以不能将文件名碰撞、某个上传器删除临时文件或学校文件系统故障写成已证实的底层根因。本包针对已证实的临时路径丢失及监控异常升级路径提供恢复。

2026-10-27 08:44 北京时间的固定回传：587/10270 完成；775288、775289 仍有数值推进；2 个原作业已失败；4 项阻断中 2 项是上述训练，另 2 项是其依赖。调度快照仍停在前一晚，必须现场重新观察队列。论文 A 已回传 575 项，完整 B 已验证 407 项。本包保留两个上传进程，继续沿用其缓存和回执。

## 修复行为

1. `atomic_io.py`：同文件系统内用每次操作独占的 0700 暂存目录、完整写入/flush/fsync 后替换；最多 6 次瞬时 I/O 重试。对“服务器已提交但客户端收到错误”的情况，只有目标完整字节 SHA 与拟写入内容完全一致才接受。只清理自己创建的暂存文件，不扫描删除其他进程文件。永久错误或耗尽重试仍抛出。
2. `resume_io.py`：在新 worker、独立科学子进程及新控制器中注入运维补丁。仅监控状态、数值观测和 slot 心跳允许延期写入，并记录 `R11_TELEMETRY_WRITE_DEFERRED`。`tasks.json`、调度意图、模型、result、completion 等持久科学记录仍必须成功写入，绝不把失败写入当成完成。
3. 新控制器先仅观察原 marker 下的 Slurm 队列/记账，不提交；确认已结束的槽位后才按显式 `--submit` 补位。最多 4 个独立作业，各 1 GPU、16 CPU、64G。已运行或排队的 allocation 原样接管，不取消、不向其发送信号。
4. 只对已识别的临时文件到监控目标的错误恢复任务：先取得任务租约及独立科学锁，避免碰仍工作的子进程。完整 result 身份和所有工件 SHA 核验通过，才补封 completion 并登记完成；否则仅该类错误任务恢复为 PAUSED，从原有结果/断点继续。其他配置错误不重置，原失败历史保留。
5. 新 worker 命令只在新控制器内存中覆盖，原 `launch.json`、任务计划及科学代码不改。仍在运行的旧 allocation 保留原进程；如果其后自然退出，控制器补上的 allocation 使用新入口，不能声称现存旧进程已经被在线注入补丁。
6. `supervisor.py` 只替换同 UID、精确 Python 入口、controller 子命令、原计划 SHA 和 PID 启动身份均匹配的旧管理控制器。先 SIGTERM；必要时只对该管理 PID 升级 SIGKILL。绝不操作进程组、科学 worker、Slurm 作业或上传进程。不明归属则不发信号并明确返回未恢复状态。

## 使用

完整固定 URL 与入口 SHA 见 `delivery/SCHOOL_COMMANDS.md`。入口默认只离线安装，不导入 Torch、不访问网络、不申请 GPU。`--check` 只读原运行记录；`--repair --submit` 才启动后台修复和失败槽位续跑。使用已有学校 Python：

```text
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
```

安装目录：`~/paper3-r11-io-repair/<payload_sha_prefix>/`。
管理日志：`~/paper3-r11-io-repair/work/97e4cadb17278532/controller.log`。
结果：同目录 `STARTED.json`、`controller-ready.json`、`controller-status.json`。
四个 allocation 的日志仍保留在原 `runs/r11_campaign/runtime/logs/`，新的 task/claim/result 身份仍由原科学计划定义。

`IO_REPAIR_CONTROLLER_READY` 表示新控制器已确认接管并完成首次队列观察；`STARTUP_PENDING` 表示后台启动尚未确认。它们都不代表四张 GPU 已分配，更不代表论文或完整归档全部上传完毕。启动后可以关闭终端。

## 发布前验证和限制

测试在 CPU、合成/确定性输入及原生产运维模块上执行，不占学校 GPU、不提交真实 Slurm。覆盖原始 ENOENT 复现与私有暂存恢复、并发写入、永久存储失败、监控写失败继续科学任务、科学完成写失败不能伪造 COMPLETE、两个存活槽位保留和两个失败槽位补位、完成结果原字节复用、活跃科学锁不被抢占、未知/PID 复用进程不被信号、默认离线安装及每个安装文件 SHA。

完整验证计数与来源 SHA 见 `proof/FINAL_VERIFICATION.json`；独立审查见 `proof/INDEPENDENT_REVIEW.json`。本地通过不能代替学校现场恢复回执，也不能保证学校存储或调度永远不再出错。

## 第一轮检查结论（本包不改变 R1）

08:40 的第一轮主动作 101/207，较前夜 74 有进展；但 B4 在 08:42 再次 `returncode=-11`，随后 B5 继续尝试，不能判为正常持续计算。槽位 1、2 日志分别停在 10 月 26 日 14:38、15:35 的成功 meta 任务。原 R1 调度源码把当前 calibration/development 阶段固定到 slot0，其余槽位等待，故 Slurm RUNNING 不等于三卡都在训练。重复源依赖校验消耗明显；缓存 bytes_reused 是累计避免的哈希读取量，不能当成真实吞吐或训练进度。`-11` 的具体原生崩溃根因还需对应 kernel.log，不能用定时堆栈输出代替诊断。
