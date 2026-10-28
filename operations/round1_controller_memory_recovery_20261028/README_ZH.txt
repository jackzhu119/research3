R1 调度器内存故障恢复（2026-10-28）

第2版修正：旧控制器报出 MemoryError 后，进程可能因残留线程继续存在，不能将主线程报错视为整个进程已经退出。先请求旧CPU管理进程正常退出；若未退出，重新核对用户、完整启动命令、启动nonce和进程启动时间，只结束这个已确认的CPU管理进程，再接续原任务派发。原科学worker及Slurm作业不发退出信号。内核尚未释放进程时明确记录接管等待状态，不误报调度恢复。

已观察到的故障
三个原作业仍是 Slurm RUNNING，但 worker-status 均为 WAITING_READY_ACTION，child_pid=null，claims.json={}。
CPU 调度器在 kernel_bridge._read 读取已完成回执时发生 MemoryError，最后一次进度写入为2026-10-28T05:10:10Z。
因此既不是三个GPU仍在训练，也不能据单个子任务 COMPLETE 判断整轮完成。

恢复方式
1. 标准库程序，使用学校已安装的 Python；不执行 pip、不新增GPU检查作业。
2. 只在CPU管理进程中流式读取回执，只保留原调度所需的状态、配置和扩展臂字段；完整扫描JSON语法，损坏回执不会被当作完成。
3. 每轮调度运行一个独立短进程，结束即释放该轮缓存和线程状态；异常和超时记录后重试，不让常驻管理进程累积大型回执对象。
4. 原始科研代码、已完成回执、检查点、数据划分和训练更新数保留。科学worker继续执行原验证规则。
5. 仅恢复已有三个作业的任务派发；本修复不提交、取消或迁移Slurm作业，不发送科学worker信号，不运行原硬件探测watcher。
6. 主线确认后仍有 strongest-control、freeze、internal；之后两个原有扩展臂。队列真正 COMPLETE 后worker正常退出。245是当时已发现的动作数，不是整轮固定总任务数。

学校执行
下载并校验固定版本 RECOVER-PAPER3-R1-CONTROLLER.py 后：
"$HOME/.venvs/paper3-hpc-cu121-offline/bin/python" -I -S -B -u "$HOME/RECOVER-PAPER3-R1-CONTROLLER.py" --apply

查看当前管理状态：同一命令将 --apply 换为 --check。
管理恢复就绪不等于训练完成；以新队列阶段、worker实际claim和原生完成回执为准。
本地CPU/模拟Slurm测试不能替代学校现场执行结果。
