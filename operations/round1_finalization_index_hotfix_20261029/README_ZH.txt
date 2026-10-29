第1轮最强对照校验缓存索引误报热修复（2026-10-29）

学校现场证据：单卡作业779398仍RUNNING；strongest-control多次因
FINALIZATION_INPUT_CHANGED: runs/training_campaign_parallel_v1/digest-cache-v1/index.json退出后重试。
原因：上一版开放读取依赖捕获把会自行读写的哈希缓存索引当成不可变科研输入；
真实DigestCache.flush在验证过程中先读取index再原子替换，首遍native验证结束即被误拦截。

本次仅精确排除这一个操作缓存索引，要求lexical与canonical路径都准确命中。
不忽略整个runs或runtime；真正receipt/model/data/plan/code变化仍拒绝发布。
旧科研函数的首次验证、原最强对照选择、最终不可变写入器和数值结果均保留。

执行：学校现有Python以 -I -S -B -u 执行 FIX-PAPER3-R1-FINALIZATION.py --apply。
不取消/重提Slurm作业，不安装依赖，不启动测试训练，不改变原单卡资源。
仅在原可变operational action_startup.py末尾装载内容寻址的新helper；
旧82bf不可变包完全不变，原START入口--check及显式续跑仍能校验通过。

后台激活：若当前正是被严格绑定的strongest-control/freeze旧校验进程，且其KERNEL_READY、
当前action/claim/job/slot/父子PID/UID/进程开始时间/完整参数均相符，
只对这一段CPU对照验证请求一次SIGINT。原子进程自身写PAUSED并释放原claim，
同一Slurm作业的原parent自然重试并载入新helper。
这两个动作不做模型训练；internal、development、meta、任何不匹配进程均不发信号。
不向parent、进程组或Slurm allocation发信号，也不使用TERM/KILL刷新科研进程。
请求先持久记账并持per-claim锁，远端步骤超时不能重复发送。
若正在导入、已经结束、已载入修复或已进入真实训练，只观察或等原流程继续。

工作目录：$HOME/paper3-r1-finalization-hotfix/work
 applied.json：仅证明安装操作已完成
 activation.log、status.json：实际激活与当前原生结果回执
 status=FIXED_VERIFICATION_SCOPE_ACTIVE：当前claim绑定日志确认新v2验证作用域已生效
 status=STRONGEST_CONTROL_COMPLETED_NATIVE_RECEIPT_OBSERVED：原最强对照回执已完成
 REFRESH_REQUESTED/WAITING等状态不代表整轮完成。
后台启动 terminal_can_close:true 后可关闭终端；原自动进度/结果回传保持独立运行。
只读检查：同一入口 --check。

本地验证包括真实旧worker_runner导入顺序、保留旧包字节和旧入口校验、
真实DigestCache原子更新回归、科研输入变化/符号链接仍拒绝、原选择器结果一致性、
精确进程身份/只读动作范围/持久一次性刷新测试。开发环境没有连接学校运行真实CUDA或发送学校信号。

最新预部署回传已标记STOPPED_BY_USER，安装器保留该STOP，不会恢复作业。
用户明确需要续跑时，再执行已有组合入口 --r1-only --apply；新任务会经已安装operational hook载入本修复。
