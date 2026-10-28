第一轮单作业续跑：1 GPU / 4 CPU / 64G

用途：继续原有第一轮主线和两条既有扩展臂，保留原科学动作、种子、输出目录和已有结果。
分区沿用 hpc_gpu；已观察的主线硬件为 A100 PCIe 40GB，原生硬件/计时兼容检查保留。
4核为Slurm申请，保留原数值线程策略，避免改变已有校准计时身份。

启动：现有Python执行 START-PAPER3-R1-1GPU.py --apply。
返回 R1_SINGLE_GPU_BACKGROUND_STARTED / terminal_can_close:true 后可断开终端。
该回执只证明后台启动；GPU开始工作以Slurm实际分配和绑定任务日志为准。

工作目录：$HOME/paper3-r1-single-gpu/work
  controller.log / controller-status.json：管理器和真实派发结果
  allocation-state.json：单作业提交历史、排队、运行、退出
  observer.log / observer-status.json：自动回传状态
  STARTED.json：后台启动回执
  config.json：非敏感运行配置，不保存token

学校不安装依赖，不运行全量测试，不提交GPU测试作业。
仅原CPU管理进程和旧CPU进度监控被精确交接；若有旧计算子进程仍活跃，先让其完成当前动作。
新调度仅一项GPU分配；原先仅供槽位1/2的扩展路由改到唯一槽位0，仍保留原科学阶段顺序。
首遍原生结果验证保留；strongest-control/freeze内同参数重复成功验证在当前动作内复用，
输入依赖发生变化会拒绝发布结果，缓存不跨动作持久化。

取消行为：网页取消本包新作业后，管理器观察到CANCELLED会写STOP并禁止再补交。
主动停止文件：原项目 runs/training_campaign_parallel_v1/STOP。
--apply是显式重新授权续跑，会清除先前STOP；不要用--apply作为只读状态命令。
只读检查：START-PAPER3-R1-1GPU.py --check。

作业超时/故障按原断点机制继续；Slurm时限与强杀风险仍存在，没有任意累计次数/12小时停止限制。
管理循环与上传分别运行于独立进程；网络/上传/报告错误不停止训练。
管理元数据周期900秒deadline只终止自己创建的管理子进程，不向科学worker发送信号。
进度每180秒回传到 jackzhu119/research3 的 paper3-r1-progress 分支。
原结果自动摘要在完成数变化时由独立低优先级进程回传到 paper3-results 分支；
同一内容不重复上传，网络/结果汇总失败单独重试，不阻塞进度采样和训练。

验证边界：开发侧实际隔离入口、原worker接口、原生选择器结果一致性和模拟Slurm验证，
没有从开发环境连接学校运行真实CUDA核。学校首个正式任务的日志是最终现场运行证据。
