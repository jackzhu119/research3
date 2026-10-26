# 第九轮 CUDA 确定性累计求和修复（2026-10-26）

第九轮六个 `feature_reference` 任务在收集完源训练前缀特征后，计算加权中位数长度尺度时反复失败。学校 PyTorch 2.5.1+cu121 在严格确定性模式下拒绝 CUDA 浮点 `torch.cumsum`；原任务在 395/396 会话之后进入同一个异常。该依赖没有完成也阻塞 GP 参数选择和相应评估。

本修复只把排序后权重的累计求和移至 CPU float64，并将累计向量返回权重原设备。长度尺度公式、加权中位数边界、核函数、超参数、支持集、随机种子、任务图均保留，严格确定性模式继续开启。模型拟合和预测继续使用原 GPU。

修复在第九轮子进程导入 `paper3r9.gp` 时编译内存副本，保留此前 `balanced_loss` 精度修复。每个任务的 `runtime.json`、`result.json`、`error.json` 中会记录 worker 和 GP 原始/执行代码 SHA256。科学源码、原 plan、配方散列、checkpoint、已完成结果和原共享修复 hook 不改写。

`FIX-PAPER3-R9-CUMSUM.py` 只更新已安装第九轮的 `runtime/backend_entry.py`，覆盖正在运行作业的下一次子任务重试，以及尚在排队和后续续跑的第九轮 worker。不会停止或重提任何 Slurm 作业，也不改分区和资源。第九轮位于 `hpc_gpu`，三个单卡作业，每项 16 CPU、64G 内存。当前运行的失败子进程结束后，原调度器会自动采用新启动入口。

安装仅使用 Python 标准库，不加载 Torch，不下载数据，不扫描训练数据。失败任务已有的 `reference-sessions/*.pt` 将由原代码按其身份验证后复用，无需重新提取全部会话特征。第一、五、六、八轮不安装本修复。

下载并执行 `RUN-ONE-LINE.sh` 中的一整行。安装成功标志为 `R9_CUMSUM_REPAIR_APPLIED`；这是安装回执，不等于 GPU 已恢复。新子任务日志应出现 `R9_DETERMINISTIC_CUMSUM_READY` 和 `R9_RUNTIME_CORRECTION`，随后观察 `feature_reference` 完成数由 0 增长，以及 `select` 开始。一般需等正在运行的子任务退出及原调度器的重试间隔。

不要用旧的固定 SHA 在线安装器恢复已修改启动入口；如原控制器确实退出，请使用已安装的 `future-path-v2/resume_existing.py` 保存计划恢复路径。本次正常安装不需要恢复控制器。

第九轮已有的独立论文证据快传和全文件慢传进程继续运行，上传目的地保持按轮次分区；本修复不新建上传器、也不声称当前部分结果就是完整归档。

验证见 `VALIDATION.json`：CPU 软件回归与 CUDA 路由约束模拟分别标注。当前执行环境无 GPU，学校实际运行情况仍需以修复后的新任务回执确认。
