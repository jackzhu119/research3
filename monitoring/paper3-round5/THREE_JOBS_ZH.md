# 第5轮：三个独立单GPU作业实时进度

[新三作业轻量状态](three-jobs-status.json) · [旧单作业监控历史](status.json) · [按轮分类结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-05)

当前方案为 hpc_gpu1、三个独立单GPU作业，每作业12CPU/64G；总3GPU/36CPU/192G。目标资源组显卡为A100-SXM4-80GB，资源申请与实际队列状态分列。

首次启动立即回传，此后约180秒；只读控制器完成计数和实际worker进度，不重算统计、不读取逐试次预测。网络失败只影响同步，不控制训练。

旧监控仍可继续写status.json；它是历史单作业流，不应当作此处新三作业的状态。新状态独立存放，避免两个监控进程覆盖同一文件。
