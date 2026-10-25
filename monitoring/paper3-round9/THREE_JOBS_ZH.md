# 第9轮：三个独立单GPU作业实时进度

[三作业轻量状态](three-jobs-status.json) · [按轮分类结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-09)

当前方案为 hpc_gpu、三个独立单GPU作业，每作业16CPU/64G；总3GPU/48CPU/192G。目标资源组显卡为A100-PCIe-40GB，资源申请与实际队列状态分列。

首次启动立即回传，此后约180秒；只读控制器完成计数和实际worker进度，不重算统计、不读取逐试次预测。网络失败只影响同步，不控制训练。

第九轮使用独立目录、作业标记与状态分支，已有轮次继续各自运行。
