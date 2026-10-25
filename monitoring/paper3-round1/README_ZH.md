# 第一轮实时进度

[最新轻量状态](status.json) · [按轮分类结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-01)

同一原始科学计划，三个独立单GPU作业：hpc_gpu、每作业12CPU，内存43G/43G/42G，共36CPU/128G。

首次启动即回传，之后约180秒一次；不重新统计逐试次结果。主线mainline与两条extension_arms分栏显示；主线完成不等于全部原任务完成。两条既有扩展为BASE4_PLUS_YANG与BASE4_PLUS_DREYER，扩展包装动作可包含很多原生fit。coordinator记录与live_queue分别显示，运行/排队以live_queue本次观察为准。网络失败只影响回传，不控制训练。
