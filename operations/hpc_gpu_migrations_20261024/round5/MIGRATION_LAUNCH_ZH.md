# 第5轮资源迁移与原结果续跑

目标为学校hpc_gpu组的A100-PCIe-40GB；每轮一个节点、一个普通Slurm作业、三GPU独立工作进程，总36CPU、128G共享内存，每进程12CPU亲和组。不会生成数组或为每个逻辑任务单独提交作业。计划仍是原来的105项，原数值文件、recipe、任务顺序、结果目录和checkpoint身份保持不变。

新安装目录~/paper3-round5-hpcgpu/<hash>/，结果继续存原/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5/。CPU/GPU/内存资源以runtime/launch.json为准，既有plan.json原字节和里面的历史操作字段保留。

必须先完整下载并验证发布包，才执行MIGRATE.py --submit。入口准备时不会取消任何作业。用户--submit后启动独立后台交接：原controller/allocation的STOP机制请求保存；仅查询同用户且精确匹配本轮marker的旧job。PENDING仅用scancel --state=PENDING取消；RUNNING用USR1请求checkpoint，不强杀GPU。等待队列与accounting都确认结束，并取得原controller锁后，归档历史操作状态、保留旧jobID/单调序号，复用原完成项和checkpoint提交新的hpc_gpu单作业。观察/配额故障持续后台重试，不重新跑全量pytest/pilot/全数据树扫描。

迁移日志为原结果目录runtime/hpc_gpu_migration/migration.log及progress.json。后续节点日志仍为logs/train-作业号.out/.err，逐卡任务和步数/会话进度在allocation_progress.json。单任务错误隔离；没有累计12小时限制。

GitHub沿用第5轮专属分支paper3-r5-progress，汇总路径monitoring/paper3-round5/status.json，状态内experiment标明paper3-round5；三个轮次不会共写一个status。现有本地私有凭据自动发现，公开下载包不包含凭据。监控/网络失败不会停止训练；完整checkpoint、逐会话预测及CSV先存学校本地。

本轮复用学校已有源模型和Stieger缓存，不新增数据下载；可选Weibo外测不是本次迁移自动启动内容。

本地CPU及模拟Slurm验证不能替代学校PCIe40GB三卡现场实跑；GPU型号变更只记录runtime变化，不宣称跨硬件逐位等价。
