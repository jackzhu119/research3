# 第一、五、六轮：三独立单卡作业续跑

这是运行层迁移，不建立新的科研轮次。原数据、种子、算法、已完成回执与checkpoint原地保留。

|轮次|旧作业/分区|新分区与显卡|新作业资源|
|---|---|---|---|
|第1轮|770043 / hpc_gpu1|hpc_gpu / A100 PCIe 40GB|3个单卡作业，各12CPU；43G、43G、42G，共128G|
|第5轮|772012 / hpc_gpu|hpc_gpu1 / A100 SXM4 80GB|3个单卡作业，各12CPU、64G，共192G|
|第6轮|772013 / hpc_gpu|hpc_gpu1 / A100 SXM4 80GB|3个单卡作业，各12CPU、64G，共192G|

在线入口：[启动命令](online/LAUNCH_ZH.md)。每轮分别执行，重复执行会复用控制器、已完成结果和断点。

## 交接顺序

下载并校验内嵌完整运行包 → 提交三个held单卡作业 → 实际队列/资源核对三份回执 → 旧任务保存checkpoint或取消旧pending → 等旧锁/作业退出 → release新作业。

提交被学校QoS拒绝或新作业没有全部确认时，保留旧任务并后台重试；不会为了腾提交名额先取消旧任务。学校的资源配额和排队时长仍由调度器决定。

第一轮先停用精确匹配的旧自动单卡重提交监控，避免它在checkpoint退出后把旧作业再次拉起。旧作业先通过USR1请求原kernel保存状态；确认科学子进程退出、账本关闭、checkpoint SHA及保存游标一致后，可结束旧的备份整理过程，避免无意义全树SHA拖慢交接。迁移边界保存在 `runs/training_campaign_parallel_v1/migration-progress-snapshot.json`。不会删除原 `runs/meta/` 或 `runs/training_campaign_v1/`。

第五、六轮原 `runtime/STOP` 保留为旧控制器停用墓碑，新任务只使用 `runtime/three_jobs`。原计划和数值worker保持原样。

## 第一轮的任务依赖

剩余独立元训练配置/种子可三卡并行；计时比较与开发评价固定slot0，避免混入不同物理设备的计时。主线结束后，原登记的Yang、Dreyer两个扩展arm分别在slot1/2继续，slot0自行退出释放资源。不能把主线完成误称成两个扩展也全部完成。

跨A100 80GB→40GB恢复保留原checkpoint/identity，记录实际新硬件，只对已授权的型号/显存/UUID和重复驱动枚举数量作行政兼容；不改数据、算法、种子或科学更新数，不宣称跨卡位级一致。旧80GB开发计时不伪装成40GB测量。相同40GB硬件族的换UUID比较有独立审计。

## 自动GitHub回传

- 第1轮：`paper3-r1-progress:monitoring/paper3-round1/status.json`
- 第5轮：`paper3-r5-progress:monitoring/paper3-round5/three-jobs-status.json`
- 第6轮：`paper3-r6-progress:monitoring/paper3-round6/three-jobs-status.json`

原第五、六轮旧 `status.json` 留作旧监控流，不会覆盖新三作业状态。

状态首次立即推送，此后每180秒；现成报告、CSV和任务回执摘要由独立后台线程推到 `paper3-results/results/paper3/round-NN/automatic-summary/`。摘要只使用已有文件，不重新统计，不等待数小时的聚合。`AUTO_SUMMARY_LATEST.json`不等同于完整原始预测/checkpoint归档的`LATEST.json`。

## 验证与运行边界

见 `VALIDATION.json`、`install-proof.json`、`download-proof.json` 和各轮测试。已完成160项本地调度/交接/恢复/监控测试、原kernel CPU中断恢复及完整包隔离安装准备验证。学校启动命令不运行pytest、pilot或数据全树核验，不重新安装环境。每个Slurm作业按学校时限运行，提前保存断点后自动续提交；没有新增累计12小时限制。

源码已发布和实际下载核验。学校真实GPU和调度动作需要用户执行启动命令后发生，本次准备没有提交或取消学校作业。
