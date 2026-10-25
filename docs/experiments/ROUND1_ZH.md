# 第1轮：原论文主线：Lie空间校准P/PCT与完整基线

[八轮总索引](README_ZH.md) · [最新回传汇总](../LIVE_STATUS_ZH.md) · [后续安排](../ROADMAP_ZH.md)

## 已发布的拆分续训修复

2026-10-25完整日志确认旧适配漏掉实际NVIDIA A100-SXM4-40GB。已发布[修复及恢复入口](../../operations/round1_repair_20261025/README_ZH.md)，覆盖当前筛选和后续比较、同动作重复核验与阶段日志，沿用已完成结果与350断点。用户已授权必要的断点退出和重新排队。修复包已发布不等于学校已应用，以新续训更新计数为准。第六轮迁移已取消，保持原配置。

## 本轮研究内容

小预算空间适应能否改善新会话识别；与冻结源、调头、微调、协方差/切空间、加性basis及低秩方法比较。

计划规模：**分阶段计划：15源神经fit、45元筛选、10额外元确认、开发/确认与扩展**。任务成本不同，完成任务比例不能直接换算剩余GPU时间。

## 本次整理核对

核对日期：2026-10-25（北京时间）；每条原始回传有独立更新时间，后续以[自动汇总页](../LIVE_STATUS_ZH.md)和[本轮原始回传](https://github.com/jackzhu119/research3/blob/paper3-r1-progress/monitoring/paper3-round1/status.json)为准。

1作业运行、2作业排队；旧350更新记录尚未见RESUMED，需查看当前attempt日志。

资源配置：**`hpc_gpu`，3个独立单卡作业；各12CPU，内存43G/43G/42G，总36CPU/128G**。`hpc_gpu`对应A100 PCIe 40GB资源组，`hpc_gpu1`对应A100 SXM4 80GB资源组；节点名`gpu1`与分区名`hpc_gpu1`不同。以上GPU型号为资源组配置，不替代作业内的实测设备记录。

## 源码、设计与结果

- [本轮代码与数学设计](https://github.com/jackzhu119/research3/tree/research/paper3-protocol-v2-w0/p3work)
- [实时回传原始文件](https://github.com/jackzhu119/research3/blob/paper3-r1-progress/monitoring/paper3-round1/status.json)
- [按轮分类的结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-01)
- 学校输出：`/public/home/zhuziyuan/projects/paper3/runs/training_campaign_parallel_v1`
- [本轮整理前的设计与部署记录](../records/20261025-before-reorganization/ROUND1_ZH.md)

历史说明中的“未提交”、旧作业号和旧资源只对应当时交付时刻；当前入口不沿用这些旧状态。源码分支与固定提交下载继续保留。

## 后续步骤

- 核对当前attempt内部日志，确认350断点续训恢复
- 完成剩余元筛选、选择及原始后续校准/开发/确认/内部评估和扩展计划

## 保存与分析口径

学校文件先落盘，完成项与匹配断点继续复用。运行监控推送进度，自动摘要推送表格/报告；只有本轮`LATEST.json`及其manifest/Release分片上传器回执确认后，才标完整归档发布。主计划完成不自动包含“可选外测”或扩展训练。

统计以参与者为单位，区分预先登记对比、敏感性、机制探针和多轮探索。数学性质与软件测试不直接证明未来识别性能改善。

原始输出还包括`runs/source_training_v2`、`runs/source_classical_v2`、`runs/meta`、`runs/training_campaign_v1`；并行目录是新编排入口。当前47/61为已观测主线动作数，不是第一轮全部科研工作的总进度。后续含校准profile、开发筛选/选择、确认、内部检查和原有两条源扩展。
