# 第2轮：空间成分门控与可靠性保护

[八轮总索引](README_ZH.md) · [最新回传汇总](../LIVE_STATUS_ZH.md) · [后续安排](../ROADMAP_ZH.md)

## 本轮研究内容

从前20个试次估计旋转与相对增益的可用程度；比较固定/条件门控、误标与空间漂移稳健性。

计划规模：**66：36核心＋24敏感性＋6参考**。任务成本不同，完成任务比例不能直接换算剩余GPU时间。

## 本次整理核对

核对日期：2026-10-25（北京时间）；每条原始回传有独立更新时间，后续以[自动汇总页](../LIVE_STATUS_ZH.md)和[本轮原始回传](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)为准。

66/66主计划完成，报告完成；完整包正在学校打包。

资源配置：**`hpc_gpu`，最多3个独立单卡作业；各12CPU/64G**。`hpc_gpu`对应A100 PCIe 40GB资源组，`hpc_gpu1`对应A100 SXM4 80GB资源组；节点名`gpu1`与分区名`hpc_gpu1`不同。以上GPU型号为资源组配置，不替代作业内的实测设备记录。

## 源码、设计与结果

- [本轮代码与数学设计](https://github.com/jackzhu119/research3/tree/paper3-v2-progress/experiments/followup_20261023)
- [实时回传原始文件](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/monitoring/paper3-followup-v2/status.json)
- [按轮分类的结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-02)
- 学校输出：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2`
- [本轮整理前的设计与部署记录](../records/20261025-before-reorganization/ROUND2_ZH.md)

历史说明中的“未提交”、旧作业号和旧资源只对应当时交付时刻；当前入口不沿用这些旧状态。源码分支与固定提交下载继续保留。

## 后续步骤

- 保留正在打包的上传进程
- 全部分片上传并核验，发布LATEST.json完整归档回执

## 保存与分析口径

学校文件先落盘，完成项与匹配断点继续复用。运行监控推送进度，自动摘要推送表格/报告；只有本轮`LATEST.json`及其manifest/Release分片上传器回执确认后，才标完整归档发布。主计划完成不自动包含“可选外测”或扩展训练。

统计以参与者为单位，区分预先登记对比、敏感性、机制探针和多轮探索。数学性质与软件测试不直接证明未来识别性能改善。
