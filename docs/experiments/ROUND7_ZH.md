# 第7轮：有限窗时空Lie、头容量与源相对风险

[八轮总索引](README_ZH.md) · [最新回传汇总](../LIVE_STATUS_ZH.md) · [后续安排](../ROADMAP_ZH.md)

## 本轮研究内容

拆解静态空间与时间成分、支持集可辨识性和源相对风险目标；Pan重复探索单列。

计划规模：**273：15字典＋72内部核心＋72敏感性＋21参考＋93重复Pan；Weibo另60**。任务成本不同，完成任务比例不能直接换算剩余GPU时间。

## 本次整理核对

核对日期：2026-10-25（北京时间）；每条原始回传有独立更新时间，后续以[自动汇总页](../LIVE_STATUS_ZH.md)和[本轮原始回传](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/monitoring/paper3-round7/status.json)为准。

学校主计划273/273完成，报告COMPLETE；仓库旧20/273为滞后计数，完整归档待上传。

资源配置：**`hpc_gpu`，1节点1作业3GPU；总36CPU/128G共享**。`hpc_gpu`对应A100 PCIe 40GB资源组，`hpc_gpu1`对应A100 SXM4 80GB资源组；节点名`gpu1`与分区名`hpc_gpu1`不同。以上GPU型号为资源组配置，不替代作业内的实测设备记录。

## 源码、设计与结果

- [本轮代码与数学设计](https://github.com/jackzhu119/research3/tree/paper3-r7-progress/experiments/followup_20261024_r7)
- [实时回传原始文件](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/monitoring/paper3-round7/status.json)
- [按轮分类的结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-07)
- 学校输出：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r7`
- [本轮整理前的设计与部署记录](../records/20261025-before-reorganization/ROUND7_ZH.md)

历史说明中的“未提交”、旧作业号和旧资源只对应当时交付时刻；当前入口不沿用这些旧状态。源码分支与固定提交下载继续保留。

## 后续步骤

- 上传器在第2/3轮后处理第7轮完整归档
- 分析迁移/主指标/消融及多重比较；保留探索性标记

## 保存与分析口径

学校文件先落盘，完成项与匹配断点继续复用。运行监控推送进度，自动摘要推送表格/报告；只有本轮`LATEST.json`及其manifest/Release分片上传器回执确认后，才标完整归档发布。主计划完成不自动包含“可选外测”或扩展训练。

统计以参与者为单位，区分预先登记对比、敏感性、机制探针和多轮探索。数学性质与软件测试不直接证明未来识别性能改善。

## 2026-10-25：完整总报告定向补传

第七轮273/273已完成；论文通道已上传完整14张CSV。旧快速上传器因64MiB限制跳过了 reports/report.json，现已发布[不限文件总大小的单文件补传工具](../../operations/round7_report_supplement_20261025/README_ZH.md)及[单行执行指令](../../operations/round7_report_supplement_20261025/RUN-R7-REPORT-ONE-LINE.sh)。只读这一份报告、按约1MiB分片保留全部原始字节与SHA；不重打包任务目录，不修改训练、已有CSV或慢速全归档。

当前状态是工具已发布、真实下载及隔离安装已验证，等待学校执行后的 REPORT_JSON_LATEST.json 回执，不能据此声称总报告已经上传。未来Codex读取该指针、manifest及restore_report.py即可按回执commit恢复完整JSON。原14张CSV仍可先分析；完整逐试次与模型归档另记账。
