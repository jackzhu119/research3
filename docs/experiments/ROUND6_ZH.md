# 第6轮：曲率形状预测平均与Pan外部评价

[八轮总索引](README_ZH.md) · [最新回传汇总](../LIVE_STATUS_ZH.md) · [后续安排](../ROADMAP_ZH.md)

## 本轮研究内容

比较Schur形状预测平均、条件头、同中心/同边际控制，并检验Pan外部会话迁移。

计划规模：**198：6字典＋60内部核心＋48敏感性＋12参考＋72Pan；Weibo另36**。任务成本不同，完成任务比例不能直接换算剩余GPU时间。

## 本次整理核对

核对日期：2026-10-25（北京时间）；每条原始回传有独立更新时间，后续以[自动汇总页](../LIVE_STATUS_ZH.md)和[本轮原始回传](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/monitoring/paper3-round6/three-jobs-status.json)为准。

0/198；772073/772074/772075已提交，均因AssocGrpGRES排队。

资源配置：**`hpc_gpu1`，3个独立单卡作业；各12CPU/64G，总36CPU/192G**。`hpc_gpu`对应A100 PCIe 40GB资源组，`hpc_gpu1`对应A100 SXM4 80GB资源组；节点名`gpu1`与分区名`hpc_gpu1`不同。以上GPU型号为资源组配置，不替代作业内的实测设备记录。

## 源码、设计与结果

- [本轮代码与数学设计](https://github.com/jackzhu119/research3/tree/paper3-r6-progress/experiments/followup_20261024_r6)
- [实时回传原始文件](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/monitoring/paper3-round6/three-jobs-status.json)
- [按轮分类的结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-06)
- 学校输出：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r6`
- [本轮整理前的设计与部署记录](../records/20261025-before-reorganization/ROUND6_ZH.md)

历史说明中的“未提交”、旧作业号和旧资源只对应当时交付时刻；当前入口不沿用这些旧状态。源码分支与固定提交下载继续保留。

## 后续步骤

- 等待关联组GPU额度允许作业分配
- 按既定任务DAG训练/评估并自动回传摘要
- 完成后分析及归档

## 保存与分析口径

学校文件先落盘，完成项与匹配断点继续复用。运行监控推送进度，自动摘要推送表格/报告；只有本轮`LATEST.json`及其manifest/Release分片上传器回执确认后，才标完整归档发布。主计划完成不自动包含“可选外测”或扩展训练。

统计以参与者为单位，区分预先登记对比、敏感性、机制探针和多轮探索。数学性质与软件测试不直接证明未来识别性能改善。

固定发布启动说明：[online/LAUNCH_ZH.md](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/online/LAUNCH_ZH.md)。已有作业正在排队，查看说明不代表需要再次提交。
