# 第8轮：部分协方差输运与保持协方差的类别方向旋转

[八轮总索引](README_ZH.md) · [最新回传汇总](../LIVE_STATUS_ZH.md) · [后续安排](../ROADMAP_ZH.md)

## 本轮研究内容

比较recolor/Bures/源度量、部分输运及证据门控；新增保持正则化协方差的类别对比交换子旋转与解析步长验证。

计划规模：**2835：3类参考＋24元训练＋207核心＋792压力＋27标签空效应＋1566敏感性＋216旋转敏感性**。任务成本不同，完成任务比例不能直接换算剩余GPU时间。

## 本次整理核对

核对日期：2026-10-25（北京时间）；每条原始回传有独立更新时间，后续以[自动汇总页](../LIVE_STATUS_ZH.md)和[本轮原始回传](https://github.com/jackzhu119/research3/blob/paper3-r8-progress/monitoring/paper3-round8/three-jobs-status.json)为准。

0/2835；772121/772122/772123已提交，均因AssocGrpGRES排队。

资源配置：**`hpc_gpu`，3个独立单卡作业；各16CPU/64G，总48CPU/192G**。`hpc_gpu`对应A100 PCIe 40GB资源组，`hpc_gpu1`对应A100 SXM4 80GB资源组；节点名`gpu1`与分区名`hpc_gpu1`不同。以上GPU型号为资源组配置，不替代作业内的实测设备记录。

## 源码、设计与结果

- [本轮代码与数学设计](https://github.com/jackzhu119/research3/tree/paper3-round8-20261025/experiments/followup_20261025_r8)
- [实时回传原始文件](https://github.com/jackzhu119/research3/blob/paper3-r8-progress/monitoring/paper3-round8/three-jobs-status.json)
- [按轮分类的结果](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-08)
- 学校输出：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261025_r8`
- [本轮整理前的设计与部署记录](../records/20261025-before-reorganization/ROUND8_ZH.md)

历史说明中的“未提交”、旧作业号和旧资源只对应当时交付时刻；当前入口不沿用这些旧状态。源码分支与固定提交下载继续保留。

## 后续步骤

- 等待三张A100 PCIe40GB可用配额
- 执行协方差输运及保持协方差的类对比旋转验证、压力/消融/迁移矩阵
- 训练中持续保存断点及摘要回传；完成后进行预设配对检验

## 保存与分析口径

学校文件先落盘，完成项与匹配断点继续复用。运行监控推送进度，自动摘要推送表格/报告；只有本轮`LATEST.json`及其manifest/Release分片上传器回执确认后，才标完整归档发布。主计划完成不自动包含“可选外测”或扩展训练。

统计以参与者为单位，区分预先登记对比、敏感性、机制探针和多轮探索。数学性质与软件测试不直接证明未来识别性能改善。

[完整数学设计](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/DESIGN_ZH.md) · [文献](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/LITERATURE.md) · [前轮依据](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/EVIDENCE_ZH.md) · [交付验证](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/VALIDATION.json)。2835项是固定有限矩阵，其中仅24项元训练，其余为参考/评价/压力等任务。
