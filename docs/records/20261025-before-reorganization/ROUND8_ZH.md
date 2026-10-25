# 第八轮实验与运行入口

<!-- PAPER3_ROUND8_20261025 -->
## 第八轮：已发布，等待用户提交

第八轮基于第二、三、七轮已回传摘要，研究部分协方差输运、Bures/源度量几何和保持协方差的类别方向校正。固定计划 **2835项任务，其中24项元训练**；Stieger内部、Pan与Weibo分开分析，探索性重复验证，不预先声称提升。

- 资源：`hpc_gpu` A100 PCIe40GB，三个独立单卡作业，各16CPU/64G；合计48CPU/192G。
- [设计与公式](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/DESIGN_ZH.md) · [前轮依据](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/EVIDENCE_ZH.md) · [文献](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/LITERATURE.md) · [单行启动](https://github.com/jackzhu119/research3/blob/paper3-round8-20261025/experiments/followup_20261025_r8/LAUNCH_ZH.md)
- 无计算节点下载、全量test/pilot或累计12小时限制；保存双代训练断点与逐会话结果，独立GitHub回传。
- 进度分支`paper3-r8-progress`；摘要归档`paper3-results:results/paper3/round-08`。完整预测和模型仍在学校本地。
- 已通过学校核心软件版本的本地CPU验证，覆盖118项独立测试；线上raw/API完整包下载与安装核对通过。学校GPU现场尚未提交验证。

此前轮次的历史资源/计数记录保留，实际队列请看各轮最新观测。
<!-- /PAPER3_ROUND8_20261025 -->

冻结源EEGNet，只使用新会话前20个已获取位置中的有效试次。新增算法利用训练域类别参考与支持集类别差的交换子生成受限旋转；同一正则协方差下比较零旋转、正反方向、等幅随机方向和标签置换。解析步长改善的是所定义类别二阶对比目标，不保证未来准确率提升；已有数学与候选贡献边界见设计和文献。

全量网格仅用于机制分析；独立统计单位仍是参与者。已查看过的队列不是重新盲测。完整代码、精确清单、数据来源与恢复说明均位于上面独立分支。安装包已包含两个外部裁剪缓存，提交到学校时直接使用，源模型和Stieger缓存沿用学校既有结果。

