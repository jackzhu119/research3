# 第九轮：支持信息约束的源预测残差GP

[独立源码与在线入口](https://github.com/jackzhu119/research3/tree/paper3-round9-20261025/experiments/followup_20261025_r9) · [完整设计](https://github.com/jackzhu119/research3/blob/paper3-round9-20261025/experiments/followup_20261025_r9/DESIGN_ZH.md) · [实测依据](https://github.com/jackzhu119/research3/blob/paper3-round9-20261025/experiments/followup_20261025_r9/evidence/RESULTS_FOR_R9_ZH.md)

第七轮同一ST-relative主体增加head自由度时，支持BCE下降而未来BA由61.31%降至56.84%；EA达到62.21%。这是探索性机制线索，不是独立的因果证明。第九轮冻结EEGNet，以源logit为均值做支持空间GP残差，比较MAP、Laplace积分、固定权重与支持信息位置相关的锚定；零信息时精确回锚定预测。

1548项：6特征参考、12有限meta_validation选参、234核心、432压力、702先验敏感性、108锚定强度、54标签置换。三个独立单卡作业进入hpc_gpu，各16CPU/64G。已有三种子源模型直接复用，无新源训练。数据沿用Stieger/Pan/Weibo并明确探索性重复使用。

状态：设计、代码及开发环境测试已发布；未提交学校GPU作业，无第九轮效果结论。默认在线入口仅准备，显式--submit才启动队列控制器。每会话续跑，冻结特征共享缓存，节点不下载、不pip、不全量test/pilot；无累计12小时限制。

论文通道先发布完整CSV及人/会话小表、配对身份与信息分箱；完整通道独立增量归档逐查询数据、验证和断点。结果分区为paper3-results/results/paper3/round-09，PAPER_DATA_LATEST、ARCHIVE_PROGRESS、LATEST分别记账。源码已有但不代表学校已跑通；实际GPU现场表现、资源排队和科研增益仍待运行确认。
