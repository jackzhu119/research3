# Paper3 R10：优化后的探索性续跑

本包消费学校已经完成的原15项机制实验，自动登记选型修订并准备后续165项配对实验。它不覆盖原R10或R11代码和结果，不重训原14项评估；原seed20261018特征参考按已封存SHA精确复用，其余两个种子建立参考。165个节点为3个参考节点加18方法×3种子×3已用队列。

资源固定为 **4个独立单卡Slurm作业，每个1GPU、16CPU、64G内存**。科学进程在每个allocation内持久运行，冻结模型和只读元数据缓存复用；任务异常、进程崩溃及网络故障分别处理，失败任务不拖停全部实验。完成会话、模型及归档按身份恢复。

- [选型修订、配对公式、原始结果和证据边界](R10_CONTINUATION_SCIENCE_ZH.md)
- [部署、恢复、165任务与完整上传](R10_CONTINUATION_IMPLEMENTATION_ZH.md)
- [包含下载与安装的一行正式启动](delivery/SCHOOL_COMMANDS.md)
- [发布前完整验证](research/CONTINUATION_FINAL_VERIFICATION.json)
- [原机制阶段CSV、JSON、来源与SHA](research/evidence/r10_mechanism/PROVENANCE.json)
- [保留的R10数学算法](R10_MATH.md) · [原研究方案](R10_RESEARCH_PROPOSAL_ZH.md)

新选型只使用原selection四人相对EA的参与者配对增益标准误，check四人不重排候选。旧性能门槛均保留为诊断，不再据此中断后续探索。当前固定规则在原开发数据选择a0.3-c0.1，原规则选a0.3-c0.5/STOP的只读重建同时保留。不能把新选择写成原协议的PASS，也不能将已看过的check或holdout/Pan/Weibo重新称为独立确认。

主算法数学未改。工程兼容优化包括R9原生float32概率到统一float64概率格式的可审计归一化；保留原生输出审计，不改GP/EEGNet内核。真实历史基线缺少已冻结artifact时只隔离对应方法/种子；固定统计家族保留缺失比较，不伪造基线或悄悄取交集。

自包含入口 `delivery/START-PAPER3-R10-OPTIMIZED.py` 默认仅安装，不导入Torch、不运行pytest、不申请GPU、不提交。只有显式 `--continue --submit` 才会在学校正式启动四个后台作业；显示 `DETACHED_CONTROLLER_STARTED` 后可关闭终端。安装、准备、CPU诊断与提交的边界见启动说明。本地交付验证不代表学校CUDA实测或新165任务效果，结果正、负、无效均完整报告。

原 `R10_IMPLEMENTATION.md` / `R10_EXPERIMENT_PLAN.md` 描述初始分阶段协议，保留供审计；当前续跑应使用上述CONTINUATION说明与OPTIMIZED入口。
