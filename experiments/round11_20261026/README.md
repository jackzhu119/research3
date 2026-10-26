# Paper3 R11：前缀—未来得分输运与置换正则研究

R11以实际已上传结果及R10设计为依据，研究冻结EEGNet下从少量校准标签到后续预测的有向更新。基础机制与既有meta-learning/噪声正则的重叠已明确，不提前宣称性能突破。

- [最新原始证据审计](R11_EVIDENCE_AUDIT_ZH.md)
- [中文研究提案与文献差异](R11_RESEARCH_PROPOSAL_ZH.md)
- [完整公式、证明及退化情形](R11_MATH.md)
- [10,270任务、550配置、25方法与统计](R11_EXPERIMENT_PLAN.md)
- [实际实现、恢复与后台上传](R11_IMPLEMENTATION.md)
- [学校完整一行入口](delivery/SCHOOL_COMMANDS.md)

固定4个独立单卡作业，每个16CPU、64G。默认campaign自动完成开发→选型→旧队列探索性评价。持久科学进程、只读缓存、按任务隔离失败和增量双通道上传用于避免重复启动和全局中断。

安装默认不提交；正式执行须显式`--start --submit`。CPU及合成验证不代表真实EEG或学校CUDA结果。R1–R10代码、作业和结果不改写。
