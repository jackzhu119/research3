# Paper3 R10：证据保留的查询函数校准

基于实际 R1–R9 CSV、JSON、预测归档和代码核查的独立算法实验包。核心是用支持集可辨识的完整查询函数集合，决定 EA 源预测值得修改多少；不是给已有 GP 方差门控更名。

- [中文研究方案及原始效应证据](R10_RESEARCH_PROPOSAL_ZH.md)
- [数学定义、证明、有限数值界及消融](R10_MATH.md)
- [15→165→57任务实验、主要比较和停止条件](R10_EXPERIMENT_PLAN.md)
- [完整实现、四作业学校部署与恢复](R10_IMPLEMENTATION.md)
- [真实来源及文献审查](research/literature_review.md)
- [R2–R7原始证据](research/evidence_r2_r7.md) · [R8/R9最新原始证据](research/evidence_r8_r9.md)

资源固定：**4作业，各1GPU、16CPU、68G内存**。自包含安装包默认只安装，只有显式 `--submit` 才提交。CPU/合成验证已经执行；本轮真实 EEG 实验、学校 GPU 预检及新数据独立确认尚未执行，不能提前声称性能突破或论文录用价值。

`delivery/START-PAPER3-R10.py` 无需 git clone 或 pip，完整指令见 `delivery/SCHOOL_COMMANDS.md`。原始 R1–R9 代码与结果保持原位；此包的历史模块副本有来源 SHA，修复只影响新包。
