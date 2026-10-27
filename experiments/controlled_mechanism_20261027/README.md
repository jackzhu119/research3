# 少样本空间适配失效机制：受控验证研究

主问题：**支持集约束弱的空间更新是否真实参与负迁移，并具有独立于容量和整体预测改变量的影响？**

- [完整设计、数学干预、分阶段条件及计算方案](MECHANISM_EXPERIMENT_ZH.md)
- [六项主要比较、统计单位、功效与证伪](STATISTICS_ZH.md)
- [机器可读设计配置](EXPERIMENT_CONFIG.json) · [主要比较清单](PRIMARY_COMPARISONS.csv)
- [既有原始CSV和来源SHA](evidence/MANIFEST.json) · [最新R8–R11证据事实](evidence/LATEST_RESULT_FACTS.json)

本次核对到R10正式探索阶段165/165完成，主要BA比较尚无超过EA的校正后证据；R11仍不完整。研究以同容量方向、同参考函数RMS、真实更新成分删除与普通收缩对照为核心，避免重复命名R7/R10已有的head秩或Schur机制。

先导为8名开发参与者×3源seed的24个完整配对块，覆盖原有5预算；机制不获支持就停止方向路线。旧holdout/Pan/Weibo仅作探索复现，独立确认的新数据尚未落实。计算方案为4个作业，各1GPU/16CPU/64G；本次只交付设计与证据，没有新训练引擎、执行包或学校提交。

可离线核对设计与证据：

```bash
python -I -S -B experiments/controlled_mechanism_20261027/audit_design.py
```

这条命令只读本地配置与已复制的证据，不拟合模型、联网或调用Slurm。字典、source与参考UID的真实SHA以及先导τ_B仍需在开发冻结时填入，不声称本设计已可直接部署训练。
