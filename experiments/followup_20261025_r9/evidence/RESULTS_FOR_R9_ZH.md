# 从第 2、3、7 轮实测结果到第 9 轮设计

本记录是已完成结果的探索性复核。读取的是 `paper3-results` 分支的不可变 paper-data 快照，未重跑训练，也未把本次结果再次作为盲测。2026-10-25 核对：36 份计划/CSV、106 个文件分片通过 SHA-256 验证；33 张 CSV 解析完整。大 CSV 按原字段和行序恢复，重复表头、行偏移和总行数均核对；原始 CSV 字节摘要保留于 `DOWNLOAD_AUDIT.json`，重建 CSV 格式字节可能不同。

## 已有结果的主线

| 结果 | 实测数值 | 对第 9 轮的含义 |
| --- | --- | --- |
| R2 dual 相对 P | −0.00806 个百分点；Holm p=1 | 更复杂的旋转/增益门控尚未带来可辨认收益 |
| R2 P / support-EA | 内部 BA 61.4911% / 62.2132% | 新算法必须与简单且强的 EA 比较 |
| R3 profiled 相对 SGD10 / SGD100 | −4.6367 / −5.1090 个百分点；Holm p=.04785 / .02930 | 高精度拟合支持集目标不等于未来泛化提升 |
| R3 保护相对无保护 | +4.6589 / +4.6430 个百分点；Holm p=.04395 | 主要是修复相对弱版本的退化，不能写成优于强 SGD |
| R7 static full-head / support-EA | 内部 BA 56.8918% / 62.2132%；NLL .98535 / .64661 | 优先研究小支持集下的 head 适配，而不是继续增加时空自由度 |
| R7 static full-head / support-EA | Pan BA 60.4524% / 63.8214%；NLL .87047 / .63038 | 在已重复使用的外部数据上也见同方向现象，但仍属探索 |
| R7 预设主比较家族 | 12 项 BA Holm 均为 1；24 项 BA+NLL 家族最小 Holm=.13477，无正证据 | 目前不能声称 R7 新算法显著优于参照 |

## 最值得追踪的机制：支持集自由度不等于泛化证据

第 7 轮 B20 内部支持集经过 QC 后，平均仅 **15.5301 个有效试次**；所有获取位置仍计入预算，没有补抽样。全 head 有 **64 个特征系数及 1 个偏置**，中心化支持特征秩平均 **14.5301**。汇总诊断记录 `head_absorbed_spatial_fraction=1.0`、`data_only_spatial_info_rank=0.0`。这与有限样本上线性 head 的列空间已能覆盖空间校准的支持集一阶预测变化一致；它不表示所有新会话或所有未来试次上空间变化都不可辨识。

令支持集 head 雅可比为 $Z\in\mathbb R^{n\times(d+1)}$、空间雅可比为 $J\in\mathbb R^{n\times r}$。如果 $\operatorname{rank}(Z)=n$，则 $P_Z=ZZ^+=I_n$，从而 $(I-P_Z)J=0$。这解释为什么一味更好地优化经验目标、或增加空间复杂度，可能仍无法分辨有效的空间修正与对少量标签的记忆；它不是对未来误差的提升保证。

同一 ST-relative 主体下的探索性消融进一步显示：

| head 设置 | 支持集 BCE | 未来 BA | 未来 NLL | 负翻转 balanced rate |
| --- | ---: | ---: | ---: | ---: |
| rank 0（仍有偏置） | .67213 | 61.3097% | .65776 | 1.7666% |
| rank 4 | .51114 | 59.1832% | .81371 | 8.7702% |
| rank 8（平均有效 7.91） | .28334 | 56.9915% | .96813 | 18.1821% |
| full 64 | .12134 | 56.8356% | .98454 | 21.2705% |

rank 0 相对 full 的已报告差为 **+4.4741 个百分点，描述性 95% CI [2.2126, 6.6961]**；该消融没有预设确认性显著性声明。全 head 先验从 .01 增加到 1.0 时，内部 BA 从 56.2663% 改善至 58.7509%，仍低于 EA。不能凭该表直接选一个最佳 rank 再用同一内部 holdout 宣称新发现已确认。

按预算机械增加 head 维数也没有解决问题：

| 预算 | R7 budget-head BA | R7 budget-head NLL | R2/R7 共用 EA 参照 BA | EA NLL |
| --- | ---: | ---: | ---: | ---: |
| 4 | 61.0788% | .66449 | 61.2256% | .66978 |
| 8 | 58.9477% | .89842 | 61.4402% | .65604 |
| 12 | 56.8297% | 1.03423 | 61.8486% | .65003 |
| 16 | 56.1273% | 1.03788 | 61.9641% | .64814 |
| 20 | 56.7375% | .98678 | 62.2132% | .64661 |

这些趋势支持第 9 轮研究 **由支持外预测证据决定适配强度、源预测为中心的受控残差 head、非机械的有效自由度和不确定性**。第 8 轮已有协方差运输和保协方差旋转，不应把相同公式换名算新方向。

## 统计和可写论文的边界

- R2/R3 内部有效独立单位为 11 位参与者；R7 内部也是 11 位，Pan 为 14 位。105 会话×3 个种子=315 行、Pan 28 会话×3=84 行均不是新增独立样本。参与者内先聚合，再做配对推断。
- 本次复核保留原报告统计，没有新增显著性检验。R7 主比较家族属于 `exploratory_repeated_external_multimetric_family`，Pan 已被多次探索，不能称为新的独立盲测。
- `cv_selection_frequency.csv` 是空文件，不能据此声称已有 CV 稳定性证据。R3 的外部 transfer 两张表也为空。
- 所有已报告配对比较的 unmatched 行最大值为 0，汇总表 `(group,partition,budget,scenario)` 无重复键。各诊断可能单独排除不可定义会话，应看其行数，不能把缺失当作零。
- 第 7 轮 14 张 CSV 完整，但此前上传器因 64MiB 限制跳过聚合 `reports/report.json`。本记录不等于聚合 JSON 已补传；逐参与者机制相关性或新的会话模型还需要该补传或专门的紧凑指标导出。现有 CSV 没有逐参与者的完整诊断向量，不能从汇总均值编造相关系数。
- 负翻转和修正错误是 balanced rate，不应乘总试次数冒充原始计数。

## 可复核来源

- R2 快照：`results/paper3/round-02/paper-data/d3366bed6441dd7d8888d0e3660bbf167388870f5c447d221e6e6afe74c35494/manifest.json`
- R3 快照：`results/paper3/round-03/paper-data/4c4d33c077891ece00b2cc34b36261a8db1aa33ede9a388e7fe4703034b920f0/manifest.json`
- R7 快照：`results/paper3/round-07/paper-data/4d44fa6d6f6d754f70d70f1cb4d5b8439ea0721bac938ce54dc85c7138352a24/manifest.json`
- 关键原表：R2 `comparisons.csv`、`budget_curves.csv`；R3 `reports/primary_comparisons.csv`；R7 `reports/evaluation_summaries.csv`、`diagnostics.csv`、`primary_multimetric_comparisons.csv`、`sensitivity_comparisons.csv`、`budget_curves.csv`。
- `fetch_paper_evidence.py` 重建全部现有表；`analyse_paper_evidence.py` 产生 `FINDINGS.json`。没有训练/部署修改，也没有重新检验模型。
