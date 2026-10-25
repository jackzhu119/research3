# 写论文所需结果：轻量传输审计

核对2026-10-25已有第2、3、7轮回传及实际report/worker代码。未修改仓库或学校任务，未扫描逐试次预测、checkpoint或原始EEG。

## 结论

不必等15万文件的完整结果包，已有GitHub摘要已可阅读主要效果、预设比较、置信区间、预算曲线、压力与迁移结果。优先另传学校已经生成的报告/CSV与任务回执，即可形成完整的论文分析输入；模型断点和逐试次文件留给原完整备份流程。

现有GitHub摘要不是空数据，但也不是完整可重算数据。它们被monitor代码每表截为最多400行，且移除了参与者级向量。R3诊断只有400/2420行；R7计时400/1480、诊断400/31799行。必须补传已有完整reports文件才能完整分析这些表。

## 已经能读的结果

统一分类路径：

- 第2轮：https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-02
- 第3轮：https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-03
- 第7轮：https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3/round-07

本次整理已在上述目录monitor-snapshots/<blob>/status.json保存原字节：

| 轮次 | 原监控状态blob | 内容 |
|---|---|---|
| 2 | 700c1e026b910d1fa4eddb053e408d8c6f2c9b7a | 66/66；10主比较、75预算曲线、50预算比较、20参考比较、4敏感性、30诊断、18诊断比较，全部表行已回传 |
| 3 | 59198a756412743f285f13e36365bd4926d274d4 | 174/174；10主比较、195评价汇总、预算/敏感性/压力表、195计时；2420诊断中只回传400；本报告transfer为空，不可声称Weibo已完成 |
| 7 | 469d3e2d0c0ee776e8c12e27dc66df41ff123e88 | 主计划本地273/273、报告COMPLETE；12BA比较、24BA/NLL主比较、370评价汇总、173迁移、258迁移比较；计时/诊断被截断。顶层20/273是旧计数，不能抹去后续完成证据 |

主要比较已有参与者数量、候选/基线均值、差值、配对sign-flip p、Holm p、个人bootstrap置信区间、种子覆盖与配对会话覆盖。统计单位是参与者，不把种子或会话当独立样本；不是新的盲确认实验。

## 最快的第一层白名单

每轮输出分别为runs/followup_20261023_v2、runs/followup_20261024_r3、runs/followup_20261024_r7。直接读取以下已知浅层路径，不递归任务目录：

1. plan.json：精确矩阵、变体、种子、预算、依赖和主比较族。
2. reports/report.json、reports/*.csv，以及可能存在的根目录report.json、*.csv。R3/R7真实完整报告在reports/report.json，不能仅找根report.json。
3. 每个计划task.key下result.json、selected.json、error.json（存在才读）：完成状态、同源/数据身份、选模依据、elapsed_seconds。result.json不包含评价会话数据，不能误当参与者结果表。
4. 精简manifest：每文件SHA256/字节数、来源目录、源时间、计划task覆盖、报告状态/表总行数、读失败与未传文件、声明不含checkpoint和完整预测。

已有完整报告已含计算好的比较、区间与多重校正，不需要重跑训练或重跑报告。CSV/JSON直接入GitHub按轮paper-ready/分区，后续Codex可直接读取。若单文件较大，可按原report的table拆成独立JSON/CSV，保留值、完整行数与源文件SHA，明确这是无重统计的重排；不要因API大小限制默默截断或丢表。

## 需要重新统计或参与者散点图时

R3/R7每个评价目录为task/evaluations/{partition}/B{budget}/{scenario}/summary.json。该摘要含participants数组（participant_id、BA、BCE）、会话数量、expected_sessions/查询签名、data_identity/parameter_id。它可支持BA/BCE的参与者配对重算，读一个summary不用读105个会话预测文件。

但summary的participants仅BA/BCE，不含完整NLL/Brier/ECE/负翻转等参与者指标。要重算这些指标，可额外导出一个紧凑session-metrics.csv/jsonl，将每个会话现有JSON的标量指标、参与者/会话/seed/partition/budget/scenario、匹配查询签名汇集；NLL等若只在概率数组中可得，需读取概率并按既有report.row_metrics公式生成，不得伪称纯拷贝。这个补充可放后续阶段，不阻塞当前论文报告阅读。

所有逐会话JSON都含预测/概率/labels/run ids等数组，正是文件数量巨大的来源。不应为了第一层上传去逐个扫描它们，更不应递归validation历史或.pt断点。

## 写论文能与不能做什么

完整报告/CSV足够写既定方法比较、预算曲线、压力/敏感性、迁移、计时、已有机制指标及置信区间表。旧摘要已可做初稿，但被截断的计时/诊断不能当全量机制证据。参与者原始差值、逐run/逐试次分析、修改检验规则、误差审计或精确独立复现仍需要上面第二层表或完整归档。

本次不根据测试数量、数学恒等式或软件CPU验证宣称实验显著改善。R2/3/7的主要结果已经包含未改善甚至下降的比较，论文应如实报告，不因上传目录名paper-ready就预先认定成功。
