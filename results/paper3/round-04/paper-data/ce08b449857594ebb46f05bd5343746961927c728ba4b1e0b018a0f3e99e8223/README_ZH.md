# 第4轮 · 已有论文分析数据

原样保存已有报告、CSV、计划及任务回执；没有重新训练或重新计算统计。

[完整文件与校验清单](manifest.json)

| 原文件 | 字节数 | 精确分片清单 |
|---|---:|---|
| analysis/collection-metadata.json | 6750 | [原始字节与 SHA256](source-manifests/analysis/collection-metadata.json.json) |
| analysis/condition-summaries.jsonl | 10172262 | [原始字节与 SHA256](source-manifests/analysis/condition-summaries.jsonl.json) |
| analysis/selected-basis-validation.jsonl | 293575 | [原始字节与 SHA256](source-manifests/analysis/selected-basis-validation.jsonl.json) |
| analysis/source-index.json | 317925 | [原始字节与 SHA256](source-manifests/analysis/source-index.json.json) |
| analysis/task-index.csv | 88951 | [原始字节与 SHA256](source-manifests/analysis/task-index.csv.json) |
| analysis/task-results.jsonl | 801153 | [原始字节与 SHA256](source-manifests/analysis/task-results.jsonl.json) |
| plan.json | 240710 | [原始字节与 SHA256](source-manifests/plan.json.json) |
| reports/budget_comparisons.csv | 1301423 | [原始字节与 SHA256](source-manifests/reports/budget_comparisons.csv.json) |
| reports/budget_curves.csv | 738918 | [原始字节与 SHA256](source-manifests/reports/budget_curves.csv.json) |
| reports/cv_selection_frequency.csv | 848537 | [原始字节与 SHA256](source-manifests/reports/cv_selection_frequency.csv.json) |
| reports/diagnostics.csv | 2919419 | [原始字节与 SHA256](source-manifests/reports/diagnostics.csv.json) |
| reports/evaluation_summaries.csv | 1088942 | [原始字节与 SHA256](source-manifests/reports/evaluation_summaries.csv.json) |
| reports/primary_comparisons.csv | 78876 | [原始字节与 SHA256](source-manifests/reports/primary_comparisons.csv.json) |
| reports/report.json | 31280754 | [原始字节与 SHA256](source-manifests/reports/report.json.json) |
| reports/sensitivities.csv | 148914 | [原始字节与 SHA256](source-manifests/reports/sensitivities.csv.json) |
| reports/sensitivity_comparisons.csv | 193812 | [原始字节与 SHA256](source-manifests/reports/sensitivity_comparisons.csv.json) |
| reports/stress.csv | 178149 | [原始字节与 SHA256](source-manifests/reports/stress.csv.json) |
| reports/stress_comparisons.csv | 697910 | [原始字节与 SHA256](source-manifests/reports/stress_comparisons.csv.json) |
| reports/timings.csv | 159101 | [原始字节与 SHA256](source-manifests/reports/timings.csv.json) |
| reports/transfer.csv | 0 | [原始字节与 SHA256](source-manifests/reports/transfer.csv.json) |
| reports/transfer_comparisons.csv | 0 | [原始字节与 SHA256](source-manifests/reports/transfer_comparisons.csv.json) |
| runtime/runtime-corrections.json | 3366 | [原始字节与 SHA256](source-manifests/runtime/runtime-corrections.json.json) |

按各源清单的分片顺序直接拼接，不添加换行。所有分片均经完整远端 GET/SHA256 核验。
缺失项、报告时效和任务完整性见清单 metadata；文件已回传不代表科研结论已验证。
本快速通道与完整逐试次输出归档的 LATEST.json 分别保留。
