# 第6轮 · 已有论文分析数据

原样保存已有报告、CSV、计划及任务回执；没有重新训练或重新计算统计。

[完整文件与校验清单](manifest.json)

| 原文件 | 字节数 | 精确分片清单 |
|---|---:|---|
| analysis/collection-metadata.json | 7139 | [原始字节与 SHA256](source-manifests/analysis/collection-metadata.json.json) |
| analysis/condition-summaries.jsonl | 8328891 | [原始字节与 SHA256](source-manifests/analysis/condition-summaries.jsonl.json) |
| analysis/selected-basis-validation.jsonl | 83836 | [原始字节与 SHA256](source-manifests/analysis/selected-basis-validation.jsonl.json) |
| analysis/source-index.json | 341322 | [原始字节与 SHA256](source-manifests/analysis/source-index.json.json) |
| analysis/task-index.csv | 79147 | [原始字节与 SHA256](source-manifests/analysis/task-index.csv.json) |
| analysis/task-results.jsonl | 573983 | [原始字节与 SHA256](source-manifests/analysis/task-results.jsonl.json) |
| plan.json | 220596 | [原始字节与 SHA256](source-manifests/plan.json.json) |
| reports/budget_comparisons.csv | 1213786 | [原始字节与 SHA256](source-manifests/reports/budget_comparisons.csv.json) |
| reports/budget_curves.csv | 601218 | [原始字节与 SHA256](source-manifests/reports/budget_curves.csv.json) |
| reports/cv_selection_frequency.csv | 59883 | [原始字节与 SHA256](source-manifests/reports/cv_selection_frequency.csv.json) |
| reports/diagnostics.csv | 5632957 | [原始字节与 SHA256](source-manifests/reports/diagnostics.csv.json) |
| reports/evaluation_summaries.csv | 1655455 | [原始字节与 SHA256](source-manifests/reports/evaluation_summaries.csv.json) |
| reports/primary_comparisons.csv | 70918 | [原始字节与 SHA256](source-manifests/reports/primary_comparisons.csv.json) |
| reports/primary_multimetric_comparisons.csv | 7309 | [原始字节与 SHA256](source-manifests/reports/primary_multimetric_comparisons.csv.json) |
| reports/report.json | 45237336 | [原始字节与 SHA256](source-manifests/reports/report.json.json) |
| reports/sensitivities.csv | 97360 | [原始字节与 SHA256](source-manifests/reports/sensitivities.csv.json) |
| reports/sensitivity_comparisons.csv | 126993 | [原始字节与 SHA256](source-manifests/reports/sensitivity_comparisons.csv.json) |
| reports/stress.csv | 0 | [原始字节与 SHA256](source-manifests/reports/stress.csv.json) |
| reports/stress_comparisons.csv | 0 | [原始字节与 SHA256](source-manifests/reports/stress_comparisons.csv.json) |
| reports/timings.csv | 769522 | [原始字节与 SHA256](source-manifests/reports/timings.csv.json) |
| reports/transfer.csv | 838555 | [原始字节与 SHA256](source-manifests/reports/transfer.csv.json) |
| reports/transfer_comparisons.csv | 1362588 | [原始字节与 SHA256](source-manifests/reports/transfer_comparisons.csv.json) |
| runtime/runtime-corrections.json | 3366 | [原始字节与 SHA256](source-manifests/runtime/runtime-corrections.json.json) |

按各源清单的分片顺序直接拼接，不添加换行。所有分片均经完整远端 GET/SHA256 核验。
缺失项、报告时效和任务完整性见清单 metadata；文件已回传不代表科研结论已验证。
本快速通道与完整逐试次输出归档的 LATEST.json 分别保留。
