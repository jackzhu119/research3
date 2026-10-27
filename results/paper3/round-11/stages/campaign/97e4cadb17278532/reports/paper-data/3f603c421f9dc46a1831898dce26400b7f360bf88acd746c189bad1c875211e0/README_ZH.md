# 第11轮 · 已有论文分析数据

原样保存已有报告、CSV、计划及任务回执；没有重新训练或重新计算统计。

[完整文件与校验清单](manifest.json)

| 原文件 | 字节数 | 精确分片清单 |
|---|---:|---|
| plan.json | 20250104 | [原始字节与 SHA256](source-manifests/plan.json.json) |
| reports/comparisons.csv | 103386 | [原始字节与 SHA256](source-manifests/reports/comparisons.csv.json) |
| reports/method_summary.csv | 143105 | [原始字节与 SHA256](source-manifests/reports/method_summary.csv.json) |
| reports/participant_metrics.csv | 2601381 | [原始字节与 SHA256](source-manifests/reports/participant_metrics.csv.json) |
| reports/participant_seed_metrics.csv | 7346548 | [原始字节与 SHA256](source-manifests/reports/participant_seed_metrics.csv.json) |
| reports/report-status.json | 433 | [原始字节与 SHA256](source-manifests/reports/report-status.json.json) |
| reports/report.json | 1712212 | [原始字节与 SHA256](source-manifests/reports/report.json.json) |
| reports/session_metrics.csv | 84949827 | [原始字节与 SHA256](source-manifests/reports/session_metrics.csv.json) |

按各源清单的分片顺序直接拼接，不添加换行。所有分片均经完整远端 GET/SHA256 核验。
缺失项、报告时效和任务完整性见清单 metadata；文件已回传不代表科研结论已验证。
本快速通道与完整逐试次输出归档的 LATEST.json 分别保留。
