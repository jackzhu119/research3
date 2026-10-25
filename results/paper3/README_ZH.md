# 论文3 · 按轮分类的结果归档

[仓库主页](https://github.com/jackzhu119/research3) · [最新进度](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/LIVE_STATUS_ZH.md) · [八轮实验](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/experiments/README_ZH.md) · [全部分支](https://github.com/jackzhu119/research3/blob/research/paper3-protocol-v2-w0/docs/BRANCHES_ZH.md)

## 八轮结果入口

| 轮次 | 本次整理确认 | 结果分区 |
|---|---|---|
| 1 | 1作业运行、2作业排队；旧350更新记录尚未见RESUMED，需查看当前attempt日志。 | [第1轮](round-01/README_ZH.md) |
| 2 | 66/66主计划完成，报告完成；完整包正在学校打包。 | [第2轮](round-02/README_ZH.md) |
| 3 | 174/174主计划完成，报告完成；完整归档待顺序上传。 | [第3轮](round-03/README_ZH.md) |
| 4 | 772011运行；仓库11:09快照48/204（23.5%），该计数不是当前实时采样。 | [第4轮](round-04/README_ZH.md) |
| 5 | 0/105；772070/772071/772072已提交，均因AssocGrpGRES排队。 | [第5轮](round-05/README_ZH.md) |
| 6 | 0/198；772073/772074/772075已提交，均因AssocGrpGRES排队。 | [第6轮](round-06/README_ZH.md) |
| 7 | 学校主计划273/273完成，报告COMPLETE；仓库旧20/273为滞后计数，完整归档待上传。 | [第7轮](round-07/README_ZH.md) |
| 8 | 0/2835；772121/772122/772123已提交，均因AssocGrpGRES排队。 | [第8轮](round-08/README_ZH.md) |

## 文件分类

| 文件或目录 | 内容 | 是否完整预测或模型备份 |
|---|---|---|
| monitor-snapshots/ | 本次分类冻结的原始回传及来源 | 否 |
| previous/ | 先前归档的历史回传 | 否 |
| automatic-summary/ 与 AUTO_SUMMARY_LATEST.json | 已声明的分析摘要与报告 | 否 |
| final/ 与 LATEST.json | 上传器发布的完整本轮输出索引、manifest和Release分片 | 按manifest声明范围确认 |

第2/3/7轮完整上传器已在学校运行，最后人工观测为第2轮打包37.53%；这不是GitHub网络上传百分比。当前表是带日期的分类记录，之后完整归档以上传器的新回执为准。

完整输出归档范围不自动包括原始EEG、共享源模型与Python环境。原始回传继续写各轮progress分支，学校本地文件继续保留。
