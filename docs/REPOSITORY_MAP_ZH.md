# 仓库内容地图

[默认主页](../README.md) · [全部26个分支](BRANCHES_ZH.md) · [八轮实验](experiments/README_ZH.md)

## 按内容定位

| 内容 | 位置 | 用途 |
|---|---|---|
| 当前首页、最新回传、后续安排 | 默认分支README、docs/LIVE_STATUS_ZH.md、docs/ROADMAP_ZH.md | 了解当前阶段与后面要做什么 |
| 八轮研究分类 | 默认分支docs/experiments/ROUND1至ROUND8 | 研究问题、数学路线、规模、当前资源与跨分支源码入口 |
| 原始方法、协议、合同 | 默认分支p3work/、protocol/、contracts/、plans/ | 核心实现和科学定义 |
| 数据/预处理/来源 | manifests/、provenance/、audit/、学校数据说明 | 追溯信号、划分、缓存和输入证据 |
| 学校部署与恢复 | tools/及两个migration操作分支 | 当前操作封装和历史交付；固定提交下载保留 |
| 第2至7轮实验实现 | 各轮progress分支experiments/对应目录 | 代码、设计、矩阵、数据缓存与运行手册 |
| 第8轮实现 | paper3-round8-20261025分支experiments/followup_20261025_r8/ | 独立源码与可下载启动包 |
| 运行进度 | 各轮progress分支monitoring/ | 已提交异步快照，含各自源时间 |
| 分轮结果 | paper3-results分支results/paper3/round-NN/ | 历史快照、自动摘要、完整归档分开保存 |
| 历史交付与记录 | distribution/*、旧research部署分支、archives/、docs/records/ | 固定版本与旧页面原文，保留可追溯性 |
| 软件检查 | tests/及各实验包tests/ | 本地软件行为验证，不代表学校GPU已现场实测或效果显著 |

## 分支使用约定

默认分支负责总导航；实验实现按轮隔离；运行监控继续写既有progress分支；结果按轮写paper3-results。分支名、目录和固定提交仍被已启动程序引用，因此通过分类入口整理，不搬移正在使用的代码或输出。

## 三种结果范围

`previous/<commit>/`为历史回传快照；`automatic-summary/<hash>/`与`AUTO_SUMMARY_LATEST.json`为分析摘要；`final/<snapshot>/`与`LATEST.json`为上传器声明完整归档的入口。完整模型和逐试次预测是否已离校备份，必须按本轮回执的scope与manifest判断。
