# 论文3第三轮：已设计，尚未提交

第一、二轮保持原目录和各自进度分支；这里的第三轮尚无识别效果结果。

- [完整数学设计](experiments/followup_20261024_r3/DESIGN_ZH.md)
- [实验矩阵：174项](experiments/followup_20261024_r3/EXPERIMENT_MATRIX.json)：21字典、78核心、72敏感性、3参考；可选18项外部迁移另计。
- [程序与运行说明](experiments/followup_20261024_r3/RUNBOOK_ZH.md)
- [131项软件测试与验证范围](experiments/followup_20261024_r3/VALIDATION.json)
- [固定下载启动包](experiments/followup_20261024_r3/online/START-PAPER3-R3-ONLINE.py)
- [第三轮状态](monitoring/paper3-round3/status.json)
- [Weibo衍生缓存及许可](data/paper3_round3/WEIBO_CACHE.json)

默认学校hpc_gpu，最多3单卡独立任务，各12CPU/64G；可按配额传24CPU/128G。无累计12小时限制；复用已完成项，保留字典和逐会话评价断点，配额不足等待。GitHub同步失败不停止训练。学校GPU尚未实跑，数学和软件验证不代表显著提升。
