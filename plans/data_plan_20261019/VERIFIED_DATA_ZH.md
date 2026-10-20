# 本轮真实缓存与训练分工

以下数量来自本次原生解析、固定版本缓存 manifest 和原生产回读证明，不是文献规模或训练结果。最终部署验收另须固定 index 的全部对象独立完整 GET/SHA256；以交付包的最终验收回执为准。

| 数据 | 实际覆盖 | 有效 LR trial | 用途与门禁 |
|---|---:|---:|---|
| 主线 source4：BNCI2014-001、Cho2017、Lee2019-MI、PhysioNet-MI | 505 原档，224 个 provider ID | 28,830 | 主线共同 f0/Rref；跨库自然人身份独立未证实 |
| Stieger2021 官方 v1 | 62 人、598 会话 | 72,522 | 固定 40/11/11；对应 396/97/105 会话。原 LR 89,699，剔除 17,177；内部留出缓存完整性检查不等于内部科学检验 |
| Yang2025 官方 v5 的 2C 子集 | 51 个 provider ID、153 个 data/evt pair | 30,591 | 独立 `BASE4_PLUS_YANG`；925 个非二类事件排除。源分析窗不冒充已证实物理 MI 起止；未知 SI 只接受已证实共同正增益并在模型前 joint RMS |
| Dreyer2023，NEMAR nm000250 v1.0.5 sourcedata | 87 个 provider ID、520 EDF | 20,792 | 独立 `BASE4_PLUS_DREYER`；原生完整 MI 与有效 UID 均 20,792，波形 QC 排除 0。128 份新旧缓存已实际数值对照；论文1已有暴露，不作新盲法外测 |
| Ma2022 官方公开 v1，native_v2 缓存 | 25 人、125 个 EDF/events pair | 11,988 | 独立三通道开发资料，左/右 5,983/6,005；原始删除 trial 的采集位置未知，不作原始付费 B20；分组未冻结，不自动 fit |
| CAR3 source 缓存 | 与 source4 相同的 224 个 ID/原 trial | 同一批 28,830 | 不增加独立 N；独立 f0 工程准备不代表三通道 Rref/G/完整14方法已验收 |
| 已登记外测与新增候选 | 当前仅有元数据证据 | 不计新增 trial/N | HS、Wang、Leeuwis、MILimbEEG 等遵循 W7 权限；不下载或解封 query。Zuo/NETBCI 等不匹配任务不能改名并入左右手 MI |

Yang 与 Dreyer 是两条分别由 source4 加一个新库组成的五 source 支路，不组成默认六库池。比较方法共享各支路真实 f0/Rref、相同 seed 的基线所选 epoch、相同更新量与选择预算；不把来源变化单独解释成人数增加的因果效应。上述 provider ID 不可直接相加成已证实独立自然人数。

完整原始 EEG 不进入云端训练恢复集合。固定包携带原协议 ZIP、新研究数据方案 DOCX、代码、角色/版本/原生证据、portable Python 和最小 SDK；数值缓存与固定依赖 wheel 从已有 R2 按 index 的明确路径恢复。逐文件大小、SHA256、角色、dataset 和 IO profile 都必须一致；不能用 HEAD、ETag、已存在文件或清单生成冒充新的完整回读。

GPU 开机前科学 fit、GPU 训练与科研性能结果均为 0。开机后先执行唯一受控恢复 pilot，通过真实检查点/预测/数值恢复与阶段门后，才继续主线及两条独立增强支路。失败保留原 claim、预算和检查点，不换目录重复 fit。Ma、CAR3 与封存外测继续保留各自门禁。
