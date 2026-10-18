# 固定 API 与证据来源（核对日期：2026-10-18）

本清单支持训练包的接口和方法说明。领域论文、最新患者数据及创新设计由独立研究报告覆盖。可移动的 latest 文档可能比实际安装版本更前沿；执行以以下固定 tag 和已记录版本源码为准。

| 内容 | 官方来源 | 本包据此落实的事实 |
|---|---|---|
| MOABB 1.7.2 | https://github.com/NeuroTechX/moabb/releases/tag/v1.7.2 | 固定发布版，2026-09-14 发布；本地实际导入 1.7.2 |
| MNE 1.13.2 | https://github.com/mne-tools/mne-python/releases/tag/v1.13.2 | 固定发布版，2026-09-11 发布；本地实际导入 1.13.2 |
| BNCI2014_001 | https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/bnci/bnci_2014.py | 公开 subject list、标注与 interval、通道和 uV 到 V 转换 |
| Cho2017 | https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/gigadb.py | 类别拼接 Raw 的构造方式、通道与幅度单位转换；拼接坐标不是原始采集时钟 |
| Lee2019_MI | https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/Lee2019.py | train/test/resting kwargs，左右原码，两个 session，uV 到 V 转换 |
| PhysionetMI | https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/physionet_mi.py | imagined/executed 分开；MI hand run 为 4/8/12，另外返回 6/10/14 双手/双脚 MI，不能误当单侧左/右手；原始数据经 MNE EDF 进入 SI |
| `get_data()` 管线 | https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/base.py | `_create_process_pipeline` 插入 `SetRawAnnotations` |
| Canonical event anchor | https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/preprocessing.py | `SetRawAnnotations.transform` 已对事件加 `interval[0]*sfreq`，并向 MNE 传 `first_samp`；下游不得再加 |
| Ma2022 可调用性 | https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/__init__.py | 安装的固定版没有导出 Ma2022；本地 `getattr` 和 doctor 确认，故只登记待适配 |
| MNE Raw 数据约定 | https://mne.tools/1.13/generated/mne.io.Raw.html | Raw EEG 读取采用 SI volts；自建 Raw 用真实 V 量级合成信号验证输入输出 |
| MNE annotation 转事件 | https://github.com/mne-tools/mne-python/blob/v1.13.2/mne/annotations.py | 无匹配事件会报错；不能依靠 broad `except ValueError` 把全部非目标/未知 run 混为一谈 |
| 数学基线 API | https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html | `lsqr` 与 `shrinkage="auto"` 的 LDA；实际固定 scikit-learn 1.8.0 |
| Logistic regression | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html | `sample_weight` 与 `lbfgs`；本包 C=1、最多 2000 次迭代并检查收敛 |
| PyTorch 固定版本安装 | https://pytorch.org/get-started/previous-versions/ | 使用官方 CPU/cu128 index 的 torch 2.8.0；本地实际验证 2.8.0+cpu，未验证 CUDA 变体 |

原始数据引用标识保存在 `p3/datasets.py` registry：BNCI 2014-001 对应 10.3389/fnins.2012.00055；Cho 数据档 10.5524/100295；Lee 数据档 10.5524/100542；PhysioNet EEG Motor Movement/Imagery 数据 10.13026/C28G6P；Ma2022 原文 10.1038/s41597-022-01647-1。这些 DOI 是来源标识，不表示本轮已下载并完成数据质控。

## 交付审查中纠正的实际问题

1. 预处理身份原先只绑定配置与 provider adapter，现补充本包 `datasets.py`、`core.py` 与 numpy/scipy 身份；所有 shard 共同契约包含实现哈希，不能在同一 prepared pool 中混入不同处理实现。
2. 全源 EEGNet 的 epoch 选择现在同时核对 source LODO 输入、配置、源码与运行环境；确定性基线状态 receipt 也绑定完整 fit identity。
3. Physionet 的 imagined-only adapter 仍会返回双手/双脚 MI run；只有明确符合已核查 canonical `hands`/`feet`/`rest` 的 run 可以带审计原因跳过。新增合成 Raw 回归测试覆盖此分支与未知标签失败。
4. `raw.n_times` 的 NumPy 整数在哈希 JSON 中显式转为 Python int，避免真实 MNE API 结果破坏序列化。

独立只读审查还确认了双向滤波的 3 秒输入截止边界、source-validation 选择的开发属性以及 CPU 断点验证不能推广到跨 GPU 硬件位级复现。这些限制已写入使用文档。
