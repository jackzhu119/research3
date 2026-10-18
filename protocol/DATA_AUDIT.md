# 第三篇论文数据审计与适配合同

审计日期：2026-10-18。本文是**研究与实现交接**，不是已下载、已训练或外部测试完成的回执。本轮未下载任何 EEG 大文件，未启动训练，未修改 P1/P2 或既有训练包。实际看过的 HS v6 小文件为 channels.tsv、eeg.json 与官方 manifest；其余代码、论文和归档页的证据等级在下文逐项说明。机器可读登记见同目录 DATA_REGISTRY.json。

## 1. 推荐的数据层级

采纳本轮最终分工：**主 f0 只依赖四个稳定源 BNCI、Cho、OpenBMI、PhysioNet；Stieger2021 负责大规模长期开发；Wang2026 保留全库作健康确认；HS-Stroke v6 保留全库作临床主确认。Yang2025 仅作为通过 gate 后的可选 enhanced-source 支路，当前不作为主链依赖。** 不把所有新库同时加入训练，也不为凑规模把任务不同的 ME、上下肢或指级任务当左右手 MI 合并。

| 层级 | 数据 | 本轮允许的研究用途 | 必须保留的边界 |
|---|---|---|---|
| 基础预训练 | BNCI2014_001、Cho2017、OpenBMI/Lee2019_MI、PhysioNet MI | 源域预训练、源域按人/按库验证、固定分类器与编码器比较 | 源人身份不可同时出现在开发或最终测试；只用明确左右手 MI |
| 可选增强预训练，当前 gate 未过 | Yang2025 2C；P1 报告 51 人，P3 eligible 未核 | 通过单位/锚点/原档 gate 后，可复用 P1 BDF 与明确标签资格，生成独立增强支路缓存 | 不构成主 f0 依赖、不计入主源规模；不是全新外测；不能以 P1 18 通道 NPZ 冒充 clinic11 |
| 长期主开发 | Stieger2021，62 人、598 个原始 session 文件 | 按人划分开发拟合/开发验证；开发序贯停止、更新强度、窗口及比较器 | 先固定人名单；全部 62 人都是开发资源时不得再称其中人为第三篇全新确认；默认不同时加入源编码器预训练 |
| 健康确认 | Wang2026 全库 | 冻结模型、超参数、输入合同、比较器后，执行一次预定确认；允许按协议消耗 support 标签 | 全体不用于调参；纯 LR 与 2D 中 LR 子任务分开；分实验组报告 |
| 临床主确认 | HS-Stroke v6 全体 14 人 | 封存后的跨 session、少校准确认 | 14 人不能支持康复疗效结论；23,382 是作者去重后 trial 报告数，P3 eligible 数仍未知 |
| 已探索的迁移开发 | Kumar2024/P2；Dreyer2023/P1 | 调试、历史复现、敏感性比较 | 不能重新包装为第三篇全新独立确认 |
| 独立输入族 | Ma2022 | 3 通道 C3/Cz/C4 或自有 montage 的敏感性开发 | 不含全部 clinic11；禁止补零/插值后混称同输入主结果 |
| 临床/硬件储备 | Liu2024；UET175；Alawieh2025 | 各自通过许可、时钟、身份和通道 gate 后，预先指定用途 | UET175 需要作者密码且时间戳被 ERD 筛选；Alawieh 有刺激/交叉组并非额外新人数 |
| 任务扩展 | Ding/Finger2025；SEFMEID2026 | 后续独立任务论文或附加实验 | Finger 不是左右手二分类；SEFMEID 每个周期 ME 后 MI，不能把两段当独立人的新样本 |

本方案仍可“多训练”：四个主源做受试者/数据集均衡的预训练规模曲线、种子和特征消融；在 Stieger 的固定开发人群上系统训练和审计适应策略。Yang 的 gate 全部通过后再运行独立的增强源支路，不延迟主流程、不把预期参与者计入已可用规模。保存所有已定义协议的中间 checkpoint、逐 trial 预测、适应轨迹和标签成本即可服务后续写作。不要靠反复查看 Wang/HS 结果来决定第三篇方法。

所有跨库“生物学受试者无重叠”默认为未验证。特别是同 Bin He 实验室的 Stieger、Wang、Finger 没有可核对的跨研究身份表。Wang 主队列的 BCI-naïve 与 Finger 主队列既往 BCI 经验是相关线索，**不是跨研究无重叠的证明**。现阶段可说“整库留出的外部研究”，不能无条件说“所有参与者均证实从未在任何开发集出现”。如要作最严格 unseen-person 主张，需作者级身份不重叠说明；在此前保留该限制，且 Finger 不进入本轮训练。

## 2. Yang2025：能复用什么，不能复用什么

### 已核证据

原文：Yang 等，Scientific Data，2025-03-23，DOI [10.1038/s41597-025-04826-y](https://link.springer.com/article/10.1038/s41597-025-04826-y)。数据 DOI：[10.25452/figshare.plus.22671172](https://doi.org/10.25452/figshare.plus.22671172)。

原研究分 51 人 2C 与 11 人 3C，均 3 天。原始记录 64 个槽位，其中前 59 是 EEG，其余 5 个辅助槽位未记录有效辅助信号；作者处理后为 58 EEG、250 Hz、4 秒片段。**51 与 11 的 subject 编号分别从 001 开始，跨子库不能只用编号作全局 identity。** 当前 P3 只纳入已审计的 51 人 2C；3C 不自动扩大源名单。

P1 的已批准资格修订位于 [YANG_LABEL_ELIGIBILITY README](https://github.com/jackzhu119/cross-subject-mi-eeg/blob/0539379578818b3d71f995192e5e8a81cf4d5aef/research_runs/YANG_LABEL_ELIGIBILITY_20261018/README_zh.md)，固定 commit 为 `0539379578818b3d71f995192e5e8a81cf4d5aef`。其资格 manifest SHA256：

`5291d362185a8db9bf45ddc95d9cb49d40de57893d2fd4da8681a802b5ab36f9`

该 P1 审计报告 153 session 的明确 1/2 事件合计 **30,591**：1=15,295，2=15,296。此数是 P1 原始明确标签总数，不是已验证的 P3 可用 trial 数。保留全部 51 人，不猜 255、不补缺失 onset。五个不足计划计数的 session 以及 017/02 额外 255 均已明确登记。

[P1 p1_data.py](https://github.com/jackzhu119/cross-subject-mi-eeg/blob/0539379578818b3d71f995192e5e8a81cf4d5aef/research_runs/SOURCE_EXPANSION_20261017/p1_data.py) 的真实 CHANNELS18 是：

`Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CP2, CP4, Pz`

它**包含全部 clinic11**；因此原档通道适配有根据。

### 原始层可以复用

- 已下载的 `data.bdf / evt.bdf`、解压成员、完整文件 SHA256、下载回执；P3 使用独立引用/只读路径，不更改 P1 原档。
- BDF header 解析、24 bit 解码、每通道物理/数字标定、事件提取和时钟对齐代码，可在新适配器内复用，但新适配器有自己的 commit/hash。
- P1 v3 的明确标签资格、原始事件索引、实际 onset、原始 9 号标记分隔位置和异常清单。
- 旧 P1 数值缓存只用于历史复现或**明确保持原合同的独立基线**，不能假装是重新生成的 P3 缓存。

### P3 数值缓存必须重建

P1 是 CAR18、160 Hz、3 个频带视图、6 秒滤波上下文、320 点的 2 秒结果、trial broad-view RMS 无量纲化；P3 v1 是 CAR11、128 Hz、3 秒上下文、单个 8–30 Hz 视图、256 点 2 秒结果。CAR18 后再切 11、再重采样不会等价于从原始 11 通道生成 CAR11；上下文与滤波也不同。旧 checkpoint 的空间核/采样率/输入长度不能默认为兼容。

P1 源码明确写：Yang BDF 的单位字符串为 `?V`，尚未确认是伏还是微伏。P1 的已审计规则仅在完整、相同的标定证据下利用公共正比例增益在 trial RMS 中抵消，并保留“物理单位未解决”标记。P3 v1 的模型端 `standardized_trials` 已是 trial 去均值/RMS，CSP/协方差与 EEGNet 均调用它。因此**可在新 adapter/缓存合同中保留 calibrated numeric 和 unit_unresolved，再统一进入已有无量纲模型输入**。不能把中间数组标成 V；不能对该源计算或解释 250 µV 绝对阈值。每通道共同标定条件不成立时，尺度不变性也不能当作补救理由。

### 时钟不能过度解释

P1 已明确：`marker + 1.5 s` 是分析参考，不是已证实的 MI 开始时刻。MOABB 解释与作者归档处理脚本的 epoch 解释不同；P1 用原 marker 相对 `[2,4) s` 作为交集分析窗，原上下文 `[0,6) s`。P3 如保持这个分析窗，可把新 3 秒滤波上下文明确写成 marker `[1.5,4.5) s`，但不得称为“精确统一的 MI 起点后 0.5–2.5 秒”。此选择应在 source/development 冻结。滤波上下文可能包含准备/任务结束附近信号，其含义需保留。

9 号标记目前仅是**原始操作性分隔位置**，无独立官方语义证明其一定是实际 block 开始。禁止每保留 40 个事件重新划 run；禁止把保留顺序号写成已恢复的原计划试次号。

## 3. Stieger2021：真正的大规模长期开发库

原文：[10.1038/s41597-021-00883-1](https://link.springer.com/article/10.1038/s41597-021-00883-1)。**官方原始归档**：[Human EEG Dataset for Brain-Computer Interface and Meditation](https://figshare.com/articles/dataset/Human_EEG_Dataset_for_Brain-Computer_Interface_and_Meditation/13123148)，DOI 10.6084/m9.figshare.13123148，2021-02-13；当前官方页显示 **351.01 GB、CC BY 4.0**。这是本轮核验后对之前镜像容量和许可记录的修正。不能把 NEMAR 722.5 GB 或挑战镜像 399 GB 写成同一原始归档的精确大小。

原文报告 62 人、598 个 session 文件、269,099 个**全部任务** trial、7–11 次访问、超过 600 小时。使用 64 通道帽，发布数据矩阵实际 **62 EEG×time**、1000 Hz、µV。每次 session 计划 450 trial；3 种任务 LR/UD/2D 各 6 run×25 trial。纯 LR 的计划上限是 **598×150=89,700**，不是 269,099。P3 真正保留的左右手、时间窗完整、通道合格 trial 数必须由原始文件审计生成，当前为 null。

原文给出足够恢复结构的字段：

| 原字段 | 含义/合同 |
|---|---|
| BCI.data | 按 trial 的 cell，62×nTime，单位 µV |
| BCI.time | 相对于 target presentation 的 ms；0 是目标出现，反馈从 +2000 ms 开始 |
| TrialData.tasknumber | 1=LR；2=UD；3=2D |
| TrialData.targetnumber | 1=right，2=left，3=up，4=down/rest |
| TrialData.runnumber / trialnumber | 原 run 与 trial 顺序，必须保存 |
| TrialData.triallength / resultind | 反馈控制时长/结束索引；不能把其后的反馈结果/休息凑进任务窗 |
| TrialData.artifact、chaninfo.noisechan | 作者原始 QC 标志；与我们自己的 QC 分开 |
| metadata.date / day / time | 原论文明确的 session 时间字段；真实解析、缺失率仍需 gate |
| chaninfo.label | 原始通道顺序，精确映射 clinic11 |

**不能直接把 MOABB 的全 subject Raw 当正式时间源。** 已读固定 v1.7.2 的 [stieger2021.py](https://github.com/NeuroTechX/moabb/blob/v1.7.2/moabb/datasets/stieger2021.py)：其循环将 trial 拼接、按 targetnumber 放事件，没有保存原 tasknumber/runnumber/trialnumber；简单选 left/right 会把 2D 的左右方向也混入，物理 run 边界丢失。该版本在原 cue 0 处放事件，却以反馈时长 triallength 和配置 tmax 比较；这不是“纯 MI 反馈开始后的 3 秒”的证明。正式 adapter 直接读原 BCI MAT，先按 tasknumber=1 限制任务，再显式定义 cue/feedback 锚点。

### 短试次与选择偏差

反馈在击中/误中时提前结束，最短 trial 不能凭 rest 或事后结果补成 3 秒。若主输入以反馈开始为零并要求上下文 `[0,3)`，必须要求 3 秒真实反馈 MI；这会按原控制速度/成败选择记录，可能减少快速成功试次，不能忽略。

执行前只在 Stieger 的开发人群统计：每人/类/session/实验组的可用时长分布、各候选窗口保留率、原 QC 标记，以及完整 run query 覆盖。主候选为反馈开始后的 3 秒上下文与 [0.5,2.5) 秒分析窗。预先登记的 fallback 为：固定 Stieger meta-train人群的受试者平均 3 秒完整反馈保留率不足 80% 时，全源/开发统一改为 2 秒上下文与 [0.5,1.5) 秒的 1 秒分析窗，并重新匹配训练。分母限定所有真实纯 LR trial，另报完整性/坏道/QC 排除；同时展示各人/类/组/session 分布，避免 trial 总量加权掩盖弱覆盖人群。正式定义以 EXPERIMENT_PROTOCOL.md 为准。阈值与最终输入合同在开发集冻结；对 Wang/HS 不根据模型成绩或可用比例临时改窗。短窗族若采用不同长度，应重新匹配源预训练，不能和 v1 checkpoint 直接交换。主报告须给按窗口的选择流程和排除原因；跨窗口效果比较优先用共同 eligible trial，额外覆盖率独立报告。

Stieger 还含 8 周 MBSR/waitlist 干预与后续训练，原研究先筛掉 baseline 天花板表现者。因此它适合漂移/学习开发，但不是无干预自然漂移随机人群。记录 MBSR/control，不把组别构成变化当算法进步。

## 4. Wang2026：新健康整库确认，先把语义补全

原文：[10.1038/s41467-026-75435-5](https://link.springer.com/article/10.1038/s41467-026-75435-5)，Nature Communications，2026-07-15。官方归档：[32293995](https://kilthub.cmu.edu/articles/dataset/EEG_Datasets_for_Sensory-Guided_Joint_Learning_in_Motor_Imagery_Brain-Computer_Interfaces/32293995)，2026-05-29，**59.36 GB，CC BY-NC 4.0**。作者代码：[bfinl/SensoryGuidedJointLearning](https://github.com/bfinl/SensoryGuidedJointLearning)，本轮 main commit `57710f76a5ddd6f9a3aad28566ca55776a424400`。

### 人数与实际 session

原文的主研究是 31 人：joint learning 15、BCI2000 8、tactile 8；Methods 另明确独立招募 EEGNet control 8 人。当前 development MOABB 为四归档预留 39 个全局 ID，但官方页仍用“31 人四组”的简略表述。**实际释放可读人数、每组文件清单与 session 数还未下载验收，因此 registry 不把 39 作为已验证可用总数。**

主 31 人均完成前 4 session；21 人完成 session 5–6；16 人有超过 2 个月后的新增 LR 和 2D 随访。不能写“31/39 人每人全部 6/8 个可用 LR session”。原文 session 1–2 是 LR、3 混合 LR/UD、4–6 为 2D。主二分类确认限定纯 LR session/run；2D 中抽出左右方向应作为任务上下文变化的附加分析。

### 时长、来源结构和字段

原文 Methods 的实验结构为：2 秒 fixation，方向 cue 后 **5 秒 MI**，再 2 秒休息。作者 [updateModel.py](https://github.com/bfinl/SensoryGuidedJointLearning/blob/57710f76a5ddd6f9a3aad28566ca55776a424400/updateModel.py) 明确公共数据是 trial-segmented MATLAB `runData`，**不是原 BCI2000 .dat**；固定长信号 `trialSignal` 是 samples×channels×trials，例如 **5000×62×30**，读取 `meta.sampling_rate_hz` 与 `meta.selected_channels`。BCI2000 变长文件为每 trial 的 cell，不能统一补 5 秒；原始 trialInfo 和 meta 应完整保留。

[development MOABB wang2026.py](https://github.com/NeuroTechX/moabb/blob/ed1aacdb39a445efe13a4f026673b14cc2913b85/moabb/datasets/wang2026.py) 对固定 EEGNet 类型与 BCI2000 类型采用不同标签映射，并保留四归档 file ID：
- JointLearning: 64710999
- BCI2000Control: 64710750
- TactileControl: 64710993
- EEGNetControl: 64710990

这些可作下载清单线索，正式下载前仍对官方 version manifest 核对。该 adapter **忽略 meta/trialInfo，并把 run 改成计数键**，因此不能原样支持正式时间审计；它还明确留注公共信号单位未说明，当前 1e-6 转换须作者确认。P3 不得沿用这个推测，未知单位保留 null/calibrated_numeric，并只在已证明公共比例因子的尺度不变支路使用。

原作者代码示范按当前文件出现的 label 值重新映射成 0 开始类别，P3 **不能**逐文件如此重映射。必须先验证 group/task/字段类型的全局语义，保存原 label code；校准、LR、UD、2D 不能共享不经检查的数字 code。

固定 5 秒结构使 ≥3 秒输入可行的证据较强，但还必须用实际小样本 header 确认分段起点是否正好 MI 开始、时间轴、trial 顺序、单位和每组变体。当前只是代码/协议级通过，不是已读 EEG schema 通过。

### 支持/查询方案

每 session 通常是 run1 无反馈校准；run2 无触觉反馈测试；run3–5 训练；run6 无触觉终测。主要可执行方案：**run1 的前 Bmax=20 个合格 trial 中按预算请求标签，所有方法共同在完整 run2 上评估**。若任一数据条件未满足此结构，按预先的 gate 报告缺失，不把文件的第几个条目猜成该 run。

run6 可作为跨中间真实人机训练后的远期 query。其间参与者已接受原研究 decoder/触觉训练，离线重放的 P3 没有因果控制这些经历；不能把 run6 改进归因于 P3 促进了人体学习。joint/tactile 组试次成对、copy/new 受前一个 trial 成绩影响；不随机打散，query 不拆到 pair 中间，反馈/触觉段分层，置信区间以人聚合。

## 5. HS-Stroke v6：4 秒原 trial 可用，时钟只是分层序数的证据更强

原文：[10.1038/s41597-026-07742-x](https://doi.org/10.1038/s41597-026-07742-x)。固定归档：[30747806/6](https://figshare.com/articles/dataset/HS_Stroke_dataset_an_upper-limb_motor_imagery_EEG_dataset_of_chronic_stroke_patients/30747806/6)，DOI **10.6084/m9.figshare.30747806.v6**，2026-07-24，CC BY 4.0。

本地实际已恢复并只读核对的官方 manifest：
`p3_workflow_sources/paper3_pretrain_20261018/catalog/hs_stroke_v6_official_metadata.json`。

其 21 文件精确总量 **6,903,852,376 bytes**；本轮推荐只为主适配器下载 source 和所需小 metadata，避免重复下载源与衍生：

| 成员 | bytes | 官方 file ID | MD5 |
|---|---:|---:|---|
| sourcedata.zip | 4,894,533,454 | 66772196 | a079fa4a6f3073e0ebb92102ade23731 |
| derivatives.zip | 2,005,900,019 | 66772193 | f9eff51ad9a3de6fad728b542804c73c |
| README.md | 2,502 | 66878795 | accaf4a89cfd1486d38a375c343b42b9 |
| updateNotice.pdf | 516,476 | 66960068 | 01a11dabc6341b8ce60486a0069c8a50 |

官方 channels.tsv/eeg.json 实际验证为 **11 EEG、256 Hz、V、linked mastoids reference**：
`FC3,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP4`。不能写 16 通道。eeg.json 的 RecordingType=continuous 并不改变 MAT 实际按 trial 保存的事实。

作者 repo 固定 commit `33d2f269256442f7b5af7199f5d5259d5b390182`：
- [sourcedata.py](https://github.com/KINLAWWW/HS-Stroke/blob/33d2f269256442f7b5af7199f5d5259d5b390182/hs_stroke/datasets/sourcedata.py)
- [preprocess_derivatives.py](https://github.com/KINLAWWW/HS-Stroke/blob/33d2f269256442f7b5af7199f5d5259d5b390182/hs_stroke/datasets/preprocess_derivatives.py)
- [dataset/README.md](https://github.com/KINLAWWW/HS-Stroke/blob/33d2f269256442f7b5af7199f5d5259d5b390182/dataset/README.md)

源码读取源 `EEGdata` 为 channel×time×trial，transpose 后逐 trial 处理；`EEGdatalabel[:,0]` 一 trial 一 label；从每原 trial 的 **9–13 秒**截出 4 秒 MI。因此**不需要把 1 秒复制/补齐为 3 秒**。作者 derivative 生成器本身也保存 trial×11×1024 的 4 秒数组，然后其训练 loader 才按 1 秒切窗。93,528 是 4×23,382 的窗口数，不是独立 trial 数。

源码还暴露出三个必须修正的问题：

1. `sourcedata.py` 中循环变量 `run_id` 实际是 trial 枚举索引，不是物理 run。不能复制该字段命名后宣称恢复了真实 run。保留 source_record_id=原 MAT 文件身份、trial_index_in_record；physical_run_id 只有证据支持时才写，否则 null。
2. 作者导出的 derivative 将同 session 多文件拼为一个文件，未保存足够的原文件/run 边界映射。因此正式时序支路优先 source。已读代码只按 sub/ses/文件路径排序、累计 trial offset；没有读取绝对 recording timestamp，不能声称恢复了真实记录日历间隔。session/trial 顺序应标明“提供方保留序数”；原计划缺口/删除重复产生的间隔不能补造。
3. 不照搬作者按整个 trial 的多窗口拟合 AutoReject、跨该 trial 窗口 min-max 等处理来进行严格时序比较；也不照搬原 cross-session odd/even split，它不是只用过去预测未来的设计。

主 pipeline 从真实源 trial 抽 `[9,12)` 的 3 秒上下文，再按统一相对 MI offset 取 `[9.5,11.5)` 两秒分析窗，前提是实际 shape/label定义验证通过。这样数据完整性足够，所有窗口仍挂同一个独立 trial ID，计费一次标签。分类 label code 的 left/right 对应需实际读取 v6 events.tsv/events.json 校验，目前本轮不猜数字映射。

14 人、278 session 来自论文；23,382 trial 为 v6 作者更正后的报告。当前没有读到全部 ZIP 内 EEG，不保证每个 session 都能形成前20支持与下一完整run查询。若 physical_run 无法证实，但有独立证据确认源 trial 数组仍为 acquisition order，则可预定采用**同 session 前20 + 后续完整 original-trial block**，明确是 ordinal within-session sequential，不伪称跨run；仅有可重复的数组索引不等于取得该顺序证据。顺序 gate 不过则该序贯支路仍阻塞，不能用猜测代替；此 fallback 必须在最终查询解封前登记，不能看到成绩后选择。

## 6. UET175：可作新临床储备，尚不能替代封存主确认

原文 PDF：[UET175, Frontiers in Neuroscience, 2025-06-17](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1580931/pdf)，DOI 10.3389/fnins.2025.1580931。作者入口：[nmk-k66-uet/UET175](https://github.com/nmk-k66-uet/UET175)，固定 commit `e05a1ae12421514d2392db169f20437b1c60673d`。

它是越南 Hospital 175 的 **30 名卒中患者**，2024-10 至 2025-01 收集；不是 HS 华山医院 14 人的重命名。使用 Emotiv EPOC Flex 22 通道、128 Hz，4 类分别左右上肢/下肢 MI；每 run 12 trial，各类3次；2 秒放松、2 秒 cue、4 秒 MI，run 间休息20秒。主左右上肢任务可取两类，但“lifting arm”不能无说明写成 HS 的 wrist extension 或 finger MI。

原文报告初始 **2,640 trial**、97 被拒、**2,543 保留 trial**；其“220 sessions”与同页“每人平均1.43、最多3 sessions”不相容。220×12=2,640 表明其可能混用 run/session，但**未经原档不能自动纠正为220物理session**。真实 session 数、日期、可用 LR 数均为 null 待验收。

容量有两种不同证据：
- 原文：解压约 434 MB，Raw CSV 中位数约 2 MB。
- 作者 Git LFS 指针：`Data_UET175.rar` 对象 **178,930,296 bytes**，SHA256 `0c203151d1127b2c38f6ae1646237d455b3a16bfe590e36c728bf6e4705cdd2e`。134 bytes 是 Git 指针本身，不能错算成数据大小。未下载验证 RAR 完整性。

**实际获取门槛：**README 要求向作者索要解压密码。本轮未联系作者，也未获密码，因此是 access_pending；文章的 CC BY 不自动等于已确认数据的独立授权条款，取得密码时同时保留科研使用条件。

**最重要的时间偏差：**原文的 Event timestamp 是从 cue 最后0.5秒至 MI 结束寻找 ERD≥30% 的 1秒片段，并移除不显著/noisy trial。它是 response-selected timestamp，不是原始刺激 onset。直接用此窗训练可能有生理筛选和选择偏差，不能用来证明在未经筛选的新患者连续流上有效。正式 adapter 应以 Raw CSV + Session Setup JSON + Action Label TXT + 可核对的原始同步信息恢复固定 cue/MI 时轴。恢复失败则只列为“作者选窗数据的独立探索”，不加入主序贯协议、不声称任意3秒原MI可恢复。

22 通道图说明类似 BNCI22，但仍需原文件逐列通道证据才能确认 clinic11。不能仅看图“很像”就指定列号。相比 Liu2024 的40 trial/人，UET175 有更丰富的真实 run/日期潜力，但当前密码、原始 cue 时钟、session计数歧义未解决，故并列储备，不替代 HS 主确认。2026-08-31 的 [LoRA foundation-model preprint](https://arxiv.org/abs/2609.00282) 使用过 UET175，是应比较的方法近邻；该文的预印本结果不构成本项目预期效果承诺。

## 7. 其它数据的操作性限制

- **OpenBMI**：54 人、2 session，当前包固定只加载 train_run。论文中的 online/test 段不能在本地标签未验明前自动加倍 trial 数。其真实日历间隔与 adapter session名分开记录。
- **PhysioNet**：只取 run 4/8/12 的左右手 MI；run 3/7/11 是 ME，run 6/10/14 是双手/双脚 MI。全库 ZIP 1.9 GB/未压缩3.4 GB是官方页面给出的全库大小，不是目标 LR 子集大小。
- **Liu2024**：报告50人、每人单session40trial，development MOABB 指出3对内容重复并默认47人，但这不是本轮对原始版本的完整独立校验。保留原始ID与去重映射，不能悄悄换成“作者50人均独立”或“已确认47人”。主 Bmax20 才有后20作为 query；B40 无 query。它只作跨人少校准附加任务。
- **Alawieh2025**：[Zenodo v1 15454355](https://zenodo.org/records/15454355)，2025-05-18，18.3 GB显示大小；25健康+2SCI。主20人5天；cross-over 4人是主rest组的再试，不能加成31人；刺激与rest/单脉冲条件分开。官方页采样512Hz、32/64帽；同组每个run时钟、MI长度、真实clinic11以及数据许可仍需原档核验。现 MOABB离线入口不能证明已加载全部5天online。
- **Finger2025**：DOI [10.1038/s41467-025-61064-x](https://doi.org/10.1038/s41467-025-61064-x)，[KiltHub 29104040](https://kilthub.cmu.edu/articles/dataset/EEG-BCI_Dataset_for_Real-time_Robotic_Hand_Control_at_Individual_Finger_Level/29104040)，v2显示502.77GB/CC BY-NC4。21人主组与后续16人扩展的同一研究角色应由身份表确认；包含MI和ME、128EEG1024Hz。它不是左右手任务，且主组有先前BCI经验/离线筛选；禁止将全502.77GB称为全新的LR-MI预训练集。
- **SEFMEID2026**：Scientific Data，2026-09-19，DOI [10.1038/s41597-026-08298-6](https://doi.org/10.1038/s41597-026-08298-6)，[ScienceDB 10.57760/sciencedb.30795](https://doi.org/10.57760/sciencedb.30795)。50人32EEG1000Hz+fNIRS，需注册、DUA与审批。每周期为8s rest→8s ME→8s rest→8s MI；原周期是分组单位，ME与MI不能随机拆到不同折；通道表实际验明前不能承诺clinic11兼容。

## 8. 所有 adapter 的最小验收合同

先做 metadata/schema audit，之后才生成训练缓存。每个 dataset/group/subject/session/source_record/trial 记录至少包含：

1. 官方 DOI 与版本、原下载 URL/file ID、压缩和解压文件 hash、字节数、许可证据、adapter commit。
2. dataset+cohort+provider_subject 的全局键；原 session/run/file/trial标识和各层数字排序；伪连续 Raw 索引与真实时间轴分开。
3. 原 sampling rate、原单位字符串、标定映射、原通道顺序、精确 clinic11 索引；缺通道不补零/猜列/插值。
4. 原 event code、明确左右手语义、原 marker onset、analysis anchor、是否已验证 MI onset、trial真实开始/结束与可用上下文。
5. 原任务、反馈、触觉/刺激、配对/原周期、日期以及缺失标记；不知道的字段写 null，不造假。
6. 不依赖模型成绩的资格规则和逐项排除原因；保留 unknown label，不猜标签；去重保留原对象映射，试次级与窗口级分别计数。
7. P3数值合同 hash、原始到新缓存的对账、每trial全部窗口同一身份、模型输入无量纲化之前的单位/QC证据。
8. 来源、开发、最终测试的人级名单和可见性；模型/比较器冻结后才解封最终 query 真值。单纯下载归档不等于该数据已用于训练。

主实验固定B20、无门控。对于次级序贯协议，B4初始化、B8首次独立审计；所有标签含audit均计独立trial预算。Bmax=20主比较，各方法共同query；停止后到Bmax以及query期间冻结全部学习状态，包括BN、normalizer和未标注自适应。部署实际被独立audit过的checkpoint，不在audit后重新拟合再声称同一审计效力。query可固定为下一整run，无法核实run时才使用预先登记的顺序block分支。B40仅限有足够数据的预定次级曲线，使用自己的共同Q40，不与Q20混成同一曲线。

## 9. 当前仍未解除的 gate

- Wang 四归档真实人数/session/trial、单位、trialInfo时间起点、组别code与run身份：尚未看 EEG 原档，不能宣称全量可用。
- Stieger 纯LR/候选时长/完整query覆盖、实际日期解析、clinic11坏道覆盖：需原MAT审计；89,700仅计划上限。
- HS v6 原MAT shape与labels、物理run语义、绝对日期：只有小metadata+作者代码证据；4秒可用性是代码级高可信，仍需schema receipt。
- Yang P3 新数值缓存、已批准v3 raw/label manifest现场复核、未知物理单位的尺度不变支路身份：待实施。
- 同BinHe实验室各库身份重叠未知，未经作者说明不得从“不同研究”推断“零受试者重叠”。
- UET175 密码/数据使用条款/固定cue恢复未通过；不发送邮件、不把该库放到自动下载训练队列。
- 新数据的 MOABB 在线页面属于 development：固定包v1.7.2不自动支持所有最新类。必须固定新增 adapter commit并保存版本，不能直接安装移动 develop 后替换实验身份。
