# 第三篇方法近邻：原文阅读与可借鉴边界

核验截止：2026-10-18。本文件服务于方法和实验设计，不按影响因子凑文献。Nature Communications 是 Nature 子刊；ICLR、ICML、CVPR、EMBC 是会议；arXiv 条目是预印本。下述效果属于各自任务和协议，不能作为本研究将获得“大提升”的承诺。

**本轮结论：应把主贡献放在跨会话学习可迁移的空间适应方向。** 负向翻转正则、轻量参数适应、冻结骨干、元学习低维基和无标签候选门控都有直接先例。新方法至少要击败同字典维度的 learned-additive、同源元训练的 Meta-LoRA，以及 PCT+LoRA；不能只比未对齐、未校准的弱基线。

## M01. 2026 Nature Communications：感知引导的人机共同学习

**Sensory-guided human-machine joint learning accelerates the acquisition of motor imagery brain computer interface control**，Hanwen Wang 等，2026-07-15，Nature Communications 17, 6177。DOI: 10.1038/s41467-026-75435-5。

- [期刊全文与 Methods](https://www.nature.com/articles/s41467-026-75435-5)
- [作者代码](https://github.com/bfinl/SensoryGuidedJointLearning)，MIT；[数据](https://doi.org/10.1184/R1/32293995)

**读到的方法。** 模型在 run 内固定、训练 run 结束后更新，测试 run 不微调。CSP/LDA 的置信度先选约 20% 样本初始化 EEGNet，之后依逐样本损失排序逐渐扩大纳入比例并加权训练。Copy/New 反馈与触觉指导根据前一试次表现触发；主体是人和机器的共同学习，不是单纯离线更换损失。作者另招募了 8 名 EEGNet 对照，不能在没有文件对应关系时直接与主组总人数相加。

**借鉴。** 学习 run 与评价 run 分开、加入强 EEGNet 对照、拆开机器更新和人类训练。其自步加权是必须承认的先例。**边界。** Copy/New 试次存在行为依赖，滑窗不是独立样本；重放公开数据不能宣称获得新的触觉干预因果效果。与本法的差异在源纵向元训练空间方向和固定少试次校准，而非“首次选择可靠样本”。

阅读定位：Methods 的数据采集、在线分类、模型更新和自步学习段；作者 `updateModel.py` 与 README 的公开 `.mat` 入口。

## M02. 2025 Nature Communications：个体手指在线控制

**EEG-based brain-computer interface enables real-time robotic hand control at individual finger level**，Yidan Ding、Chalisa Udompanyawit、Yisha Zhang、Bin He，2025-06-30，Nature Communications 16, 5401。DOI: 10.1038/s41467-025-61064-x。

- [期刊全文](https://www.nature.com/articles/s41467-025-61064-x)
- [作者代码](https://github.com/bfinl/Finger-BCI-Decoding)，MIT；[数据](https://doi.org/10.1184/R1/29104040)

**读到的方法。** 离线收集个体手指 ME/MI 数据；在线每项任务先用基础 EEGNet 运行前 8 个 run，再用这些 run 微调后控制后 8 个 run。21 名完成者有既往肢体 MI-BCI 经验，16 人继续额外 MI 会话；研究还用离线重放拆解模型与人类练习的作用。

**借鉴。** 同一后续数据上重放旧/新模型、按阶段解释提升、把在线平滑和模型更新分开消融。**边界。** 研究筛选了响应者，任务是同一只手的手指，而非左右手二分类；其人群、任务表现和机器人控制效果不能搬作本研究预期值。我们可以学习这种成对比较方式，不能把普通微调本身当新方法。

阅读定位：Results 的 fine-tuning/human learning 分解；Methods 的离线任务、在线反馈、参与者及训练效应设计。

## M03. MIRepNet：必须加入的 MI 专用预训练基线

**MIRepNet: A pipeline and pre-trained model for EEG-based motor imagery classification**，Knowledge-Based Systems 343, 115966，2026-06-15 期次。DOI: 10.1016/j.knosys.2026.115966。

- [正式出版页](https://www.sciencedirect.com/science/article/pii/S0950705126006921)
- [作者预印本 PDF](https://arxiv.org/pdf/2507.20254)；[代码](https://github.com/staraink/MIRepNet)；[权重](https://huggingface.co/starself/MIRepNet)

**读到的方法。** 本轮细读的是作者预印本，标题使用 Foundation Model；不能声称逐项核对了无法访问的最终出版 PDF。方法包含通道模板/插值、8–30 Hz、250 Hz、欧氏对齐，卷积嵌入加 Transformer，以掩码重建和 MI 分类联合预训练，再做个体微调。

**借鉴。** 作为强 MI 骨干，按完全相同目标预算比较 head-only、LoRA 与本法。**边界。** 预训练表含 Dreyer2023，官方权重不能把该库当完全未见外测；部分原文比较的目标训练比例不同，不能直接继承数值排名。预处理、筛选低表现受试者及权重训练来源均须记录。主体算法不是重新包装“有预训练就提高”。

阅读定位：预印本方法、训练/微调策略、预训练数据 Table 1；代码与权重发布页。

## M04. CBraMod：跨范式 EEG 基础模型

**CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding**，ICLR 2025；作者预印本首次发布于 2024-12-10。

- [ICLR 正式论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/bbbd6d915cb90be21c1254a82d45cedd-Paper-Conference.pdf)
- [代码](https://github.com/wjq-learning/CBraMod)，MIT；[权重](https://huggingface.co/weighting666/CBraMod)

**读到的方法。** 以通道×时间片为基本单元，结合时域卷积和频域 FFT 嵌入、非对称条件位置编码，以及分开的空间/时间注意力；通过掩码重建在 TUH 数据上预训练，再迁移到多种 EEG 下游任务。

**借鉴。** 用作通用预训练强基线，与 MI 专用 MIRepNet 对照。**边界。** 其贡献是表征与预训练，不是少试次会话校准的风险控制。必须遵循权重所需单位、通道标识和 patch 契约；把原始 11 通道矩阵强行填入模型、再把性能下降当成基础模型无用是不公平的。更大骨干不自动保证患者或少标签场景更优。

阅读定位：正式 PDF 的架构、patch/位置编码、预训练目标与下游设置。

## M05. BTTA-DG：2026 ICLR 的在线无标签适应近邻

**Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding**，Huan Luo、Na Lu、Xiaopeng Wang、Xu Niu，ICLR 2026。

- [正式论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/36eeec256f89eba9ed60cbca73ea92ed-Paper-Conference.pdf)
- [作者代码](https://github.com/luo-huan-123/BTTA-DG)，MIT

**读到的方法。** 冻结 SincAdaptNet，用每试次的时序类别概率拟合 Dirichlet 特征；历史伪标签记忆库按类拟合 GMM，将特征似然与原预测先验组合。先预测当前试次，再按置信度/熵门槛加入有限记忆库并更新密度；不对编码器做梯度微调。

**借鉴。** 把“先预测后更新”的因果顺序作为比较契约，保留一个强无标签适应参照。**边界。** 置信度门槛不能保证伪标签正确；Dirichlet 中的独立时点是建模假设。该算法依赖时序 logits，并非对所有只输出 pooled trial embedding 的模型都可直接插入。它与有 20 个真实标签的校准属不同监督成本设置，应明确分组。

阅读定位：SincAdaptNet、Dirichlet 投影、GMM 记忆库和在线算法段。

## M06. Wimpff：纵向连续微调已经有直接先例

**Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study**，Martin Wimpff、Bruno Aristimunha、Sylvain Chevallier、Bin Yang，IEEE EMBC 2025。DOI: 10.1109/EMBC58623.2025.11253543。

- [作者论文 PDF](https://arxiv.org/pdf/2502.06828)
- [作者代码](https://github.com/martinwimpff/eeg-continual)，MIT

**读到的方法。** 在 Stieger 的长期多会话记录上，以其他受试者第一会话预训练，再仅用当前会话之前的数据适应。比较只用上一次与汇总所有历史数据，以及每次重置基础模型与顺序继承模型；结合在线 EA、AdaBN 和实时自适应池化。

**借鉴。** 真实 session 顺序、固定 source 起点、历史量与更新策略的析因比较。**边界。** 累积更多旧会话、持续微调、EA/AdaBN 均不能称为第三篇新贡献。其窗口密集且反馈段可能早终止，不能把窗口数当独立数据规模。我们的主线应是第一次少试次校准时预先学到的空间方向，而非再次证明历史数据有帮助。

阅读定位：Methods 的 LOSO 预训练、四种微调路径及预处理/在线池化。

## M07. PCT：错误转移正则最直接的先例

**Positive-Congruent Training: Towards Regression-Free Model Updates**，Sijie Yan 等，CVPR 2021，14299–14308。DOI: 10.1109/CVPR46437.2021.01407。

- [CVPR 正式页](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html)
- [作者 PDF](https://arxiv.org/pdf/2011.09161)

**读到的方法。** 负向翻转明确指旧模型正确、新模型错误。其 focal distillation 在普通蒸馏距离外，对旧模型预测正确的样本加权；距离包含温度 KL 和 logit matching。原文也检验了仅给旧正确样本加权 CE 的朴素方法，并发现它不足以稳定解决问题。

**借鉴。** 保存成对旧/新预测、同时报告纠错与新增错误；实现 FD-LM、普通 KD、PCT+LoRA。**边界。** 本项目任何“错误转移惩罚”“保护旧正确样本”都处于这条先例之内。PCT 也不构成在新患者分布上零新增错误的保证；不能只下降 H、不报告同时损失的 G 与 BA。

阅读定位：§3 Negative Flips；§4.1/4.2、公式 7–9 与 §5.2 消融。

## M08. 2026 卒中基础模型 LoRA：普通 PEFT 路线已被覆盖

**Lightweight Adaptation of EEG Foundation Models for Stroke Motor Imagery Decoding: Domain Shift and Subject-Level Robustness**，Anh T. Nguyen、Zihua Sun、Michelle J. Johnson，arXiv:2609.00282，2026-08-31。预印本，不能写成已正式期刊发表。

- [作者预印本](https://arxiv.org/abs/2609.00282)
- [UET175 数据作者仓库](https://github.com/nmk-k66-uet/UET175)

**读到的方法。** 对 LaBraM/REVE 比较 frozen-head 和 rank-8 LoRA，使用 30 名 UET175 卒中受试者，以参与者分组 CV/LOSO、通道和时窗消融查看个体差异。原文明确直接使用数据作者发布的 accepted event timestamps，这些试次已排除噪声和不显著 ERD；不代表全部原始提示试次。

**借鉴。** 强 PEFT、个体尾部表现和健康→患者域移位。**边界。** “基础模型+LoRA+卒中”本身不足以立项。其协议不是本方案的真实前 20 试次→未来完整 run；ERD 选择段也不能当作已确认的 cue onset。数据压缩包需作者密码，当前应标 access-pending，不能列为已能直接训练。不可把该预印本的筛选后性能与本项目全事件统计直接比较。

阅读定位：III-B 接受试次时间戳；III-D LoRA；III-E/III-F 的 CV、时间窗和失败个体分析。

## M09. MUSML：元学习空间基最接近的通用方法

**Subspace Learning for Effective Meta-Learning**，Weisen Jiang、James Kwok、Yu Zhang，ICML 2022，PMLR 162:10177–10194，会议日期 2022-07-17 至 23。

- [正式论文页](https://proceedings.mlr.press/v162/jiang22b.html)
- [正式 PDF](https://proceedings.mlr.press/v162/jiang22b/jiang22b.pdf)

**读到的方法。** 外层学习多个参数子空间的基 S；每个任务固定这些基，只用 support 梯度拟合组合系数 v，形成 w=Sv。训练时对多个子空间选择做可微松弛，通过独立源任务 validation loss 更新基；测试用 support 表现选子空间。作者在特定假设下分析了复杂度、优化和近似误差。

**借鉴。** 把共享方向与目标系数拆开，明确完整 bilevel 梯度；控制维度和子空间近似误差。**边界。** “外层学基、内层学少数系数”已明确存在。本项目具体差异是 EEG CAR 的有效物理空间、无迹有界指数变换和真实纵向未来 run 目标；不能继承该文在假设模型上的理论界，不能声称首次元学习适应子空间。

阅读定位：§3.2、公式 1、Algorithm 1；§3.3 假设和界。

## M10. Meta-Transfer：冻结特征的双层迁移也不是新概念

**Meta-Transfer Learning for Few-Shot Learning**，Qianru Sun、Yaoyao Liu、Tat-Seng Chua、Bernt Schiele，CVPR 2019。DOI: 10.1109/CVPR.2019.00049。

- [正式论文页](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.html)
- [作者 PDF](https://arxiv.org/pdf/1812.02391)；[代码](https://github.com/yaoyao-liu/meta-transfer-learning)

**读到的方法。** 先以大样本监督预训练特征抽取器，之后冻结其权重；内层用任务 support 更新分类器，外层用任务 query 学 scaling/shifting 元参数。其元测试固定已学元参数，只适应任务分类器。

**借鉴。** 冻结与更新对象清楚、source support/query 分工严格；为“相同双层训练是否所有轻量适应都能获益”建立强对照。**边界。** 我们的目标 θ 也会在目标阶段学习，具体变换空间不同；但不能把冻结大模型、少参数个性化与双层训练本身包装成新算法。

阅读定位：§4.1/4.2、公式 3–5、Figure 2/3。

## M11. Confidence-gated source selection：置信区间门控的直接 EEG 近邻

**A confidence-gated source selection strategy for cross-session transfer in brain–computer interfaces**，Yiming Shen、David Degras，2026-08-28，Frontiers in Human Neuroscience。DOI: 10.3389/fnhum.2026.1895016。

- [期刊 Methods](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2026.1895016/full)
- [作者代码](https://github.com/Yiming-S/MSDA-Bench)

**读到的方法。** 对 source-target MMD 做 trial bootstrap，用区间重叠定义近邻源集合，再用近/桥接源上的代理验证选择表示、分类器与对齐配置。候选包含 CSP、log-variance、tangent space 和多种对齐。区间重叠在文中就是启发式近邻判断。

**借鉴。** 检验源选择是否解释收益，比较强经典组合。**边界。** 原设计 leave-one-session-out 可使用目标之外的未来会话和完整无标签目标数据，与严格按时间校准不同；改为因果协议必须标注为本研究改编。MMD 区间重叠不等于风险置信界，更不保证少试次适应无害。

阅读定位：§2 方法中的 MMD/置信区间、代理验证及候选空间。

## M12. EEG-Fusion：应比较增量门控收益，而非整个管线对弱基线

**EEG-Fusion: Failure-Informed Source-Free Expert Routing for Robust Motor Imagery EEG Decoding**，Abdul Basit、Saim Rehman、Muhammad Shafique，arXiv:2609.33962v1，2026-09-27。当前按预印本引用。

- [当前作者 PDF](https://arxiv.org/pdf/2609.33962)
- [当前作者 HTML](https://arxiv.org/html/2609.33962)

**读到的方法。** 先 EA 和仅归一化参数 TTA，源留出记录训练 ridge 质量预测器，以无标签熵、置信度、多样性和塌缩指标选择三个神经候选。当前全文明确为完整目标流可见的 batch-transductive 协议；协方差/生理特征模型是独立对照，不在最终路由集合中。

**借鉴。** 把 EA、TTA、路由的增量贡献拆开。当前 PDF 相对 EA+Safe-BN 的 Cho52 路由 F1 增量仅约 0.0004，主要准确率改善来自前两步；它特别提醒不能只拿完整新管线对比未对齐 EEGNet。

**边界与版本。** 本轮已打开 7 页一手 PDF，页眉标 v1/27 Sep 2026；旧 abs 检索缓存仍出现 Cho9 和不同专家构成。本文件按当前 PDF/HTML 方法，不混合旧摘要数字。正式复现必须保存 PDF SHA 与代码版本。此工作不支持“通用门控本身一定带来大准确率提升”。

阅读定位：III-A 至 III-F；V-B 的阶段贡献与摘要对强 EA+Safe-BN 的配对比较。

## 三条补充近邻：用于修正主张，不扩张首轮训练矩阵

| 论文与一手来源 | 已核实的方法关联 | 本方案必须承认的边界 |
|---|---|---|
| [Graph Adapter for Parameter-Efficient Fine-Tuning of EEG Foundation Models](https://arxiv.org/pdf/2411.16155)，作者预印本 v2，2025-02-18 | §2.1.3 在冻结 BENDR 前加入基于电极距离的两层 GNN，学习空间输入表示 | 冻结骨干前加空间适应模块已有先例；该文下游是 MDD/TUAB，不能直接当 MI 效果 |
| [EEG-Based Mental Imagery Task Adaptation via Ensemble of Weight-Decomposed Low-Rank Adapters](https://arxiv.org/pdf/2412.17818)，ICPR 2024 工作的作者稿 | §3 将 DoRA 的幅度/方向分解与多个低秩模块用于 EEG Conformer；在 speech/MI 间迁移 | EEG/MI 的 LoRA、DoRA 及适应集成均已有先例 |
| [Subspace Adaptation Prior for Few-Shot Learning](https://link.springer.com/article/10.1007/s10994-023-06393-y)，Machine Learning，DOI 10.1007/s10994-023-06393-y | 元学习每层适合调整的操作子集，以受限适应减少过拟合 | “学习哪些参数值得调整”是通用既有研究方向；本法要提出和验证具体 EEG 结构 |

Nature/Science 本刊的侵入式神经假体结果不应直接作为头皮 EEG 少试次分类的可复现模板。本轮没有补入一个已核验可直接用于本任务的 Science 本刊开放多会话头皮 EEG 方法，以免把期刊声望误当方法适配性。
