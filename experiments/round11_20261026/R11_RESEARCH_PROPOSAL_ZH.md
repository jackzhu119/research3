# R11研究提案：以前缀到未来损失的有向算子学习校准

版本日期：2026-10-26。**研究设计，尚无R11真实EEG效果结论。**下述默认FULL campaign配置和任务计数已由本轮方案确定；代码、计划及发布身份以最终生成的hash记录为准。

本轮建议研究一个具体问题：在冻结源EEGNet和合法Support-EA坐标下，前20个真实采集位置形成的标签分数，究竟应向哪个方向修正未来预测？R9已研究固定核残差GP及局部信息锚定，R10已研究固定几何上的相容函数集合和anchor-relative决策。R11应改变离线学习对象：直接用meta_train的真实prefix→future任务，学习从前缀分数到未来校准系数的**有向算子**；以固定标签数置换诱导的centered扰动作正则，并在同一个全局log-loss上界中完成凸拟合。

这不是新发明meta-learning、有限总体协方差、logistic曲率界或噪声正则。可提出的新颖性假设仅是上述对象与部署数据角色的联合构造、相应可计算适配器，以及能够推翻机制解释的严格对照。

## 1. 现有证据要求解决什么

历史证据支持“前缀拟合得更好不等于未来预测更好”，并不支持复杂几何已经超过Support-EA。R10前置审计复算了R3–R7的参与者配对主要结果，发现一些保护方法能挽救弱拟合器退化，但不能据此声称强基线优势；R4早期标作Support-EA的结果还存在实际分派为head_newton的身份问题，不能继续用它作为EA基线。R8完整主家族未显示超过EA的可靠证据。已恢复的R9 raw-GP预测表明锚定可以减轻退化，但未证明胜过EA，且EA预处理GP的证据覆盖仍不能从缺失作负面推断。

本轮新鲜度核查另指出：R9任务/上传指针的完成计数与实际统计报告新鲜度不同。`research/evidence/EVIDENCE_SUMMARY.json`所对应固定读取快照中，完成指针已到1432，而实际report仍为120任务；这不是当前运行状态声明，也不能以1432替代可配对效应。R10在同一采集快照中尚无可用的真实EEG效果证据。因此不能说R10已经失败，也不能将R11设计包装成被新结果验证的改进。

事实来源为本轮固定来源缓存及复用证据：`research/evidence/PROVENANCE.json`、`REUSED_EVIDENCE.json`、`EVIDENCE_SUMMARY.json`，以及R10审计的`evidence_r2_r7.md`、`evidence_r8_r9.md`。历史holdout、Pan、Weibo已被反复用于研发，R11中仍为探索性；新的独立确认集须在方法冻结后按既定权限与协议使用。

## 2. 三个候选及取舍

| 候选 | 所学对象和目标适配 | 优点 | 主要碰撞与失败风险 | 决策 |
|---|---|---|---|---|
| A：跨会话稳定分数协方差/经验Bayes先验 | 从重复session前缀估计稳定、会话局部和观测分量；目标用Gaussian收缩或受限logistic head | 简单，直接响应少标签的方差问题，可作强对照 | Laird–Ware、Bonilla MTGP、ALPaCA、DKT/ADKL已覆盖大部分结构；score covariance也不能无条件等同head先验；不同EEG块未必同latent mean/独立noise | 保留为后续候选，未列入本轮已实现25方法；本轮不声称已比较该学习型EB基线 |
| B：有向prefix→future损失算子＋固定计数置换正则 | 用meta_train的support score、未来score及未来特征二阶矩学习一般A；目标只算β=A g_S | 训练对象对应未来预测；同一目标可闭式或数值凸拟合；在线只有分数计算和矩阵乘法 | 接近Meta-Curvature、learned optimizer和marginalized corruption；全局界可能太松；d²参数可能meta-overfit；配对分布可能不匹配部署 | 推荐主候选，必须通过等容量/等数据消融 |
| C：非Gaussian有限状态的因果校准 | 学稳定类型、瞬态状态及转移；前缀更新状态，未来只传播状态 | 与静态GP和线性head有明确结构差别 | HMM/mixture-of-experts已有；仅少量前缀不能识别未来run转移；需额外长记录与时间元数据契约，模型选择成本高 | 暂缓，不以“非Gaussian”本身声称新方法 |

选择B不是因为另两项不实用，而是B能把“前缀方向是否对未来有益”定义成可直接训练和证伪的对象；同时允许将几何学习、置换正则与求解方式分开检验。A和C各自的概率假设不会被悄悄移植成B的理论保证。

## 3. 数据角色和目标端边界

源EEGNet、分类器、BN/dropout、训练参考白化规则及Support-EA算法保持冻结。预算B最多为20个**原始acquired位置/UID**；无效位置仍付费，不补位。窗口、重采样或多个随机种子不产生新的独立标签。单类/空有效support采用预先固定的anchor回退并记录，不通过类平衡补采或事后删除episode修饰结果。

训练episode由前缀S和其合法未来Q组成。只用S的输入拟合`EA_S`，将同一个`EA_S`用于S和Q，并用同一冻结source产生特征和anchor logits。若Q使用自己的EA或batch统计，训练目标就不再对应部署，不能默认为同一方法。

- **meta_train**：允许使用S和Q标签离线学习A及训练端矩；Q的未来角色必须与部署定义匹配，并锁定原UID和query签名。
- **meta_validation**：按预先定义的选择规则选择配置并冻结；选择使用的Q标签必须明确记账。
- **目标评价episode**：拟合仅接收已经采集的S输入、标签、源logits和UID。Q标签仅交给隔离的评价器；Q输入不更新EA、白化、归一化、A、超参数或任何batch状态。

目标未来Q集合沿用共同的冻结B20 query定义，不能不同方法或不同预算各自换Q。跨session prefix→prefix可以作为机制对照，但不得冒充选定的prefix→future任务。所有训练聚合先对participant等权，再在其合法episode内部按冻结权重聚合；同participant的多个session/pair不是独立受试者样本。

## 4. 候选B的精确定义

### 4.1 分数与有向适配

在冻结坐标中记`φ_i∈R^d`。数值anchor先由源二元logits之差或源margin得到`m_raw`，再定义`b_ε=log((1−ε)/ε)`、`m0=clip(m_raw,−b_ε,b_ε)`、`p_i=σ(m0_i)`；核心默认`ε=10⁻¹²`。**源margin裁剪发生在加入`A g_S`之前**，support score、future训练目标和目标推理均使用这个同一数值anchor。两类均出现的有效support有n=n1+n0个位置，定义balanced loss权重

`a_i = 1/(2 n_{y_i})`，`g_S = Σ_{i∈S} a_i φ_i(y_i−p_i)`。

于是`Σa_i=1`。这是平均balanced NLL的score；它与以n倍缩放的权重不是同一参数尺度。学得一般实矩阵A后，目标校准参数为

`β_S = A g_S`，`p_A(x) = σ(m0(x)+φ(x)ᵀβ_S)`。

A允许非对称，不作GP posterior或PSD covariance解释。非对称表达能力不同于同坐标PSD收缩，但不能声称所有multi-task GP、role-dependent kernel或learned optimizer都无法表示它。

### 4.2 精确固定计数置换协方差

条件于**固定的support features、source probabilities及n1/n0**，令π在所有保持n1个正标签、n0个负标签的分配上均匀取值。令gπ按同样balanced score公式计算，并定义

`h_i = φ_i[(1−p_i)/(2n1) + p_i/(2n0)]`，`h̄=(1/n)Σh_i`。

则组合随机化给出

`N_S = Cov_π(gπ) = [n1 n0/(n(n−1))] Σ_i (h_i−h̄)(h_i−h̄)ᵀ`。

这里“精确”指对预先定义的人工置换分布精确求矩，不指真实EEG标签可交换或真实生理噪声被识别。公式无需trial iid或稳定latent参数假设；但这个局部组合恒等式也不证明未来泛化。bias为常数时，相关centered方向可退化，这是公式本身的结构。

必须使用**centered perturbation**：`ξπ=gπ−Eπ[gπ]`，再取`g̃=g_S+√η ξπ`。一般`Eπ[gπ]≠0`，所以`g_Sg_Sᵀ+ηN_S`不能冒称原始shuffled-label score的二阶矩。

还有一项必须公开的严格退化：若`n1=n0=n/2`，则`h_i=φ_i/n`，故

`N_S = S_φ/(4n)`，`S_φ=Σ_i(φ_i−φ̄)(φ_i−φ̄)ᵀ/(n−1)`。

此时N与p_i及实际标签排列无关，完全等于缩放的中心化feature covariance；不能称它挖掘了新的label correctness或reliability信息。该等价式须有数值检查和匹配control，而不是以不同方法名重复计入优势。

### 4.3 同一logistic全局majorizer中的训练目标

二元log-loss满足`ℓ''(m)=σ(m)(1−σ(m))≤1/4`。Q按固定合法权重定义

`r_Q = Σ_{j∈Q} b_j φ_j(y_j−p_j)`，`M_Q = Σ_{j∈Q} b_j φ_jφ_jᵀ`，`Σb_j=1`。

因此对任意β均有全局上界

`L_Q(β)−L_Q(0) ≤ −r_Qᵀβ + (1/8)βᵀM_Qβ`。

这是经典smoothness/MM算术，不是R11新定理。将`β=A(g_S+√η ξπ)`代入并对π精确取期望，得到拟训练目标

`J(A)= Ê[−r_Qᵀ A g_S + (1/8) tr{M_Q A(g_Sg_Sᵀ+ηN_S)Aᵀ}] + (λ/2)‖A‖_F²`，

其中η≥0、λ>0，Ê为冻结的participant等权meta_train经验聚合。N控制指定人工扰动方向；r_Q和M_Q则保留future角色。目标不是把稳定score covariance直接当参数先验，也不在目标episode重新求GP或profile集合。

### 4.4 凸拟合、可检查性质及其范围

记`C_S=g_Sg_Sᵀ+ηN_S`、`a=vec(A)`、`b=vec(Ê[r_Qg_Sᵀ])`、`H=(1/4)Ê[C_S⊗M_Q]`。实数算术中H为PSD，因λ>0，

`â=(H+λI)^{-1}b`。

该系数统一采用`(λ/2)‖A‖²`口径，不与`λ‖A‖²`混用。大维度不必显式建立d²×d²矩阵，可用与同一目标等价的matrix-free求解。核心求解器默认最多5000步；未达收敛但存在已检查的最佳有限可行迭代时接受该迭代并如实记录partial/convergence状态，不因达到迭代预算而无声回退零算子。

零算子始终可行。在精确最优和正确实现下，训练经验majorizer加非负ridge不高于零；由N项非负，可得到训练episode平均未扰动future loss相对其anchor的相应上界。有限求解时应检查实际目标与残差，仅对实际检查到的数值结论负责。这是**训练目标的确定性性质**，不是每个episode都不下降，不是新participant/未来会话安全保证，也不是BA保证。

数值实现先按§4.1裁剪源margin，再以稳定softplus/logit loss表达`m0+φᵀA g_S`上的光滑目标；这不只是报告时裁剪概率。最终sigmoid概率再应用共同的ε floor时，须将其报告口径与该光滑loss目标区分；不把适配后margin重新裁剪当成同一数学更新。浮点线性求解检查不冒充形式化IEEE区间证明。证书属于离线训练可审查结果，不作为每个query再次执行的重型门控。

## 5. 最接近的真实先例与有限差异

本轮原文及SHA记录见`research/literature/permutation_noise_priorart.json`；更广的20篇核查见`verified_paper_reviews.json`。以下5篇原文均已实际获取并阅读。

| 原始文献 | 已覆盖内容 | R11可保留的有限差异 |
|---|---|---|
| [Park与Oliva，Meta-Curvature](https://arxiv.org/abs/1902.03356)，2019 | 通过meta_train held-out表现学习梯度变换和适配几何 | 本提案的指定score随机化、future-role矩及统一凸目标；general A本身不是新原则 |
| [Bishop，Training with Noise is Equivalent to Tikhonov Regularization](https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/)，1995，DOI10.1162/neco.1995.7.1.108 | 训练扰动诱导显式正则；原文为小输入噪声及平方损失等语境 | R11的固定计数score扰动和logistic全局majorizer不同；不能把noise→regularizer关系称新 |
| [Wager、Wang与Liang，Dropout Training as Adaptive Regularization](https://arxiv.org/abs/1307.1493)，2013 | GLM无偏feature noising可写为adaptive regularization，联系Fisher/L2/AdaGrad | R11针对support score的人工置换并使用future M_Q；正则化原理已有 |
| [Eshratifar、Eigen与Pedram，Gradient Agreement as an Optimization Objective for Meta-Learning](https://arxiv.org/abs/1810.08178)，2018 | task-loss权重、proximal/Taylor近似、quadratic子问题闭式解、gradient-inner-product权重 | gradient agreement与quadratic closed fit都不是新原则；有限差异是指定general A、centered置换N及全局majorizer的联合目标 |
| [Freedman，On regression adjustments to experimental data](https://www.stat.berkeley.edu/~census/neyregr.pdf)，2008 | 固定组大小无放回assignment的有限总体推断；randomization不自动给出通常回归/iid假设 | 这里只支持条件随机化算术，不能据此称EEG真标签具有交换性或真实noise被估计 |

补充的[Reptile分析](https://arxiv.org/abs/1803.02999)原文§5也已明确同任务不同minibatch梯度inner-product与within-task generalization。[MCF原文](https://proceedings.mlr.press/v28/vandermaaten13.html)也已核查，边际化人工corruption而避免显式采样本身已有先例。

特别是Balanced-count情形的N退化，使“新的标签噪声建模”不适合作为主claim。最审慎的贡献表述是：**在冻结源与前缀EA的部署约束下，将有向prefix→future监督矩、固定计数centered score扰动和全局logistic majorizer合为一个可审查的凸训练目标，形成在线轻量适配算子，并通过等容量、等信息的对照检验何时该联合结构有必要。**本次有界检索未提供全球首次证明；若已有同构目标，必须缩小方法主张。

## 6. 验收对照必须回答什么

主比较必须包含正确分派、同source、同UID、同query集合的Frozen、Support-EA、低容量和全头SGD/正则头、R9代表性强控制、R10可获得的冻结版本。历史不可配对或基线身份不合格的数据不进入“胜过EA”的论断。

本轮已实现的候选B机制对照及其覆盖边界如下；具体25方法以实验计划为准：

- η=0的有向majorizer，隔离固定计数正则的净贡献；η>0不预设更好。
- identity/scalar、对称PSD、自由对称及general A，检验方向耦合是否超过常规收缩；不要把general A击败弱PSD当作击败所有GP。
- 等容量full-matrix learned-optimizer控制；在同样的prefix→future数据上直接优化**exact logistic ERM**，分别提供匹配配置和独立调参版本。这不是官方全编码器Meta-Curvature的原样复现。若只换majorizer带来速度收益，应将贡献和效果相应缩小。
- feature-covariance匹配control；balanced-count时必须与N逐数值等价。配合diagonal、无约束scalar与统一ridge控制，检查是否仅是常规容量或惩罚效应；本轮没有额外独立的isotropic-noise实验族。
- 打乱episode的prefix→future关联作为已实现负对照；换入不同meta_train session的future仍使用源前缀EA。它未必打乱participant，不要求其必然退化为零。本轮没有额外独立的within/cross-session训练家族，若关联被破坏仍不损害结果，不能主张配对关系是收益来源。
- 同一prefix EA用于support和future由软件契约测试验证。本轮没有实现独立的query自行拟合EA诊断任务；该诊断不在实际25方法中，不声称已执行或作为合法基线完成比较。

指标包含全query上的participant等权BA、balanced NLL、Brier、负迁移、适配幅度、耗时及条件数/求解残差。会话和seed先在participant内聚合；不能将窗口、session-pair或多seed当独立N。统计比较方向、主要家族、缺失处理和确认/探索边界须在新结果前冻结。

## 7. 默认FULL campaign、比较家族及发布身份

先完成公式、有限排列穷举小例、balanced退化、majorizer全局性、矩阵求解等必要CPU核查；它们验证实现与数学一致，不构成EEG性能证据。随后用meta_train学习、meta_validation冻结，再运行已获授权的数据阶段。历史cohort仍用于探索，真正独立确认须依既定数据权限与预声明方案执行。

默认FULL campaign共 **10,270项任务**，不是运行完成数：

| 组成 | 计算 | 任务数 |
|---|---|---:|
| 源/参考准备 | 3项 | 3 |
| 三类候选的训练 | 550配置×3seed×5预算 | 8,250 |
| 独立开发评价 | 550配置×3seed | 1,650 |
| EA/Frozen开发参考 | 6项 | 6 |
| 冻结选择 | 1项 | 1 |
| 匹配控制训练 | 9项×3seed×5预算 | 135 |
| 冻结后匹配评价 | 25方法×3seed×3cohort | 225 |
| **总计** | 3+8250+1650+6+1+135+225 | **10,270** |

550配置由三类独立调参家族组成：主方法440配置（5个rank×8个η×11个ridge）；无置换正则no_noise家族55配置；exact-logistic ERM家族55配置。后两者分别选择自身最优配置，不能强迫沿用主方法最优超参数来制造弱对照。实际参数值、五个预算、三个seed及各cohort身份以最终plan为准；本表不虚构尚未写入plan的参数列表。

预先定义的主要家族在B20比较主方法与六个参考的BA：正确Support-EA、SGD100、R9 raw、R9 EA、R10 profile和独立调参的ERM。跨K个cohort统一作Holm **6K**；默认K=3，即18条主要假设。匹配机制家族为9个匹配比较×BA/NLL，跨K个cohort统一作Holm **18K**；默认K=3，即54条。未取得或不可配对的参考明确缺失，不通过缩小家族挑选显著项。具体方法ID和比较方向由最终plan逐项绑定。

上述三个旧cohort均已参与历史研发，10,270项完整运行也不会使其变成独立确认。默认FULL结果仍按探索性解释，不宣称新增独立cohort或独立确认优势。

| 项目 | 状态 |
|---|---|
| 正则与线性求解口径 | `(λ/2)‖A‖²`，`H+λI`；默认5000步，记录最佳可行partial迭代 |
| 默认配置/任务规模 | 550配置；10,270项任务；25种冻结后匹配方法 |
| 主要/机制家族 | B20 BA Holm6K；匹配BA+NLL Holm18K；默认K=3 |
| 最终方法ID、参数值、版本与plan hash | 由最终生成计划锁定，不由本文臆造 |
| 数据/source/UID/query签名 | 由最终plan和实际收据绑定；旧证据身份保留 |
| 包清单、CPU验证、安装及发布身份 | 实际完成后以对应记录填写 |
| R11真实EEG效果 | 尚无；任务数不冒充完成数或成功结论 |

若majorizer过松而A几乎为零、η正则没有净收益、general A不胜过同容量exact-logistic learned optimizer，或新目标在meta_validation/独立确认失败，应保留失败结果并收缩或否定机制解释。只回到anchor、超过退化GP、或获得漂亮训练证书，均不足以宣称新的少标签校准能力。
