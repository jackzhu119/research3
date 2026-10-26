# R11 文献审查与原创性边界

> 选题更新：主候选已转为有向prefix→future损失算子与固定class-count、centered标签置换正则的联合凸fit；初轮第4节的稳定协方差/EB建议现为候选A/强对照。最终判断见 `literature/permutation_noise_priorart.md` 与项目根目录 `R11_RESEARCH_PROPOSAL_ZH.md`。

核查截止 **2026-10-26**。本次审查20篇真实论文，并保留本轮实际抓取的URL、HTTP状态、时间、字节数与SHA256；正文缓存只放在`/tmp/r11-lit-primary`及统计支线缓存，不将版权原文重新发布到交付目录。下列“全文已读”“完整摘要”“仅官方题录”明确分级。R10已有原文核查只作为起点，本轮新增主题做了实际公开获取。

**结论：仅把R9固定核改成learned kernel，把R10相容集合换成EB收缩，或加入noise whitening、source anchor，都不足以构成新算法。** 与R11最接近的先例是ALPaCA、DKT/ADKL、Meta-Curvature、Bonilla多任务GP及经典混合效应模型。R11值得研究的具体差异只能落到：在与部署一致的采集前缀观测单位上，区分跨会话可复现的标签score方向、会话效应及观测noise，并由这一分解约束适配operator。该结构仍是待证伪的研究设计，不能宣布全球首次或已优于EA。

## 1. 最接近的碰撞

| R11口号/结构 | 已有直接先例 | 必须保留的边界 |
|---|---|---|
| 学可迁移核或PSD feature几何 | DKL、DKT、ADKL、Bonilla MTGP | 学核不是新贡献；R9只换核属learned-kernel对照 |
| 固定特征，学权重先验后用少样本后验 | ALPaCA、Grant层次Bayes、Laird–Ware | BLUP/EB后验和Gaussian conditioning均非新推导 |
| 学一个让梯度泛化更好的更新矩阵 | Meta-Curvature | learned preconditioner不能重新命名为新原理 |
| support条件化低秩调制 | KML、ADKL | KML的kernel指卷积filter，不能与GP covariance混写 |
| 白化/二阶域对齐/小样本协方差稳定化 | CORAL、LW、OAS、RDA | feature covariance不等于label-score signal/noise covariance |
| 用配对噪声观测学习稳定信号 | Noise2Noise及重复测量模型 | 不同EEG时间块不天然同latent mean或独立noise |
| 可靠Bayesian prior与少任务不确定性 | F-PACOH、2026 BNNP | EB点估计未自动计入超参数不确定性 |
| 20标签识别每trial的真实可靠性 | Scott等label-noise identifiability | 需要额外假设；低SNR不等于cue标签错误 |

## 2. 二十篇具体审查

以下SHA指实际获取的原始响应字节；完整多来源记录见`literature/verified_paper_reviews.json`和`literature/primary_source_fetch_manifest.json`。arXiv的`10.48550/...`是仓储DOI，不冒充出版DOI。

### 1. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks（2017）

来源：https://proceedings.mlr.press/v70/finn17a.html；不虚构未提供的DOI。核查等级：官方PMLR题录和完整摘要。SHA256 `5755fd5cd75c1d67f629e808573ff69da9762e0ae2791ff4888fda7d089e8f60`。

已有机制：在任务分布上学习可经少步梯度更新获得良好泛化的初始化；内外循环以目标任务适配后的表现训练。

与R11关系：episodic训练、少标签校准和meta_train/meta_validation分工均不是R11原创。 边界：它并不提供20个相关EEG采集位置下的未来会话风险保证；不能把R11只冻结编码器与小矩阵更新当成新元学习范式。

### 2. Recasting Gradient-Based Meta-Learning as Hierarchical Bayes（2018）

来源：https://arxiv.org/abs/1801.08930；DOI `10.48550/arXiv.1801.08930`。核查等级：作者arXiv原文PDF已读。SHA256 `0148a875f243347e9cdbc18d24874c75e9af92175a7c525fb9db6dbd09e3abf1`。

已有机制：将MAML解释为层次模型的经验Bayes近似；§2.2式(2)积分任务参数、估计共享先验，§3把内循环更新与隐含先验连接，并讨论曲率/Laplace近似。

与R11关系：从meta_train学习先验协方差、目标小样本形成后验或以EB解释适配已经被覆盖。 边界：改变为固定源anchor或冻结EEG特征是应用约束；必须标清新学习目标与可识别对象，不能把EB解释重新命名为理论贡献。

### 3. Deep Kernel Learning（2016）

来源：https://proceedings.mlr.press/v51/wilson16.html；不虚构未提供的DOI。核查等级：官方PMLR题录和完整摘要。SHA256 `6fdba4efd98b1414690008b50fcfe152d5e17231ea0fea8308f4f87b47be1fdf`。

已有机制：结合深度特征映射与核方法，以边际似然学习核及表示。

与R11关系：φ(x)^T M φ(x′)、学习SPD几何或将冻结表示放入核均属于已有核学习语言。 边界：R11若只学习M后替换R9核，最合理定位是learned-kernel基线；必须区分标签score协方差和边缘feature协方差。

### 4. Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels（2020）

来源：https://arxiv.org/abs/1910.05199；DOI `10.48550/arXiv.1910.05199`。核查等级：作者arXiv原文PDF已读。SHA256 `2eb3ca8cc7573d156dfea70519b11edfc68427f5a55a47e7e0e1a943614fa1ba`。

已有机制：共享跨任务深核超参数，积分任务内GP并优化各任务边际似然；Algorithm1训练用meta_train任务S∪Q标签，测试仅对S条件化。分类§3.2采用±1 label regression，再对预测均值作sigmoid，并非精确Bernoulli GP分类。

与R11关系：跨任务学习可迁移核、Bayesian inner loop、small-support矩阵推断、query侧不重新拟合都已有直接先例。 边界：R11若以Bernoulli score/任务前缀噪声作不同观测模型，须明确这一差异；不得把Gaussian label regression与R9的logistic GP声称逐项同一。

### 5. Adaptive Deep Kernel Learning（2019）

来源：https://arxiv.org/abs/1905.12131；DOI `10.48550/arXiv.1905.12131`。核查等级：作者arXiv原文PDF已读。SHA256 `2b3a447fa28932e762d7d8f47046d51ae641db0a6e12b42ea4fd6246a043a98d`。

已有机制：用任务编码器产生任务适配的核族，并与可微kernel ridge/GP类推断结合；不是所有任务共用一个固定核。

与R11关系：若R11根据20标签support生成协方差、核参数或任务条件几何，与ADKL直接接近。 边界：配对prefix的统计识别设计可能是额外结构，但support-conditioned kernel本身不足以支撑原创。此处以2019预印本/2020版本引用，不虚构期刊发表信息。

### 6. Meta-Learning Priors for Efficient Online Bayesian Regression（2018）

来源：https://arxiv.org/abs/1807.08912；DOI `10.48550/arXiv.1807.08912`。核查等级：作者arXiv原文PDF已读。SHA256 `0c738152f15b4e7d6a6030d80ff679f4e0a98ba1371faa25dd5d9bcb1645b80d`。

已有机制：ALPaCA联合学习固定online特征、matrix-normal权重先验均值与precision；式(5)–(8)给BLR后验；§4.1–4.2以held-out conditional predictive NLL训练，并明确固定feature时就是empirical Bayes。

与R11关系：先验协方差＋冻结编码器＋小样本解析后验＋跨任务预测目标，是与R11最直接的先例之一。 边界：ALPaCA原文主要为Gaussian回归且noise covariance作为给定量；R11若学习score观测noise需给独立识别证据，不能仅说“我们同时学noise”就证明原创/可靠。

### 7. Meta-learning with differentiable closed-form solvers（2018）

来源：https://arxiv.org/abs/1805.08136；DOI `10.48550/arXiv.1805.08136`。核查等级：作者arXiv题录和完整摘要。SHA256 `6be727572b3624b631fe06cf41b08e1e3f60f962504a433582513252eaad7ffd`。

已有机制：用ridge regression及logistic regression求解器作为few-shot适配模块，误差反传过求解器，并以Woodbury利用少样本降低矩阵成本。

与R11关系：冻结特征、小头ridge/logistic求解以及n≤20的dual解法不能算新算法。 边界：R11应保留等特征、等20trial预算的ridge/logistic baseline；只比昂贵全参数fine-tune不能隔离新operator的贡献。

### 8. Meta-Learning With Differentiable Convex Optimization（2019）

来源：https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html；不虚构未提供的DOI。核查等级：官方CVF题录和完整摘要。SHA256 `224df0cc8d81c7f9517fd71c2663fc27780fcaeae93d8da3876becf1f7b099d4`。

已有机制：将可微凸优化分类器纳入元学习，以任务内凸分类问题和外层表示学习结合。

与R11关系：“内层凸、外层学几何/特征、少样本快速适配”已有成熟先例。 边界：R11的PSD约束或凸logistic inner loop仅是工具；需报告operator的数据对象、学习准则及跨session验证的实质变化。

### 9. Meta-Curvature（2019）

来源：https://arxiv.org/abs/1902.03356；DOI `10.48550/arXiv.1902.03356`。核查等级：作者arXiv原文PDF已读。SHA256 `c3427b78fc6d6ba2d01b6014acd89a81020950df35ddc88316db206dfc8d27c8`。

已有机制：学习跨任务梯度变换以提高适配泛化；§3式(3)用张量mode-product建模参数依赖，学习的curvature并非仅是当前support Hessian或Fisher。

与R11关系：“学一个适配几何让20标签gradient更可靠”是直接碰撞，不应把learned preconditioner称新原则。 边界：R11需要证明其operator是在配对观测模型下学习稳定label-score、而不是常规meta-curvature的另一种参数化。

### 10. Revisit Multimodal Meta-Learning through the Lens of Multi-Task Learning（2021）

来源：https://arxiv.org/abs/2110.14202；DOI `10.48550/arXiv.2110.14202`。核查等级：作者arXiv原文PDF已读。SHA256 `42caf4f6afee24f6c6d0817613a1ce8fde3e9326fb171b3c5b10b2480b6f08c1`。

已有机制：KML在该文指Kernel ModuLation：task embedding生成卷积权重残差调制；§5.2式(6)为W⊙(J+M)，式(8)用向量外积降参数；同时以任务间loss ratio分析正负transference。

与R11关系：support条件化调制、低秩hypernetwork、以跨任务迁移解释适配都已有先例。 边界：这里kernel是卷积滤波器，不是GP covariance kernel；不能误写为同一种核方法，也不能声称R11首次分析negative transfer。

### 11. Meta-Learning Reliable Priors in the Function Space（2021）

来源：https://arxiv.org/abs/2106.03195；DOI `10.48550/arXiv.2106.03195`。核查等级：作者arXiv原文PDF已读。SHA256 `039a6351d7e2cb97a1545abf28f1171b1da4a79b9a425fb91409e1a9fc32051e`。

已有机制：F-PACOH将先验、hyper-prior与hyper-posterior当随机过程；用有限measurement-set上的functional KL近似正则，改善meta-data稀少处的过度自信。

与R11关系：“学习可靠先验”“函数空间保守不确定性”“防止meta-prior过度自信”都已有实质先例。 边界：其PAC/过程KL条件不自动传到R11；R11的单个EB协方差估计忽略hyperparameter uncertainty，不能由后验variance直接宣称目标会话可靠。

### 12. Amortising Inference and Meta-Learning Priors in Neural Networks（2026）

来源：https://arxiv.org/abs/2602.08782；DOI `10.48550/arXiv.2602.08782`。核查等级：2026作者arXiv原文PDF已读。SHA256 `ad8f0b0f1c7005e406cbf526f30a22ea4598472404da88337f094f60b420b745`。

已有机制：通过逐dataset amortised variational inference学习BNN权重先验，将模型构造成以BNN权重为latent的neural process；讨论极少任务下学习先验以及within-task minibatching。PDF自标ICLR2026，本记录主引arXiv。

与R11关系：截至本轮日期，跨dataset学习Bayesian prior与amortised task inference仍有最新直接先例；泛称“元学习先验”已远不足新颖性。 边界：本文不是EEG配对score观测识别模型；R11可比较其思想，但不能凭简单EB更便宜就获得新原理。

### 13. Classification with Asymmetric Label Noise: Consistency and Maximal Denoising（2013）

来源：https://proceedings.mlr.press/v30/Scott13.html；不虚构未提供的DOI。核查等级：官方PMLR题录和完整摘要。SHA256 `819d91c33b0d65520414459802f6e7359f7590b587ff9b3eef94559963ec7e66`。

已有机制：给出非对称label contamination下可识别、可一致估计的假设及maximal-denoising策略；没有把标签噪声率可从任意小样本无条件推断当成事实。

与R11关系：用20个标签推断每trial“标签可靠性”与真实潜在分类误差，不是自动可识别；需额外结构。 边界：EEG cue标签通常是实验任务指令；信号低SNR、被试不执行任务、source分类错与标签误标必须区分。R11的score稳定性不能被命名为已识别的label correctness。

### 14. Random-Effects Models for Longitudinal Data（1982）

来源：https://doi.org/10.2307/2529876；DOI `10.2307/2529876`。核查等级：官方题录和完整PubMed摘要；JSTOR正文未取得。SHA256 `badd91e63a3da6a1da86cf3a08fe71096490a44fcc3fae7366bd4ca44e57b3b1`。

已有机制：纵向同一个体重复观测的依赖可以通过两层随机效应模型表示，个体参数来自总体分布。摘要明确说明 empirical Bayes、maximum likelihood 与 EM 的结合。

与R11关系：跨受试者/会话学习总体协方差，再用小样本目标观测计算收缩更新，属于经典混合效应/经验Bayes结构；仅把随机效应改名为适配几何不足以成为新算法。 边界：R11若建立score-block观测模型，须明确每层随机效应以及within-session依赖；20个目标标签不能同时无约束估计总体D、会话R与任务偏移。此处未声称已读付费全文。

### 15. Multi-task Gaussian Process Prediction（2007）

来源：https://proceedings.neurips.cc/paper_files/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html；不虚构未提供的DOI。核查等级：原文PDF已读。SHA256 `b50b8656f7e4686f37423ac78ef34cc10494e227a02721ca2dd3df813300ae6e`。

已有机制：学习输入核与自由形式PSD任务协方差Kf，并使用任务独立噪声D；预测协方差为Kf⊗Kx+D⊗I，超参数可通过边际似然梯度或EM估计。

与R11关系：跨会话operator kernel、学习跨任务covariance、单独估计噪声并经验Bayes收缩均有直接先例。若R11只是φ(x)^T M φ(x′)并将M从meta-train拟合，这仍是学习核/多任务协方差。 边界：该文的无噪声block design会发生任务迁移消除（§2.3）；因此不能把covariance非零当作必有跨任务迁移。R11须展示其noise与不完全支持如何实际改变目标更新。

### 16. A well-conditioned estimator for large-dimensional covariance matrices（2004）

来源：https://doi.org/10.1016/S0047-259X(03)00096-4；DOI `10.1016/S0047-259X(03)00096-4`。核查等级：Crossref官方题录已核；作者PDF受限；机制另由Chen原文§I–II交叉核对。SHA256 `e70389b938e885d68048ee00053ce236e41c6d3ff32b1d399223efb4af9fb29d`。

已有机制：将样本协方差向结构化目标收缩以改善高维小样本条件数，是已成熟的协方差估计方法。

与R11关系：给meta-train协方差加ρI、trace-normalization、自动选收缩强度或求逆稳定化不能独立算新机制。 边界：原论文全文本次未取得；不把其大维渐近最优结论扩张为20个相关EEG标签下的最优分类/泛化结论。

### 17. Shrinkage Algorithms for MMSE Covariance Estimation（2010）

来源：https://doi.org/10.1109/TSP.2010.2053029；DOI `10.1109/TSP.2010.2053029`。核查等级：作者arXiv原文PDF已读；Crossref核出版年份DOI。SHA256 `0a74b89eb3e63ee49b15272668185186e3d3a9a3da64a79d9d14abe912379333`。

已有机制：针对独立同分布Gaussian样本估计协方差，提出RBLW与oracle-approximating shrinkage (OAS)，在样本协方差与缩放identity之间选择收缩。RBLW对LW的MSE优势依赖Gaussian设定；OAS是对未知oracle的近似。

与R11关系：R11的noise covariance whitening如果使用OAS，OAS本身是基线/数值组件；对协方差做广义特征分解也不会由此获得新的噪声可辨识性。 边界：采集位置自相关、异方差score、共享标签或嵌套prefix都违反简单iid设定；不能机械把trial数n=20代入并宣称OAS风险保证。

### 18. Regularized Discriminant Analysis（1989）

来源：https://doi.org/10.1080/01621459.1989.10478752；DOI `10.1080/01621459.1989.10478752`。核查等级：Crossref官方题录已核；publisher/author全文未取得。SHA256 `84a86414f6fbc9d05b5fd6a7d7c9ee7115c25292952eb3aeb9e1d5dfb6721bd8`。

已有机制：正则化判别分析通过对类协方差进行汇集和收缩，改善小样本判别的方差；在LDA/QDA及更强收缩之间连续调节。

与R11关系：冻结EEG特征、由source总体协方差与目标少标签均值产生线性判别，或在noise-whitened特征上训练收缩LDA，可能只是RDA/LDA迁移变体。 边界：R11需要至少与source-pooled/shrinkage-LDA以及仅在20trial更新均值的版本比较，才能隔离学习适配operator的收益。原文未取得，不能声称已核对其每个定理或具体公式。

### 19. Return of Frustratingly Easy Domain Adaptation（2016）

来源：https://doi.org/10.1609/aaai.v30i1.10306；DOI `10.1609/aaai.v30i1.10306`。核查等级：作者arXiv原文PDF已读；Crossref核出版元数据。SHA256 `63f38d2cf1e46551afa68993e51d8a73b23b8871bbdad6242a40db46962c770f`。

已有机制：CORAL最小化源变换协方差与目标协方差的Frobenius距离，给出源whitening后按目标covariance重着色的线性变换。

与R11关系：特征whitening、second-order domain alignment、学习一个SPD线性几何以便迁移都已有非常直接先例。 边界：原论文利用无标签目标输入，不等于R11可使用冻结query分布；本项目必须只用实际前20trial支持输入。R11若学习的是标签score可靠性而非边缘特征协方差，应以同等输入限制的CORAL/EA为对照证明区别。

### 20. Noise2Noise: Learning Image Restoration without Clean Data（2018）

来源：https://proceedings.mlr.press/v80/lehtinen18a.html；不虚构未提供的DOI。核查等级：官方PMLR题录；作者原文PDF由统计支线读取。SHA256 `742dd098012e3dceec5ee6f2572431259ebad0442b7170fbaa9f4c9e10f9089c`。

已有机制：平方损失下，保持给定输入的目标条件均值可用有噪声目标学习相同回归函数；以同潜变量的配对噪声观测替代clean target具有直接先例。

与R11关系：配对meta_train block提取稳定潜信号不是首次；R11的cross-block矩协方差虽非该文估计器，也不能把“以noise识别signal”作为全新原则。 边界：两个EEG前缀块不天然是同潜信号的独立重复；类组成、serial dependence和会话漂移会改变条件均值。此文不提供R11协方差分解可识别性的定理。

## 3. 20trial可辨识性与配对观测的必要条件

令冻结源模型的trial score为`g_i = J_i^T(y_i-p0_i)`，部署前缀统计量为` s=(1/B)Σg_i`。若每个真实采集trial可产生多个窗口，B仍计**原始acquired trial/UID**，窗口不是新增独立标签。d维目标样本协方差的rank至多B−1；当d≥20时，仅20个相关trial不能无约束识别高维signal、noise和session效应。meta_train估计的协方差是转移先验，不能说它由目标20标签独立识别。

最简单重复测量模型是`s_a=m+e_a, s_b=m+e_b`。只有`E[e_a|m]=E[e_b|m]=0`且条件交叉noise为零时，`Cov(s_a,s_b)=Cov(m)`。只有两块noise covariance相同，`Cov(s_a-s_b)/2`才给该noise；不等时只得到两者和的一半。PSD投影只是有限样本正则，不会证明这些假设。

跨会话可进一步写`s_{p,s,b}=μ+u_p+v_{p,s}+e_{p,s,b}`。不同session的同participant交叉矩可识别可复现的`u_p`，同session不同非重叠块还混有`v_{p,s}`。这是已知crossed/random-effects矩分解的应用。它能成为R11的研究设计，前提是数据有足够真实重复session，并且显式建模/检验类组成、任务状态、时间趋势及跨块相关。不能把嵌套B=5/10/20前缀当条件独立复制；也不能为了让模型假设成立而偷偷将前20trial改成类平衡采样。

Gaussian近似下`μ+D(D+R/B)^{-1}(s−μ)`就是经典后验均值/BLUP。若noise存在serial dependence，`R/B`应换为实际块均值covariance，包括lag covariance项；盲目按1/B外推没有根据。将Bernoulli score近似为Gaussian是计算近似，不是20标签下的精确概率模型。

EEG cue标签是否正确执行、source预测是否错误、trial是否低SNR、以及潜在P(y|x)是否可辨识是不同问题。没有额外观测或假设，不能把score不稳定估计命名为“真标签错误概率”，也不能借此自动剔除support标签。

## 4. 候选的取舍与具体可研究机制

1. **简单EB learned covariance**：实用，必须实现为强基线；上述先例已实质覆盖，不作为论文唯一原创点。
2. **support-conditioned deep kernel/低秩调制**：与ADKL/KML/Meta-Curvature重叠，参数更多且目标20trial不足以证明几何可辨识；不优先作为R11主线。
3. **部署前缀观测驱动的可复现score operator**：本轮推荐的研究假设。先固定source anchor、score定义、合法物理trial预算及prefix/block观测协议；在meta_train的真实重复session中学习“跨session可复现部分→目标前缀观测”的transport/noise operators，以participant等权构造矩或likelihood。最终目标适配只接收本会话已采集前B≤20的features、source logits、labels和UID，不读取target query分布或标签。优先把operator定义在label score而非任意feature covariance上，使学习对象直接对应分类适配的可迁移信号。

该候选相对R9的区别是**学习哪个score方向能跨会话复现及其采集观测noise**；相对R10的区别是**改变可学习的更新几何与观测模型**，而非只在固定kernel上换更保守动作。其新意不是交叉协方差、EB或随机效应定理，而可能是针对冻结源EEG与真实前缀部署机制的联合统计对象、受约束训练目标、有限支持适配器及其系统证伪。

source anchor可以保留为安全工程基准；除非另有严格模型条件和有效外界，不能承诺未来BA/NLL不下降。新geometry并不由R10的anchor-relative模型集合保证自动保护；应分别报告几何收益和保护动作收益。

## 5. 必须能推翻主张的实验

- 等源特征、同一物理前20trial：source、Support-EA、ridge/logistic head、pooled/shrinkage-LDA、R9固定核、R9 learned-feature covariance kernel、isotropic EB、full score covariance EB。先验证新operator不是这些baseline的等价改写。
- paired-score主线对照：正确配对、participant/session打乱配对、同session与跨session配对、只feature covariance、去noise校正、只diagonal noise、忽略lag项。若打乱配对不损失，稳定跨会话信息不是收益机制。
- 逐层验收：meta_train配对矩的split stability和有效rank；meta_validation中score transport预测误差及真实适配BA/NLL；成本/条件数；冻结后只一次使用新的confirmatory participants/cohort。历史R2–R10已影响设计的cohort都保持探索性。
- 记录零更新率、每participant负迁移、20trial下适配幅度和耗时；若只因总回到anchor而表现稳定，不称新的适配能力。若增益只出现在相同session，而跨session消失，应缩小claim并拒绝“transferable geometry”表述。

有限文献检索无法证明某个组合全球首创。投稿主张应以精确operator/likelihood、与上述直接先例的公式级差异及冻结后的实证证伪为依据。本文件不含新EEG性能结论，也未读取原始EEG、使用GPU或修改学校作业。
