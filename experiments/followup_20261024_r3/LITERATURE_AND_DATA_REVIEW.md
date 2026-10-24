# Paper3 第三轮：数学方向与外部数据查新

核验日期：2026-10-24。范围：在现有11通道、128 Hz、8–30 Hz、最前20个已采集试次的会话校准合同下，研究比V2双门、softPCT和时间尾部风险更实质的数学改进。本文件是设计证据，不包含新实验结果，也不声称穷尽所有文献。已下载4篇公开论文PDF作全文核对，并在随后获得授权后完整下载、逐文件校验及处理Weibo2014的10个原MAT（4.34 GB）。这些是数据与软件核验，没有训练模型或生成识别效果；未联系作者、未操作学校服务器，既有源模型与旧缓存未被改写。

## 1. 建议主线及真正的新颖性边界

更值得推进的主线是：**把校准写成支持集可辨识性受限的低维预测更新问题，在去除线性头的局部补偿方向后，以预测空间曲率定义空间步长，并明确控制线性化误差与旧预测变化。** 这比再增加一组条件门更可形成统一方法；成立与否必须通过同预算对照验证。

但自然梯度、Gauss–Newton、信赖域、变量投影、隐式可微凸分类器、低秩不确定性、SPD对齐都已有成熟研究。即使统一这些工具，也不能写成“首次几何适应”“首次少样本不确定性”或保证顶刊创新。可主张的候选贡献须具体落在：**EEG CAR空间生成元的解析身份切空间；消除头补偿后的低秩空间可辨识性；固定前缀预算下可检验的预测变化控制；以及实际非线性查询预测中的近似误差证据。**

最接近而V2文献表未充分覆盖的是 **SPDIM（ICLR 2025）**：其已研究EEG源数据不可用时，每个目标域仅优化一个SPD流形约束参数；并分析统计对齐在标签先验改变时可能失效。Paper3有标签前20试次、原始信号空间、之后不再适应的合同不同。不能用“目标域只有一个几何参数”作为独有贡献。[SPDIM原论文与官方会议记录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc6cd575b8340be6ee70beaed9ca4015-Abstract-Conference.html)

## 2. 逐方向的第一手依据与可执行含义

| 方向 | 经核验第一手文献 | 已有内容；对第三轮的具体约束 |
|---|---|---|
| 自然梯度 | Amari, *Natural Gradient Works Efficiently in Learning*, Neural Computation 10(2):251–276, 1998. [DOI/出版社](https://doi.org/10.1162/089976698300017746) | Fisher度量与重参数化适应不是新理论。其渐近效率不能移植为20个有时间依赖EEG试次的保证。 |
| Fisher/GGN与信赖域 | Martens, *New Insights and Perspectives on the Natural Gradient Method*, JMLR 21(146):1–76, 2020. [原文](https://jmlr.org/papers/v21/17-678.html) | 分析Fisher和广义Gauss–Newton在特定输出模型下的相等、阻尼与信赖域，亦讨论近似重参数化不变性。须说明采用的矩阵、采样分布和阻尼。 |
| 不应把梯度外积叫作Fisher | Kunstner, Hennig & Balles, *Limitations of the empirical Fisher approximation for natural gradient descent*, NeurIPS 2019. [原文](https://papers.nips.cc/paper_files/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html) | 支持标签下的梯度外积通常不同于模型期望Fisher；不能仅给二阶更新更换名词。二分类可直接精确计算输出模型Fisher/GGN。 |
| 经典信赖域 | Moré & Sorensen, *Computing a Trust Region Step*, SIAM J Sci Stat Comput 4(3):553–572, 1983. [出版社](https://doi.org/10.1137/0904038) | 小维度信赖域可精确/高精度求解；“接受比率+收缩半径”有既有依据。应比较固定步长、固定范数球和曲率信赖域，分清预条件与步长控制。 |
| 头作为可分离变量 | Golub & Pereyra, *The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate*, SIAM J Numer Anal 10(2):413–432, 1973. [出版社](https://doi.org/10.1137/0710036) | 消去线性变量/变量投影是经典技术。原论文是可分离非线性最小二乘；不能直接称岭逻辑回归存在同样闭式解。 |
| 可微少样本闭式头 | Bertinetto, Henriques, Torr & Vedaldi, *Meta-learning with differentiable closed-form solvers*, ICLR 2019. [作者页](https://www.robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/) / [PDF](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf) | 已有R2-D2岭回归及LR-D2逻辑回归适应。只把SGD头换成凸头不足以构成主要创新。 |
| 隐式可微凸头 | Lee, Maji, Ravichandran & Soatto, *Meta-Learning With Differentiable Convex Optimization*, CVPR 2019:10657–10665. [官方原文](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html) | MetaOptNet已将可微凸分类器用于少样本元学习。若不再展开10步SGD，仍需正确区分精确最优解的隐式梯度和有限迭代算法梯度。 |
| 隐式内层梯度 | Rajeswaran, Finn, Kakade & Levine, *Meta-Learning with Implicit Gradients*, NeurIPS 2019. [官方原文](https://papers.neurips.cc/paper_files/paper/2019/hash/072b030ba126b2f4b2374f342be9ed44-Abstract.html) | 隐式元梯度及降低内层展开内存是已有方法。小型凸内层更容易检查残差，但不能在未收敛时宣称精确隐式梯度。 |
| 训练后网络线性化 | Maddox et al., *Fast Adaptation with Linearized Neural Networks*, AISTATS 2021, PMLR130:2737–2745. [原文](https://proceedings.mlr.press/v130/maddox21a.html) | 已研究训练后线性化网络的快速适应和不确定性，包含分类和回归。Paper3若用身份切空间缓存，贡献需体现EEG空间结构与消去头后信息，而非“首次线性化适应”。 |
| Fisher选择低秩方向 | Feng, Lin & Kao, *Learning in the Fisher Subspace: A Guided Initialization for LoRA Fine-Tuning*, ICML 2026, PMLR306:30132–30155. [原文](https://proceedings.mlr.press/v306/feng26h.html) | 2026年已有数据感知Fisher引导的适应子空间。不能宣称首次用预测曲率选择低秩方向；其LoRA参数空间不同于Paper3信号空间。 |
| 低秩贝叶斯元学习 | Park, Almecija, Sharma & Azizan, *Tractable Uncertainty-Aware Meta-Learning*, AISTATS 2026, PMLR300:1486–1494. [原文](https://proceedings.mlr.press/v300/park26b.html) | LUMA已用线性化模型、Fisher相关低秩先验协方差、贝叶斯任务适应及OoD证据。全文明确为回归/Gaussian观测；分类Laplace近似不能声称继承其精确后验。 |
| 信息设计 | Kiefer & Wolfowitz, *The Equivalence of Two Extremum Problems*, Canadian J Math12:363–366, 1960. [出版社PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/B8B0626C11F52B0FD8C67C5D54BDDD43/S0008414X00010002a.pdf/the-equivalence-of-two-extremum-problems.pdf) | logdet/D-optimality是经典设计原则。学习字典使去除头补偿后的信息矩阵较好识别可以成为问题特定目标，但不能称新信息论。正确DOI为10.4153/CJM-1960-030-4。 |

### 2.1 推荐精确定义的曲率与局部消元

令二分类对数几率为 `a_i(θ,w)`，其中θ是r维空间系数，w是相对源头的二分类对比头参数；避免两套softmax权重同时平移带来的冗余。固定支持集权重 `q_i≥0, Σq_i=1`，并对头的权重和截距施加明确正则，保证小样本线性分离时最优解仍有限。

身份点或当前点的局部模型可写为：

```
a_i(δθ,δw) ≈ a_i0 + j_iᵀδθ + φ_iᵀδw
G = Σ_i q_i p_i(1−p_i) [j_i;φ_i][j_i;φ_i]ᵀ
H_local = G + blockdiag(Λθ, Λw)
        = [[A, B], [Bᵀ,D]]
S = A − B D⁻¹ Bᵀ
g_profile_local = gθ − B D⁻¹ gw
```

此处G是所定义支持权重下的逻辑损失GGN/模型期望输出Fisher；如果q是按真实标签平衡得到的固定权重，最好称“加权GGN度量”，不要含混成对未知目标分布的真实Fisher。S衡量的是**局部允许头补偿后的空间曲率**，不是原始信号的生理真实性或全局辨识定理。

`D≻0`时S是正则局部凸二次模型的Schur补。完整非线性目标的精确profile Hessian必须使用完整Hessian并在头最优点满足一阶条件；GGN Schur补只能称局部Gauss–Newton消元。岭逻辑回归头没有一般闭式最优解；闭式是牛顿/IRLS二次子问题，或改成平方损失后的岭回归。

若把整个网络在身份点固定线性化，并以此线性logit模型优化带正则的逻辑损失，内层确实成为凸问题；但在未来查询上再用真实矩阵指数与EEGNet推理，便存在**训练代理与实际查询预测的模型失配**。需要记录支持集及查询的线性化logit差、预测翻转、误差随ρ变化，以及对至少一个小规模精确非线性校准参考的差异；不能将凸代理的全局最优性转写成原非线性问题的最优性。

若预测变化约束依赖头，不能先把头按无约束最优解消掉再声称解了联合约束问题。应求联合凸子问题，或者使用含约束的条件profile问题。求解容差、最大迭代和回退规则须固定，不把难求解会话静默剔除。

### 2.2 比“SGD换Newton”更有研究价值的检验

1. 固定同一个字典和同一适应参数，比较SGD、联合阻尼GGN、头消元后的等价GGN解、曲率信赖域。精确算术下联合线性系统和Schur补应一致；若得到不同收益，原因应是正则/约束/近似/迭代预算，不能把纯代数重写当成额外准确率机制。
2. 区分Fisher预条件、对θ的欧氏球、预测KL球和非线性接受判据。阻尼 `λI` 会破坏一般线性重参数化不变性；正交换基下的等价性更容易严格成立。
3. 通过 `λmin(S)`、条件数和 `tr(G(G+Λ)⁻¹)` 等诊断可识别方向；这些是支持信息诊断，不是自动可靠的置信区间。若使用logdet(S)元目标，必须给无logdet以及同强度普通正则对照，并解释它可能鼓励敏感而不稳健的方向。

## 3. SPD运输与Lie几何：应明确研究对象

| 第一手来源 | 对当前设计的含义 |
|---|---|
| Zanini et al., *Transfer Learning: A Riemannian Geometry Framework With Applications to Brain-Computer Interfaces*, IEEE TBME, DOI10.1109/TBME.2017.2742541. [原始记录](https://pubmed.ncbi.nlm.nih.gov/28841546/) | EEG协方差仿射对齐已有直接先例。 |
| Rodrigues, Jutten & Congedo, *Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces*, IEEE TBME66(8):2390–2401, 2019, DOI10.1109/TBME.2018.2889705. [原始记录](https://pubmed.ncbi.nlm.nih.gov/30596565/) | 平移、尺度和旋转校正已有先例，不能仅靠对称/反对称分解宣称新机制。 |
| Yair, Ben-Chen & Talmon, *Parallel Transport on the Cone Manifold of SPD Matrices for Domain Adaptation*, IEEE TSP67(7):1797–1811, 2019, DOI10.1109/TSP.2019.2894801. [作者预印本与正式DOI](https://arxiv.org/abs/1807.10479) | SPD切空间运输及几何分析已成熟。应分清“协方差点的运输”与“施加到原始时序上的线性变换”。 |
| Bleuzé, Mattout & Congedo, *Tangent space alignment: Transfer learning for Brain-Computer Interface*, Front Hum Neurosci16:1049985, 2022. [原文](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1049985/full) | 切空间对齐及跨通道数迁移已有方法；不等于给既有11通道EEGNet更换输入通道而无需重训。 |
| Li, Kawanabe & Kobler, *SPDIM*, ICLR 2025. [原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc6cd575b8340be6ee70beaed9ca4015-Abstract-Conference.html) | 目标域单SPD偏置、固定特征提取器/分类器、信息最大化已有EEG实例；其目标数据使用方式与固定20试次合同须分开。 |

现有 `exp(S+K)` 是一般可逆Lie群变换，作用后的协方差为 `A C Aᵀ`；它本身一般不是SPD矩阵，更不能把 `exp(S+K)` 直接称为SPD测地线。白化/CAR去掉公共模态以后空间为10维。行列式、距离或条件数结论均应在10维可逆作用上陈述，不能在11维秩亏矩阵上套用。

可行的第三轮几何研究是先从前20试次估计收缩协方差，得到确定性的基点运输，再只学习低维残差；同时给“支持运输单独”“同样运输+头”“运输+低维残差”对照。必须计入协方差估计噪声与类先验不平衡。禁止用整场未来查询信号估计目标协方差而仍称前20预算；即使不读查询标签也增加了可用信息。

## 4. 预测保护：从软惩罚到有边界的约束

- Yan et al., *Positive-Congruent Training: Towards Regression-Free Model Updates*, CVPR 2021:14299–14308. [官方原文](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html) 已直接优化减少negative flips；这就是已有PCT边界。
- Zhao et al., *ELODI: Ensemble Logit Difference Inhibition for Positive-Congruent Training*, IEEE TPAMI, 2024, DOI10.1109/TPAMI.2024.3392724. [原始记录](https://pubmed.ncbi.nlm.nih.gov/38652615/) 已研究logit差抑制与负翻转保留；不应把logit保持本身称为新颖。
- Jiang et al., *Churn Reduction via Distillation*, ICLR 2022. [作者机构原文说明](https://research.google/pubs/churn-reduction-via-distillation/) 明确建立蒸馏与显式预测churn约束的关系，因此“从惩罚改成约束”也不是独立新理论。
- Träuble et al., *Backward-Compatible Prediction Updates: A Probabilistic Approach*, NeurIPS 2021. [官方原文](https://papers.neurips.cc/paper_files/paper/2021/hash/012d9fe15b2493f21902cd55603382ec-Abstract.html) 已研究何时保留/更新预测的概率决策。无需为了保护而额外读取未来真实标签。
- Lopez-Paz & Ranzato, *Gradient Episodic Memory for Continual Learning*, NeurIPS 2017. [官方原文](https://papers.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) 将记忆损失不增加转为局部梯度不等式；支持集margin约束需承认这一类先例。亦有[作者代码原始记录](https://github.com/facebookresearch/GradientEpisodicMemory)。

Paper3最容易严格成立的是**有限支持集的预测保持性质**：对源模型在支持集判对的点保留正margin；对其判错点不强制保留。线性logit模型给线性约束，真实非线性网络需复核约束再收缩/回退。此性质只针对选定已标注支持点，不能称未来查询零负翻转。平均KL限制只控制平均分布差，不保证每个点argmax不变；很小margin仍可翻转。

若在每个未来查询输入上无标签地计算新旧margin并按固定规则回退，可保证该输入上预测不变的某些条件，但这是**查询依赖的预测选择器**，须明确描述。永远保留源argmax会同时禁止所有纠错。任何希望保留错误修复的方案都无法仅靠未知真实标签保证未来负翻转为零。

## 5. 支持不确定性：可以量化，暂不能包装为安全保证

20个前缀试次的类别不平衡、时间相关、源模型失配、协方差估计噪声是不同不确定性。V2六个证据特征只是代理量。第三轮可使用正则逻辑凸代理的局部Laplace协方差，或按训练受试者学得的低秩先验；但应称**模型内局部后验近似**，不能称经过校准的人群置信度。

LUMA 2026已给出低秩先验/任务不确定性的强先例；Bayesian MAML也早已研究少样本任务分布，而非只取一个参数点。[Yoon et al., NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/e1021d43911ca2c1845910d84f40aeae-Abstract.html)；[Finn et al., Probabilistic MAML, NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/8e2c381d4dd04f1c55093f22c59c3a08-Abstract.html)。

如要用不确定性控制步长，比继续增加门更可解释的是对梯度不确定集合进行最坏情况局部优化，例如 `gᵀδ + 0.5 δᵀHδ + κ sqrt(δᵀV_gδ)`。椭球鲁棒化及其范数形式已有经典理论：[Ben-Tal & Nemirovski, *Robust Convex Optimization*, Mathematics of Operations Research23(4):769–805,1998](https://doi.org/10.1287/moor.23.4.769)。新的工作在于V_g如何仅用前缀/元训练数据可重复估计，以及这项对EEG是否改善；20个相关试次的朴素iid bootstrap不能自动给可靠覆盖率。应保留固定κ、纯阻尼与同均值步长对照，不用留出结果挑置信阈值。

最新旁证：[Lee, Pradeepkumar & Sun, *Test-Time Adaptation for EEG Foundation Models: A Systematic Study under Real-World Distribution Shifts*, MLHC2026,PMLR340:948–984](https://proceedings.mlr.press/v340/lee26a.html) 比较TENT/SHOT/T3A，报告直接移植TTA常退步。已核对全文，其数据为TUEV、TUAB、CHB-MIT、SleepEDF和EarEEG，**不含本任务的运动想象前20有标签校准**；它支持必须报告失败和适应代价的动机，不能用于预言Paper3的准确率或认定所有梯度适应无效。

## 6. 外部数据：先排除重复与任务不匹配

本地依据：`/workspace/paper3_hpc/protocol/DATA_REGISTRY.json`、`DATA_AUDIT.md`，以及V2设计。源数据BNCI2014_001、Cho2017、Lee2019_MI、PhysionetMI不能充当独立外部；Stieger2021是元开发体系。注册表中Yang2025_2C是optional_source，Dreyer2023及Kumar2024标为exposed_exploratory，Ma2022为development，Wang2026和HSStroke2026_v6已有locked_test角色。它们是否已实际进入原运行仍应读冻结manifest与run产物，不能仅凭“没用来训练V2”重命名为未见外测。

注册表已登记但不适合当前默认新外测的项目：UET175需要作者RAR密码；SEFMEID2026为申请/DUA待批；Ding2025_Finger任务是右手不同手指而非左右手；Alawieh2025虽有公开Zenodo且可能纵向，但有TESS干预、32/64通道混合、交叉人群及现有reserve曝光；未做原schema和身份审计前不应直接套11通道合同。

### 6.1 已准备的可选外部稳健性集：Weibo2014 / Yi2014 / NEMAR nm000146

原论文：Yi et al., *Evaluation of EEG Oscillatory Patterns and Cognitive Process during Simple and Compound Limb Motor Imagery*, PLoS ONE9(12):e114853,2014。[第一手原文与原始数据可用性](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0114853)。原始Harvard Dataverse DOI为[10.7910/DVN/27306](https://doi.org/10.7910/DVN/27306)，不是Cho2017的GigaDB100295。当前主f0的四个来源和本地注册表不含Weibo/Yi2014/NEMAR nm000146；正式结果仍应以冻结来源清单、别名和原档哈希记录说明身份，不能仅凭“未登记”认证跨研究参与者绝无重合。

**10个原MAT现已全部匿名下载，并逐文件计算本地SHA-256，与固定版本发布清单完全一致。** 总计4,340,450,480 bytes，约4.34 GB。按一次只保留一个原MAT的顺序下载、处理、保存并校验缓存；原MAT随后删除，未上传原始信号。保留的新缓存有效载荷共11,019,151 bytes，约11 MB，含1,580条LR、200个B20支持位置及1,380条固定查询。9人原数据为64×1600×560、各160条LR；第6人为64×1600×500、140条LR。数值轴和标签来自原MAT完整解析，未复用BIDS事件顺序。

10人各1会话，原任务为六种运动想象加静息；公开MAT是200 Hz、8秒epoch，MI起点位于第3秒。实际原MAT中每人的发布顺序前20条LR均为10L+10R；9人的B20查询各140条，第6人为120条。**原MAT发布顺序前20条全部任务只有4–8条LR**；不能将LR子任务的20条校准写成整场只经历20次试验。早期NEMAR BIDS事件表的顺序与原MAT不一致，原数据已覆盖该旧元数据推断；真实物理采集时序并未因此得到证明。

固定clinic11为FC3, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP4，对应原MAT解析器零基列号[16,20,24,25,26,27,28,29,30,34,38]；原MAT不含独立通道名/单位变量，位置和微伏×1e−6转换依据[原MOABB解析器](https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/Weibo2014.py)，不声称从MAT中独立完成物理标定。只读取MI后原始600:900采样的1.5秒上下文，做精确CAR11、4阶双向8–30 Hz SOS、200→128 Hz polyphase重采样、0.25–1.25秒裁剪及每通道时域去均值，保存N×11×128 float32的**pre-RMS**缓存。联合RMS由既有模型流程在适应/预测时应用。对真实W01试次与现有原预处理函数比较，float64最大差为0，float32逐位一致，QC相同；软件检查还覆盖上下文之外改成NaN/Inf不影响结果、无效前缀付费及全部短预算查询身份固定。

全部MAT末尾均为80条静息；原始物理run身份、区段顺序及真实跨trial时钟仍不可据此还原。BIDS onset=0.25+8.5k是拼接坐标，未用于缓存。这里的 **published-order LR-prefix20** 只表示发布顺序LR子任务的离线稳健性协议，**不代表真实物理采集前20条，不是新的前瞻性独立确认**；所有预算4/8/12/16/20固定评估B20之后同一批查询，不补造run，不计算真实未来运行段TERM/P10。

许可来自[原Harvard数据DOI的作者登记DataCite rightsList](https://api.datacite.org/dois/10.7910/DVN/27306)，明确为CC0-1.0；[NEMAR固定版本页](https://www.nemar.org/dataset/nm000146?v=v1.0.3)亦标CC0。[CC0官方条款](https://creativecommons.org/publicdomain/zero/1.0/)允许复制、修改与分发，因此可以发布选通道/预处理后的数值缓存；这不把论文PDF或代码的单独许可改成CC0，也不表示作者认可本衍生实验。包内保存原来源、转换、SHA-256及许可证据，保留建议学术引用。

完整实际核验副本在`compatibility/weibo_all10.json`，许可证据在`compatibility/weibo_license.json`。便携gzip tar缓存使用相对manifest路径；其SHA/bytes及从解包副本经ExternalCatalog读取10人的结果记录在`compatibility/weibo_portable.json`：gzip包8,441,250 bytes，SHA-256为`253bfe3e09e19bd72a456474f426273c37e4ee20f2cb1dd38f441c4ca9a4a34b`。在禁止网络请求条件下，解包副本与既有缓存的10人数值数组、5个预算支持/查询张量和UID逐位相同。导入器仅在显式准备命令下载；工作进程只读缓存，可选数据失败不阻断核心训练。

### 6.2 第二个可选方向与排除项

Pan2023/Pan2025更接近两次访问的二分类任务：原始发布元数据确认28通道、250 Hz、左右手MI、CC0和约0.74/0.81 GB原MAT。前者14人×2访问×每次120试次；后者10人×2访问，后一次有最初30条训练后接在线反馈。原始记录分别为[10.7910/DVN/251NOW](https://doi.org/10.7910/DVN/251NOW)和[10.7910/DVN/GH74ZG](https://doi.org/10.7910/DVN/GH74ZG)。本次官方DataCite元数据匿名可取，但具体Dataverse数据文件下载在当前网络返回403反机器人页；所以不把它列作已经验证能在本环境下载执行的默认数据。可保留为第二可选协议，文件实际获取成功后才执行，失败日志写清原因，不自动转为作者邮件请求。

AlexMI排除：其16通道缺clinic11中的FC3、FC4、C5、C1、C2、C6、CP3、CP4，且任务为右手、双脚、休息，无左右手二分类。[原维护方代码与协议](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/alex_mi.py)。Guttmann-Flury2025 Eye-BCI虽有31人63会话、62EEG/1000 Hz、每会话40条LRMI及可匿名读的2026Zenodo衍生镜像，但原论文坏通道清单包括FC3，且镜像转换了Trig语义；不能在未确定同合同坏道规则、转换来源和原序号前视为干净clinic11主确认。不要因它比较新就优先于协议更明确的数据。

GigaDB100295/100542是已有Cho2017/Lee2019源库；GIN高伽马Schirrmeister2017虽公开且通道充分，但原任务为运动执行ME，不能当成相同左右手MI外测。独立下载端点、许可证、每文件预算、元数据审计与原序号解释详见同目录 `EXTERNAL_DATA_NOTES.md`；10个Weibo原MAT的URL、bytes与发布方SHA256已提取为 `WEIBO_DOWNLOAD_MANIFEST.json`。本次已完成Weibo10个原MAT的全量取数、SHA核验和数值处理；Pan/Eye-BCI仍仅核对公开元数据与少量头部，未发起作者联系。

## 7. 最小可发表证据建议

主机制只选一个统一方向：身份切空间凸校准/局部消元与可辨识性，或精确非线性曲率信赖域；避免把全部新点拼成一个无法归因的大方法。至少保留源冻结、原P/原PCT、头单独、同参数普通SGD、同正则凸头、曲率/消元/约束各自对照。比较需固定前20采集预算、同源模型、同查询集、同训练选择规则和实际校准时间。

数学性质分别标注为：精确代数恒等式、凸代理性质、局部近似性质、仅支持点约束、经验不确定性诊断。收益则以受试者配对的未来BA、负翻转/纠错及代价回答；内部留出只11人，不能靠增加会话/种子伪增独立样本数。外部新队列冻结后只作一次预定义确认；仅20条查询/会话的短数据不适合强行估计未来物理运行段尾部风险。

## 8. 全文核验产物

公开论文原文件及pdftotext在 `references/`：`martens2020`、`spdim2025`、`luma2026`、`neuroadapt2026`。这些是少量可公开取得的文献，不包含账户凭证。核验依据为论文/官方会议/出版社/作者机构页；未以博客或聚合网站作为核心技术结论来源。
