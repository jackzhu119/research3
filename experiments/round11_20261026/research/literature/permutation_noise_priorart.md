# R11新增主候选：标签置换正则与有向损失算子的先例核查

核查截止2026-10-26。该补充取代初轮literature_review.md中把稳定协方差/EB作为主推荐的意见；稳定协方差保留为候选A及强对照。下面5篇最接近原文均实际获取并阅读，完整URL/HTTP/字节/SHA见同名JSON。

## 1. Meta-Curvature（2019）

https://arxiv.org/abs/1902.03356；DOI `10.48550/arXiv.1902.03356`。原文PDF §3、式(3)，已读。SHA256 `c3427b78fc6d6ba2d01b6014acd89a81020950df35ddc88316db206dfc8d27c8`。

已有机制：meta-train中学习全局梯度变换，使适配后在held-out任务数据上泛化；对参数依赖作张量结构化。

碰撞：R11的Ag是learned optimizer/preconditioner的直接近邻。非对称矩阵或用future loss训练都不能单独声称首次；需要同容量、同数据、直接logistic ERM的full-matrix meta-optimizer对照。 差异：R11拟把固定class-count的centered score perturbation与全局logloss majorizer合为同一个凸二次目标；这与MC原文采用的训练细节不同，但尚不证明该联合构造全球未有。

## 2. Training with Noise is Equivalent to Tikhonov Regularization（1995）

https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/；DOI `10.1162/neco.1995.7.1.108`。作者原文PDF §2–3，已读。SHA256 `1fdda79bdb7b374041b325a8bac12bfe3af76104afc99db209bdfaad84154fec`。

已有机制：训练扰动与显式正则的关系已有经典推导；在小输入噪声/平方误差语境下得到广义Tikhonov项，并讨论正定化。

碰撞：不能把“期望零均值扰动产生quadratic covariance penalty”作为R11的新原理。 差异：Bishop不是固定标签数置换的score covariance，也不训练prefix→future决策算子；其小噪声渐近不能充当R11全局logistic bound的证明。

## 3. Dropout Training as Adaptive Regularization（2013）

https://arxiv.org/abs/1307.1493；DOI `10.48550/arXiv.1307.1493`。作者原文PDF §2–3、GLM noising regularizer，已读。SHA256 `13c288585fb3b285ab7ed268239ad0cf319cc3eb5db7bba4c3f74f2842285b04`。

已有机制：对无偏feature corruption积分可写为GLM经验损失加无标签adaptive regularizer；二阶近似联系Fisher加权L2与AdaGrad。

碰撞：R11的tr(M_Q A N A^T)作为由设计扰动诱导的方向相关正则，属于已有noising-as-regularization脉络；不能叫新的noise learning定理。 差异：Wager处理feature dropout等人工扰动，R11拟对support label-count-preserving score perturbation精确取covariance，并用future M_Q作role-coupled惩罚；这只构成限定的结构差异。

## 4. Gradient Agreement as an Optimization Objective for Meta-Learning（2018）

https://arxiv.org/abs/1810.08178；DOI `10.48550/arXiv.1810.08178`。作者原文PDF §2–3、式(3)–(11)，已读。SHA256 `a914268b20fa44381e36c38f3f6452ca4dd77a281f255f1b13f406b4c5bb334c`。

已有机制：对任务loss设置正/负权重，利用proximal一阶Taylor近似，得到quadratic子问题闭式解及gradient-inner-product权重，以促进任务间generalization。

碰撞：gradient agreement、由held-out/generalization目标决定更新方向、以及quadratic/proximal闭式fit均有先例；这些不能独立算R11创新。 差异：该文学习task-loss权重而非固定source下general A映射，使用一阶近似而非R11全局logistic majorizer，也不含固定class-count centered permutation covariance。R11仍需同数据、同容量的exact-logistic learned-optimizer对照。

## 5. On regression adjustments to experimental data（2008）

https://www.stat.berkeley.edu/~census/neyregr.pdf；不附未经核实的DOI。作者原文PDF §3–4和无放回抽样方差部分，已读。SHA256 `9e6d04fab99c2b99588e285f757f02fca39dbe1b21e12ef49a2cbcac4894c09e`。

已有机制：固定有限总体响应/协变量，随机性只来自固定组大小的无放回assignment；明确randomization不自动证明回归模型、iid残差或通常标准误。PDF标Advances in Applied Mathematics40(2008)180–193。

碰撞：在固定n1,n0下使用assignment indicator的负相关和有限总体covariance，是经典随机化算术；R11的N不能当新的抽样原理。 差异：这篇不是训练label-permutation regularizer的方法论文；这里仅支持设计条件随机化框架。R11的标签置换是人工reference distribution，不能借随机化文献将真实EEG标签说成可交换，也不能推出真实noise covariance。

## 主张边界

1. 人工固定class-count置换下的covariance可以是精确有限总体量，但真实EEG标签的交换性、真实噪声方差及未来风险并未因此得到识别。无需iid的仅是条件于固定features/logits/counts的组合恒等式。
2. `ggᵀ+ηN`应由`g+√η(gπ−Eπgπ)`的二阶矩解释；不能直接说它是raw shuffled-label score的二阶矩。
3. 当n1=n0=n/2，`h_i=φ_i/n`、`N=Sφ/(4n)`，其中Sφ是分母n−1的中心化sample covariance。此时N与source概率及实际标签排列无关；必须提供逐数值等价的feature-covariance对照，不能声称N估计了新label reliability。
4. logistic `ℓ''≤1/4`给1/8的全局二次majorizer是已知smoothness算术。优化general/non-symmetric A是learned-optimizer近邻，不意味着任何GP或role-dependent kernel都不能表示它。
5. R11仅可将“deployment-role匹配的prefix→future监督矩、centered固定计数score扰动与同一全局majorizer的联合凸fit”作为待验证的具体贡献；η=0、identity/scalar、PSD/symmetric、普通Meta-Curvature、same-data exact-logistic ERM必须匹配比较。

补充：van der Maaten等2013 [Learning with Marginalized Corrupted Features](https://proceedings.mlr.press/v28/vandermaaten13.html)原文也已取到并读，记录于JSON的supplementary_marginalization_reference。其期望corruption loss边际化覆盖“无需显式采样无限augmentation”的一般原则。

另一个补充：Nichol、Achiam与Schulman2018 [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)原文§5已明确分析同任务不同minibatch梯度inner-product与within-task generalization，实际获取PDF的SHA亦保留在JSON。
