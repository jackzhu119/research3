# R11统计先例与可辨识性审计

截止2026-10-26。6篇真实论文，出版元数据与实际获取SHA记录见同名JSON。付费正文未取得时明确限定核查等级。

## 1. Nan M. Laird等（1982）：Random-Effects Models for Longitudinal Data

原始来源：https://doi.org/10.2307/2529876；DOI：10.2307/2529876。核查等级：官方题录和完整PubMed摘要；JSTOR正文未取得。

纵向同一个体重复观测的依赖可以通过两层随机效应模型表示，个体参数来自总体分布。摘要明确说明 empirical Bayes、maximum likelihood 与 EM 的结合。

**与R11的碰撞：**跨受试者/会话学习总体协方差，再用小样本目标观测计算收缩更新，属于经典混合效应/经验Bayes结构；仅把随机效应改名为适配几何不足以成为新算法。

**边界：**R11若建立score-block观测模型，须明确每层随机效应以及within-session依赖；20个目标标签不能同时无约束估计总体D、会话R与任务偏移。此处未声称已读付费全文。

## 2. Edwin V. Bonilla等（2007）：Multi-task Gaussian Process Prediction

原始来源：https://proceedings.neurips.cc/paper_files/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html；DOI：出版页未提供，不虚构。核查等级：原文PDF已读。

学习输入核与自由形式PSD任务协方差Kf，并使用任务独立噪声D；预测协方差为Kf⊗Kx+D⊗I，超参数可通过边际似然梯度或EM估计。

**与R11的碰撞：**跨会话operator kernel、学习跨任务covariance、单独估计噪声并经验Bayes收缩均有直接先例。若R11只是φ(x)^T M φ(x′)并将M从meta-train拟合，这仍是学习核/多任务协方差。

**边界：**该文的无噪声block design会发生任务迁移消除（§2.3）；因此不能把covariance非零当作必有跨任务迁移。R11须展示其noise与不完全支持如何实际改变目标更新。

## 3. Olivier Ledoit等（2004）：A well-conditioned estimator for large-dimensional covariance matrices

原始来源：https://doi.org/10.1016/S0047-259X(03)00096-4；DOI：10.1016/S0047-259X(03)00096-4。核查等级：Crossref官方题录已核；作者PDF受限；机制另由Chen原文§I–II交叉核对。

将样本协方差向结构化目标收缩以改善高维小样本条件数，是已成熟的协方差估计方法。

**与R11的碰撞：**给meta-train协方差加ρI、trace-normalization、自动选收缩强度或求逆稳定化不能独立算新机制。

**边界：**原论文全文本次未取得；不把其大维渐近最优结论扩张为20个相关EEG标签下的最优分类/泛化结论。

## 4. Yilun Chen等（2010）：Shrinkage Algorithms for MMSE Covariance Estimation

原始来源：https://doi.org/10.1109/TSP.2010.2053029；DOI：10.1109/TSP.2010.2053029。核查等级：作者arXiv原文PDF已读；Crossref核出版年份DOI。

针对独立同分布Gaussian样本估计协方差，提出RBLW与oracle-approximating shrinkage (OAS)，在样本协方差与缩放identity之间选择收缩。RBLW对LW的MSE优势依赖Gaussian设定；OAS是对未知oracle的近似。

**与R11的碰撞：**R11的noise covariance whitening如果使用OAS，OAS本身是基线/数值组件；对协方差做广义特征分解也不会由此获得新的噪声可辨识性。

**边界：**采集位置自相关、异方差score、共享标签或嵌套prefix都违反简单iid设定；不能机械把trial数n=20代入并宣称OAS风险保证。

## 5. Jerome H. Friedman等（1989）：Regularized Discriminant Analysis

原始来源：https://doi.org/10.1080/01621459.1989.10478752；DOI：10.1080/01621459.1989.10478752。核查等级：Crossref官方题录已核；publisher/author全文未取得。

正则化判别分析通过对类协方差进行汇集和收缩，改善小样本判别的方差；在LDA/QDA及更强收缩之间连续调节。

**与R11的碰撞：**冻结EEG特征、由source总体协方差与目标少标签均值产生线性判别，或在noise-whitened特征上训练收缩LDA，可能只是RDA/LDA迁移变体。

**边界：**R11需要至少与source-pooled/shrinkage-LDA以及仅在20trial更新均值的版本比较，才能隔离学习适配operator的收益。原文未取得，不能声称已核对其每个定理或具体公式。

## 6. Baochen Sun等（2016）：Return of Frustratingly Easy Domain Adaptation

原始来源：https://doi.org/10.1609/aaai.v30i1.10306；DOI：10.1609/aaai.v30i1.10306。核查等级：作者arXiv原文PDF已读；Crossref核出版元数据。

CORAL最小化源变换协方差与目标协方差的Frobenius距离，给出源whitening后按目标covariance重着色的线性变换。

**与R11的碰撞：**特征whitening、second-order domain alignment、学习一个SPD线性几何以便迁移都已有非常直接先例。

**边界：**原论文利用无标签目标输入，不等于R11可使用冻结query分布；本项目必须只用实际前20trial支持输入。R11若学习的是标签score可靠性而非边缘特征协方差，应以同等输入限制的CORAL/EA为对照证明区别。

## 总判断

D(D+R/B)^(-1)score、noise-whitening+LDA、φ^TMφ learned kernel、或将它们添加source anchor，均不能独立支撑新算法主张。

若将目标固定支持协议对应的可部署score充分统计量作为单位，预先在meta_train构造不同支持块长/采集间隔/会话的paired观测，联合学习可加性与跨会话可迁移的信号/观测noise operator，并将其转移到20trial的目标后验适配，可形成一项具体研究设计；核心差异必须是对部署支持机制的观测算子建模，而非普通learned covariance。仍须与近年meta-kernel/EB文献进一步核查。

配对识别公式（本记录的简单模型推导，不冒充论文新定理）：

- s_ta = m_t + e_ta; s_tb = m_t + e_tb; E[e_ta|m_t]=E[e_tb|m_t]=0; Cov(e_ta,e_tb|m_t)=0

- Under that model Cov(s_ta,s_tb)=Cov(m_t)=D; symmetrize finite-sample cross covariance; finite-sample PSD projection is regularization, not a proof that assumptions hold.

- If equal within-block noise covariance R and equal block sizes, Cov(s_ta-s_tb)/2=R. With unequal covariances, only (R_a+R_b)/2 is identified; conditional cross-noise adds bias.

- For Gaussian scores s=m+e, m~N(mu,D), e~N(0,R/B), the posterior mean is mu+D(D+R/B)^(-1)(s-mu). This is standard Gaussian conditioning/EB shrinkage, not a new R11 theorem.

前20trial不足以在目标会话无约束分离signal/noise covariance；源端估计是转移先验。配对假设和部署支持观测机制必须进入实验消融。
