# R11 候选比较：学习什么可迁移，而非再次包装 GP 或保守门控

这是算法选择前的数学讨论，不是实现完成或真实 EEG 性能报告。没有运行 GPU 或提交 Slurm。旧 holdout/Pan/Weibo 已参与研发，仍只能作探索性证据。

**选择后的定位：**主线已从候选 B 演化为 prefix→该episode固定futureQ 的有向 score transport，并加入精确 fixed-class-count、中心化 permutation score covariance 的离线正则。meta_train futureQ标签因此是合法训练监督；目标query标签仍禁止用于拟合或选择。当前精确定义及实验规模以 [R11_MATH.md](../R11_MATH.md) 和 [R11_EXPERIMENT_PLAN.md](../R11_EXPERIMENT_PLAN.md) 为准。下面保留选择前的 cross-session prefix-only 讨论，避免把它误称为已实施的主训练任务。

**建议：不要把“跨会话估计协方差，然后做 Gaussian prior/ridge”直接定为显著的新算法。**它是合理、应保留的强对照，但与 empirical Bayes、mixed-effects/BLUP、GP kernel meta-learning 很接近。若希望得到与 R9/R10 更实质不同而且较简单的主算法，优先考察下面的候选 B：用独立会话前缀上的真实标签分数，学习一个可非对称、可反向的校准传输算子；训练目标使用 logistic 全局上界，离线是凸二次问题，在线只作一次支持分数计算及矩阵乘法。其新颖性仍须与 learned optimizer、Meta-SGD/Meta-Curvature 和跨域线性回归逐条比较，不能把下面的闭式解或上界本身称为原创。

## 1. 共同对象与必须保持的数据角色

冻结 Support-EA、EEGNet 编码器、原分类器以及 BN/dropout。预算 B 为原始采集位置数，取 4/8/12/16/20；无效位置付费，不补位。目标未来 query 集保持原 B20 所定义的同一集合。目标 query 不提供标签，也不更新白化、EA 或 batch statistics。

记经过冻结训练参考白化、可含 bias 的特征为 h(x)∈R^d，EA anchor margin 为 m₀(x)，p₀(x)=sigmoid(m₀(x))。在一个合法前缀块 U 上定义

\[
g_U=\sum_{i\in U}a_i h_i(y_i-p_{0i}),\qquad
M_U=\sum_{i\in U}a_i h_i h_i^\top,
\qquad a_i\ge0,\quad\sum_i a_i=1.
\]

两类均存在时，可用 a_i=1/(2n_{y_i})；这是 balanced NLL 的分数归一化。它不是 R10 的总和权重 n/(2n_class)，不能在参数尺度上混淆。空/单类支持的处理须固定；最简单是明确回退 EA，而不是事后挑掉 episode。

meta_train 的输入是每个 participant/session 的原 first20 acquired prefix。可在不同会话的这些前缀之间配对；不必打开这些会话原来的 future-query 标签。participant 等权，其内有序 session-pair 等权，不能把 S(S−1) 个配对当作 S(S−1) 个独立参与者。

**重要的 EA 配对细节：**若训练“输入前缀 U → 另一会话前缀 V 的损失”，应先只从 U 拟合 EA_U，再将同一个 EA_U 应用于 U 和 V。目标部署也是用 EA_support 处理未来 query。如果 V 用自己的标签无关 EA_V，虽然没有标签泄漏，训练的传输问题已与部署不同，必须另作消融，不能默认为同一条件风险。

## 2. 三类候选的结构差异

| 候选 | 离线所学对象 | 目标前缀校准 | 相对 R9/R10 的差别 | 主要风险与建议 |
|---|---|---|---|---|
| A：对比识别的稳定分数子空间/先验 | 跨会话稳定分数协方差、会话局部协方差、有限前缀噪声 | 稳定子空间中的低维 logistic head，或 Gaussian shrinkage | 学习跨会话结构，而不是 R9 的固定 linear/RBF 几何；没有 R10 profile | 数学上很接近 EB/BLUP、repeatability/CCA/TRCA；保留为强对照，不宜仅靠它宣称重大原创 |
| B：独立块有向损失传输算子 | 支持分数到另一会话校准系数的有向算子 A，可非对称/非 PSD | β=A g_support，p=sigmoid(m₀+hᵀβ) | 直接学“哪个支持方向会降低另一块的损失”；不是 posterior variance、近最优函数集或目标支持再拟合 GP | 推荐优先研究；需击败同容量 learned-optimizer/PSD 对照，防 meta_train 配对过拟合 |
| C：有限状态的稳定/瞬态因果校准 | 稳定 participant 状态、瞬态状态及状态转移 | 前缀标签作有限状态过滤；未来只传播状态，不读取未来标签 | 非 Gaussian 的有限状态预测混合，区别于静态 GP/静态 residual head | HMM/mixture-of-experts 已有；转移规律对 first20 以外是否可识别是大问题，不优先 |

候选之间不是换一个正则系数：A 是分层噪声与先验估计，B 是监督的跨块决策算子，C 是有时间状态的非 Gaussian 生成模型。三者均不因名称或组合而自动获得新颖性。

## 3. 候选 A：对比识别稳定分数，能识别到哪里

### 3.1 一个可证明的分解

对 participant p、session s，将 first20 内分为不重叠前缀子块 a/b，假设在同一个固定分数坐标中

\[
g_{psa}=\mu+u_p+v_{ps}+e_{psa},\qquad
g_{psb}=\mu+u_p+v_{ps}+e_{psb}.
\]

要求 u、v、e 均已中心化；不同 session 的 v 不相关；两个子块的 e 条件不相关；各成分互不相关。令其协方差为 C_u,C_v,C_e，两个子块噪声同尺度。则

\[
\begin{aligned}
C_{\rm between}
 &=\mathbb E[(g_{psa}-\mu)(g_{pta}-\mu)^\top]=C_u,\quad s\ne t,\\
C_{\rm within}
 &=\mathbb E[(g_{psa}-\mu)(g_{psb}-\mu)^\top]=C_u+C_v,\\
\tfrac12\mathbb E[(g_{psa}-g_{psb})(g_{psa}-g_{psb})^\top]
 &=C_e.
\end{aligned}
\]

所以两个对比可以区分稳定 participant 分数、会话局部变化及有限前缀噪声。半块均值 ḡ=(g_a+g_b)/2 的噪声为 C_v+C_e/2。

若目标是在线性均方损失下从 ḡ 估计稳定分数 u，则最优仿射估计是

\[
\widehat u=C_u(C_u+C_v+C_e/2)^{-1}(\bar g-\mu).
\]

这就是 LMMSE/BLUP/empirical-Bayes shrinkage；不是一个新的概率推导。有限样本 cross-covariance 可能不对称/非 PSD，PSD 投影与 ridge 属于估计正则化，不能称作观测上“确证”的稳定信息。

### 3.2 分数协方差不等于 head 先验协方差

如果真正的 residual head 是 θ，而 p(y=1|h)=sigmoid(m₀+hᵀθ)，则

\[
\mathbb E[g\mid\theta]
 =\mathbb E\{h[\sigma(m_0+h^\top\theta)-\sigma(m_0)]\}
 \approx F_0\theta,
\quad F_0=\mathbb E[p_0(1-p_0)hh^\top].
\]

因此 C_g≈F₀ C_θ F₀，只有固定/正确可逆的 Fisher 几何和局部线性条件下，才可把 C_g 经 F₀⁻¹ 去卷积解释为参数先验。直接拿 score covariance 当 head covariance 或 kernel metric，单位和估计对象均可能不对。Fisher whitening 只能使这个局部关系更清楚，不会把非线性关系变成精确恒等式。

### 3.3 直接负例与新颖性判断

- 若 session nuisance 跨天也相关，C_between 包含 nuisance，不能识别“可迁移到未来”的方向。
- 若目标有用信号主要是当前 session 特有而在未来 run 仍持续，删除 C_v 反而有害。
- 同一 EA_U 从输入前缀估计后作用于两个块，会共享变换误差；简单独立噪声分解未必成立。需说明条件化对象或把它当统计模型假设。
- 完全可以出现 C_u=0：诚实方法应退化为 EA，不能以此证明算法成功。

最接近的已知脉络包括 mixed-effects/BLUP、empirical Bayes、跨重复测量的可靠性协方差、CCA/CorrCA/TRCA、learned Gaussian prior/GP meta-learning。若最后的在线更新仍是 Gaussian prior 下的 ridge/logistic MAP，那么“采用跨会话配对”是一个有意义的估计设计，但不是足够的新算法区别。

## 4. 候选 B：有向、跨会话的损失传输

### 4.1 直接估计决策所需的量

对每个 meta_train 有序配对 (U,V)，U、V 来自同一 participant 的不同 session 前缀。只用 U 的输入拟合 EA_U；在这同一 anchor 下计算 g_U、V 的 r_V=g_V 和 M_V。选择一个冻结的算子 A，令候选校准参数

\[
\beta_U=A g_U.
\]

这里 A 的输出单位是 head 参数，不把输入分数协方差误当 head 先验。A 可非对称且可有负向作用：训练任务不是拟合支持标签本身，而是让某个支持分数在另一个块上形成合适的修正。

### 4.2 无需局部 Taylor 假装成真：logistic 的全局二次上界

二元交叉熵 ℓ(m,y)=softplus(m)−ym 满足 ℓ''(m)=p(1−p)≤1/4。因此对任意 β，**全局**有

\[
L_V(\beta)-L_V(0)
\le -r_V^\top\beta+\tfrac18\beta^\top M_V\beta.
\tag{1}
\]

这不是 R10 的相容函数区间，也不是 Laplace posterior。它是经典 logistic smoothness/MM 上界，不能把上界本身当原创。

定义 participant 等权、session-pair 等权的经验目标

\[
U(A)=\widehat{\mathbb E}_{U,V}
\left[-r_V^\top A g_U+\tfrac18(Ag_U)^\top M_V(Ag_U)\right],
\]

并解

\[
\widehat A=\arg\min_A\;U(A)+\frac\lambda2\|A\|_F^2,
\qquad\lambda>0.
\tag{2}
\]

它只使用 meta_train 已合法的前缀标签。既不在目标 query 上拟合，也不要求在目标支持上重复 Newton/profile 求解。

### 4.3 精确凸解与能证明的性质

采用列堆叠 a=vec(A)，定义

\[
b=\operatorname{vec}(\widehat{\mathbb E}[r_Vg_U^\top]),\qquad
K=\tfrac14\widehat{\mathbb E}[(g_Ug_U^\top)\otimes M_V].
\]

K 为 PSD，目标为 (1/2)aᵀKa−bᵀa+(λ/2)‖a‖²，故

\[
\widehat a=(K+\lambda I)^{-1}b.
\tag{3}
\]

有两个不同层级的结论：

1. **训练配对的确定性性质。**由最优性及 (1)，

\[
U(\widehat A)
=-\tfrac12\widehat a^\top K\widehat a-\lambda\|\widehat a\|^2\le0,
\]

从而训练配对的平均 V-block logistic loss 不高于该配对的 EA anchor；b≠0 时上界严格为负。这不需要 Gaussian 假设，也不需要独立配对的概率解释。零算子始终可行。

2. **总体/新会话性质不是自动的。**若将期望定义在正确、固定的 episode 分布上，且 K,b 是其真实矩，则同样的总体上界成立；实际用经验 K,b 时，需要 participant 独立、合适的有界性/矩条件及训练—部署分布匹配才能讨论泛化。重复 session-pair 不能创造独立样本量。不能把第1条叫作未来安全保证。

若离线使用 CG 或其他有限求解器，不必假称精确最优；直接计算实际 U(A) 和 ridge 目标。只要实际 U(A)≤0，训练平均的上界结论仍成立。数值线性求解残差和训练证书是离线小矩阵检查，不是每次启动或每个 query 的重型门控。

### 4.4 一个 PSD shrinkage 不能做到的例子

为展示表达能力而非 EEG 性能，设标准化 g 满足 E[ggᵀ]=I₂，目标二次风险的最优系数为 Rg，

\[
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\]

不受对称约束的 A=R 可以精确传输。对任意对称 PSD P，

\[
\mathbb E\|Pg-Rg\|^2=\|P-R\|_F^2
=\|P\|_F^2+\|R\|_F^2\ge2,
\]

因为对称与反对称矩阵 Frobenius 正交，最优 PSD 解是 P=0。说明“该信号该往另一方向转”与“缩小同一个方向”是不同的模型能力。

**边界必须保留：**这个反例只排除同一坐标/各向同性曲率下的对称 PSD shrinkage，不排除一般 multi-task GP、role-dependent kernel、learned optimizer 或 Meta-Curvature。带 prefix/future 角色的联合 GP 完全可能表示同样的交叉协方差。因此不能写“任何 GP/任何 kernel 都不可能做到”。

### 4.5 成本、容量与真正需要击败的基线

在线一次分数计算，加 A g 和 query head 推理。若 d 大，直接存 d²×d² 的 K 不合算；可用 matrix-free CG，或在 meta_train 冻结的低维坐标 d₀ 中求完整 A。低维选择属于容量控制，不是新颖性。固定预算可各自训练 A_B，λ/坐标规则在 meta_validation 冻结；不能查看目标 cohort 再调。

最重要的控制是：

- EA、Frozen、bias/rank4/full head-SGD、SGD100、R9 raw-RBF/EA-linear、R10 MAP/profile；
- 同一 g、同一坐标与目标的 scalar A=aI、对称 PSD A、自由对称 A、自由 A；
- 相同参数数量、直接优化 meta_train held-out-block CE 的 full-matrix learned optimizer；
- within-session 配对、跨 session 同 participant 配对、打乱 participant 配对；
- 一个不读取目标支持标签的全局 residual head，排除收益只是再次训练源头；
- 固定全局曲率上界与更激进局部曲率近似，分开标明哪一个保有 (1)。

不能把“超过弱 MAP”作为成功。若自由 A 并不胜过 PSD/同容量 learned optimizer，方向传输的创新主张应缩小；若跨 session 配对不胜过打乱配对，所谓可迁移结构没有得到支持。

### 4.6 应允许的失败

- 跨 session 关系不代表当前 session 的未来 run，训练配对问题与部署不匹配。
- g 在很少有效样本上几乎全是噪声；b 很小或估计方向不稳定。
- 全局 1/4 曲率上界过松，最佳 A 接近零，不能产生实际 BA 收益。
- 有益在线校准是强非线性的，单个线性算子表达不足。
- b 的有限样本误差与参数数 d₀² 使 meta_train 上界漂亮、meta_validation 失败。
- 目标表现改善可能来自普通 full-matrix meta-learning，而不是独立块目标；该对照不可省略。

## 5. 候选 C：非 Gaussian 有限状态的因果校准

为避免只是把 R9 换成一个 temporal GP，使用有限状态而非 Gaussian 随机过程。设稳定 participant 类型 k∈{1,…,K}，瞬态状态 z_t∈{1,…,J}，转移矩阵 P；排放为

\[
\Pr(y_t=1\mid h_t,k,z_t)
=\sigma(m_{0t}+h_t^\top\theta_{k,z_t}).
\]

meta_train 学 θ、P 和类型先验；目标只在已付费前缀上通过有限状态 forward recursion 更新 q_B(k,z_B)。对未来位置 t，先计算 q_t=q_B P^{t-B}，再输出

\[
p_t=\sum_{k,z}q_t(k,z)\sigma(m_{0t}+h_t^\top\theta_{k,z}).
\]

未来 query 标签不进入过滤，其他 query 的有无/次序也不改变某个固定 t 的结果。必须从真实 acquisition metadata 定义时间步，不能把随机 split 或 published-order 代理当成物理时间。

有限混合恒等式给出已观测前缀的经典 log-loss 结论：对于任何允许的状态路径 π，混合模型累计 log loss 不超过该路径累计 loss−log prior(π)。正确生成模型下的 posterior predictive 对条件期望 log loss 最优。这两者都是 HMM/Bayesian mixture 的既有性质，不是真实未来 EEG 的无条件保证。

该方向更大的问题是可识别性：如果只读每个 session first20，且它们全在同一物理 run，未来跨 run 的 P 无法从这些数据直接识别。可以假设 stationary trial transition 再外推，但这是假设；或明确允许使用 meta_train 的更长因果记录训练，此时需要改数据契约。不能凭少量 first20 的拟合给出长时间预测保证。HMM、sticky state model、switching experts 均是直接先例；因此不优先于 B。

## 6. 新颖性和发布边界

最需查证的近邻包括：empirical-Bayes few-shot head、mixed-effects/BLUP、learned Gaussian process priors/deep kernels；CCA/CorrCA/TRCA 的 repeatability 结构；Meta-SGD、Meta-Curvature、full-matrix learned optimizers；logistic 全局二次上界/MM；HMM/非 Gaussian 状态混合。精确文献应由独立检索矩阵补齐，本文不凭记忆编 DOI。

候选 B 最可能成立的论文贡献是一个具体问题定义与方法：**在只合法使用跨会话前缀的条件下，从支持 label score 直接识别有向校准决策，以同一可计算目标学习一个在线轻量的算子，并用严格配对控制验证何时需要非 PSD 传输。**它不是“发明全矩阵 meta-learning”或“发明 logistic majorizer”。如果检索发现与 (2)–(3) 及其数据角色相同的现成方法，必须缩小贡献或换方向。

数学段落采用实数 sigmoid 概率。共同数值 probability floor 可保留，但不能无声地把裁剪后的目标当成同一光滑函数；每个样本由数值概率裁剪造成的正向 loss 增量最多为 −log(1−ε)，约为 ε。该微小数值项与浮点求解误差应分开说明，而非声称机器级形式化证明。

软件方案若继续开发，资源按用户最新要求为4个单GPU作业、每个16CPU/64G；本讨论未启动作业。兼容性验证应在实际发布前集中完成一次适当测试；不可把大量重复启动门控、反复全量哈希或零收益的运行时保护当作算法贡献。
