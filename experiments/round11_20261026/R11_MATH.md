# R11：条件置换扰动正则化的有向分数传输

本文定义已选择的数学机制，不宣称真实 EEG 性能已经验证。交叉熵的全局二次上界、线性 learned optimizer、有限总体置换协方差与 noise-injection/Tikhonov 的关系均为既有工具。研究贡献的候选是它们在固定前缀 EEG 校准中的具体问题定义与联合结构，不是上述基础对象的发明。

## 1. 数据角色与预测形式

冻结 EEGNet、原分类器、BN/dropout、训练参考白化及特征尺度。对目标前缀 S，只用已付费原始采集位置内的有效输入拟合 Support-EA_S。这个同一变换用于 S 与未来 query；query 不提供标签或更新任何统计。

先用固定 meta_train mean/whitener/eigenaxes，将冻结特征投影到 r 维 z；逐行令 b=z/max(1,‖z‖₂)。不含 bias 时 φ=b；含 bias 时 φ=[b;1]/√2，故 ‖φ‖₂≤1，最后一维为真正的固定常数。归一化在追加 bias 之前进行，不能把整行追加后的向量再按其 norm 缩放，后者会把 bias 变成输入相关量。所有 R11 matched controls 使用同一映射，原生历史基线则保留自己的原算法。

记映射后的特征为 φ∈R^d。原冻结网络的二元 raw margin 为 m_raw=logit₁−logit₀；实现先按共同数值floor ε=10⁻¹²定义

\[
L_\varepsilon=\log\frac{1-\varepsilon}{\varepsilon},\qquad
m_0=\operatorname{clip}(m_{\rm raw},-L_\varepsilon,L_\varepsilon),
\qquad p_0=\sigma(m_0).
\]

因此本文m₀是共同数值anchor的截断source margin，而非未截断的raw EEGNet margin。支持分数与未来训练目标都在这个同一固定m₀坐标中计算；实数算术下p₀就是相应floored anchor概率。在线支持两类计数为 n₁=k、n₀=n−k；两类都有且 n≥2 时，定义 balanced-mean 权重

\[
v_i=\frac1{2n_{y_i}},\qquad \sum_i v_i=1,
\quad g=\sum_{i=1}^n v_i\phi_i(y_i-p_i).
\tag{1}
\]

这是平均分数，不是 R10 的总和权重 n/(2n_class)。参数尺度、训练目标与 λ 均按 (1) 定义，不得混用。

离线已学算子 A∈R^{d×d}，在线仅计算

\[
\beta=A g,\quad \delta(q)=\phi_q^\top A g,\quad
p(q)=\sigma(m_0(q)+\delta(q)).
\tag{2}
\]

可以为固定预算 B=4/8/12/16/20 学习对应 A_B；若采用一个共享算子，训练中预算权重必须明确。是否共享由实现配置冻结，不能在目标结果出来后更改。φ 的坐标与白化也必须固定，因为 Frobenius ridge 不对任意非正交坐标变换保持不变。

空支持、单类支持、非有限状态按固定规则回退共同数值 EA anchor。g=0 时 (2) 精确返回 anchor。在线没有 GP posterior、R10 profile 求界或按 query 标签学习混合权重。

## 2. 固定类别计数的精确置换协方差

条件于前缀输入、冻结 φ_i、p_i 和类别计数 k，把正类位置集合 I 在全部大小为 k 的子集中均匀抽取。置换时，anchor、EA 和特征保持不变；只改变标签位置。定义置换分数

\[
g_\pi=\sum_i\frac{\phi_i(y_i^\pi-p_i)}{2n_{y_i^\pi}}.
\]

令

\[
c=-\sum_i\frac{p_i\phi_i}{2n_0},\qquad
h_i=\phi_i\left[\frac{1-p_i}{2n_1}+\frac{p_i}{2n_0}\right],
\quad \bar h=\frac1n\sum_i h_i.
\tag{3}
\]

则精确地有

\[
g_\pi=c+\sum_{i\in I}h_i,\qquad
\mathbb E_\pi g_\pi=c+k\bar h.
\tag{4}
\]

因此中心化置换扰动 ξ=g_π−E_πg_π 满足 Eξ=0，且

\[
\boxed{N=\operatorname{Cov}_\pi(g_\pi)
=\frac{k(n-k)}{n(n-1)}\sum_i(h_i-\bar h)(h_i-\bar h)^\top.}
\tag{5}
\]

**证明。**记 I_i=1{i∈I}。均匀不放回抽样给出 E I_i=k/n，Var(I_i)=k(n−k)/n²，且 i≠j 时 Cov(I_i,I_j)=−k(n−k)/[n²(n−1)]。将这些系数代入 Cov(ΣI_i h_i)，并使用 Σ(h_i−h̄)(h_i−h̄)ᵀ=Σh_i h_iᵀ−n h̄h̄ᵀ，即得 (5)。常数 c 不贡献协方差。

N 为 PSD、rank(N)≤min(d,n−1)。不需要枚举 binomial(n,k) 个置换，也不需要 Monte Carlo。有限 CPU 测试可在小 n 枚举全部标签位置，独立检查 (4)–(5)。

**解释边界。**N 是指定条件置换分布的精确协方差，既不是实际 EEG 生理噪声估计，也不是 iid Bernoulli 抽样方差、稳定参数协方差或未来风险置信区间。实际标签并不被假设为均匀置换产生。

### 2.1 平衡类别时的精确简化

若 n₁=n₀=n/2，则 h_i=φ_i/n，与 p_i 无关。因此

\[
\boxed{N=\frac1{4n(n-1)}\sum_i(\phi_i-\bar\phi)(\phi_i-\bar\phi)^\top.}
\tag{6}
\]

这时该机制就是一个有确定尺度的中心化特征协方差扰动，而非一种依赖 anchor 不确定性的全新量。若 φ 含常数 bias，N 在该方向上为零；固定正负计数的置换不改变相应 balanced class total。不能以 η→∞ 必然恢复 EA。

若额外 p_i=1/2 且特征已中心化，则普通固定权重 Fisher-score covariance N_F=Σφ_iφ_iᵀ/(4n²)，与 N 仅差 n/(n−1)。相应控制若匹配总尺度，二者应完全相同；这项等价性必须测试并如实报告。

## 3. meta_train 的前缀→未来目标

对每个合法 meta_train episode e=(S,Q)，S 是原 acquired prefix，Q 是固定 B20 之后的合法未来集合。只用 S 输入拟合 EA_S，同一个 EA_S 处理 S、Q。**meta_train 的 Q 标签用于离线监督是允许的；目标 holdout/外部/独立确认 Q 标签不进入训练或选参。**这两个角色不能混写为“从不读取任何 query 标签”。

Q 上的 balanced-mean 权重 w_j≥0、Σw_j=1，定义

\[
r_Q=\sum_{j\in Q}w_j\phi_j(y_j-p_j),\qquad
M_Q=\sum_{j\in Q}w_j\phi_j\phi_j^\top.
\tag{7}
\]

M_Q 为 PSD。participant 等权，其内 session 等权；多预算或重复 episode 不能增加独立参与者数。

## 4. 全局 logistic 上界与中心化扰动

对 ℓ(m,y)=softplus(m)−ym，有 ℓ''(m)=σ(m)(1−σ(m))≤1/4。因此任意 β∈R^d 均满足

\[
L_Q(\beta)-L_Q(0)
\le-r_Q^\top\beta+\frac18\beta^\top M_Q\beta.
\tag{8}
\]

这是真正的全局 smoothness 上界，不是把局部 Taylor 近似当作外界；但该上界本身是经典结果。

取一个设计扰动

\[
\widetilde g=g+\sqrt\eta\,\xi,\qquad
\xi=g_\pi-\mathbb E_\pi g_\pi,\qquad\eta\ge0.
\tag{9}
\]

条件于 S,Q，把 β=A g̃ 代入 (8)，使用 Eξ=0、Covξ=N，得到精确的期望上界

\[
\mathbb E_\pi[L_Q(A\widetilde g)-L_Q(0)]
\le-r_Q^\top Ag+
\frac18\operatorname{tr}\{M_Q A(gg^\top+\eta N)A^\top\}.
\tag{10}
\]

这是对**中心化、平移后的合成 score 扰动**的平均 loss 上界。它不等同直接使用未经中心化 g_π 的随机标签预测；也不是所有扰动的 worst-case loss 证书。预测时仍使用未扰动 Ag，而不必采样置换。

## 5. 唯一的离线凸优化问题

以冻结的 participant/session/budget 权重作平均，令

\[
U_\eta(A)=\widehat{\mathbb E}_e\left[
-r_Q^\top Ag+
\frac18\operatorname{tr}\{M_Q A(gg^\top+\eta N)A^\top\}\right],
\]

\[
\boxed{\widehat A=\arg\min_A
U_\eta(A)+\frac\lambda2\|A\|_F^2,\qquad\lambda>0.}
\tag{11}
\]

η>0 是主方法；η=0 是相同容量、相同目标的 learned-transport 对照。η 仅正则化离线学习：若在线公式仅为 β=Ag，则目标 N 不另外控制每个 episode/query 的收缩。不能称作“逐 query 估计真实噪声后适配”。

定义 S_η=ggᵀ+ηN，C=Ê[r_Qgᵀ]。导数给出正常方程

\[
\boxed{\frac14\widehat{\mathbb E}[M_Q A S_\eta]+\lambda A=C.}
\tag{12}
\]

采用列堆叠 a=vec(A)，K_η=(1/4)Ê[S_η⊗M_Q]，b=vec(C)，则

\[
\widehat a=(K_\eta+\lambda I)^{-1}b.
\tag{13}
\]

因为 K_η 为 PSD、λ>0，问题强凸且解唯一。可用小维度直接解，也可 matrix-free CG；后者计算 Ê[M_Q A S_η]/4+λA，无需存 d²×d² Kronecker 矩阵。

### 5.1 训练平均证书，不能升级成未来保证

在精确解处 bᵀâ=âᵀK_ηâ+λ‖â‖²，故

\[
\boxed{U_\eta(\widehat A)
=-\tfrac12\widehat a^\top K_\eta\widehat a
-\lambda\|\widehat a\|^2\le0.}
\tag{14}
\]

由 (10)，训练 episode 的平均中心化扰动损失不超过其 EA anchor。又因为 U₀(A)≤U_η(A)，未扰动训练预测也满足该平均上界。b≠0 时 (14) 严格为负。

这些是指定训练样本上的确定性结论，不要求 iid episode，也不把重复 session 当独立样本；但恰恰因此它们不是泛化证书。新 participant/session 上需要分布匹配与适当统计假设。不能声称每个训练 episode、每个 query 或未来真实 BA/NLL 都不劣于 EA。

有限求解可以保存实际 U_η(A)、ridge 目标及正常方程残差。只要实际 U_η(A)≤0，训练平均上界仍成立，不必伪造精确最优。浮点检查和概率 floor 的影响应另记，不声称形式化机器区间证明。

求解器达到迭代预算但已有有限且不劣于零算子的训练目标时，可以保存该实际迭代值并标注 `converged=false, accepted_feasible_iterate=true`；不能仅因未达到极严残差阈值就抹成零算子。非有限、目标劣于零或状态损坏才按明确规则回退。对 majorizer，带非负 ridge 的目标≤0足以推出 U_η≤0。

### 5.2 精确 logistic ERM 控制也有自己的扰动上界

定义

\[
J_{\rm ERM}(A)=\widehat{\mathbb E}[L_Q(Ag)-L_Q(0)]
+\frac\eta8\widehat{\mathbb E}\operatorname{tr}(M_QANA^\top)
+\frac\lambda2\|A\|_F^2.
\]

在 β=Ag 处对扰动 √ηAξ 使用同一个全局 smoothness 上界，并用 Eξ=0，可得

\[
\widehat{\mathbb E}\mathbb E_\pi[L_Q(A(g+\sqrt\eta\xi))-L_Q(0)]
\le J_{\rm ERM}(A)-\frac\lambda2\|A\|_F^2.
\]

所以实际 ERM 目标≤0也支持相应训练平均扰动结论，即使 anchor 展开的 U_η(A) 未必≤0。两种证书须分别命名，不能把带噪声/ridge 的 ERM 目标写成观测 NLL。主 majorizer 的优势候选是凸二次结构、可复用统计及求解成本，而非声称真实ERM没有任何训练上界。

## 6. 有向算子不是正定 kernel 预条件器的改名

C=Ê[r_Qgᵀ] 一般不对称，A 也不要求对称或 PSD。其学习目标是支持分数到未来有用校准参数的传输，未假设 g 就是 head 参数。

考虑理想化矩关系 E[ggᵀ]=sI₂、M_Q=mI₂、N=κI₂，其中 s,m>0 可取足够小以兼容有界特征，C=ρR，ρ非零且足够小以满足联合矩的Cauchy约束（例如要求ρ²<ms），且

\[
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad\rho\ne0.
\]

由 (12)，A*=ρR/[λ+m(s+ηκ)/4]，为非零反对称算子。对任何对称算子 P，tr(PᵀC)=0，二次项与 ridge 非负，所以最优对称（因此也包括 PSD）解只能是零。该例展示自由方向传输能严格区别于同坐标的 PSD shrinkage。它是说明算子类的理想化矩反例，不宣称实际 EEG 分布已实现这些矩。

边界：一般 learned optimizer、Meta-Curvature、带 prefix/future 角色的 multi-task GP 可以表示非对称交叉关系。此例只比较明确受限的对称/PSD 算子类，不能证明“任何 GP/kernel 都做不到”，也不是 EEG 数据已存在这种旋转结构的证据。

## 7. λ、η 的极限与不可声称的单调性

- λ→∞ 时 ‖â‖≤‖b‖/λ→0；对固定有限 g、φ_q，预测收敛共同 EA anchor。
- λ↓0 时，有限维精确矩问题的最小范数极限为 K_η†b。这里 b∈range(K_η)：若 a∈null(K_η)，则对每个正权 episode 有 M_Q^{1/2}A g=0；r_Q 属于 M_Q 的列空间，故 r_QᵀAg=0，进而 aᵀb=0。该论证依赖定义一致的 r_Q、M_Q，而非任意外加 b。
- η=0 精确恢复无置换正则的同容量传输模型。
- λ>0 固定时，η 增大使最优解的置换惩罚能量 tr Ê[M_Q A N Aᵀ] 单调不增。这可由两个 η 下的最优性不等式相加得到。
- 但每个 query 的修正幅度、A 的逐元素值、BA 或真实 NLL 不保证随 η 单调。K₀ 与置换惩罚矩阵通常不交换。
- η→∞ 只迫使可惩罚的方向消失；若 N/M_Q 有零空间，可能保留非零解，尤其平衡计数下 bias 方向不受 N 惩罚。因此该极限不等同 EA。

## 8. 必须区分的机制与负例

1. 固定类别计数置换条件化，不是真实噪声识别；n₀=n₁ 时精确退化为 (6)。
2. 学习非对称 A 本身已有 learned-optimizer 先例；必须比较相同容量的直接 logistic meta-ERM。
3. 训练 global upper bound 成立不意味着未来泛化；过大的 d² 容量可把训练配对噪声变成虚假 C。
4. η 可能过度抑制真实 signal，因为标签置换会破坏 signal；真实 meta_validation 才能判定权衡。
5. meta_train 的 prefix→future 关系若与目标 session 不同，A 可能反向迁移。
6. 若 C=0，则 A=0；总是复制 EA 不能包装成主要性能贡献。

合理的新颖性范围是具体的 fixed-count centered-permutation regularization、prefix→future 决策传输目标、轻量在线实现及它们的机制证据。finite-population covariance、global majorization、noise injection 或非对称矩阵均不应单独标为原创。

## 9. 有界性、数值与 API 对应

由于 ‖φ‖≤1，balanced-mean 权重和为1，故 ‖g‖≤1、‖r_Q‖≤1、‖M_Q‖₂≤1；置换分数同样满足 ‖g_π‖≤1，因此 tr(N)≤1。正常方程 Hessian 的谱范数至多 λ+(1+η)/4。这提供简单的离线一阶求解步长界，不需要目标 query 标签或每次启动重新求界。

实现参数 `feature_rank` 对应 r，`noise_strength` 对应 η，`ridge` 对应 λ。`train_operator` 只接受明确 `training_role=meta_train` 的 support/future 配对；`adapt` 只接受目标 support；`predict` 接受 query 特征及 anchor logits，不接受 query 标签。主方法 `operator_mode=directed, noise_type=permutation, training_loss=majorizer`；identity 控制为 A=αI、α∈R，不应偷偷限制 α≥0 而弱化 learned-step 基线。

实现诊断中的 `majorizer_value` 包含ridge，即 U_η+λ‖A‖²/2；`observed_training_erm_difference` 包含真实NLL difference、噪声惩罚及ridge；`observed_training_balanced_nll_difference` 才是未加惩罚的实际训练NLL差。这三个数不可互换，也不可把第一个直接代入 (14) 当作未加ridge的U_η。

feature-noise 控制固定为标签无关的 balanced-matched 中心化特征矩 Σ(φ_i−φ̄)(φ_i−φ̄)ᵀ/[4n(n−1)]。它在观察类别平衡时与主N完全一致，包括尺度和常数offset零方差；不得声称此时主方法多估计了一种噪声信息。类别不平衡时，该控制只是标签无关参考，不假称对应实际固定计数的置换分布。Fisher-noise 控制为 Σv_i²p_i(1−p_i)φ_iφ_iᵀ，保留其不同的条件模型和可能非零的offset方差。

特别地，若某预算的全部meta_train前缀都类别平衡，主方法与feature-noise控制的全部训练矩相同，因强凸性得到同一A。因为在线只用Ag，这会使二者在所有目标episode上相同，包括类别不平衡的目标episode。差异来自训练前缀的不平衡，而非在线读取目标N后作额外调整。

数学式 (8)–(14) 使用固定截断anchor margin m₀上的光滑实数logistic loss ℓ(m₀+δ,y)，并不对更新后的margin再次作截断后再证明Hessian上界。实际保存/评分时，最终概率另统一裁剪到[ε,1−ε]；任意这种概率裁剪对单个真实类 NLL 的正向增量最多 −log(1−ε)，约 ε。最终评分裁剪、anchor margin截断与浮点线性求解误差是三个不同问题，不构成机器级区间证明。

## 10. 与 bilinear kernel/attention 和普通凸回归的关系

在线公式可精确改写为

\[
\delta(q)=\sum_i v_i\,k_A(q,i)(y_i-p_i),\qquad
k_A(q,i)=\phi_q^\top A\phi_i.
\]

这是一个可非对称的跨角色bilinear transfer kernel，也可视为没有softmax的线性support-query交互。由于没有对目标support Gram求逆或形成GP posterior，它区别于R9的具体推断；但bilinear attention、一次learned gradient step及full-matrix meta-optimizer均是直接近邻。

将 (g⊗φ_q) 当作设计向量，真实ERM正是对vec(A)的凸logistic回归，加同一噪声二次项与ridge；它不是必须用非凸MAML来求解的模型。majorizer是这个问题的全局二次上界。新颖性检索应覆盖这些等价表述，而不能只检索新命名“PFST”。
