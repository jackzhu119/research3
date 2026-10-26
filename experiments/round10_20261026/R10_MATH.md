# R10：证据保留的查询函数校准——数学定义与可证明性质

> 本文定义一个待真实实验检验的校准算法。核方法、Rashomon/目标相容集合、Schur 补、凸对偶和 minimax regret 均有既有文献。研究贡献的候选是它们在少支持 EEG 校准中形成的单一可计算机制：保留观测支持证据增益的完整函数集合，显式允许支持之外的函数完成，再据查询的最不利 anchor-relative proper loss 决定是否修正及修正多少。这里没有真实未来 BA/NLL 保证，也没有把相容集合误称为覆盖率已校准的置信集合。

## 1. 研究对象与协议

冻结 EEGNet 编码器、分类层及 BN/dropout 状态。最大校准集为真实前20个采集位置，经合法 QC 后得到 `n≤20` 个有效 `(x_i,y_i)`；预算为4/8/12/16/20，未来 query 集固定。

对每个预算，Support-EA 只用该预算内校准输入拟合协方差变换。冻结网络给出 anchor margin `m_0(x)` 与概率

\[
p_0(x)=\sigma(m_0(x)),\qquad \sigma(t)=(1+e^{-t})^{-1}.
\]

主算法以 Support-EA 为 anchor；Frozen EEGNet 为单独强基线/anchor 消融。实现统一使用固定的二元概率数值floor（默认1e−12）定义共享数值anchor与NLL评价；所有baseline、fallback和matched消融必须同样处理。文中“严格恢复anchor”指该共享数值anchor，不声称复原未裁剪的无限精度极端概率。特征 `\phi(x)` 来自同一冻结网络。用于特征白化、尺度、核选择和超参数选择的参考几何只来自冻结的 meta_train/meta_validation 开发流程；目标 query 既不进入拟合，也不更新任何归一化统计。

校准函数只接受支持特征、支持 anchor logits、支持标签和固定配置；预测函数只接受拟合状态、单批 query 特征及 query anchor logits，接口不接受 query 标签。增加、删除或重排其他 query 不得改变一个固定 query 的输出。

这里严格区分标签角色：每个目标episode的`fit/predict`均不接受其query标签；meta_validation的开发query标签可以用于预先规定的selection/check，因此记录`development_query_labels_used=true`；正式目标query不参与R10配置选择，记录`target_query_labels_used_for_selection=false`。另一方面，R1–R9旧holdout/Pan/Weibo结果已经影响本轮研究动机与算法设计，这一历史不能被后一个字段抹去，所以旧cohorts仍全部属于探索性证据。该字段不表示研究者从未看过任何历史目标结果。

## 2. 为什么不把严格 head 商空间更新当作主创新

令空间参数为 θ，支持 head 设计矩阵为 `Z_θ∈R^{n×p}`。若 `rank(Z_θ)=n`，则任意支持 margin `u∈R^n` 都存在 head `h` 使

\[
Z_\theta h=u-m_\theta.
\]

因此，在仍满行秩的空间参数范围内，支持 logits 的任意改变都可被分类头吸收。仅凭支持标签，不能证明某个空间更新具有 head 之外的独立预测价值。

局部加权 Jacobian 记为 A，head 设计记为 Z，则

\[
S_{\rm free}=A^\top(I-ZZ^\dagger)A.
\]

若 `range(Z)=R^n`，则 `S_free=0`。有限 head ridge 可产生非零 Schur 信息，但这依赖约束/先验，不能描述为额外观测信息。R3 已有这种有限 ridge Schur 与信息半径；R7 已检验不同 head 容量。因此 R10 不将它们更名为新机制，也不要求执行新的空间参数更新。

R10 改为讨论**完整预测函数**：即使支持 margin 可拟合，某个查询的修正也可能被支持不可见的函数方向撤销。只有这种歧义不足以撤销修正时才执行它。

## 3. 核函数类、标签权重与支持目标

选定固定 PSD 核 k。主实现使用开发集冻结几何上的线性加截距残差核；RBF 只有在开发阶段预先规定并冻结后才能成为另一个候选，不可按探索 query 结果改选。

例：若开发集白化后特征为 z∈R^d，可定义

\[
k(x,x')=\tau^2\left(\frac{z(x)^\top z(x')}{d}+1\right).
\]

核/τ 是函数先验，不是新颖性本身。线性核相当于带固定有限先验的残差 head 类；MAP 的支持系数数目小于20，并不意味着原函数类不再具有支持之外的自由度。

令 `n_c=Σ_i1(y_i=c)`。两类均存在时，使用显式固定标签平衡权重

\[
w_i=\frac{n}{2n_{y_i}},\quad \sum_iw_i=n.
\]

它对应平衡交叉熵总和；两类数量相等时与普通 Bernoulli 总和一致。若实现选择平均形式 `1/(2n_{y_i})`，必须同步缩放 λ 并在配置中记录，不能把目标缩放变化冒充算法收益。空支持、少于2个有效样本或单类支持直接返回 anchor 并记录原因。

对于残差 `f∈H_k`，定义严格凸目标

\[
J(f)=\sum_{i=1}^nw_i\left[\operatorname{softplus}(m_{0i}+f(x_i))-y_i(m_{0i}+f(x_i))\right]
+\frac\lambda2\|f\|_{H_k}^2,\quad\lambda>0.
\]

数值拟合得到一个**实际可行、有限且不劣于零修正**的 `\widehat f`：

\[
\widehat J=J(\widehat f)\le J(0)=J_0,\qquad G=J_0-\widehat J\ge0.
\]

不要求把有限迭代拟合假称为精确最优。若该条件不成立、输入非有限或求解失效，返回 anchor。

## 4. 证据保留集合：不是概率置信区间

唯一的 evidence-retention 参数 `c∈[0,1]` 在开发阶段固定。定义

\[
T=\widehat J+cG=(1-c)\widehat J+cJ_0,
\qquad
\mathcal C_T=\{f:J(f)\le T\}.
\]

集合中每个函数至少保留 `(1−c)` 比例的**观测 regularized support gain**。这是一个可审计的模型集合；它没有自动的95%覆盖率、真实标签生成模型正确性或跨会话泛化含义。

对查询 q，完整可达 residual 区间为

\[
\ell_q=\inf_{f\in\mathcal C_T}f(q),\qquad
u_q=\sup_{f\in\mathcal C_T}f(q),
\]

相应概率区间为

\[
I_q=[L_q,U_q]
=[\sigma(m_0(q)+\ell_q),\sigma(m_0(q)+u_q)].
\]

集合必须允许保持支持拟合、但改变查询预测的所有函数完成；只保留 MAP 的支持张成空间会漏掉核心的不可辨识性。

## 5. 支持规模的精确有限维表示

### 5.1 基本 RKHS 分解

支持 Gram 矩阵为 `K=UΛUᵀ`，正特征值方向数 r。定义

\[
B=U\Lambda^{1/2},\quad
 a_q=\Lambda^{-1/2}U^\top k_{Sq},\quad
 s_q=k(q,q)-\|a_q\|^2\ge0.
\]

将 f 分成支持 kernel 张成空间与其正交补。端点最优解可写成

\[
f_S=Bv,\quad f(q)=a_q^\top v+\sqrt{s_q}\,t,
\quad \|f\|^2=\|v\|^2+t^2.
\]

这里 `s_q` 是未由支持评估函数确定的查询完成自由度。它是经典核插值 Schur innovation/power function，不是新的量，也不是 R9 Laplace posterior variance。新增机制是把它放入同一个支持证据相容问题，再求 anchor-relative 决策。

令

\[
F(v)=\sum_iw_i[\operatorname{softplus}(m_{0i}+(Bv)_i)
-y_i(m_{0i}+(Bv)_i)]+\frac\lambda2\|v\|^2.
\]

则查询区间来自

\[
\min/\max_{v,t}\ a_q^\top v+\sqrt{s_q}t
\quad\text{s.t.}\quad F(v)+\frac\lambda2t^2\le T.
\tag{P_q}
\]

不改变 encoder，不读取 query 标签，也不对 query 批次拟合一个新模型。

### 5.2 有限精度中 PSD 对象必须明确

不能把正的支持特征值按相对阈值丢弃后，直接把遗漏方向描述成支持的精确零空间。R9 的谱截断可作为它自身的近似，但不能不加说明地用于本节精确完成证明。

部署可使用显式声明的事件-ID nugget kernel

\[
k_\epsilon(x,x')=k(x,x')+\epsilon\,1\{\operatorname{UID}(x)=\operatorname{UID}(x')\},\quad\epsilon>0.
\]

支持 UID 唯一，未来 query UID 与支持不相交。支持矩阵用 `K_ε=BBᵀ` 的全部 Cholesky 方向；`a_q=B^{-1}k_Sq`，query diagonal为 `k(q,q)+ε`，从而联合支持–查询 Gram 对应同一个 PSD 对象。ε 是声明的有限模型 nugget，不是隐藏的数据证据；τ=0时仍显式返回 anchor，不靠 nugget单独拟合标签。

当只以数值误差判定轻微负 s 时，应使用向保守方向的明确误差膨胀；超过容差的 PSD 不一致返回 anchor并记录。不得把严重负 innovation静默截为0后宣称得到安全界。

## 6. 最终动作：相对 anchor 的 minimax proper-loss regret

二元交叉熵记为

\[
\operatorname{CE}(\eta,p)=-\eta\log p-(1-\eta)\log(1-p).
\]

对给定查询，定义

\[
p^*(q)=\arg\min_{p\in(0,1)}\sup_{\eta\in I_q}
\{\operatorname{CE}(\eta,p)-\operatorname{CE}(\eta,p_0(q))\}.
\tag{R_q}
\]

### 命题1：闭式解与模型集合内的相对损失性质

\[
p^*(q)=\operatorname{clip}(p_0(q),L_q,U_q).
\]

**证明。** 对固定 p，regret关于 η仿射。若 p>p0，斜率 `logit(p0)−logit(p)<0`，最坏值在 η=L；该分支最小值在 p=L（只在 L>p0时有必要）。若 p<p0，最坏值在 η=U，最小值在 p=U。p0位于区间内时，取η=p0可知任何偏离会产生非负最坏regret，p=p0给出0。因此得到clip形式。

若 p0<L，最坏regret为

\[
\operatorname{CE}(L,L)-\operatorname{CE}(L,p_0)
=-\operatorname{KL}(\operatorname{Ber}(L)\|\operatorname{Ber}(p_0))\le0;
\]

另一侧对称。因此对每个 `η∈I_q`，`CE(η,p*)≤CE(η,p0)`。

**边界：** 真实未来条件概率未被证明属于 `I_q`。该命题不是实际未来 NLL 改善、BA 改善、分布漂移鲁棒性或临床安全保证。最终 BA 只能由预注册实验判定。

若只求出保守外区间 `I_q⊂Ĩ_q`，对 Ĩ 的同样clip动作仍对 I 的所有成员满足相对损失性质；松界只使更新更保守。

## 7. 有意义的零更新证书与退化性质

### 命题2：支持不可见完成可以否决查询更新

令实际可行拟合状态为 `\widehat v`，`μ_q=a_qᵀ\widehat v`，且 `s_q>0`。取

\[
v=\widehat v,\qquad t=-\frac{\mu_q}{\sqrt{s_q}}.
\]

该函数保持所有支持 residual不变，并使查询 residual恰为0；它的目标只增加

\[
\frac{\lambda\mu_q^2}{2s_q}.
\]

所以只要

\[
\boxed{\frac{\lambda\mu_q^2}{2s_q}\le cG,}
\tag{Z_q}
\]

零修正在真实查询区间内，算法必须返回 anchor。这是精确充分条件，不需要把 \widehat v 假称为全局最优，不需要 posterior方差近似，也无需求两个端点。

零查询修正的精确可行性条件（s>0）为

\[
\min_v\{F(v)+\lambda(a_q^\top v)^2/(2s_q)\}\le T.
\]

廉价证书 Z 是选取已有可行 v的充分证书；未通过Z不表示一定需要修正。

### 命题3：极限与保守性单调

1. c=1时，T=J0，零函数属于集合，所有查询严格恢复 anchor。
2. c=0且\widehat f是精确极小值时，由λ强凸性，集合退化为 MAP；有限优化时应称有限目标相容集，不能强称精确 MAP极限。部署代码把 `mode=map` 或 `c=0` 明确作为直接 MAP 消融，标注 `not_profile_certified`；有限拟合时该消融不继承相容集合稳健性质，不能将其与主线 `0<c≤1` 的证书混淆。
3. 对同一固定拟合状态，c增大使集合嵌套扩大，因此**精确区间对应的精确minimax动作**满足每个查询的 `|p*_c−p0|` 单调不增。该结论不自动延伸到有限预算实现：不同c可能触发不同的自适应tilt序列、松紧不同的外界与停机路径，实际近似修正不保证普遍单调。
4. 若 `k_Sq=0`，已有支持拟合与t=0给出零查询修正，所以返回 anchor。
5. 对固定支持模型和固定 a_q，增大 s_q使每个v对应的可达查询区间扩大，因此**精确完整区间及其精确minimax动作**向anchor收缩。有限预算、自适应求界得到的近似动作不据此获得普遍单调保证。**不**可据此声称改变整体 kernel超参数也满足相同单调性。
6. s=0只表示该查询在函数类中没有额外正交完成方向；支持近最优解之间仍可存在预测歧义。

## 8. 严格凸倾斜问题与有限数值证书

令方向 ε∈{−1,+1}、ζ>0，定义

\[
v_{\epsilon\zeta}=\arg\min_v\{F(v)-\epsilon\zeta a_q^\top v\}.
\]

其 Hessian

\[
H(v)=B^\top\operatorname{diag}(w_i p_i(1-p_i))B+\lambda I\succeq\lambda I
\]

保证唯一最小值。若需要求精确端点，令

\[
D_\epsilon(\zeta)=F(v_{\epsilon\zeta})+\frac{\zeta^2s_q}{2\lambda}.
\]

因为 `∇F(v_(εζ))=εζa`，有

\[
D'_\epsilon(\zeta)=\zeta a_q^\top H(v_{\epsilon\zeta})^{-1}a_q
+\frac{\zeta s_q}{\lambda}\ge0.
\]

若T高于精确最小值、查询functional非零，可对 `Dε(ζ)=T` 做doubling bracket和二分。KKT给出端点

\[
\operatorname{endpoint}_\epsilon=a_q^\top v_{\epsilon\zeta}
+\epsilon\frac{\zeta s_q}{\lambda}.
\]

**关键：** 一个近似可行 extremizer只给出内界，不能当作保守外界。

对任意ζ>0，凸对偶给出 `max_C εf(q)` 的上界

\[
U_\epsilon(\zeta)=
\frac{T-\inf_v\{F(v)-\epsilon\zeta a_q^\top v\}}{\zeta}
+\frac{\zeta s_q}{2\lambda}.
\]

对任意实际迭代值v，令

\[
g=\nabla F(v)-\epsilon\zeta a_q,
\quad M=F(v)-\epsilon\zeta a_q^\top v-\frac{\|g\|^2}{2\lambda}.
\]

λ强凸性保证 tilted最小值≥M。因此

\[
\boxed{\widehat U_\epsilon(\zeta)=\frac{T-M}{\zeta}+
\frac{\zeta s_q}{2\lambda}\ge\max_{f\in\mathcal C_T}\epsilon f(q).}
\tag{D_q}
\]

这条梯度残差修正使有限Newton状态也能产生外界。多个ζ的最小有效上界仍是有效上界。求解预算、Newton停机阈值只影响保守程度，不允许把内界冒充外界。

### 一侧求界即可作保守决策

已有可行函数的查询值为μ。若μ=0，anchor属于集合，返回anchor。令方向 `d_q=sign(μ)`；只有所有相容函数的修正同号时，才可能作非零最优动作。因此只需求 `ε=−d_q` 的对偶上界，记

\[
e_q=-\min_{\zeta\in\mathcal Z}\widehat U_{-d_q}(\zeta)
\le\inf_{f\in\mathcal C_T}d_q f(q).
\]

取

\[
\delta_q=d_q\max(0,e_q),\qquad p(q)=\sigma(m_0(q)+\delta_q).
\]

若界紧，这是精确minimax动作；若界松，则位于anchor与真正最优端点之间，仍有命题1的模型集合内相对损失性质（以前述外界有效为条件）。这是一条逐查询、逐配置的性质，不意味着不同c或不同预算求出的近似动作必然形成嵌套/单调序列。若计算预算耗尽而未能证实 e>0，返回anchor。这减少一半端点计算，并允许证书驱动早停。

代码中方向命名 `direction`，激活函数命名 `sigmoid`。

### 有限精度边界

D_q是实数算术中的数学证书。NumPy float64残差计算、保守roundoff padding与PSD容差构成数值实现，但未采用形式化IEEE-754区间算术，因此不等于对每个输入、每次机器舍入都有效的机器级证明。有限测试检查了有代表性的数值构造，不证明所有浮点执行上的外界保守性。若界非有限、几何非PSD、梯度/目标状态矛盾、证书超出已有可行见证值且超过容差，应返回anchor并记录。不能以CPU测试声称所有GPU浮点行为已形式化验证。

## 9. 算法伪代码

```text
FIT(S, fixed_config, frozen_reference):
    校验支持身份、预算、标签、维度；若空/单类/样本不足，保存anchor fallback
    用冻结开发几何变换支持特征；构造明确kernel与UID nugget支持矩阵
    Cholesky使用全部SPD方向，得到B
    从v=0做单调线搜索Newton，保存实际可行vhat、Fhat、F0、gradient residual
    若Fhat非有限或Fhat>F0，保存anchor fallback
    G=F0−Fhat；T=Fhat+cG；保存固定状态及哈希

PREDICT_ONE(q, anchor_logit):
    不接受query标签，不改变训练状态或统计
    计算kSq、a=B^{-1}kSq和PSD一致的query创新s
    若拟合fallback、c=1、μ=aᵀvhat=0，返回anchor
    若s>0且λμ²/(2s)≤cG，记录zero-completion witness并返回anchor
    direction=sign(μ)
    在固定求解预算内迭代若干正tilt ζ：
        用支持规模Newton近似最小化F(v)+direction*ζ*aᵀv
        计算带gradient residual修正的有效dual upper bound
        保留最小有效upper，得到 signed lower correction e
        可按固定证书精度规则提前停机，不读query标签
    若无有限有效正e，返回anchor；否则δ=direction*e
    记录query-null证书、dual gap/residual、实际修正、运行时间与fallback
    返回sigmoid(anchor_logit+δ)
```

## 10. 参数、成本与缓存

- encoder与原分类器没有可训练参数更新。
- 支持MAP拟合系数r≤n≤20；线性函数类本身的维数仍可为d+1，不能把r误称为全部真实容量。
- 查询profile临时变量r+1，不是从query标签学到的额外参数。
- 支持Gram与Cholesky存储O(n²)，参考特征/支持特征O(nd)，配置与证书另记。
- MAP每个Newton迭代约O(nr²+r³)。每个未被廉价零证书否决的query，若使用R次tilt、每次I次Newton，约O(RI(nr²+r³))；特征核计算另计。
- 主线只要单侧证书；full interval可作数学测试或少量机制诊断。
- 冻结EEGNet特征、EA变换、支持/未来query身份和已合法的参考几何可缓存。缓存必须绑定源checkpoint、预处理、预算、支持/query UID、配置和代码哈希；不能仅按文件名复用。
- 数值小系统可在CPU float64运行，GPU仅负责冻结encoder特征提取；部署计时必须包含CPU/GPU转移及证书求解，不宣称零开销。主实现的root迭代上限在真实EEG结果产生前固定为12；有限预算只返回已有有效对偶外界所支持的保守动作，不声称得到精确端点。

## 11. 与前轮的实质区别及必要消融

| 前轮 | 已有机制 | R10需要被验证的区别 |
|---|---|---|
| R3 | 有限head ridge的Schur Newton、有效信息半径、支持KL保护 | 不再把空间参数信息量缩放当作query可靠性；对完整支持相容函数及未识别query完成求稳健动作 |
| R4 | OOF选择、one-SE、Brier收缩/stacking | c及kernel规则在meta开发阶段固定；目标支持内不反复选候选，也不从query标签学融合权重 |
| R8 | 支持几何/特征recolor及多种距离结构 | 主线复用EA和冻结特征，不增加几何复杂度；创新假设针对可执行的预测修正而非更复杂的对齐 |
| R9 | 残差GP MAP/Laplace、query posterior variance information fraction与anchor混合 | 同一个非二次objective相容集同时约束支持证据与支持外函数完成；以dual outer certificate求anchor-relative动作，无另加variance gate |

必须比较EA、R9原实现及matched-weight MAP；必须消融 `s=0`（忽略不可辨识query完成）、c=0/MAP、c=1/EA、二阶椭球替代完整logistic profile、R9 variance-fraction gate替代稳健动作。不同支持loss权重/先验尺度不能混进算法机制比较。

预期失败也必须被允许：若方法只是几乎处处复制EA、没有带来预先要求的配对BA/NLL或负迁移改善，不能将保守性包装为主指标突破。探索性已用holdout/Pan/Weibo与独立新数据确认必须分开报告。


## 12. 两个已实现的matched机制控制（均无主方法证书）

实现入口为`paper3r10/ablations.py::predict_ablation`。二者复用完全相同的支持拟合状态、class-balanced权重、参考白化、核幅度、λ、UID nugget与共享数值anchor，不能因比较而另选参数。

### 12.1 局部二次profile控制

在实际有限拟合值 vhat 处计算 g=∇F(vhat)、H=∇²F(vhat)。局部目标写为

`Q(v)=Fhat+gᵀ(v−vhat)+0.5(v−vhat)ᵀH(v−vhat)`。

完成平方得到中心 `v_c=vhat−H⁻¹g` 与 `Q_min=Fhat−0.5gᵀH⁻¹g`。使用原主方法阈值T，查询近似区间为

`aᵀv_c ± sqrt(2(T−Q_min)*(aᵀH⁻¹a+s/λ))`。

按该近似区间作anchor投影。它只隔离“完整非二次profile”相对局部椭球的作用；局部Q既不是原F的普遍上下界，也不构成原非线性模型集合的稳健证书。实现始终标注`not_profile_certified`。c=1时额外使用明确的anchor极限控制分支，不假称它来自Q(0)=F0。

### 12.2 R9式matched方差门控控制

使用相同有限拟合μ=aᵀvhat与H，定义

`V=s/λ+aᵀH⁻¹a`，`V_prior=kε(q,q)/λ`，`I=clip((V_prior−V)/V_prior,0,1)`。

以20点Gauss–Hermite求 `p_PP=E_Z sigmoid(m0+μ+sqrt(V)Z)`，输出

`p_gate=(1−0.75I)*p0+0.75I*p_PP`。

这是在R10匹配class-balanced目标与几何上的R9式门控消融；R9原生ordinary-likelihood基线仍单独保留，不能将二者冒充同一已完成旧方法。c不进入该门控公式，因为本控制特意移除了证据保留相容动作。它不具有原非线性模型集合的relative-regret证书，始终标注`not_profile_certified`。

两项控制均不读取query标签，不改变拟合状态或query统计；数值非有限/几何不一致时显式退回共享anchor并记录原因。
