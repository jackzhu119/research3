# R10 数学候选审查（设计前置记录）

本记录依据实际实现而非仅依赖轮次简介：R3 `paper3r3/algorithm.py` 的 `information/_propose/_protect`；R4 `paper3r4/algorithm.py` 的 `support_folds/shrink_weight/simplex_brier_weights/fit`；R9 `paper3r9/gp.py` 的 `fit_gp/posterior_moments/predict_proba`。最新 R8/R9 数值证据由独立审计代理核查；本记录不据旧监控选择获胜方法。

## 1. 已实现内容与不可绕过的限制

R3 已有有限 head ridge 下的 Schur 信息、信息有效秩缩放空间半径，以及支持集实际预测 KL 保护；R4 已有获得位置分块 OOF、one-SE、Brier 收缩及 2–3 分量概率 simplex stacking。R9 已有源/EA margin 为 offset 的 logistic 残差 GP、Laplace 方差和查询权重 `cap*(prior−posterior variance)/prior`。这些概念均不能更名后当作 R10 创新。

### 不可辨识性命题

对固定空间参数 θ，令支持集 head 设计矩阵 Zθ∈R^(n×p)，若 rank(Zθ)=n，则任意期望支持 margin u∈R^n 都存在 head h 满足 Zθ h=u−mθ。因而对每个使 Zθ 满行秩的空间参数，支持 margin 都可以由 head 完全吸收。仅凭这些支持标签，不能辨别某个空间方向具有 head 之外的独立预测价值。

局部形式：加权空间 Jacobian A 和加权 head 设计 Z 满足 `range(Z)=R^n` 时，精确 head 商空间信息 `Aᵀ(I−ZZ†)A=0`。有限 ridge 所产生的非零 Schur 矩阵依赖先验强度；它不是新增的数据可辨识信息。约15个有效支持点与65维 head 正是此限制的实质情形。

## 2. 三个技术候选

| 候选 | 核心数学对象 | 非平凡之处 | 主要风险与处理 |
|---|---|---|---|
| A：严格 head 商空间校准 | `Aᵀ(I−P_Z)A` 的正特征空间内做受限 Newton 更新 | 具有明确重参数化不变性、真实独立信息条件 | 完整 head 时通常精确归零；限制 head 容量再做 Schur 属于 R3/R7 已探索机制的组合，不推荐作为主创新 |
| B：可预测前缀证据驱动更新 | calibration prefix 上的 prequential likelihood ratio / e-process，仅用过去标签拟合下一次预测 | 在明确条件 null 下可证明 optional-stopping 有效，区别于反复 OOF 选最优 | B≤20 时检验力弱；EEG 不同会话的漂移破坏简单未来推断；e-process 是已有统计工具，不能仅加阈值当作创新 |
| C：保留支持证据增益的查询函数歧义优化 | 所有仍保留规定比例支持 objective gain 的函数，在该查询上的完整可达区间；其上求相对固定 anchor 的 minimax proper-loss regret | 支持拟合证据、支持以外的不可辨识方向、查询更新与 anchor 决策由同一优化问题相连；可得精确小系统求解、零更新证书 | 目标相容集/Rashomon interval、profile likelihood、robust Bayes 和 baseline-relative pessimism 均有先例；原创性只能归于具体耦合机制和可计算的 EEG 校准结构，不能声称这些基础对象新颖 |

优先继续研究 C，但须等待最新结果与文献碰撞核查。它不是已验证超过 EA 的方法。

## 3. 候选 C：定义

固定 anchor 概率 `p0(x)`，优先使用真实支持输入上合法计算的 Support-EA + 冻结 EEGNet。所有模型选择仅使用 meta_train/meta_validation；目标 query 的输入只能逐点用于预测，不用于白化、拟合或超参数选择。

令 support 为 `(xi,yi),i=1..n`，`n≤20`（QC 后可更少），anchor margin `mi=logit p0(xi)`。固定 PSD kernel k 的几何与超参数来自开发集。支持 residual 函数 f 的 penalized logistic objective 为

`J(f)=Σ_i[softplus(mi+f(xi))−yi(mi+f(xi))]+λ||f||_H²/2`, `λ>0`。

为与 R9 MAP 匹配，推荐将 amplitude 放在 kernel 中、λ固定为1，并使用 ordinary Bernoulli support likelihood。最终评价仍按协议报告 participant-equal BA、balanced NLL、Brier。这里不是把训练损失当作未来风险。

记 `fhat=argmin J`，`G=J(0)−J(fhat)≥0`。定义唯一 evidence-retention 参数 `c∈[0,1]` 和集合

`C_c={f:J(f)≤J(fhat)+cG}`。

每个集合成员至少保留 `(1−c)` 比例的观测 regularized support gain。这是操作性相容集合，不是置信区间；没有自动的95%覆盖率含义。

对查询 q 求

`l(q)=inf_{f∈C_c} f(q)`, `u(q)=sup_{f∈C_c}f(q)`，

并转成概率区间 `I_q=[sigmoid(m0(q)+l),sigmoid(m0(q)+u)]`。

最终动作直接解

`p*(q)=argmin_{p∈(0,1)} sup_{η∈I_q}{CE(η,p)−CE(η,p0(q))}`。

## 4. 有限维精确表示：必须保留支持外的函数自由度

先以真正 PSD 支持 Gram 矩阵 K 分解 `K=UΛUᵀ`，仅精确零特征值用 Moore–Penrose 处理，r=rank(K)。令

`B=UΛ^(1/2)`, `a_q=Λ^(−1/2)Uᵀk_Sq`, `s_q=k_qq−||a_q||²≥0`。

由 RKHS 正交分解，区间端点的任意最优函数可写成

`f_S=Bv`, `f(q)=a_qᵀv+sqrt(s_q)t`, `||f||²=||v||²+t²`。

对同一支持输出仍允许 t 变化；若只保留 span{k_xi} 并强行 t=0，就会遗漏支持标签没有识别的查询自由度，产生过度确定的修正。

令

`F(v)=Σ_i[softplus(mi+(Bv)i)−yi(mi+(Bv)i)]+λ||v||²/2`。

记 vhat 为其唯一最小值点，`T=F(vhat)+κ`, `κ=c[F(0)−F(vhat)]`。查询端点为

`max/min aᵀv+sqrt(s)t`, subject to `F(v)+λt²/2≤T`。

若包含 head 与空间/RBF 两个组成部分，只要给它们显式有限、固定正定先验，就可采用 direct-sum RKHS kernel，完成同样的**完整函数**歧义计算。有限 head prior 不得被描述为支持标签独立识别了空间方向；严格无先验 head 商空间仍可能为零。

## 5. 可计算端点：一维单调根 + 支持规模 Newton

对符号 ε∈{−1,+1} 和 ζ≥0，定义

`v_(εζ)=argmin_v[F(v)−εζ aᵀv]`, `t_(εζ)=εζ sqrt(s)/λ`。

只需求非负根

`D_ε(ζ)=F(v_(εζ))−F(vhat)+ζ²s/(2λ)=κ`。

因为 `∇F(v_(εζ))=εζ a` 且 `H=∇²F≽λI`，有

`D'_ε(ζ)=ζ aᵀH^(−1)a+ζs/λ≥0`。

因此 κ>0、`||a||²+s>0` 时可通过 doubling bracket 与 bisection 稳定求解。ε=+1 得上端点，ε=−1 得下端点：

`endpoint_ε=aᵀv_(εζ)+εζs/λ`。

每次 tilt 都是严格凸 logistic 小问题；从前一次 Newton 值 warm-start，无需重新训练 EEGNet，也不拟合 query 标签。可批量处理 query，时间与状态须包含实际求解开销。

## 6. 两个严格性质

### 6.1 Anchor-relative minimax regret 的闭式解

令 `I=[L,U]⊂(0,1)`。对任意固定 p，`CE(η,p)−CE(η,p0)` 关于 η 是仿射函数。若 p>p0，其最大值在 η=L；若 p<p0，则最大值在 η=U。分段最小化得到

`p*=clip(p0,L,U)`。

若 p0∈I，p*=p0 且最坏 regret为0。若 p0<L，p*=L，其最坏 regret为 `−KL(Ber(L)||Ber(p0))≤0`；p0>U 对称。因此，对所有 η∈I，`CE(η,p*)≤CE(η,p0)`。

**严格边界**：这个性质只对给定模型集合的概率区间成立。真实未来 η 未被证明属于 I；不构成真实未来 NLL、BA 或跨会话泛化保证，也不是无条件 safety guarantee。

### 6.2 零更新的查询歧义证书

MAP query residual 为 `μ=aᵀvhat`。若 s>0，取相同的支持最优 vhat 和 `t=−μ/sqrt(s)`，则查询 residual恰为0，支持拟合完全不变，代价只增加 `λμ²/(2s)`。

因此只要

`λμ²/(2s)≤κ`，

就有一个仍保留规定支持 gain 的函数在该查询上不作修正。故 0 在真实 residual 区间内，minimax 结果必须严格等于 anchor。这是可在求两端点前执行的廉价、精确充分条件。

更一般的精确零可行性条件为

`min_v[F(v)+λ(aᵀv)²/(2s)]≤T`。

该条件依赖未观测查询方向所允许的**函数完成**；它不是 R9 的 Laplace posterior-variance information fraction。

### 6.3 退化与单调性

- c=1：零函数在集合中，所有查询恢复 anchor。
- c=0 且精确优化：集合退化为 MAP residual；这是必要的 MAP 消融，不是安全机制。
- c 增大：集合嵌套扩大，精确区间扩大；对每个固定查询，精确minimax动作`|p*_c−p0|`单调不增。有限自适应tilt求界可能产生非嵌套松界，所以实际近似修正没有普遍跨c单调保证。
- `k_Sq=0`：取 support MAP 与 t=0 即得到零查询修正，所以恢复 anchor。
- `k_qq=0`：该查询在函数类中只能零修正。
- s=0：查询函数值由支持 RKHS 投影表示，但仍存在 support objective 近最优解之间的歧义；不能因 s=0 就视为标签充分。
- 单类/空有效支持：沿用协议显式 anchor fallback；不以无约束分类头完成 separation。
- kernel amplitude为0：显式 anchor fallback，不能只凭数值 nugget产生科学更新。

## 7. 数值证书、有限精度与计算成本

不能把不精确的可行点误当成端点外界。可用 dual 外界：对任意 ζ>0，upper endpoint有

`U(ζ)=[T−min_v(F(v)−ζaᵀv)]/ζ+ζs/(2λ)`。

近似 tilted optimizer v 的梯度 g 满足强凸下界

`min F_tilt ≥ F_tilt(v)−||g||²/(2λ)`。

将该下界代入 U 可得到保守 upper bound；对 −a 同样求出 lower bound。用保守外区间得到的 anchor projection，仍对原相容区间满足上述 relative-loss 性质。若 Newton/dual 证书不收敛，应明确 anchor fallback 与原因，不能静默宣称精确求解。

若基准 MAP只是近似值，可以把集合明确定义为 `J≤J(vhat_approx)+c[J(0)−J(vhat_approx)]`，这是可审计的有限计算相容集合；不把 vhat_approx 当成已证明精确极小值。若 J(vhat_approx)>J(0) 或非有限，直接 anchor。

R9 的相对1e−8谱截断不能直接当作精确零空间。丢掉正支持特征值后，所谓“支持外方向”仍可能改变被丢弃的支持输出，精确 RKHS 完成证明便失效。可采用显式声明的事件-ID nugget kernel `kε(x,x')=k(x,x')+ε1_{UID(x)=UID(x')}`，支持 UID 唯一、未来 query UID与支持不交；Kε严格 SPD，query diagonal也包含 ε，用全部 Cholesky方向，不静默改变理论对象。另一条路是固定开发集有限特征映射并明确所用投影后的模型类。

支持 n≤20，latent r≤n：一次 MAP约 `O(I(nr²+r³))`；每个非零候选查询的两端点约 `O(2RI(nr²+r³))`，R为根迭代数、I为Newton数；query-null证书可跳过部分查询求解。模型存储 `O(n²+nd)`，batch预测状态 `O(Qr²)`；encoder仍冻结，query没有梯度或统计更新。实际成本可能明显高于 R9 的一次 posterior solve，必须报告，不能称免费。

## 8. 最小证伪实验

1. 主方法相对 Support-EA、R9 anchored GP、R9 MAP、最强 R4 stacking 配对比较，而非只对失败空间校准器。
2. 去掉 query completion（人为 s=0）检验不可辨识方向的保护是否必要。
3. κ=0/MAP；c=1/anchor两个极限检查；用 R9 variance-fraction gate替代完整 robust resolver。
4. 用局部二阶椭球替代完整 logistic objective-profile，检验非二次 profile是否带来有意义收益。
5. 小预算 B4/B8必须检查有效样本、核秩、创新量、证书零更新比例、负迁移、纠正/引入错误。
6. 合成可识别 rank1 correction应出现非零并改善预测；相同支持拟合但查询正交方向可任意变化的构造应恢复 anchor。若两者都无更新，机制没有分辨能力。
7. 元开发阶段若绝大多数 query始终复制 EA，且未改善 NLL/负迁移，不应把“稳定性”包装成超过强基线的算法突破；停止或作为负结果记录。
8. 已参与设计的 holdout/Pan/Weibo只标探索；独立确认需要未用于设计的新数据。

## 9. 原创性声明边界

目标相容集、Rashomon区间、profile likelihood、minimax regret、kernel Schur complement各自已有文献。当前有希望作为原创研究问题的是：**少校准 EEG 中，对每个查询保留完整未识别函数完成空间，以已观察支持证据增益为门槛，统一求出 anchor-relative proper-loss 的稳健更新，并给出支持规模的可证数值解与零更新证书**。这是一项待检验的具体算法机制；不是对这些数学概念的发明声明。若文献已有同构完整算法，应重新评估或放弃，而不是更名。
