# 第七轮支持风险：相对变化与绝对难度的分离

本模块只读取已获取支持输入对应的预测和支持标签，不接收未来query。源参考显式detach。它不新增训练启动检查；目标值与证书都是算法诊断，正值不终止其他独立任务。

## 1. 同一基准分布与四个目标

每个有效支持试次的二类NLL为 ℓᵢ(w)=−log p_w(yᵢ|xᵢ)。基准qᵢ=1/(2n_yᵢ)，故各类基准质量1/2。所有NLL从log_softmax计算，不用概率1e−12截断。

源相对变化 δᵢ(w)=ℓᵢ(w)−ℓᵢ(0)，源模型/BN/源参考不随内层或元梯度更新。四个数据目标：

| kind | 数据项 | 识别对象 |
|---|---|---|
| ce | Σqᵢℓᵢ(w) | 普通类别平衡CE |
| absolute_dro | τ log Σqᵢ exp(ℓᵢ(w)/τ) | 强调绝对高损失样本 |
| relative_dro | τ log Σqᵢ exp(δᵢ(w)/τ) | 强调适应相对源模型变差的位置 |
| variance_regret | E_qδ + Var_qδ/(2τ) | 相对指数目标的二阶近似 |

父拟合器另外加相同科学ridge λθ‖θ‖²+λa‖a‖²/d。模块不偷偷把ridge或数值damping混入逐试次风险。

这里absolute/relative使用**KL惩罚的有限重加权**，不是固定KL半径DRO的精确解。q是类别平衡基准，Gibbs最坏权p可改变类别总质量；不能描述为已约束每类最坏质量仍为1/2。诊断显式记录其类别质量、最大权重和有效样本量。

## 2. 精确变分恒等式与证书范围

定义 Rτ=τ log E_q exp(δ/τ)。有限支持上

Rτ = sup_p {E_pδ−τKL(p‖q)},

其中p为同一批支持点上的任意概率向量，Gibbs最大者pᵢ∝qᵢexp(δᵢ/τ)。因此对任意这样的p：

E_pδ ≤ Rτ + τKL(p‖q).

若父拟合器从真实identity开始，对**实际模型**的 F=Rτ+ridge 采用不增的接受规则，并实际达到F≤0，则

E_pδ ≤ τKL(p‖q)−ridge.

这是对已观测支持点重权的确定性代数结论。它不证明未知未来query风险不增加，不认证真实分布偏移属于离散重权，也不是有限样本置信区间或标签噪声鲁棒定理。只用身份切线代理优化时，结论也只能约束该代理；必须另读真实指数图的支持损失才能报告实际模型证书。

identity逐项δ=0，所以Rτ(0)=0。earned_KL_radius定义为−Rτ/τ，数值为负时保持负值，不能截成0后冒充成功。Rτ+ridge为正时，条件证书字段为None，仍记录一般变分界。

父拟合器若允许数值接受容差，需按实际保存F和容差解释；本模块的support_certificate_holds严格检查F≤0，numerical_certificate_tolerance=0，不把轻微正值标为精确证书成立。

## 3. 有力的空效应与局部机制

τ趋向无穷时 Rτ→E_qδ，而E_qδ与普通CE仅差不依赖w的源常数，所以optimizer和梯度相同。若ℓᵢ(0)对所有试次为常数c，relative与absolute目标也仅差c，optimizer完全相同。这些属于软件/机制空效应，不增加学校启动门槛。

有限指数矩下，较大τ或较小δ时：

Rτ=E_qδ+Var_qδ/(2τ)+O(第三累积量/τ²).

相对目标的领先修正是“适应造成的损失变化方差”，与绝对高损失方差不同。variance_regret只保留二阶近似；它不自动拥有相对指数目标的同一精确证书。

打乱源baseline会破坏identity逐项δ=0。若做该探针，需明确它是错误anchor控制，不能赋予相同证书。源baseline为全局常数则与绝对DRO严格等价，无需另训练一个重复主方法。

## 4. 数值与API

`support_risk(logits, source_logits, y, kind='relative_dro', temperature=.5)` 返回可微的float64数据标量，候选的一阶及元梯度保留，参考梯度完全切断。

`risk_diagnostics(..., temperature=.5, ridge=0., kl_radius=0.)` 返回JSON标量：相对指数风险、平均变化、变化方差、log指数矩、regularized_relative_objective、证书是否成立、signed earned_KL_radius、指定KL下的一般界与条件证书界，以及Gibbs权集中度。

使用max平移避免指数溢出；接近质量1时用expm1/log1p，避免极大τ乘logsumexp归一化舍入误差。诊断保存log指数矩而非直接取exp，避免无必要的溢出。缺少一类支持由父拟合器按既有回退处理；底层API不会猜造缺失类权重。

## 5. 先前结论与候选创新

- [Agarwal与Zhang，COLT2022](https://proceedings.mlr.press/v178/agarwal22b.html)已提出分布偏移下Minimax Regret Optimization，并指出绝对DRO不自动意味着均匀低遗憾。
- [DoReMi，NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html)已将reference excess loss用于领域重加权，本轮不能声称首次使用源相对损失。
- [Kouw与Loog，Pattern Recognition Letters2021](https://doi.org/10.1016/j.patrec.2021.05.005)及其[原稿](https://arxiv.org/abs/1706.08082)讨论相对源模型的保守目标域适应；该文特定模型/假设下结论不直接搬到当前神经网络和未观测query。
- [Iterative Robust Satisficing，ICML2026](https://proceedings.mlr.press/v306/agirman26a.html)研究名义性能目标下的脆弱性优化，是另一条已有鲁棒学习路线。

候选增量是在严格小支持预算的Lie空间适应中，对真实支持变化指数矩、二阶变化方差、绝对难度、源保护和头/空间结构作可辨识对照，并用未参与选择的会话/数据评价。源模型很差、支持错标、τ过小或分布变化超出支持重权时都可能失败；保留这些反例，不许将支持代数性质直接写成未来准确率提升。
