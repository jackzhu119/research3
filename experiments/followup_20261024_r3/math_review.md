# Paper3 第三轮数学审查与可执行研究方案

本方案保持原任务：冻结已有 EEGNet、BN 和源参考协方差，只用新会话最前面的 20 个已采集试次进行有监督校准，随后预测未来完整物理运行段。无效试次仍占采集预算，不用未来样本补齐。第三轮代码独立位于 `paper3r3/algorithm.py`，不修改前两轮包。这里提出可检验的机制，不承诺准确率、统计显著性或最优性。

已有 V2 的微小/零改进不能证明某个新机制必然有效。第三轮首先检验一个更基础的问题：**20 个标签是否足以把空间调整与残差头调整区分开，以及数值优化有没有真正使用可观测的方向？** 支持集曲率、头部混淆、变换半径和预测保护分别有明确的公式和对照。

## 1. 原模型、维数和不能混淆的性质

令 (Q\in\mathbb R^{10\times11}) 为正交 CAR 坐标映射，(QQ^T=I_{10})、(Q^TQ=H_{CAR})。固定正定源参考为 (R\in\mathbb R^{10\times10})。正交无迹基 (E\in\mathbb R^{100\times99})，字典 (B\in\mathbb R^{99\times r})，(B^TB=I_r)，(r\in\{2,4,8\})。采用行优先向量化：

\[
 M(\theta)=\operatorname{unvec}(EB\theta),\quad
 A(\theta)=Q^TR^{1/2}\exp M(\theta)R^{-1/2}Q,\qquad\|\theta\|_2\le\rho.
\]

每试次联合 RMS 归一化为 (N(Y)=Y/s(Y))，(s(Y)=\sqrt{\|Y\|_F^2/(11T)})。冻结编码器给出特征 (h_\theta(x)=\phi(N(A(\theta)x))\in\mathbb R^p)，原二分类头的差分 logit 为 (m_{0,\theta}(x))。唯一残差头采用二分类对比参数 (a=(w,b)\in\mathbb R^{p+1})：

\[
 m_{\theta,a}(x)=m_{0,\theta}(x)+[h_\theta(x)^T,1]a.
\]

预测实际使用上述非线性指数变换。残差加回原两个 logits 时为 \((-\Delta/2,+\Delta/2)\)，避免额外的二分类 softmax 公共偏移自由度。

由正交性可得 \(\|M\|_F=\|\theta\|_2\le\rho\)、\(\operatorname{tr}M=0\)，所以白化 10 维空间中 \(\det\exp M=1\)、\(\kappa_2(\exp M)\le e^{2\rho}\)。物理 CAR 子空间中的共轭变换只能得到上界 \(\kappa_2\le\kappa_2(R)e^{2\rho}\)。11 维 (A\) 秩亏，不能说其行列式为 1 或条件数受同一界控制。RMS 归一化后的完整预测映射也不具有线性体积守恒性质。

无迹和小半径不能保证分类信息可辨识。RMS 本身消除了正的试次整体幅度；编码器还有其他可能不敏感的方向。协方差在各向同性分布中无法辨识纯旋转，这也是不宜只根据协方差偏移决定空间调整的原因。

## 2. 方向一：精确的单位点 Lie 切线，加上有头部消元的约束 Newton 校准

### 2.1 一次输入反传得到全部 99 维切线

记 \(L=Q^TR^{1/2}\)、\(Z_i=R^{-1/2}Qx_i\)、\(Y_i=LZ_i\)、\(s_i=s(Y_i)\)。这里使用 \(Y_i\) 而非直接假设原始浮点数组恰好在 CAR 子空间，保证和实际几何前向一致。

在单位点 \((\theta,a)=(0,0)\) 上，对源差分 logit 求归一化输入梯度：

\[
 u_i=\left.\frac{\partial m_0}{\partial N}\right|_{N=Y_i/s_i},\qquad
 v_i=\frac1{s_i}\left(u_i-Y_i\frac{\langle u_i,Y_i\rangle_F}{\|Y_i\|_F^2}\right).
\]

由于 \(D\exp_0[G]=G\)，精确的一阶导数为

\[
 Dm_0[G]=\langle v_i,LGZ_i\rangle_F
          =\langle L^Tv_iZ_i^T,G\rangle_F.
\]

因此令

\[
 C_i=\operatorname{vec}(L^Tv_iZ_i^T)^TE\in\mathbb R^{1\times99},
 \qquad J=CB\in\mathbb R^{n\times r}.
\]

**(C_i\) 不依赖 (B\)**。源网络处于 eval 模式，BN 使用固定统计量、dropout 关闭且试次之间无交互时，`autograd.grad(margin.sum(), normalized_x)` 一次反向传播即可得到所有试次的 (u_i\)。随后计算 (C\)、单位点特征和 (m_0\)，可全部 detach。对固定单位点替代模型而言，这不是丢弃元梯度：这些量确实与 (B\) 无关。外层仍通过 (CB\)、小矩阵求解、约束半径和真实指数变换后的查询损失更新 (B\)。

需要区分两件事：切线导数在单位点是精确的；用这条切线处理非零 \(\theta\) 的支持集损失是局部近似。实际查询仍用非线性 (\exp M\)，所以存在支持替代模型和最终预测之间的误差，必须记录实际支持损失/切线损失并用 `relinearized` 参考检验。

### 2.2 固定、严格凸的支持集替代问题

令 (H_0=[\phi(N(Y_i))^T,1]_{i=1}^n\in\mathbb R^{n\times(p+1)})。(n\le20\)。有两类支持时定义固定权重 \(q_i=(2n_{y_i})^{-1}\)，因此 \(\sum_iq_i=1\)。主替代问题为

\[
 \min_{\|\theta\|\le\rho}
 \sum_iq_i\{\operatorname{softplus}(\tilde m_i)-y_i\tilde m_i\}
 +\lambda_\theta\|\theta\|^2
 +\frac{\lambda_h}{p+1}\|a\|^2,
 \quad\tilde m=m_0+J\theta+H_0a.
\]

正的两项先验使这一固定线性预测模型具有唯一的约束最小值；代码使用固定 10 步有回溯的 Newton 迭代，并不宣称在所有支持集上已达到这个最小值。原尺度的先验曲率是 **\(2\lambda_\theta I\)** 和 **\(2\lambda_h/(p+1)I\)**，不能遗漏 2 或头维数的除数。

主实验所有优化器/空间方法采用同一个 \(\lambda_h=.1\)，由匹配基线隔离优化与信息机制；\(\lambda_\theta=.01\)、\(\rho=.15\)、\(r=4\)。保留 \(\lambda_h=.001\) 的旧 SGD10 命名参考，明确它与主实验既有优化器差异，也有头正则差异。头正则敏感性 \(.001,.01,1\) 的空间方法须和同值的 SGD/头基线配对，不能把更强头正则本身的收益算成新几何机制。

### 2.3 GGN、头部消元和不能宣称的新颖性

在当前替代模型上，\(p_i=\sigma(\tilde m_i)\)，\(D_{ii}=q_ip_i(1-p_i)\)。数据曲率是

\[
 F=[J,H_0]^TD[J,H_0].
\]

这是该二分类线性预测模型的精确数据 Hessian，也是固定设计权重下的条件 Fisher/GGN。它不同于用当前真标签形成的逐样本梯度外积“经验 Fisher”。在实际非线性模型的 `relinearized` 参考中，同一个公式仅是 GGN；那里遗漏了差分 logit 的二阶导数项。[Martens, JMLR 2020](https://jmlr.org/papers/v21/17-678.html) 已系统研究这些区别。

对当前完整目标的梯度 \(g_\theta,g_a\)，在两个对角块加入先验曲率及 Levenberg 阻尼 \(\mu I\)，记得到的正定块为 \(K_{\theta\theta},K_{\theta a},K_{aa}\)。消元得到

\[
 S=K_{\theta\theta}-K_{\theta a}K_{aa}^{-1}K_{a\theta},\qquad
 \bar g=g_\theta-K_{\theta a}K_{aa}^{-1}g_a.
\]

无约束时

\[
 \Delta\theta=-S^{-1}\bar g,\qquad
 \Delta a=-K_{aa}^{-1}(g_a+K_{a\theta}\Delta\theta).
\]

只有把头真正求到驻点时才可以令 (g_a=0\)。当前每步没有这一假设。头部在**当前二次模型中**被精确条件优化，不等于已经解析求解了原 ridge-logistic 头；后者没有一般闭式解。

Schur 消元和完整联合正定系统的解在约束不活跃时完全相同。因此“用 Schur 而非全矩阵”本身不是新的模型或收益来源，也不能把内部数值舍入差异写成科研贡献。经典变量投影、可微 ridge/logistic 内求解和可微凸优化已有充分先例：[Golub & Pereyra 1973](https://epubs.siam.org/doi/10.1137/0710036)、[Bertinetto 等 ICLR 2019](https://www.robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/)、[Lee 等 CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html)。本实验检验的是受前缀预算约束的 Lie 校准、头部竞争和约束优化的具体组合。

### 2.4 必须约束新参数端点，不能只约束步长

设当前参数为 \(\theta\)。正确的局部问题是

\[
 \min_{\|\theta+\Delta\theta\|\le\rho}
 \bar g^T\Delta\theta+\tfrac12\Delta\theta^TS\Delta\theta.
\]

令 (t=\theta+\Delta\theta\)、\(b=S\theta-\bar g\)，则 (t\) 解

\[
 (S+\nu I)t=b,\quad \nu\ge0,\quad
 \|t\|\le\rho,\quad \nu(\|t\|-\rho)=0.
\]

半径不活跃时直接解线性系统；活跃时对标量 \(\nu\) 作括区间和二分。随后以这个受约束的 \(\Delta\theta\) 重新计算头部二次模型的条件最优更新。`tangent_joint` 对照先求完整无约束 Newton 步再朴素裁剪 \(\theta\)，保持原无约束头更新；两者的可检验差异发生在半径活跃时。`tangent_diag` 则丢弃交叉曲率，作为常见预条件近似的对照。

代码再沿旧参数至候选端点回溯，检查实际的固定替代目标不增加。球是凸集，所以已可行的两端之间保持可行。固定小半径、固定 10 步、同一阻尼和回溯预算应在所有对应方法之间匹配。

### 2.5 边界元梯度不能把二分出的乘子当常数

在边界，记 (A=S+\nu I\)。对输出梯度 (z\)，定义

\[
 v=\frac{t^TA^{-1}z}{t^TA^{-1}t},\qquad
 u=A^{-1}(z-vt).
\]

通过 KKT 方程得到

\[
 \nabla_b\ell=u,\quad
 \nabla_S\ell=-\operatorname{sym}(ut^T),\quad
 \partial\ell/\partial\rho=\rho v.
\]

内部时 (u=S^{-1}z\)、半径梯度为 0。`ball_quadratic` 自定义 autograd 实现这个一阶反向；特别是第三项保证“信息决定半径”的外层梯度成立。仅把前向二分的 \(\nu\) detach 后调用普通求解，会漏掉边界运动项。活跃集切换、回溯分支和保护比例选择仍使学习器仅分段可微；实现只承诺第一阶外层梯度，不承诺边界处的高阶光滑性。

## 3. 方向二：用“扣除头部竞争后的信息”决定空间预算

空间 Fisher 大，并不表示空间参数可以单独识别。头部可能用另一组参数产生几乎相同的 20 个支持预测。只在单位点计算、不加入空间先验或数值阻尼的惩罚头 Schur 数据矩阵：

\[
 \mathcal I=J^TD_0J-J^TD_0H_0
 (H_0^TD_0H_0+\eta_hI)^{-1}H_0^TD_0J,\quad
 \eta_h=2\lambda_h/(p+1),\quad
 (D_0)_{ii}=q_i\sigma(m_{0i})(1-\sigma(m_{0i})).
\]

理想精确算术中 (0\preceq\mathcal I\preceq J^TD_0J\)。当 (H_0\) 满行秩且头先验趋近 0 时，头部能够插值全部支持差分 logits，\(\mathcal I\to0\)。本任务常见 (p+1=65>20\)，这个退化风险是真实的结构问题，增加到 (r=8\) 并不会自动解决。

它是**给定头正则后的局部惩罚信息**，不是“20 个标签严格证明了真实生理空间参数的唯一可辨识性”。必须记录头先验、原始/扣除后特征值和有效维数，不能把空间先验产生的正定性误当成数据证据。

定义

\[
 d_{eff}=\operatorname{tr}\{\mathcal I(\mathcal I+\tau I)^{-1}\}
 =\sum_j\frac{s_j}{s_j+\tau},\quad
 \tau=2\lambda_\theta c_I,\quad
 \rho_S=\rho\sqrt{d_{eff}/r},\quad c_I>0.
\]

`information_scale=c_I` 主值 1，敏感性只改变 (c_I\)，不改变实际损失中的先验。有效维数在 \([0,r]\)，从而信息半径不超过原半径。代码通过矩阵求解和迹计算 (d_{eff}\)，不必对重根特征向量反向传播；特征值只作诊断。半径在支持单位点确定后固定整个适应过程，避免同一次迭代中“缩小球却仍拿旧可行点”造成不一致。

这个规则的假设是：在空间调整相对于头校准仍有可观测信息的会话中，允许更多空间移动。必须同时比较相同平均收缩强度的固定半径对照，才能区分“当前支持信息有价值”和“总体收缩更强”。固定对照半径只能从元训练/验证选择，不能根据内部留出平均 (d_{eff}\) 回填。

全矩阵 Schur、迹、各向同性先验和端点球在 (B\mapsto BO,\theta\mapsto O^T\theta\) 下保持等变，最终物理变换不变。一般的逐坐标对角 Fisher 不具备此性质，`tangent_diag` 应明确是这个近似对照。

有效维数/低秩先验/线性化适应都属于已有方法谱系，不能声称首次提出。尤其需与 [Maddox 等 AISTATS 2021 的线性化网络适应](https://proceedings.mlr.press/v130/maddox21a.html) 和 [LUMA AISTATS 2026](https://proceedings.mlr.press/v300/park26b.html) 划清边界；后者研究线性化回归，不是本二分类任务的现成后验保证。

**重要失败情形：**源模型自信但判错时，\(p_0(1-p_0)\) 很小，信息规则可能错误地阻止最需要的适应；头正则过弱时它可能几乎总是收缩到 0；特征同质性也可能令它对所有会话给出相近半径。平衡平均损失的权重和为 1，信息量不会像独立样本对数似然之和那样自动随 (n\) 线性增长，故不要承诺预算曲线的 (d_{eff}\) 单调增长。

## 4. 方向三：在实际预测上保护源模型，而非把参数球误当预测保护

参数球控制几何大小，却没有直接限制高增益编码器附近的预测变化。`tangent_protected` 和 `information_protected` 在获得候选 \((\hat\theta,\hat a)\) 后，把整个更新按同一比例 \(\alpha\in\{1,1/2,\ldots,0\}\) 缩小：

\[
 (\theta,a)=\alpha(\hat\theta,\hat a).
\]

使用真实的非线性指数变换和真实变换后特征，逐级检查相对于冻结源模型的支持集平衡 Bernoulli KL：

\[
 \sum_iq_i\operatorname{KL}\{
 \operatorname{Bern}(\sigma(m_{0i}))\;\|\;
 \operatorname{Bern}(\sigma(m_{\alpha\hat\theta,\alpha\hat a}(x_i)))\}
 \le\epsilon,\qquad\epsilon=.01.
\]

这个约束锚定的是**源模型**；逐步限制相邻迭代的 KL 并不等价于限制最终偏离源模型的 KL。实际检查也避开了另一个陷阱：正则化目标的 Schur 曲率并不自动等于剖面预测的 KL 曲率。若要直接在联合二次规划中加入预测 Fisher 约束，必须对 (K+\mu F\) 重新消元，而不能把先算出的任意 Schur 矩阵当成 KL 度量。这里的后验收缩规则更简单、可审查，且不宣称解出了最优联合 KL 约束问题。

在被检查的支持分布上，Pinsker 加 Jensen 给出

\[
 \sum_iq_i|p_{new,i}-p_{0,i}|\le\sqrt{\epsilon/2}.
\]

这个结论只涉及当前支持集的平均概率变化。它不保证零负翻转，不保证未来试次准确率或校准误差提高，也不控制未观察的会话漂移。源模型系统性错误时，保护可能妨碍纠错；接近 0.5 的源概率即使发生小 KL 变化也可能翻转类别。需同时报告 BA、负翻转、纠错和概率质量，而不能只挑保护更好的指标。

KL 近端约束和信赖域本身有成熟先例：[Schulman 等 ICML 2015](https://proceedings.mlr.press/v37/schulman15.html)。该强化学习文献的策略改进理论不转移到本离线 EEG 分类任务。这里实现的支持保护不具有“未来不退步”的理论保证。

## 5. 已实现的对照和最小可证伪比较

| 方法 | 支持集求解 | 主要作用 |
|---|---|---|
| `sgd10` | 原非线性模型 10 步同步 SGD | 与前一轮内核数值一致；主实验采用匹配头正则 |
| `sgd100` | 同模型 100 步 SGD | 排查 10 步欠优化；实际耗时另行报告，不能先称严格等算力 |
| `head_sgd` | 空间固定，只校准头 | 检验空间部分是否真的有增益 |
| `head_newton` | 固定特征上的 ridge-logistic Newton | 排除“只是头优化得更好” |
| `tangent_diag` | 固定切线、对角曲率 | 检验完整交叉曲率的价值 |
| `tangent_joint` | 联合 Newton 后朴素裁剪空间端点 | 内部等价性和边界处理对照 |
| `tangent_profiled` | Schur、精确二次球端点、条件头更新 | 主优化机制 |
| `tangent_information` | 同上，信息决定半径 | 头竞争之后的信息是否可用 |
| `tangent_protected` | 主方法加真实支持 KL 后验收缩 | 预测保护机制 |
| `information_protected` | 信息半径与真实支持 KL | 预先指定的组合，不事后拼装 |
| `relinearized` | 每步真实非线性 Jacobian/GGN | 仅后验参考；拒绝 `create_graph=True`，不伪装成已元训练方法 |

最有解释力的比较是：profiled 对匹配 SGD10/100 和 head Newton；profiled 对 joint 仅在球活跃时的差异；information 对 profiled 和固定收缩半径；protected 对 profiled；组合对单组件。字典是否重新训练也必须匹配：不能用重新元训练的 B 对旧 B 然后把差异全部解释成目标优化器。无约束内部等价检查是软件正确性要求，不作为独立科研贡献。

建议每个会话至少保存：有效支持数、类别计数、切线/实际支持损失、\(\|\theta\|\)、球活跃比例、条件头参数范数、原始/Schur 曲率谱、\(d_{eff}\)、有效半径、线搜索比例、实际源 KL、保护比例、目标校准延迟、相对源模型负翻转和纠错。以上诊断用于解释预先定义的比较，不用留出曲线反向修改方法。

预算曲线仍使用同一最前 20 个已采集试次的短前缀 B4/8/12/16/20，并始终评价 B20 定义的相同未来查询。单类别支持遵循原合同，回退冻结源模型并报告比例。无标签查询统计、未来协方差和未来标签均不进入 `fit`。

## 6. 数值实现、资源和验证边界

实现依赖 PyTorch 和已有 NumPy；没有 scipy 求解器。几何、切线和小矩阵求解使用 float64，冻结 EEGNet 可保留原 float32；无 AMP。端点球前向只把至多 8×8 的正定矩阵、右侧向量和半径传至 CPU，单次 NumPy 谱分解后进行标量二分，最后将结果复制回输入设备。这样避免每个二分步骤都触发 GPU 小矩阵求解及主机同步；自定义一阶 KKT 反向仍在原设备上计算。校准延迟必须包括这些 CPU/GPU 小量传输，不能只计 GPU 预测时间。训练逐 episode 累积 4 项外层梯度并及时释放图，保持原受试者/会话抽样和训练预算。主 learner 不需要通过网络的内层二阶导数反传，真正元训练图主要经过 4/8 维约束求解和实际查询前向。`relinearized` 需要逐输出求输入方向导数，明确只作较贵的后验参考。

学校资源按本轮实际要求设置为最多同时 3 个单 GPU 作业，每个 A100 PCIe 40 GB、12 CPU、64 GB 内存；小矩阵算子线程从 4 开始。不能从矩阵维数推断完整元训练一定低于 40 GB，也不能预先承诺速度收益；需要记录学校实际峰值内存和 episode 用时。旧轮次训练和产物保持原位。

代码开发时针对实际 CompactEEGNet 的 11×128 输入验证：单位点解析切线与直接 autograd 一致、SGD10 与旧内核一致、Schur 与全联合内部解一致、端点球 KKT 一阶梯度有限差分、信息半径完整外层梯度、字典正交换基等变、头插值导致的信息退化、四 episode 外层累积和 Stiefel 更新、源模型冻结、实际 KL 检查及单类别回退。它们是合成软件验证，不是 EEG 效果、GPU 性能或统计功效验证。最终精确版本和测试数以根目录验证记录为准。

三个方向可以失败，而且失败各有可解释结果：优化对照无差异意味着原 SGD 未明显欠优化；空间不优于头 Newton 意味着新会话证据主要支持头校准；信息半径几乎总为零暴露头/空间无法在 20 试次下区分；保护改善负翻转却降低 BA 表明纠错受到约束。应保留这些结果，不在观察留出以后改写主终点或用“稳定性”掩盖准确率没有提升。
