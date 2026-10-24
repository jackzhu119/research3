# Paper3 第三轮：有限前缀证据下可辨识的 Lie 空间校准

版本：2026-10-24。本文与 `paper3r3/algorithm.py`、`plan.py` 和 `report.py` 对齐。第三轮为独立代码与输出目录，前两轮源码、作业和产物不被覆盖。本轮主矩阵有 **174 个逻辑任务：21 个字典训练/复用任务、78 个核心评价任务、72 个敏感性评价任务、3 个 EA 参考任务**；174 不是元训练次数。可选 Weibo 发布顺序迁移另有 18 个评价任务，单独建计划。

当前已有合成软件与精确版本兼容性验证，尚无第三轮学校 GPU 的真实 EEG 效果、显著性或运行时间结论。V2 部分种子的微小/零收益只说明需要检查更基础的假设，不能据此宣布新方法必然更好。当前内部队列已属于前两轮研究体系，本轮属于**探索性内部重复验证**，不能重新称为首次盲测或独立外部确认。

## 1. 科研问题与三个可证伪方向

任务保持为：**冻结源 EEGNet，只用一个新会话最前 20 个已采集试次校准，再预测其后完整物理运行段。** 不随机重排会话、不补采有效试次、不用未来无标签信号估计协方差、不更新源网络 BN 或权重。

核心问题是：20 个标签提供的有限信息，能否把“空间变化”与“残差分类头变化”区分开？如果不能，如何减少没有充分证据的空间移动？第三轮围绕以下三个方向，而非继续增加条件门：

1. **解析身份切空间与约束校准。** 由 CAR、白化、RMS 和冻结网络的输入导数，得到完整 99 维无迹 Lie 方向的解析一阶信息；在低秩字典中用固定凸代理求解，并在空间约束激活时重新优化条件头更新。
2. **头部竞争之后的信息收缩。** 把残差头可以补偿的支持预测变化扣除，再用剩余局部信息决定空间预算，检验是否比统一半径更有用。
3. **实际预测保护与联合字典训练。** 在真实指数变换后的支持预测上约束相对源模型的 KL；通过实际未来查询损失，学习与该校准学习器匹配的空间字典。

可证伪的结果包括：更充分的 SGD 与新求解器无差异；空间调整不优于充分优化的头；信息半径总是接近零；保护减少负翻转却妨碍纠错；联合学习字典不优于旧字典。任何一种结果都应保留，不能在观察内部留出后更换主终点。

## 2. 数据、时间预算与冻结合同

源模型复用 BNCI2014_001、Cho2017、Lee2019_MI、PhysionetMI 的已有匹配种子 EEGNet 和源参考协方差。Stieger2021 沿用原参与者清单：预期 40 人元训练、11 人元验证、11 人内部留出，实际身份以冻结 manifest 为准，不重新划分。每个源模型种子对应自己的参考协方差、字典和结果，种子固定为 **20261018、20261019、20261020**。

输入合同保持 clinic11：FC3、FC4、C5、C3、C1、Cz、C2、C4、C6、CP3、CP4；128 Hz、8–30 Hz、原始 MI 上下文 0–1.5 秒，裁剪 0.25–1.25 秒为 128 点；按原规则做 11 通道 CAR、时域去均值与联合 RMS。使用现有缓存，不另行重滤全库或更换时间窗。EEGNet 本身及轻量跨范式建模属于已有基础。[Lawhern 等，EEGNet](https://doi.org/10.1088/1741-2552/aace8c)

采集预算按原始事件计数。失败 QC 的试次仍占前 20 名额，只将有效子序列送入拟合；不拿第 21 条以后补足。B4/8/12/16/20 均截取同一个 B20 原始前缀。**所有预算评价同一组由 B20 定义的未来完整物理运行段查询**，以防短预算通过额外获得更早、更容易的查询改变结果。支持集不足两类时，所有有监督校准回退源模型并记录比例，不能悄悄删掉该会话。支持 EA 的无监督参考仍可使用有效信号。

元训练查询标签只用于外层训练目标；元验证标签只用于选择字典 checkpoint；内部留出查询标签只用于评价。`fit` 不接收查询输入或标签。训练、验证和留出在参与者层面隔离，而非把同一个人的不同会话当独立划分。

## 3. 几何与分类模型

设 \(Q\in\mathbb R^{10\times11}\)，\(QQ^T=I_{10}\)、\(Q^TQ=H_{CAR}\)。固定源参考 \(R\succ0\)。令 \(E\in\mathbb R^{100\times99}\) 是 Frobenius 正交的 10×10 无迹矩阵基，\(B\in\mathbb R^{99\times r}\)、\(B^TB=I_r\)，\(r\in\{2,4,8\}\)。行优先向量化下：

\[
 M(\theta)=\operatorname{unvec}(EB\theta),\qquad
 A(\theta)=Q^TR^{1/2}\exp M(\theta)R^{-1/2}Q,\qquad
 \|\theta\|_2\le\rho.
\]

对 \(Y\in\mathbb R^{11\times T}\)，联合 RMS 为 \(s(Y)=\sqrt{\|Y\|_F^2/(11T)}\)，\(N(Y)=Y/s(Y)\)。冻结编码器给出 \(h_\theta(x)=\phi(N(A(\theta)x))\in\mathbb R^p\)，本短窗 EEGNet 的 \(p=64\)。原二分类差分 logit 为 \(m_{0,\theta}(x)\)。目标只拟合 \(r\) 维空间参数和 \(p+1\) 维对比头 \(a=(w,b)\)：

\[
 m_{\theta,a}(x)=m_{0,\theta}(x)+[h_\theta(x)^T,1]a.
\]

残差加回原二 logits 为 \((-\Delta/2,+\Delta/2)\)，保留原头及其偏置，避免二分类 softmax 的公共 logit 平移冗余。最终查询预测使用**真实的非线性指数变换和变换后特征**。

正交基和字典给出 \(\|M\|_F=\|\theta\|_2\)，并且 \(\operatorname{tr}M=0\)。因此白化 10 维空间中 \(\det\exp M=1\)、\(\kappa_2(\exp M)\le e^{2\rho}\)。物理 CAR 子空间的共轭变换只有上界 \(\kappa_2\le\kappa_2(R)e^{2\rho}\)；11 维 \(A\) 秩亏，不能说其行列式为 1。RMS 后的整个映射也不是线性体积保持映射。

一般 \(\exp(S+K)\) 并不是 SPD 矩阵，也不能拆成独立旋转与缩放；这里的对象是施加于原始时序的可逆 Lie 作用，不是直接在协方差 SPD 流形上预测生理电极位置。协方差对齐、旋转和切空间迁移都有充分先例，不能作为“首次”贡献。[He & Wu，欧氏对齐](https://pubmed.ncbi.nlm.nih.gov/31034407/)、[Rodrigues 等，Riemannian Procrustes](https://pubmed.ncbi.nlm.nih.gov/30596565/)、[Bleuzé 等，切空间对齐](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1049985/full)

## 4. 解析 99 维身份切空间：何处精确，何处近似

令 \(L=Q^TR^{1/2}\)、\(Z_i=R^{-1/2}Qx_i\)、\(Y_i=LZ_i\)、\(s_i=s(Y_i)\)。使用几何前向的 \(Y_i\) 而非假设浮点输入恰好处于 CAR 子空间，保证导数路径一致。在 \(\theta=0,a=0\) 计算源差分 logit 对归一化输入的梯度：

\[
 u_i=\left.\frac{\partial m_0}{\partial N}\right|_{N=Y_i/s_i},\qquad
 v_i=\frac1{s_i}\left(u_i-Y_i\frac{\langle u_i,Y_i\rangle_F}{\|Y_i\|_F^2}\right).
\]

因为 \(D\exp_0[G]=G\)，对任意生成元 \(G\) 有

\[
 Dm_0[G]=\langle v_i,LGZ_i\rangle_F
 =\langle L^Tv_iZ_i^T,G\rangle_F.
\]

因此完整 99 维坐标以及低秩支持 Jacobian 为

\[
 C_i=\operatorname{vec}(L^Tv_iZ_i^T)^TE,\qquad
 C\in\mathbb R^{n\times99},\qquad J=CB\in\mathbb R^{n\times r}.
\]

冻结网络处于 eval 模式、BN 统计固定且试次没有批间交互时，一次 `autograd.grad(margin.sum(), normalized_x)` 得到全部试次的 \(u_i\)。\(m_0\)、\(C\) 和身份点特征均与 \(B\) 无关，可以 detach，外层对**固定切线学习器**的梯度仍然完整地通过 \(CB\) 和后续求解传播；无需经过 EEGNet 的内层二阶导数展开。

这项恒等式精确的是身份点一阶导数。将其用于非零空间调整的整个支持拟合，是明确的局部模型近似。训练后网络线性化已有先例，候选贡献须落在 CAR/RMS/Lie 结构、头部竞争和前缀预算的具体结合上。[Maddox 等，AISTATS 2021](https://proceedings.mlr.press/v130/maddox21a.html)

## 5. 固定凸代理、Schur 消元与端点约束

### 5.1 支持目标及公平先验

设 \(H_0=[h_0(x_i)^T,1]_{i=1}^n\)，\(n\le20\)，有两类支持时 \(q_i=1/(2n_{y_i})\)、\(\sum_iq_i=1\)。固定支持替代目标为

\[
 \min_{\|\theta\|\le\rho_S}
 \sum_iq_i\{\operatorname{softplus}(\tilde m_i)-y_i\tilde m_i\}
 +\lambda_\theta\|\theta\|^2
 +\frac{\lambda_h}{p+1}\|a\|^2,
 \quad\tilde m=m_0+J\theta+H_0a.
\]

主点为 \(r=4\)、\(\rho=.15\)、\(\lambda_\theta=.01\)、**\(\lambda_h=.1\)**。所有主要空间优化器、头单独对照和信息/保护组件在目标会话拟合时使用相同先验；不能将头正则加大带来的变化解释成新曲率方法的收益。另保留明确命名的 `original_sgd10`，它使用旧 \(\lambda_h=.001\)，只作原算法参考，不和 `.1` 主基线合并。

正则同时覆盖头权重和截距，防止少样本可分时解发散。先验曲率为 \(2\lambda_\theta I\) 与 \(2\lambda_h/(p+1)I\)，遗漏因子 2 或维数除数会改变模型。正的两类先验使固定凸代理存在唯一约束最小值，但代码只运行固定迭代预算，并未宣称所有会话都已精确求到该值；更不能由此推导真实非线性网络问题的全局最优性。

### 5.2 正确的加权 GGN 与局部头消元

记 \(p_i=\sigma(\tilde m_i)\)，\(D_{ii}=q_ip_i(1-p_i)\)。数据曲率为

\[
 F=[J,H_0]^TD[J,H_0].
\]

对固定线性 logits 的逻辑损失，这是精确数据 Hessian，也可称所定义设计权重下的输出 Fisher/GGN。由于 \(q_i\) 按已知支持类别固定，不把它含混为未知目标总体分布的“真实 Fisher”；也不把使用当前真标签的梯度外积称为同一矩阵。[Martens 2020](https://jmlr.org/papers/v21/17-678.html)、[Kunstner 等 2019](https://papers.nips.cc/paper_files/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html)

令 \(g_\theta,g_a\) 为当前目标梯度，\(K=F+\operatorname{diag}(2\lambda_\theta I,2\lambda_h I/(p+1))+\mu I\)，\(\mu=10^{-4}\) 为匹配数值阻尼。分块消元：

\[
 S=K_{\theta\theta}-K_{\theta a}K_{aa}^{-1}K_{a\theta},\qquad
 \bar g=g_\theta-K_{\theta a}K_{aa}^{-1}g_a,
\]
\[
 \Delta a=-K_{aa}^{-1}(g_a+K_{a\theta}\Delta\theta).
\]

没有把头求到驻点就不能令 \(g_a=0\)。这里精确消掉的是**当前二次子问题**的头；ridge-logistic 原问题没有一般闭式头解。经典变量投影和可微凸头不是新发明。[Golub & Pereyra 1973](https://doi.org/10.1137/0710036)、[Bertinetto 等，ICLR 2019](https://www.robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/)、[Lee 等，MetaOptNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html)

### 5.3 约束新的参数端点，并重新计算条件头

正确的空间约束是 \(\|\theta+\Delta\theta\|\le\rho_S\)，不是仅约束 \(\|\Delta\theta\|\)。对 \(t=\theta+\Delta\theta\) 解

\[
 \min_{\|t\|\le\rho_S}\tfrac12 t^TSt-b^Tt,
 \quad b=S\theta-\bar g,
 \quad (S+\nu I)t=b,
\]
\[
 \nu\ge0,\quad\|t\|\le\rho_S,\quad\nu(\|t\|-\rho_S)=0.
\]

内部时 \(\nu=0\)；边界时一维二分确定乘子。然后使用实际受约束的 \(\Delta\theta\) 在上述公式中重新求头更新。每步再沿旧点至候选端点回溯，检查固定凸代理目标不增加，最多 16 个比例；失败则保留旧点，不能删除会话。主代理更新 10 步，敏感性覆盖 1/3/20 步。

`tangent_joint` 对照先解完整联合无约束系统，再只裁剪空间参数，沿用原头步；`tangent_profiled` 解受约束二次问题并重新计算头。在球不活跃时，Schur 解与完整联合解是同一个解，数值测试必须一致。**纯消元不是独立模型，也不是准确率机制；候选机制在约束、先验、信息半径和学习器匹配。** 小维信赖域求解本身也有经典基础。[Moré & Sorensen 1983](https://doi.org/10.1137/0904038)

在边界，\(A=S+\nu I\)，对解的输出梯度 \(z\) 计算

\[
 v=\frac{t^TA^{-1}z}{t^TA^{-1}t},\quad u=A^{-1}(z-vt),\quad
 \nabla_b\ell=u,\quad\nabla_S\ell=-\operatorname{sym}(ut^T),\quad
 \frac{\partial\ell}{\partial\rho_S}=\rho_Sv.
\]

自定义一阶 KKT 反向包含半径导数；不能将二分得到的乘子当常数而遗漏边界运动。活跃集、回溯和保护比例发生切换时学习器仅分段可微，代码不承诺高阶边界导数或对精确最优解的完整隐式梯度。有限迭代算法梯度与精确内层最优解梯度应区分。[Rajeswaran 等，隐式元梯度](https://papers.neurips.cc/paper_files/paper/2019/hash/072b030ba126b2f4b2374f342be9ed44-Abstract.html)

## 6. 头部竞争后的可观测信息与半径收缩

只看 \(J^TD_0J\) 会把可被头补偿的方向也计为空间证据。在身份点以源概率形成 \(D_0\)，定义不含空间先验和数值阻尼的惩罚头 Schur 数据矩阵：

\[
 \mathcal I=J^TD_0J-J^TD_0H_0
 (H_0^TD_0H_0+\eta_hI)^{-1}H_0^TD_0J,\qquad
 \eta_h=2\lambda_h/(p+1).
\]

精确算术下 \(0\preceq\mathcal I\preceq J^TD_0J\)。当 \(H_0\) 满行秩且 \(\eta_h\to0\)，头能够插值所有支持 logits，\(\mathcal I\to0\)。本任务的 65 维头面对最多 20 个标签，这个退化风险不能靠增加空间秩自动解决。

因此它应称**给定先验后的局部惩罚可观测信息**，不是生理空间参数的严格可辨识性定理。严格正定可能来自先验，不代表数据证明了唯一真实空间变换。RMS 已移除试次整体正幅度，分类器还可能对别的生成元不敏感；各向同性协方差也无法单独识别纯旋转。

信息半径采用

\[
 d_{eff}=\operatorname{tr}\{\mathcal I(\mathcal I+\tau I)^{-1}\}
 =\sum_j\frac{s_j}{s_j+\tau},\qquad
 \tau=2\lambda_\theta c_I,\qquad
 \rho_S=\rho\sqrt{d_{eff}/r}.
\]


主 \(c_I=1\)，敏感性为 `.1` 与 `10`；它只改变信息尺度，不改变目标中的空间先验。\(0\le d_{eff}\le r\)，故 \(\rho_S\le\rho\)。半径由身份点一次确定，在整个适应过程固定。通过矩阵求解和迹求 \(d_{eff}\)，特征值仅作诊断，避免对重根特征向量求导。全 Schur、谱函数、各向同性先验和球在 \(B\mapsto BO,\theta\mapsto O^T\theta\) 的正交换基下等变；一般逐坐标对角 Fisher 没有这种性质。

小 Fisher 可能来自**自信但错误**的源模型，不能解释成“不需要适应”。头先验太弱也可能让全部半径坍缩。固定设计权重和为 1，信息量不会自动随支持数线性增长，不保证预算曲线单调。当前信息规则是局部证据代理，未给出校准后的贝叶斯后验、覆盖率或人口风险置信度。线性化低秩不确定性与 Fisher 引导子空间已有相近工作；LUMA 的精确观测模型为回归，不能把其结论直接移到这里的分类代理。[LUMA，AISTATS 2026](https://proceedings.mlr.press/v300/park26b.html)、[Feng 等，Fisher 子空间 LoRA，ICML 2026](https://proceedings.mlr.press/v306/feng26h.html)

若信息方案优于统一半径，只能先支持“该信息收缩规则有用”。目前敏感性给出固定半径 `.05/.15/.35`，没有依据内部留出平均半径反向构造完全同均值的收缩对照；因此不夸大为已彻底排除平均收缩强度的解释。

## 7. 在实际支持预测上保护源模型

`tangent_protected` 和 `information_protected` 得到候选 \((\hat\theta,\hat a)\) 后，沿**身份点至整个候选更新**取 \(\alpha\in\{1,1/2,\ldots,0\}\)：

\[
 (\theta,a)=\alpha(\hat\theta,\hat a),\qquad
 \sum_iq_i\operatorname{KL}\{\operatorname{Bern}(p_{0i})\|\operatorname{Bern}(p_{\theta,a}(x_i))\}
 \le\epsilon.
\]

主 \(\epsilon=.01\)，敏感性为 `.002/.05`。每个比例都用真实的 \(\exp M\)、RMS 和变换后特征复核，而非只检查切线预测。源概率固定；不读取未来查询统计。它限制最终偏离源模型的支持 KL，不能以“每步对上一迭代的小 KL”替代。

这个后验收缩规则不宣称解出了最优联合 KL 约束问题。目标正则 Schur 不自动等于预测 KL 度量；预测约束依赖头时，不能无约束消掉头再声称解决了完整联合约束。接受的离散比例在局部外层梯度中视为常数。

在被检查的支持分布上，Pinsker 与 Jensen 给出 \(\sum_iq_i|p_{new,i}-p_{0i}|\le\sqrt{\epsilon/2}\)。它控制的是支持平均概率变化，不保证每点 argmax 不变，更不保证未来零负翻转。源模型系统性错误时，保护可能直接阻碍纠错。预测保持、蒸馏和显式 churn 约束均已有研究，不能把“惩罚改约束”单独包装成新理论。[Yan 等，PCT](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html)、[ELODI](https://pubmed.ncbi.nlm.nih.gov/38652615/)、[Jiang 等，Churn Reduction via Distillation](https://research.google/pubs/churn-reduction-via-distillation/)

## 8. 联合字典元学习与模型选择

定义固定预算学习器 \(\mathcal A_B(S)\) 为前述有限步支持校准算法。每个外层更新先按参与者、再按其会话均匀抽 4 个 episode，以实际非线性查询预测最小化

\[
 \mathcal L_{meta}(B)=\frac14\sum_{e=1}^4
 \mathcal L_{balanced}\left(f_{B,\mathcal A_B(S_e)}(X_{Q_e}),y_{Q_e}\right),
 \qquad B^TB=I_r.
\]

逐 episode 累积一阶外层梯度并释放图。对 \(G=\nabla_B\mathcal L\)，使用 \(G_T=G-B\operatorname{sym}(B^TG)\)，全局梯度范数裁剪至 1，以 `.01` 步长更新并做正对角 QR 回缩。每种字典最多 1000 次外层更新，每 50 步在元验证会话评价；按“参与者等权 BA 最高 → BCE 最低 → 步数更早”选择 checkpoint。固定 `selected.json` 后才允许相应内部留出评价。

字典家族分别为：`legacy`（原 SGD10，训练头先验 `.001`）、`matched_sgd`（SGD10，训练头先验 `.1`）、`profiled`（新 profiled，`.1`）、`information`（新信息半径，`.1`）、`protected`（信息+真实 KL 保护，`.1`）。`matched_sgd` 与三个新家族使用相同先验、种子初始化、episode 序列、更新预算和验证选择规则。旧字典在来源、种子、秩、输入、更新预算和选择记录完整匹配时复用，否则独立训练。复用不等于重新选择旧留出结果。

固定 `legacy` 字典上的主要优化器比较保持同一个 B，隔离目标会话适应变化。主要字典比较改为新家族对 `matched_sgd`，在同一头先验 `.1` 下检验“与新校准学习器匹配的元训练目标是否有额外价值”。新字典与旧 `legacy` 的交叉差异仍保留，但只作描述：两者训练先验不同，不能把这种差异全部归于新字典目标。相同外层更新数控制训练样本预算，不意味着不同内求解器耗费严格相同算力。低秩元学习与任务适应子空间本身已有先例。[Jiang 等，Subspace Learning for Effective Meta-Learning](https://proceedings.mlr.press/v162/jiang22b.html)

`relinearized` 每步在真实网络重新计算空间 Jacobian/GGN，并复核真实支持目标，只是固定字典的较贵后验参考；代码拒绝把它当成已经支持高阶展开的外层训练方法。不得称所有比较都优化了同一个凸模型，或所有学习器都已元训练。

## 9. 固定实验矩阵：174 个逻辑任务

| 家族 | 组成 | 任务数 | 是否元训练 |
|---|---|---:|---|
| 旧字典 | `legacy` × r2/r4/r8 × 3 种子 | 9 | 可复用完整匹配结果，否则训练 |
| 匹配先验字典 | `matched_sgd` × r4 × 3 种子 | 3 | 各自 1000 次外层更新 |
| 新字典 | profiled/information/protected × r4 × 3 种子 | 9 | 各自 1000 次外层更新 |
| 旧字典核心评价 | 11 个适应器 + 独立 original_sgd10 × 3 种子 | 36 | 否 |
| 新字典交叉评价 | 3 新字典 × 4 适应器 × 3 种子 | 36 | 否 |
| 匹配先验字典评价 | profiled、信息+保护 × 3 种子 | 6 | 否 |
| 敏感性评价 | 下表设置 | 72 | 否，固定相应 legacy 字典 |
| 支持 EA 参考 | 3 个源模型种子 | 3 | 否 |
| **合计** | **21 basis +78 core +72 sensitivity +3 reference** | **174** | **最多 21 个字典 fit** |

若所有字典均需重新训练，上限为 21×1000=**21,000 次外层更新**，每次 4 个 episode。旧字典中最多 9 项可能复用；实际节省量由完成记录与输入身份决定，不能预先当成已经省下。核心、敏感性和参考任务只做目标校准评价，不重复元训练。

11 个适应器为 `sgd10`、`sgd100`、`head_sgd`、`head_newton`、`tangent_diag`、`tangent_joint`、`tangent_profiled`、`tangent_information`、`tangent_protected`、`information_protected`、`relinearized`。`sgd100` 用 100 步真实非线性 SGD，属于较多更新预算对照，只有实际计时相近时才可称近似等算力。`head_newton` 对固定特征上的逻辑头充分迭代，检验新结果是否只是头拟合得更好。

3 种新字典上的 4 个交叉适应器固定为 `sgd10`、`tangent_profiled`、`tangent_information`、`information_protected`；匹配先验 SGD 字典只加 `tangent_profiled`、`information_protected` 两组，以服务两个主要字典比较，不扩大为全笛卡尔积。所有 78 个核心评价覆盖五个预算，清洁设置共 390 个评价坐标，而不是额外的 390 次字典训练。支持 EA 只用支持协方差做 5% 收缩和源参考重着色，不拟合标签头，也不宣称完整复现原 EA 论文。

| 敏感性轴 | 非主设置 | 方法 | 数量 |
|---|---|---|---:|
| 秩/半径 | (r2,.15)、(r8,.15)、(r4,.05)、(r4,.35) | profiled、信息+保护，各 3 种子 | 24 |
| 头先验 | `.001/.01/1` | 同上，各 3 种子 | 18 |
| 代理迭代数 | `1/3/20` | 同上，各 3 种子 | 18 |
| 实际 KL 预算 | `.002/.05` | 信息+保护，各 3 种子 | 6 |
| 信息尺度 \(c_I\) | `.1/10` | 信息+保护，各 3 种子 | 6 |
| **合计** | **均只评价 B20，清洁条件** | | **72** |

这些是围绕固定主点的单轴分析，不在内部留出选择“最优”点。头先验敏感性中的两种空间方法使用同一先验，可以描述保护/信息规则的条件变化；`.01/1` 下并没有完整额外的匹配 SGD 元训练矩阵，因此不夸大成已经彻底分离所有正则与优化效应。秩敏感性使用对应秩的 legacy 字典；半径、头先验和迭代敏感性不重新元训练 \(B\)，代表对已选字典的目标校准变化。

## 10. 一次运行保留的机制、误差和压力证据

逐会话保存有效支持数、类别计数、回退、空间范数、有效半径、Schur 谱和有效维数、代理/实际支持损失、代理与实际差分 logit RMSE/最大误差、头范数、回溯比例、真实源支持 KL、保护比例及校准耗时。代理误差随半径增大而恶化，是可能导致方法失败的证据，不得隐藏。

冻结已选字典后，在 legacy 字典的 SGD10、头 Newton、profiled、information、信息+保护，以及 protected 新字典的信息+保护上固定执行六种 B20 压力条件：支持标签翻转 10%/20%；校准之后的查询 CAR 对称增益漂移，生成元范数 `.10/.25`；查询反对称旋转漂移，生成元范数 `.10/.25`。各方法/种子共享同一会话的方向或标签排列，两档共享方向。

标签压力只改支持标签，未来真标签不变。空间压力只改校准之后的查询信号；支持不变。生成元 Frobenius 范数是无量纲 log 变换强度，不是“电压增加 10%”或“旋转 10 度”。它检验校准后未知漂移的鲁棒性，不能称算法观察漂移后主动追踪，也不能当作自然误标率或真实电极变化分布。压力数据独立生成，不覆盖缓存。

平均准确率之外同时报告：相对同一源模型的平衡负翻转、纠错、参与者退步比例、平衡 NLL/Brier、固定 10 箱平衡 ECE 和校准时间。ECE 小样本估计有噪声；本轮没有延续 V2 的运行段尾部训练目标，不能把先前的 temporal 方法混入本轮机制结论。

负翻转和纠错均以全部查询的类别平衡权重为分母：前者统计“源判对、新方法判错”，后者统计“源判错、新方法判对”；不把它们含混成只在源判对/判错子集内的条件率。因此同一查询集上有精确核算关系 `BA_new − BA_f0 = corrected_error_rate − negative_flip_rate`，可直接检查所谓保护是否以牺牲纠错换来。

## 11. 十项主要比较、统计单位与解释边界

主终点为 **B20、clean、内部留出参与者等权平均 BA**。命名 `字典:适应器` 中 `legacy` 表示旧字典，除独立 `original_sgd10` 外，下列主要目标校准均为头先验 `.1`。固定 10 项比较为：

| # | 候选 − 对照 | 要回答的问题 |
|---:|---|---|
| 1 | legacy:profiled − legacy:SGD10 | 代理约束校准是否优于相同字典/先验的原更新 |
| 2 | legacy:profiled − legacy:SGD100 | 是否只是 10 步欠优化 |
| 3 | legacy:profiled − legacy:head_newton | 空间调整是否提供头之外的收益 |
| 4 | legacy:profiled − legacy:tangent_diag | 全交叉曲率是否有价值 |
| 5 | legacy:profiled − legacy:tangent_joint | 约束激活后的端点/条件头处理是否有价值 |
| 6 | legacy:information − legacy:profiled | 支持局部信息决定半径是否有价值 |
| 7 | legacy:tangent_protected − legacy:profiled | 真实支持 KL 保护是否改变精度 |
| 8 | legacy:information_protected − legacy:information | 信息收缩之上保护是否有增量 |
| 9 | profiled:profiled − matched_sgd:profiled | 相同训练先验下，匹配新学习器的字典目标是否有增量 |
| 10 | protected:information_protected − matched_sgd:information_protected | 相同训练先验下，组合方法的字典目标是否有增量 |

表中简写与代码精确名称对应：profiled=`tangent_profiled`、information=`tangent_information`、SGD10/100=`sgd10/sgd100`。主要 family 为 `plan.py::COMPARISONS` 的 10 对；information 新字典等其他交叉对照仍保留为描述性证据，不临时挤入 family。

对参与者 \(u\)、种子 \(s\)、同一会话 \(j\) 的方法差异，先在每个参与者/种子内对相同会话平均，再对三个种子等权平均，得到一个参与者差值 \(d_u\)，最后对参与者等权平均。查询 UID 和真标签必须完全一致；预算曲线也要复用同一 B20 查询。**会话、试次和 3 个种子都不是额外独立参与者**。更多试次可以降低会话测量噪声，不能把内部留出的 11 人改写成数百独立样本。

报告绝对 BA 差值“百分点”、20,000 次参与者配对 bootstrap 的 95% CI、双侧配对符号翻转 p 和整组 10 项 Holm 校正 p。11 人时可枚举 \(2^{11}\) 个符号组合；符号翻转依赖零假设下参与者差值符号对称，不具有临床随机试验的随机化保证。bootstrap CI 为各比较的未同时校正区间，不能当成 family 同时覆盖区间。相对误差下降单独写作 \((BA_{new}-BA_{base})/(1-BA_{base})\)，不把 +2 个百分点含混写成“提升 2%”。

只有上述 10 对涉及的组都完成三个种子且参与者/会话网格匹配后，才输出整组 Holm 推断；不必等待无关敏感性任务，但部分 family 未齐时只输出描述，不连续窥看 p。敏感性、短预算、压力和可选外部评价均只报告差值与个别描述性区间，不新增未经控制的显著性声明。

参与者数有限，小收益可能无法检出。主终点无改善不能靠挑某个压力、某个种子、某个预算或某个概率指标宣称全面有效。报告所有完成/失败/回退与有效配对覆盖，不以难处理参与者或负结果会话为由静默删除。

## 12. 可选外部迁移：Weibo 发布顺序，不能冒充物理采集前缀确认

Weibo2014 / Yi2014 / NEMAR nm000146 是同一来源的别名。原论文为 10 名健康参与者、每人一次会话、7 类简单/复合运动想象与休息；原数据 DOI 为 10.7910/DVN/27306。它不是已用作源库的 Cho2017/GigaDB100295。最终仍需以别名和原文件哈希检查是否与已有输入重叠，不能仅凭注册表未列出就宣称整个项目绝无历史接触。[Yi 等，原论文与数据声明](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0114853)、[原始 Dataverse](https://doi.org/10.7910/DVN/27306)、[NEMAR 固定版本](https://www.nemar.org/dataset/nm000146?v=v1.0.3)

可选适配只取左右手子任务，保持 MAT 发布行序及原 trial 索引，每人取最前 20 条已发布 LR 条目；9 人 160 条 LR、第 6 人 140 条，共 1580 条 LR。已查元数据的发布顺序 LR20 均为 10L+10R。它**不是整场开始后只采集 20 条所有任务试次**，也未核实为真实物理时钟顺序。

原数据为 200 Hz、8 秒提供方预处理 epoch，MI 起点为第 3 秒。按既有源模型合同取 MI 0–1.5 秒上下文、裁剪 .25–1.25 秒，并转换为 11 通道、128 Hz、8–30 Hz、CAR 与 RMS。原 MAT 不嵌入通道名和单位字段，通道映射/微伏转伏依据原维护方解析器，需保留为可追溯假设而非独立物理单位校准。提供方更早的预处理不能逆转。

真实 run 边界、跨 trial 采集时钟、80 条休息的位置和第 6 人缺失区段尚不完整；BIDS 的合成拼接 onset 不能当真实采集时间。外部 run 记 null，不制造时间窗或伪物理 run，不用于真实未来运行段尾部结论。因此本外测应称**发布顺序下左右手子任务的描述性外部稳健性分析**，不能单凭它证明严格采集前缀协议在独立队列得到确认。

可选计划不更新任何源模型、字典、超参数或阈值。固定 legacy 字典的 SGD10、头 Newton、profiled、information、信息+保护，以及 protected 新字典的信息+保护，6 组×3 种子=**18 个评价任务**，均覆盖五预算、同一 LR20 后的发布顺序查询。统计单位为 10 名参与者，三种子仍先平均，不变成 30 名外部个体；只做描述性配对分析。

固定版本原 MAT 共约 4.34 GB，建议只下载原 MAT 和小元数据，避免与完整 6.06 GB 衍生版本重复保存。可选数据失败不阻塞核心 174 任务，不能自动改用另一个已曝光数据集。SPDIM 已直接研究 EEG 目标域小几何参数适应，需据任务数据使用边界比较，而非声称首次冻结特征后进行几何适应。[SPDIM，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc6cd575b8340be6ee70beaed9ca4015-Abstract-Conference.html)

## 13. 实现资源、验证与预期论文主张

学校配置为最多同时 3 个独立单 GPU 任务，每个 **A100 PCIe 40 GB、12 CPU、64 GB RAM**，不要求同节点；小矩阵算子线程从 4 开始。使用现有 Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6 环境，不在作业中安装依赖。旧训练继续保留，新目录断点与评价分开。具体提交、监控、续跑和可选数据命令见 README/RUNBOOK，不把操作细节作为科研贡献。

几何、切线和小矩阵使用 float64；冻结 EEGNet保留源 float32，无 AMP。端点球只将至多 8×8 系统传至 CPU，NumPy 一次谱分解后进行标量二分，再复制回输入设备；一阶 KKT 反向在原设备执行。这样避免每次二分都同步 GPU 小算子，但不预先承诺真实速度提升。校准计时包括输入导数、求解、回溯、KL 复核和 CPU/GPU 小量传输，排除未来查询推理，并按 CUDA 完成时点计时。报告学校实测内存峰值和耗时，不能凭参数少就保证全部负载一定低于 40 GB。

已在精确版本 CPU 环境对实际 CompactEEGNet 11×128 的合成输入验证：解析切线对直接 autograd、旧 SGD10 数值一致、Schur/全系统内部一致、球端点及半径 KKT 导数、完整信息半径外层梯度、四 episode 累积与 Stiefel 更新、正交换基等变、头插值信息退化、真实支持 KL、回退和各适应器有限性。它们验证软件，不验证 EEG 提升、学校 GPU 性能或统计功效；最终测试数和证据文件以包内验证记录为准。

可写入论文的性质应分层：解析切线和 Schur 是代数恒等式；固定替代问题有严格凸结构；实际网络使用局部近似；KL 只限制已观察支持平均概率变化；信息半径是受先验影响的局部证据规则。可能的组合贡献是**在固定采集前缀预算下，将完整 CAR/Lie 身份切空间、头竞争后的信息、约束边界梯度和实际预测保护连接到可训练字典，并用严格对照说明收益和失败来自哪里**。自然梯度、GGN、Schur、凸头、低秩元学习和预测蒸馏不应被重新命名为新发明，尚无穷尽式查新或成功发表保证。

更完整文献及外部数据来源、排除理由见 [LITERATURE_AND_DATA_REVIEW.md](LITERATURE_AND_DATA_REVIEW.md)；详细推导审查见 [math_review.md](math_review.md)。冻结方法之后仍需要具有真实采集顺序、可比通道/任务且未参与开发的独立队列，才能更强地检验临床或真实在线情境下的外推；当前离线内部与发布顺序外部证据不能替代这种验证。
