# 第七轮：头容量、先验曲率与空间可辨识性的区分

本页定义局部机制诊断与头容量对照，不增加训练前审批或启动条件。必要条件不足只写入结果，不能因此停止任务。实现为`paper3r7/head_capacity.py`。

## 1. 固定支持代理与有效信息

设当前实际有效支持数为n，固定源EEGNet在这些支持输入上的身份特征为F∈R^(n×p)，允许的头投影为U_h∈R^(p×h)。增广头设计矩阵为

\[
Z=[FU_h,\mathbf 1],\qquad
\tilde m=m_0+J\theta+Za,\quad J\in\mathbb R^{n\times r}.
\]

r应为本方法全部目标适应系数的维数；若分别给空间和时间各r个自由参数，应使用总维数2r，而非只计算其中一块。

类平衡模型-logistic曲率为W=diag(q_iσ(m_i)σ(−m_i))。它不是实际标签分数外积组成的empirical Fisher，也不是任意非线性网络的精确Hessian。未正则分类头作为nuisance后，数据部分的局部空间信息为

\[
I_{data}=J^TW^{1/2}(I-P_{W^{1/2}Z})W^{1/2}J.
\]

令n_W为正曲率行数，则

\[
\operatorname{rank}(I_{data})\le
\min\{r,n_W-\operatorname{rank}(W^{1/2}Z)\}.
\]

若Z满行秩且W正，投影P=I，因而I_data=0。对固定线性代理这不只是一个Hessian现象：任意Jθ都在Z的列空间中，所以任意θ的支持logits改变均能由a补偿。数据本身不能区分这些空间参数。

这说明固定支持上的参数解释受头容量影响，不说明高维头预测一定差，也不说明空间校准无用。未来泛化由先验、源表征和任务分布共同决定；不能由支持信息秩推导未来BA排序。

## 2. 头ridge提供曲率，不等于数据已经识别

原始头特征维为p，科学ridge为λ_a||a||²/(p+1)，η=2λ_a/(p+1)。排除求解器damping后，惩罚头的空间Schur量为

\[
S_\eta=J^TWJ-J^TWZ(Z^TWZ+\eta I)^{-1}Z^TWJ.
\]

S_η一般比I_data大；加空间prior后再得到S_η+2λ_θI。即使I_data=0，两种prior也可使总矩阵正定。不能从正定性推断数据已识别真实空间混合。

满行秩Z且η趋0时，

\[
S_\eta=\eta J^T(ZZ^T)^{-1}J+O(\eta^2).
\]

本模块分别输出未正则信息、惩罚头Schur、加空间prior的总Schur，以及头ridge增加的trace。后两者明确标为penalized curvature，不混称真实数据信息。

## 3. 支持预算约束头容量

`full`使用U=I_p。`fixed`使用预定h个方向。`budget`请求

\[
h_{request}=\max\{0,\min(h_{max},n-1-r)\},\quad h_{max}=16.
\]

再限制h不超过p、n−1及中心化支持特征的实际数值秩。`none`或h=0保留一个bias，而非删除所有头参数。

该规则尝试为r个空间方向保留观测残差维数，只是必要维数条件。J可能仍落入头空间，或其残差信息非常弱；因此继续记录残差J奇异值、信息谱和rank。不能把“有足够维数”写成“空间参数已经可辨识”。

尤其B4且r4时，即使全部4位置有效，h_cap=0，bias仍占1维，数据空间秩最多3，必要条件依然不足。B4正常运行并利用预定prior，诊断标不足，不暂停、不要求补采。QC失败后按实际有效n计算h，但失败位置仍消耗付费预算。

## 4. PCA与随机等维对照

PCA只使用当前已获取支持的固定源身份特征，先中心化，再取右奇异向量。它不读取标签，不读取query，不通过目标外层梯度反传该选择。固定源身份特征不依赖待学习字典B，所以detach与该定义一致。

随机方向用固定种子的p×p Gaussian QR正交基，取前h列。各预算使用嵌套的同一随机基；仅h会受支持数值秩影响。相同h的PCA/随机对照分离方向选择与参数数目。full始终返回I，不随机旋转完整头。

PCA重复奇异值子空间切分有普通SVD基选择歧义，不宣称所有数值实现或任意截断秩都给相同轴。固定符号可消除孤立奇异向量的正负选择，但不能消除重根内的任意旋转。

## 5. 固定原始物理头范数先验

U列正交时δw=Ua，故||δw||²=||a||²；bias保持原坐标。原full维数p固定，原头prior定义保持为

\[
\frac{\lambda_a}{p+1}(\|\delta w\|^2+\delta b^2).
\]

若复用core求解器，其内部对当前h维头除以h+1，则传入

\[
\lambda_{a,reduced}=\lambda_a\frac{h+1}{p+1}.
\]

于是实际惩罚与原full头范数完全一致。直接改为λ_a/(h+1)会同时改变先验强度，不能把全部收益归于头容量。实现`effective_head_l2`和projector metadata显式记录该换算。

拟合完成后重构δw_full=Ua；后续真实空间/时间特征仍使用这个full维头。不能用query重新选择U或根据未来效果调整h。

## 6. 数值与结论范围

小矩阵诊断使用detached float64 CPU SVD。rank阈值为max(rows,cols)×eps64×最大奇异值；空间残差阈值以原加权J的尺度定义，避免把投影roundoff当作新信息。满行秩头时数据残差按P=I精确置零。

惩罚Schur通过加权头矩阵的SVD谱η/(σ²+η)计算，避免反复求逆高维且接近奇异的头矩阵；没有加隐含damping或新prior。极端logit使部分曲率在机器精度下归零时，记录有效正曲率行数，不能用原n虚增信息维数。

所有结论限定于所输入的固定身份切线代理。真实非线性特征随θ改变、约束球和有限步求解都可改变完整问题，不能将代理局部信息直接解释成唯一恢复真实电极、脑源或跨会话混合矩阵。

## 7. 文献近邻与潜在任务贡献

- [Cox与Reid，Parameter Orthogonality and Approximate Conditional Inference，JRSSB，1987](https://doi.org/10.1111/j.2517-6161.1987.tb01422.x)：nuisance与参数正交有成熟统计基础，Schur投影不冒称新普遍定理。
- [ANIL，Rapid Learning or Feature Reuse，ICLR2020](https://iclr.cc/virtual_2020/poster_rkgMkCEtPB.html)：少样本只调头已有经典研究。
- [BOIL，Towards Representation Change，ICLR2021](https://openreview.net/pdf/1a8c0d92bdde5a60d7b354b43c36e22abacfc99a.pdf)：冻头更新表征同样已有工作。
- [Martens，New Insights and Perspectives on the Natural Gradient Method，JMLR2020](https://jmlr.org/beta/papers/v21/17-678.html)：Fisher、近似曲率与参数化不变性的区别已有系统分析。

因此仅减少头维、冻结头或使用Fisher矩阵不足以宣称全新算法。值得检验的论文特定问题是：严格付费前缀下，头容量如何吞没可观测空间/时空方向，先验曲率与数据识别怎样分离，以及预算自适应头容量能否在保持原先验定义的情况下改善新会话泛化。
