# 第三篇论文：空间适应方向学习的实施规格

版本：2026-10-18。状态：研究方案与实现契约，尚无本方案的训练结果。本文只定义一条主算法；数据入口、最终窗口、统计推断和运行预算由同目录的数据及执行契约统一冻结。

## 1. 主问题、主张与研究边界

主问题是：**在只采集 20 个目标会话校准试次时，能否利用其他受试者的纵向记录，预先学到值得调整的少数空间变化方向，使新会话的校准优于仅更新分类器及自由空间适应？**

主算法在固定 EEG 编码器之前加入一个参考方式一致、可逆、幅度受限的空间变换。源域纵向元训练只学习变换方向；目标会话只拟合少量方向系数和分类头增量。主实验没有候选门控、序贯审计或额外 CVaR 目标。4/8/12/16/20 个采集试次的预算曲线是次级实验。

以下内容已有明确先例，不能作为“首次”贡献：PCT 的负向翻转正则；LoRA/EEG 参数高效微调；冻结预训练骨干再做元迁移；元学习低维适应子空间。最接近的通用先例是 MUSML（ICML 2022），其外层学习子空间基、内层学习线性组合系数；必须正面比较。[M07 M09 M10]

可检验的新假设收紧为：**跨完整记录段的未来预测目标，可以学到适用于少试次 EEG 校准的物理空间变化方向；在共同平均参考的有效空间中约束这些变化，有助于控制无意义的空间变形。** 这是待验证的研究假设，不是已证实结论。

## 2. 数据角色与可见性

| 对象 | 可以使用的数据 | 不能使用的数据 |
|---|---|---|
| 基础模型 f0、源参考协方差 Rref | 四个稳定源库的预训练训练 split，名单和 event_uid 由预训练 manifest 固定 | Stieger 元验证/内部测试，任何最终外测 query |
| 生成元字典 G | Stieger 元训练组的 40 名受试者；真实时间顺序的 support → 后续完整 run query | 元验证 11 人的梯度、内部测试 11 人、外部队列 |
| 全局超参数与训练轮次选择 | Stieger 元验证 11 人 | 内部测试及最终外测 |
| 目标会话系数和分类头增量 | 该会话真实顺序前 20 个已采集校准试次中通过预先固定质量规则的输入和标签 | 任何后续 query 的信号统计、标签、准确率或最优轮次 |
| 最终性能计算 | 参数冻结且预测落盘后，由独立评估入口读取 query 标签 | 不允许返回训练或选择入口 |

40/11/11 是拟定参与者分组数，须以原始数据资格审计及最终 manifest 为准；开发不能根据效果重新分组。源编码器及其初始分类头在 G 的元训练中均不更新。每个目标会话重新从 f0、零系数、零头增量开始；主实验不继承该受试者上一会话的适应状态。

### 2.1 时间与采集成本

算法输入必须携带 `protocol_freeze_sha`，并使用统一的 MI 指令/反馈开始锚、滤波上下文及分类窗口。当前设计拟在 Stieger 使用 cue + 2 s 的 feedback/MI 开始锚，统一窗口仍须通过开发组覆盖率审计后冻结；不得把旧 v1 的 cue 窗口静默沿用。窗口改变时，受影响的 f0、Rref、字典和全部对照一并重训。

主预算为真实采集的 20 个 LR-MI 试次，而非筛选后 20 个好试次、20 个窗口或 10 个/类。短试次和预先定义的坏试次仍计入采集负担；不得补采到 20 个可用样本后仍记作 20。有效 support 缺失任一类别时，返回 f0 并记录 `insufficient_class_support`，不删除该会话。数据契约另报告短试次覆盖率、类别差异和被排除 query 的范围。

相邻滑窗及 Wang 的 Copy/New 成对试次不能计作独立受试者。主算法使用每个事件的一份固定 trial 输入；若扩展为多窗，必须先在 trial 内汇总损失，仍只记一个事件。

## 3. 唯一约定：Q 是 10 × 11 的行正交矩阵

主通道集合有 C=11 个通道，d=C−1=10。令

\[
H=I_C-\frac{1}{C}\mathbf1\mathbf1^\top,
\qquad Q\in\mathbb R^{d\times C},
\qquad QQ^\top=I_d,\quad Q^\top Q=H.
\]

Q 使用固定通道顺序下的确定性 Helmert 基，并保存实际数组及 SHA256；不能只保存“Helmert”字符串。X 是上游完成共同平均参考、统一滤波/重采样/裁窗，并按通道去除时间均值之后、联合 RMS 归一化之前的 C×T 试次。理论上 HX=X。

若旧 f0 的管线没有同一 11 通道 CAR，零系数只会恢复 HX，不能恢复旧的未参考输入；该旧 checkpoint 不可无改动复用。已恢复的 v1 管线在选定 11 通道后做 CAR、在模型入口做按通道时间去均值和联合 RMS，可在完整协议 SHA 一致时做身份重放验证。

## 4. 固定源参考协方差

仅在四个 f0 预训练训练 split 上构造 Rref。对每个非退化 trial：

\[
C_i=\frac{QX_iX_i^\top Q^\top}{\operatorname{tr}(QX_iX_i^\top Q^\top)}.
\]

依次等权平均 dataset、subject、该 subject 的 trial，得到 \(\bar C\)。采用固定收缩，不增加调参维度：

\[
R_{\mathrm{ref}}=0.95\bar C+0.05I_d/d.
\]

用 float64 对称特征分解求 Rref^(1/2) 和 Rref^(−1/2)，并保存来源 event_uid 集合摘要、等权方式、收缩参数和矩阵数组。白化只用于定义变换坐标；不会作为另一个无条件预处理改变 f0 的输入。主方法不从目标 query 估计协方差，也不在目标阶段更新 Rref。

## 5. 空间变换族及结构保证

学习 r 个矩阵 \(G_k\in\mathbb R^{d\times d}\)，满足

\[
\operatorname{tr}(G_k)=0,
\qquad \langle G_k,G_l\rangle_F=\delta_{kl}.
\]

目标系数为 \(\theta\in\mathbb R^r\)，定义

\[
M_\theta=\sum_{k=1}^r\theta_kG_k,
\qquad \|\theta\|_2\le\rho,
\qquad T_\theta=\exp(M_\theta).
\]

作用于原始 11 通道空间的算子固定为

\[
\boxed{A_\theta=Q^\top R_{\mathrm{ref}}^{1/2}
\exp(M_\theta)R_{\mathrm{ref}}^{-1/2}Q.}
\]

之后使用原有联合 trial-RMS：

\[
\mathcal N(Y)=Y/\sqrt{\|Y\|_F^2/(CT)}.
\]

先 Aθ 再 N，不允许在输入适应后再次做逐通道标准差归一化，否则会移除算法试图学习的相对增益。RMS 的求和与平方根用 float64；对每个有限、非零能量 trial 直接除以其真实 RMS，不设 epsilon 截断或幅值下限。全零和非有限值由预先定义的质量规则直接处理。这样对任意共同正增益 a，N(aY)=N(Y)；绝对单位未知但共同增益可证的数据才可使用这一路径。

### 5.1 可直接证明的性质

由于字典 Frobenius 正交，\(\|M_\theta\|_F=\|\theta\|_2\)，从而 \(\|M_\theta\|_2\le\rho\)。矩阵指数及其逆满足

\[
e^{-\rho}\le\sigma_{\min}(T_\theta)
\le\sigma_{\max}(T_\theta)\le e^\rho,
\quad \kappa_2(T_\theta)\le e^{2\rho},
\quad \det T_\theta=e^{\operatorname{tr}M_\theta}=1.
\]

当 θ=0 时，A0=Q^TQ=H，因此 N(A0X)=N(X)；这是身份重放的理论依据。对于任意原始参考偏置 a(t)，Aθ(X+1a^T)=AθX；输出仍在 CAR 子空间。

这些性质只约束**白化后的有效 10 维坐标**。11×11 的 Aθ 因 CAR 本来秩为 10、行列式为 0；不能把 det=1 写到 Aθ 上。Rref 的相似变换会改变通常欧氏坐标下的奇异值，不能把上述界直接挪到原始电压坐标。RMS 后的非线性归一化也不保持该体积解释。以上均不推出准确率提升、无新增错误或患者安全保证。

### 5.2 为什么不用仅 SPD 的适应

G 不要求对称，可以同时包含反对称的旋转成分和对称无迹的各向异性增益成分。仅允许对称 G 会限制可表达的变化；必须通过消融检验反对称自由度是否有用，不能先把所有变化解释成真实电极位移。

去掉 trace 是为了排除联合 RMS 下不可辨识的整体正增益。r 指目标需学习的空间系数个数，**不是矩阵更新的秩**；exp(M)−I 通常满秩。span(G) 通常不对交换子闭合，exp(span(G)) 也不对组合闭合，因此本方案不宣称学到 Lie 子群或完整的 Lie 子代数。

## 6. 字典参数化与初始化

构造固定的无迹矩阵正交基 \(E_1,\ldots,E_{99}\)：90 个非对角标准基矩阵，加 9 个由 d 维 Helmert 对比向量放在对角线得到的基矩阵。向量化后记 \(E\in\mathbb R^{100\times99}\)，E^TE=I。

优化变量是 \(B\in\mathbb R^{99\times r}\)，B^TB=I，且

\[
\operatorname{vec}(G_k)=EB_{:,k}.
\]

每个种子生成一个 99×r 标准正态矩阵，薄 QR 得到 B，调整 QR 符号使 R 对角线非负。初始 B 不使用元验证、内部测试或外部数据。每个 episode 的 θ 和分类头增量均初始化为 0；G 不可初始化为全零，否则不满足约束。

字典有基旋转的不唯一性：G 的列做正交旋转、θ 逆向旋转可以产生同一 M。不能把不同种子的“G1”当成同一个神经生理成分。主机制分析比较字典投影矩阵 BB^T、子空间主角、实际拟合的 Mθ/Aθ，以及整个字典的对称/反对称能量比例。

## 7. 冻结编码器与分类头的精确定义

令固定源模型为 \(f_0=h_{\omega_0}\circ\phi\circ\mathcal N\)。φ 的参数、BN running mean/variance 及其他 buffers 全部冻结；固定 `eval()` 以关闭 dropout。仅设置 `requires_grad=False` 不足以冻结 BN 状态。

针对二分类，源头有两组 logit 权重。令其 margin 参数为

\[
w_0=w_{0,1}-w_{0,0},\quad b_0=b_{0,1}-b_{0,0}.
\]

目标只学习 \(\delta w\in\mathbb R^p\) 和 \(\delta b\in\mathbb R\)。对于 \(u_\theta=\phi(\mathcal N(A_\theta X))\)，

\[
m_{\theta,\delta}(X)=(w_0+\delta w)^\top u_\theta+b_0+\delta b,
\quad p(y=1|X)=\operatorname{sigmoid}(m).
\]

这等价于更新一个二分类线性头，消除了 softmax 对共同 logit 平移的不辨识方向。实现时仍调用原始双 logit source head：先算 zbase=hω0(uθ)，再令 δm=δw^T uθ+δb、znew=zbase+(−δm/2,+δm/2)，最后 m=znew,1−znew,0。不要改用先合并源头权重再做一次矩阵乘法的不同数值路径，以免零增量重放出现额外舍入差。数学上 sigmoid(m)=softmax(znew)1，BCEWithLogits(m,y) 与二类 cross-entropy 等价；保存同一 znew 导出的概率、标签和 PCT logits。

总目标可训练参数为 r+p+1，不可只报告 r。源 ω0 和 φ 的 checkpoint 始终不变；每个目标会话复制初始化并拟合独立增量。保存的概率统一由 `softmax(znew)` 得到，预测统一由 `argmax(znew)` 得到，恰好相等时取类别 0。它在精确算术下等价于阈值 0.5；不另用一次低精度 sigmoid 舍入后的比较改变近零 margin 的标签，也不在 query 上调阈值。

## 8. 内层优化：固定 10 步，不用 query 早停

对有效 support \(S\)，先按真实类别计算类平衡 BCE：

\[
L_{\mathrm{bal}}(S)=\frac12\sum_{c=0}^1
\frac1{n_c}\sum_{i\in S:y_i=c}\operatorname{BCE}(m_i,y_i).
\]

主方法内层损失固定为

\[
L_{\mathrm{in}}=
L_{\mathrm{bal}}(S)+0.01\|\theta\|_2^2
+0.001\frac{\|\delta w\|_2^2+\delta b^2}{p+1}.
\]

从 θ=0、δw=0、δb=0 开始，执行 10 次 full-support 梯度下降：θ 学习率 0.05，头增量学习率 0.01，不使用 momentum、Adam 或 target batch statistics。每步后将 θ 投影到半径 ρ 的闭球：

\[
\theta\leftarrow\theta/\max\{1,\|\theta\|_2/\rho\}.
\]

训练顺序固定、无 support 数据增强、无基于 support 训练准确率选择最优轮次。头和 θ 在同一步、同一前向结果上求导并同时更新。第 10 步即最终模型。上述常数是可实施的初始研究规格，不是已验证最优配置；任何开发修改必须在内部测试/外测前以新版本整体冻结，不能以某个外部患者结果为理由修改。

## 9. 外层优化：只学习 G

### 9.1 Episode 构造

等概率抽取 Stieger 元训练 subject，再等概率抽取其合格 session。该 session 前 20 个真实采集 LR 试次定义最大 support 前缀。外层 query 使用该前缀所在最后一个 run 之后的完整 LR run；同一个 run 的剩余 trial 不混到 query 中。

**主元训练的每个 episode 固定使用 20 个采集试次的前缀**，与 `EXPERIMENT_PROTOCOL.md` §6.2 一致。全套 episode manifest 预先生成并保存真实 session/run/trial/event_uid。缺类 support 返回 f0；不得为了产生梯度而按标签重新抽样。将该 episode 的全部合法 future query 原始 trial 合并，整体至少 20 个有效原始 trial、每类至少 5 个；不要求每个单独 run 都达到这两个门槛。整体不满足则按数据契约记录为该协议下不可评价，不能临时随机拆窗口补足。

次级预算曲线将同一个已用 B=20 训练的 G 应用于 {4,8,12,16,20} 嵌套前缀，**query 始终按 20-trial 边界固定**。另行登记的混合预算元训练消融才在这些预算中均匀采样；它不是主 G，不可看到外测后替换主模型。

元训练 outer query 可以用标签求梯度，这是源开发阶段；最终目标 query 严禁如此。其区别必须通过 split 清单、代码入口和权限对象落实，而非仅靠命名。

### 9.2 外层目标和更新

记上节固定内层算子为 \((\theta_e^{10},\delta_e^{10})=\mathcal U(S_e;B)\)。主外层目标只有未来 run 的类平衡损失：

\[
\min_{B^\top B=I_r}\quad
\mathbb E_e\left[L_{\mathrm{bal}}(Q_e;
\theta_e^{10}(B),\delta_e^{10}(B),B)\right].
\]

以 4 个 episode 为一个 meta-batch，1000 次外层更新。对每个 episode，将所有合法 future query 原始 trial 合并后按 §8 的 Lbal 计算：每类等权、类内每 trial 等权，不另加 run 权重。episode 按 subject/session 均匀抽取。某个 run 缺少一类不导致除零；类别分母来自整个 episode 的 query，并受 §9.1 的资格门槛约束。计算完整的 unrolled meta-gradient，保留内层对 B 的依赖；使用 `create_graph=True` 或等价的函数式自动微分。不允许把 detach 内层后得到的近似梯度标作同一算法。

采用无 momentum 的 Stiefel 投影梯度下降，学习率 0.01。设欧氏梯度为 J，

\[
J_T=J-B\operatorname{sym}(B^\top J),\qquad
B\leftarrow\operatorname{qf}(B-0.01J_T).
\]

梯度 Frobenius 范数上限为 1；超出时整体缩放。QR retraction 在每次外层更新后执行，保持同一符号约定。B、Rref、matrix_exp 用 float64；编码器可用 float32，转换仍须保留梯度。首版不使用混合精度，以降低二阶梯度和矩阵运算的数值风险。

每 50 次更新在元验证组按 B=20 的固定协议评价，选择受试者等权 BA 最高的 checkpoint；完全相同的 BA 用较低验证 BCE，再用较早 checkpoint 打破并列。内部测试只在配置冻结后读一次。训练末尾不是把元验证或内部测试重新并入元训练。

### 9.3 有限搜索

只搜索 r∈{2,4,8}、ρ∈{0.05,0.15,0.35}，共 9 个设置。所有设置共用来源、seed 列表、episode manifest 和内层步数。首先用预先指定的开发种子完成 9 设置选择，再对冻结配置重复种子；不得挑选最好种子。训练预算分别记录 f0 拟合、G 的 episode/梯度更新、验证和目标适应，不把它们混称一个训练次数。

### 9.4 数据规模机制：增加人数有没有真实价值

在 Stieger40 元训练身份中，以预先固定的元数据分层、稳定随机顺序建立嵌套 10/20/40 人子集。采用同一个已冻结 r、ρ、f0、Rref、验证组、1000 次 outer update 和 4-episode meta-batch 分别学习 G；不同规模的独立参与者数改变，优化步数不变。若合格元训练不足 40 人，按数据契约预登记约 25%/50%/100% 的嵌套规模并报告整数。

该实验专门检验更多独立纵向参与者能否改善方向学习。保存每个规模实际接触的独立参与者、会话、原始 trial 数及重复抽取次数；不能把更多重复 episode 当更多数据。它与 Yang 加入 f0 预训练的 source 扩展是两个不同问题，不在同一比较中同时改变。规模消融先在开发阶段完成，最终是否外测由冻结清单规定。

## 10. PCT 为已知对照与消融，不是主创新

主方法 λPC=0。附加版本 `GENEXP+PCT` 只在内层增加 PCT 的 focal logit matching，outer 仍为相同未来 CE。先在原始未适应输入上计算 f0 的两类 logits 并 detach；这些值和 old-correct mask 在该 episode 内固定。

\[
L_{\mathrm{PC}}=\frac1{|S|}\sum_{i\in S}
\left[1+5\mathbf1\{f_0(X_i)=y_i\}\right]
\frac12\|z_{\theta,\delta}(X_i)-z_0(X_i)\|_2^2.
\]

附加内层项为 0.1 LPC。计算 zθ,δ 时保留源双 logit 头在变换后特征上的输出，再分别加 −δm/2 和 +δm/2；这样 θ=δ=0 时两个 logits 逐项与源模型一致。不要把 margin matching 的缩放版本静默称作逐项相同的 FD-LM。

`PCT+LoRA` 使用同一 teacher、mask、α=1、β=5、λPC=0.1 和同一目标 support。普通 KD 对照令 β=0，其余相同。这里 α/β 的形式来自 PCT；0.1 是本研究固定的 EEG 实施权重，并不是原文移植即可保证有效的经验值。若开发验证发现损失尺度异常，须在开启外测前统一修订该权重并同步所有 PCT 对照。[M07]

## 11. 必须保留的强对照

| 对照 | 必须保持一致的条件 | 回答的问题 |
|---|---|---|
| f0 / head-only | 相同 f0；head-only 用相同 20 试次、同一头和 10 步 | 空间适应有没有超出分类器校准的价值 |
| Learned additive basis | 独立元训练，同一 r、相同无迹正交字典约束及 episode；T=I+ΣθG | 作用来自学到方向，还是矩阵指数几何 |
| Random-generator exp | 同一 r、ρ、θ/head 步数；固定随机字典不做元学习 | 源纵向方向学习是否必要 |
| Source drift-PCA exp | 只用 Stieger40：类平衡会话协方差的 log、受试者内去均值、去 trace、Frobenius PCA；取前 r 个对称基 | 任务驱动元学习是否优于高方差漂移方向 |
| Symmetric-only / skew-only learned generators | 同一 r、预算、bilevel，仅将 E 限制为相应子空间 | 各向异性增益与旋转自由度是否各有贡献 |
| Full traceless exp | 99 个目标系数，完整无迹 E，源参考及半径相同 | 低维约束是否减少小样本过拟合 |
| Free spatial LoRA | T=I+UV^T，k∈{1,2,4}，一因子非零另一因子零，更新两因子和头；约束 ||UV^T||F≤ρ | 自由低秩适应是否已经足够 |
| Meta-LoRA | 外层学习非零 U0，V0=0 保证 identity；内层更新 U、V 和头；相同 episodes、查询次数及源数据 | 是否只是所有方法都能得到的元学习收益 |
| PCT+LoRA；GENEXP±PCT；普通 KD+LoRA | 同一 teacher/support/正则尺度，记录新增错误与纠错 | 改善是否完全由既有负向翻转正则解释 |
| EA+head、CSP/LDA classifier-only 与 full 更新、tangent-space shrinkage 分类器 | 严格同预算；query 不参与 EA/协方差估计；统一事件划分 | 神经适应是否优于可靠的经典校准 |

自由 LoRA 的空间目标参数数目是 2dk=20k（有分解冗余），本法是 r；**k=r 不代表参数匹配**。真正严格匹配 r 个空间系数的对照是 learned-additive、random-generator、PCA 和相应几何消融。分别报告参数数目、更新秩、累计梯度步和实际耗时；不得把自由 LoRA 写成虚假的“同参数量”对照。

对于 additive，ρ<1 已可给出 I+M 的可逆性和 [1−ρ,1+ρ] 奇异值界，所以仅声称 exp 可逆不构成实证优势。exp 和 additive 必须各自在同一小搜索集合上完成源验证选择，不能只用 exp 的最优超参数压制对照。

MIRepNet、CBraMod 作为后阶段强骨干对照，须先完成权重来源、预训练包含库、单位、通道映射和 token 时长契约。不能默认官方 MIRepNet 的 Dreyer 评估为未见外部数据，因为预训练表已含 Dreyer2023。不同骨干可各自元训练 G；这证明方法的适用范围，不等于同一个 G 自动跨所有骨干迁移。[M03 M04 M08]

## 12. 结果和机制证据

主要结果按冻结统计方案报告受试者等权 BA。保存每个 query event_uid 的 source、head-only、各适应模型预测和 logits。定义每类纠错率 Gc 与新引错率 Hc：

\[
G_c=\frac{\#\{y=c,f_0\ne y,f_{new}=y\}}{N_c},
\quad H_c=\frac{\#\{y=c,f_0=y,f_{new}\ne y\}}{N_c}.
\]

则 ΔBA=(G0−H0+G1−H1)/2。未按类别加权的 G−H 等于普通 accuracy 差，不能冒称 BA 差。另对 head-only 做相同配对分解，检验“新增空间变化”是否有效。

机制分析提前固定为：实际 Mθ 的范数与边界命中率；整体字典对称/反对称能量；跨 seed 子空间稳定性；学到方向与 source drift-PCA 的主角；有效 support 数量与性能关系；旧错纠正和旧对变错的配对图。拓扑图展示的是模型输入变换，不能直接等同脑连接变化或病灶恢复。

可在元训练/元验证数据上增加小规模受控扰动实验：使用预先固定、标签无关的旋转及相对通道增益，比较识别性能和已知变换恢复误差。它只能说明受控机制，不能替代真实新会话或患者数据，也不能把按作者想法构造的扰动当成自然漂移事实。

## 13. 实施顺序与必要验收

1. 冻结通道、锚、上下文、分类窗口、质量规则和参与者角色，生成原始事件 manifest。
2. 复用或重训协议完全一致的四源 f0；从其训练事件构造 Rref。
3. 实现 Q/E/B、matrix_exp、原始坐标变换、joint RMS、函数式 margin head。
4. 实现 inner update，再实现保留完整依赖的 outer update；最后接入真实 source episodes。
5. 先跑身份、几何和小样本梯度验收，再开始 G 元训练；本轮交付不执行训练。
6. 完成必须对照和源验证选择，冻结正式配置；通过预定开发门槛后才解封内部测试与外测。

必须通过的科学验收：

- θ=δ=0 的输入、logits、预测在规定 float32 容差内重放 f0；同时对多个共同参考偏置重放。
- Q、E、B 正交和 G 无迹成立；数值奇异值/行列式符合对应坐标的理论界；不检查 11 维 det=1。
- 非零反对称 G 的变换确能产生旋转；禁止因一次强制对称化把该自由度丢失。
- 小型合成模型上，以有限差分验证一个完整 inner→outer 的 B 方向梯度，而不仅验证一次 matrix_exp 梯度。
- 冻结 encoder 的参数及所有 buffers 前后 hash 一致；源头 ω0 不变。
- 真实 support 和 query 不重叠且 query 在后续完整 run；同一目标会话不因不同方法重新划分。
- 完整运行与断点继续在同实现、同环境、同设备上结果一致；保存 RNG、episode 游标、B 和 fast-weight 状态。
- 主流程不读取目标 query 来计算 Rref、EA、伪标签、早停或选择 r/ρ。

这些是必要科学检查；不要求为简单文档变更或每个张量操作制造重复测试。

## 14. 停止、降级与可证伪标准

- 若身份重放、时间标签或 meta-gradient 检查失败，停止训练并修实现，不解释为方法效果差。
- 若源开发中 learned dictionary 一贯不优于随机/漂移 PCA 的同系数基，不把结果包装成“学会有用空间方向”；优先停止扩大昂贵训练。
- 若仅 head-only 已解释全部增益，或自由/Meta-LoRA/PCT+LoRA 达到同样收益而新方法无明确效率/稳定性优势，降低算法创新主张，不靠更多门控模块弥补。
- 若 exp 与 learned-additive 无可重复差异，保留“方向学习”的证据，删除“矩阵指数带来性能提升”的主张；数学约束仍可如实陈述。
- 若改善只来自丢弃早终止、低 ERD、低表现用户或更长输入窗口，不能归因于适应算法。
- 若健康内部测试有效、患者外测无效，应报告迁移边界；不把患者结果回流后再把同一数据称最终外测。

效应阈值、主要对比、多重性与置信区间由统计规格统一定义，不在实现过程中临时决定。“大提升”是希望检验的结果，不能作为训练前保证。

## 15. 本文引用

M 编号对应 `REFERENCE_METHODS.md`。优先阅读 M07 PCT、M09 MUSML、M10 Meta-Transfer，它们直接限制新颖性表述；M01/M02 是实验组织和反馈学习的参考，不是本离线方法可继承的在线收益。

- M07: https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html
- M09: https://proceedings.mlr.press/v162/jiang22b.html
- M10: https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.html
- 补充直接近邻 GraphAdapter: https://arxiv.org/pdf/2411.16155
- 补充直接近邻 EDoRA: https://arxiv.org/pdf/2412.17818
- 补充 SAP: https://link.springer.com/article/10.1007/s10994-023-06393-y
