# 论文3第四轮：用前缀内样本外证据决定如何适应

设计日期：2026-10-24（Asia/Shanghai）。本文件记录第四轮的数学依据、可证伪问题和文献边界。最终计划为**204项核心逻辑任务：21字典、90核心评价、90敏感性评价、3EA参考**；可选Weibo另21项。只有21项涉及字典训练/匹配复用，并非204次源模型训练。本文件不把软件测试或公开论文的结果写成本轮已经取得的识别效果。

## 1. 与第三轮的区别

第三轮研究固定身份切线、分类头的Schur消元、空间信息半径和在同一支持集上的实际源预测KL。它回答“前20条在给定先验下支持多大空间变化”。第四轮进一步回答：**这次变化在支持内未参与拟合的试次上是否有益，是否应该保留源预测、仅调头或启用空间校准？**

这不是增加更多启动门槛。交叉拟合是每个会话内的数值算法：支持不足时自然返回源模型，产生可分析的回退记录，不阻断其他会话或独立任务。旧轮作业、源码和结果目录保持独立。学校资源为图中 `hpc_gpu1` 队列的A100-SXM4-80GB：**一个节点上的一个3GPU作业，内部3个单卡计算槽，每槽16核CPU与64G主机内存**，总请求48核CPU、192G内存。不把一个3GPU分配误写成三个独立Slurm分配；不能仅凭网页分区名称绑定到某个未给出的GPU主机名。

沿用冻结源模型、clinic11、128Hz、相同时间窗、相同参与者划分和种子。目标预算按原始采集位置计数；QC失败仍付费，不能从第21条补入。各短预算共用B20之后同一查询集。目标查询输入和标签均不进入交叉拟合、权重选择或校准；其标签仅评价最终冻结预测。

## 2. 主要方向A：交叉拟合的源锚定收缩

### 2.1 原支持内拟合改善为何不够

设源正类概率为 `p0(x)`，空间校准学习器在支持集S上拟合后为 `pS(x)`。直接在S上选混合强度，会利用同一标签两次：复杂的校准器可以拟合标签，然后在同一标签上证明自己值得信任。第三轮的源KL测量的是预测变化大小，不能判断这种变化是否改善新样本。

将有效支持划入固定折 `V1,...,VK`。第k折的校准器只使用 `S\Vk`，然后预测Vk，得到每条支持的交叉拟合概率

\[
 \widetilde p_i=p_{S\setminus V_{k(i)}}(x_i),\qquad
 d_i=\widetilde p_i-p_0(x_i).
\]

折模型不得用被留出标签拟合头、空间参数、类别平衡权重或早停。冻结源模型以及身份切线的无标签输入导数可以在全部预算内支持X上一次计算，再按训练/留出索引切片；这不读取未来查询，也不使用被留出标签拟合该折模型。

为避免标签排序后“好看”的折分，主折按原始付费位置连续分块，不按标签重新分层。QC后按原位置映射到折；空有效折跳过，单类训练折按既定学习器回退源模型并记录原因。不得因为某折失败而补入额外试次。最终实现以5个block为主，折数不超过有效样本数；LOO和确定性类别分层为预先固定敏感性，不在观察留出后择优。前向链是有价值的后续协议，但本次204项没有实现，不把它列为完成任务。

### 2.2 Brier收缩的闭式解

两类支持时，令 `qi=1/(2 n_yi)`，总和为1。考虑

\[
 \min_{0\le\alpha\le1}
 \sum_i q_i\{p_0(x_i)+\alpha d_i-y_i\}^2+gamma\alpha^2.
\]

其解为

\[
 \widehat\alpha=\Pi_{[0,1]}
 \frac{\sum_iq_id_i\{y_i-p_0(x_i)\}}
 {\sum_iq_id_i^2+\gamma}.
\]

分母无信息时取0；正的gamma给出向源模型收缩的先验。最终再用全部预算内有效支持重拟合一次，未来预测为

\[
 p_{final}(x)=(1-\widehat\alpha)p_0(x)+\widehat\alpha p_S(x).
\]

这是**概率混合**，不是theta/head参数插值或logit混合，不能互相替代。已冻结的源和完整支持校准器都需要用于查询推理，需计入实际部署推理成本。

`alpha=0`是可行点，因此最小化后的交叉拟合Brier加罚项不大于源的同样经验Brier；未加罚的同一交叉拟合Brier也不大于它。这只是一个经验最优化性质。完整支持重拟合后的函数和各折函数不同，支持试次存在相关性，alpha又使用全部交叉拟合标签训练，故**不是未来风险不增加定理，也不是完全独立验证的alpha风险估计**。应保存OOF风险、完整支持重代入风险和真正未来查询风险的三者差距。

## 3. 主要方向B：源、头、空间的机制分解式凸堆叠

单一校准器可能在部分会话只需要头调整，另一些会话需要空间调整。固定小字典 `source / head-only / spatial`，为各成员生成同一折划分下的OOF概率矩阵 `P`。源列无需训练。解

\[
 \widehat w=\arg\min_{w\ge0,\;\mathbf1^Tw=1}
 \|W^{1/2}(Pw-y)\|_2^2
 +\frac\gamma2\|w-e_0\|_2^2,
\]

其中 `e0=(1,0,0)` 表示源锚，`W=diag(qi)`。正的gamma给出严格凸二次问题。仅三个成员时，可以枚举7个非空活跃集，在每个集合上解带和为1约束的小线性系统，检查非负性并取目标最小者；无需额外大型优化器。

梯度系统为

\[
 H=2P^TWP+\gamma I,\qquad
 g=2P^TWy+\gamma e_0.
\]

注意系数2；概率型二分类Brier有时写成两类误差之和，是此正类Brier的2倍，若混用定义却不改gamma会改变正则强度。两成员情形的 `gamma/2 ||w-e0||²` 恰好是 `gamma alpha²`，应当与A的闭式解一致。

完整支持重拟合各非源成员后，用同一个w形成最终概率混合。此方法可以产生“支持不足以证明空间校准有益，因此只用头或源”的结果，但不要把某一权重直接解释为真实生理漂移类型。高度相关的成员可交换权重而几乎不改变预测；gamma的作用部分是稳定分解。

必要对照包括：源、单头、单空间、固定等权混合、同支持重代入拟合权重（明确标为乐观机制对照）、OOF收缩、OOF堆叠；若OOF优于重代入才有证据说明样本外选择是有用机制。还需固定先验的源+头双成员与源+空间双成员，以识别收益是否只来自多模型平均。

## 4. 独立方向C：有限候选中的one-SE选择

`xfit_select` 与 `xfit_select_stack` 在支持内固定比较六个组合：头先验 `.01/.1/1` 与空间半径 `.05/.15` 的2×3网格。每个组合都只在折的训练子集拟合，并在对应留出支持上用**真实非线性空间变换**产生概率；不是在代理margin上计算OOF分数。参数候选以空间校准器的OOF类别平衡Brier打分，stack版本在选出此候选后再拟合组合权重；不能称已经联合搜索了最优stack风险或完全独立地验证了所选权重。

记候选c的总体分数为 `Rc`。对于第k折，`wk=sum_{i in Vk} qi`，`rck=sum(qi loss_i)/wk`，则 `Rc=sum(wk rck)`。代码采用

\[
 s_c^2=\frac{\sum_kw_k(r_{ck}-R_c)^2}{1-\sum_kw_k^2},\qquad
 SE_c=\sqrt{s_c^2\sum_kw_k^2}.
\]

只有一个有效折或分母退化时SE取0；折质量与回退次数仍记录。取最低分数候选 `cmin` 后，容许集合为

\[
 \mathcal C_{1SE}=\{c:R_c\le R_{cmin}+SE_{cmin}\}.
\]

在该集合中固定优先更小半径，其次更大头先验；完全相同时沿固定遍历顺序选择。最终对照 `minimum` 直接选最低CV风险，不使用该容许区间。选定的头先验同时用于完整支持的head-only成员和空间成员，保证stack中的头与空间共享这次选择。

这是一项**偏向简单模型的选择启发式**，不是新统计检验；OOF折共享训练样本且EEG有时间依赖，SE不可当成独立折下的未来风险置信区间。Bates、Hastie、Tibshirani的JASA研究专门说明CV风险对象与朴素区间的复杂性，[原论文](https://doi.org/10.1080/01621459.2023.2197686)。候选网格也会过拟合CV准则，问题并不会因为候选只有六个就消失，[Cawley与Talbot，JMLR2010](https://www.jmlr.org/papers/v11/cawley10a.html)。因此本轮保存所有候选分数、SE、选中配置、OOF/full-refit实际查询差距，而不是让one-SE变成启动前需要“通过”的门。

## 5. 独立方向D：对疑似误标更稳健的校准目标

这个方向可与A/B分开，避免所有参数的全笛卡尔积。广义交叉熵GCE定义

\[
 \ell_q(p_y)=\frac{1-p_y^q}{q},\qquad0<q\le1.
\]

q趋于0为交叉熵，q=1为MAE型损失。二分类margin m下，令 `py` 为真实支持标签的概率，则

\[
 \partial_m\ell_q=p_y^q\{\sigma(m)-y\}.
\]

因此极端低py的样本被下权。可取预先固定q=.7，.3/1作敏感性，q=0由普通CE实现；不要因当前内部留出最好而更改主q。这个做法早已有NeurIPS2018先例，不能作为本轮独立新理论。

本轮实现每轮更新 `wi=py^q` 的工作加权CE/GGN子步，并对真实GCE加正则目标回溯。其梯度为真实GCE梯度，但正定工作曲率忽略权重对margin的导数，因此不是其一般非正定Hessian。每轮重新计算权重，空间端点受原半径球约束，再重新求相应条件头更新。目标是不增加所定义的固定身份切线GCE加罚项，不是第三轮的全局凸代理，也不承诺找到了非凸问题的全局最优解。回溯仍为有限预算，找不到下降步则保留旧点并记录，不中断其他会话。

`xfit_stack_gce` 使用同样的源/head/spatial堆叠结构，其中空间成员的校准目标换成GCE，head-only成员仍为匹配先验的普通逻辑头；不把它含混为三个成员都进行了GCE训练。GCE并不直接优化最终查询的BA或Brier。

最关键反例是：源模型自信但错误时，真正有纠错价值的困难试次也会被当作噪声下权。因此主清洁结果、支持标签10%/20%翻转、类别下权分布、纠错与负翻转都必须保留。没有人工真值证据时，不能把自然困难样本断言为“标签错误”。

## 6. 本次未实施的后续方向：有条件的各向异性先验

R3中空间先验为 `lambda_theta ||theta||²`。更深入的问题是元训练会话是否支持某些空间方向比另一些方向更可靠。可用

\[
 \lambda_\theta\theta^T\Omega\theta,
 \quad\Omega\succ0,\quad\operatorname{tr}\Omega=r,
\]

并仅从元训练/元验证学习Omega或估计校准参数协方差，留出不得参与。可对经验协方差做向各向同性收缩，再逆转和归一化trace；必须同时比较单位阵、对角和完整矩阵，并匹配总强度。这个估计量基于已拟合theta，其噪声、边界截断和字典坐标会影响协方差，不能把它当成已识别的真实生理漂移先验。

Omega与B共同学习时存在换基问题：`B→BO` 应伴随 `Omega→O^T Omega O`；只允许对角Omega则会破坏一般换基等变性。放任矩阵尺度还能等价改变lambda，因此trace/谱约束及相同强度对照是必要数值定义。元学习的层次贝叶斯解释和协方差收缩都已有明确先例。除非最终代码和测试包含这个方向，不把它计为已实现的第四轮任务。

## 7. 可证伪问题与实验分解

1. **OOF是否只是更保守？** 本轮比较OOF alpha与固定alpha=.5、原空间校准器，并报告实际alpha分布；不是只报告平均准确率，也没有实现完整`.25/.5/.75/1`固定混合网格。
2. **时间相关是否让CV看起来过好？** 对照连续block、LOO和类别分层；在全部预算内形成折，不换用未来查询。前向链留作后续，不把它写成已完成测试。
3. **堆叠是否只重复源模型？** 报告source/head/spatial权重和有效成员数、成员预测相关性、相对源的纠错/负翻转。源权重接近1也属于有效阴性结论。
4. **稳健目标是否抹掉有用难样本？** 清洁与人工误标压力成对比较，未来查询真标签不变；仅对支持标签施加固定种子翻转。
5. **混合是否值得推理开销？** 分开计时OOF拟合、完整支持重拟合、未来查询各成员推理和混合；每次计时按CUDA完成点测量。
6. **同一方法能否跨数据集？** 已准备的Weibo外部数据仅作发布顺序LR前缀稳健性。仍不能声称它拥有已核实的真实物理采集前20条，也不能把20条LR前缀写成只经历20条全部任务。

所有目标权重、正则与fold模式只可由元训练/元验证或预先固定规则确定。对当前已反复使用的内部参与者队列继续分析，应标为探索性内部重复验证；即使p<.05也不是重新获得的首次盲测。参与者是统计单位，先在参与者内对会话和种子取均值，不能把试次、fold和种子堆成独立n。预先固定主要比较族，并使用同一族的多重比较校正；其他宽矩阵用于描述与发现，不事后升级成主假设。

保持精确核算关系：类别平衡的 `BA_new−BA_source = corrected_error_mass−negative_flip_mass`。最终混合概率用于NLL/Brier/ECE和argmax，不能用某个成员预测充当混合效果。

## 8. 一手文献与真正可声称的增量

以下已通过原论文站点、作者页或作者所属机构存档核对。它们支持方法来源与边界，不预测本轮会取得多少百分点收益。

| 来源 | 本轮采用或需要避免的误读 |
|---|---|
| Wolpert, *Stacked Generalization*, Neural Networks5(2):241–259,1992，[原文/DOI](https://doi.org/10.1016/S0893-6080(05)80023-1) | 用部分数据拟合基学习器、用未参与拟合的输出训练上层组合早已存在；OOF组合不是新发明。 |
| Breiman, *Stacked Regressions*, Machine Learning24:49–64,1996，[作者机构原稿](https://statistics.berkeley.edu/tech-reports/367) | 交叉验证预测与非负最小二乘堆叠是经典方法。本轮的三成员凸二次问题是其任务特定实例。 |
| van der Laan, Polley & Hubbard, *Super Learner*,2007，[原作者机构技术报告](https://biostats.bepress.com/ucbbiostat/paper222/)，[DOI](https://doi.org/10.2202/1544-6115.1309) | 交叉验证选择与组合已有oracle研究；不能把它的渐近条件直接转成20条相关EEG的有限样本保证。 |
| Hu & Zhang, *Optimal Parameter-Transfer Learning by Semiparametric Model Averaging*,JMLR24(358):1–53,2023，[原论文](https://jmlr.org/papers/v24/23-0030.html) | 已用J折CV为迁移源分配数据驱动权重。其半参数模型及渐近最优性不等同冻结EEGNet+空间适配。 |
| Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*,JMLR11(70):2079–2107,2010，[原论文](https://www.jmlr.org/papers/v11/cawley10a.html) | CV准则自身可因方差而过拟合；支持内候选选择是完整学习算法的一部分，不能在内部留出选最优网格后再冒充盲测。 |
| Ryan Tibshirani，*Model Selection and Validation1: Cross-Validation*，CMU作者课程讲义，2013，[作者原稿](https://www.stat.cmu.edu/~ryantibs/datamining/lectures/18-val1.pdf) | one-SE是已有的保守选择规则；本轮只明确其多参数复杂度排序，不把规则本身作为新算法理论。 |
| Bates, Hastie & Tibshirani, *Cross-Validation: What Does It Estimate and How Well Does It Do It?*,JASA119(546):1434–1445,2024，[原论文](https://doi.org/10.1080/01621459.2023.2197686) | 标准CV区间可能低覆盖，重拟合还会改变推断对象。本轮one-SE只称选择启发式，不能当风险证书。 |
| Bergmeir, Hyndman & Koo, *A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction*,CSDA120:70–83,2018，[作者页](https://robjhyndman.com/publications/cv-time-series/) | 普通CV的论证依赖纯自回归和不相关误差等条件，不能因标题含time series就认定任意EEG前缀CV已获保证。 |
| Roberts等，*Cross-validation Strategies for Data with Temporal, Spatial, Hierarchical, or Phylogenetic Structure*,Ecography40:913–929,2017，[作者页](https://florianhartig.github.io/publications/roberts2017cross/) | 验证设计应与依赖结构和目标一致。block减少某些相邻泄漏风险，但20条的有限统计能力仍不能绕过。 |
| Zhang & Sabuncu, *Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels*,NeurIPS2018，[原论文](https://papers.nips.cc/paper_files/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html) | CE/MAE的q家族及噪声鲁棒动机已有先例；困难样本抑制是本任务必须实测的代价。 |
| Ledoit & Wolf, *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*,JMA88:365–411,2004，[作者原稿](https://www.ledoit.net/Well-conditioned2004.pdf) | 样本协方差向单位阵收缩已有严格理论；基于校准theta估计先验并不自动满足其数据条件或转移全部最优性结论。 |
| Grant等，*Recasting Gradient-Based Meta-Learning as Hierarchical Bayes*,ICLR2018，[原论文](https://openreview.net/pdf?id=BJ_UL-k0b) | 跨任务先验与元学习的联系已有直接先例，不把一般“学习正则”当作新贝叶斯理论。 |
| Angelopoulos等，*Conformal Risk Control*,ICLR2024，[原论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) | 对单调损失的风险控制需要其校准/交换性等设置。OOF权重在同20条上拟合且EEG随时间依赖，不能把它命名为分布无关未来负迁移保证。 |

可合理探索的组合贡献是：**在严格采集前缀和不接触未来查询的校准中，利用交叉拟合证据分离源、分类头和空间变换的作用，并共同刻画支持噪声、源错误、自适应收缩与未来负迁移之间的关系。** 这需要外部稳健性、匹配成本及真实误差分解支持；不是把stacking、Schur、GCE或经验贝叶斯重新命名。

## 9. 固定实验矩阵：204项

任务种子均为 `20261018/20261019/20261020`。主空间秩4，半径`.15`，头先验`.1`，空间先验`.01`，内更新10，OOF5个连续块，混合ridge`.01`，GCE q`.7`。单独旧字典的元训练头先验为`.001`，匹配SGD和新学习器字典的元训练头先验均为`.1`，避免新字典比较混入头正则强度变化。

| 家族 | 构成 | 任务数 |
|---|---|---:|
| legacy字典 | 秩2/4/8，各3种子 | 9 |
| matched_sgd字典 | 秩4，各3种子 | 3 |
| xfit_shrink/xfit_stack/gce字典 | 3种元学习器×3种子 | 9 |
| legacy字典核心评价 | 下述14适应器×3种子 | 42 |
| 匹配/新字典交叉评价 | 4字典家族×4适应器×3种子 | 48 |
| 敏感性评价 | 下表固定单轴改变 | 90 |
| 支持EA参考 | 3种子 | 3 |
| **合计** | **21basis +90core +90sensitivity +3reference** | **204** |

最坏需要21,000次外层更新，每次4个episode；这不是源EEGNet从头训练。旧legacy及第三轮matched_sgd最多12项仅在数值配置、源/数据身份、训练完成和验证选择记录匹配时复用。未满足复用条件则独立训练，不为了等待旧轮无限阻塞本轮。三种新学习器字典共9项本身需要元训练。

14适应器精确命名为：`sgd10`、`head_newton`、`tangent_profiled`、`information_protected`、`fixed_half`、`uniform_stack`、`insample_shrink`、`insample_stack`、`xfit_shrink`、`xfit_stack`、`xfit_select`、`xfit_select_stack`、`gce_profiled`、`xfit_stack_gce`。最后一个在配置内部表示为 `xfit_stack` 加 `base_variant=gce_profiled`，分组名称仍单独保留。

4字典家族为matched_sgd、xfit_shrink、xfit_stack、gce；各自只交叉评价 `tangent_profiled/xfit_shrink/xfit_stack/gce_profiled`。没有对全部新字典做14适应器全笛卡尔积。90核心任务均评价支持4/8/12/16/20预算，共450个clean评价坐标；每个坐标含所有预先登记的参与者/会话，不是450次元训练。

| 敏感性轴 | 非主设置 | 方法数×设置数×种子数 | 任务数 |
|---|---|---|---:|
| CV折方式 | 确定性类别分层、LOO | shrink/stack/select-stack，3×2×3 | 18 |
| 折数 | 2、10 | shrink/stack，2×2×3 | 12 |
| 混合ridge | `.001/.1` | shrink/stack，2×2×3 | 12 |
| 空间基学习器 | information_protected | shrink/stack，2×1×3 | 6 |
| 空间秩 | 2、8 | shrink/stack，2×2×3 | 12 |
| GCE q | `.3/1` | GCE、stack-GCE，2×2×3 | 12 |
| 参数选择规则 | 最小CV风险 | select/select-stack，2×1×3 | 6 |
| 空间半径 | `.05/.35` | shrink/stack，2×2×3 | 12 |
| **合计** | **只跑B20、clean** | | **90** |

压力条件为支持标签10%/20%翻转、校准后查询对称增益与反对称旋转各`.10/.25`两档。在legacy字典的profiled、information_protected、shrink、stack、GCE、stack-GCE六组×3种子执行，共18核心任务×6条件=108额外评价组合，只用B20。校准后查询漂移没有被算法观察并适应，不称主动漂移追踪；生成元范数不是电压百分比或物理旋转角度。所有方法共享种子、会话和对应扰动。

可选Weibo阶段另有7组×3种子=21评价任务：legacy字典的profiled、information_protected、shrink、stack、GCE、stack-GCE六组，另加新stack字典的stack。它读取本轮已选字典，不重新元训练，不修改核心计划。外部数据真实采集时序未验证，其发布顺序协议只作描述性稳健性。

## 10. 十二项预先固定主要比较

主要坐标为内部holdout、B20、clean，终点为参与者等权平均BA。`legacy`表示相同旧字典，主目标会话头先验为`.1`。主要比较固定为：

| # | 候选 − 对照 | 要检验的机制 |
|---:|---|---|
| 1 | legacy:xfit_shrink − legacy:fixed_half | 支持内估计的alpha是否优于固定一半混合 |
| 2 | legacy:xfit_shrink − legacy:insample_shrink | OOF拟合alpha是否优于同支持重代入选择 |
| 3 | legacy:xfit_stack − legacy:uniform_stack | 源锚定OOF权重是否优于固定等权 |
| 4 | legacy:xfit_stack − legacy:insample_stack | OOF堆叠是否优于重代入堆叠 |
| 5 | legacy:xfit_stack − legacy:xfit_shrink | 增加头成员和组合自由度是否值得 |
| 6 | legacy:xfit_select − legacy:tangent_profiled | 六候选one-SE参数选择是否比固定主点更有效 |
| 7 | legacy:xfit_select_stack − legacy:xfit_stack | 选择头先验/半径是否给固定stack额外收益 |
| 8 | legacy:gce_profiled − legacy:tangent_profiled | 稳健目标是否改善清洁识别 |
| 9 | xfit_stack:xfit_stack − matched_sgd:xfit_stack | 同训练先验下，适配stack的字典元目标是否有效 |
| 10 | gce:gce_profiled − matched_sgd:gce_profiled | 同训练先验下，GCE字典目标是否有效 |
| 11 | xfit_shrink:xfit_shrink − matched_sgd:xfit_shrink | 同训练先验下，适配shrink的字典目标是否有效 |
| 12 | legacy:xfit_stack_gce − legacy:xfit_stack | 在相同组合中只改变空间目标是否有价值 |

第5项同时增加成员和参数，不能把全部差异归于更好的风险估计；第6/7项包含会话头先验与半径变化，不能称纯求解器消融。第8项以clean为主要坐标，标签压力的差异另作预先命名描述，不能在clean不利时事后把压力结果改作唯一主结论。

主要12比较涉及的各组3种子及共同参与者/会话均齐备后计算整族Holm校正；中途可回传效应量和明确覆盖，不能挑已完成种子的最小p。敏感性、预算、压力、外部及其他交叉比较报告描述性CI与效应，不把未校正p混为主要发现。重复使用的原内部队列始终标注探索性内部重复；若今后需要真正确认，应预先冻结整条算法，并在此前未用于设计的新参与者/数据上单独验证。

## 11. 字典元学习的数值对象与边界

仍以每参与者、每会话均匀抽4个episode，使用实际未来查询标签作为**元训练外层监督**，不把它传给该episode的支持适配器。外目标为真实非线性变换后最终混合预测的类别平衡交叉熵，字典约束 `B^T B=I`。每次外层更新累积4个episode梯度后释放图，投影至Stiefel切空间、梯度裁剪并以`.01`做QR回缩；1000更新、每50步元验证，按参与者等权BA最高/BCE最低/更早更新选择。

OOF闭式alpha、所选simplex活跃面上的连续权重和相应fold学习器都参与当前分支的外层梯度，不应仅学习full-fit校准器而把组合当成与B无关常数。source/head/spatial的推理混合保持概率定义，数值clip只用于对数损失稳定。候选argmin/one-SE、simplex活跃集、投影与回溯比例切换时都是分段算法；不承诺全局光滑、高阶元梯度或精确内层最优解的隐式梯度。

若任一折/会话两类不足，源回退是算法输出；若数值或输入身份确有错误，应保存明确error与断点并继续独立任务，不能把错误伪装成COMPLETE。这与程序重复进行全量pytest、pilot或冻结库全盘验证无关。本轮保持轻量启动、现有环境、每10更新断点和逐会话评价保存；不存在累计12小时科学预算上限。GitHub同步与本地训练独立，网络错误只影响同步重试。


