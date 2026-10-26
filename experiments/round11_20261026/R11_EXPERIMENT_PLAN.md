# R11：有向前缀分数传输的预先规定实验方案

本方案对应 `R11_MATH.md` 中的 PFST：以 meta_train 的前缀→固定未来配对学习算子，中心化固定类别计数置换仅作离线正则化；目标预测不读取 query 标签。当前没有真实 R11 EEG 效应结果，不能以合成测试或旧轮结果声称超过 EA。

用户允许充分计算。默认将完整开发、选择及旧 cohort 探索任务放在一个有依赖的 campaign DAG 中自动推进，不再增加“必须先显著改善”的人工 PASS 阶段或4/4小样本检查。资源为 **4个作业，每个1GPU、16CPU、64G**。本文作者本轮没有运行 GPU 或提交 Slurm。

## 1. 不能变动的数据与评价单位

- seeds：20261018、20261019、20261020。它们是三个固定 pipeline/source 版本，不能把参与者数乘3。
- 校准预算：原始前4/8/12/16/20个采集位置；坏点仍付费，不补位。
- 每个 episode 的未来集合按原始 B20 定义一次，所有预算和方法使用同一 query UID、顺序和 tie-breaking。
- Support-EA 只用当前预算内有效支持输入；同一个变换应用于该 episode 的支持与未来 query。
- 主方法的 whitening/axes 来自 meta_train；选择 rank/η/λ 只用 meta_validation。
- **meta_train future-query 标签是本轮离线训练监督的一部分。**meta_validation query 标签用于预先规定的候选选择；目标 holdout、Pan、Weibo、独立确认 query 标签只在持久化预测之后评价，不回流学习。
- 每个 seed 内 session 等权，再在 participant 内 seed 等权；participant 为独立统计单位。query 数、预算数、session 数、重复配对数都不能增加名义 N。

旧 holdout/Pan/Weibo 已参与 R1–R10 研发，R11 在这些 cohort 上的正式执行仍是**探索性**。Pan/Weibo 若只有 published-order 代理，要保留其顺序范围标注。新独立确认必须是未参与源训练、开发和历史目标分析的数据；不能靠旧数据换名、重划分或换 seed 获得确认资格。

## 2. 学习目标、特征和容量

主方法参数为 `feature_rank=r, noise_strength=η>0, ridge=λ>0`。先在固定 meta_train 坐标中投影，再仅将特征向量按 norm 封顶；含 offset 时追加常数并统一除√2。该顺序保留真正的常数坐标和 ‖φ‖≤1。

算子 A 是 d×d，d=r+1（有 offset 时），不是只有 d 个离线参数。在线 β=Ag 是 d 维 head；每个固定预算有相应训练算子。报告离线参数 d²、在线状态大小、拟合时间和 query 吞吐，不能仅把在线 β 的维数当作整个学习器容量。

主训练目标固定为

\[
\widehat{\mathbb E}\left[-r_Q^\top Ag+
\frac18\operatorname{tr}\{M_Q A(gg^\top+\eta N)A^\top\}\right]
+\frac\lambda2\|A\|_F^2.
\]

λ/2 是明确的目标归一化；正常方程中为 `+λA`，不能误写为 λ‖A‖² 后仍使用相同 λ。

没有 target query profile、posterior 方差门、目标模型选择或按 query 标签估计的收缩权重。N 的精确性只针对指定 fixed-count permutation 分布，不等同真实 EEG 噪声估计。平衡类别时 N 精确等于缩放的中心化特征协方差，必须公开这一简化。

## 3. 完整开发网格与任务数

主网格440个配置：

\[
\begin{aligned}
r&\in\{4,8,16,32,64\},\\
\eta&\in\{.03,.1,.3,1,3,10,30,100\},\\
\lambda&\in\{10^{-6},3\!\times\!10^{-6},10^{-5},3\!\times\!10^{-5},10^{-4},3\!\times\!10^{-4},10^{-3},.003,.01,.03,.1\}.
\end{aligned}
\]

另有两组各55配置、**独立选参**的强 learned-optimizer 对照。它们都用同一个 rank×λ 网格，η=0、无置换正则、自由方向算子：

1. `operator_no_noise_tuned`：相同 global logistic majorizer。
2. `operator_erm_tuned`：直接优化真实 balanced future logistic loss difference，加 λ/2‖A‖²。

不能仅比较主配置去掉 η 后的一个固定参数点，然后声称胜过充分调优的 learned optimizer。

| 部分 | 任务数 | 组成 |
|---|---:|---|
| meta_train reference | 3 | 每 seed 一个固定参考 |
| 完整候选算子训练 | 8250 | 550配置×3seed×5预算 |
| meta_validation 候选评价 | 1650 | 550配置×3seed；每任务包含5预算 |
| meta_validation EA/Frozen | 6 | 2方法×3seed |
| 决定并记录配置 | 1 | 完整候选表、one-SE、LOPO和状态 |
| 开发合计 | **9910** | 上述五项 |
| 所选配置的9个 matched controls | 135 | 9×3seed×5预算训练 |
| 旧3cohort探索评价 | 225 | 25方法×3seed×3cohort；每任务5预算 |
| campaign总计 | **10270** | 9910+135+225 |

任务数不是参与者数，也不是单次廉价拟合数。每个训练或评价任务可以包含大量 session/query；应记录实际 encoder forward、meta_train pair、求解迭代及缓存复用量。

## 4. meta_validation 选择规则

对每个候选 c，在同一冻结 participant/session/seed/query 集上计算 BA 曲线及相对 EA 的归一化 AUBC：

\[
A_{c,p}=\frac1{20-4}\sum_{j=1}^{4}
\frac{B_{j+1}-B_j}{2}
\{BA_{c,p}(B_j)+BA_{c,p}(B_{j+1})\},
\quad \Delta_{c,p}=A_{c,p}-A_{EA,p}.
\]

先按 participant mean Δ 最大者定义 best；用该候选 participant 配对差的 SE=sd(Δ)/√N 定义 one-SE band。不能使用 query/session/seed 级 SE。

在 band 内按固定顺序选择：**较小 feature_rank、较大 ridge、较大 noise_strength、candidate ID 字典序**。较大 η 只是固定 tie preference，不意味着每个 query 更安全或实际风险单调。两组55点独立对照各自在自己的完整候选集合内使用同样的 one-SE原则。

LOPO 仅作为选择稳定性诊断：每次移除一个 meta_validation participant，重做上述选择，报告所选配置分布与效应敏感性；不据此重排最终完整人群的选择，也不把这些留一结果当独立确认。

缺少任何计划内 participant/session/seed/budget/query身份配对的候选，**整个候选**不进入 eligible 集；不能在方便的交集上保留它。报告排除名单、原因及完整覆盖表。若某一族没有 eligible 候选，采用预先声明默认配置，选择状态为 **`DEFAULT_UNVALIDATED`**，科学解释明确标注 **`NO_VALIDATED_SELECTION`**；该状态随正式探索结果传播，不能伪装为开发成功。campaign 可继续，禁止把缺失或负面结果改写成 PASS。

选择完成后，后续节点自动读取保存的配置/所选 operator artifacts。代码、特征映射、class weights、η/λ、算子约束、数值 floor、求解停止规则和选择/比较家族均固定。目标结果不得用于重新选参；若需要新设计，应建立新研究版本。

## 5. matched controls 与25个正式方法

所有 learned matched controls 使用所选主配置的 rank、ridge、训练数据、特征图、预算与 seed，除声明的变化外均相同。

| method | 唯一声明变化 |
|---|---|
| `operator` | 主方法：自由 A、exact centered-permutation N、η>0、majorizer |
| `operator_no_noise` | N=0、η=0；其余主配置匹配 |
| `operator_symmetric` | A=Aᵀ，可有负特征值 |
| `operator_psd` | A为对称PSD |
| `operator_diagonal` | A为自由符号的对角矩阵 |
| `operator_identity` | A=αI，α∈R；不弱化为α≥0 |
| `operator_feature_noise` | 标签无关、balanced-matched中心化特征协方差 Σ(φ−φ̄)(φ−φ̄)ᵀ/[4n(n−1)]；平衡时与主N精确相同 |
| `operator_fisher_noise` | 用固定权重Bernoulli/Fisher分数协方差控制替代N |
| `operator_shuffled_pairs` | 前缀与不同meta_train session的future重新配对；确定性置换，Q仍使用输入前缀EA |
| `operator_erm` | 同配置的真实logistic meta-ERM，保留相同η噪声惩罚与ridge |
| `operator_no_noise_tuned` | 独立55点网格选择的无噪声majorizer |
| `operator_erm_tuned` | 独立55点网格选择的无噪声真实ERM |

其中前10行中的9个非主方法是135训练任务的 matched controls；最后两行复用完整开发族的合法所选 operator。真实ERM的观测 NLL difference、噪声惩罚与ridge应分别报告，不能把带惩罚目标值写成实际NLL。

feature-noise 已固定为标签无关的 balanced-matched 中心化特征矩，与主N在平衡类别下连尺度和offset处理都相同；须有数值等价回归检查，不能在此情形宣称主N提供新信息。差异由训练前缀的不平衡产生；若某预算全部训练前缀平衡，二者得到同一A且对所有目标episode预测相同。Fisher-noise 为 Σv_i²p_i(1−p_i)φ_iφ_iᵀ，使用不同条件模型。应按预算报告各矩trace/rank和meta_train不平衡episode比例。独立调优ERM族η=0，matched ERM用主η；本次没有穷尽η>0的独立ERM全网格，因此不能声称主majorizer胜过所有充分调优的noise-regularized ERM。

`shuffled_pairs` 必须记录真实 source/future session映射并排除原配对。若只要求不同session，它可能保留同participant信号；只能解释为 episode-association 控制，不能声称已破坏全部participant关联或必应得到零收益。

另有13个固定强基线：Frozen、真实Support-EA、bias/rank4/full head、SGD10/100、R3保护、R4 select-stack、R8 fiber-bounded、R9 raw-RBF anchored、R9 EA-linear anchored，以及保留默认设计的R10 profile。R10尚无真实EEG性能结果，使用它作设计基线不等于已有正面效应证据。原生方法按实际执行语义及冻结meta配置运行，不能因缺artifact就复制Frozen或换成较弱方法。

除原生基线外，匹配 rank/特征映射的在线head控制也有助于区分特征容量与operator作用；若不在本次固定25方法中实施，应明确列为范围限制，不能称全部容量解释已被排除。

## 6. 主要家族、机制家族与成功解释

主要终点：B20 participant-equal BA。每个 cohort 预先固定6个对比：主方法分别对 EA、SGD100、R9 raw-RBF、R9 EA-linear、R10 profile、独立调优 `operator_erm_tuned`。

若有 K 个 cohort，这构成 **6K** 项双侧主要家族，使用 Holm。旧3cohort时为18项，且始终标为探索。主要比较完整性不能因某个参照缺失而减少家族或偷偷删除参与者。

9个 matched controls 的 BA/NLL（B20）形成单独 **18K** 项预先规定机制家族，Holm单独校正。独立调优无噪声majorizer的完整结果也报告；若未列入这些家族，则明确为次要结果，不能借用别的家族显著性。

报告每个对比 participant N、完整差值向量、均值/标准差、95%CI、原始p、家族大小和Holm p。CI为participant配对bootstrap；p为双侧participant sign-flip。两者不是互相反演，小N的边缘CI/p不一致不能被选用有利口径。

辅助结果：五预算 BA/NLL/Brier、normalized AUBC、participant负迁移比例、纠错/引错、全部query的非零更新比例、算子对称/反对称能量与谱、训练目标/实际训练NLL、求解残差、运行时间和内存。

正向BA效应、CI与对应家族校正检验共同支持时，才可针对相应方法声称优势。打败固定η=0但不胜过充分调优的ERM，不能证明超过已有learned optimizer；全部接近EA也不能包装成突破。置换机制若与平衡类别featurecov控制等价，应作为等价结果而非人为制造差异。

## 7. 一次适当兼容性验证与运行效率

发布前集中完成：Python3.12.14、Torch2.5.1+cpu、NumPy2.2.6的CPU测试（学校为Python3.12.3/Torch2.5.1+cu121；微版本和CUDA构建不同，不声称已完成真实GPU兼容验证）、fixed-count置换枚举、global-bound/正常方程检查、类平衡简化、常数offset、非对称反例、η=0/λ极限、query标签隔离、固定B20未来集、paid-invalid位置及native强基线派发。

离线达到求解迭代上限时，有限且目标不劣于零的实际迭代值可以继续使用，保存 `converged=false, accepted_feasible_iterate=true`；不能仅因未达到极严停止阈值就删除候选。只有非有限、状态损坏或目标劣于零等实际失败才回退，并记录原因。选择有效性与算法目标证书是不同字段。

学校CUDA验证应使用有限真实缓存/冻结checkpoint做一次小批前向、EA及保存恢复检查，并明确记录设备与版本；CPU结果不能冒充GPU实测。本讨论及文档编写未执行GPU/Slurm。

部署保持4个作业各1GPU/16CPU/64G；DAG依赖自动完成reference、完整候选、选择、matched controls和探索评价，不要求用户逐阶段人工确认科研PASS。配置/输入错误应清晰报错并有限重试，禁止无限吞错或无限重提作业。

特征/EA/训练moment缓存按实际 source、数据、UID、预算、特征rank及变换身份复用；λ、η扫描不重复encoder提取。适合预计算的正常方程统计也可复用。身份验证在首次读取/真正变化时执行，不能把每个轻量算子任务都变成全量数据重哈希、GPU导入和重复大型预检。

## 8. 独立确认与本轮论断边界

独立确认不混入本次旧cohort自动探索campaign。获得真正新数据、采集顺序与历史独立性记录后，使用已固定算法直接评价，不再以确认结果调参。source种子、预算与共同数值评分策略保持一致。

本方案可检验“置换扰动正则是否相对充分调优learned optimizer有效”及“自由方向传输是否相对对称/PSD约束必要”。训练平均上界、完美的软件测试、任务完成数及消耗GPU时间都不能替代这两项真实证据，也不能代替有限检索下的新颖性判断。
