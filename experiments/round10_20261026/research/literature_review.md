# R10 文献核查、候选算法与原创性边界

核查截止日：**2026-10-26**。本记录先阅读现有实现，再核对真实论文的出版页、作者预印本或原文 PDF。结论是：**不能将 Schur 信息、正交 GP、近最优模型集合、查询预测区间、方差门控或 minimax 原则本身作为 R10 的发明。**目前值得实现和证伪的是一个具体的、以支持拟合证据增益约束查询函数歧义、再决定相对固定 anchor 的更新的算法耦合机制。其原创性属于有限检索下的研究假设，尚不构成“全球首次”或已经达到投稿标准的结论。

## 1. 现有代码已经做了什么

实际阅读的本地审计副本及 SHA256：

- R3：`paper3r3/algorithm.py` 中 `information`、`_propose`、`_protect`。R4 的 `core.py` 保留了这套原实现。已有 head-ridge Schur 信息、有效维数缩放半径、profiled Newton 和支持预测 KL 保护。
- R4：`paper3r4/algorithm.py`，SHA256 `e595d0256424b9f3e23a653d78de3a940140c667ace5118524060e1396fe9226`。已有按采集位置分块的 OOF、one-SE、Brier 收缩、概率 simplex stacking；其注释明确把 SE 当选择启发式，不把时间相关折当独立推断样本。
- R9：`paper3r9/gp.py`，SHA256 `5956354a457c95db4503494c74bb0e1a6e6e2f3cfaf661c5c04ca1d49ee93ad7`。`fit_gp` 是带源/EA margin offset 的 Bernoulli logistic GP；`posterior_moments` 已计算查询的相对方差缩减；`predict_proba` 已按 `anchor_cap * (prior-variance)/prior` 对源预测与残差 GP 概率混合。

这些副本用于辨明算法重叠；最终证据审计还须把仓库中使用的实际版本和任务代码身份一并固定。数学实现与原始结果的统计核查是两项不同工作。

## 2. 与核心构想最接近的先例

### 2.1 nuisance 投影与可辨识性并非新概念

[Plumlee 与 Joseph，2018，Orthogonal Gaussian process models](https://www3.stat.sinica.edu.tw/sstest/j28n2/j28n23/j28n23.html)，DOI `10.5705/ss.202015.0404`，原文 §2.1、式(2.2)–(2.3)，构造与参数均值空间正交的 GP，处理均值项与随机残差的混淆。这里的正交性可定义在整个输入区域，而非只在已观测位置。因此“对分类头空间做投影，再放一个 GP”不足以支持原创性。

[Tuo，2019，Adjustments to Computer Models via Projected Kernel Calibration](https://doi.org/10.1137/17M1128769)，[作者原文](https://arxiv.org/abs/1705.03422)，已用投影核和与参数导数相关的正交条件解决校准辨识问题，并讨论半参数效率。其应用和渐近条件与少标签 EEG 不同，但投影核这一数学结构不能重新命名为新方法。

R7 的完整分类头维数大于有效支持样本时，还存在更基本的限制：若支持 head 设计矩阵满行秩，精确商空间信息为零。head ridge 使 Schur 矩阵非零，是引入先验约束后的结果，不能解释成数据独立识别了空间方向。

### 2.2 相容函数集和查询范围已有 Rashomon 文献

[Marx、Calmon 与 Ustun，2020，Predictive Multiplicity in Classification](https://proceedings.mlr.press/v119/marx20a.html) 研究近似同样准确的模型给出互相冲突预测的现象及其精确计算。

特别重要的是截至本次日期已经发表的 [Nguyen 等，2026，When Does Model Multiplicity Affect Prediction Intervals? A Sharp Phase Transition](https://proceedings.mlr.press/v337/nguyen26c.html)，**UAI 2026，PMLR 337:4873–4889**。原文 §3.1 定义1–2、式(2)–(3)已定义目标近最优集合及其查询预测最小/最大值；§4 定理3、式(4)–(5)给出 ridge 椭球及精确区间。该文的主要覆盖率结果还另有 conformal 校准条件。**因此 R10 的目标次水平集和查询区间都必须明确归入已有 Rashomon/profile 思路。**不能把模型集合内的范围直接称作未来标签的95%预测区间。

[Haghighat、Anahideh 与 Rudin，2026，Resolving Predictive Multiplicity for the Rashomon Set](https://ojs.aaai.org/index.php/AAAI/article/view/41076)，DOI `10.1609/aaai.v40i44.41076`，使用离群点处理、局部修补和模型配对协调减少预测分歧。其局部修补依赖验证集邻域标签，不能直接迁移成在目标 query 上重新校准的许可。

### 2.3 kernel innovation 也是经典量

`k(q,q)-k(q,S)K_SS†k(S,q)` 是 RKHS 投影残差范数的平方，在核插值中对应 power function，在无噪声 GP 中对应条件方差。[Kanagawa、Hennig、Sejdinovic 与 Sriperumbudur 的专著预印本](https://arxiv.org/abs/2506.17366)（2025首版，**2026-09-09更新v3**）及其 [2018综述](https://arxiv.org/abs/1807.02582)系统讨论这类等价性。R10 可利用它保留未被支持位置识别的函数自由度，但不能声称该 Schur 补或 GP/RKHS 等价性是新推导。

### 2.4 基线相对悲观决策也有先例

[Loog，2016，Contrastive Pessimistic Likelihood Estimation for Semi-Supervised Classification](https://doi.org/10.1109/TPAMI.2015.2452921)，[原文](https://arxiv.org/abs/1503.00269)，原文 §3–4 已把监督解明确放入对比悲观目标，以控制半监督估计相对原解的训练集对数似然变化。它的保证对象是完整训练集，不是任意未来风险；R10 也必须保留这种保证范围的区分。

[Nematollahi、Jafari Jozani 与 Jafaraghaie，2021，A Unified Theory for Robust Bayesian Prediction Under a General Class of Regret Loss Functions](https://www3.stat.sinica.edu.tw/sstest/j31n4/j31n406/j31n406.html)，DOI `10.5705/ss.202018.0063`，原文 §2 的定义2.1把后验 regret 的 Γ-minimax 作为预测原则。其 regret 是相对每个先验的 Bayes 解，而非固定 EEG anchor；但 minimax regret 和对不确定集合的稳健预测不是新原则。

## 3. 更广的相关文献矩阵

下表各行只给出与本项目有关的范围差异；不把不同数据划分、校准预算或使用 query 输入更新的论文数字当作本项目可直接比较的性能基线。

| 领域与真实论文 | 已有内容 | R10 必须保留的区别 |
|---|---|---|
| [Lawhern等，2018，EEGNet](https://pubmed.ncbi.nlm.nih.gov/29932424/)，DOI `10.1088/1741-2552/aace8c` | 紧凑卷积 EEG 解码器 | 冻结并复用网络是既有设计，不是新架构贡献 |
| [He与Wu，2020，Euclidean alignment](https://pubmed.ncbi.nlm.nih.gov/31034407/)，DOI `10.1109/TBME.2019.2913914` | 通过均值协方差实现欧氏空间数据对齐 | 本项目 Support-EA 只能使用真实已采集支持输入，不能把原论文中更广的目标数据使用方式带入冻结 query 协议 |
| [Rodrigues、Jutten与Congedo，2019，Riemannian Procrustes Analysis](https://pubmed.ncbi.nlm.nih.gov/30596565/)，DOI `10.1109/TBME.2018.2889705` | 黎曼空间域对齐 | 几何对齐及旋转不能单独作为创新；必须公平对照 EA |
| [Ahmed等，2026，EEG数据对齐比较](https://pubmed.ncbi.nlm.nih.gov/42368409/)，DOI `10.3389/fnsys.2026.1840121` | 比较和优化 EEG 域对齐 | 表明对齐仍是活跃强基线领域，不支持“增加几何复杂度必有收益” |
| [Finn、Abbeel与Levine，2017，MAML](https://proceedings.mlr.press/v70/finn17a.html) | 学习易于少步适配的初始化 | episodic meta-learning 已有；R1 也已实现元学习主线 |
| [Raghu等，2020，ANIL](https://arxiv.org/abs/1909.09157) | 只适配任务头并分析特征复用 | “冻结编码器、只更新小头”并非原创 |
| [Hu等，LoRA，2021预印本/ICLR2022](https://arxiv.org/abs/2106.09685) | 冻结预训练模型、低秩参数增量 | 低秩参数化本身不保证少支持样本下的可辨识性或泛化 |
| [Wimpff等，2025，Longitudinal EEG fine-tuning](https://arxiv.org/abs/2502.06828) | 因果纵向、多会话微调及在线测试时适配 | 其更新历史和数据使用方式需与本项目单次前缀校准区分 |
| [Wang等，2025，TCPL](https://pmc.ncbi.nlm.nih.gov/articles/PMC12682862/)，DOI `10.3389/fnins.2025.1689286` | 任务条件 prompt 与 few-shot EEG meta-learning | EEG + prompt/PEFT + 元学习的组合也已有实质先例 |
| [Li等，2026，Co-adaptive Few-shot Calibration](https://pubmed.ncbi.nlm.nih.gov/42776881/)，DOI `10.1109/JBHI.2026.3735157` | 受试者相关 MI 模式选择和少样本元校准 | 2026-09-23已在线发表；不能宣称首次少样本自适应 EEG 校准 |
| [Miao等，2026，Meta-learning multi-source domain adaptation](https://pubmed.ncbi.nlm.nih.gov/41825840/)，DOI `10.1016/j.jneumeth.2026.110742` | 元学习与多源 EEG 迁移 | 多源、meta、domain adaptation的普通组合不构成R10原创核心 |
| [Rasmussen与Williams，2006，GPML第3章](https://gaussianprocess.org/gpml/chapters/RW3.pdf) | GP分类、Laplace近似和预测积分 | R9的GP/MAP/Laplace及方差属于这些基础工具 |
| [Farajtabar等，2020，Orthogonal Gradient Descent](https://proceedings.mlr.press/v108/farajtabar20a.html) | 投影新任务梯度以保护旧任务输出 | Jacobian投影与保护旧函数值不是新原则；旧任务输出保护不等于未来会话风险保证 |
| [Titsias、Schwarz与de G. Matthews，2020，Functional Regularisation](https://arxiv.org/abs/1901.11356) | GP后验函数信念及持续学习函数空间正则 | 使用函数空间而非参数空间保护本身已有 |
| [Niu等，2022，EATA](https://proceedings.mlr.press/v162/niu22a.html) | 可靠样本筛选与Fisher防遗忘约束 | 样本可靠性门控及Fisher保护不能直接作为创新；该方法通常会更新测试数据，与本项目 query 禁更新不同 |
| [Tan等，2024预印本/2025修订，EATA-C](https://arxiv.org/abs/2403.11491) | 区分模型/数据不确定性并校准TTA | “利用不确定性决定更新”这一口号已被覆盖 |
| [Cawley与Talbot，2010，Model Selection Bias](https://www.jmlr.org/papers/v11/cawley10a.html) | 有限样本选择准则自身可被过拟合 | 支持内部CV不自动消除选择偏差；反复检索已用holdout更不能视为确认 |
| [Yao等，2018，Using stacking to average Bayesian predictive distributions](https://arxiv.org/abs/1704.02030)，DOI `10.1214/17-BA1091` | 基于proper scoring rule的交叉验证预测分布组合 | OOF、概率融合和stacking并非原创，R4亦已有 |
| [Chernozhukov等，2018，Double/debiased ML](https://arxiv.org/abs/1608.00060)，DOI `10.1111/ectj.12097` | 正交分数及cross-fitting用于结构参数推断 | 不能把“正交 + cross-fit”借名为新的分类泛化定理 |
| [Henzi与Ziegel，2022，Sequential forecast inference](https://arxiv.org/abs/2103.08402)，DOI `10.1093/biomet/asab047` | proper-score差异的序贯e-value推断 | 若选候选B，必须正确写明条件forecast-dominance null，并核对2022更正 `10.1093/biomet/asac043`；20个相关采集位置不自动保证未来会话有效 |
| [Geifman与El-Yaniv，2019，SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) | 联合学习分类和拒绝 | R10保留anchor预测、不拒绝输出；须报告所有query上的BA/NLL，不能仅报告接受更新子集 |
| [Angelopoulos等，2024，Conformal Risk Control](https://arxiv.org/abs/2208.02814) | 在相应校准条件下控制期望单调损失 | R10的模型集合一致性不是conformal coverage；反复用于开发的内部holdout也不能替代独立校准样本 |
| [Wang等，2026，Prediction Fragmentation Controls TTA](https://arxiv.org/abs/2609.20700) | 用源/适配预测分歧几何决定病例更新 | 泛泛的输入相关router已被覆盖；R10应以函数相容集合上的优化问题定义动作，而非另加经验分数门控 |

## 4. 三个候选的淘汰与推荐

| 候选 | 真正新增部分的可能范围 | 数学与现有证据的关系 | 能否被证伪/超过强基线 | 成本及决策 |
|---|---|---|---|---|
| A：精确 head 商空间的空间校准 | 在指定数据/头限制下设计可辨识子空间；不是Schur或正交核本身 | full head 满行秩时严格商空间为零；有限ridge不是独立信息；与R3/R7重叠明显 | 可用合成head吸收例和真实有效秩直接证伪；不能仅把退化成EA当性能胜利 | 小矩阵低成本，但原创性和非零有效更新空间都弱；**不选为主方法** |
| B：可预测采集前缀的证据过程 | 把更新决策与逐采集位置可预测证据预算连起来；不是e-value/OOF本身 | 可在明确条件null下研究可选停止；小样本和会话漂移使统计解释困难 | 可测20位置下的检测力与负迁移，必须检验是否总不更新 | 串行前缀拟合有额外成本，假设风险高；**保留为后续候选，不以复杂统计承诺包装主贡献** |
| C：支持证据保留约束下的完整查询函数歧义与anchor-relative proper-loss动作 | **同一个支持目标增益约束**同时决定可接受函数完成、查询歧义、可证外界与相对anchor动作；不是把几种可靠性分数相乘 | 不靠独立空间参数可辨识性；显式保留支持之外的自由度，响应R7所示“支持拟合好但未来方向未识别”问题 | 去掉query completion、改用MAP/R9方差门控/局部二阶区间均是直接消融；必须在完整query上胜过EA，或至少给出真实成本下可重复的NLL/负迁移收益 | 支持n≤20；完整区间需两侧profile，主实现只求决策所需的单侧外界；零更新证书、warm-start与dual外界需真实计时。**推荐作为分阶段实验主算法，原创性和性能均待验收** |

候选C的合理参数化是 `G=J(0)-J(fhat)`、`C_c={f:J(f)≤J(fhat)+cG}`。精确优化时`c=0`、`c=1`分别覆盖MAP和完整anchor退化；有限拟合下`c=0`仅为不继承profile证书的直接MAP消融。c增加使精确集合嵌套，并使精确minimax动作的更新幅度单调不增；有限预算自适应求界的实际近似动作不保证跨c普遍单调。固定kernel几何、c和求解规则只由meta_train/meta_validation开发并冻结。目标前B个采集位置是唯一允许进入拟合的本地标签。

最有希望的算法贡献应围绕一个统一机制表述：**保留规定比例支持证据增益的函数集合，对每个查询保留完整未识别函数完成空间，利用可证的凸profile外界决定相对anchor的proper-loss动作。**相对基础对象，新增的是具体问题的耦合、有限支持可计算解、零更新证书、数值实现与系统证伪，而不是新发明一个统计名词。

## 5. 应允许和不应允许的论断

可证明：在事先定义且数值正确包住的模型相容区间内，所选动作对所有区间概率的期望proper loss不劣于固定anchor；在查询零交叉核或明确零可行性证书成立时，输出严格恢复anchor；证据容忍度增大时，精确minimax动作的更新幅度不增；这不自动适用于有限预算的自适应近似动作。

不可从上述性质推出：真实未来条件概率在该区间内、未来BA/NLL必然提升、跨会话漂移下覆盖率有效、逐位数值结果与GP相同、或发表价值已经成立。尤其不能把原型的保守不更新描述成性能突破。

对于原创性，本次检索未发现与“gain-relative logistic RKHS完整查询完成 + certified profile outer bounds + 固定anchor-relative proper-loss动作”完全同构的已发表算法。但是这只是**截至2026-10-26的有界检索结论**。若后续审稿或扩展检索发现同构先例，应缩小贡献或放弃原创算法主张。

## 6. 获取与核查证据

- `literature/primary_source_fetch_manifest.json`：逐次下载URL、HTTP状态、字节数、SHA256、检索UTC时间和本地审计缓存路径。HTTP200但实际是cookie challenge的页面被单独标注，不能当作原文证据。
- `literature/pubmed_primary_bibliography.json`：NCBI官方XML提取的作者、题名、年份和文章级DOI；刻意排除了参考文献的ArticleId，避免把引用论文DOI误配给本文。
- 完整HTML/PDF仅缓存于 `/tmp/r10-lit-primary` 供审计；仓库交付包含引用、哈希与自己的有限摘要，不重新发布版权原文。
- 原文重点细读包括：Plumlee2018§2、Tuo投影核段落、Nguyen2026§3–4及后续conformal边界、Loog2016§3–4、Haghighat2026局部验证标签使用、Nematollahi2021§2。

这些文献记录不替代R1–R9的原始CSV/JSON统计审计，也不构成R10真实EEG效果证据。


数值与标签边界补注：c=0在生产代码中是直接MAP消融，有限拟合时标注`not_profile_certified`；只有精确极小值下的数学集合才退化为单点MAP。实数算术的对偶外界证明不等于NumPy float64/padding对所有机器输入的形式化IEEE区间算术证明。开发meta_validation query标签可用于固定selection/check并单独记true；目标query标签不参与episode拟合或R10配置选择，但历史旧cohort结果已影响研究设计，所以旧cohorts始终探索性。
