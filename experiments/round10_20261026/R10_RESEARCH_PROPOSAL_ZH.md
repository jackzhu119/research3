# R10 研究提案：由支持证据决定查询函数修正

研究与文献核查截止日：**2026-10-26**。R2–R8初查及早期R9审计固定于 [`paper3-results@68a8d5c5714ad2a4796bfc21b9203ed29027878d`](https://github.com/jackzhu119/research3/tree/68a8d5c5714ad2a4796bfc21b9203ed29027878d)；交付前R9补审另固定于`a752d00a7b56972d67e1786eb122b218828e3616`，见第2节。本文提出并实现一个待真实 EEG 实验证伪的算法；**没有宣称 R10 已优于 EA，也没有把模型集合内的数学性质当作未来会话风险保证。**

完整原始证据、文献矩阵与公式分别见 [R2–R7 审计](research/evidence_r2_r7.md)、[R8/R9 审计](research/evidence_r8_r9.md)、[文献核查](research/literature_review.md)和 [R10 数学定义](R10_MATH.md)。本文给出科研决策及其可检验依据。

## 1. 要解决的具体问题

最大校准预算只有真实前20个采集位置，经 QC 后有效标签通常更少。此时“支持集拟合改善”与“未来预测值得改变”是不同问题。一个高容量分类头可以吸收全部支持 margin，空间参数或残差函数仍可能在未来 query 上产生完全不同的修正。增加优化步数、几何自由度或额外模块，不能自动增加可辨识证据。

R10 研究：**在保留明确比例的已观测支持拟合增益时，所有仍然相容的预测函数，是否都要求某个查询偏离强 anchor？若不是，保留 anchor；若是，只执行仍被这些函数共同支持的修正。**

这里的 anchor 主线是 Support-EA 后的冻结 EEGNet。EEGNet、EA、开发集特征白化、残差 logistic 核模型均是既有工具；它们本身不是本研究的原创贡献。目标是一个统一的、可计算并可逐机制否证的更新原则，而非再添一个经验不确定性权重。

## 2. 原始结果实际上支持什么

### 2.1 数据身份、完成状态与统计口径

本次读取实际 CSV、JSON、任务 manifest、收据和算法源码，校验分片与整文件 SHA256，而非从 README 摘要推断。R2–R7 的论文数据均已取得；R4–R6 读取的是最新完整 v2 论文分析包。R8 已完成 **2835/2835** 任务，最新 v2 论文包有33文件、545,964,970字节，原40项主要比较能够从参与者 summary 独立精确复算。

R9状态与分析上传不同步：初查03:51:48Z监控为774/1548，core234/234、stress432/432，论文report仍只有120任务。初次从Channel B恢复39个GP任务档案、521,741,539压缩字节；交付前又从新固定提交`a752d00a7b56972d67e1786eb122b218828e3616`补回**33个RBF任务档案、429,807,754压缩字节**，逐会话核对预测、log概率及query签名。故本次既核查raw linear，也已核查完整三种子的raw RBF anchored对EA；不能再沿用“RBF全部未取得”的旧判断。原24项主要比较仍有4项EA-RBF相关档案缺失，剩余缺档不代表无效或无增益；保守缺项处理与完整原始效应分别保留。

R1仍是部分完成的元学习校准主线。最新只读快照为2026-10-26T04:58:11Z：meta_screening完成38/45、meta_selection3/5、meta_engineering5/5；已观察主包装动作52/61，但这是动态依赖包装动作数，不能解释为全部原研究完成52/61。三个Slurm作业RUNNING，其中两个槽位有4–5秒内的新训练进度（外层更新673和915），另一个无具体训练观测；`all_original_complete=false`，两扩展臂未开始。因此R1不能作为已完成的主要对照结论，其未完成结果不用于挑选R10超参数。原始状态与SHA见`evidence/late/R1_LATEST_STATUS*.json`。

R10对全部方法统一采用`1e−12`概率floor计算balanced NLL，并保留该评价版本身份。历史轮次若使用未截断log-probability/BCE-with-logits，其NLL标量不能与R10截断后的NLL直接混合；需要从身份匹配的原预测/log-probability按共同规则重算，保留原结果，不覆盖历史文件。BA配对仍须同时满足source、支持/query UID与聚合身份。

下表均为预算 B20、方法减基线的参与者等权 BA 差，单位为百分点（pp）。内部 holdout 为11名参与者、105会话，Pan为14名参与者、28个 published-order 会话，Weibo为10名参与者。先在参与者与种子内平均会话，再平均三个种子；种子和会话不充当独立参与者。R2–R7 使用20,000次参与者 bootstrap，R8/R9 使用10,000次；p 为两侧精确配对 sign-flip，依赖零假设下参与者差值符号对称。CI 与检验不是互相反演，边缘结果不能只看 CI 判显著。所有已反复用于研发的 holdout、Pan、Weibo 均为**探索性证据**；Pan/Weibo 的 published order 也不能自动等同已核实的物理采集时间。

### 2.2 历史主要比较：保护有用，但强基线仍未被超过

|轮次与比较|队列 / N|ΔBA(pp)|95%CI|原始 p|原家族 Holm p|
|---|---|---:|---|---:|---:|
|R2 dual − P|holdout / 11|−0.0081|[−0.0207, 0]|0.5000|1|
|R3 profiled − SGD100|holdout / 11|−5.1090|[−7.2518, −2.9517]|0.002930|0.029297|
|R3 protected − profiled|holdout / 11|+4.6589|[2.3965, 6.9094]|0.004883|0.043945|
|R3 information-protected − information|holdout / 11|+4.6430|[2.4077, 6.8751]|0.004883|0.043945|
|R4 xfit-select − profiled|holdout / 11|+1.7458|[0.9930, 2.5579]|0.000977|0.011719|
|R4 select-stack − xfit-stack|holdout / 11|+0.5592|[0.3187, 0.8388]|0.001953|0.021484|
|R5 left-rotation − single|holdout / 11|+0.0257|[0.0047, 0.0467]|0.062500|0.500000|
|R5 symmetric − single|holdout / 11|0|[0, 0]|1|1|
|R6 `map:uq_joint−map:uq_map`（两者均MAP）|Pan / 14|−0.0357|[−0.0952, 0.0238]|0.453125|1|
|R6 `joint:uq_joint−map:uq_joint`（目标推断比较）|Pan / 14|−0.0119|[−0.0595, 0.0357]|1|1|
|R7 spatiotemporal − static|Pan / 14|−0.1190|[−0.4762, 0.2381]|0.586670|1|

原比较家族大小分别为R2/R3/R4/R5的10/10/12/8，R6为8项BA加8项NLL，R7为12项BA加12项NLL。

**R6的比较对象必须写全。** Pan B20、14名参与者下，`map:uq_joint`（joint训练几何、目标仍做MAP校准）相对`map:uq_map`的balanced NLL从0.870367393降至0.869336720，改善量0.001030673，95%CI[0.000743873,0.001333462]，原始p=.00024414、原16项家族Holm=.00341797；对应BA差为−0.035714pp、p=.453125、Holm1。这里的改善是训练几何的比较，不能称为目标joint posterior优于MAP。真正目标推断的`joint:uq_joint−map:uq_joint`，NLL仅从0.869336720降至0.869272302，改善0.000064418，CI[−0.000138584,0.000347287]，p=.809814、Holm1；BA差−0.011905pp、p=1，均无显著优势。原R6 NLL表的改善量方向为baseline减candidate；R10新协议的NLL差统一candidate减baseline，符号不能直接混用。R6的这些NLL仍明显差于同队列EA的0.63038335。R7全24项原主要比较最小Holm=.134765625。

**R4 存在已经源码确认的 EA 参考分派错误。**任务标签为 `support_EA`，但 `_configuration` 使用 `setdefault`，没有覆盖原配置中已存在的 `head_newton`；三种子收据都保留该实际配置。因此该行 holdout BA56.8531%、NLL0.986081与 head_newton 完全相同。R5 的正确 EA 分派修复及对应代码身份已记录；在匹配源 checkpoint、输入与 query 签名下，正确 EA 为 BA**62.2132%**、NLL**0.646613**。不能用错误 R4 EA 行证明超越强基线，也不覆盖或改写历史结果文件。R4 内部交叉拟合比较仍可作为其指定方法之间的证据。

在身份严格匹配后的**新增事后探索比较**中，R4 select-stack 相对正确 EA 为 **−1.5390pp**（95%CI[−2.1961,−0.8119]，原始 p=.005859，11人中10人负差）；相对 SGD10 为−0.8168pp（[−1.7962,−0.0603]，p=.099609）。R3 protected 相对 SGD100 为−0.4501pp（[−0.8571,−0.0701]，p=.064453）。这些不是历史主要家族的预注册检验，不借用旧 Holm 显著性。R5 single 相对 EA −5.3215pp、R6 `map:uq_joint`（joint训练几何+MAP校准）相对EA在holdout −5.2877pp、Pan −3.3690pp；完整 CI、p 和配对文件在证据附录。

所以，R3证明的是**保护可显著缓解强拟合伤害**；R4提供了有限的选择/组合积极结果；R5–R7没有证明增加几何复杂度能超过强 EA。

### 2.3 R7 为何直接指向可辨识性问题

下表为 holdout、B20、315个会话×种子观测，先按参与者等权汇总；它是机制描述，不把315当独立 N。

|head 限制|实际平均参数数|支持 balanced BCE|未来 BA(%)|未来 NLL|data-only 空间有效秩|head 吸收比例|
|---|---:|---:|---:|---:|---:|---:|
|rank0（仍含偏置）|1|0.672126|61.309739|0.657763|4.000000|0.343623|
|rank4|5|0.511141|59.183174|0.813710|3.906730|0.625039|
|rank8|8.906730|0.283337|56.991491|0.968129|3.579693|0.818542|
|full64|65|0.121421|56.891774|0.985351|0|1|

有效支持样本均值仅**15.530106**，中心化特征秩14.530106。增加 head 容量使支持损失骤降，未来 BA/NLL却恶化；full head 能吸收全部支持空间变化。但“支持 Schur 为零”不意味着任意未来 query 没有独立影响，恰恰说明支持拟合不能识别其未来延伸。R10不把R7已有的Schur诊断重新命名为算法创新。

R10的`head_rank4`容量基线由当前支持集的**无标签冻结特征**做PCA，保留至多4个方向并另带bias；不读取query，也不是meta_train预先固定的4维子空间。

### 2.4 最新 R8/R9 改变了哪些判断

|轮次与比较|队列 / N|ΔBA(pp)|95%CI|原始 p|Holm p / 家族|
|---|---|---:|---|---:|---|
|R8 full-bures − EA|Pan / 14|−0.0119|[−0.2262, 0.1905]|1|1 / 原40项|
|R8 full-bures − EA|Weibo / 10|−0.0754|[−0.4325, 0.2857]|0.726562|1 / 原40项|
|R8 fiber-bounded − EA|Pan / 14|+0.2619|[−0.2381, 0.7143]|0.331665|1 / 原40项|
|R8 fiber-bounded − EA|Weibo / 10|+0.3889|[−0.0238, 0.7621]|0.113281|1 / 原40项|
|R8 fiber-bounded − reversed|Weibo / 10|+1.0278|[0.5278, 1.5794]|0.003906|0.156250 / 原40项|
|R9 linear-MAP − EA|holdout / 11|−2.4168|[−3.5029, −1.3986]|0.001953|0.044922 / 事后linear24项|
|R9 linear-anchored − EA|holdout / 11|−0.8713|[−1.7318, −0.0815]|0.071289|1 / 事后linear24项|
|R9 linear-anchored − EA|Pan / 14|−0.2619|[−1.2738, 0.7857]|0.647217|1 / 事后linear24项|
|R9 linear-anchored − EA|Weibo / 10|+0.2897|[−0.9048, 1.4762]|0.658203|1 / 事后linear24项|

R8 原40项主要 BA/NLL 比较**无一通过 Holm .05**，最小值.15625；相对反向/随机几何的未校正改善不等于超过 EA。R9已取得的线性GP说明：局部信息锚定能显著缩小 MAP 的性能退化，但没有得到对 EA 的主要增益证据。Holdout MAP NLL 相对 EA 恶化+0.048236（CI[0.033683,0.063885]，p=.000977、上述事后家族 Holm=.023438）。

R9 raw local-anchored锚定Frozen源预测，其holdout负翻转约1.0092%，低于EA的8.3404%，但纠错率也只有1.2507%，低于EA的9.4532%。**少引错不等于净收益**。R9代码已经有 EA 预处理GP、posterior variance信息比例及anchor概率混合；把这些再加一层权重不构成R10创新。最新raw RBF anchored相对EA的BA差：holdout **−0.5963pp**（95%CI[−1.5680,0.3620]，N=11，p=.262695），Pan **+0.4643pp**（[−0.3214,1.2857]，N=14，p=.308228），Weibo **+0.5000pp**（[−0.5006,1.5714]，N=10，p=.429688）；三者原始p均不显著。Pan/Weibo属于原主要家族的可用项，按原24项、缺档项p=1的保守Holm均为1；holdout为额外探索比较，不能借用原家族显著性。尚缺的EA-RBF相关比较仍须在正式冻结前补审，不能用缺失替代负面结论。

### 2.5 可复算出处

上述数字来自固定提交恢复的原始结果，不是模拟值。关键机器可读材料为：

- [R2–R7主要效应](evidence/early/PRIMARY_EFFECTS.csv)、[补充强基线比较](evidence/early/SUPPLEMENTAL_EFFECTS.csv)、[代码身份](evidence/early/CODE_PROVENANCE.json)、[恢复与SHA审计](evidence/early/DATA_RESTORATION_AUDIT.json)。
- [R8全部40项](evidence/late/R8_MAIN_COMPARISONS.csv)、[独立复算](evidence/late/R8_PRIMARY_REPRODUCED.json)、[R9已取得线性GP比较](evidence/late/R9_B20_AVAILABLE_LINEAR_COMPARISONS.csv)、[逐参与者指标](evidence/late/R9_PARTICIPANT_METRICS.csv)。
- [R9档案恢复审计](evidence/late/R9_GP_ARCHIVE_RESTORE_AUDIT.json)、[覆盖缺口](evidence/late/R9_UPLOAD_COVERAGE.json)、[计划与报告身份](evidence/late/PLAN_REPORT_IDENTITY_CHECKS.json)。`research/`中的读取、恢复、分析脚本可重复这些统计。

## 3. 原创性审查与候选淘汰

检索覆盖 EEG 会话适配/对齐、PEFT/meta-learning、GP/RKHS、参数可辨识性、函数保护、预测多解、不确定性适配、交叉拟合和选择性预测。使用出版页、作者预印本和原文；逐条URL、SHA、日期及节/公式位置见[文献矩阵](research/literature_review.md)和[来源manifest](research/literature/primary_source_fetch_manifest.json)。HTTP200的cookie挑战页有明确标记，不充当论文证据。

### 3.1 不能声称新发明的部分

1. **投影和Schur辨识性**：Plumlee与Joseph的[Orthogonal GP (2018)](https://www3.stat.sinica.edu.tw/sstest/j28n2/j28n23/j28n23.html)，§2.1式2.2–2.3，以及Tuo的[Projected Kernel Calibration (2019)](https://doi.org/10.1137/17M1128769)，已讨论均值/参数导数空间与GP残差的混淆及投影。R3/R7也已实现相关机制。
2. **近最优模型集合和查询范围**：Marx等[Predictive Multiplicity (2020)](https://proceedings.mlr.press/v119/marx20a.html)已有预测多解；特别是Nguyen等[UAI2026/PMLR337](https://proceedings.mlr.press/v337/nguyen26c.html) §3.1定义1–2、式2–3已给出目标次水平集与查询最小/最大预测，§4给出ridge精确范围。Haghighat等[AAAI2026](https://doi.org/10.1609/aaai.v40i44.41076)已有局部模型协调。R10不能把次水平集或区间本身称为原创。
3. **核query innovation**：`kqq−kqS KSS† kSq`是经典RKHS power function/无噪声GP条件方差；见Kanagawa等[2025专著、2026-09-09 v3](https://arxiv.org/abs/2506.17366)。它不是新统计量，也不同于R9的Laplace后验方差。
4. **相对基线的悲观优化和regret**：Loog的[CPLE (2016)](https://doi.org/10.1109/TPAMI.2015.2452921)已有监督基线相对悲观似然；Nematollahi等[Robust Bayesian Prediction (2021)](https://doi.org/10.5705/ss.202018.0063)已有稳健regret预测原则。R10的clip闭式动作不是独立的原创性充分条件。
5. **小头、元学习、方差门、交叉拟合**：ANIL、LoRA、GPML、EATA、stacking及R4均构成直接先例。2026年已有[Co-adaptive Few-shot Calibration](https://doi.org/10.1109/JBHI.2026.3735157)、[meta多源EEG适配](https://doi.org/10.1016/j.jneumeth.2026.110742)，及输入相关的[Prediction Fragmentation TTA](https://arxiv.org/abs/2609.20700)。不能声称首次few-shot EEG或首次按查询可靠性决定适配。

### 3.2 三个技术不同的候选

|候选|潜在新增问题|数学风险与直接证据|成本/复现|决定|
|---|---|---|---|---|
|A：精确head商空间中的空间更新|只保留不能被head吸收的空间方向|head满行秩时`Aᵀ(I−ZZ†)A=0`；有限ridge产生非零信息是先验约束，不是独立观测。R7已显示此退化；正交GP又构成文献重叠|小矩阵便宜，但可能没有非零可识别方向，易退化成旧方法|放弃作为主创新；保留数学反例和容量基线|
|B：按采集前缀构造可预测证据过程|每个标签揭示前固定预测，以序贯证据决定更新预算|[Henzi与Ziegel 2022](https://doi.org/10.1093/biomet/asab047)已有proper-score/e-value；R4已有OOF。20个相关采集位置下检测力弱，条件forecast null不等于未来会话风险，亦须核对原文更正|逐前缀重拟合且假设多，难在低成本下证明优于EA|保留后续研究，不借e-value名义承诺泛化|
|C：证据保留的完整查询函数校准|同一支持目标增益约束完整query函数完成，再以有证书的外界作anchor相对动作|响应R7的支持吸收及R9的保护/收益折衷；不依赖空间参数可独立辨识。完整logistic目标、未识别completion和有限求解证书共同决定一个动作|n≤20，但逐query profile可能昂贵；CPU float64、单侧对偶和零更新证书可降低成本，须真实计时|**主选：先数学/软件，再开发集机制，失败即停止**|

有限检索未找到与C完整构造完全同构的已发表算法；这不是世界首次的证明。合理贡献范围是**特定EEG校准问题的联合结构、可计算解及其机制证据**。若后续发现同构先例，必须缩小原创性主张或放弃算法创新定位。

## 4. 主算法：证据保留的查询函数校准

### 4.1 一个目标、一个相容集合、一个决策

冻结Support-EA/EEGNet给出margin `m0`，在meta_train冻结白化后的特征上使用标准线性加截距核。实际主线不新增几何编码器，也不以EA anchor作原创点。支持两类均存在时，权重固定为 `w_i=n/(2n_yi)`，总和为n；λ=1。

\[
J(f)=\sum_i w_i\{\operatorname{softplus}(m_{0i}+f(x_i))-y_i(m_{0i}+f(x_i))\}
+\frac\lambda2\|f\|_{\mathcal H_k}^2.
\]

从零修正拟合实际可行的`fhat`，要求`Jhat≤J0=J(0)`。记`G=J0−Jhat`，固定`0<c≤1`：

\[
\mathcal C_c=\{f:J(f)\le\widehat J+cG\}.
\]

每个成员至少保留`1−c`比例的已观测regularized support gain。对query q求该集合上完整函数值范围`[lq,uq]`，映成概率范围`[Lq,Uq]`。最终动作解：

\[
\min_p\sup_{\eta\in[L_q,U_q]}
\{\operatorname{CE}(\eta,p)-\operatorname{CE}(\eta,p_0(q))\},
\qquad p^*(q)=\operatorname{clip}(p_0(q),L_q,U_q).
\]

只有**所有相容函数都把query推向anchor同一侧**时，才有非零修正。这不是把MAP均值乘一个方差权重，也不使用query标签估计融合系数。

### 4.2 非平凡部分在完整query完成与有限求解证书

支持Gram分解后，完整问题可写成最多`n+1`维：

\[
f_S=Bv,\quad f(q)=a_q^\top v+\sqrt{s_q}t,\quad
F(v)+\lambda t^2/2\le T,
\]

其中`sq=kqq−||aq||²`保留支持外未识别函数自由度。若只拿MAP支持张成空间做区间而设置`t=0`，会删除本研究要检测的query歧义。实现采用声明的UID nugget与完整SPD Cholesky，不把正特征值的数值截断冒充精确零空间。

一个无需端点求解的**零更新证书**为：令`μq=aqᵀvhat`，当`sq>0`且

\[
\lambda\mu_q^2/(2s_q)\le cG,
\]

存在保持所有支持预测但令该query修正为零的相容函数，必须返回anchor。其他query用严格凸的单侧倾斜logistic问题求对偶外界；λ强凸性给出梯度残差修正，使有限Newton迭代仍产生保守界。近似可行端点只是内界，不能当稳健外界。

实际单侧界足够决定动作：松界让修正更接近anchor，计算预算耗尽或证书不一致时返回anchor并记录。完整公式、单调root、KKT、残差证书、数值边界与伪代码在[R10_MATH.md](R10_MATH.md)。这些优化工具本身已有先例，贡献检验聚焦整个联合机制是否必要且有效。

### 4.3 能证明和不能证明什么

可严格证明：对定义好的模型相容概率区间内每个η，精确minimax动作的期望log loss不高于anchor；扩大c使**精确区间与精确决策**嵌套并使修正幅度不增；c=1、零交叉核和零更新证书等情形恢复anchor。有限求解只要返回有效保守外界，仍保留逐查询的同一集合内相对损失性质；但自适应tilt和有限预算可能产生松紧不同的界，**不保证实际近似动作跨c普遍单调**。模型集合没有被证明包含真实未来概率，因此**不能推出未来BA/NLL改善或临床安全**。NumPy float64加保守padding也不是形式化IEEE区间算术证明；CPU合成测试不能证明所有机器浮点输入上的外界都保守。

支持拟合系数r≤n≤20，不能误称整个线性函数类只有r个自由度；原特征函数类仍可有d+1维。MAP每次Newton约`O(nr²+r³)`；未被零证书跳过的query再乘倾斜次数和Newton次数。需包含冻结encoder、数据传输和CPU求界的端到端时间，不能仅报告小矩阵理论成本。

## 5. 怎么让实验真正能否证核心机制

保持既有冻结预处理、source checkpoint与数据划分。真实前20个采集位置为最大支持；预算4/8/12/16/20；固定未来query集合。QC不补采未来位置；query不更新归一化、EA、特征白化或模型，也不选择超参数。每个固定query的预测不随其他query的加入、删除或重排改变。meta_train/validation负责开发；holdout、Pan、Weibo只能报告探索性；独立确认须注册此前未用于本算法设计的新数据及身份。

这里严格区分标签角色：每个目标episode的`fit/predict`均不接受其query标签；meta_validation的开发query标签可以用于预先规定的selection/check，因此记录`development_query_labels_used=true`；正式目标query不参与R10配置选择，记录`target_query_labels_used_for_selection=false`。另一方面，R1–R9旧holdout/Pan/Weibo结果已经影响本轮研究动机与算法设计，这一历史不能被后一个字段抹去，所以旧cohorts仍全部属于探索性证据。该字段不表示研究者从未看过任何历史目标结果。

强基线必须包括Frozen、正确执行的Support-EA、低容量与完整head-only、现有SGD10/100空间校准、R3保护方法、R4 select-stack、R8有完整证据的代表性方法、R9原实现及其已有EA版本。R4错误EA行排除；R9覆盖不足的方法在全档案补审后决定代表性，不默认缺失等于无效。历史重用预测只在checkpoint、输入、query、方法及代码身份一致时合法。

对C设置直接机制消融：

- **matched MAP**：同一anchor、核、特征、λ和标签权重，只去掉稳健集合动作，隔离支持拟合与保守预测的区别；`c=0/map`明确不继承有限求解的profile证书。
- **无query completion**：人为置`s=0`，检验未识别函数完成是否真的影响未来负迁移。
- **二阶近似**：用局部椭球替代完整logistic profile，检验非二次证据集合是否必要。
- **R9 variance gate替代**：其余匹配，检验对偶相容动作相对普通方差混合是否有独立收益。
- **c=1/完整EA**及Frozen-anchor：检验是否只是几乎处处复制EA、是否依赖强anchor。

主要终点是参与者等权BA；辅以balanced NLL、Brier、参与者负迁移比例、纠错/引错、全部query上的更新比例、证书类型、运行时间与真实函数容量。不得只在接受更新的query子集上汇报。主比较家族及多重校正规则在正式运行前锁定，参与者为独立单位，报告配对效应、CI及Holm校正；统计成功不能由显著性与非显著性混用构成。完整方案见[R10_EXPERIMENT_PLAN.md](R10_EXPERIMENT_PLAN.md)。

## 6. 分阶段成本控制与完成条件

|阶段|执行内容|完成条件|停止条件|
|---|---|---|---|
|1 数学/软件|合成可辨识与零空间反例、梯度、证书、退化、身份及无泄漏测试|全部必要CPU测试通过；精确小例与数值证书吻合；假设和fallback可审计|证书错误、依赖query标签/批次、身份不可追踪即阻断|
|2 开发机制|仅meta_train/validation，小网格和严格任务上限；同权重MAP及机制消融|更新非平凡、至少一个新增机制有可重复开发收益且成本可接受|处处复制EA、只改善支持损失、无独立机制收益或成本超限即停止扩展|
|3 冻结|冻结核/规则/主比较、统计方案、代码与新数据资格|只用开发数据选一次；签名与manifest固定；完成R9缺口补审|依赖旧目标query调参或缺乏合法独立确认资格，不进入确认|
|4 正式配对|探索复用队列与独立新数据分开执行、同支持/query配对|完整任务/预测/收据、双通道归档，按固定家族报告|不据正式query结果改方法；失败如实报告|

初始候选网格幅度`τ∈{.3,1,3}`、证据比例`c∈{.01,.1,.5}`，λ=1，标签权重和为n。这些范围来自合成数值/检测力检查，**不是搜索holdout、Pan或Weibo后得到的参数**；最终值仍须由合法meta开发流程锁定。核主线固定为meta_train白化的线性加截距，避免同时展开多种复杂几何。正式目标不是重复R8的数千任务穷举。

当前已实现算法、与前轮原生代码对照的基线接口、真实数据worker和HPC平台代码；CPU数学/合成/接口验证与实际EEG实验分开记录。学校GPU预检和正式实验尚未执行；本次未提交、取消Slurm作业或占用GPU。学校正式资源配置固定为**4个并行Slurm作业，每个作业1张GPU、16核CPU、68G内存，队列`hpc_gpu`**；总计4张GPU、64核CPU、272G申请内存。四个任务各申请一个单卡allocation，不要求全部放在同一节点；实际启动仍取决于学校调度配额和资源可用性。此处是后续提交配置，本次未自动提交或取消作业。安装、验证、显式`--submit`及双通道上传的准确命令见[R10_IMPLEMENTATION.md](R10_IMPLEMENTATION.md)。网络失败不得终止训练，完成任务按身份复用，科学结果与运行状态独立保存。

## 7. 预期贡献与投稿判断

R10候选贡献可以清楚表述为：**以支持目标的实际证据增益定义一个完整函数相容集合，显式保留支持不可辨识的query完成方向，利用有数值残差证书的单侧凸profile决定相对强anchor的最小可靠修正。**核心不是新发明GP、Schur补、Rashomon集合、交叉拟合、minimax或clip，而是它们针对本问题形成的一个具体、可解、可审计的联合算法及其机制证据。

现有实验充分支持“小支持下容量与未来可靠性脱节”的动机，也支持“减少伤害可能同时丢掉有效纠错”的反例；它们尚未提供R10真实效果。若新方法只有相对失败MAP的改善、没有超过EA，或几乎总退回EA，不能包装为主指标突破。若在此前未用于设计的新数据上，按锁定家族获得对EA的有意义配对BA优势，并通过query完成/完整profile/稳健动作消融及成本验收，才有较强的算法SCI投稿基础。若只能稳定改善NLL或负迁移，应收窄论文主张，并先确认收益不来自牺牲有效纠错或增加不可接受成本。

当前状态是**原创机制的可复现候选与已实现实验系统**，不是已成立的性能突破或录用价值保证。失败结果同样可证伪该候选，下一步应严格执行开发阶段停止条件，再决定是否投入独立确认。
