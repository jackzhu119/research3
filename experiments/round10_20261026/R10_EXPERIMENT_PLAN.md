# R10 实验计划：分阶段检验查询函数证据校准

## 1. 实验目的与目前边界

R10的主假设是：少量支持标签虽然可以降低训练目标，却未必能确定某个未来查询的预测修正。保留规定比例支持目标增益的完整函数集合、显式支持外函数完成及anchor-relative稳健决策，能否比Support-EA和已有GP/SGD方法带来更好的参与者等权BA，并减少NLL恶化与负迁移？

本计划不预设算法优于EA，也不把R1–R9已反复使用的holdout、Pan、Weibo当作独立确认。R8/R9的新证据、R4 EA调度问题及相关既有方法差异，须按证据审计文件解释，不能仅复述旧README。R10的OOF/one-SE选择规则、核函数、GP/MAP、Rashomon集合和统计检验均不是原创性声明；算法贡献的候选机制见`R10_MATH.md`。

本次设计与交付不提交Slurm作业、不取消旧作业、不占用学校GPU。阶段1的CPU/合成验证与阶段2以后的真实EEG实验必须区分。后续正式提交须有显式`--submit`，并通过各阶段冻结/完成条件。

## 2. 共同科学协议

### 2.1 支持预算与固定未来query

- 每个episode使用真实前20个**采集位置**作为最大校准窗口，预算B为4、8、12、16、20。
- QC后有效支持数可以少于B；不能用后来的有效样本补足B，也不能改成“前B个有效样本”。
- 各预算使用同一个已冻结、位于B20校准边界后的未来query集合。改变预算不能同时改变query难度或分母。
- 不在query上调参、挑选epoch/模型、更新BN/协方差/白化统计、选择报告结果或筛选session。
- 目标query输入只在固定拟合状态下逐点计算预测；其标签仅进入独立评价阶段。加入其他query不能改变已给定query的预测。
- 支持集单类、过少有效点、EA几何不可用、数值求解失败等必须按固定fallback规则记录。不能因某方法困难而从配对人群中删除episode。

这里严格区分标签角色：每个目标episode的`fit/predict`均不接受其query标签；meta_validation的开发query标签可以用于预先规定的selection/check，因此记录`development_query_labels_used=true`；正式目标query不参与R10配置选择，记录`target_query_labels_used_for_selection=false`。另一方面，R1–R9旧holdout/Pan/Weibo结果已经影响本轮研究动机与算法设计，这一历史不能被后一个字段抹去，所以旧cohorts仍全部属于探索性证据。该字段不表示研究者从未看过任何历史目标结果。

### 2.2 数据身份与采集顺序

每个episode固定participant/session、源模型seed、最大校准采集UID、有序支持位置、各预算有效UID、固定未来query UID、有序物理run/时间来源、QC/预处理/数据划分SHA256。源EEGNet、EA实现、开发参考几何、残差核、代码版本及各方法配置同样绑定身份。

真实采集顺序与physical run身份是独立确认的必要条件。Pan/Weibo若仅有published-order代理，必须保留该标注，只能作为探索性证据；不能升级为“真实时间外推确认”。

对新确认cohort，必须核对参与者、会话及数据版本没有进入源预训练、meta开发、R1–R9查询分析或R10方法设计。数据集改名、旧数据重新划分或旧会话新seed不是新独立数据。

### 2.3 冻结模型与参考几何

EEGNet参数、原分类头、BN与dropout均保持冻结。Support-EA只用本预算校准输入。残差特征参考几何只来自meta_train/source training角色，不能来自目标query。

优先复用R9已存在的EA meta_train reference及其对应源/data/role证据，但必须核对源checkpoint、特征预处理、参考role、UID和SHA完全相同；存在文件不等于可以复用。若不匹配，则按事先固定的8个meta_train prefix-session策略重建reference，不能看指标后选session。

## 3. 任务规模与实际计算单位

| 阶段 | 任务声明上限 | 构成 | 本次状态 |
|---|---:|---|---|
| 1 数学与软件验证 | CPU测试，不是EEG任务 | 合成输入、数学性质、数据/身份/无泄漏、缓存/恢复/归档 | 可在当前环境运行并记录实际通过项 |
| 2 小规模机制筛选 | 15 | 1个reference + 9个profile候选 + 3个matched MAP + Frozen + EA | 真实EEG实验尚未运行 |
| 3 冻结 | 0个拟合任务 | 固定阶段2所选配置、比较家族、源/data/code身份和统计方案 | 依赖阶段2PASS |
| 4A 正式探索性配对 | 165 | 3个reference + 18方法×3seeds×3已用cohorts | 未提交；须有冻结文件 |
| 4B 独立确认 | 57 | 3个reference + 18方法×3seeds×1个新cohort | 缺少新cohort时必须BLOCKED |
| 全部阶段理论声明上限 | 237 | 15+165+57；并非一次性展开 | 各阶段门控后才能继续 |

“165任务”不是165次廉价拟合。每个evaluation任务包含多个participant/session，且五个预算分别校准。若`E_(seed,cohort)`为冻结可评价episode数，则18方法的正式阶段共有

`18×5×Σ_(seed,cohort) E_(seed,cohort)`

个method–episode–budget评价单元；需要拟合的方法各自在这些单元上拟合。Frozen没有参数拟合，EA仍需支持协方差估计；参考提取、源特征推理和query profile求解另计。报告必须列出实际episode、支持拟合、query预测及缓存复用次数，不能只用任务总数掩盖计算成本。

## 4. 阶段1：数学与软件验证

### 必须完成

1. λ强凸目标的梯度/Hessian和下降线搜索；有限可行拟合满足`Fhat≤F0`，否则明确anchor fallback。
2. Cholesky全部SPD方向、声明的UID nugget、query创新的PSD一致性；不把正特征值截断冒充精确零空间。
3. 非二次logistic profile的对偶外界包含独立二维可行网格，保留gradient residual修正，不能用近似可行extremizer当外界。
4. 同一支持拟合下，query-null completion证书可否决修正；同时构造R9式方差information gate仍非零的对照，说明两者并非同一机制。
5. 可识别共享方向应产生非零更新；完全无交叉信息、c=1、零amplitude、单类/空支持等退化情形应复原anchor。
6. 对固定模型，验证精确相容区间的嵌套与精确minimax动作向anchor收缩的数学性质，并在合成例上检查有限实现；不同c的自适应有限求界并不保证实际近似动作普遍单调，不能由几个数值例宣称这一保证。query批次、顺序和额外query不能反向影响拟合或其他预测。
7. 旧head满行秩时空间Schur商信息为零的合成构造，防止虚构独立参数信息。
8. 状态保存/加载绑定支持、配置、参考几何和派生状态；代码/数据/预测身份变化不能静默复用已完成结果。
9. 断点恢复、锁、有限失败重试、结果/运行状态分离、上传失败不重训、双通道独立回执，以及凭据不进入输出。
10. 学校环境兼容性与有限GPU预检命令应可用；本次没有GPU时必须明确标为尚未执行，不能由CPU测试替代A100实测声明。

### 完成条件

规定CPU/合成回归通过，未发现目标query标签进入episode拟合或R10配置选择的路径，允许并单独记录开发meta_validation query标签按固定规则进入selection/check；数值失败有可审计fallback，部署dry-run不会分配GPU或提交作业。实际测试报告记录命令、时间、环境和代码哈希。

### 停止条件

任何数学证书被独立可行反例推翻、发现隐藏的query统计更新、缓存身份错配、方法标签/真实调度不一致或恢复重训完成任务：停止后续阶段，修复并重新固定阶段1版本。通过阶段1不代表真实EEG有效。

## 5. 阶段2：15任务的小规模机制实验

### 5.1 固定开发人群

仅使用meta_validation内事先允许的8名参与者。按participant ID的固定hash排序选择，恰好8名，每人选择一个已有协议判定合格的B20未来query session，session同样按固定ID/hash规则选，不能看query表现。

8名参与者在任何候选指标计算前，按固定hash分成4名selection和4名check。若不足8名合格参与者，阶段2为`BLOCKED_INSUFFICIENT_DEVELOPMENT_COHORT`，不能临时放宽成更少样本或替换query困难session。

- selection部分：用于固定候选选择。
- check部分：只执行一次PASS/STOP检验，不重新排名，不从失败配置里挑一个在check上更好的替代品。
- 两部分都属于算法开发，均不是独立确认数据。

### 5.2 候选与预算

固定seed20261018；预算4、8、20。主选型指标为B20的participant-equal BA；B4/B8用于机制、小预算稳定性和成本诊断，不能据其结果事后切换主选型指标。

9个主机制候选是完整笛卡尔积：

- kernel amplitude：0.3、1、3；
- evidence fraction c：0.01、0.1、0.5；
- 主kernel为冻结参考白化特征上的linear+intercept，λ=1，标签平衡权重`n/(2*n_class)`；
- 每个amplitude对应一个matched MAP消融，共3个；
- Frozen与Support-EA各1任务；
- reference 1任务（若合法命中已有reference，则完成回执明确标记复用）。

不得在阶段2边看已用holdout/Pan/Weibo的R10表现，边改c网格、amplitude或求解规则。

### 5.3 选型规则

对selection的4名参与者，先计算各候选的B20平均BA。令最高平均BA为`best`，其参与者平均BA的标准误为`SE_best=SD/sqrt(4)`。one-SE候选集为平均BA至少`best−SE_best`的**9个robust候选**。集合内按以下固定次序选择：c更大优先、amplitude更小优先、最后候选key字典序。Frozen、EA和MAP不是替代候选，而是独立acceptance/消融基准。

这是已有保守选择规则，不能当作R10原创贡献。样本只有4名selection参与者，SE与点估计高度不稳定；此规则是小规模淘汰程序，不是显著性结论。

### 5.4 PASS/STOP条件（选出后只检查，不再排名）

所选候选只有同时满足以下条件才PASS：

1. selection部分B20相对EA的participant-equal BA差至少`+0.25个百分点`；
2. check部分B20平均BA不低于EA；
3. selection和check上的balanced NLL均不得比EA恶化超过0.01；
4. 不出现已验证的对偶/模型集合证书违反、数据身份错配、无泄漏失败或方法语义不一致；
5. 数值错误导致的全局anchor fallback比例不超过1%；协议规定的空/单类支持fallback与有意义的query-null保守选择分别统计，不能混成数值错误；
6. 所选方法有非平凡预测修正。预先固定以B20下`|p1_method−p1_EA|≥1e−4`定义非平凡，按participant等权统计；若全部候选完全复制EA，或所选方法在固定8名开发参与者的B20上合并计算的participant-equal修正比例低于1%，STOP；
7. 在学校实际CPU配置上，缓存冻结特征后新增profile预测耗时不超过5秒/1000个query；同时报告非零候选query的单位成本、支持拟合耗时和完整pipeline耗时，不能靠大量零更新快路径隐瞒昂贵求解。

主实现的query root求解预算在任何真实EEG候选结果产生前固定为`max_root=12`。该工程预算依据阶段1合成CPU压力与独立可行网格/证书验证调整，而非按目标BA/NLL选择。任意有效tilt的残差修正dual仍给出外界；较少迭代可能产生更保守的更新，不声称已得到精确端点。合成数据的耗时外推不算通过学校真实CPU的5秒/1000query门槛。

本阶段不使用内部holdout/Pan/Weibo进行重新选型。任一条件失败应输出`STOP_NO_EVIDENCE_FOR_ADVANCEMENT`及原因，保留完整候选结果。可以发表或记录负结果，但不能静默扩大网格后继续称同一预注册R10。

阶段2通过也不表明算法真实超过EA：上述小样本门槛只决定是否值得投入完整配对实验。

## 6. 阶段3：冻结主算法、比较及身份

PASS后生成不可变冻结文件，至少包含：

- 所选kernel、amplitude、c、λ、class weights、nugget、所有求解/早停/roundoff/fallback规则；
- selection/check的participant/session UID和选择算法，所有15任务原始结果及阶段门槛判定；
- 三个pipeline seeds、源checkpoint及EA、参考几何、数据/预处理/支持/query UID身份；
- 本文主要/次要比较家族、统计单位、CI/p值方法、随机seed及停止条件；
- 完整代码版本/文件SHA和环境约束；
- `development_query_labels_used=true`，`target_query_labels_used_for_selection=false`；
- 旧cohorts全为探索、新确认cohort的独立性与数据来源审查状态。

没有冻结文件、配置/身份不匹配或阶段2不是PASS，正式阶段不得提交。已用目标cohort的结果不能回流修改freeze；如确需新算法版本，必须开启新的研究版本并将旧结果保持原样。

## 7. 阶段4A：165任务正式探索性配对实验

### 7.1 固定范围

- seeds：20261018、20261019、20261020。
- cohorts：现有内部holdout、Pan、Weibo；三者均已参与设计/既往比较，因此全部标为exploratory。
- 所有方法使用共同冻结episode与query UID；每个任务内部包含B4/8/12/16/20五个预算。
- 3个meta_train reference任务加18方法×3seeds×3cohorts，共165任务。三seed不是独立参与者重复数。

### 7.2 固定18个方法

| key | 方法语义与必要验证 |
|---|---|
| frozen | 真正冻结EEGNet，无目标统计更新 |
| support_ea | 真实Support-EA；必须通过协方差变换/dispatch测试，不能fallthrough成Frozen |
| head_bias | 只更新bias的低容量head |
| head_rank4 | 仅用当前有效支持的无标签冻结特征做PCA取至多4维，另带bias；不是meta_train固定子空间，禁止读取query或按标签选方向 |
| head_full | 完整head；编码器与BN冻结 |
| sgd10 | 已有10步空间SGD校准，按原方法字典/几何/先验配置执行 |
| sgd100 | 已有100步空间SGD校准；主比较强基线，不得偷换为弱化实现 |
| r3_protected | 已有R3保护校准；核对真实变换/完整head及KL规则 |
| r4_xfit_select_stack | 预先固定的R4交叉拟合选择+stacking；正确记录组件，不按R10目标结果重选 |
| r8_fiber_bounded | 按完整R8证据预先指定的代表性方法与原配置；不能按目标cohort另挑最佳版本 |
| r9_raw_anchored | 已有raw-feature RBF anchored GP原实现，使用其原meta_validation选择配置（最新已取得完整三种子证据） |
| r9_ea_anchored | 已有EA-feature anchored GP，主要比较对象 |
| profile | R10主方法，阶段3冻结配置，EA anchor |
| profile_map | 与profile相同几何、权重、amplitude及λ的直接MAP；`not_profile_certified` |
| profile_no_completion | 同一配置但强行忽略s_q；检验未识别query完成是否必要，属于非安全机制消融 |
| profile_quadratic | 同一拟合状态与相容阈值，但用局部Hessian二次椭球；是近似机制消融，不继承非线性目标证书 |
| profile_variance_gate | 同一拟合状态/权重/几何，以20点Gauss–Hermite与0.75信息比例混合代替稳健动作；是R9式matched控制，不替换原生R9基线 |
| profile_frozen_anchor | 固定Frozen anchor替代EA anchor；其余配置固定，不按目标数据重调 |

已有完成预测/特征可以复用，但必须核对实际method语义、代码/源/数据身份、各预算支持及固定query UID。旧汇总标量若缺少Brier、paired flips或不同aggregation，不能冒充完整R10结果；可从合法原预测重新计算评价而不重训。缺少必要基线artifact应BLOCKED/显式运行真实原方法，不能填随机预测、复制Frozen、删除方法或写“近似等价”。

R4曾出现EA标签与真实调度不一致；因此旧`support_EA`名称本身不能作为已正确执行EA的证据。R4其他候选也要按其真实recipe审查，而非一概视为正确或一概作废。

## 8. 阶段4B：独立确认，57任务

需要一个此前未用于算法设计和源/开发/目标比较的新cohort，包含核验后的真实采集顺序、物理run身份、参与者来源及标签隔离规则。新cohort完全按阶段3 freeze运行，不再修改方法或超参数。

计算范围：3个已冻结meta_train reference任务（允许严格复用）+18方法×3seeds×1个新cohort=57任务，各evaluation任务包含五预算。

没有满足条件的新数据与身份证明时，状态必须是`BLOCKED_NO_INDEPENDENT_CONFIRMATION_DATA`；不得将Pan/Weibo换名、旧holdout新seed、旧数据新划分或只隐藏结果摘要当作独立确认。

如后续扩大确认cohort数，应在读取新cohort标签/结果前扩展统计家族并固定，不在显著性结果出来后选择报告哪个cohort。

## 9. 评价指标与配对单位

### 9.1 层级聚合

每个session先在全部冻结未来query上计算class-balanced BA、NLL与Brier；同一participant内部等权平均合格session，再对三个pipeline seed等权平均，最后对participant等权平均。

participant是独立统计单位。query trial、session、预算点及seed均不能增加名义样本量。每个contrast必须使用相同冻结participant集合与同一query UID配对；缺失方法/seed/session使主要比较未完成，不能按方便的交集缩小分母后标为完成。

BA效应量以百分点表示：`Δ_BA_pp=100*mean_i(BA_profile,i−BA_baseline,i)`。NLL/Brier以原单位报告差，负值为改善。

R10对全部方法统一采用`1e−12`概率floor计算balanced NLL，并保留该评价版本身份。历史轮次若使用未截断log-probability/BCE-with-logits，其NLL标量不能与R10截断后的NLL直接混合；需要从身份匹配的原预测/log-probability按共同规则重算，保留原结果，不覆盖历史文件。BA配对仍须同时满足source、支持/query UID与聚合身份。

### 9.2 主要与辅助终点

主要终点：B20参与者等权BA。

辅助终点：balanced NLL、balanced Brier、B4/8/12/16/20预算曲线、预先固定的预算曲线面积、相对EA的participant负迁移比例、纠正错误/引入错误、fallback/证书覆盖、时间、状态存储及参数复杂度。

负迁移定义：participant B20的`ΔBA<0`；零差不计负迁移。另报告`ΔBA≤−1个百分点`等严重负迁移描述，但不能替换预先固定主定义。

纠正/引入错误必须按同一query UID与相同tie-breaking计算：EA错误而新方法正确为纠正；EA正确而新方法错误为引入；另报告class-balanced及participant-equal聚合，避免只给长session的总trial数。

复杂度区分：冻结encoder参数、支持拟合系数r≤n、完整残差函数类维数、query profile临时变量、实际自由度/核谱及缓存字节。不能用r≤20宣称原65维head的所有不确定性已经消失。

## 10. 统计方案与多重比较

### 10.1 主要确认家族

每个预先规定的新确认cohort，B20 BA固定四个contrast：

1. profile − Support-EA；
2. profile − SGD100；
3. profile − R9 EA-anchored GP；
4. profile − matched profile MAP。

初始独立确认仅一个cohort，因此Holm家族大小4；若预先增加cohort，家族扩大为4×cohort数。双侧α=0.05；方向和完整效应量都报告，不能把负向显著解读为成功。

探索性旧三cohorts的相同四contrast构成单独12项描述性家族，报告Holm结果并明确探索性质。它们不能因Holm校正而变回独立确认，也不用于重新挑主算法。

### 10.2 估计与检验

- 每个contrast保存所有participant配对差，报告N、均值、标准差、95%CI、原始p及Holm校正p。
- CI：participant配对bootstrap 10,000次，固定seed20261026及预先声明的contrast/cohort子stream。重采样participant时携带其全部session和seed，不能独立重采样trial制造精度。
- p：N≤20时精确枚举sign-flip；更大N时100,000次固定seed Monte Carlo sign-flip，用`(extreme+1)/(M+1)`。零差/all-zero情形p=1，N不足无法合理估计时明确不可推断。
- sign-flip的“精确”指所用符号零分布的枚举，不代表无条件因果随机化；它依赖participant独立和相应零假设下的符号交换/对称性。小N bootstrap与检验的局限必须保留。
- 只有完整方法、seed、人群及session覆盖通过后才计算主要家族的最终推断；中途监控不反复生成“主结论p值”。

预先固定机制消融次要家族：在B20下，主方法分别减`profile_no_completion`、`profile_quadratic`、`profile_variance_gate`、`profile_frozen_anchor`，对BA与balanced NLL各作一个contrast，共`8×cohort数`项统一Holm校正。matched MAP的BA对比已属于四项主要家族，不借此重复另建家族；其NLL可完整描述，但不以未预先规定的显著性升级主结论。Brier、预算曲线/AUBC各自按冻结的独立次要家族报告，不把次要指标显著升级成“主要终点突破”。不新增数据驱动亚组，必要探索亚组须单独标注。

## 11. 成功标准、失败标准与论文解释

最低主要终点成功标准固定为：新独立cohort的`profile−Support-EA`在B20上BA效应为正、95%CI下界大于0且四项主要家族内Holm校正p<0.05。其他三个主要contrast必须同时完整报告；只有各自正向效应和校正检验支持时，才能声称优于SGD100、R9 EA-anchored或自身MAP。不能把EA一项成功外推成全部基线优越。

具备进一步算法论文价值的必要证据还至少包括：

- 新独立cohort上的主方法B20 BA优于真实EA，效应正、95%CI和Holm校正结果支持预先规定主结论；
- 与强SGD/R9方法及自身MAP的比较说明不是只超过一个失败空间方法；
- `profile_no_completion`等消融能定位新增机制，而非只有核/超参数/预处理改变；
- NLL、Brier、负迁移与小预算结果与主机制一致，或完整解释不一致；
- 新增计算成本、复现与数值失败率值得实际应用投入。

这些是待检验标准，不是当前已取得结果。若仅在旧探索cohort得到小幅提升、只改善训练loss、几乎复制EA、主要指标不显著或计算成本超门槛，应保留负结果并降低论文贡献判断。阶段2停止或独立数据缺失时，不能宣称已完成SCI算法验证。

样本量与power目前未知，不能由“57任务”“三seed”推导具有足够统计功效。独立确认开始前，可只用meta开发人群的participant差异方差做透明power/precision敏感性分析；必须在查看确认数据标签/结果前固定招募/样本数，不从最终结果倒推所需N。

## 12. 工程与提交边界

学校正式资源配置固定为**4个并行Slurm作业，每个作业1张GPU、16核CPU、68G内存，队列`hpc_gpu`**；总计4张GPU、64核CPU、272G申请内存。四个任务各申请一个单卡allocation，不要求全部放在同一节点；实际启动仍取决于学校调度配额和资源可用性。此处是后续提交配置，本次未自动提交或取消作业。

每阶段plan及freeze均记录精确任务key、依赖、method语义、数据/source/code哈希、资源配置和阶段门控。科学结果与runtime状态、重试次数及上传状态分开保存；运行失败不改科学配方，完成任务不重训。

训练不依赖外网；凭据从学校既有安全配置读取。双通道独立：A优先上传完整论文CSV/participant/session/统计/配置证据，B上传完整预测、检查点、验证轨迹和归档及SHA/回执。任一网络故障不终止训练，不因上传重试重新拟合。

安装、CPU验证、有限GPU预检和正式提交必须有不同命令/状态。正式提交要求显式`--submit`；阶段3未冻结、独立数据不合格或baseline身份缺失时不得提交相应正式阶段。本次交付只设计并验证软件，不自动执行GPU实验。
