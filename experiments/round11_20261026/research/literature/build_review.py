import json, pathlib, datetime
B=pathlib.Path('/workspace/paper3-round11/research/literature');R=B.parent
fetch=json.loads((B/'primary_source_fetch_manifest.json').read_text());F={x['key']:x for x in fetch}
rows=[]
def add(key,year,level,known,collision,boundary,sections='',pdf=None):
 f=F[key];m=f['meta'];title=m.get('citation_title',[f.get('title')])[0];authors=m.get('citation_author',[]);arxiv=m.get('citation_arxiv_id',[None])[0]
 ev=[f]+([F[pdf]] if pdf else [])
 rows.append({'id':key,'title':title,'authors':authors,'year':year,'doi':('10.48550/arXiv.'+arxiv) if arxiv else None,'doi_note':'arXiv repository DOI, not an asserted journal DOI' if arxiv else 'No DOI asserted; official proceedings URL used.','primary_url':f['requested_url'],'reading_level':level,'relevant_sections':sections,'established_mechanism_zh':known,'collision_with_R11_zh':collision,'limits_and_R11_boundary_zh':boundary,'fetch_evidence':ev})
add('maml',2017,'官方PMLR题录和完整摘要','在任务分布上学习可经少步梯度更新获得良好泛化的初始化；内外循环以目标任务适配后的表现训练。','episodic训练、少标签校准和meta_train/meta_validation分工均不是R11原创。','它并不提供20个相关EEG采集位置下的未来会话风险保证；不能把R11只冻结编码器与小矩阵更新当成新元学习范式。')
add('hierarchical_bayes',2018,'作者arXiv原文PDF已读','将MAML解释为层次模型的经验Bayes近似；§2.2式(2)积分任务参数、估计共享先验，§3把内循环更新与隐含先验连接，并讨论曲率/Laplace近似。','从meta_train学习先验协方差、目标小样本形成后验或以EB解释适配已经被覆盖。','改变为固定源anchor或冻结EEG特征是应用约束；必须标清新学习目标与可识别对象，不能把EB解释重新命名为理论贡献。','§2.2–3，式(2)–(3)',pdf='hierarchical_bayes_pdf')
add('deep_kernel',2016,'官方PMLR题录和完整摘要','结合深度特征映射与核方法，以边际似然学习核及表示。','φ(x)^T M φ(x′)、学习SPD几何或将冻结表示放入核均属于已有核学习语言。','R11若只学习M后替换R9核，最合理定位是learned-kernel基线；必须区分标签score协方差和边缘feature协方差。')
add('dkt',2020,'作者arXiv原文PDF已读','共享跨任务深核超参数，积分任务内GP并优化各任务边际似然；Algorithm1训练用meta_train任务S∪Q标签，测试仅对S条件化。分类§3.2采用±1 label regression，再对预测均值作sigmoid，并非精确Bernoulli GP分类。','跨任务学习可迁移核、Bayesian inner loop、small-support矩阵推断、query侧不重新拟合都已有直接先例。','R11若以Bernoulli score/任务前缀噪声作不同观测模型，须明确这一差异；不得把Gaussian label regression与R9的logistic GP声称逐项同一。','Algorithm1；§3，式(3)–(8)',pdf='dkt_pdf')
add('adkl',2019,'作者arXiv原文PDF已读','用任务编码器产生任务适配的核族，并与可微kernel ridge/GP类推断结合；不是所有任务共用一个固定核。','若R11根据20标签support生成协方差、核参数或任务条件几何，与ADKL直接接近。','配对prefix的统计识别设计可能是额外结构，但support-conditioned kernel本身不足以支撑原创。此处以2019预印本/2020版本引用，不虚构期刊发表信息。','方法和任务核编码部分',pdf='adkl_pdf')
add('alpaca_correct',2018,'作者arXiv原文PDF已读','ALPaCA联合学习固定online特征、matrix-normal权重先验均值与precision；式(5)–(8)给BLR后验；§4.1–4.2以held-out conditional predictive NLL训练，并明确固定feature时就是empirical Bayes。','先验协方差＋冻结编码器＋小样本解析后验＋跨任务预测目标，是与R11最直接的先例之一。','ALPaCA原文主要为Gaussian回归且noise covariance作为给定量；R11若学习score观测noise需给独立识别证据，不能仅说“我们同时学noise”就证明原创/可靠。','§3–4，式(5)–(9)，Algorithm1',pdf='alpaca_pdf')
add('r2d2',2018,'作者arXiv题录和完整摘要','用ridge regression及logistic regression求解器作为few-shot适配模块，误差反传过求解器，并以Woodbury利用少样本降低矩阵成本。','冻结特征、小头ridge/logistic求解以及n≤20的dual解法不能算新算法。','R11应保留等特征、等20trial预算的ridge/logistic baseline；只比昂贵全参数fine-tune不能隔离新operator的贡献。')
add('metaoptnet',2019,'官方CVF题录和完整摘要','将可微凸优化分类器纳入元学习，以任务内凸分类问题和外层表示学习结合。','“内层凸、外层学几何/特征、少样本快速适配”已有成熟先例。','R11的PSD约束或凸logistic inner loop仅是工具；需报告operator的数据对象、学习准则及跨session验证的实质变化。')
add('metacurvature_correct',2019,'作者arXiv原文PDF已读','学习跨任务梯度变换以提高适配泛化；§3式(3)用张量mode-product建模参数依赖，学习的curvature并非仅是当前support Hessian或Fisher。','“学一个适配几何让20标签gradient更可靠”是直接碰撞，不应把learned preconditioner称新原则。','R11需要证明其operator是在配对观测模型下学习稳定label-score、而不是常规meta-curvature的另一种参数化。','§2.3–3，式(2)–(3)',pdf='metacurvature_pdf')
add('kml_revisit',2021,'作者arXiv原文PDF已读','KML在该文指Kernel ModuLation：task embedding生成卷积权重残差调制；§5.2式(6)为W⊙(J+M)，式(8)用向量外积降参数；同时以任务间loss ratio分析正负transference。','support条件化调制、低秩hypernetwork、以跨任务迁移解释适配都已有先例。','这里kernel是卷积滤波器，不是GP covariance kernel；不能误写为同一种核方法，也不能声称R11首次分析negative transfer。','§4–5.2，式(6)–(9)',pdf='kml_pdf')
add('reliable_prior',2021,'作者arXiv原文PDF已读','F-PACOH将先验、hyper-prior与hyper-posterior当随机过程；用有限measurement-set上的functional KL近似正则，改善meta-data稀少处的过度自信。','“学习可靠先验”“函数空间保守不确定性”“防止meta-prior过度自信”都已有实质先例。','其PAC/过程KL条件不自动传到R11；R11的单个EB协方差估计忽略hyperparameter uncertainty，不能由后验variance直接宣称目标会话可靠。','§2–4，式(1)–(3)',pdf='reliable_prior_pdf')
add('amortising_2026',2026,'2026作者arXiv原文PDF已读','通过逐dataset amortised variational inference学习BNN权重先验，将模型构造成以BNN权重为latent的neural process；讨论极少任务下学习先验以及within-task minibatching。PDF自标ICLR2026，本记录主引arXiv。','截至本轮日期，跨dataset学习Bayesian prior与amortised task inference仍有最新直接先例；泛称“元学习先验”已远不足新颖性。','本文不是EEG配对score观测识别模型；R11可比较其思想，但不能凭简单EB更便宜就获得新原理。','§2–3，layerwise posterior与BNNP构造',pdf='amortising_2026_pdf')
add('label_noise_identifiability',2013,'官方PMLR题录和完整摘要','给出非对称label contamination下可识别、可一致估计的假设及maximal-denoising策略；没有把标签噪声率可从任意小样本无条件推断当成事实。','用20个标签推断每trial“标签可靠性”与真实潜在分类误差，不是自动可识别；需额外结构。','EEG cue标签通常是实验任务指令；信号低SNR、被试不执行任务、source分类错与标签误标必须区分。R11的score稳定性不能被命名为已识别的label correctness。')
# Reuse independently fetched primary records from the statistical sub-audit.
S=json.loads((R/'literature_stats_priorart.json').read_text());rows.extend(S['papers'])
n2n_ev=[x for x in S['all_fetch_attempts'] if 'lehtinen18a' in x.get('url','')]
rows.append({'id':'noise2noise','title':'Noise2Noise: Learning Image Restoration without Clean Data','authors':['Jaakko Lehtinen','Jacob Munkberg','Jon Hasselgren','Samuli Laine','Tero Karras','Miika Aittala','Timo Aila'],'year':2018,'doi':None,'primary_url':'https://proceedings.mlr.press/v80/lehtinen18a.html','reading_level':'官方PMLR题录；作者原文PDF由统计支线读取','relevant_sections':'条件均值与配对噪声监督部分','established_mechanism_zh':'平方损失下，保持给定输入的目标条件均值可用有噪声目标学习相同回归函数；以同潜变量的配对噪声观测替代clean target具有直接先例。','collision_with_R11_zh':'配对meta_train block提取稳定潜信号不是首次；R11的cross-block矩协方差虽非该文估计器，也不能把“以noise识别signal”作为全新原则。','limits_and_R11_boundary_zh':'两个EEG前缀块不天然是同潜信号的独立重复；类组成、serial dependence和会话漂移会改变条件均值。此文不提供R11协方差分解可识别性的定理。','fetch_evidence':n2n_ev})
assert len(rows)==20,len(rows)
# Explicitly retain failed/wrong-title probes as rejected evidence, not citations.
rejected={'alpaca':'404; guessed proceedings path invalid. Correct author source is arXiv1807.08912.','metacurvature':'arXiv1902.03399 title mismatch, unrelated computability paper. Correct source is1902.03356.','arxiv_kernel_search':'No matching paper returned; search result only.','kml_search':'Broad query result only; not a paper citation.'}
for x in fetch:
 x['citation_status']='rejected_or_discovery_only' if x['key'] in rejected else ('supplementary_not_one_of_20_reviews' if x['key'] in ['mtl_feature','kernel_modulation','pmaml'] else 'verified_primary_source')
 if x['key'] in rejected:x['rejection_reason']=rejected[x['key']]
(B/'primary_source_fetch_manifest.json').write_text(json.dumps(fetch,ensure_ascii=False,indent=2)+'\n')
obj={'schema_version':'paper3-r11-literature-audit-v1','cutoff_date':'2026-10-26','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_count':len(rows),'papers':rows,'scope_note_zh':'20篇有界原始文献审查；完整PDF、完整摘要和仅题录核查分级。不声称全球首次或穷尽截至日期所有论文。','rejected_source_probes':rejected,'statistical_sub_audit':'research/literature_stats_priorart.json'}
(B/'verified_paper_reviews.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
intro='''# R11 文献审查与原创性边界

核查截止 **2026-10-26**。本次审查20篇真实论文，并保留本轮实际抓取的URL、HTTP状态、时间、字节数与SHA256；正文缓存只放在`/tmp/r11-lit-primary`及统计支线缓存，不将版权原文重新发布到交付目录。下列“全文已读”“完整摘要”“仅官方题录”明确分级。R10已有原文核查只作为起点，本轮新增主题做了实际公开获取。

**结论：仅把R9固定核改成learned kernel，把R10相容集合换成EB收缩，或加入noise whitening、source anchor，都不足以构成新算法。** 与R11最接近的先例是ALPaCA、DKT/ADKL、Meta-Curvature、Bonilla多任务GP及经典混合效应模型。R11值得研究的具体差异只能落到：在与部署一致的采集前缀观测单位上，区分跨会话可复现的标签score方向、会话效应及观测noise，并由这一分解约束适配operator。该结构仍是待证伪的研究设计，不能宣布全球首次或已优于EA。

## 1. 最接近的碰撞

| R11口号/结构 | 已有直接先例 | 必须保留的边界 |
|---|---|---|
| 学可迁移核或PSD feature几何 | DKL、DKT、ADKL、Bonilla MTGP | 学核不是新贡献；R9只换核属learned-kernel对照 |
| 固定特征，学权重先验后用少样本后验 | ALPaCA、Grant层次Bayes、Laird–Ware | BLUP/EB后验和Gaussian conditioning均非新推导 |
| 学一个让梯度泛化更好的更新矩阵 | Meta-Curvature | learned preconditioner不能重新命名为新原理 |
| support条件化低秩调制 | KML、ADKL | KML的kernel指卷积filter，不能与GP covariance混写 |
| 白化/二阶域对齐/小样本协方差稳定化 | CORAL、LW、OAS、RDA | feature covariance不等于label-score signal/noise covariance |
| 用配对噪声观测学习稳定信号 | Noise2Noise及重复测量模型 | 不同EEG时间块不天然同latent mean或独立noise |
| 可靠Bayesian prior与少任务不确定性 | F-PACOH、2026 BNNP | EB点估计未自动计入超参数不确定性 |
| 20标签识别每trial的真实可靠性 | Scott等label-noise identifiability | 需要额外假设；低SNR不等于cue标签错误 |

## 2. 二十篇具体审查

以下SHA指实际获取的原始响应字节；完整多来源记录见`literature/verified_paper_reviews.json`和`literature/primary_source_fetch_manifest.json`。arXiv的`10.48550/...`是仓储DOI，不冒充出版DOI。
'''
parts=[intro]
for i,p in enumerate(rows,1):
 evid=[x for x in p['fetch_evidence'] if (x.get('status',x.get('http_status'))==200)]
 ev=next((x for x in evid if str(x.get('content_type','')).startswith('application/pdf')),evid[-1] if evid else None)
 sha=ev.get('sha256') if ev else 'unavailable'
 parts.append(f"\n### {i}. {p['title']}（{p['year']}）\n\n来源：{p['primary_url']}"+(f"；DOI `{p['doi']}`" if p.get('doi') else '；不虚构未提供的DOI')+f"。核查等级：{p['reading_level']}。SHA256 `{sha}`。\n\n已有机制：{p['established_mechanism_zh']}\n\n与R11关系：{p['collision_with_R11_zh']} 边界：{p['limits_and_R11_boundary_zh']}\n")
ending='''
## 3. 20trial可辨识性与配对观测的必要条件

令冻结源模型的trial score为`g_i = J_i^T(y_i-p0_i)`，部署前缀统计量为` s=(1/B)Σg_i`。若每个真实采集trial可产生多个窗口，B仍计**原始acquired trial/UID**，窗口不是新增独立标签。d维目标样本协方差的rank至多B−1；当d≥20时，仅20个相关trial不能无约束识别高维signal、noise和session效应。meta_train估计的协方差是转移先验，不能说它由目标20标签独立识别。

最简单重复测量模型是`s_a=m+e_a, s_b=m+e_b`。只有`E[e_a|m]=E[e_b|m]=0`且条件交叉noise为零时，`Cov(s_a,s_b)=Cov(m)`。只有两块noise covariance相同，`Cov(s_a-s_b)/2`才给该noise；不等时只得到两者和的一半。PSD投影只是有限样本正则，不会证明这些假设。

跨会话可进一步写`s_{p,s,b}=μ+u_p+v_{p,s}+e_{p,s,b}`。不同session的同participant交叉矩可识别可复现的`u_p`，同session不同非重叠块还混有`v_{p,s}`。这是已知crossed/random-effects矩分解的应用。它能成为R11的研究设计，前提是数据有足够真实重复session，并且显式建模/检验类组成、任务状态、时间趋势及跨块相关。不能把嵌套B=5/10/20前缀当条件独立复制；也不能为了让模型假设成立而偷偷将前20trial改成类平衡采样。

Gaussian近似下`μ+D(D+R/B)^{-1}(s−μ)`就是经典后验均值/BLUP。若noise存在serial dependence，`R/B`应换为实际块均值covariance，包括lag covariance项；盲目按1/B外推没有根据。将Bernoulli score近似为Gaussian是计算近似，不是20标签下的精确概率模型。

EEG cue标签是否正确执行、source预测是否错误、trial是否低SNR、以及潜在P(y|x)是否可辨识是不同问题。没有额外观测或假设，不能把score不稳定估计命名为“真标签错误概率”，也不能借此自动剔除support标签。

## 4. 候选的取舍与具体可研究机制

1. **简单EB learned covariance**：实用，必须实现为强基线；上述先例已实质覆盖，不作为论文唯一原创点。
2. **support-conditioned deep kernel/低秩调制**：与ADKL/KML/Meta-Curvature重叠，参数更多且目标20trial不足以证明几何可辨识；不优先作为R11主线。
3. **部署前缀观测驱动的可复现score operator**：本轮推荐的研究假设。先固定source anchor、score定义、合法物理trial预算及prefix/block观测协议；在meta_train的真实重复session中学习“跨session可复现部分→目标前缀观测”的transport/noise operators，以participant等权构造矩或likelihood。最终目标适配只接收本会话已采集前B≤20的features、source logits、labels和UID，不读取target query分布或标签。优先把operator定义在label score而非任意feature covariance上，使学习对象直接对应分类适配的可迁移信号。

该候选相对R9的区别是**学习哪个score方向能跨会话复现及其采集观测noise**；相对R10的区别是**改变可学习的更新几何与观测模型**，而非只在固定kernel上换更保守动作。其新意不是交叉协方差、EB或随机效应定理，而可能是针对冻结源EEG与真实前缀部署机制的联合统计对象、受约束训练目标、有限支持适配器及其系统证伪。

source anchor可以保留为安全工程基准；除非另有严格模型条件和有效外界，不能承诺未来BA/NLL不下降。新geometry并不由R10的anchor-relative模型集合保证自动保护；应分别报告几何收益和保护动作收益。

## 5. 必须能推翻主张的实验

- 等源特征、同一物理前20trial：source、Support-EA、ridge/logistic head、pooled/shrinkage-LDA、R9固定核、R9 learned-feature covariance kernel、isotropic EB、full score covariance EB。先验证新operator不是这些baseline的等价改写。
- paired-score主线对照：正确配对、participant/session打乱配对、同session与跨session配对、只feature covariance、去noise校正、只diagonal noise、忽略lag项。若打乱配对不损失，稳定跨会话信息不是收益机制。
- 逐层验收：meta_train配对矩的split stability和有效rank；meta_validation中score transport预测误差及真实适配BA/NLL；成本/条件数；冻结后只一次使用新的confirmatory participants/cohort。历史R2–R10已影响设计的cohort都保持探索性。
- 记录零更新率、每participant负迁移、20trial下适配幅度和耗时；若只因总回到anchor而表现稳定，不称新的适配能力。若增益只出现在相同session，而跨session消失，应缩小claim并拒绝“transferable geometry”表述。

有限文献检索无法证明某个组合全球首创。投稿主张应以精确operator/likelihood、与上述直接先例的公式级差异及冻结后的实证证伪为依据。本文件不含新EEG性能结论，也未读取原始EEG、使用GPU或修改学校作业。
'''
parts.append(ending)
(R/'literature_review.md').write_text(''.join(parts))
print(json.dumps({'review_count':len(rows),'files':[str(B/'verified_paper_reviews.json'),str(R/'literature_review.md')],'titles':[x['title'] for x in rows]},ensure_ascii=False))
