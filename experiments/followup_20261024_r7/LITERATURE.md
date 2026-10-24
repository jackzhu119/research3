# 第七轮一手文献与研究边界

检索核对日期：2026-10-24。优先核对正式论文集、期刊或原作者文章；以下文献支持数学工具与问题定位，不代表本轮EEG已取得效果。有限检索不足以证明任何方向“无人提出”。未将搜索摘要中的改进幅度外推到当前数据。

## 1. 时空、可分离结构与正交变换

1. **Higashi、Tanaka，2013，Common Spatio-Time-Frequency Patterns for Motor Imagery-Based Brain Machine Interfaces。** Computational Intelligence and Neuroscience，Article537218。[原文](https://doi.org/10.1155/2013/537218)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/24302929/)；[作者机构目录](https://hgshrs.github.io/publications.html)。时空频联合建模在MI中已有先例，不能仅增加时间滤波便宣称新范式。

2. **Separable Common Spatio-Spectral Patterns for Motor Imagery BCI Systems，IEEE TBME63(1):15–29，2016，2015年在线发表。** [DOI](https://doi.org/10.1109/TBME.2015.2487738)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/26452197/)。可分离时空谱建模和小样本结构约束有直接先例。本轮关注冻结骨干的少参数适配及有限窗守恒，不将这类已有结论抹去。

3. **Singla、Feizi，Skew Orthogonal Convolutions，ICML/PMLR139:9756–9766，2021。** [正式论文](https://proceedings.mlr.press/v139/singla21a.html)。反对称生成元指数的正交性、卷积构造和近似界均已有工作。本轮使用小块谱指数实现有限窗时空适配，不称正交指数是新定理，也不套用其CIFAR鲁棒性为EEG结论。

4. **Prichard、Theiler，Generating surrogate data for time series with several simultaneously measured variables，Physical Review Letters73:951–954，1994。** [期刊](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.951)；[作者稿](https://arxiv.org/abs/comp-gas/9405002)。多变量共同相位surrogate已有长期基础。第七轮的共同时间正交null只主张零滞后空间cov不变，不把零边界算子扩称任意全频交叉谱不变。

5. **Schreiber、Schmitz，Surrogate time series，PhysicaD142:346–382，2000。** [DOI](https://doi.org/10.1016/S0167-2789(00)00043-9)。原数据、约束和替代数据检验需要对应明确零假设。人工压力只检验所构造机制，不说明真实跨会话漂移具有同一生成过程。

6. **Sztyler、Królak、Strumiłło，Influence of EEG Signal Augmentation Methods on Classification Accuracy of Motor Imagery Events，Sensors26(4):1258，2026。** [期刊原文](https://www.mdpi.com/1424-8220/26/4/1258)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/41755195/)。该文比较包含Fourier-surrogate在内的EEG增强，相位增强在MI中已有直接应用。本文只据其存在排除“MI从未用相位”的空白主张，不把其方法等同本轮的受限前缀适配。

本轮可以检验的具体增量是：核心同r、同支持前缀下，有限窗实反对称时空生成元能否补充静态通道图，及收益能否排除边界、频谱泄漏和整体时间变化。rotation-only和temporal-only另作成分消融，排除仅去掉gain即可解释的收益；投影后幅度/有效rank可能改变，不把它们称严格等幅方向比较。非因果整trial操作、完整ambient与时间零均值子空间的生成元度量、白化/原CAR能量差别必须保留。

## 2. nuisance参数、头容量与可辨识性

7. **Cox、Reid，Parameter Orthogonality and Approximate Conditional Inference，JRSSB49(1)，1987。** [出版社](https://doi.org/10.1111/j.2517-6161.1987.tb01422.x)；[Oxford记录](https://academic.oup.com/jrsssb/article-abstract/49/1/1/7027797)。nuisance参数与有效信息是经典主题。第七轮的投影/Schur秩结论是任务化线性代数推论，不是新建立的一般统计定理。

8. **Raue等，Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood，Bioinformatics25(15):1923–1929，2009。** [原文](https://doi.org/10.1093/bioinformatics/btp358)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/19505944/)。结构性与有限数据下实际识别应区分。当前支持代理的full-head吸收空间作用，不自动证明神经生理变量可识别或不可识别；它针对所定义的预测参数。

9. **Raghu等，Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML，ICLR2020。** [正式会议页](https://iclr.cc/virtual_2020/poster_rkgMkCEtPB.html)。ANIL只保留任务head内循环，说明head/body分工已有系统研究。“只调头”本身不是本轮创新。

10. **Oh、Yoo、Kim、Yun，BOIL: Towards Representation Change for Few-shot Learning，ICLR2021。** [正式论文PDF](https://openreview.net/pdf/1a8c0d92bdde5a60d7b354b43c36e22abacfc99a.pdf)；[作者实现](https://github.com/jhoon-oh/BOIL)。冻结head、更新body同样有既有方法。第七轮需验证预算相关的头容量、必要秩条件和先验信息来源，而非简单将冻头改名。

可检验贡献是付费B20中“头空间何时吞没全部支持空间变化”的相图、与ridge正定性的区别，以及保留原head范数先验后的容量分配。必要条件不是充分保证，缺乏秩不使任务中断。

## 3. 相对风险、最坏风险与已有近邻

11. **Agarwal、Zhang，Minimax Regret Optimization for Robust Machine Learning under Distribution Shift，COLT/PMLR178:2704–2729，2022。** [正式论文](https://proceedings.mlr.press/v178/agarwal22b.html)。最坏绝对风险与低遗憾并不等价，已有明确理论。因此本轮不称相对源模型风险是全新概念；只研究真实B20适配中的变化稳定性和对应机制控制。

12. **Xie等，DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining，NeurIPS36，2023。** [正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html)。参考模型的excess loss用于重加权已有跨领域应用。本轮迁入同一会话支持点上的风险差，不将语言模型的效果作为当前MI性能证据。

13. **Ağırman、Saday、Tekin，Iterative Robust Satisficing: Minimizing Performance Degradation Under Distribution Shift，ICML/PMLR306:939–960，2026。** [正式论文](https://proceedings.mlr.press/v306/agirman26a.html)。固定名义性能并降低脆弱性已有近期方法。第七轮有限支持KL重权界的范围须与该文完整理论区分，不能据同样“稳定”措辞套用未来风险保证。

14. **Kouw、Loog，Robust domain-adaptive discriminant analysis，Pattern Recognition Letters148:107–113，2021。** [期刊DOI](https://doi.org/10.1016/j.patrec.2021.05.005)；[机构收录正式版本](https://repository.tudelft.nl/file/File_bec63d82-2984-44d1-9fe7-3927ec26c1d8)。文中target-contrastive pessimistic风险及源相对保守适应已有邻近思想。本轮目标、输入监督、有限支持重权条件和已有方法不同，不能只因使用baseline就主张首次。

第七轮相对指数风险、二阶变化方差展开、τ极限和常数anchor null均基于已有log-moment/变分工具。可探索的是它们与有限窗时空图、头预算一起在真实前缀外推中何时有效，及支持上的性质为何可能在未来失效。

## 4. 已发表的近期方向，不能重复空白主张

15. **Feng、Lin、Kao，Learning in the Fisher Subspace: A Guided Initialization for LoRA Fine-Tuning，ICML/PMLR306:30132–30155，2026。** [正式论文](https://proceedings.mlr.press/v306/feng26h.html)。Fisher引导低秩子空间已有直接近邻；仅叠加Fisher和LoRA不构成本轮的新问题。

16. **LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters，ICLR2026。** [正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8505a7e1da40320bf5692d7edec5aa45-Abstract-Conference.html)。参数化无关的低秩几何优化也已有正式工作，不以“黎曼＋低秩＋不变”作为无人研究的标签。

17. **Xing、Nagarajan，From confidence to caution: conformal gating for cross-session motor imagery brain-computer interfaces，Journal of Neural Engineering，2026-09-28在线发表，DOI10.1088/1741-2552/aead26。** [PubMed](https://pubmed.ncbi.nlm.nih.gov/42805221/)；[DOI](https://doi.org/10.1088/1741-2552/aead26)。跨会话MI门控已有直接文献。前缀与未来非交换、同20样本同时适配/校准时，不能直接引用标准conformal覆盖率。

18. **Gulrajani、Hashimoto，Identifiability Conditions for Domain Adaptation，ICML/PMLR162:7982–7997，2022。** [正式论文](https://proceedings.mlr.press/v162/gulrajani22a.html)。高阶矩条件与线性域映射可辨识性已有理论，不把一次协方差对齐当成唯一正确对应。

19. **Shrestha等，Domain Transfer Becomes Identifiable via a Single Alignment，ICML/PMLR306:112760–112782，2026。** [正式论文](https://proceedings.mlr.press/v306/shrestha26a.html)。其识别结论要求结构稀疏Jacobian及真实跨域配对anchor。当前仅有target标签，不具备同试次source–target配对，不能直接套该定理。

## 5. 新颖性陈述的可接受范围

有限检索发现大量邻近工作，因此目前可写“提出并检验一个任务特定的模型与机制分解”，不能写“首次使用时空/相位/低秩/Schur/DRO/冻头”。应以同预算、同输入、固定主比较和外部重复证据说明新增内容。

数学恒等式、软件正确性、支持风险性质、真实EEG识别与生理解释是五种不同证据。任一失败或负结果都应保留；引用顶刊顶会不替代本实验效果。数据和运行证据分别见PAN2023_DATA_NOTES.md、VALIDATION.json及线上交付记录。
