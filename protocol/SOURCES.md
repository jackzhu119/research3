# 参考文献与原始数据入口

核查日期：2026-10-18。方法描述应与 METHOD_SPEC.md、REFERENCE_METHODS.md 配合阅读；数据的版本、字节与未完成审计见 DATA_AUDIT.md。参考论文的结果不是第三篇的预期结果。

## 核心引用

- [R01] Wang et al. Sensory-guided human-machine joint learning accelerates the acquisition of motor imagery brain computer interface control. Nature Communications, 2026-07-15. DOI: 10.1038/s41467-026-75435-5. https://www.nature.com/articles/s41467-026-75435-5 。代码 https://github.com/bfinl/SensoryGuidedJointLearning 。数据 https://doi.org/10.1184/R1/32293995 。
- [R02] Ding et al. EEG-based brain-computer interface enables real-time robotic hand control at individual finger level. Nature Communications, 2025-06-30. DOI: 10.1038/s41467-025-61064-x. https://www.nature.com/articles/s41467-025-61064-x 。代码 https://github.com/bfinl/Finger-BCI-Decoding 。数据 https://doi.org/10.1184/R1/29104040 。
- [R03] Lee et al. Brain–computer interface control with artificial intelligence copilots. Nature Machine Intelligence, 2025-09-01. DOI: 10.1038/s42256-025-01090-y. https://www.nature.com/articles/s42256-025-01090-y 。代码 https://github.com/kaolab-research/bci_raspy 。公开数据 https://zenodo.org/records/15165133 。
- [R04] Yan et al. Positive-Congruent Training: Towards Regression-Free Model Updates. CVPR 2021, 14299–14308. 作者版本v3 https://arxiv.org/abs/2011.09161v3 。会议原文 https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html 。negative flip与focal distillation是既有工作。
- [R05] Jiang, Kwok and Zhang. Subspace Learning for Effective Meta-Learning. ICML 2022, PMLR 162:10177–10194. https://proceedings.mlr.press/v162/jiang22b.html 。原文PDF https://proceedings.mlr.press/v162/jiang22b/jiang22b.pdf 。外层学basis、内层学系数是直接相关的先例。
- [R06] Nguyen, Sun and Johnson. Lightweight Adaptation of EEG Foundation Models for Stroke Motor Imagery Decoding: Domain Shift and Subject-Level Robustness. arXiv:2609.00282v1, 2026-08-31. https://arxiv.org/abs/2609.00282v1 。这是预印本，不写成已发表SCI文章；队列内交叉验证不等于健康到患者的零样本迁移。
- [R07] Liu et al. MIRepNet: A pipeline and pre-trained model for EEG-based motor imagery classification. Knowledge-Based Systems 343, 115966, 2026. DOI: 10.1016/j.knosys.2026.115966. https://www.sciencedirect.com/science/article/pii/S0950705126006921 。作者预印本 https://arxiv.org/abs/2507.20254 。代码 https://github.com/staraink/MIRepNet 。权重 https://huggingface.co/starself/MIRepNet 。使用前按原文Table 1和实际checkpoint核对预训练来源。
- [R08] Luo et al. Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding. ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/36eeec256f89eba9ed60cbca73ea92ed-Abstract-Conference.html 。代码 https://github.com/luo-huan-123/BTTA-DG 。
- [R09] Liu et al. EEG-FM-Compass: progress, benchmarking and future directions for EEG foundation models. National Science Review, nwag466, 2026-08-03，排版版本2026-10-12. DOI: 10.1093/nsr/nwag466. https://doi.org/10.1093/nsr/nwag466 。代码 https://github.com/Dingkun0817/EEG-FM-Benchmark 。NSR不是Nature子刊，ICLR/ICML/CVPR是会议而非SCI期刊。
- [R10] Stieger, Engel and He. Continuous sensorimotor rhythm based brain computer interface learning in a large population. Scientific Data 8, 98, 2021-04-01. DOI: 10.1038/s41597-021-00883-1. https://link.springer.com/article/10.1038/s41597-021-00883-1 。
- [R11] Stieger. Human EEG Dataset for Brain-Computer Interface and Meditation. 官方归档 https://figshare.com/articles/dataset/Human_EEG_Dataset_for_Brain-Computer_Interface_and_Meditation/13123148 ，DOI: 10.6084/m9.figshare.13123148 。本轮页面显示351.01GB、62个发布电极、CC BY4.0；精确逐文件字节仍须Work获取manifest。
- [R12] HS-Stroke原论文 DOI: 10.1038/s41597-026-07742-x 。固定修订数据v6 https://doi.org/10.6084/m9.figshare.30747806.v6 ，官方API https://api.figshare.com/v2/articles/30747806/versions/6 。作者修订与适配代码 https://github.com/KINLAWWW/HS-Stroke 。以作者去重说明和v6身份为准，不使用论文旧摘要的重复trial数。
- [R13] Wang joint-learning数据 https://doi.org/10.1184/R1/32293995 及[R01]的Experimental design、Data availability和原始作者代码。代码版本、对象shape、组别、MI起点及physical units需逐项核验。
- [R14] 论文1固定研究快照0539379578818b3d71f995192e5e8a81cf4d5aef。明确标签审计 https://github.com/jackzhu119/cross-subject-mi-eeg/blob/0539379578818b3d71f995192e5e8a81cf4d5aef/research_runs/YANG_LABEL_ELIGIBILITY_20261018/README_zh.md 。时钟审计 https://github.com/jackzhu119/cross-subject-mi-eeg/blob/0539379578818b3d71f995192e5e8a81cf4d5aef/research_runs/YANG_MARKER_AUDIT_20261018/README_zh.md 。Yang原数据论文 DOI: 10.1038/s41597-025-04826-y 。
- [R15] 论文2固定科学快照19a5909fd76d96be9316559602010932ccec81a8。 https://github.com/jackzhu119/few-shot-mi-eeg/blob/19a5909fd76d96be9316559602010932ccec81a8/docs/kumar_innovation_revision_20261017.md 。已有后续方案 https://github.com/jackzhu119/few-shot-mi-eeg/blob/19a5909fd76d96be9316559602010932ccec81a8/docs/kumar_component_exploratory_protocol_20261016.md 。Kumar NEMAR v1.0.5 DOI: 10.82901/nemar.nm000177.v1.0.5 。
- [R16] Trocellier et al. Visual cues in MI-BCI induce biases in EEG classification models with deep learning but not with standard machine learning. Frontiers in Neuroergonomics, 2026-09-28. DOI: 10.3389/fnrgo.2026.1804143. https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2026.1804143/full 。
- [R17] PyTorch2.8. Reproducibility. https://docs.pytorch.org/docs/2.8/notes/randomness.html 。
- [R18] PyTorch2.8. CUDA semantics. https://docs.pytorch.org/docs/2.8/notes/cuda.html 。

## 其余直接方法与数据来源

- Sun et al. Meta-Transfer Learning for Few-Shot Learning. CVPR2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.html 。
- Wang et al. CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. ICLR2025. https://github.com/wjq-learning/CBraMod ，会议页 https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbbd6d915cb90be21c1254a82d45cedd-Abstract-Conference.html 。
- Wimpff et al. Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study. EMBC2025. DOI: 10.1109/EMBC58623.2025.11253543. https://arxiv.org/abs/2502.06828 ，https://github.com/martinwimpff/eeg-continual 。
- Shen and Degras. A confidence-gated source selection strategy for cross-session transfer in brain–computer interfaces. Frontiers in Human Neuroscience, 2026-08-28. DOI: 10.3389/fnhum.2026.1895016. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2026.1895016/full ，https://github.com/Yiming-S/MSDA-Bench 。
- Basit, Rehman and Shafique. EEG-Fusion: Failure-Informed Source-Free Expert Routing for Robust Motor Imagery EEG Decoding. arXiv:2609.33962v1, 2026-09-27. https://arxiv.org/pdf/2609.33962 ，https://arxiv.org/html/2609.33962 。预印本；本轮以v1 PDF内容为准，不继承旧搜索摘要中互相冲突的样本数或效果数字。
- BNCI2014_001 https://bnci-horizon-2020.eu/database/data-sets 。Cho2017 DOI: 10.1093/gigascience/gix034，数据10.5524/100295。OpenBMI DOI:10.1093/gigascience/giz002，数据10.5524/100542。PhysioNet https://physionet.org/content/eegmmidb/1.0.0/ 。
- Liu2024急性卒中数据论文 DOI:10.1038/s41597-023-02787-8，数据 https://doi.org/10.6084/m9.figshare.21679035.v5 。
- UET175原论文 DOI:10.3389/fnins.2025.1580931，https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1580931/full 。作者原档和访问限制见DATA_AUDIT，不能照搬反应强度选出的时间戳进行确认。
- SEFMEID2026 DOI:10.1038/s41597-026-08298-6，数据 DOI:10.57760/sciencedb.30795。PROTEUS公开版 https://www.nemar.org/dataset/nm000290 。
- Alawieh2025 DOI:10.1073/pnas.2418920122，https://zenodo.org/records/15454355 。Forenzo2023 DOI:10.1109/TBME.2023.3298957，https://pmc.ncbi.nlm.nih.gov/articles/PMC10803074/ 。
- 科研借鉴而非主训练数据：Willett et al. A high-performance speech neuroprosthesis. Nature2023, DOI:10.1038/s41586-023-06377-x；Flesher et al. A brain-computer interface that evokes tactile sensations improves robotic arm control. Science2021, DOI:10.1126/science.abd0380。它们使用侵入式记录/刺激，不能直接并入头皮左右手MI。

## 固定版本规则

Work应在首次实际访问时保存论文版本、数据版本、代码commit、checkpoint SHA-256、许可和下载时间。官方页面滚动更新时，记录新旧差异并生成新的输入版本；不能只把文献年份写成版本锁定。除明确已核验的HS v6字节和旧基线ZIP哈希外，未实测值维持null。
