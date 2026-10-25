# 第八轮一手文献与可声称的增量

检索核对日期：2026-10-25。链接指向原作者、正式期刊/会议、PubMed索引或预印本原文。数学性质与论文中其他数据上的效果不等于本轮已获得收益。

| 文献 | 本轮使用及边界 |
|---|---|
| He、Wu，**Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach**，IEEE TBME，2020。[PubMed](https://pubmed.ncbi.nlm.nih.gov/31034407/)，[作者IEEE介绍](https://brain.ieee.org/newsletter/2020-issue-1/transfer-learning-for-brain-computer-interfaces-euclidean-alignment-and-label-alignment/) | 协方差白化/对齐早已有直接EEG应用。本轮原support_EA是支持内单位迹均值、5%收缩和固定源重着色参考，并非完整复现其所有实验。 |
| Bhatia、Jain、Lim，**On the Bures–Wasserstein distance between positive definite matrices**，Expositiones Mathematicae 37(2):165–191，2019。[正式DOI](https://doi.org/10.1016/j.exmath.2018.01.002)，[作者原文](https://arxiv.org/abs/1712.01504) | SPD根、协方差最小位移及Bures路径是经典结果。应用到正则估计不使其成为未知总体或分类风险保证。 |
| Brockett，**Dynamical systems that sort lists, diagonalize matrices, and solve linear programming problems**，Linear Algebra and its Applications 146:79–91，1991。[正式DOI](https://doi.org/10.1016/0024-3795(91)90021-N) | 正交轨道上的commutator/双括号梯度已有研究。R8的单步方向、导数和二阶界是任务化推导，不称新发明的一般矩阵理论。 |
| Rodrigues等，**Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces**，IEEE TBME 66(8)，2019。[PubMed](https://pubmed.ncbi.nlm.nih.gov/30596565/) | EEG的重心、尺度和类中心旋转已有直接先例。R8研究冻结原始信号网络及付费前缀，不把有限一步控制称为完整RPA复现。 |
| He、Wu，**Different Set Domain Adaptation for Brain–Computer Interfaces: A Label Alignment Approach**，IEEE TNSRE 28(5):1091–1108，2020。[PubMed](https://pubmed.ncbi.nlm.nih.gov/32167903/) | 使用极少目标标签作类别协方差对齐已有先例。“使用标签校正协方差方向”本身不是首次。 |
| **Transfer learning with data alignment and optimal transport for EEG based motor imagery classification**，Journal of Neural Engineering，2024。[PubMed](https://pubmed.ncbi.nlm.nih.gov/38232381/) | 数据对齐与OT已用于MI，不以“首次OT用于EEG”作为创新定位。当前有限支持与冻结分类器协议不同，不能直接套用其效果。 |
| **Transfer Learning With Optimal Transportation and Frequency Mixup for EEG-Based Motor Imagery Recognition**，IEEE，2022。[IEEE原论文](https://ieeexplore.ieee.org/document/9910147/) | 联合分布OT和频率混合已有相关方法。R8不把本轮小矩阵闭式运输等同于该方法或其完整理论。 |
| Li等，**SPDAlign: Interpretable Riemannian Alignment for EEG Forward Modeling Shifts**，arXiv预印本，2026-10-05。[原文](https://arxiv.org/abs/2610.06315)，[全文](https://arxiv.org/html/2610.06315v1) | 近期直接近邻：SPD表示、均值和Wasserstein Procrustes旋转。它是预印本，未核实为正式期刊论文。R8需用严格前20输入、无未来无标签数据、冻结EEGNet及同协方差控制说明任务增量。 |

可检验的成果定位是：**少样本校准中，平均协方差已对齐后还剩哪些与分类有关的方向；运输强度与估计收缩如何分别影响未来预测；很少标签是否足以改善受限方向而不必拟合残差头。** 需要实际配对结果、阴性控制和明确的数据范围才能形成论文结论。

有限检索不能证明“无人提出”。Bures最优位移、commutator局部上升、协方差匹配和真实EEG准确率是不同证据；其中任何一个成立都不自动证明另一个。已查看过的内部/Pan结果使R8属于探索性复验，海量网格不恢复盲测资格。
