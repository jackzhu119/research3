# 第五轮一手文献核验与贡献边界

核验日期：2026-10-24（Asia/Shanghai）。下列原始论文站点、作者/机构存档和官方会议记录支持方法来源及限制，不预测本轮性能。未做穷尽式新颖性检索，不能声称首先将极分解、BCH、splitting或Lie方法用于EEG。

| 一手来源 | 已核验的事实及本轮边界 |
|---|---|
| N. J. Higham, *Computing the Polar Decomposition—with Applications*, SIAM Journal on Scientific and Statistical Computing7(4):1160–1174,1986. [作者机构存档](https://eprints.maths.manchester.ac.uk/694/), [DOI](https://doi.org/10.1137/0907079) | 极分解及其稳定计算为经典矩阵分析。这里使用expK和expS直接构造旋转/正定伸展，不是提出新的极分解算法，也不需要将每次10×10矩阵先数值极分解再拟合。 |
| G. Strang, *On the Construction and Comparison of Difference Schemes*, SIAM Journal on Numerical Analysis5(3):506–517,1968. [DOI](https://doi.org/10.1137/0705041), [作者MIT论文目录](https://math.mit.edu/~gs/papers/papers2.html) | 对称分裂及误差阶是经典数值分析；本轮检验其作为低维输入校准图的归纳偏置，不把它改名为新的splitting定理。 |
| N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM,2008,第8章. [出版社章节](https://doi.org/10.1137/1.9780898717778.ch8) | 正交×正定的极分解、矩阵指数及矩阵函数基础。小字典图与完整矩阵空间须区分，不能由完整矩阵分解存在推断rank4目标校准有全局表达优势。 |
| P. Neff, Y. Nakatsukasa & A. Fischle, *A Logarithmic Minimization Property of the Unitary Polar Factor in the Spectral and Frobenius Norms*, SIAM Journal on Matrix Analysis and Applications. [出版社/DOI](https://doi.org/10.1137/130909949) | 摘要的谱范数结论为任意n，Frobenius对数结论只写n≤3。本文10维任务不能直接援引其Frobenius结论声称polar全局最优；本轮的正交相似和相同奇异值性质有直接初等证明，不依赖超范围引用。 |
| P. L. C. Rodrigues, C. Jutten & M. Congedo, *Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces*, IEEE TBME66(8):2390–2401,2019. [PubMed原始记录](https://pubmed.ncbi.nlm.nih.gov/30596565/), [DOI](https://doi.org/10.1109/TBME.2018.2889705) | 用平移、缩放和旋转对齐EEG分布已有直接先例。不能把“分开讨论旋转与增益”当独立创新。该工作处理协方差分布，本轮固定源神经网络并在原始时序输入上以相同r个参数检验组合图。 |
| P. Zanini et al., *Transfer Learning: A Riemannian Geometry Framework With Applications to Brain–Computer Interfaces*, IEEE TBME. [PubMed](https://pubmed.ncbi.nlm.nih.gov/28841546/), [DOI](https://doi.org/10.1109/TBME.2017.2742541) | 协方差流形上的域迁移框架已有基础，不能声称首次几何迁移。本文保持原信号/冻结编码器，研究的是低秩Lie作用的耦合曲率，不重新建立协方差分类器。 |
| A. Bleuzé, J. Mattout & M. Congedo, *Tangent space alignment: Transfer learning for Brain-Computer Interface*, Frontiers in Human Neuroscience16:1049985,2022. [原文](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1049985/full) | 切空间中的EEG域对齐及跨数据集评价已有先例。本轮的“同身份切线、不同高阶耦合”与单纯再做切空间映射不同；不能把其多个数据库上的结果当成本任务预期收益。 |
| S. Li, M. Kawanabe & R. Kobler, *SPDIM: Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG*, ICLR2025. [官方会议记录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc6cd575b8340be6ee70beaed9ca4015-Abstract-Conference.html) | 已有单个SPD目标参数的高效几何域适应以及生成式漂移分析。本轮不是首次parameter-efficient EEG geometry；区别在有标签前20原始位置、共享低维字典、旋转/增益同谱的图对照及其反例。 |
| Z. Chen, Y. Song, Y. Liu & N. Sebe, *A Lie Group Approach to Riemannian Batch Normalization*, ICLR2024. [官方会议记录](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7f2257d2b291b8d7e712c70b67e09412-Abstract-Conference.html), [作者机构存档](https://iris.unitn.it/handle/11572/423094) | Lie群归一化及EEG分类已有直接研究。本轮不改变冻结网络BN统计，不把Lie群一词本身当创新，也不把群均值/方差控制结论移植成少样本未来风险保证。 |
| A. Partovi, D. B. Grayden & A. N. Burkitt, *POC-CSP: a novel parameterised and orthogonally-constrained neural network layer for learning common spatial patterns (CSP) in EEG signals*, Journal of Neural Engineering22(3):036037,2025. [PubMed原始记录](https://pubmed.ncbi.nlm.nih.gov/40367961/), [DOI](https://doi.org/10.1088/1741-2552/add8bc) | 已有Lie参数化、保持数学约束的EEG空间滤波神经层。不能声称第一次训练正交EEG空间参数；其训练/微调数据预算与本轮冻结网络前20标签不同，不能拿论文报告准确率作直接基准。 |
| V. J. Lawhern et al., *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*, Journal of Neural Engineering15:056013,2018. [DOI](https://doi.org/10.1088/1741-2552/aace8c) | 保留现有紧凑源编码器及其输入合同，不将骨干性能当本轮新算法贡献。 |

## 本轮可审查的增量

候选题目是“固定前缀采集与低秩参数预算下，旋转—相对增益的高阶耦合是否影响冻结EEG分类器的迁移”。数学部分应明确三个层级：

1. **精确代数**：同r、同身份Jacobian；三个split在共享θ时正交相似，白化增益谱、位移酉不变范数及旋转极因子相同；纯分量/交换空效应及逆向对称性。它们不是新增基础定理，可由公式直接证明。
2. **算法选择**：同参数实际非线性支持优化与元训练图、共同B的对照、共享θ后验探针、同预算与相同预测/数据处理。它们需实现和梯度测试验证，不能只换推理公式便称训练算法进步。
3. **经验科研结论**：真实未来查询上的配对识别差异、失效样例、计算开销及是否与控制好的非交换漂移交互。即使代数性质成立，效果也可能为零或变差；拟合后几何诊断与准确率相关不自动构成因果证据。

本轮控制压力共用同一个增益S，并改变相同旋转角谱K的作用平面，形成可交换/不可交换的谱匹配对照；不能只把S改成另一种谱后将结果全部归于非交换性。跨条件的方向、实际变形量与冻结分类器关系仍需记录，不能由同谱直接推断风险匹配。

统计单位、主要比较族和软件验证范围须按同版实验计划记录。对既有内部队列反复比较应一直标注探索性，外部发布顺序数据也不伪装成严格真实采集顺序的前瞻试验。
