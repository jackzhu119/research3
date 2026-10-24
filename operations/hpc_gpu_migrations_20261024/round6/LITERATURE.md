# 第六轮一手文献依据与新颖性边界

检索核对日期2026-10-24。文献支持数学背景与对照，不代表本轮EEG已提升；不用二次综述当原始实验结果。

## 局部曲率与Bayes近似

**Daxberger等，Laplace Redux—Effortless Bayesian Deep Learning，NeurIPS34，2021。** [正式论文](https://proceedings.neurips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html)；[作者实现](https://github.com/runame/laplace-redux)。该文讨论从全网到最后层的实用Laplace近似。它支持研究小子网络的曲率不确定性，不证明准确率必升。本轮迹归一化、球投影后的有限空间点不冒称真实后验方差。

**Immer、Korzepa、Bauer，Improving predictions of Bayesian neural nets via local linearization，AISTATS，2021。** [作者论文](https://arxiv.org/abs/2008.08400)。GGN可解释为局部线性模型，其推断与原非线性预测可能不一致。本轮因此保留uq_linear，并将实际指数图平均明确标为候选受限预测规则，而不是由代理GGN自动得到的精确Bayesian预测。

**Botev、Ritter、Barber，Practical Gauss–Newton Optimisation for Deep Learning，ICML/PMLR70:557–565，2017。** [正式论文](https://proceedings.mlr.press/v70/botev17a.html)。提供Gauss–Newton近似的计算背景。本轮只处理r维空间及小残差头，不做整个EEGNet二阶训练，也不新增相应训练依赖。

## 有限空间点与数值求积

**Julier、Uhlmann，Unscented Filtering and Nonlinear Estimation，Proceedings of the IEEE92(3):401–422，2004。** [DOI](https://doi.org/10.1109/JPROC.2003.823141)；[大学托管论文](https://www3.nd.edu/~lemmon/courses/ee67033/pubs/unscented-filter-julier-proceedings-2004.pdf)。确定性点传播非线性矩是已有工具。本轮中心权1/2和2r个边点各1/(4r)只在投影前匹配指定一二阶矩；球投影后必须重新记录真实中心/协方差。

**Arasaratnam、Haykin，Cubature Kalman Filters，IEEE Transactions on Automatic Control54(6):1254–1269，2009。** [正式论文](https://ieeexplore.ieee.org/document/4982682/)。球面径向求积是少点近似多维Gaussian矩的经典背景。本轮不是Kalman滤波器，不读取未来query更新状态。点数/方向敏感性及GH9的数值精度属于近似质量，不能宣称精确积分。

一维Gauss–Hermite同为标准数值工具。r4的9个空间点无需9×9个骨干前向；独立配对的81个头组合复用9组空间特征。

## 概率质量与硬分类分开

**Rasmussen、Williams，Gaussian Processes for Machine Learning，MIT Press，2006，第3章。** [作者书站](https://gaussianprocess.org/gpml/)；[第3章](https://gaussianprocess.org/gpml/chapters/RW3.pdf)。概率预测、分类决策及非Gaussian似然近似是不同概念。同一中心margin上的对称logistic噪声积分不改变0.5决策边界，是本轮直接验证的经典对称性推论，不称新发现的普遍定理。

**Guo等，On Calibration of Modern Neural Networks，ICML/PMLR70:1321–1330，2017。** [正式论文](https://proceedings.mlr.press/v70/guo17a.html)。该工作说明概率校准值得独立评价，并给出简单温度缩放。本轮不借新外部query学习温度；NLL/Brier与BA分别报告，避免将更保守概率写成识别率提高。

## 新数据的来源

**Pan等，Journal of Neural Engineering，2023，DOI10.1088/1741-2552/ad0a01；Pan2023数据DOI10.7910/DVN/251NOW。** [原论文](https://doi.org/10.1088/1741-2552/ad0a01)；[作者数据](https://doi.org/10.7910/DVN/251NOW)；[登记元数据](https://api.datacite.org/dois/10.7910/DVN/251NOW)。登记14人、每人两会话、左右手MI、28通道，覆盖clinic11。实际schema/label/顺序及预处理按原文件核验，不依摘要猜轴。

两次真实会话有助于研究跨会话重复性，但发布试次顺序不自动等于真实物理采集顺序；缺失run边界不能伪造。固定外部评价减少继续在内部队列挑方案的偏差，不保证14人显著，也不认证真实受试者跨库绝无重合。已有Weibo仅作可选发布顺序LR稳健性，不把旧库别名包装成新取数。

## 本轮可探索贡献

在20个付费位置、冻结EEGNet和同低维空间预算下，以头竞争校正曲率决定点云方向，通过投影后同中心与保持完整离散边际的独立配对，分离“中心收缩”“对称概率平滑”“实际空间非线性”，再用预先固定的新双会话队列检验复现。

若只有NLL改善，写概率质量结论；若中心已解释全部收益，写收缩结论；若全形状不优于同迹isotropic，保留不需要额外复杂性的负结果。没有在运行前承诺显著、最优或临床有效。Schur补、Laplace、sigma点和GH保留经典归属，不改名冒充新理论。

数学定义见DESIGN_ZH.md；主比较/统计族见STATISTICS.md及同版计划。实际取数与软件/GPU执行范围按审计记录分别说明。
