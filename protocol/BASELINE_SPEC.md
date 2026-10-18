# 对照方法的实施规格

本文件把 EXPERIMENT_PROTOCOL 中的 B0–B11 落到可复现设置。所有数值是预先规定的开发起点，不是已验证最优值。每个方法最多9个开发配置；修改必须在内部留出或外测之前登记新版本。数学基线与神经基线不假装具有同样的optimizer步数，分别报告标签、参数、更新、耗时。

## 1 共同契约

相同原始20trial前缀、相同有效support、相同未来query、同一输入窗口和clinic11。supervised方法有效support缺类时均回退各自source模型，计入失败/回退统计。参与者、session、trial资格对全部方法相同。不会为了凑类别向未来补采。

所有基于EEGNet的更新共享完全相同f0；其源权重、source classifier和BN buffers作为只读起点。除B2明确允许的参数以外都冻结，所有神经方法在目标阶段均使用eval模式，关闭dropout并固定BN统计。源logits的标签0/1顺序固定；由同一logits生成probability和argmax，tie选择0。

空间方法统一使用METHOD_SPEC的Q与Rref坐标。变换之后才做相同的联合trial RMS，不加原始幅值epsilon；相对通道增益不会被逐通道z-score消除。Rref来自四个source，任何对照不得额外使用query或完整无标签目标session来估计参考。

## 2 基础神经与协方差对照

| ID | 参数与优化 | 开发配置 |
|---|---|---|
| B0 frozen | f0直接推断，无目标优化 | 1个固定设置 |
| B1 head-only | θ固定0；原双logit加对称margin头增量；full-support SGD 10步，类平衡BCE | head lr={0.003,0.01,0.03} × 头L2系数={0,0.001,0.01}，按p+1归一化，共9 |
| B2 full fine-tuning | 复制f0，更新全部可训练权重与头，BN running统计固定；full-support SGD10步；以源参数作L2起点 | lr={0.0001,0.0003,0.001} × L2-to-source={0,0.0001,0.001}，按可训练参数数归一化，共9 |
| B3 source-covariance alignment + head | 从合法support估计trace-normalized CAR协方差Ctarget，收缩得到Rt；固定A=QᵀRref^(1/2)Rt^(−1/2)Q，再做B1同形头10步 | target covariance shrink={0.01,0.05,0.1} × head lr={0.003,0.01,0.03}；头L2=0.001，共9 |

B3是“将目标协方差匹配到固定source参考”的对齐，必须给出上述方向和矩阵次序；不把它无说明写成作者EA原版。传统EA会把source和target都映射到其各自平均协方差的白化空间，需另行训练匹配source输入的编码器。若补做原版EA，使用独立source checkpoint与同预算support-only目标参考，列为补充强对照；不能把未对齐的f0只做一端白化就冒充完整原算法。

B2的目标明确为 `Lbal(S) + lambda*||psi-psi0||2^2/Ntrainable`：Lbal沿用METHOD_SPEC的类别等权BCE，psi包含全部可训练参数及分类头，含偏置；psi0是同一source checkpoint，Ntrainable是psi元素数，BN running buffers不在其中。

相关原始先例：He and Wu, Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach，IEEE TBME，DOI10.1109/TBME.2019.2913914，作者稿 https://arxiv.org/abs/1808.05464 。Sun, Feng and Saenko, Correlation Alignment for Unsupervised Domain Adaptation，https://arxiv.org/abs/1612.01939 。本项目B3是明确适配到固定f0坐标的比较版本，不是宣称逐行复现以上论文。

## 3 传统小样本基线

### B4 目标CSP加源先验和LDA

在相同10维CAR坐标中计算每trial的trace-normalized协方差。source各类别均值按source数据集、参与者等权，target各类别均值只来自合法support。类别c的拟合协方差为 `(1-alpha)*Ctarget_c + alpha*Csource_c`，再用0.1的单位迹各向同性收缩。解CSP广义特征问题，取最大与最小各m个方向，提取log normalized variance特征，在target support上拟合`LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5,0.5])`。

开发网格alpha={0,0.5,1}、m={1,2,3}，共9。alpha=0是target-only完整CSP重拟合，alpha=1是source CSP固定、只用target重新拟合LDA；中间值是源先验收缩。源CSP和源LDA均保存frozen参照。所有分母、排序和特征符号采用确定性约定；不把source全量trial与20个target简单拼接后将source权重淹没目标，除非作为另行标明的对照。

B4先验方案是本研究的强传统实现，不能冒称与论文2的源标签规模、窗口和训练公式完全相同。论文2原协议的复核放在Kumar已探索数据中。

### B5 切空间特征的目标分类器校准

保留旧包已经定义的log-Euclidean参考切空间特征，不能写成默认仿射不变均值。每trial在10维CAR空间做trace归一化和SPD收缩；source参考为source各log协方差加权均值的矩阵指数。切空间特征取参考白化后协方差的矩阵log，上三角非对角乘sqrt(2)。source特征标准化器与source logistic classifier固定作为初始模型。

target仅使用support，通过类平衡BCE加L2-to-source校准logistic margin；使用确定性L-BFGS、最多200次迭代、梯度容差1e-8，记录是否收敛。开发网格trial covariance shrink={0.01,0.05,0.1}、L2-to-source={0.01,0.1,1}，共9。每个shrink对应合法source参考与分类器，source模型拟合另计任务数。不能把传统精确优化的迭代次数冒充与神经10步是相同运算量。

B5的目标明确为 `Lbal(S) + lambda*(||w-w0||2^2+(b-b0)^2)/(q+1)`，其中q是固定source标准化后的切空间特征维数，w0、b0来自匹配shrink的source logistic classifier；偏置参与正则。Lbal沿用METHOD_SPEC的类别等权BCE，标准化器完全冻结，不用support重新定义特征尺度。

## 4 自由空间更新与元学习强对照

### B6 与 B7

B6在白化10维空间使用 `T=I+UVᵀ`。rank k={1,2,4}；U0用固定pipeline seed生成后薄QR，V0=0，初始T严格为I。目标更新U、V与相同margin头10步，因子学习率0.05、头0.01。内层用类平衡BCE、0.01*||UVᵀ||F²及METHOD_SPEC相同头正则；每次更新后若||UVᵀ||F>rho，将V乘rho/该范数。rho={0.05,0.15,0.35}，共9配置。记录真实2dk+p+1参数数。

B7在B6上加METHOD_SPEC的既有PCT focal logit matching，alpha=1、beta=5、lambda=0.1。teacher和old-correct mask在目标episode内固定，不能用适应后的模型重新生成“原本正确”集合。PCT的形式来自既有论文，系数是本研究冻结的实施规格。

### B8 与 B9

B8使用METHOD_SPEC同样无迹正交字典、同样r与rho网格、同样Stieger40数据、1000×4episode外层与10步内层，但把`exp(M)`改为`I+M`。它必须独立元训练并在相同9格中选择，不能直接套P的最优配置后称公平。由于rho<1，B8同样可逆；指数映射不是本研究唯一可逆方案。

B9以pipeline seed初始化相同r个无迹正交生成元并永久冻结。空间系数与头按P拟合；它没有G元训练。r、rho同9格。重复seed用于随机字典和整个pipeline的稳定性，不能挑最好的随机字典。

### B10 与 B11

B10为Meta-LoRA。外层只学习U0，保持V0=0；用10×k的列正交参数和与METHOD_SPEC相同的Stiefel投影/QR更新使起点确定。内层更新U、V及头的规则与B6一致；外层按相同的40人episode、20trial、1000次更新和4episode batch，用完整unroll的未来类平衡损失训练U0。k、rho同9格，outer学习率0.01、无momentum、梯度整体范数上限1。

B11在Meta-LoRA内层加同一PCT项；需要独立元训练，使比较涵盖“元学习加PCT”这条强组合。其调参、验证频率、seed和query权限与P一致。所有基线都由同一source f0开始，不能只给P完整纵向训练数据。

## 5 主候选和最强对照选择

P为METHOD_SPEC唯一主方法，PCT系数为0；P+PCT另作消融，不自动替换P。B*候选集合是B1到B11，B0只作冻结参照。B*按3条确认pipeline在Stieger validation上20trial的参与者均值选择；先取距最高均值不超过0.5个百分点的集合，再按同设备预定计时的校准耗时中位数、最后按编号选择，记录在外测前冻结。P若低于单列的B0冻结参照，不能仅凭超过B*称为实用改善。

确认阶段同一输入、source分支、seed和query给全部方法。未成功实现或未经过相同开发搜索的强对照不能被悄悄删除；应先完成它，或在打开外部query前明确缩小尚不能证明的主张。开发工程smoke仅验证实现，不用于报告科学效果。

## 6 输出要求

每次拟合保存公式与实现版本、具体网格项、source依赖、目标训练参数数、实际更新数、target prefix UID、query UID、耗时/显存、失败或回退原因、最终probability/logits及checkpoint SHA。所有方法在同一participant层面比较，不用窗口或seed充当独立样本。
