# 第六轮：头竞争校正的曲率形状与受限空间预测平均

## 1. 研究问题与增量

前五轮已分别研究低维Lie方向、支持证据门控、Schur信息半径、支持内交叉拟合和非交换几何图。本轮进一步问：**只有20个已获取位置时，是否应沿分类头竞争后仍不确定的空间方向作受限预测平均，而不是只用一个空间拟合点？**

保持冻结EEGNet、同一个rank-4字典、共同支持拟合中心和原分类头。增加预测点，不增加目标会话可学习参数或目标数据，不根据未来query选点/选超参数。主要结论优先来自新Pan双会话外部评价，内部队列继续标为探索性复验。

最关键的对照是投影后同中心预测、保持空间/头边际但删除配对相关性的预测，以及硬分类严格不变的线性平滑控制。本轮称“曲率形状引导的受限空间预测平均”，不称精确贝叶斯后验、精确截断高斯积分、无害迁移保证或全局最优算法。

## 2. 固定输入、空间图与共同中心

沿用clinic11、CAR后10维、128Hz、8–30Hz、固定窗与pre-RMS缓存；源EEGNet、BN和源参考不更新。预算按原获取位置付费，QC失败不从后面补齐；B4/8/12/16/20共用B20之后的未来query。

固定原联合指数图，不用第五轮部分结果挑图：
\[
M(\theta)=\operatorname{mat}(EB\theta),\qquad
A(\theta)=Q^TR_{ref}^{1/2}e^{M(\theta)}R_{ref}^{-1/2}Q .
\]
B为99×r列正交字典。主点r=4、ρ=.15、λθ=.01、λa=.1。实际变换和联合RMS后的增广特征为 \(z_\theta(x)=[\phi_\theta(x);1]\)，预测margin
\[
m(x;\theta,a)=m_\theta(x)+z_\theta(x)^Ta .
\]
共同中心采用稳定的core.tangent_profiled，执行10次Newton/Schur更新。它是有限步、受约束的固定身份切线代理中心，不是精确非线性MAP，也不保证有限步已到代理最优。
\[
\tilde m_i=m_{0i}+j_i^T\theta+z_i^Ta,\quad
L_S=\sum_iq_i\ell(\tilde m_i,y_i)+\lambda_\theta\|\theta\|^2+
\lambda_a\|a\|^2/d,\qquad\|\theta\|\le\rho .
\]
身份源导数j与源特征z固定，q为类别平衡且和为1的权，d为增广头维数。正式预测仍经过实际指数图和实际变换后特征。

## 3. 头竞争校正曲率与迹预算

在中心代理margin上取 \(w_i=q_i\sigma(\tilde m_i)(1-\sigma(\tilde m_i))\)，W为其对角矩阵：
\[
\Lambda=\begin{bmatrix} A&C\\C^T&D\end{bmatrix},\quad
A=J^TWJ+2\lambda_\theta I,\quad C=J^TWZ,\quad
D=Z^TWZ+(2\lambda_a/d)I.
\]
这里包含科学ridge，**不包含求解器数值damping**。两个ridge正，因此全块与Schur补 \(S=A-CD^{-1}C^T\) 正定。定义 \(K=D^{-1}C^T\)，它给出局部头竞争补偿。

原损失是平衡均值，直接把Hessian逆当20独立样本真实后验有尺度问题。本轮只使用曲率形状：
\[
\tau=\frac{\kappa\rho^2}{\operatorname{tr}(S^{-1})},\qquad
\Sigma_\theta=\tau S^{-1},\qquad V_a=\tau D^{-1}.
\]
主κ=.25，敏感性κ=.0625/1；投影前 \(\operatorname{tr}\Sigma_\theta=\kappa\rho^2\)。τ不是已校准Bayes温度，固定迹后不能声称绝对空间不确定性自然随支持数下降。支持改变形状、头补偿与预测分散。

将整个科学目标同乘正数时，以上Σθ、Va、K不变；只缩放数据项、只改ridge或混入数值damping不具该等价性。S是固定代理的剩余曲率，不是实际指数网络的精确Hessian。

## 4. 正权空间点、球投影与中心偏移

令 \(LL^T=\Sigma_\theta\)，使用2r+1点：
\[
\delta_0=0,\ \omega_0=\tfrac12;\qquad
\delta_{j,\pm}=\pm\sqrt{2r}Le_j,\ \omega_{j,\pm}=\tfrac1{4r}.
\]
投影前加权均值0、协方差Σθ精确成立，高阶矩不等于Gaussian；r4为9点。
\[
\theta_i=\Pi_\rho(\hat\theta+\delta_i),\quad
\Delta_i=\theta_i-\hat\theta,\quad a_i=\hat a-K\Delta_i .
\]
负号由局部块二次完成平方给出，补偿使用**投影后**Δi。投影后重算 \(\bar\theta=\sum\omega_i\theta_i,\bar a=\sum\omega_i a_i\) 与实际协方差。它们不再自动等于原中心与Σθ，不能称精确截断Gaussian矩。

有限轴规则有方向依赖：Cholesky列、对称平方根列或重根特征轴，都不自动具有连续Gaussian积分完整的B→BO换基不变性。保留预定方向/点数敏感性，不用未来query选择最有利stencil，也不宣称有限求积全局坐标不变。

## 5. 实际空间预测与头余项积分

空间点i给出实际margin \(m_i=m(x;\theta_i,a_i)\) 与增广特征zi。头余项ε~N(0,Va)，其标量方差 \(v_i=z_i^TV_az_i\)。固定GH9：
\[
P_i(x)\simeq\frac1{\sqrt\pi}\sum_{k=1}^9h_k
\sigma(m_i+\sqrt{2v_i}\,t_k),\qquad
p_{\rm joint}(x)=\sum_i\omega_iP_i(x).
\]
GH9是数值近似。固定特征时头线性，所以无需为每个头积分点重复EEGNet前向；9个空间点最多9组实际空间特征。实际执行及峰值内存/时延记录，不只报参数数目。

## 6. 十种方法及关键控制

| 程序名 | 定义/识别对象 |
|---|---|
| uq_map | 原有限步中心的一次实际指数图预测；名称为简写，不宣称精确MAP |
| uq_head | θ固定原中心，仅积分对称头余项 |
| uq_linear | 以同一个实际中心margin为均值的线性Gaussian控制，空间梯度取实际图 |
| uq_spatial | 同空间点及条件头均值，去GH头余项 |
| uq_joint | Schur形状空间点＋条件头负相关补偿＋GH余项，主候选 |
| uq_isotropic | 同迹的各向同性空间形状，保持同τ、K、D和头余项 |
| uq_independent | 空间点与头均值点独立笛卡尔配对，保留完整离散边际，删除配对相关 |
| uq_mean | 投影后加权θ/a均值处的一次实际预测 |
| uq_mean_head | 与uq_mean同中心，加同头GH余项，是主要中心匹配对照 |
| uq_diagonal | 同迹对角空间形状，保持同τ、K、D和余项，检查方向相关性 |

独立性控制严格为
\[
p_{\rm indep}(x)=\sum_{i,j}\omega_i\omega_j
E_\epsilon\,\sigma(m(x;\theta_i,a_j)+z_{\theta_i}(x)^T\epsilon).
\]
它保留joint的空间点边际、头均值点边际与头余项，只删除i=j配对相关。r4为81个标量头组合，仍复用9组空间特征。仅把ai改回ahat会同时删除头边际方差，不是这里的控制。

joint对mean_head分离预测平均与投影中心漂移；joint对isotropic衡量形状及其投影影响的整体贡献，不能单独称“纯形状因果效应”。辅助报告每个点云的平均预测与其自身中心预测的差。

## 7. 可验证空效应和曲率机制

若Z是对称零均值标量噪声，\(g(m)=E\sigma(m+Z)\) 满足g(−m)=1−g(m)、g′>0，因此g(m)>1/2当且仅当m>0。正权成对GH亦满足。

因此uq_head、uq_linear应与原中心硬分类逐位相同（数值零margin按固定tie规则），BA/纠错/负翻转不变；NLL/Brier仍可改变。这是软件与机制检验，不给学校新增启动门槛。

小零均值扰动有二阶展开：
\[
E\sigma(m(w))\approx\sigma(m_0)+
\tfrac12\sigma'(m_0)\operatorname{tr}(H_m\Sigma)+
\tfrac12\sigma''(m_0)\nabla m_0^T\Sigma\nabla m_0 .
\]
m0=0时σ″(0)=0，对称方差平滑的领先边界位移为零。实际空间平均改变分类，可能来自logit曲率或明确记录的投影中心偏移；该局部展开不保证有限ρ下风险改善。

## 8. 精确矩阵与主要比较

主计划198逻辑任务＝126内部＋72新Pan；另可选Weibo36，不自动加入主包。

| 内部家族 | 数量 |
|---|---:|
| map/joint两种字典目标×3种子 | 6 |
| 两种字典×10种预测器×3种子 | 60 |
| 预定敏感性 | 48 |
| 预定参考 | 12 |
| 内部总计 | 126 |
| Pan固定交叉评价60＋参考12 | 72 |
| 主包总计 | 198 |

只有6项训练本轮字典，其他主要校准评价，不是198次重训EEGNet。字典最多1000outer×4episode，验证/断点/种子规则匹配。

敏感48＝κ=.0625/1×joint/isotropic/mean_head×3种子（18）＋GH5/17×joint×3（6）＋中心权0的joint/mean_head×3（6）＋预定旋转轴的joint/mean_head×3（6）＋ρ=.05/.35×joint/mean_head×3（12）；均只评价B20。与主点相比只改变所指变量，不用敏感性挑外部最佳方案。

参考12＝仅头Newton、支持EA、第五轮实际图精化geometry_refined、第四轮xfit_stack，各3种子。可选Weibo＝map字典10预测器×3种子＋head/EA×3＝36。精确输出身份和依赖以同版EXPERIMENT_MATRIX.json/plan.py为准。

八项预定比较：同map字典上joint分别对map、mean_head、isotropic、independent、spatial、head、linear，以及joint字典上的joint对map字典上的joint。mean与diagonal完整报告，不根据结果临时升主。

## 9. 新Pan数据与统计强度

补充Pan2023登记14人×2次真实会话、两类左右手MI、28通道，含clinic11；实际取数、原schema/label、缓存与许可证据见PAN2023_DATA_NOTES.md和兼容性记录。元数据可读不等于已下载，下载完成也不等于真实物理采集顺序已确认。

原物理run/试次采集顺序若未独立确认，只称发布试次顺序prefix；不能等长伪造run或冒称前瞻采集前20条。两次真实会话可提供跨会话重复证据。跨数据库真实参与者绝无重合不能仅凭注册表无别名认证。

预先锁定方法与κ等参数，不读取Pan query调参或选字典。主要概率终点balanced NLL，BA为关键辅助；若NLL或BA任一可作正证据，则8×2统一16项Holm，不从两个独立8族中择优。参与者为单位，同人会话及三个种子内聚合，不把会话/种子/试次扩成独立n。

既有Stieger内部留出仍属探索性复验。新Pan固定评价增强外部证据，但14人不自动保证功效、显著性或临床有效。具体配对覆盖、缺失和报告规则见STATISTICS.md。

## 10. 可证伪成果与创新边界

有价值结果包括：joint优于同中心与同迹isotropic，删除头配对相关后变差，在新数据两会话复现；或BA不变而NLL/Brier改善；或全部收益由投影中心收缩解释，多点平均不必要。应保留退步、负结果和时延。

若head/linear的BA不变但概率改善，结论是概率质量与分类改善不同。若只在大κ退步，提示局部形状与大位移非线性失配，不应事后删除该条件。

Schur、Laplace、sigma点、GH与logistic对称性均有已有基础。任务特定候选贡献是固定获取预算下的空间—头竞争、球投影、同中心/同边际机制拆解和独立外部复验，不是重新发明贝叶斯/求积，也不预承诺最优。

## 11. 连续运行

沿用hpc_gpu1、一个节点一个Slurm作业三GPU、总36CPU及共享164G内存，每槽12核；Python3.12.3/PyTorch2.5.1+cu121/NumPy2.2.6既有环境，不在作业装依赖，前五轮作业和结果独立保留。

双代训练checkpoint、评价逐会话保存、完成项复用、独立任务失败隔离、GitHub同步独立重试。无累计12小时上限，保留Slurm单次时限前保存和对账续交；学校启动不重跑全量pytest/pilot/全树SHA。实际软件验证、上线地址与学校GPU实跑范围以交付文件为准。
