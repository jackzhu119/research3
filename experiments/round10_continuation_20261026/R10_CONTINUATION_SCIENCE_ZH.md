# R10 探索性续跑：选型规则修订登记

登记版本：`paired-ea-gain-one-se-exploratory-amendment-v1`。本文件说明原机制结果已产生后作出的修订，不替代原R10方案，不称作原结果产生前的预注册。新交付使用独立代码包、计划、输出目录和修订文件；原15任务及原规则的STOP重算结论必须原样保留。

## 1. 已知结果与修订范围

原机制阶段在固定seed 20261018、8名meta_validation参与者、预算4/8/20下完成15任务：1个reference、9个robust候选、3个matched MAP、Frozen和Support-EA。固定GitHub结果HEAD为 `4307c8e354f3ad46865bcb7473a96ce017037f0a`；08:00:40Z的原report含15个sealed任务、14个指标任务、336个session/budget记录。原report SHA256为 `058882cdec21bdf7f5ddc4ee1d7cd58d3dec10b30046e69b1ecaa00fafbec859`，recipe为 `d9bbaad2f1af0159da345b4c1e73a90bf0aa05268f9466bc01150a5e72cf0574`。

原规则用最优候选绝对BA在4名selection参与者之间的标准误构建one-SE带。由于参与者基线难度差异较大，其SE约6.0144个百分点，9候选全部入带，再按较大c、较小amplitude排序，选中B20完全复制EA的 `a0.3-c0.5`。根据原规则只读重算为STOP：selection BA增益0，非平凡更新比例0，分别未达到+0.25个百分点和1%的门槛。若没有原学校decision文件，此结论明确标为“原规则重算”，不能伪装成学校当时已经运行并保存的STOP回执。

本修订纠正选型中不确定性所对应的量，并取消把小样本性能结果作为整个探索实验的中断条件。它不修改R10的RKHS/logistic目标、query-null completion、残差修正dual、锚相对minimax预测、EA定义、共同概率floor、query划分或9点候选网格。已看过开发结果是选择新规则的背景，因此后续结果仍为探索性证据。

## 2. 修订后的唯一选型规则

保持原固定hash选出的8人及4人selection/4人check划分。只允许selection四人的B20数据决定配置。每个方法先在同一participant/seed内对session等权平均，再对seed等权平均；参与者等权。该机制数据只有一个seed，每人一个预选session。

对候选j与参与者i定义配对增益

\[
\Delta_{ij}=BA_{ij}-BA_{i,EA},\quad
\bar\Delta_j=\frac14\sum_{i\in S}\Delta_{ij},\quad
SE_j=\frac{SD_{ddof=1}(\Delta_{ij}:i\in S)}{\sqrt4}.
\]

令j*为平均配对增益最大的候选，完全相同的平均增益按candidate ID字典序决定j*。one-SE候选集合为

\[
\mathcal C=\{j:\bar\Delta_j\ge\bar\Delta_{j^*}-SE_{j^*}\}.
\]

实现只使用固定的1e−12数值比较容差。在集合内保持原保守排序：evidence fraction c更大优先，amplitude更小优先，最后candidate ID字典序。全部9个数据完整且合法的原robust候选都参与，不新增正增益、NLL或非零更新筛选，不扩大网格，也不改为直接硬选最高均值候选。

这条规则具有具体的不变性：若给同一参与者所有候选及EA的BA同时加上一个共同难度偏移h_i，则Δ_ij、其SE、候选集合与选择均不变。原绝对BA标准误不具备这一性质。该不变性并不证明任何未见参与者的增益。

one-SE是开发启发式，不是95%置信区间、候选间显著性检验或风险上界。n=4时标准误本身不稳定。该规则仍可能选择零更新或负增益候选；这种情况下明确记录`NO_POSITIVE_DEVELOPMENT_GAIN`，保持完整探索评估，不再据此把全部任务中断。

## 3. check与旧门槛只作诊断

check四人的BA/NLL不进入候选排序、one-SE带或tie-break，不用于失败后改选其他配置。新选择固定后报告check四人的配对BA增益和balanced NLL差。修订前check结果已经可见，故check也不能被重新描述为未接触的验证或独立确认。

旧selection增益、check BA/NLL、非平凡更新、全局数值fallback、证书计数及预测成本阈值全部完整保存为诊断。即使诊断未达旧门槛，也不阻止后续正式形状的探索性配对任务。身份错配、损坏或缺失的选择输入不能伪装成完整配对数据；选择器返回`BLOCKED_INVALID_DEVELOPMENT_DATA`，不能偷偷删掉缺失参与者或取交集。

取消性能中断不等于接受错误实现。运行时数值错误仍按算法既有的可审计fallback处理；真实证书反例、数据身份错误或代码与方法语义不一致须标明技术错误。完整选择输入之外的历史基线缺失只影响其对应方法/seed及比较，不阻断其他可运行方法；缺失比较仍保留在固定家族中。

## 4. 当前开发数据的可复核结果

下表用于记录修订在已有selection数据上的结果，不是新增EEG实验，不是独立验证。

| 量 | 值 |
|---|---:|
| 配对均值最大的候选 | a0.3-c0.01 |
| 最优候选selection平均增益 | +1.087520个百分点 |
| 最优候选配对增益SE | 0.641742个百分点 |
| one-SE增益下界 | +0.445778个百分点 |
| 入带候选 | a0.3-c0.01、a0.3-c0.1 |
| 最终选择 | a0.3-c0.1 |
| 所选候选selection增益 | +0.860247个百分点 |
| 所选候选selection配对SE | 0.589870个百分点 |

所选候选check BA增益为+0.277778个百分点，balanced NLL差为−0.000606064；这些量只作诊断。原规则选择与STOP重算同时保存在`legacy_decision_reconstruction`，并保留原始source plan、report和每个完成结果的SHA来源。

## 5. 后续165任务的范围与解释

后续仍保留18方法×3seed×3已用cohort，以及3个reference任务，共165任务；每个evaluation任务包含原固定5预算4/8/12/16/20。三个cohort为内部holdout、Pan2023与Weibo2014，均已参与既往设计或比较，全部标为探索性，不把新的代码版本或新目录称为新的独立数据。

完整方法为Frozen、Support-EA、head_bias、head_rank4、head_full、SGD10、SGD100、R3 protected、R4 cross-fit stack、R8 fiber bounded、R9 raw RBF、R9 EA linear，以及主profile、matched MAP、no completion、Frozen anchor、quadratic与variance gate。原生强基线的已冻结训练产物/配置和本轮目标支持适配协议均保留；不根据续跑目标结果选择更弱基线。

主要家族保持B20 BA的四类比较：profile对EA、SGD100、R9 EA anchored、matched MAP，跨3cohort共12项Holm校正。预设四个机制消融的BA/NLL形成8×cohort数的独立次要家族。历史产物缺失的比较不能从家族里删除或当作成功；缺少完整配对的比较明确NOT_READY。重复seed不增加独立参与者N，session先聚合到participant。

原15任务已经完成，不为改变规则而抹除或反复重训这些相同任务。续跑通过独立amendment记录重新解释其选型信息，并在新正式探索任务中执行所选配置。若实现因缓存/兼容优化变化，必须绑定新执行代码SHA与原结果来源SHA，不能声称原机制阶段使用过后来修改的代码。

## 6. 实现与验证边界

`paper3r10.amendment_selection.amended_development_decision`是纯stdlib函数，接收经SHA验证的原plan对象、session_metrics字典、coverage与source_receipts，返回新决策对象；它不加载旧plan、不初始化NumPy/Torch、不写原结果或freeze。平台将该对象写入新amendment文件并绑定后续计划。原`selection.py`的development_decision/freeze接口保留，用于原规则复现。

针对性CPU回归包括：参与者共同难度偏移不变性；check结果任意合法变化不改选型或探索继续资格；带内较低均值的保守候选可以胜出；全零/负增益也诚实继续；行顺序和query数不改变参与者等权选择；缺对/重复/身份错配/非有限指标不能被静默接受；纯stdlib导入与原输入对象不变。兼容测试在发布前执行，完整原始EEG重哈希或反复重跑测试不作为每个任务的启动门槛。

本修订不声明future-risk保证、真实EEG总体提升、论文价值或独立确认成功。旧结果保留，新结果完整报告，正、负与无效结果均保持可检查。
