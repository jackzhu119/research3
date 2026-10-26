# R10 前置证据复核：R2–R7 原始结果、统计与可用性

核查冻结点：`paper3-results@68a8d5c5714ad2a4796bfc21b9203ed29027878d`，2026-10-26 03:59 UTC（北京时间11:59）。本页来自实际 CSV、JSON、manifest、任务回执及源代码；未使用仓库 README 的效应摘要代替结果。

## 1. 数据是否已传到仓库

| 轮次 | 最新论文数据 | 完成任务证据 | 本次读取范围 |
|---|---|---|---|
| R2 | v1 9个文件，684,619字节 | 原报告 COMPLETE 66/66；全归档 LATEST 存在 | 全部CSV、plan及446,022字节report.json |
| R3 | v1 14个文件，25,311,991字节 | 完整归档174/174；原论文表完整 | 全部CSV、plan；50项统计重算中的R3部分读取参与者配对向量 |
| R4 | **新v2** 22个文件，51,559,447字节，collection COMPLETE | 原report COMPLETE 204/204 | 完整31,280,754字节report、全部CSV、未清洗condition-summaries/task-results、plan |
| R5 | **新v2** 22个文件，29,937,940字节，collection COMPLETE | 原report COMPLETE 105/105 | 完整18,293,291字节report、全部CSV、未清洗条件汇总和任务回执 |
| R6 | **新v2** 23个文件，67,312,160字节，collection COMPLETE | 原report COMPLETE 198/198 | 完整45,237,336字节report、全部CSV、未清洗条件汇总和任务回执 |
| R7 | v1 15个文件，43,818,712字节；另有150,424,829字节聚合report的144片原字节补传 | 全归档273/273 | 全部CSV、plan，含32MB诊断表；本次未重复下载150MB聚合report，因为所需配对向量及机制量已在CSV |

本次验证104个源文件、298个分片，合计原声明198,811,487字节；101个文件恢复为**原始字节**，3个旧版大CSV按manifest恢复为**完全相同的字段值与行序**，仅可能改变CSV格式字节。每个分片都检查SHA256。R4–R6聚合report已在本地无损gzip后检查解压字节SHA；为节约磁盘删除的仅为本次新建的重复下载分片缓存。

R4–R6目前论文数据完整上传，不应再沿用“只有自动清洗摘要、缺大报告”的旧判断。相反，R4–R6的`LATEST.json`全归档指针仍未见；论文数据完成与全归档完成是不同通道。

可复核文件：`evidence/early/POINTER_AUDIT.json`、`DATA_RESTORATION_AUDIT.json`、`LOCAL_COMPRESSION.json`。源码指纹另见`CODE_PROVENANCE.json`。

## 2. 统计单位与重复使用范围

- 内部holdout：11名Stieger参与者，105个固定未来会话，3个冻结pipeline seed，通常315条会话×seed配对记录。独立统计单位是**11名参与者**，不是315条记录或33个人。
- Pan：14名参与者，28个published-order会话，3个seed，共84条会话×seed记录。该顺序是原MAT试次索引，不能等同已验证的物理采集时间。
- 对每名参与者先在同一seed内平均配对会话，再平均3个seed，最后参与者等权。
- 95%CI为20,000次参与者percentile bootstrap，seed20261018，针对单条比较、条件于现有划分与pipeline；并非同时置信区间。
- 主比较p值为双侧精确配对sign-flip（这些队列N≤18），依赖零假设下参与者差值的符号对称性。Holm控制原来固定的比较家族。
- R2/R3/R4/R5家族大小分别10/10/12/8。R6为8个BA+8个NLL，共16；R7为12个BA+12个NLL，共24。不能只拿其中BA表再调整，也不能将多种seed或会话当作独立N。
- 本次重新计算了R3–R7的**50条主BA比较**：参与者向量、效应、CI、精确p和对应历史Holm均与原表一致。R2轻量报告未导出参与者差值向量，故其10条比较的效应/CI/p读取自实际原CSV/JSON，Holm可复算，但本次未为它重新下载826MB全预测归档来重算参与者向量。
- 从R10立场看，这些内部holdout及Pan结果均已被设计者看到，均为探索性证据；不能将其重新命名为独立确认。R6当时的首次Pan评价标签不改变它现已被R10设计使用的事实。

## 3. 历史预声明主要比较

以下效应均为候选减基线的BA百分点；N=独立参与者数。每行均B20、clean、3seed。

| 原比较 | 队列/N | ΔBA pp [95%CI] | 原始p / 原家族Holm p | 判断 |
|---|---|---|---|---|
| R2 dual − P | holdout/11 | −0.00806 [−0.02074,0] | .500 / 1.000 | 没有门控成分带来的主指标优势 |
| R3 profiled − SGD10 | holdout/11 | −4.63668 [−6.85425,−2.36855] | .006836 / .047852 | 解析强拟合显著劣化 |
| R3 profiled − SGD100 | holdout/11 | −5.10898 [−7.25184,−2.95169] | .002930 / .029297 | 相对更强SGD同样劣化 |
| R3 protected − profiled | holdout/11 | +4.65887 [2.39647,6.90942] | .004883 / .043945 | 保护有效地挽救退化，非强基线优势证明 |
| R3 information-protected − information | holdout/11 | +4.64302 [2.40773,6.87505] | .004883 / .043945 | 同上 |
| R4 xfit-select − profiled | holdout/11 | +1.74577 [0.99299,2.55789] | .000977 / .011719 | 支持内样本外选择优于弱拟合器 |
| R4 select-stack − xfit-stack | holdout/11 | +0.55921 [0.31874,0.83881] | .001953 / .021484 | 某些选择结构有增益 |
| R4 xfit-stack − uniform-stack | holdout/11 | +2.28056 [0.93171,3.70325] | .011719 / .093750 | 主家族校正后不显著 |
| R5 left-rotation − single | holdout/11 | +0.02573 [0.00470,0.04667] | .062500 / .500000 | 极小变化，不能称显著 |
| R5 symmetric − single | holdout/11 | 0 [0,0] | 1 / 1 | 当前预测没有收益 |
| R6 `map:uq_joint−map:uq_map`（均为目标MAP） | Pan/14 | −0.03571 [−0.09524,0.02381] | .453125 / 1 | joint训练几何无BA收益 |
| R6 `joint:uq_joint−map:uq_joint`（目标推断比较） | Pan/14 | −0.01190 [−0.05952,0.03571] | 1 / 1 | 目标joint推断无BA收益 |
| R7 时空 − 静态，BA | Pan/14 | −0.11905 [−0.47619,0.23810] | .586670 / 1 | 没有BA收益 |

R5中个别bootstrap CI不跨零而精确p≥.05并不构成自相矛盾：这是两种有限样本推断程序，不可用CI端点替换预声明的精确p与Holm规则。

**R6的比较对象必须写全。** Pan B20、14名参与者下，`map:uq_joint`（joint训练几何、目标仍做MAP校准）相对`map:uq_map`的balanced NLL从0.870367393降至0.869336720，改善量0.001030673，95%CI[0.000743873,0.001333462]，原始p=.00024414、原16项家族Holm=.00341797；对应BA差为−0.035714pp、p=.453125、Holm1。这里的改善是训练几何的比较，不能称为目标joint posterior优于MAP。真正目标推断的`joint:uq_joint−map:uq_joint`，NLL仅从0.869336720降至0.869272302，改善0.000064418，CI[−0.000138584,0.000347287]，p=.809814、Holm1；BA差−0.011905pp、p=1，均无显著优势。原R6 NLL表的改善量方向为baseline减candidate；R10新协议的NLL差统一candidate减baseline，符号不能直接混用。R6的这些NLL仍明显差于同队列EA的0.63038335。R7全24项原主要比较最小Holm=.134765625。

完整60条主BA比较与40条R6/R7多指标原表（其中BA条目重复出现在多指标表）保存在`EFFECT_AUDIT.json`与`PRIMARY_EFFECTS.csv`，不可只挑上表正向行。

## 4. 必须排除的R4基线标签错误

本次发现不能忽略的证据质量问题：R4的`support_EA`报告B20 holdout BA=56.853139%、NLL=0.986080735，与其`head_newton`输出完全一致；R5/R6/R7的正确Support-EA为BA=62.213243%、NLL≈0.646613403。

原R4 worker `_configuration`先执行`values.setdefault("variant", task.variant)`，而R4 reference任务的`config.variant`已是`head_newton`，因此返回给评价器的variant也是`head_newton`；并未进入`fit_reference(..., "support_EA")`。后续修复的内存overlay明确加入`if task.variant == "support_EA": variant="support_EA"`，但R4这三个已完成回执不带该修复的`runtime_corrections`，完成结果被复用而没有重跑。

确认依据是：

1. 实际R4 task-results JSONL中3个seed均`task.variant=support_EA`、`task.config.variant=head_newton`，且无`runtime_corrections`。
2. 实际R4 condition summary的参与者及会话BA与head_newton一致。
3. R5正确EA回执明确携带`support_EA_reference_dispatch`修复、执行源SHA与hook SHA。
4. R4与R5相应seed的source checkpoint、source receipt、reference receipt、input contract SHA全部相同；固定query signatures也相同。

**因此R4中标作Support-EA的结果应标为不合格基线，不可用于声称新算法超过EA。** 历史文件保留，不应改写它的标签来伪装原先执行的是正确EA。R10须实际重新验证EA分派，或仅复用来源、任务执行指纹和query身份都已验证的R5/6/7正确EA输出。此发现不否定R4对weak baseline的预声明配对结果，但改变了对强基线突破的解释。

## 5. 与真实强基线的补充配对检查

以下检查是在阅读旧结果之后为R10动机作出的**事后探索性审计**，不是原轮次预声明优势检验。CI为单比较CI，所列p未作新的确认性多重校正，不得与原主家族Holm混用。

| 候选 − 强基线 | 队列/N | ΔBA pp [95%CI] | 事后原始p | 负差参与者比例 |
|---|---|---|---|---|
| R4 select-stack − 正确EA（R5重用、同源同query） | holdout/11 | −1.53900 [−2.19606,−0.81191] | .005859 | 10/11 |
| R4 select-stack − SGD10 | holdout/11 | −0.81683 [−1.79624,−0.06028] | .099609 | 8/11 |
| R3 protected − SGD100 | holdout/11 | −0.45011 [−0.85710,−0.07011] | .064453 | 7/11 |
| R3型information-protected（R4复评）− 正确EA | holdout/11 | −0.69803 [−1.52635,0.09596] | .141602 | 8/11 |
| R5 single − EA | holdout/11 | −5.32147 [−7.45924,−3.27085] | .000977 | 11/11 |
| R6 `map:uq_joint`（joint训练几何+MAP校准）−EA | holdout/11 | −5.28773 [−7.39709,−3.27209] | .000977 | 11/11 |
| R6 `map:uq_joint`（joint训练几何+MAP校准）−EA | Pan/14 | −3.36905 [−5.57143,−1.27381] | .011475 | 11/14 |
| R7 full-head静态 − EA | holdout/11 | −5.32147 [−7.45924,−3.27085] | .000977 | 11/11 |
| R7 rank0头限制 − EA | holdout/11 | −0.90350 [−1.74596,−0.10266] | .057617 | 9/11 |

跨R4/R5的比较来自未清洗condition summaries的实际参与者指标，逐seed核对同一source identity、data identity和全部105个query signatures后配对。R7的rank0与EA使用同一坐标下原CSV参与者差值向量的精确代数合成。R3 protected与SGD100同样使用两条匹配的参与者差值向量合成；未将两条不配对的总体均值相减来制造CI。

上述现象表明：减少伤害与打败EA是两个不同的实验问题。后续方法必须把EA、SGD100和低容量头纳入主比较；仅打败全65维Newton头无法支撑算法优势论文。

R10的`head_rank4`是当前支持集无标签特征上的PCA容量基线，不是meta_train固定子空间。

## 6. R7容量与可辨识性机制

内部holdout、B20、3seed、11人/315条会话×seed：

| 头特征限制 | 头参数数（含bias） | 支持BCE | 未来BA% | 未来balanced NLL | data-only空间信息秩 | 头吸收空间比例 |
|---|---:|---:|---:|---:|---:|---:|
| rank0 | 1 | .672126 | 61.309739 | .657763 | 4.000000 | .343623 |
| rank4 | 5 | .511141 | 59.183174 | .813710 | 3.906730 | .625039 |
| rank8 | 8.90673平均 | .283337 | 56.991491 | .968129 | 3.579693 | .818542 |
| full64 | 65 | .121421 | 56.891774 | .985351 | 0.000000 | 1.000000 |

同一批支持集有效样本均值为15.530106，中心化支持特征秩14.530106。rank8实际有效rank均值7.90673，故参数数均值不是固定9。rank0仍允许一个bias，因此不能把它写成“完全没有头参数”。

这直接支持“高维头在小支持集上吞没空间Jacobian的信息、支持损失下降却未来风险恶化”的研究动机，但不能推出任何未来风险定理。full head的data-only Schur残差秩为0，首先说明**这些支持位置上的函数变化可被头参数张成**；不等于所有未来query位置都无独立空间效应。R7已经计算Schur残差秩、头竞争空间与吸收比例，因此仅重新命名这个残差投影也不构成R10创新。

## 7. 对R10的可证伪要求

- 新方法应同时对“支持集可辨识”与“在未用于拟合的位置可预测”负责；只用ridge抬高Schur特征值不应冒充额外的真实数据秩。
- 将低容量头、正确EA和SGD100作为强控制；验证每个新增数学机制是否改变未来损失与负迁移，而非只改变训练目标。
- R4已有blocked/LOO/stratified支持内交叉拟合、one-SE、收缩、stacking与参数选择；这些本身不能在R10中声称原创。
- 固定的B4/8/12/16/20未来query已在R4的93、R5的54、R6的144个多预算task/partition/seed分组上核对query签名不变。对跨轮缓存复用还必须同时匹配source、算法执行指纹与输入身份；仅名为同一方法或同一个plan SHA不够。
- 真正算法选择只应在元训练/验证范围进行。当前内部holdout、Pan、Weibo不能再充当全新独立确认集。

## 8. 复算入口与已知边界

依次运行`fetch_pinned.py`、`fetch_manifests.py`、`fetch_data.py`、`audit_statistics.py`可重做本次冻结证据；`fetch_code.py`单独保存各轮代码分支的不可变头与源文件SHA。所有脚本只读GitHub，不操作学校作业。远端大源文件通过manifest分片读取，没有截断行或删除参与者。

主要产物：

- `EFFECT_AUDIT.json`：60条主BA比较、R6/R7完整原多指标行、9条事后强基线检查、R7机制表及固定query检查。
- `PRIMARY_EFFECTS.csv` / `SUPPLEMENTAL_EFFECTS.csv`：紧凑可读表。
- `POINTER_AUDIT.json` / `DATA_RESTORATION_AUDIT.json` / `CODE_PROVENANCE.json`：冻结提交、git blob、源SHA与范围。

本次未读取R7完整150MB report的每个冗余字段，也未下载R2完整预测归档；对所需现象已经有原表、参与者向量、诊断量和合法来源身份支持，以上边界不以“完整全档已下载”冒充。


### 与R10新协议的数值评价边界

R10对全部方法统一采用`1e−12`概率floor计算balanced NLL，并保留该评价版本身份。历史轮次若使用未截断log-probability/BCE-with-logits，其NLL标量不能与R10截断后的NLL直接混合；需要从身份匹配的原预测/log-probability按共同规则重算，保留原结果，不覆盖历史文件。BA配对仍须同时满足source、支持/query UID与聚合身份。

R10后续提交资源固定为4个并行作业，每作业1张GPU、16核CPU、68G内存，`hpc_gpu`队列；本轮设计交付未自动分配这些资源。
