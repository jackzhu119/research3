# 第八轮统计与结果记录

第八轮由前七轮设计及已经观察到的第2、3、7轮结果启发。它是一轮可持续扩展的探索性实验，不是新盲测，也不把大量计算任务当作大量独立受试者。统计检查发生在独立报告进程，缺表、GitHub超时和报告错误不停止GPU训练。

## 1. 回传证据决定研究重点

实际结果与版本证据见 [evidence/PRIOR_ROUNDS_AUDIT.md](evidence/PRIOR_ROUNDS_AUDIT.md) 和 [evidence/PRIOR_SUMMARY_PROVENANCE.json](evidence/PRIOR_SUMMARY_PROVENANCE.json)。公开来源为：

- [第2轮状态与结果表](https://github.com/jackzhu119/research3/blob/4706017ef85ff2fcf702e55da9258fd101fdbbf0/monitoring/paper3-followup-v2/status.json)：66/66，报告完整。P平均BA 61.4911%，双门61.4830%，support_EA 62.2132%。双门、软PCT与时间风险主比较均没有校正后正向证据。
- [第3轮状态与结果表](https://github.com/jackzhu119/research3/blob/b56484bff67fd593b70e972a9e4c62d9d44d98aa/monitoring/paper3-round3/status.json)：174/174。tangent_profiled为56.8544%，SGD100为61.9634%；前者低5.1090个百分点，Holm p=.02930。实际预测KL保护恢复约4.66个百分点，但这是相对较弱拟合器的恢复，不能写成超越SGD或EA。
- [第7轮结果表](https://github.com/jackzhu119/research3/blob/f6804b9a72ff118b732643e6b8e05b9f0f6c3cc6/monitoring/paper3-round7/status.json)：Pan B20 support_EA为63.8214%，冻结f0为63.1786%，静态CE/full为60.4524%。十二个BA主比较Holm p均为1，24项BA/NLL统一家族没有p<.05。该历史监控顶部20/273计数与完整报告不一致，较新学校本地回执为273/273；不能用滞后顶部计数判未完成，也不能把报告表当全量模型档案已上传。

因此本轮以协方差输运及其仍未确定的旋转方向为主线，保留冻结源模型、原support_EA、头SGD10/100为参考。无论新方法是否提高均值，都应分析输运距离、标签方向、退步与纠错的机制。已有经典Bures/白化公式本身不是新理论；具体候选贡献由本轮严格控制与真实结果决定。

## 2. 数据与统计单位

| 数据 | 本轮作用 | 人数 / 会话数 | 解释边界 |
|---|---|---|---|
| Stieger既有元训练与验证 | 学习全局/证据参数、选择checkpoint | 40人396会话 / 11人97会话 | 只在这些阶段学习或选择，不读取内部/外部结果选参数 |
| Stieger既有内部留出 | 重复内部评价 | 11人105会话 | 前几轮已反复使用，明确探索性 |
| Pan2023缓存 | 重复外部评价 | 14人28会话 | R6/R7已用，本轮不能称首次外测 |
| Weibo/Yi2014缓存 | 追加发布顺序左右手子任务评价 | 10人10个发布会话 | 原始物理采集顺序/run尚未恢复；前20条LR不等于七任务原会话前20条 |

每种方法均使用固定三个源模型流水线种子20261018/19/20。种子重复降低算法随机性，不能把11人写成33个独立样本。会话在同一个人内相关，不能将105会话、315会话种子行或试次当独立抽样单位。

预算B4/8/12/16/20按既有采集/发布位置计费，无效位置仍占预算；所有B使用由B20固定的同一未来查询集合。不同方法、不同种子需匹配参与者、会话、查询UID和真标签。支持不足两类时产生明确回退结果，保留该会话；不能为某方法删掉难会话。任何真实前缀时间顺序的主张应限于有可靠原始时间/区段元数据的数据集。

## 3. 预先指定的40项主比较家族

正式定义以同版本 `plan.json` 的 `comparisons`、`primary_partitions`、`primary_metrics` 为准。主点为B20、clean；10对方法在Pan及Weibo各比较BA和类别平衡NLL，共 **10×2×2=40** 项，统一Holm校正。

| 候选 − 对照 | 主要问题 |
|---|---|
| full_bures − support_EA | 匹配协方差时的几何对应是否有用 |
| full_metric_bures − full_bures | 改变输运度量的影响 |
| partial_bures − full_bures | 部分输运与全输运 |
| learned_bures_global_pooled − support_EA | 元学习全局输运/收缩强度 |
| learned_bures_evidence_pooled − learned_bures_global_pooled | 支持证据条件化是否优于常数 |
| learned_bures_evidence_class_balanced − learned_bures_evidence_pooled | 类别平衡协方差与标签信息增量 |
| fiber_bounded − support_EA | 同协方差旋转方向是否提供增量 |
| fiber_bounded − fiber_reversed | 标签导出的旋转方向而非仅扰动大小 |
| fiber_bounded − fiber_random | 与等类随机方向的区别 |
| fiber_bounded − fiber_fixed | 支持证据下受限步长与固定步长 |

上述是算法组件识别问题；即使其中某个对照较弱，新算法也必须同时展示与冻结f0、support_EA、头SGD的绝对数值。不能只挑一种弱基线宣称总体先进。此家族是在查看既有轮结果后固定，Holm控制不消除前七轮探索或数据复用偏差。

## 4. 配对估计和方向

对每个参与者i和固定种子s，先在完全匹配的会话集合内求候选与基线差值平均；再对三个种子等权平均，得到每个人一个差值d_i；最后对参与者等权平均。不是按试次数或会话多寡加权。

- BA：`d_i=BA_candidate−BA_baseline`，报告`100×mean(d_i)`，单位**百分点**。
- NLL：`d_i=NLL_baseline−NLL_candidate`，因此正值同样表示改善，单位为类别平衡NLL。
- 正确性、负翻转和纠错按查询类别等权，保留`BA_new−BA_source = corrected_error_rate−negative_flip_rate`的恒等检查。
- 使用逐试次log-probability重算NLL；极端尾概率即使浮点下溢为0也不把损失任意截断为有限上限。历史只存概率的兼容路径才有数值下界，当前worker提供log-probability。

95%区间是10000次**参与者配对bootstrap**百分位区间，统计随机种子固定20261025。它们是各比较的描述性区间，非40项同时置信区间；跨参与者抽样假设和小样本限制仍存在。

双侧p值使用配对均值统计量的符号翻转。人数≤20枚举全部2^n种符号；人数>20使用固定种子的100000次Monte Carlo并采用`(1+extreme)/(1+replicates)`。这需要零假设下参与者差值符号对称，并非临床随机分配保证。均值为正且统一Holm p<.05才标记本家族的正向证据。所有零差值返回p=1。

任何主比较缺种子、缺配对会话、查询身份不一致、预期参与者/会话不足或有无效输出时，**报告保留描述结果并暂缓整个40项家族的p值**。这只影响统计表的解释，其他任务继续。每一项主比较齐备后才计算校正，避免每到几个新会话就重复挑显著结果。训练任务总完成与统计家族完整是两个独立字段；不可混同。

## 5. 大规模探索矩阵如何避免无效膨胀

谱/收缩/混合强度/旋转幅度响应面、短预算、时空压力、错标压力和标签置换是预定探索性数据。它们输出参与者等权指标、支配/退步分布及机制诊断，不产生未经登记的“最优模型”或主检验。大网格摘要不逐格计算数万项bootstrap，避免报告进程占用大量时间；主40项有配对区间和检验，完整预测保留供后续明确分析。

相同输入、相同强度、相同seed的对照用于解释机制；标签置换不改变 pooled covariance 时，冻结源/无标签EA应不变，这是软件/算法性质，不是新增受试者结果。同协方差旋转能否改善分类、是否仅发生在某些支持类结构下，要依据真实未来查询观察。人工压力不等于已经发现真实生理漂移模型。

任务清单有限且可续跑、可另建扩展批次。取消累计12小时限制并不意味着将一个训练任务无限优化、不断改变公平比较预算。更多外层步/种子/网格属于不同预定实验条件，应以新任务身份保留，而非覆盖原结果。

## 6. 可复核且不阻塞训练的报告

`paper3r8.report.build_report(plan)`只使用NumPy和标准库，不导入PyTorch、不接触原始EEG缓存。独立监控器在完成数变化后启动报告进程，最短间隔10分钟；训练worker不等待报告或GitHub。

每个COMPLETE结果回执绑定计划、任务、source/data身份。报告首次读取该任务的逐会话预测，核对summary列出的全部预期会话、查询、概率、log-probability与指标，随后缓存紧凑指标和已读取文件SHA。后续依据结果回执的内容SHA复用缓存，不反复遍历训练数据或全部预测。这里依赖“COMPLETE任务产物不再改写”的worker约定；显式`--refresh`可重新读取已完成预测，不重新训练。

- `reports/full-report.json`：完整本地统计、全部诊断、逐文件哈希、source/data身份及参与者差值。
- `reports/report.json`：供线上查看的汇总；诊断预览最多400行，明确总行数，不伪装全部已显示。
- `reports/primary_comparisons-*.csv`、`evaluation_summaries-*.csv`、`diagnostics-*.csv`：每1000行分片，保留完整聚合表供异步GitHub分类同步，避免单个摘要超过上传大小。
- `reports/report-cache.json`：本地增量缓存，不是科学完成凭证。
- 每份报告保留recipe、计划SHA、输入哈希清单的总SHA、统计seed、精确/Monte Carlo方法和解释范围。

单任务失败、回执不完整或报告读取错误会出现在`issues`/`task_status`，不算完成，也不静默删掉后给出全体成功数字。模型断点、逐试次预测和原始完整结果保留学校本地；GitHub自动摘要上传不是完整checkpoint档案已经异地备份的证明。

