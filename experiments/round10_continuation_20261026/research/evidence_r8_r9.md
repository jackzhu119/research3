# R8/R9 原始证据复核（R10 研究依据）

本分析以参与者为独立统计单位：先在每个参与者与种子内平均会话，再等权平均三个种子；种子与会话不当作独立参与者。holdout、Pan2023 与 Weibo2014 均已反复用于先前研发，本文件全部推断为探索性，不能作为 R10 独立确认。

## 数据身份与可用范围

- R8及最初linear档案固定提交：`68a8d5c5714ad2a4796bfc21b9203ed29027878d`；最新RBF档案追加固定提交：`a752d00a7b56972d67e1786eb122b218828e3616`，读取于2026-10-26T04:56:31Z。原始字节、分片和整文件 SHA256 均校验；复现脚本在本目录。
- R8：2835/2835 完成；新 `PAPER_DATA_LATEST` 为完整的 v2 论文数据包（33 文件，545,964,970 字节），不再使用旧 AUTO_SUMMARY 的1688任务子集。完整采集包含2835任务收据、3636条件summary与参与者BA/NLL。`reports/full-report.json` 的40项预先声明主要 BA/NLL 比较均可从独立采集的参与者 summary 精确复算，效应量、bootstrap区间与sign-flip p全部匹配；最小Holm p=0.15625。
- R8 主报告更新时间2026-10-25T22:22:42Z，状态COMPLETE；数据包采集时间2026-10-26T02:39:48Z。原始plan字节SHA与报告语义plan哈希不同是编码口径不同；解析后的规范JSON哈希及recipe均完全匹配，见 `PLAN_REPORT_IDENTITY_CHECKS.json`。
- R9最新监控及paper指针为1149/1548（2026-10-26T04:56:31Z），core234/234已完成。**实际论文CSV/report仍停留在120任务（2026-10-26T01:22:21Z）**；所有报告CSV和JSON的原始字节SHA与先前120任务版本完全相同。因此任务计数不能代替统计报告新鲜度；详见`FINAL_FRESHNESS_RECHECK.json`。
- 为补足真实GP证据，最初从Channel B读取180份group manifest，恢复39个GP档案共521,741,539压缩字节；最新已发布group总数增至213，本次再恢复33个raw-RBF档案共429,807,754压缩字节。合计**72个GP任务、951,549,293压缩字节**逐分片及逐源文件SHA验证。每个会话JSON均用原生、SHA锁定的`paper3r7.report.row_metrics`复核query签名、预测、概率与log概率，并复算BA/NLL/Brier；不执行Torch模型。
- 现可完整配对的8种raw-feature GP为linear/RBF各自的MAP、PP、global_mixture、local-information anchored：每种均3种子×3队列×5预算；全部17160个会话×种子×预算单元的query/source/data身份严格匹配基线。**EA预处理GP的完整预测档案仍未取得**，不能把缺失当作失败，也不能声称已评估R9全部核心方法。
- 原R9预先声明家族为6组比较×2个外部队列×BA/NLL，共24项。现在取得20项，缺失4项均为`ea_rbf_anchored−support_EA`。`R9_ORIGINAL_PRIMARY_AVAILABLE.json`保留原家族与缺失列表；对可得20项，未取得假设以p=1保守补齐原24项后Holm均为1。未取得的效应不作任何赋值。另保留明确标为事后探索的linear家族24项（4linear×3队列×BA/NLL），与原家族严格区分。
- 95%CI为10000次参与者配对percentile bootstrap，种子20261025；p为两侧参与者配对符号翻转（N≤20全枚举），假设零假设下参与者差值符号对称；Holm作用于指定家族。Bootstrap CI与sign-flip检验不是同一种方法的反演，边缘情形可能不一致，显著性只按所声明p及家族校正判断。

## R8：复杂几何未证明胜过EA

|方法−基线|队列|N|BA差(pp)|95%CI|p|Holm40|
|---|---|---:|---:|---|---:|---:|
|full_bures − support_EA|pan2023_published_order|14|-0.0119|[-0.2262, 0.1905]|1|1|
|full_bures − support_EA|weibo2014_published_order|10|-0.0754|[-0.4325, 0.2857]|0.726562|1|
|learned_bures_global_pooled − support_EA|pan2023_published_order|14|0.0238|[-0.5000, 0.6429]|0.97229|1|
|learned_bures_global_pooled − support_EA|weibo2014_published_order|10|-0.1667|[-0.7857, 0.4524]|0.671875|1|
|fiber_bounded − support_EA|pan2023_published_order|14|0.2619|[-0.2381, 0.7143]|0.331665|1|
|fiber_bounded − support_EA|weibo2014_published_order|10|0.3889|[-0.0238, 0.7621]|0.113281|1|
|fiber_bounded − fiber_reversed|pan2023_published_order|14|0.5476|[-0.1548, 1.3214]|0.204346|1|
|fiber_bounded − fiber_reversed|weibo2014_published_order|10|1.0278|[0.5278, 1.5794]|0.00390625|0.15625|
|fiber_bounded − fiber_random|pan2023_published_order|14|0.3214|[-0.2857, 0.9167]|0.349121|1|
|fiber_bounded − fiber_random|weibo2014_published_order|10|0.9246|[0.3571, 1.5476]|0.0175781|0.650391|
|fiber_bounded − fiber_fixed|pan2023_published_order|14|0.0119|[-0.4286, 0.4048]|1|1|
|fiber_bounded − fiber_fixed|weibo2014_published_order|10|-0.0119|[-0.4643, 0.4167]|1|1|

虽然fiber_bounded相对反向/随机纤维在Weibo有未校正正向差异，但比较强EA均不显著；相对反向的p=.00390625经原40项家族校正为.15625。相对fiber_fixed也没有增益。这支持继续研究可靠性与可辨识性，而不支持把更多几何自由度本身作为贡献。

## R9：局部信息锚定减轻GP退化，但未证明胜过EA

最新raw-RBF追加证据改变了“只有linear可完整分析”的覆盖判断，但没有出现超过EA的可靠证据。B20、3种子时，原主要方法`rbf_anchored`相对EA如下；CI与p均为参与者配对，BA单位为百分点。

|队列|参与者N|会话数/种子|ΔBA(pp)|95%CI|p|ΔNLL|95%CI|p|
|---|---:|---:|---:|---|---:|---:|---|---:|
|holdout|11|105|−0.5963|[−1.5680,0.3620]|0.262695|+0.000911|[−0.010883,0.013229]|0.885742|
|Pan2023公开顺序|14|28|+0.4643|[−0.3214,1.2857]|0.308228|−0.004473|[−0.012187,0.002908]|0.278442|
|Weibo2014公开顺序|10|10|+0.5000|[−0.5006,1.5714]|0.429688|−0.007102|[−0.026905,0.006400]|0.640625|

原家族内两外部队列的BA/NLL比较经上述保守Holm24校正均为1。Balanced Brier差值：holdout +0.000778（95%CI[−0.003727,0.005448]，p=.75），Pan −0.001846（[−0.004828,0.001057]，p=.253418），Weibo −0.002786（[−0.010332,0.002245]，p=.640625），这些是未校正辅助描述。

相对linear_anchored，RBF_anchored在Pan的BA增加0.7262pp（95%CI[0.0476,1.4405]，p=.083008），NLL降低0.005492（[−0.010491,−0.000888]，p=.047729）；原家族校正均为1。它足以说明R10不应只选linear作为R9代表性强控制。因此R10保留原预先声明、完整取得预测的**raw-RBF anchored**；其核参数来自原meta_validation选择，不能在这些目标队列再调参。EA预处理GP仍可作为额外匹配控制，但其旧实验完整效果目前未知。

RBF_PP和RBF_MAP在两个外部队列的BA完全相同；RBF anchored相对global mixture也没有显著优势。这不支持把后验方差或局部信息权重本身称为已验证的未来预测可靠性保证。

下表是B20，三个种子，完整参与者/会话集合严格匹配（query签名、source身份及data身份相等）。差值均为方法−EA；BA为pp，NLL/Brier为原单位（负差更好）。

|方法|队列|N|BA方法|BA EA|ΔBA(pp)|95%CI|p|事后linear Holm24|ΔNLL|
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
|linear_anchored|holdout|11|61.3420|62.2132|-0.8713|[-1.7318,-0.0815]|0.0712891|1|0.004441|
|linear_anchored|pan2023_published_order|14|63.5595|63.8214|-0.2619|[-1.2738,0.7857]|0.647217|1|0.001019|
|linear_anchored|weibo2014_published_order|10|72.2024|71.9127|0.2897|[-0.9048,1.4762]|0.658203|1|-0.003229|
|linear_global_mixture|holdout|11|60.9799|62.2132|-1.2334|[-2.0446,-0.4411]|0.0126953|0.253906|0.006408|
|linear_global_mixture|pan2023_published_order|14|63.2024|63.8214|-0.6190|[-1.6786,0.6071]|0.337402|1|0.004187|
|linear_global_mixture|weibo2014_published_order|10|72.1230|71.9127|0.2103|[-1.9087,3.1429]|0.929688|1|0.006562|
|linear_map|holdout|11|59.7965|62.2132|-2.4168|[-3.5029,-1.3986]|0.00195312|0.0449219|0.048236|
|linear_map|pan2023_published_order|14|61.8810|63.8214|-1.9405|[-3.4286,-0.3095]|0.0404053|0.7677|0.041218|
|linear_map|weibo2014_published_order|10|70.4087|71.9127|-1.5040|[-4.1468,2.5198]|0.484375|1|0.005099|
|linear_pp|holdout|11|59.7965|62.2132|-2.4168|[-3.5029,-1.3986]|0.00195312|0.0449219|0.015589|
|linear_pp|pan2023_published_order|14|61.8810|63.8214|-1.9405|[-3.4286,-0.3095]|0.0404053|0.7677|0.014275|
|linear_pp|weibo2014_published_order|10|70.4087|71.9127|-1.5040|[-4.1468,2.5198]|0.484375|1|0.018833|

线性MAP/PP的分类决策相同，但预测积分显著改变NLL/Brier，不能把决策相同误解为概率也相同。Holdout线性MAP相对EA的NLL恶化+0.0482362（95%CI[0.0336829,0.0638850]，p=.0009765625；上述事后linear家族Holm=.0234375）。local-information锚定把Holdout的BA降幅从−2.4168pp缩到−.8713pp；Pan从−1.9405pp缩到−.2619pp；Weibo从−1.5040pp变为+.2897pp，但锚定对EA没有显著主指标收益。

raw linear_anchored锚定的是Frozen源预测，而非Support-EA。Holdout其negative_flip_rate约1.0092%，低于EA的8.3404%，但corrected_error_rate也只有1.2507%，低于EA的9.4532%；因此降低负翻转本身不是胜利，必须同时检查净BA与纠错损失。R9已实现EA预处理GP，不得将“把GP锚定到EA”单独宣称R10创新。

## 算法与实现核查

R9 `GPConfig` 使用logistic residual GP；meta_train加权白化参考与长度尺度，meta_validation选择超参数；正特征值子空间Newton/Laplace拟合。posterior moments给出残差均值、方差以及 `I(x)=(k(x,x)−v(x))/k(x,x)`；已存在的局部预测为 `(1−.75 I(x))p_source(x)+.75 I(x)p_GP(x)`。这是现有方法，R10不能把额外乘一个不确定性权重、信息门或同样的概率混合更名为创新。R9真实GP源码SHA `5956354a457c95db4503494c74bb0e1a6e6e2f3cfaf661c5c04ca1d49ee93ad7`；仅有独立记录的CPU cumsum确定性运行修复，未改其统计模型。

## 对R10的约束

1. 不存在已核实的R8主要性能突破。已有R9线性GP证据支持“预测保护减少伤害”，不支持“GP优于EA”。
2. 可辨识信息/后验方差与未来准确性不是一回事；R10机制应区分“可以拟合”与“值得更新”，并提供能被严格消融证伪的数学结构。
3. 强EA与SGD/低容量head、R4方案均必须保留；不能以打败明显退化GP或反向几何控制作为主要成功。
4. 未取到的RBF/EA-GP结果必须列为证据缺口；正式主方法冻结前，应导入随后完整报告重新做同样的配对核查，不能根据新query结果反复调R10。

## 机器可读复现材料

- `R8_PRIMARY_REPRODUCED.json`/`R8_MAIN_COMPARISONS.csv`：原40项主要结果及复算。
- `R8_ADDITIONAL_COMPARISONS.json`：明确标注事后探索的预算曲线比较。
- `R9_LINEAR_GP_COMPARISONS.json`/`R9_B20_AVAILABLE_LINEAR_COMPARISONS.csv`：配对效应、CI、p、家族口径、样本量。
- `R9_ALL_AVAILABLE_GP_COMPARISONS.json`、`R9_B20_FRESH_RBF_COMPARISONS.csv`、`R9_ORIGINAL_PRIMARY_AVAILABLE.json`：最新8种raw-GP及原主要家族20/24覆盖。
- `R9_PARTICIPANT_METRICS.csv`：逐参与者汇总，包括取得的强基线与GP。
- `RESTORED_SOURCE_AUDIT.json`、`R9_GP_ARCHIVE_RESTORE_AUDIT.json`：分片/整文件/档案SHA及每个原始会话文件SHA。
- `R9_FRESH_RBF_MANIFESTS.json`、`R9_FRESH_RBF_RESTORE_AUDIT.json`：最新33个RBF档案的发布Git blob、固定提交、分片与全部源文件SHA。
- `R9_UPLOAD_COVERAGE.json`：完整与不完整种子覆盖。
- `fetch_late.py`、`restore_tables.py`、`restore_archived_gp.py`、`analyze_late.py`：固定提交可复现读取与统计。所有凭据从既有配置读取，未写入证据。

## R1最新状态（只读监控，不能代替完整效应分析）

`paper3-r1-progress`的2026-10-26T04:58:11Z快照显示：已观察主包装动作52/61；这是动态依赖包装动作数，不是全部原研究实验的完成百分比。完成meta_screening38/45、meta_selection3/5、meta_engineering5/5，`all_original_complete`及`mainline_complete`均false，两扩展臂未开始。

三个Slurm作业均RUNNING。slot1在P+PCT维度4/radius.15的外层更新673，并记录21次replay，进度年龄5秒；slot2在P+PCT维度2/radius.05更新915，44次replay，进度年龄4秒。slot0显示RUNNING但没有具体训练进度观测，不能据此声称三个槽位都在有效训练。任务记录保留历史`ExitCode-7/-11/KeyboardInterrupt`，未把无新失败时间戳的缓存类型当作最新异常。原始状态及固定提交/SHA保存在`R1_LATEST_STATUS*.json`。
