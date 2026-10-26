# R11 实证依据与最新回传核查

本文件是只读证据审计，冻结于 **2026-10-26 北京时间13:43前后**的来源快照。没有运行EEG模型、提交/取消Slurm作业或改变R1–R10结果。结论基于实际JSON、CSV、分片恢复及源码身份；不根据README、任务完成计数或文件体积推断效果。

## 一、这次真正取得了哪些新资料

来源仓库为 `jackzhu119/research3`，精确分支提交见 [PROVENANCE.json](research/evidence/PROVENANCE.json)：

| 来源 | 固定提交 | 实际状态采样时间（北京时间） |
|---|---|---|
| paper3-results | `4ba8af4e4df40d67b821a377bc4eb0a0d504a4c6` | 各指针单独记录，不用提交时间替代实验时间 |
| paper3-r1-progress | `34816f37a8e35f6a34db857b958010743f73d34d` | 13:43:05 |
| paper3-r9-progress | `c0c38d6529b21ecb62549c98165c22f9b97124bd` | 13:42:36 |
| paper3-r10-progress | `b425654dbfc875673e8a708121706c5f239da17a` | 13:43:23 |

R9恢复了原始计划、`report.json`、`full-report.json`、主要比较CSV和评价汇总CSV；每个base64片段与合并整文件均校验原始字节数及SHA256。R10恢复了通道A的实际计划与收据。下载的是有限的紧凑证据，未下载原始EEG或重取庞大预测档案。

## 二、最重要的新结论：R9的新版指针仍引用旧统计报告

最新R9监控和论文指针均显示 **1432/1548**，核心234/234、压力432/432、标签置换54/54已完成；先验敏感性613/702、锚定敏感性81/108。三作业仍运行，槽位0已有当前会话70/105的评价进度。

但实际恢复的论文文件是：

| 文件 | 原始字节数 | 文件内的真实范围 |
|---|---:|---|
| `reports/report.json` | 856,044 | `updated_utc=2026-10-26T01:22:21Z`，只统计120任务 |
| `reports/full-report.json` | 4,347,168 | 同样只统计120任务，并不是1432任务的新GP报告 |
| `reports/primary_comparisons-0000.csv` | 8,464 | 24行全部 `n_participants=0`、`comparison_ready=False`、p/CI为空 |
| `reports/evaluation_summaries-0000.csv` | 358,723 | 138行，仅Frozen/EA/head和固定temperature参考，无GP组 |

完整性检查在 [EVIDENCE_SUMMARY.json](research/evidence/EVIDENCE_SUMMARY.json)，原文件在 [r9/reports](research/evidence/r9/reports)。**文件变大与已上传45份文件均不能证明GP统计已更新。**因此不能用这份报告声称R9出现新突破，也不能把空GP比较解释成GP无效。

额外枚举了该固定提交的246个`archive-groups`目录，相比R10审计时213个目录新增33份group manifest。新元数据覆盖24项EA-linear GP及9项fixed-prior RBF；所有246份可枚举manifest中没有EA-RBF组，EA-linear anchored仅取得seed20261018/19的六个队列任务，seed20261020未归档。因此本次没有为不完整两seed结果下载重型预测包，也没有把缺档当作负结果。此覆盖缺口见 [R9_NEW_GROUP_MANIFESTS.json](research/evidence/R9_NEW_GROUP_MANIFESTS.json)。这些索引与旧`ARCHIVE_PROGRESS`所列157项也不同步，原始manifest、实际目录覆盖和完成计数必须分开记录。

## 三、仍成立的完整原始效应证据

R2–R8已完成的原始证据沿用R10逐文件SHA审计，不冒充本轮新样本。紧凑表格已原字节复制到 [reused](research/evidence/reused)，每个原始路径、字节数、SHA与来源提交见 [REUSED_EVIDENCE.json](research/evidence/REUSED_EVIDENCE.json)。R4–R6完整补传和R8完整2835项论文数据均已在R10核查；R11不重新下载这些庞大档案。

所有下述BA差单位为百分点，均是方法减参考；先在参与者与种子内平均会话，再平均三个种子，参与者才是独立统计单位。三seed与多个会话不能增加参与者N。旧holdout、Pan、Weibo均被反复用于研发，全部属于探索证据。

- **R3保护确实有用，但未超过强基线。**tangent-protected相对tangent-profiled增加4.6589pp，95%CI[2.3965,6.9094]，原家族Holm p=.04395；相对SGD100为−.4501pp，CI[−.8571,−.0701]，事后原始p=.06445。CI与sign-flip检验并非互为反演，不能只据CI宣称显著。
- **R4交叉拟合有局部收益，EA参考行有源码已确认的错误分派。**select-stack相对stack +.5592pp，CI[.3187,.8388]，Holm p=.02148；R4旧Support-EA行实际跑head_newton。与匹配身份的正确EA比较，select-stack −1.5390pp，CI[−2.1961,−.8119]，事后p=.005859，N=11。R11必须调用真实EA，不继承这个错误参考。
- **R5/R6额外几何复杂度没有明确BA主收益。**R6的微小NLL优势准确属于Pan B20 `map:uq_joint−map:uq_map`（两者均MAP，改变的是训练几何），不是目标joint posterior推断获益；真实joint与MAP目标推断差没有显著性。不能以模糊“joint更好”支持新结构。
- **R7显示明确的容量与未来风险冲突。**B20 holdout的11名参与者、315会话seed行中，有效支持样本均值15.5301，full head参数65；支持集head可吸收全部一阶空间方向，数据侧独立空间信息秩为0。head rank0/4/8/full的支持损失约.6721/.5111/.2833/.1214，而未来BA约61.3097/59.1832/56.9915/56.8918%，NLL约.6578/.8137/.9681/.9854。rank0仍含bias，不是无适配。单纯训练拟合改善不足以支持未来预测修正。
- **R8全2835任务、40项原主要BA/NLL比较已完整复算。**最小Holm p=.15625，没有主要家族突破。fiber-bounded−EA在Pan为+.2619pp，CI[−.2381,.7143]，N14，原p=.3317；Weibo+.3889pp，CI[−.0238,.7621]，N10，原p=.1133，两者Holm均1。

R9此前已从完整预测档案验证72个GP任务、951,549,293压缩字节，而不是依靠当前滞后的自动报告。raw-RBF anchored相对正确EA的B20配对结果如下（[完整表](research/evidence/reused/R9_RAW_RBF_COMPARISONS.csv)）：

| 队列 | N参与者 | BA差pp | 95%CI | 双侧原始p |
|---|---:|---:|---|---:|
| 内部holdout | 11 | −.5963 | [−1.5680,.3620] | .262695 |
| Pan published-order | 14 | +.4643 | [−.3214,1.2857] | .308228 |
| Weibo published-order | 10 | +.5000 | [−.5006,1.5714] | .429688 |

raw-linear anchored的对应BA差为−.8713/−.2619/+.2897pp，均没有稳定超过EA的证据。RBF相对linear较有潜力，但不是已证明突破。原R9 24项主要BA/NLL家族在先前已恢复档案中取得20项，4项EA-RBF仍缺；保存的可用家族表没有将缺项静默删去或把家族改成更容易显著的子集。**不能把“raw-GP暂无显著优势”外推成“全部EA+GP已证实无效”。**

Pan/Weibo的published-order不能直接称为真实物理未来会话；其采集顺序证据边界继续沿用原协议。原生旧NLL和R10/R11统一概率下限后的NLL也不能直接混合检验，必要时对相同封存预测按同一规则重算。

## 四、R1与R10当前不能提供什么效果证据

R1本次40/45元筛选、3/5元选择、5/5工程验证；已观察包装动作54/61，`mainline_complete=false`，扩展未开始。槽位0外层更新760、槽位2更新492，采样时各4.6秒内有推进。槽位1原P+PCT/d8/r.05在100更新后显示`ExitCode-11`并回到等待，已换到另一个配置；具体失败原因要靠对应attempt日志，不能仅凭旧错误字段推断。本审计没有为新研究修改或停止旧作业。R1的部分结果不是已完成主线证据，也不作为R11目标集调参依据。

R10实际通道A只有48687字节的计划，SHA验证后确认**4作业，每作业1GPU/16CPU/68G，15任务机制阶段**，参考策略`reuse_r9_ea`。状态0/15、blocked0、全部NEW/attempt0、无当前错误和已分配任务。这个快照不能区分仍在队列中等待的作业与尚未到任务领取步骤；不能从job_id为空认定控制器失败，更不能当作算法已训练或已有性能提升。队列与运行入口由平台子任务独立核查。

因此 **R10尚无真实EEG效果，不能拿R10作为R11已验证成功的前提，也不能声称R11已经纠正R10实证失败**。

## 五、对R11科研判断的约束

1. 没有新完整结果推翻“正确Support-EA仍是必须超越的强基线”这一判断；GP的缺档方法仍须明确标未知。
2. R3保护、R4有限收益、R7容量冲突支持研究“哪些新增函数修正有独立且可靠的信息”，不支持单凭更多参数或更低支持损失宣称创新。
3. R11的任何新机制必须与原生正确EA、实际空间SGD、最强可合法复用R9代表及其自身matched消融比较，不能只打败崩溃或弱化旧方法。
4. 数据选择与门限选择应只用meta_train/meta_validation；旧holdout/Pan/Weibo用于反复设计之后仍是探索性。新确认需要此前未参与设计的新数据，现有快照没有这样的已完成结果。
5. 本次没有新的R9/R10显著突破可支持提前性能声明。R11应按可复现机制和小规模开发证据决定是否推进，不借滞后报告的空值、任务数增长或软件测试通过充当算法效果。

该审计保留真实缺口。若后续回传了更新统计或完整EA-GP预测，应另加有日期、固定提交的补充审计，不覆盖这里的历史边界，也不根据新目标结果反复改写已冻结的主要比较。
