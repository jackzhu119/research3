# 新数据方案的实施边界

依据用户附加的 `Paper3_Research_Data_Plan.docx`（45,711 bytes；SHA256 `74383bb55cd1287ad806e5d83812e95bdb4bf0bccf7a59292b91204f230a073a`）。该文档是研究建议，所列报告规模不作为本次真实验收数量，也不赋予外测解封权限。

主线保持 v2.1 的四个 source、11 通道信号合同及 Stieger 40/11/11 人员划分。原 52 文件协议、原 117 份软件清单与已有固定包均保留；`legacy_baseline_v1` 不是本研究的新算法。

## 数据分工

| 数据 | 本轮用途 | 必须完成的验收 | 科研限制 |
|---|---|---|---|
| 原 source4 | 主线 f0、Rref | 505 原档、224 provider ID、28,830 有效 trial 的既有真实缓存及回读证明 | provider ID 不等于已证实跨库自然人独立 |
| Stieger2021 | 固定 meta-train / meta-validation / internal-holdout | 全 598 会话原档 SHA/MD5、原生时钟、波形 QC、缓存及 R2 全 GET；正式完成后独立复核 | 缓存处理不等于进行内部检验；真实 B20 按原采集顺序计费 |
| Yang2025 v5 / 2C | `BASE4_PLUS_YANG`，独立 source 增强 | 153 data/evt pair，逐 ZIP member CRC、原档 SHA、时钟、原事件 UID、冻结源窗、缓存和 R2 证明 | 非主线样本；不能凭文件头宣称严格跨日 meta；未知 SI 仅接受已证实共同正增益并在模型前 joint RMS |
| Dreyer2023 / NEMAR nm000250 v1.0.5 sourcedata | `BASE4_PLUS_DREYER`，另一个独立 source 增强 | 87 provider ID / 520 EDF，官方发布 SHA、全量原生 TAL 资格及真实完整 MI trial、单位、V5 缓存、128份新旧实际数值比较及 R2 证明逐项验收 | 论文1已有暴露；不能称全新盲法外测；不冒充原 GDF 的逐字节副本；原生资格数不等于波形 QC 后缓存 N |
| Ma2022 公开 v1 | 独立 retained-session 三通道开发资料 | 25 人 × 5 session 的 125 EDF/events pair，官方 MD5、计算 SHA、每个真实保留 trial、原生六列表与电极/单位、缓存及回读 | 原删除 trial 的采集序号未知，不能充当原始付费 B20；不进入 source4 或主11通道 |
| 已登记外测、Leeuwis、MILimbEEG、HS、Wang | 保留其既定外测或储备用途 | 当前仅允许协议规定的元数据审计 | W7 前不得下载/解封外测 query、调参或转作训练 |
| Zuo2025、NETBCI2026 | 元数据与任务适用性记录 | 保留真实任务证据 | 下肢 LR、右手/静息不能改名后并入左右手 MI |

Yang 与 Dreyer 是两条独立 source 支路，各自仅由原 source4 加一个新增 source 组成五 source，不组成默认六库训练池。所有比较方法必须共享同支路 f0、该支路的 Rref、同一校准/未来 query 和选择预算。两支路继承同 seed 的真实 baseline4 LODO 所选 final epoch，固定每 epoch 的更新与 batch 数；这检验来源组成变化，不能单独解释为人数增加的因果效应。

三通道 source 数值缓存由相同四库、相同原 trial 产生，不增加独立 N。CAR3 的独立 f0 入口与独立 GPU 恢复 pilot 不构成三通道 Rref/G/全部14方法已完成的证明；这些依赖继续逐项验收。Ma 已实际核验全部125个 session、11,988个唯一保留 trial（左右类5,983/6,005），原始采集序号缺失这一限制仍成立。其分组及 retained-session 实验方案未冻结前，不允许该资料进入 fit。

## CPU 数据验收与版本修复

每一原档都须完成合法来源/固定版本、长度、官方 MD5 或发布 SHA 及本次 SHA、原生事件/单位/通道/采集顺序、逐 trial 缓存绑定及 R2 完整 GET/SHA。原档和数值缓存双证据完成后，只清理本流水线明确拥有的暂存原档；原 pilot、旧缓存与失败证据保留。

完整处理已暴露真实格式差异，按原始事实修复，不能靠补造 trial 绕过。Dreyer 全520份 EDF 的原生 TAL 已完成资格审查：497份保留旧规则下的40个完整 MI trial；22份虽无 `32770` 停止标记，仍有40个完整 MI trial，仅按各自原档 SHA、原文件名及实际原生 TAL 尾型白名单准入；唯一的 `sub-40_run-3` 只保留32个真实完整 MI trial（左14、右18），末尾原索引193/194的 `768/786` 前缀没有 MI cue 或配对 `800`，不赋标签或 MI UID，不计入 trial 或波形 QC 排除项，不补齐或平衡。官方 `1010` 会话结束与 `33281` 训练控制可出现在 acquisition 和 online 文件的最后一个完整 MI trial 之后；不能凭文件名阶段拒绝合法记录。零宽 `33284` EOF 原标记保留，只有实际发现的 `sub-85_run-4`、`sub-85_run-6` 允许 EOF 恰在物理文件终点，且分别绑定原档 SHA 和原文件名；其他旧停止标记/EOF规则保持不变。

Dreyer 的全量原生完整 MI 资格为20,792个（左10,394、右10,398），UID全部唯一；这是波形 QC 前的资格数，不能作为最终缓存 N 或已训练样本数。392份新增原档已实际进行 MNE `preload=False` header/TAL 交叉核验，标注类别、顺序、时钟和宽度均一致，实测时钟误差0，未调用 `get_data` 或请求返回 EEG 波形数组。V5 原生适配器 SHA256 为 `5c31e969eeabd17634864f6783d8473a0a79f1888faf274d758cd9c5ba4cef77`；cue/配对 `800`、1.5秒真实上下文、预处理数学和 UID 公式保持不变。V5 全量波形 QC 缓存、128份新旧实际数值比较及最终缓存/R2闭账验收仍未完成，不能以原生资格、MNE 元数据核验或软件测试宣称训练已就绪。

Ma 部分官方 TSV 缺标题行，严格六列读取保留第一条真实 trial。旧 parser、计划、已关闭预算及旧缓存保留，新计划扣除全部旧消耗和前代时间取整余量，Ma 新缓存使用独立 `native_v2` 路径。预处理数学和原研究主线不随 parser 修复改变。本轮仍只推进 Codex 内的 CPU 准备；GPU 训练等待用户后续手动开机，不自动启动或重启 RunPod。

本轮还修复了 Paper3 传统对照对缓存精度的识别：原始 float32 缓存经 episode 接口无损提升为 float64 后，仍使用原 float32 的 CAR 数值容差；真正高精度输入继续采用严格容差。三个真实 Stieger session 的原 float32 输入与无损提升输入所得协方差位级相等，没有重新参考或修改信号。原117文件清单和旧实现保留，唯一已变更的旧清单成员及证据见 `classical_precision_amendment.json`。

所有 COMPLETE 均要求完整 roster、实际缓存、全部回读证据和已关闭且未超限的累计预算。清单生成、代码测试、dry-run 和文件上传分别记录，不能相互冒充。

## 开机后的执行顺序

1. 用户本人手动开机；恢复固定软件、离线依赖和 cache-only 数据，逐文件 GET/SHA 校验。无需恢复数百 GB 原 EEG。
2. 主线入口 `tools/run_training_campaign.py` 先完成唯一受控 source GPU 恢复 pilot，再按真实产物串联 f0、传统对照、Rref、元学习、开发选择与一次性内部检验。默认只读；实际执行须同时给 `--execute --user-started-gpu`。累计预算和固定 claim 保留，不能换目录重训。
3. 两条额外 source 支路使用 `tools/run_source_extension_downstream.py`，各自执行 f0、Rref、强对照、14方法和三种子 B* 开发冻结；必须先验收主线14方法冻结、相同 seed 的真实基线 epoch、全量 source 数据与工程计时。主线和扩展启动前统一 BLAS 线程配置。CAR3 仍使用独立输出、预算和工程验收；未实现或未冻结的实验不自动进入 fit。
4. 本轮启动流程止于开发与内部检验，W7 冻结、外测资格审计、外测预测和科研统计按后续阶段门执行。

最终能否开机以随固定包提供的实际验收回执为准。GPU 驱动兼容性、GPU 数值恢复与计时必须开机后实测；CPU 测试不构成训练完成或算法优势。
