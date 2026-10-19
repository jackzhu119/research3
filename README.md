# Paper3 — v2.0 协议与获批 v2.1 窗口修订

原始 `Paper3_Work_Handoff_20261018.zip` 已实际读取，529,763 bytes，整体 SHA256 为 `0398664faf34682a07fb38de4e46df9829f0e5ad050f657f905d0373dbfaf175`；包内 manifest 的 **52/52 文件逐项匹配**。原件位于 `protocol/`，后续实现位于 `p3work/`，两者分开维护。`legacy_baseline_v1` 保持原字节，只是历史基础软件，不是新算法或真实训练证据。

**W0、W1 已实际完成；62 人资格审计、40/11/11 分组及获批 v2.1 共同窗口均已冻结。** 分组 SHA256 为 `1a4704555760dd1d8956b31268931617b5c12d2441a17f68dd8f34d108dd61fd`。完整时长审计覆盖已冻结 meta-train 40 人的全部396个官方 session。原 v2.0 的两个固定候选均未通过保留率门；经用户批准，只增加并审计一个 v2.1 固定候选：反馈起点不变，上下文 `[0,1.5)` 秒、裁切 `[0.25,1.25)` 秒、128 Hz、128点。该候选通过原80%保留率和10个百分点类别排除差门；原协议52文件和失败证据均保留。唯一 source CPU 协调器现按冻结505候选清单准备 W2，仍限制两条文件流程、一个原生解析器。阶段状态见 `status/tasks.json`，活动进度以私有 `audit/source_full/progress.json` 及 source worker 的 `batch_progress.json` 为准。W2 只有在实际数值缓存及 R2 完整回读均验收后才能完成。旧 `audit/w0_preflight.json` 记录原包尚未附加时的历史阻塞，不代表当前状态。

## 已完成的真实工作

- 四 source 各一份原生 schema 样本：BNCI 144、Cho 200、OpenBMI offline train 100、PhysioNet 15，共459条原始 LR 元数据；不是完整 source 队列或最终合格训练数。
- Stieger S1/Session1：450个混合任务 trial，150个纯水平 LR trial；保留 task/run/trial、原始时间、反馈终点、实测通道与 µV 转换。样本覆盖不能代替 meta-train 全体选窗。
- 全62人资格原档、原生元数据与解析审计均完成官方校验及 R2 完整 GET/SHA256 核验，原档共38,395,648,006 bytes。396个 meta-train session 的完整时长审计亦已完成，原始纯 LR 分母为59,399；获批 v2.1 候选保留48,871条，按参与者等权的保留率为82.12698%、类别排除率绝对差为4.38613个百分点。以上为波形 QC 前的元数据时长门，不是最终有效缓存数量或训练结果。
- 原始协议 ZIP 与5个 schema 原档已通过现有 R2 条件 PUT、完整 GET、字节数和 SHA256 校验。没有用 ETag 或 HEAD 冒充回读验收。
- 新几何/适配内核、B0–B11/P/P+PCT 注册表、真实 episode 接口、source 与 meta 可恢复训练入口及不可覆盖预测存档均已实现。Rref/G/U0 的真实估计、训练和选择仍需上游数据及真实产物验收。CPU 合成测试只验证软件；没有真实科研模型、性能结果或科研 pilot。
- 已增加独立数据验收：完整396会话元数据的精确计数窗口门、505个 source 原档及每条原生 LR trial 的 valid/excluded 归账、实际缓存字节与 R2 完整回读证明。只读 support 接口和模型参数恢复都保留来源、角色与付费前缀约束，不能用状态标记替代真实数据。
- CPU 依赖实际安装并锁定；CPU和 CUDA12.8 的 CPython3.12/Linux x86_64 离线 wheels 已下载。GPU profile 已在独立临时环境实际离线安装，pip check、100个版本与Torch CPU导入/小矩阵检查均通过；仍未验证GPU或驱动。安装日志及审计已实际R2完整GET核验，临时环境验证后回收，原环境和wheels保留。
- 当前统一 CPU 软件回归为1319项通过，107份 Python 代码的清单与验收范围见 `audit/software_preparation_acceptance.json`。这些测试不代表真实 EEG 拟合、GPU恢复验证或科研性能结果。

本地 CPU 双文件传输、单原生解析协调器已通过独立预算与中断恢复测试，并已在旧进程退出、锁释放后实际接管累计预算和断点；它保留原名单及阶段门，仅修改运行并发。实际单一进程及逐文件状态以私有运行进度为准。

## 数据、访问与公开仓库

当前 GitHub 仓库为 PUBLIC。因此新增 EEG、逐trial清单、官方数据catalog、R2回执及批处理状态仅保存在本地与用户现有 R2。公开分支保留代码、用户授权导入的协议和汇总验收，不公开新增数据或凭据。完整固定包在用户现有 R2 及本地共享交付目录中生成。

- `protocol/`：原始只读研究规格及52文件校验来源。
- `p3work/`：新原生适配、权限/信号合同、方法、可恢复训练和预测存档。
- `contracts/input_contract.json`：已冻结的 v2.1 信号合同；其余未到阶段的科研参数继续保持 null。
- `data/`：本次独立下载的原档和依赖 wheels；不复用论文1旧 NPZ。
- `data_catalog/`、`audit/source_schema/`、`audit/stieger_w1/`：私有原生证据及逐事件记录。
- `manifests/geometry_constants/`：实际固定 Q/E 数学数组及 SHA；不是 EEG 缓存或模型。

Stieger 分组必须先于选窗，时长决策只读已冻结 meta-train 的完整 session 元数据。HS/Wang 外测不下载、不解封；内部留出不用于方法选择。Cho按类别存储的顺序不能作为采集顺序。三个source MAT的固定loader单位证据与原生header证据分开记录，不授权绝对幅值QC或猜测物理run。

## 已实现的准备命令

```bash
python -m p3work doctor
python -m p3work status
python -m p3work pilot-plan
python tools/check_pilot_readiness.py --require-ready
python tools/source_schema_audit.py --raw-root data/source_schema
python tools/stieger_metadata_batch.py --plan plans/stieger_qualification.json
python tools/train_source.py --help
python tools/train_classical_source.py --help
python tools/train_meta.py --help
python tools/prepare_stieger_cache.py --help
python tools/verify_window_gate.py --audit-duration-batch
python tools/evaluate_development.py --help
```

Stieger 资格与完整时长批次已结束，保留原计划、累计预算、原档校验、原生审计及 R2 完整回读证据；不要重新启动历史协调器。当前 source 入口为 `tools/source_preparation_parallel.py`，默认只读，实际执行须显式追加 `--execute`。同计划重复启动由文件锁拒绝；活动进程结束后，同一命令可按原状态及累计预算恢复，已完成对象经证据校验后复用。该入口只执行 source 原档准备、原生审计、新合同数值缓存及 R2 完整回读；不运行评估或训练，也不自动生成最终交付包。候选清单固定505份原档，最终合格人数及缓存条数由实际完整审计决定，不以候选数冒充结果。

`tools/r2_backup.py` 和 `tools/r2_restore.py` 均默认dry-run；只有显式 `--execute` 才调用既有R2，严格使用文件manifest及完整GET/SHA。不扫描或复制环境、密钥文件及论文1/2上传配置。

## 阶段状态与 GPU 边界

完整meta-train时长、共同窗口和四source新缓存的实际完成状态以 `status/tasks.json` 的 W1/W2 及其验收产物为准；只有两门均通过才能生成 pilot-ready 包。尚无有效 f0/Rref/G 科研权重。强对照代码已补齐；真实数据训练、GPU恢复与阶段验收尚未执行，W5不能标为完成，详见 `p3work/methods/STATUS.md`。source神经开发/确认分别15/25次拟合，旧数学基线10次；新版B4/B5 source bank另需6次，不能将相似配方合并计数。

已有 RunPod 的关闭和后续开机由用户管理，当前状态未通过本任务观察或更改。所有 GPU 训练等待用户后续开机；代码和准备工作不会自动启动或重启服务器。受控source pilot只依赖真实W1/W2；W3 Stieger数值缓存及后续科研阶段仍单独验收。`build_delivery.py --checkpoint --execute` 只生成明确标注未就绪的可恢复快照；正式固定包必须通过W1/W2及完整缓存回执覆盖检查。不得运行 legacy 的全量执行入口跳过新合同。

受控 pilot 固定比较100次连续更新与50次更新后保存/重载再更新50次，总计200次；随后仅对固定四source训练UID批次重放预测和指标，不增加优化器更新。验收检查实际检查点、预测、指标、依赖清单及其SHA。运行后生成显式结果备份manifest，成功和失败产物均由独立手动R2命令上传并完整GET核验；本地manifest不代表已上传。单次job claim保留，不会因失败而自动重启。GPU实际恢复、预测一致性和计时结果均待用户后续开机实测。
