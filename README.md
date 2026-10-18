# Paper3 — v2.0 协议导入与研究准备

原始 `Paper3_Work_Handoff_20261018.zip` 已实际读取，529,763 bytes，整体 SHA256 为 `0398664faf34682a07fb38de4e46df9829f0e5ad050f657f905d0373dbfaf175`；包内 manifest 的 **52/52 文件逐项匹配**。原件位于 `protocol/`，后续实现位于 `p3work/`，两者分开维护。`legacy_baseline_v1` 保持原字节，只是历史基础软件，不是新算法或真实训练证据。

**本提交时 W0 本地核查完成，W1 正在执行；统一窗口与参与者分组尚未冻结。** 阶段状态见 `status/tasks.json`，运行中的逐文件进度以本地 `status/local_preparation.json` 和 worker `batch_progress.json` 为准。只有真实 W1 完成后才能创建正式数值缓存。旧 `audit/w0_preflight.json` 记录原包尚未附加时的历史阻塞，不代表当前状态。

## 已完成的真实工作

- 四 source 各一份原生 schema 样本：BNCI 144、Cho 200、OpenBMI offline train 100、PhysioNet 15，共459条原始 LR 元数据；不是完整 source 队列或最终合格训练数。
- Stieger S1/Session1：450个混合任务 trial，150个纯水平 LR trial；保留 task/run/trial、原始时间、反馈终点、实测通道与 µV 转换。样本覆盖不能代替 meta-train 全体选窗。
- 原始协议 ZIP 与5个 schema 原档已通过现有 R2 条件 PUT、完整 GET、字节数和 SHA256 校验。没有用 ETag 或 HEAD 冒充回读验收。
- 新几何/适配内核、B0–B11/P/P+PCT 注册表、真实 episode 接口、source 与 meta 可恢复训练入口及不可覆盖预测存档均已实现。Rref/G/U0 的真实估计、训练和选择仍需上游数据及真实产物验收。CPU 合成测试只验证软件；没有真实科研模型、性能结果或科研 pilot。
- CPU 依赖实际安装并锁定；CPU和 CUDA12.8 的 CPython3.12/Linux x86_64 离线 wheels 已下载。CUDA profile 仅离线解析，未在 GPU 验证；备份状态以私有回执为准。

## 数据、访问与公开仓库

当前 GitHub 仓库为 PUBLIC。因此新增 EEG、逐trial清单、官方数据catalog、R2回执及批处理状态仅保存在本地与用户现有 R2。公开分支保留代码、用户授权导入的协议和汇总验收，不公开新增数据或凭据。完整固定包在私有 R2 及本地共享交付目录中生成。

- `protocol/`：原始只读研究规格及52文件校验来源。
- `p3work/`：新原生适配、权限/信号合同、方法、可恢复训练和预测存档。
- `contracts/`：当前工作合同；未知的窗口、名单和冻结参数保持 null。
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
```

Stieger 执行需显式追加 `--execute`；同计划重复启动会由文件锁拒绝，结束后同命令可幂等恢复。资格计划固定62个最早session、38,395,648,006 bytes、单文件串行、GPU 0小时、活动wall上限14,400秒。原档校验、解析、R2完整回读成功后才清理工具自己新建的临时raw，已有S1原档保留。该基础 worker 在资格完成后仅生成 meta-train 全session计划。`tools/continue_local_preparation.py --execute` 只等待已运行的资格任务，核验62份回执后串行推进完整 meta-train 时长元数据、选窗、W2 source 新缓存和固定包备份；不重启资格任务，不执行评估或训练。完整时长批次使用独立记录的18小时CPU活动wall上限，原始选择计划及其SHA保留，名单和科学规则不变。

`tools/r2_backup.py` 和 `tools/r2_restore.py` 均默认dry-run；只有显式 `--execute` 才调用既有R2，严格使用文件manifest及完整GET/SHA。不扫描或复制环境、密钥文件及论文1/2上传配置。

## 当前未完成与 GPU 边界

本提交时，全体资格、40/11/11、完整meta-train时长和共同窗口尚未完成；四source完整新缓存未生成；无有效 f0/Rref/G 科研权重。强对照代码已补齐；真实数据训练、GPU恢复与阶段验收尚未执行，W5不能标为完成，详见 `p3work/methods/STATUS.md`。source神经开发/确认分别15/25次拟合，旧数学基线10次；新版B4/B5 source bank另需6次，不能将相似配方合并计数。

已有 RunPod 的关闭和后续开机由用户管理，当前状态未通过本任务观察或更改。所有 GPU 训练等待用户后续开机；代码和准备工作不会自动启动或重启服务器。受控source pilot只依赖真实W1/W2；W3 Stieger数值缓存及后续科研阶段仍单独验收。`build_delivery.py --checkpoint --execute` 只生成明确标注未就绪的可恢复快照；正式固定包必须通过W1/W2及完整缓存回执覆盖检查。不得运行 legacy 的全量执行入口跳过新合同。
