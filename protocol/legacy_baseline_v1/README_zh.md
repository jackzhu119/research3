# 论文 3：先缓存源数据、再积累可复用基线

本包独立于论文 1 和论文 2。它完成的是**四个公开左/右手运动想象源数据集的准备流程、两种 CPU 数学基线、紧凑 EEGNet 系列基线以及可审计的训练产物保存**。默认命令只打印计划；明确加 `--execute` 才会下载数据或拟合模型。

截至 2026-10-18，已在隔离的本地 CPU 环境检查固定版 MOABB/MNE 接口，运行软件契约测试和合成数据整流程。**本次没有下载真实 EEG、没有登录云服务器、没有执行 GPU 训练，也没有得到可报告的科研准确率或新算法提升。** 实际验证记录见 `validation/`。

## 1. 范围与数据角色

| 数据集 | 本包角色 | 固定接口行为 | 执行边界 |
|---|---|---|---|
| BNCI2014_001 | Source | MOABB 1.7.2，9 个公开编号，显式左/右手事件 | 执行时核对每个 run 的实际通道、标签与完整时间窗 |
| Cho2017 | Source | 52 个公开编号，显式左/右手事件 | Raw 是适配器重建坐标；不能把拼接后的坐标当真实采集时钟 |
| Lee2019_MI | Source | 54 个公开编号，`train_run=True, test_run=False, resting_state=False` | 保留两个 session 的训练 run；注意其原始左/右标签编码与 BNCI 相反 |
| PhysionetMI | Source | 109 个公开编号，`imagined=True, executed=False` | 仅左/右手 MI 进入模型；适配器同时返回的双手/双脚 MI run 明确记录跳过原因 |
| Ma2022 | Development，登记待适配 | 固定 MOABB 1.7.2 实际没有导出此类 | 当前不能执行准备；官网较新文档的存在不等于此版本可调用，且通道不兼容默认 clinic11 |
| HS-Stroke、SEFMEID、GuttmannFlury2025_MI、Liu2024 | Holdout，登记待审 | `prep` 和 `train` 都不自动打开 | 最终协议、患者分层、时间/标签适配与 source 重叠审核完成后另行实施 |
| Yang2025、Dreyer2023、Kumar2024 | 排除本批 | 对应论文 1 活动数据或论文 2 数据 | 避免把既有论文数据直接变成第三篇的所谓全新验证 |

这些是**适配器公布编号数**，合计 224 个 source 编号，不是本次已下载人数，也不是完成跨库生物身份去重后的 224 名独立参与者。执行后的实际保留 trial 数、异常、缺失 context 和通道状态以每个 shard 的 JSON 为准。

来源、固定 API 与审查结论列于 `API_SOURCES.md`。本包从随机初始化训练。未来如加载 MIRepNet 等官方预训练权重，必须重新审查其训练语料与所有候选测试数据的重叠；当前 source 名单不能直接沿用为这些权重的严格外部测试名单。

## 2. 安装与首轮执行

需要 Linux、Python 3.12 和独立的数据输出目录。`install.sh` 创建本包的虚拟环境，记录解析后的完整依赖版本；它本身不准备 EEG 或启动拟合。直接依赖固定，完整环境在实际安装时另存，未使用全体依赖轮子的哈希锁文件。

```bash
cd /your/path/paper3_pretrain_20261018
bash install.sh cpu
.venv/bin/python -m p3 --root /your/persistent/path/paper3_source_data doctor
.venv/bin/python -m p3 --root /your/persistent/path/paper3_source_data prep
.venv/bin/python -m p3 --root /your/persistent/path/paper3_source_data train
```

GPU 环境可将安装命令改为 `bash install.sh cu128`。GPU 型号、驱动与 CUDA 兼容性仍需在实际服务器核对；本交付没有 GPU 实测。执行前应确认 `doctor` 的 `torch.cuda_available` 和数据盘容量。默认准备步骤要求至少 100 GiB 空闲空间；这是保守的停止阈值，**不是完整四库下载量的测量值或上界**。本实现把合并的处理后数组载入主机内存，真实规模下的内存和时间需由首轮实际准备量评估。

先用一个参与者实际校验通道与下载入口，再启动全体，能让适配问题尽早暴露：

```bash
.venv/bin/python -m p3 --root /your/persistent/path/paper3_source_data prep --datasets BNCI2014_001 --subjects 1 --execute
```

四库全体准备和三种模型的 source LODO 训练：

```bash
P3_ROOT=/your/persistent/path/paper3_source_data P3_DEVICE=cuda bash run_cloud.sh
P3_ROOT=/your/persistent/path/paper3_source_data P3_DEVICE=cuda bash run_cloud.sh --execute
```

第一条显示计划；第二条实际运行。CPU 可使用 `P3_DEVICE=cpu`。也可以分别执行，以便在准备完成后检查数据清单：

```bash
.venv/bin/python -m p3 --root /your/persistent/path/paper3_source_data prep --execute
.venv/bin/python -m p3 --root /your/persistent/path/paper3_source_data train --device cuda --execute
.venv/bin/python -m p3 --root /your/persistent/path/paper3_source_data status
```

默认四库 LODO 有 4 个 fold。每个 fold 包括 CSP+LDA 一次、TS+LR 一次，以及 EEGNet 三个 seed：20261018、20261019、20261020。**计划共 20 次 fit：8 次确定性数学基线 + 12 条神经网络轨迹。** 不把确定性模型的重复 seed 计为独立实验。

完成全部 source LODO 后，可另行训练全体源数据的部署起点：

```bash
.venv/bin/python -m p3 --root /your/persistent/path/paper3_source_data train --device cuda --final --execute
```

这个步骤新增 2 次数学基线 fit 和 3 条 EEGNet 轨迹。每个 EEGNet seed 的固定训练长度取对应四个已完成 LODO 选择 epoch 的中位数向下取整，至少 1 epoch；开始前核对输入、配置、代码和运行环境。全体源数据 fit 不读取最终临床测试标签。上述 25 次是完整配置的**计划数量**。

## 3. 预处理和科学解释

默认固定 11 通道及其顺序：

`FC3, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP4`

这个选择对应已核查的 HS-Stroke v6 官方通道清单，以便后续设计临床外测。它只解决固定通道编码器的一项兼容条件；并未证明 HS 的事件时间、任务语义、session 结构和患者终点已经适配。每个真实 source run 仍须通过 exact channel-name 核验：只忽略大小写，不补零、不插值；选中通道被 provider 标坏时停止。

统一步骤为：读取 MNE Raw 的 **SI volts**；以 MOABB 1.7.2 已标准化的任务 annotation 为零点提取 `[0,3)` 秒 context；对这 11 个通道作共同平均参考；在完整 context 内使用 4 阶 Butterworth 8–30 Hz 双向 SOS 滤波；`resample_poly` 至 128 Hz；保留 `[0.5,2.5)` 秒、共 256 点，存为 float32 volts。进入模型时仅对各 trial 各通道按时间去均值，并除以该 trial 的联合 RMS；不计算跨 trial 或目标域归一化统计。

MOABB `get_data()` 已把 adapter 的 `interval[0]` 加到事件样本，不能再额外平移。JSON 保存原始采样率、原始 sample、`first_samp`、相对 Raw 的 onset、原始事件码和二分类标签（左手 0，右手 1）。**双向滤波使用了截至 3 秒的 context；不能宣称这是 2.5 秒即可完成、无未来信息的实时决策。** 当前定位为离线预训练缓存。

250 µV 峰峰值阈值和 provider 的 BAD 注释只记录 QC flag，默认保留，避免事后按效果删试次。缺少完整 3 秒 context 的事件明确记录排除；非有限数值、歧义事件或缺失选定通道会停止。Physionet 明确的非目标双手/双脚 MI run 记录跳过；任何其他无 canonical 左/右标签的未知 run 都不会被泛化地吞掉。含有多个已知任务的 run 仅抽取显式左/右手事件。

原始下载文件留在该任务 `raw/`，按 `data_path()` 返回的原档记录完整 SHA-256，另对 adapter Raw 的全 EEG 通道作分块信号哈希。这不等于与所有 provider 官方发布校验清单进行了独立比对。处理后的 NPZ 只存配置通道；保留的原档和全通道审计信息支持以后重新适配 montage，而无需重新下载。不要删除 raw 和元数据只留模型权重。

## 4. 模型、验证与未来创新的边界

| 基线 | 实际实现 | 固定说明 |
|---|---|---|
| CSP+LDA | CAR 的 Helmert 子空间，trace covariance，0.1 shrinkage，两端各 2 个广义特征向量，log-power；`lsqr`/自动收缩 LDA | CSP 类均值按 dataset/person/class 加权；LDA 各行本身没有 sample weighting，class prior 为 0.5 |
| TS+LR | 0.1 收缩协方差，训练数据的加权 **log-Euclidean 均值参考**，参考白化后的矩阵 log，上三角 √2 缩放，训练集 StandardScaler 与 LR | 不宣称等同于 pyRiemann 默认 affine-invariant Riemannian mean 的 TangentSpace；LR C=1，最多 2000 次迭代，未收敛则失败 |
| Compact EEGNet | 独立实现的 EEGNet 系列小网络，temporal 63、F1=8、D=2、F2=16、depthwise 15、pooling 4/8 | 不是新算法，也不是某一库 EEGNet 的逐参数完全复刻 |

每个 LODO fold 留出整个 source dataset。所有训练和 validation 的 provider-subject ID 严格分开，保留同一人的所有 session，不进行同人 trial 随机切分。跨 split 同 event UID 或完全相同的处理后信号会被拒绝；这不能发现所有不同加工版本的跨库重复，生物身份独立性仍需另审。

EEGNet 只按 source validation 的跨参与者平均交叉熵选择 epoch/早停。它的“epoch”定义为 64 次、每次 64 trial 的有放回优化更新，**不表示完整遍历一次所有 trial**。采样权重依次平衡 dataset、person、class，最多 80 epoch，patience=15。LODO 的 held-out source 参与了选择，因此其结果属于**开发验证**，不能包装成完全未用过的外部测试。`train_diagnostic` 也是训练诊断，不是泛化性能。

未来的纵向或少标签适应实验必须另行冻结协议。`chronological_support_query` 和 `chronological_session_split` 仅提供完整早期 run/session 与后期 query 的分割 helper，**当前没有执行任何适应实验**；adapter 的自然序号仍需与真实采集顺序核对。最终患者 holdout 不参与本包的自动评估。

先训练的价值在于：未来可以直接读取保存的 trial 概率做统计、校准分析和既定模型的误差分析，并从 source checkpoint 开始新的实验。但全新的算法、改变滤波/montage、优化器或任务定义，往往仍需新的 fit 或 finetune；本包不承诺“以后任何新算法都不用再训练”。

## 5. 可续跑产物与统一输入接口

每个 `prepared/<dataset>/` 包含 `manifest.json` 和每人的 NPZ+JSON。NPZ 的 `X` 形状为 `trials × channels × time`，float32 volts；`y` 是 int64，左 0 右 1。JSON 包含统一 schema、通道顺序、采样率、预处理契约、实现代码哈希、原档哈希和逐 trial 身份。`p3.core.save_shard` 是自定义适配器的写入入口；`load_collection` 校验 SHA、形状、有限值、标签、UID、试次信号哈希及全部 shard 的共同处理契约。默认 CLI 只允许已登记的四个 source dataset；新数据的 adapter 必须经过审查和角色登记，不通过改 manifest 把 holdout 偷渡进训练。

每个 `runs/source_baselines/<fold>/<model>/seed-<seed>/` 保存：

- `identity.json`：配置、源码 SHA-256、运行环境、输入 manifest/shard 哈希和实际 train/validation event UID。
- `train_diagnostic_predictions.npz`、`source_validation_predictions.npz`：与 `indices` 对齐的 `y`、两类概率和两类 logits；对应 JSON 保留 trial 身份和 prediction role。全源 final 没有 validation 文件。
- 数学基线的无 pickle `classical_state.npz` 和身份/哈希 receipt，可以独立重放概率。
- EEGNet 的 `last.pt`：模型、Adam、采样 generator、PyTorch CPU/CUDA RNG、早停状态和 history；`selected.pt` 和每 epoch 模型权重；每 epoch 的 source validation 概率/logits/标签/indices。
- 参与者级 metrics、`fit_summary.json`、`artifact_manifest.json` 和清楚区分 RUNNING、PAUSED_AT_EPOCH_BOUNDARY、COMPLETE、FAILED_PRESERVED 的状态文件。

重用相同命令时，已完成任务先核查产物哈希后跳过 fit；未完成 EEGNet 从 `last.pt` 的 epoch 边界续跑。崩溃时未提交的半个 epoch 会重做。已验证同一 CPU 实现、同配置、同运行环境的连续 4 epoch 与 2+2 epoch 结果完全一致；**没有验证跨 GPU 型号、驱动或设备的位级复现**。更改代码、环境、配置或输入需要新实验输出目录，不隐式接纳旧选择或覆盖原实验。

可用 `train --models eegnet --fold BNCI2014_001 --max-new-epochs 5 --execute` 分段推进；随后相同配置不带上限即可继续。准备步骤按已完成 subject shard 复用并验证哈希。如果进程恰在 NPZ/JSON 与 manifest 之间异常终止，残留文件会保留并报错，需要人工核对该 shard 后恢复；包不会自行删除或“猜测”孤立数据。

## 6. 本地软件检查

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python -m p3 --root /tmp/paper3_synthetic_check smoke --with-torch --execute
```

`smoke` 明确生成合成数组，写入 `SYNTHETIC_ONLY` 路径，只验证接口与训练/保存流程。合成数值不能作为论文结果。`validation/` 记录交付时实际通过的测试、完整 CPU smoke、final fit 和已完成产物重用哈希检查。

同包中的 `tools/fetch_holdout_archive.py` 是单独的官方原档封存工具；其默认行为仅列计划，显式执行才下载并校验原档。该步骤不等于临床任务适配、标签审查或模型评估，详情以该工具自身帮助和报告为准。
