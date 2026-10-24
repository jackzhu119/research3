# 外部数据历史记录：第三轮候选核验与Weibo准备

**第七轮复用说明：Pan2023在第六轮已从原字节公开镜像完整下载并准备，28个原MAT全部SHA验证、14人28会话3360条缓存均已就绪。第七轮沿用同一队列作探索性重复外部验证，不新增参与者或首次盲测。当前状态、许可、顺序限制和交付方式请读[PAN2023_DATA_NOTES.md](PAN2023_DATA_NOTES.md)。以下正文保留第三轮时点的候选核验；其中Pan下载403/未ready不代表第六/七轮现状。**

核验日期：2026-10-24，已纳入后续授权取数结果。Weibo2014的10个原MAT均已完整下载、逐文件验证SHA-256并处理为新缓存；Pan/Eye-BCI仍仅核对公开元数据及少量文件头。没有训练模型、计算识别效果或操作学校服务器。完成数据准备不等于完成外测，也不认证真实参与者跨研究完全独立。

## 结论

**优先候选是 Weibo2014/Yi2014 的确定性左右手子任务；第二候选是 Pan2023。** 两者都没有列在当前 `/workspace/paper3_hpc/protocol/DATA_REGISTRY.json` 的已分配数据中。Weibo 已从NEMAR保存原字节的镜像完成4.34 GB原MAT取数和SHA核验，得到约11 MB缓存；Pan2023 的作者登记元数据与 CC0 已验证，原 MAT 合计约 0.74 GB，但本环境对 Harvard Dataverse 文件端点遇到 403/机器人挑战，需要在取数前解决可用下载路径。Eye-BCI 可作独立的坏道/眼动压力测试储备，**原文明确报告 FC3 坏道问题**，不宜直接称为干净的 clinic11 主确认。

现有源为 BNCI2014_001、Cho2017、Lee2019_MI、PhysionetMI；Stieger2021 为开发；Dreyer2023 和 Kumar2024 已暴露，Yang2025 已在此前体系出现且属可选增强源。不能把它们或其 NEMAR/Zenodo 别名重新算作全新外部数据。当前 clinic11 是 `FC3,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP4`。

## 1. Weibo2014 / Yi2014：推荐的轻量外部子任务

### 一手来源与许可

- Yi W, Qiu S, Wang K, et al. **Evaluation of EEG Oscillatory Patterns and Cognitive Process during Simple and Compound Limb Motor Imagery.** *PLOS ONE* 9(12):e114853, 2014-12-09. [原论文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0114853)。论文 Data Availability 指向 Harvard Dataverse，明确数据无访问限制。
- 原始数据 DOI 是 [10.7910/DVN/27306](https://doi.org/10.7910/DVN/27306)，题名 *EEG data of simple and compound limb motor imagery*；**不是 GigaDB 100295，后者是已经用作源的 Cho2017**。
- [作者登记的 DataCite 元数据 API](https://api.datacite.org/dois/10.7910/DVN/27306) 本次匿名 GET 200；版本 1.0，开放获取，数据许可 **CC0-1.0**。原论文自身 CC BY 与数据 CC0 应分别引用。
- [NEMAR 发布记录 v1.0.3](https://www.nemar.org/dataset/nm000146?v=v1.0.3) 和 [稳定 manifest](https://data.nemar.org/nm000146/v1.0.3/manifest.json) 匿名可读。其 [sourcedata/README](https://data.nemar.org/nm000146/v1.0.3/sourcedata/README.md) 明确源文件按上游字节原样保存，2026-08-13 经 MOABB 获取，仅去掉缓存包装/文件名前缀；[源文件 provenance](https://data.nemar.org/nm000146/v1.0.3/sourcedata/sourcedata_provenance.json) 与 manifest 给 SHA-256。NEMAR 的 BIDS 是衍生物，原 MAT 可以单独获取。

### 原任务、输入与全部原MAT的实际核验

原论文为 10 位健康右利手参与者，六类运动想象加静息；8 个 MI 区段各 60 试次，每种 MI 在每区段 10 次，另有 80 次静息。每试次 8 秒：0–2 秒休息、2–3 秒准备、3–7 秒 MI、7–8 秒休息；任务次序随机，区段之间休息约 5–10 分钟。采集 1000 Hz，供分析的信号已滤波并降到 200 Hz；左右手单独想象是目标任务，不混入双手、手足组合或脚。原论文用 60 个 EEG 位置，另外 HEO/VEO/CB1/CB2 不纳入其拓扑分析。[原论文方法](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0114853)

最初读取NEMAR元数据与少量MAT头后，已继续完成10个原MAT的实际取数与解析。最终缓存只依赖原MAT的`data`/`label`，不用BIDS事件顺序：

- 本地逐文件SHA-256与固定发布清单完全匹配，总计 **4,340,450,480 bytes（约4.34 GB）**；原始信号只在准备环境短暂保存，缓存提交后逐人删除，未发布原MAT。10人原数值`data`均为float64；9人为64×1600×560，第6人为64×1600×500。MAT头中label类为double，SciPy实际读取label数组为uint8；均验证为整数1–7，未猜轴或重映射类别。
- 实际共有 **5,540条全部任务、1,580条LR**：9人各560/160，第6人500/140。各人的发布顺序前20条LR都为10L+10R；对应200个支持位置、1,380条B20固定查询。9人各140查询，第6人120。全部1,580条LR的规定上下文均通过有限性和非零能量检查，没有免费替换无效前缀。
- 原MAT发布顺序与初期NEMAR BIDS事件表不同。正式原索引、原标签和prefix均从实际MAT生成；**原MAT前20条全部任务仅有4–8条LR**。早期BIDS审计文件只保留为历史元数据，不用于数值缓存或预算计数。
- clinic11按[原MOABB MAT解析器](https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/Weibo2014.py)顺序选择，零起始索引为`[16,20,24,25,26,27,28,29,30,34,38]`，`1=left_hand,2=right_hand`。MAT没有独立通道名或单位字段；通道和微伏×1e−6转换有原解析器证据，但不称为MAT内独立物理标定。原上游滤波/降采样不能逆转，保留为外部域差异。
- 每个8秒epoch的MI锚点为3秒（200 Hz下sample600），仅取原样本`600:900`作为MI后0–1.5秒上下文；精确CAR11、8–30 Hz四阶双向SOS、16/25 polyphase（line padding）、0.25–1.25秒裁剪和每通道时域去均值后，保存 **N×11×128 float32 pre-RMS**数组。联合RMS留在已有模型步骤。实际W01数据与`p3work.preprocess.prepare_trial`比较：float64最大绝对差 **0**，float32逐位一致，QC相同。
- 真实缓存共 **11,019,151 bytes（约11 MB）**。`compatibility/weibo_all10.json`记录每个人的原schema、类数、前缀原索引、各预算相同查询身份、无prefix/query重叠、CAR/时间去均值误差和pre-RMS幅值。规定预算4/8/12/16/20全部 materialize成功；缓存源合同SHA为`5c8ea3f01634c060546af29b9330d314c4f03c265677a74b875ac7c302e31872`。

### 推荐协议与限制

冻结模型后，保留提供者的试次索引与原序，确定性选择左右手子任务，取最前20个**原MAT发布顺序左右手位置**适应，后续左右手位置查询；无效信号仍占其相应采集位置，不事后补齐。各方法、种子及预算共享同一 B20 查询身份，主统计单位为 10 人。

必须称为“左右手子任务的 B20 校准”，并报告原全任务序号/被跳过的其他任务数。原MAT发布顺序前20个全任务试次实际只有4–8个左右手事件；因此该协议不是“原七类会话开始后仅经历 20 次试验”。若坚持后者，支持集只能取原全任务前 20 个位置中的左右手标签，并单独报告可用支持量。两种预算不能混写。

原 MAT/衍生事件表没有已经核实的原始物理 run 身份。NEMAR 一个人暴露一个拼接 run，不代表原文八个 MI 区段是一整段连续采集；禁止凭等长切窗伪造物理运行段，故当前不适合验证 future-run TERM/P10 的长时段命题。实际10个MAT末尾均为80条静息，已核实的是发布存储布局；原物理区段位置和subject_6缺失60条的采集原因仍未还原，缓存只称“发布顺序LR子任务前缀”（published-order LR-prefix20）。它不代表真实物理采集前20条，也不是新的前瞻性独立确认。

### 下载端点与预算

- 已仅取manifest中 `sourcedata/subject_*.mat`，共 10 文件 **4,340,450,480 字节**，另取小型 manifest/provenance/说明；无需下整个 BIDS/Zarr。
- [subject_1 原 MAT 稳定端点](https://data.nemar.org/nm000146/v1.0.3/sourcedata/subject_1.mat)：439,447,715 字节；初期Range读取返回206；随后已完整下载439,447,715 bytes并本地核对SHA-256。
- [原 Harvard archive 1](https://dataverse.harvard.edu/api/access/datafile/2499178)、[archive 2](https://dataverse.harvard.edu/api/access/datafile/2499182)、[archive 3](https://dataverse.harvard.edu/api/access/datafile/2499179) 是 MOABB 列出的原下载地址；DataCite 三包大小合计 **4,340,453,316 字节**。Harvard 本次有机器人挑战，优先 NEMAR 保留源字节的稳定端点。
- 本次 manifest 所列整个 NEMAR 版本总计 **6,060,958,533 字节**，网页压缩 ZIP 显示约 5.8 GB；这与 4.34 GB 源 MAT 预算不同。
- 别名重合检索至少覆盖 `Weibo2014`, `Yi2014`, `10.7910/DVN/27306`, `nm000146`。当前 registry 没有它们，但没有据此认证跨研究真实参与者绝不重合。

`WEIBO_DOWNLOAD_MANIFEST.json`保存10个原MAT的稳定URL、bytes和发布SHA；本次已完整下载并重新计算全部本地SHA，均相符。该源下载清单作为缓存已绑定的历史输入保持字节不变；其中初期pending描述以`compatibility/weibo_all10.json`的完整实际核验结果为准。`prepare_weibo.py --download --delete-raw-after-prepare`采用顺序准备并逐人原子保存清单，只删除本次新下载且已验证/缓存提交的原档，保留原先就有的文件；网络失败只停止该可选取数进程，已完成人员仍可复用。运行期ExternalCatalog不下载，也不依赖SciPy/MNE/MOABB。核心实验不依赖可选取数是否成功。

### 数值衍生缓存的许可与便携发布

原Harvard DOI的[作者登记DataCite rightsList](https://api.datacite.org/dois/10.7910/DVN/27306)明确写为 **Creative Commons Zero v1.0 Universal，CC0-1.0**，且给出[正式legalcode URL](https://creativecommons.org/publicdomain/zero/1.0/legalcode)。[NEMAR v1.0.3记录](https://www.nemar.org/dataset/nm000146?v=v1.0.3)同样列CC0。[CC0官方说明](https://creativecommons.org/publicdomain/zero/1.0/)允许复制、修改、分发，包括商业用途；因此来源数值数据的选通道、滤波与裁剪衍生缓存可以分发。保留来源引用和变化说明，不将论文PDF或代码的独立许可混入，不暗示原作者背书。

实际许可证据在`compatibility/weibo_license.json`。便携包`weibo-portable.tar.gz`仅含相对路径manifest、W01–W10的裁剪数组/事件/schema，以及来源、CC0说明和去除绝对工作路径的核验JSON；不含原MAT、源模型、账户信息或其他数据集。SHA-256、精确字节数、缓存原清单SHA及从解包副本读取10人的验证记录在`compatibility/weibo_portable.json`。现有缓存不因打包而修改。便携gzip包精确为 **8,441,250 bytes**，SHA-256为`253bfe3e09e19bd72a456474f426273c37e4ee20f2cb1dd38f441c4ca9a4a34b`；已从独立解包目录在禁止网络请求条件下经ExternalCatalog读取10人，逐位核对全部数组及5个预算的支持/查询张量、UID和200/1380总量，与原缓存完全一致。原缓存manifest的SHA仍为`3080c1028898eae98045238cbded0a5ad7ac6e91543e52432e813b728096b467`。

## 2. Pan2023：更贴近真正两类跨会话的备选

[作者数据 DOI 10.7910/DVN/251NOW](https://doi.org/10.7910/DVN/251NOW) 与 [作者登记的 DataCite 元数据](https://api.datacite.org/dois/10.7910/DVN/251NOW) 确认 14 位健康参与者、每人 2 次会话、每会话 120 次左右手 MI，28 通道、250 Hz、CC0；当前登记版本 1.3，2024-03-30 更新。与 [Pan 等 JNE 2023 论文](https://doi.org/10.1088/1741-2552/ad0a01) 关联。原 MAT 元数据列出的 28 个文件大小合计 **736,301,987 字节**。

[MOABB 实现](https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/pan2023.py) 的 28 通道表含全部 clinic11。它记录 4 个 30 试次区段/会话，每区段 15 左15右、原文件没有区段边界；每试次为 2 秒休息+1秒准备+4秒 MI。源文件是预切段 MAT v7.3，MI 锚点相对段起点为 3 秒。当前 DataCite 抽象里的变量布局说明和 MOABB 具体解析并不完全相同，必须实际验 MAT，不按摘要猜轴。

具体原文件例：[subject1 day1](https://dataverse.harvard.edu/api/access/datafile/7574475)、[subject1 day2](https://dataverse.harvard.edu/api/access/datafile/7574479)；全部 28 个 ID 在上述 MOABB 源码中。本次这些端点 HEAD 403，Harvard 落地页给 JavaScript 机器人挑战。**数据开放许可和无需作者密码已核实，当前机器匿名取数成功尚未核实**，因此不能把它计入已经 ready 的数据。建议只做可选导入器/manifest，下载成功、原序和 native 通道映射通过后再纳入。

Pan2025 [10.7910/DVN/GH74ZG](https://doi.org/10.7910/DVN/GH74ZG) 是另一份 10人×2会话、每会话约180次、20个 MAT 合计 **798,523,889 字节**、CC0 的发布，不能按同一人群加成 24 人。第二会话前30次训练、后续在线反馈，试次数不定。与 Pan2023 的真实身份非重合未经证明，二者选一个做外部队列较清晰。Pan2025 的具体 [day1 文件](https://dataverse.harvard.edu/api/access/datafile/11704410)、[day2 文件](https://dataverse.harvard.edu/api/access/datafile/11704404) 也遇到相同 403；来源与版本见 [DataCite](https://api.datacite.org/dois/10.7910/DVN/GH74ZG)。

## 3. Eye-BCI：只能附带限定的备用压力测试

Guttmann-Flury E, Sheng X, Zhu X. *Dataset combining EEG, eye-tracking, and high-speed video for ocular activity analysis across BCI paradigms.* *Scientific Data* 12:587 (2025-04-08), [原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978869/)，[原数据 DOI](https://doi.org/10.7303/syn64005218)。31 人共 63 次会话：14 人一次、2 人两次、15 人三次。每次 MI 40 次（20 左20右），2秒注视+4秒想象+1–1.5秒休息，1000Hz，62 EEG 加辅助通道；原数据许可 CC0，原论文许可另为 CC BY-NC-ND。原文原 `Trig` 表示试次序号 1–40，`Cues` 才区分 Fixation/Left/Right，不能照原 Trig 直接作 1/2 类标签。

可匿名下载的 [2026-03-12 Zenodo MI 重托管版本](https://zenodo.org/records/18970793) 属 MOABB 关联的 CSV→BDF 衍生物，不是原文明确背书的新增发布；未发现作者原网页对该衍生版本的明确认可，不能写成“作者官方等价重发布已验证”。活 API 许可为 CC0，31 ZIP 总计 **26,198,687,829 字节**，[S01 ZIP 稳定端点](https://zenodo.org/records/18970793/files/S01.zip?download=1) 为 492,065,101 字节。

仅用 HTTP206 读取 S01 头、中央目录和 33,023 字节 annotation：实际 BDF 67 个信号槽（包括 66 个数据通道和 BDF Annotations），全部 clinic11/uV/1000Hz；40 个 imagery 事件、原试次计数1–40、onset递增，前20和后20各10左10右；首次 imagery onset35.396秒，比 fixation33.396晚2秒。仅此一会话的头与侧表通过，未核整个库及 BDF Trig 与 JSON 的对齐。

**更重要的是原论文的坏道处理建议列出 FC3，并建议凡任一会话被标坏的通道在整体分析排除。** 因此不能只见 header 有 FC3 就声称干净 clinic11 可用，也不能擅自插值补回。若不变更主输入合同，保留为坏道敏感性数据更诚实；不能把全部63会话都宣告合格。每会话仅剩20查询试次，亦不适合长时段尾部风险结论。

## 4. 不应误选的公开入口

- GigaDB 100295、100542 分别是现有 Cho2017、Lee2019，不是新增外部队列。
- GIN `robintibor/high-gamma-dataset` 是 Schirrmeister2017 的**运动执行**任务，左右手等类别来自实际动作；即使公开、通道充分，也不能混写为相同左右手 MI 确认。原研究 [Human Brain Mapping 2017](https://doi.org/10.1002/hbm.23730)。
- AlexMI 是 right-hand/feet/rest，且16通道不覆盖 clinic11；Shin2017A 的相邻半位点也不能当作相同 FC3/C3/CP3 通道。二者不满足本轮固定输入/任务。

## 供主报告引用的小型证据文件

`27306_datacite_metadata.json`、`251now_datacite_metadata.json`、`gh74zg_datacite_metadata.json`、`weibo_nemar_v103_manifest.json`、`weibo_nemar_v103_events_audit.json`、`weibo_nemar_v103_budget_audit.json`（两份BIDS事件/预算审计均属历史元数据，正式数值结果以`compatibility/weibo_all10.json`为准）、`eye_bci_zenodo_metadata.json`、`eye_bci_s01_header_audit.json`、`eye_bci_s01_member_manifest.json`、`eye_bci_s01_annotation_schema.json`、`eye_bci_s01_order_audit.json`。未保留带个人身份字段的原 BDF 头二进制，也未将临时签名 URL 写入本文。
