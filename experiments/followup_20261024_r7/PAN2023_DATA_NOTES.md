# 第七轮复用数据：第六轮已准备的Pan2023

**轮次说明：下列取数、许可、全部原MAT与预处理核验在第六轮准备时已经完成；第七轮只读复用同一份14人28会话缓存，不增加独立参与者，也不是新的首次盲测。原实测provenance与核验数值保留不变。**

原核验日期：2026-10-24。原记录更新了此前“Harvard下载受阻、Pan仅为候选”的状态；不是把NEMAR镜像计成另一批独立参与者。该准备过程没有加载模型、生成准确率或用外部效果调算法。

## 来源、许可与实际获取

原数据为Pan等的14人双会话左右手运动想象，[Harvard DOI10.7910/DVN/251NOW](https://doi.org/10.7910/DVN/251NOW)，关联[Journal of Neural Engineering2023论文](https://doi.org/10.1088/1741-2552/ad0a01)。原Harvard端点仍有机器人挑战；本次改从保留原字节的公开[NEMAR nm000291 v1.0.0镜像](https://www.nemar.org/dataset/nm000291?v=v1.0.0)获取。

[稳定源清单](https://data.nemar.org/nm000291/v1.0.0/manifest.json)和[镜像provenance](https://data.nemar.org/nm000291/v1.0.0/sourcedata/sourcedata_provenance.json)列出28个原文件、原DOI、MOABB版本和逐文件SHA256。**28个原MAT已全部实际下载并逐一匹配SHA256，总736,301,987bytes**，不是仅核对网页或文件头。文件ID到S1–S14/D1–D2的对应来自[公开原数据解析器](https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/pan2023.py)，固定列表保存在`PAN2023_DOWNLOAD_MANIFEST.json`。

[原作者DataCite登记](https://api.datacite.org/dois/10.7910/DVN/251NOW)明确CC0-1.0，NEMAR的dataset_description也列CC0。数值数据及本次滤波/裁剪衍生缓存可以分发；这不把论文PDF或其他代码的许可混入。许可证据见`compatibility/pan2023_license.json`。

Pan2023未在原源/开发数据登记中使用；它与Pan2025并非同一DOI。两者真实参与者是否重合未证明，本轮只纳入Pan2023，不把14人与另一份10人简单相加。也不能仅凭不同数据名证明所有跨研究真实身份绝不重合。

## 原数组与信号合同

28个文件实际均为MATv7.3/HDF5，只有`data,label,fs`三个顶层变量：`data=(120,1750,28)`、float64；`label=(1,120)`；`fs=250`。每会话60L/60R，共14人×2会话×120=3360条，全为二类MI，无需跳过其他任务。

实际HDF5轴顺序优先于摘要中的泛化数组说明。原28通道按作者/MOABB顺序选择clinic11：`FC3,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP4`，零起始索引`[1,5,7,8,9,10,11,12,13,15,19]`。MAT内没有独立通道名和单位变量；通道名、微伏转伏依赖原提供者解析器，不称MAT内独立物理标定。

每个1750点epoch包含MI开始前3秒与MI后4秒，MI起点sample750。仅读取`750:1125`的0–1.5秒上下文，先11通道CAR，再8–30Hz四阶双向SOS、64/125 polyphase(line)、裁剪0.25–1.25秒、每通道时域去均值，保存`120×11×128 float32`预RMS数组；联合RMS仍在现有冻结模型步骤，不在外部缓存提前执行。

实际S1D1首试次与原`p3work.preprocess.prepare_trial`比较：float64最大绝对差**0**，float32**逐位相同**。所有28会话均可通过现有输入合同和B4/8/12/16/20 materialize，所有3360条上下文有限且非零，没有剔除或补采。

## 前缀、会话与顺序边界

D1/D2是原作者的真实不同会话文件身份；保留两会话归属同一人。取各原MAT发布trial-index最前20位置为支持，余100位置为查询。短预算仍固定同一个B20查询，不把支持预算减少转成额外查询。

实际每会话前20位置是10L/10R，所有不重叠10位置块也是5L/5R；28个会话标签序列各不相同。原试次顺序没有被本导入器重排，但**这些标签结构与文件索引本身不证明实际物理采集时间顺序**。原文描述四个30试次区段，发布MAT没有区段边界、时间戳或trial_info；不能凭等长块恢复真实physical_run，也不能把MOABB的一次拼接run等同原物理run。

因此正式描述为**原会话MAT发布trial前缀的外部二类MI评价**，`physical_run_id=None`、`acquisition_order_verified=false`、`temporal_risk_eligible=false`。它比Weibo保留了确切的双会话身份与完整二类会话，仍不冒充新的在线前瞻采集或真实完整未来run验证。

每会话B20支持20、query100；总计**560支持位置、2800查询**。统计单位仍是**14人**，在每人内先聚合D1/D2和固定模型种子，不能用28会话或3360试次扩大独立n。

## 代码、缓存与运行

- `paper3r7/pan2023.py`：显式原MAT准备，以及只读`PanCatalog`。只有准备阶段导入h5py/scipy和下载，GPU运行期只读小缓存。
- `prepare_pan2023.py`：可选重建入口，支持逐会话原子保存与已有会话复用。
- `PAN2023_DOWNLOAD_MANIFEST.json`：28个固定文件、byte/SHA、原会话对应；不是训练时新拉取的可变索引。
- `compatibility/pan2023_all28.json`：全部28会话的实际原schema、类数、5预算/固定query与预处理一致性。
- `compatibility/pan2023_portable.json`：便携缓存包字节数/SHA256、独立解包全部会话读取比较。
- `tests/test_pan2023.py`：原轴/通道/有限上下文、失效位置仍计费、两会话不当两个人、固定query与路径/hash错误可见。

完整缓存已准备，不需在学校作业安装h5py或下载736MB原MAT。发布便携包只包含裁剪数值缓存、相对路径事件/schema、原清单、许可与引用，不含原MAT、源模型或任何账户信息。第七轮核心实验不依赖可选取数进程；采用已发布包时读取合同与各实际文件即可。

第七轮在此同一数据上固定新的方法、温度、比较族与统计规则，属于探索性重复外部验证。不能将同一14人按轮次重复计为新人，单轮多重比较校正也不能消除多轮探索影响。跨研究评价不能保证显著或把回顾性公开数据变成前瞻临床结论；若仅概率校准有益而BA无提升，应原样报告，数据与机制的阴性结果同样保留。
