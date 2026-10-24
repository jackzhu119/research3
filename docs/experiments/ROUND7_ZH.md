# 第七轮：有限窗时空Lie、头容量可辨识性与源相对风险

[返回七轮总档案](README_ZH.md)。交付核验记录：2026-10-24 UTC13:48（北京时间21:48）。526项本地CPU全套测试、源码发布与真实线上下载/禁网安装读取通过，未提交学校作业，无第七轮准确率结果。

## 三个研究轴及数学范围

保持冻结EEGNet、BN、clinic11/CAR有效10维、128Hz和既有8–30Hz/固定窗口合同。源模型与缓存复用，只用所获取前缀支持更新参数，未来query不进入训练或选模。B4/8/12/16/20共用同一个B20之后的query；QC失败仍占位置，不补采。

**有限窗时空模型。** 令白化信号Y为10×128，D为去时间均值的有限窗反对称移位算子，并归一到Frobenius范数√T。共同r=4参数经原字典形成无迹矩阵M，再分解K=(M−Mᵀ)/2和S=(M+Mᵀ)/2，定义H(Y)=KY+SYDᵀ。向量化生成元I⊗K+D⊗S反对称，因此指数变换保持白化能量、时间零均值子空间和参数取反的逆。生产实现利用D的谱分解和小10×10块，避免构造1280阶指数。

此生成元预算与静态图仅在完整10T ambient空间的Frobenius度量匹配；时间零均值子空间略有差别，更不等于每个真实信号的位移等幅。原CAR能量也不必守恒。数值用x+QᵀR¹ᐟ²(Yθ−Y)保留恒等点和浮点CAR残余。有限窗操作非因果，D模式不是生理频率，变换可产生原8–30Hz带外能量；这些量作为诊断，不变成训练前阻拦。

正式加入rotation-only与temporal-only两组消融，均用共同static_ce字典、CE目标和full头，各3种子，以排除收益仅因去掉静态gain或只保留某一成分。K/S投影同时改变幅度与可能的有效rank，不能据差值唯一归因于严格等幅的纯时间方向。global_time另为1活动参数参考，不称r4等维方法。共同时间正交压力严格保持零滞后空间协方差与联合RMS，但不保证任意filter-bank-CSP不变。

**头容量与可辨识性。** 身份代理m=m₀+Jθ+Za。消去未正则头后的空间数据信息为JᵀW¹ᐟ²(I−P)W¹ᐟ²J；其秩至多min(r,n−rankZ)。Z满行秩时，支持空间作用可全被头吸收，ridge导致的正定并不代表数据识别了空间物理变化。budget头取支持源特征PCA，h=max(0,min(16,n_valid−1−r))，另受实际秩限制；保留原p+1正则分母，避免容量变化混入先验强度变化。必要秩条件不足时依先验正常返回，不因B4秩不足中断。

**源相对风险。** δᵢ=CE(fθ(xᵢ),yᵢ)−CE(f₀(xᵢ),yᵢ)，比较均值CE、绝对entropic DRO、相对Rτ=τlogΣqᵢexp(δᵢ/τ)，及Eδ+Varδ/(2τ)近似。主τ=.5。共同10步Newton代理初始化后，以真实目标选择代理中心或identity，再作5步真实目标精化/回溯。若真实支持目标Rτ+ridge≤0，则同一有限支持任意重权p满足Eₚδ≤τKL(p∥q)−ridge；该界不保证未来新样本或噪声标签上的无负迁移，不是运行门槛。

时空谱、反对称指数、head/body分工、有效信息和相对风险均有经典及2026年已发表近邻。候选增量是付费小前缀下的具体模型与机制拆解，不声称“无人研究”或已有显著提升。正式文献与未纳入路线见独立源码分支文档。

## 精确任务、压力条件与统计

| 家族 | 数量 | 内容 |
|---|---:|---|
| 字典 | 15 | static_ce、st_ce、static_rel、st_rel、st_rel_budget，各3种子 |
| 内部核心 | 72 | 共同字典2模型×4目标×2头×3种子48；另外4字典自身与静态CE交叉24 |
| 内部敏感性 | 72 | 温度、固定头维/随机头、头ridge、半径、时间边界、split和变化方差近似 |
| 内部参考 | 21 | 仅头Newton、EA、R6 UQ、协方差-only、global_time、rotation-only、temporal-only，各3种子 |
| 重复Pan外部 | 93 | 核心72及全部参考21 |
| **主包总计** | **273** | **内部180＋Pan93，仅15项训练字典** |
| 可选Weibo | 另60 | 独立手动阶段，不自动并入主包 |

字典最多1000outer更新、每步4episode、50步元验证、10步双代断点。核心/参考使用五预算，敏感性为B20。273是正式逻辑任务数，不是从头训练273次EEGNet。

另外108个B20人工条件嵌入既有任务：共同时间正交正负方向60，支持标签10%/20%扰动48。它们增加评价数据和耗时，不增加字典数或独立参与者。白化/原CAR能量、实际输入位移、带外能量、边缘效应与数据/ridge信息分别记录。

主分析为Pan B20 clean的12个预定方法对×NLL/BA＝24项统一Holm，以14位参与者为单位，先在人内聚合两会话和三种子。内部、敏感性及人工压力为描述性探索；NLL有益不能冒称BA显著提升。

## 数据阶段：复用Pan，不是新首次盲测

本轮使用第六轮已经准备的同一Pan2023缓存：14人、28真实会话、3360条LR、560个B20支持位置和2800条固定查询。28个原MAT的下载SHA及预处理核对为R6已有取数证据，不称R7又取得一批独立参与者。R7沿用缓存manifest SHA256 `d83ba1e69a3e32dac863934accbd99fcbfdc64e11d769706025e8817611ad2cc`。

原数据缺少可验证的物理采集时间/区段，只按提供者会话MAT发布trial顺序作前缀评价。Pan属于重复外部探索，内部队列也经历过先前轮次；单轮Holm不能消除多轮观察和选题带来的选择影响。程序与方法在当前评价前固定，不能据此改称全新盲确认。

## 学校资源、恢复与GitHub同步

用户最新资源为`hpc_gpu`、A100-PCIe-40GB、一个节点一个Slurm作业内部3GPU并行，总36CPU、总128G共享主机内存，每槽12CPU亲和组。入口规划`--partition hpc_gpu --gpus 3 --cpus 12 --memory-total 128G`；CPU参数每槽计，内存参数全作业计。不是R6的hpc_gpu1/164G，也不是三个独立提交作业或DDP。

输出独立保存于`/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r7/`。双代checkpoint保存优化器与随机状态，评价逐会话保存，完整匹配结果复用，单项异常不取消其他独立任务。无累计12小时上限；集群单次作业时限仍由断点/续交处理。学校启动不重复全量test/pilot/全树SHA扫描。

自动监控设计每180秒尝试向`paper3-r7-progress:monitoring/paper3-round7/status.json`上传汇总，网络失败独立重试，不暂停训练。完整checkpoint、逐会话预测和CSV在学校共享目录；汇总回传不等于完整模型异地备份。当前未向学校提交作业，也未启动R7监控，故不能把规划路径当作已返回数据。

## 验证与线上交付记录

526项本地软件测试全部通过，0失败/错误/跳过，59.287秒。环境版本为Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6；执行主机无CUDA设备，不等同学校旧内核/驱动或三GPU现场验证。覆盖时空数值与导数、head容量/先验、真实风险、元梯度、R7生产worker、逐会话恢复、资源分配、下载与24项统计流程。

另22项交付定向测试和12项固定发布bootstrap测试通过；它们与全套测试有重叠，不相加成560个独立测试。183个发布Git blob的身份已通过远程tree SHA逐一核对。实际下载的约24.75MB wrapper与构建SHA一致，raw从1MiB保留分片继续请求收到HTTP206，API完整下载收到HTTP200，正确缓存再次执行在禁止联网时复用成功。

最终线上wrapper在禁socket/DNS及禁止子进程时，实际安装并通过同进程入口prepare273项（93Pan），所有28会话×5预算＝140个物化视图与原缓存逐位一致；四个入口隔离模式help验证通过。线上小bootstrap本身也已从raw/API实际下载核SHA，并实际调用已缓存wrapper完成安装/273项prepare。计算节点不临时下载原MAT或安装SciPy/h5py；分片、备用地址和重试在学校入口保留。以上为当前执行环境的下载/CPU交付核验，学校网络及三GPU仍未现场实测。

固定源码提交为`e804bf9febcc6e95d4e230f3a9869073440f0d38`；小bootstrap提交为`0ecb1bbac0e6365625f401e71604e4ddec6baf41`。wrapper SHA256为`72dc52eefc18c343c08f794f8f0cbc77a541df077c644376b393189973d1ae0d`；bootstrap SHA256为`61566627c1c155ef40d42c4c71089dc09355a46121837b7d31731e8aa722d092`。启动说明与proof汇总已发布，固定说明页提交为`d2bf51048f7e912990e0c936fabab90d1abe07cc`。

源码、设计及交付说明：

- [完整设计与程序目录](https://github.com/jackzhu119/research3/tree/paper3-r7-progress/experiments/followup_20261024_r7)
- [启动指令交付页](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/experiments/followup_20261024_r7/online/LAUNCH_ZH.md)
- [数学设计](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/experiments/followup_20261024_r7/DESIGN_ZH.md) · [文献](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/experiments/followup_20261024_r7/LITERATURE.md) · [候选路线](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/experiments/followup_20261024_r7/RESEARCH_OPTIONS_ZH.md)
- [精确任务矩阵](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/experiments/followup_20261024_r7/EXPERIMENT_MATRIX.json) · [本地软件验证记录](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/experiments/followup_20261024_r7/VALIDATION.json)

- [R7实际线上下载证据](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/experiments/followup_20261024_r7/online/download-proof.json) · [R7发布包禁网安装/读取证据](https://github.com/jackzhu119/research3/blob/paper3-r7-progress/experiments/followup_20261024_r7/online/published-offline-proof.json)

源码与包的上述核验不包含提交学校作业，不代表算法显著提升，也不保证学校资源/网络始终可用。
