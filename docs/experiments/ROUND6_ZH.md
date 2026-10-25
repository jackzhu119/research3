# 第六轮：曲率形状预测平均与新Pan双会话外部验证

<!-- PAPER3_PARALLEL_TARGET_20261025 -->
## 2026-10-25最新资源目标（现场尚未执行）

本次最新目标为 `hpc_gpu1` 的A100 SXM4 80GB：三个独立单卡作业，各12CPU和64G主机内存，合计36CPU/192G。旧`772013`在`hpc_gpu`排队。该目标取代2026-10-24的单节点单作业三卡迁移目标；旧配置和回传继续作为历史保留。

保留全部198项原任务声明、已完成输出、逐会话预测和checkpoint；继续使用已有Pan2023缓存，不在GPU节点下载原始MAT或安装新环境。先确认三份新held作业，再只处理本轮旧作业和旧管理器；旧STOP墓碑保留，新作业排除历史启动异常节点`gpu3`。

新监控写入`paper3-r6-progress:monitoring/paper3-round6/three-jobs-status.json`；旧`status.json`保留历史。按轮自动汇总进入`paper3-results`，不等同于完整原始输出备份。

[三作业交接、在线入口与回传配置](../PARALLEL_MIGRATION_20261025_ZH.md)。本段之后的旧资源和进度记录保留历史。
<!-- /PAPER3_PARALLEL_TARGET_20261025 -->


[返回七轮总档案](README_ZH.md)。交付记录：2026-10-24北京时间20:36；未向学校提交第六轮作业，无本轮准确率结果。

## 2026-10-24追加：迁移至hpc_gpu（包已发布核验，学校未执行）

用户最新目标为**hpc_gpu、A100-PCIe-40GB、一个节点一个Slurm作业内部3GPU，总36CPU、128G共享主机内存**，每槽12CPU。原hpc_gpu1、36CPU、164G共享主机内存保留为历史发布/提交配置，不代表迁移已在学校生效。新提交应请求`--partition=hpc_gpu --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=36 --mem=128G`；使用迁移入口处理原CLI语义，128G不乘三。

本次只迁移资源与提交封装，不改变本轮算法、数据划分、种子、198项矩阵或原输出目录；完成项和匹配checkpoint/逐会话结果继续复用。约23.4MB已准备Pan缓存随迁移包交付，计算节点不重新下载原MAT或安装环境。

独立操作分支：`paper3-hpc-gpu-migrations-20261024`，目录`operations/hpc_gpu_migrations_20261024/round6`；[迁移交付目录](https://github.com/jackzhu119/research3/tree/paper3-hpc-gpu-migrations-20261024/operations/hpc_gpu_migrations_20261024/round6)已真实发布。固定源码提交`79eabe6c456aa443e18c35c53b6c39a33d27a639`，小bootstrap提交`1d3e4860fc8a138e7be2717dae75ed55a0657e47`。本轮包24,659,326bytes，SHA256 `d993c7408b1510a061c715a32ee45002903534b148e9adfe8e18798e26dc053e`。

本轮347项本地迁移全套软件测试通过；实际raw保留分片续传收到HTTP206，API完整下载HTTP200及正确缓存禁网复用通过。下载包在禁socket/DNS和子进程场景实际安装、prepare198项，保留原计划字节及result/checkpoint/session的字节和mtime；额外读取28会话×5预算140个Pan视图。线上小bootstrap本身已从raw/API下载核SHA，并实际调用本轮缓存包安装/prepare。以上为本地CPU与交付网络验证，不是学校GPU/学校网络现场实测。

[三轮迁移启动说明](https://github.com/jackzhu119/research3/blob/c7c903fb126e3e60a7e69556bea9f7ac8ce7e489/operations/hpc_gpu_migrations_20261024/online/LAUNCH_ZH.md)交付对应`MIGRATE --submit`；用户执行后才精准处理本轮同标记旧排队，已进入运行的作业不当作排队直接取消。截至本次记录，学校取消/提交均为0，没有迁移后实际GPU训练或结果数据；启动说明固定提交为`c7c903fb126e3e60a7e69556bea9f7ac8ce7e489`；[本轮固定启动shell](https://github.com/jackzhu119/research3/blob/c7c903fb126e3e60a7e69556bea9f7ac8ce7e489/operations/hpc_gpu_migrations_20261024/online/RUN-ROUND6.sh)已实际下载并通过shell安装/prepare，未执行`--submit`。

本轮继续单独回传`paper3-r6-progress:monitoring/paper3-round6/status.json`，不写到其他轮或操作分支。下文旧资源、作业号与状态均保留原时点语境，不能据迁移计划重写为已经切换或已经训练。

## 科研问题与数学改动

继续冻结EEGNet、固定rank4空间字典、目标前20个获取位置与同一未来查询。本轮检验单点空间适应是否丢掉小支持下可用的局部不确定性；不根据前五轮部分结果选择几何图。

共同中心为固定身份切线代理上的10步Newton/Schur。对支持CE加ridge曲率分块，消去分类头竞争后取S=A−CD⁻¹Cᵀ；Σθ=τS⁻¹、τ=κρ²/tr(S⁻¹)，固定投影前迹为κρ²。κ=.25、ρ=.15，求解器damping不混入科学曲率。

rank4用9个正权空间点，投影至原球；每点头均值作负相关补偿，实际空间变换与冻结网络前向，再GH9积分头余项。不增加目标可学习参数。这是曲率形状预测平均，不是精确后验或最优性保证。

同中心单点控制分离平均与投影中心收缩；同迹各向同性/对角控制检验形状；空间点与头点独立乘积配对保留完整边际，只去相关。仅头/真实中心局部线性Gaussian平滑保持硬标签，是区分概率改善与BA改善的确定性空效应。有限点方向依赖、投影后实际矩和真实预测耗时均记录。

## 精确矩阵

| 家族 | 数量 | 内容 |
|---|---:|---|
| 字典 | 6 | 点预测/联合预测平均目标各3种子 |
| 内部核心 | 60 | 两字典×10预测器×3种子 |
| 敏感性 | 48 | 迹预算18、GH阶6、中心权6、固定轴旋转6、半径12 |
| 内部参考 | 12 | 头Newton、EA、第五轮真实图精化、第四轮OOF堆叠各3 |
| 新Pan外部 | 72 | 两字典×10预测器×3种子及12参考 |
| **主包总计** | **198** | **只有6项训练字典，复用既有源EEGNet** |
| 可选Weibo | 另36 | 独立手动阶段 |

核心/参考报告B4/8/12/16/20，敏感性只B20；短预算共用同一个B20查询。新Pan评价在依赖满足后优先调度，内部队列仍为探索性复验。

## 新增数据：已实际完成取数和预处理

Pan2023为14人、每人两份真实会话、3360条二类左右手MI。28个原MAT共736,301,987bytes，已从保留原字节的公开NEMAR镜像完整下载，逐一SHA核验；原数据CC0许可已确认。

全部3360条按原clinic11/CAR/8–30Hz/128Hz/固定窗口合同重做，与缓存float32逐位一致；每会话原预处理函数float64比较也一致。B20共560支持位置、2800固定查询。原文件没有可验证的物理时间戳/区段，保留D1/D2身份，准确称“原会话MAT发布trial前缀外部评价”，不伪造physical_run或前瞻采集顺序。

统计单位为14人，会话和三种子在人内聚合。8项预定方法比较×NLL/BA合计16项统一Holm；NLL改善不能当成BA显著提升。外部方法在查看Pan预测效果前固定，不用外部query调参或选字典。

## 原发布资源、保存与回传（历史配置）

`hpc_gpu1`，一个节点、一个Slurm作业、三张GPU，总36CPU与164G共享主机内存；每槽12CPU亲和组。入口使用`--memory-total 164G`，不乘三。

安装`~/paper3-round6/3d97b48654f44579eb49/`；输出`/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r6`。双代checkpoint、逐会话保存、完成项复用与单任务失败隔离；无累计12小时总限制，不重复启动全量test/pilot/数据树扫描。既有五轮作业独立保留。

每180秒尝试同步`paper3-r6-progress:monitoring/paper3-round6/status.json`；提交监控后生成。网络上传失败不停止训练，完整模型、概率/log概率和CSV在学校本地。

## 实际交付验证与启动

349项本地软件测试全部通过，环境匹配Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6；无学校GPU现场运行。单文件内含23.4MB已处理Pan缓存，计算节点不下载原MAT、无需访问原数据网站或安装SciPy/h5py。

约24.66MB固定包的线上raw真实HTTP206续传和API完整下载均已实际SHA通过；下载后的包在禁止socket/DNS联网时完成安装、实际入口198项prepare及全部28会话五预算读取。小bootstrap13KB，保留分片持续重试，正确缓存再次执行无需网络。

- [第六轮完整程序与设计](https://github.com/jackzhu119/research3/tree/paper3-r6-progress/experiments/followup_20261024_r6)
- [固定线上执行代码](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/online/LAUNCH_ZH.md)
- [详细数学推导](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/DESIGN_ZH.md) · [文献](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/LITERATURE.md) · [统计](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/STATISTICS.md)
- [新数据说明](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/PAN2023_DATA_NOTES.md) · [软件验证](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/VALIDATION.json)
- [线上下载证据](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/online/download-proof.json) · [发布包禁网读取证据](https://github.com/jackzhu119/research3/blob/paper3-r6-progress/experiments/followup_20261024_r6/online/published-offline-proof.json)

固定主源码提交`87a5b5c5f69080bd76a2c22910eaca242895a746`；包SHA256 `d1c506c0c2a684cae5017e7b858d7858f092b8a7f949da4d06a7a59a26ee6d82`。bootstrap固定提交`d5973a6fa7aa3d6a1bc3892b209eaa908fae8c0e`。公开历史数据、软件恒等式和CPU验证不等于识别提升；完整预定外部结果才能回答科研问题。
