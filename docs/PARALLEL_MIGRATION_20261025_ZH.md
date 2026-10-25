> 2026-10-25更新：用户已提交九份held替代作业。学校`NumNodes=1-1`触发旧解析器误判，修复已发布，尚待执行。[原九个作业接管修复指令](SLURM_NODE_RANGE_FIX_20261025_ZH.md)。

# 2026-10-25：第一、五、六轮三作业续跑交接

本页记录最新授权目标及交接实现，区别于2026-10-24的单作业三卡迁移。**本次学校作业切换尚未由本页执行或验证**；新作业ID、排队/运行状态和真实outer/session计数，应以用户执行启动命令后的回执为准。

## 1. 最新资源

| 轮次 | 分区 / GPU | Slurm作业形式 | 各作业CPU | 各作业内存 | 每轮合计 |
|---|---|---|---:|---|---|
| 第一轮 | `hpc_gpu` / A100 PCIe 40GB | 3个独立单GPU作业，各自1节点 | 12 | 43G、43G、42G | 3GPU / 36CPU / 128G |
| 第五轮 | `hpc_gpu1` / A100 SXM4 80GB | 3个独立单GPU作业，各自1节点 | 12 | 64G、64G、64G | 3GPU / 36CPU / 192G |
| 第六轮 | `hpc_gpu1` / A100 SXM4 80GB | 3个独立单GPU作业，各自1节点 | 12 | 64G、64G、64G | 3GPU / 36CPU / 192G |

这是每轮的资源合计，不是三个轮次共享一组GPU。三个独立作业可以分配到不同物理节点；没有承诺必须同节点。`hpc_gpu`、`hpc_gpu1`是分区名，不是节点名。第五、六轮排除历史启动异常节点`gpu3`。

## 2. 原结果与原科学计划保持

第一轮原作业为`770043`，最后学校只读证据为主线运行、元筛选完成32项；后续交接工具现场重新读取真实完成回执和当前断点，不把32当成固定进度。`runs/source_training_v2`、`runs/source_classical_v2`、`runs/meta`、`runs/training_campaign_v1`及原内部检查目录继续使用。已完成源模型、经典模型、MetaJob和开发结果不复制成新实验再训练。

第一轮保留45格元筛选、10新增元确认、118开发筛选和28新增开发确认的原依赖；还保留原direct入口的两条五源扩展`BASE4_PLUS_YANG`、`BASE4_PLUS_DREYER`。独立MetaJob可三卡并行；开发校准和计时工作固定slot0；后期两扩展arm分配slot1/2。选模、冻结及一次内部检查按原依赖串行执行，因此并非所有阶段都能三卡满负载。

第五、六轮仍为原105/198项任务，输出目录分别是`runs/followup_20261024_r5`与`runs/followup_20261024_r6`。任务顺序、种子、数据划分、评价预算和算法参数未因调度迁移改变。断点与逐会话预测复用原位置；第六轮继续使用已经准备好的Pan2023缓存。

## 3. 交接顺序

1. 下载并验证新启动包，读取现场计划、完成回执、旧作业与断点；准备操作目录。
2. 先提交三份带hold的新单卡作业，查询真实Slurm队列及资源，确认是本轮、三份不同作业、正确分区/GPU数/CPU和主机内存。
3. 停用本轮旧自动重提管理器。第一轮只终止精确匹配原`launch.py --execute --monitor --root`的login侧监控，不把训练PID当监控PID。
4. 第一个新作业尚未开始训练时，向原运行作业请求原生checkpoint退出；旧作业若排队则只取消本轮精确归属的旧排队项。等待旧进程组、原生锁和队列/accounting交接证据完成。
5. 保存交接审计，再放行本轮三个新作业。第一轮边界记录为`runs/training_campaign_parallel_v1/migration-progress-snapshot.json`；第五、六轮保留旧STOP墓碑，防止旧管理器再次提交原单节点三卡配置。
6. 新作业从原成功产物或原生最新checkpoint继续；节点中断、单次Slurm时限到期时，保存后由自己的调度器续交。没有恢复累计12小时总上限，科学更新数仍按原计划。

下载、提交确认、开始执行和科学完成是不同状态。上面的顺序不等于本页已经取消或提交任何学校作业；实际回执才是执行证据。第二、三、七轮结果上传器和第四轮作业不属于本次交接操作对象。

## 4. 在线入口

代码分支：`paper3-parallel-migrations-20261025`；目录：`operations/parallel_20261025/`。

[三个可直接复制的正式启动指令](https://github.com/jackzhu119/research3/blob/60dc5c81e01098e11ff18b4cb2bd0ac0b924569c/operations/parallel_20261025/online/LAUNCH_ZH.md)；固定科学续跑源码版本为`6315d2fa96b0b541009a7492b92fcd9841f91b2a`。181项本地针对性测试与接管测试及实际发布文件的下载/安装/prepare检查通过；学校GPU交接等待用户执行。

[标准库在线启动器](https://github.com/jackzhu119/research3/blob/paper3-parallel-migrations-20261025/operations/parallel_20261025/online/START-PAPER3-PARALLEL.py) 的三个入口分别为：

```bash
python -I -S -B -u START-PAPER3-PARALLEL.py --round 1 --execute
python -I -S -B -u START-PAPER3-PARALLEL.py --round 5 --execute
python -I -S -B -u START-PAPER3-PARALLEL.py --round 6 --execute
```

正式交付的命令还包含固定源码版本、SHA256校验、raw/API备用下载入口和本地缓存复用；以上只说明解析器参数，不代替最后发布的下载命令。启动沿用学校已有Python/Torch环境，不重新执行全量pytest、pilot或部署数据树SHA核验，不在GPU节点重新安装依赖或下载原始数据。软件验证与真实学校GPU运行证据分别记录，不能把本地测试通过当成学校迁移已成功。

## 5. GitHub分轮回传

| 轮次 | 新轻量状态 | 保留内容 |
|---|---|---|
| 第一轮 | `paper3-r1-progress:monitoring/paper3-round1/status.json` | 原campaign回执、主线与两扩展arm分栏 |
| 第五轮 | `paper3-r5-progress:monitoring/paper3-round5/three-jobs-status.json` | 原`status.json`及其Git历史 |
| 第六轮 | `paper3-r6-progress:monitoring/paper3-round6/three-jobs-status.json` | 原`status.json`及其Git历史 |

监控读取已有小JSON和实际队列，回传三份作业及outer/session计数，不重新扫描逐试次预测生成数小时统计。网络或GitHub暂时失败只影响回传，不控制正在训练的kernel。

已有报告、CSV和完成任务摘要按轮写入统一`paper3-results:results/paper3/round-01/`、`round-05/`、`round-06/`的自动汇总分区。**自动汇总不是完整原始输出备份**：完整checkpoint和逐试次预测仍在学校原目录，若要完整异地归档需使用完整结果上传流程。第二、三、七轮的`previous/`历史快照、`final/`完整输出和`LATEST.json`入口保留。

## 6. 历史与现场状态边界

- 第一轮旧资源：`hpc_gpu1`、SXM4 80GB、单GPU、24CPU/128G；旧job`770043`。
- 第五、六轮旧资源：`hpc_gpu`、PCIe 40GB、单节点单作业三GPU、36CPU/128G；旧job`772012`/`772013`。
- 更早第五、六轮`hpc_gpu1`、36CPU/164G单作业三卡记录同样保留历史，不能与本次192G三独立作业配置混为一谈。
- 80GB与40GB硬件差异通过单独迁移审计记录，原identity/参数/RNG计数不重置，不把80GB历史计时伪装成40GB新基准，也不宣称跨硬件逐位恢复已经现场验证。
- 初次发布时学校新作业提交数、旧作业取消数均为0；随后用户已提交九份held作业。当前应以本页上方新修复记录和真实交接回执为准，未声称修复已在学校执行。

[返回实验总档案](experiments/README_ZH.md) · [按轮结果归档](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3)
