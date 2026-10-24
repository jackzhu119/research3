# 论文3配置交接与共享聊天变更归档 · 2026-10-24

归档时间：2026-10-24T16:12:17+08:00。本文自包含记录分享聊天《排查Slurm提交限制》中的论文3配置变更；查阅和交接不依赖原聊天继续存在。[机器可读概览](EXPERIMENT_STATUS.json)与[原始关键证据摘录](records/PAPER3_SHARED_CHAT_EVIDENCE_20261024.json)一并保存。

原分享地址仅作来源标记：https://chatgpt.com/share/6adc6269-a39c-83e9-80f1-1731466f0762 。本记录不是完整聊天/截图备份，也不是学校控制器源文件或checkpoint的副本。关键配置、证据摘录和执行约定已写入仓库正文；聊天中临时下载附件不作为未来唯一恢复入口。

## 1. 最终采用的部署配置

| 项目 | 第一轮 | 第二轮扩展V2 |
|---|---|---|
| 正确仓库 | `jackzhu119/research3` | 同左；曾误查的其他论文仓库结论作废 |
| 平台 | 学校Slurm HPC | 学校Slurm HPC；与第一轮并行 |
| 分区 | `hpc_gpu1` | **`hpc_gpu`**，名称没有末尾的1 |
| 已记录节点 | `gpu4` | `gpu1`与`gpu2`，不要求全部任务同节点 |
| GPU | 已有记录A100 SXM4 80GB | 用户接受/迁移记录中的A100 PCIe 40GB |
| 并行方式 | 1个原始作业 | **最多3个独立单GPU作业，滚动补交** |
| 每作业申请 | 24CPU、128G主机内存、1GPU | **12CPU、64G主机内存、1GPU** |
| 最近作业号 | `770043` | `771066_12` / `771065_13` / `771064_14`；仅是该次观测身份 |
| 训练代码包 | 原主线及原运行时修复目录 | `/public/home/zhuziyuan/paper3-followup/0e55dba2a78b43464e8c/` |
| 共享项目 | `/public/home/zhuziyuan/projects/paper3` | 同左；仅只读复用源模型、参考和缓存 |
| 结果根目录 | `runs/meta/`、`runs/training_campaign_v1/`、`runs/source_training_v2/` | `runs/followup_20261023_v2/` |

用户粘贴的最新`squeue`/`scontrol`直接证明第二轮分区、节点、GPU数量、CPU和内存申请及当时RUNNING状态；同一份最新队列确认第一轮的运行状态、节点及24CPU，第一轮1GPU/128G来自此前同作业scontrol记录。该输出未打印GPU型号；40GB/80GB硬件描述来源于此前截图解读、迁移选择和既有归档，不能说本次又执行了`nvidia-smi`。三卡新实验合计36CPU/192G；加第一轮合计4GPU/60CPU/320G。它们是资源分配量，不是实时利用率或显存峰值。

`gpu1`、`gpu2`、`gpu4`是节点；`hpc_gpu`、`hpc_gpu1`是分区。分区汇总有空闲卡不保证单节点同时满足GPU/CPU/内存，也不消除优先级或关联组配额约束。历史记录中`hpc_gpu`节点为56CPU、5张40GB A100、约251GiB主机内存；这来自聊天对节点截图的解读，不扩展成所有节点的规格保证。

## 2. 本次聊天到底改了什么

本次可追踪的修改集中于**提交、排队、迁移、资源和监控**。聊天未给出算法、数据划分、任务编号或训练预算被改写的证据；原有66项数值实验继续使用原计划。下表是时序记录，不是需要重新执行的操作清单。

| 阶段 | 方案/现象 | 最终地位 |
|---|---|---|
| 原始云端入口 | 一次提交66个数组元素，`%3`或`%2`限制运行并发；触发`QOSMaxSubmitJobPerUserLimit` | 已被滚动提交取代；减小百分号并发不能减少整批提交计数 |
| 最初补交 | `paper3-submit-guard/auto_submit.py`，最多两个单GPU任务，24CPU/128G | 历史；曾出现文件缺失，不能把只有预检/控制器PID当成已提交 |
| 迁区与退回 | 尝试3×16CPU/128G并迁往`hpc_gpu`，又退回`hpc_gpu1` | 历史选择，不是当前12CPU/64G配置 |
| 单作业四卡80GB | `hpc_gpu`、4GPU/48CPU/512G；目标型号和单节点主机内存不匹配 | 未成为最终运行方式 |
| 单作业四卡40GB | 改为4GPU/48CPU/200G；记录作业`770753`在`hpc_gpu`因`AssocGrpGRES`等待 | 历史排队方案，已由三单卡方案取代 |
| 三单卡80GB | `hpc_gpu1`，3个单卡作业、每项16CPU；`770766_12`等因`Priority`排队 | 中间恢复方案；不是最后部署 |
| 最终40GB三单卡 | 用户接受40GB和降低内存；`hpc_gpu`，每项12CPU/64G，最多三项滚动补交 | **最新用户终端记录已确认3项RUNNING** |
| 迁移启动参数修复 | `SWITCH-PAPER3-40G.sh`曾因空参数报`Supported flags: --check`；修复末尾参数从`"${1:-}"`改为`"$@"` | 后续RUNNING输出确认切换已走通；不在这里重新修补服务器 |

`Priority`是优先级等待；`AssocGrpGRES`是关联组GRES配额约束；`QOSMaxSubmitJobPerUserLimit`是提交数量策略拒绝；迁移中的`JobHeldUser`是暂挂；几者不等同于训练数值错误。旧排队作业0秒COMPLETED也不能当成科学任务已完成。

聊天曾引用“最多提交10个、同时运行4个”等QoS说法，但保留下来的用户文字未包含相应完整`sacctmgr`数值表。本记录保留其历史来源，不把它写成永久配额；“三作业”是当前控制器并发策略，不等于已证实学校MaxJobs=3。

## 3. 科学实验保持的内容

- 任务总数66：36核心元训练（12方法×3种子）＋24参数敏感性元训练（P/dual×4配置×3种子）＋6参考任务。
- 核心方法：`P, global, scalar, dual, PCT, softPCT, temporal, dual_temporal, softPCT_mean, runmean, symonly, skewonly`。
- 三个种子：20261018、20261019、20261020。核心r=4、ρ=.15；敏感性为(r2,.15)、(r8,.15)、(r4,.05)、(r4,.35)。
- 元训练每fit1000外层步，每步4个episode、内层10步；每50步元验证。参考任务不做1000步元训练。
- 40名元训练参与者、11名元验证参与者及11名内部留出参与者；学校现场记录元训练396个session、元验证97个session。数据/源模型与划分沿用本轮冻结输入。
- B4/8/12/16/20标签预算曲线、固定六种误标/空间漂移诊断、预测概率及负翻转指标继续按原设计评价。敏感性任务仅B20；这些评价不等于重新训练一遍元模型。
- 核心36项齐备后才产生固定10主比较的Holm推断；当前部分结果不能写成最终显著提升或新的临床/外部证据。

完整数学与选择规格仍以[版本化设计](https://github.com/jackzhu119/research3/blob/paper3-v2-progress/experiments/followup_20261023/DESIGN_ZH.md)和对应代码为准。本次归档没有调整学习率、种子、方法、批量、划分或比较终点。三张GPU处理三个独立实验，没有把单次episode批量扩大三倍。

## 4. 学校本地路径和接续约定

| 用途 | 已保存位置 |
|---|---|
| Python | `/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python` |
| 第二轮原代码包 | `/public/home/zhuziyuan/paper3-followup/0e55dba2a78b43464e8c/` |
| 第二轮任务表 | `/public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/plan.json` |
| 最终迁移shell历史入口 | `/public/home/zhuziyuan/SWITCH-PAPER3-40G.sh`；修复前备份记录为`.before-fix`；不是本次要执行的命令 |
| 最终迁移工具安装目录 | `/public/home/zhuziyuan/paper3-submit-three40/` |
| 最终切换日志 | `/public/home/zhuziyuan/paper3-submit-three40/switch.log` |
| 历史三作业安装目录 | `/public/home/zhuziyuan/paper3-submit-three16/`，其中`install.log`曾由用户列出 |
| 历史三作业控制日志；最终40GB控制日志路径未独立核实 | `/public/home/zhuziyuan/projects/paper3/runs/followup_20261023_v2/_threejobs16/controller.log` |
| 每项训练日志 | `runs/followup_20261023_v2/logs/train-<jobid>_<taskid>.out`及`.err` |
| 每项结果/进度 | `runs/followup_20261023_v2/<task-key>/result.json`、`progress.json` |
| 每项断点 | 同一task目录的`latest.json`和两代checkpoint |
| 第一轮主日志 | `.paper3-hpc/logs/train-770043.out`、`.paper3-hpc/jobs/770043/campaign.log` |

`_threejobs16`在中间三作业方案中使用；先前归档曾称40GB迁移后仍沿用该控制日志，本次没有直接文件证据重新核实这一点。旧目录名不能用于判断当前资源。最终迁移脚本与服务器控制器实际文件没有在本次文档归档中重新下载/执行/哈希，也未把聊天中的临时附件链接当成完整代码备份。服务器已安装的控制器、切换日志和原任务表是后续交接的定位入口。

滚动补交继续使用原任务编号、task-key和结果根目录。任务只有`result.json.status == COMPLETE`才按成功跳过；Slurm作业退出、改号或网页显示完成都不足以替代它。未完成任务读取身份匹配的原checkpoint；已完成会话预测可复用。新种子从0开始不代表旧种子重算。已在运行的控制器继续负责补交，不为刷新网页再启动旧`START.py --submit`或另一套迁移控制器。

原worker保留USR1/TERM保存机制：接收到时间窗口停止请求可保存并尝试requeue；主动终止后不应无条件复活。账号权限、强杀和存储故障仍会影响接续；调度迁移本身不证明跨GPU逐位等价或零步骤回放。没有累计12小时的人为总限额；Slurm单次申请时限仍是独立配置。

## 5. 实验状态与GitHub保存范围

第一轮用户原始训练日志在`2026-10-24T04:39:59Z`至`04:40:59Z`显示B11/r2/ρ.05/seed20261018，外层186→198→199、内层7440→7920→7960。上层`CampaignError`字段和源模型`CLASSICAL_PENDING`文本较旧，不能仅凭它们判定这条活跃轨迹已失败或经典模型仍未运行。15个源神经fit完成；整轮完成未确认。第一轮目前只有人工GitHub快照，未确认它接入持续自动回传。

第二轮已独立核实GitHub持续上传；读取的固定提交为`1effccac6b6d1ddc761b88f1d13eab300de63e3b`，其中快照时间`2026-10-24T07:45:44Z`（北京时间15:45:44）。12/66项COMPLETE，均为seed20261018的12个核心方法；P/global/scalar的第二个种子在1000/1000/950步附近训练或评价。报告`PARTIAL`、`core_complete=false`，仍有54项未完整完成。该时间是快照时间，不是此刻的学校队列查询。

同步分支`paper3-v2-progress`，路径`monitoring/paper3-followup-v2/status.json`；已有周期性`Update Paper3 follow-up progress`提交。本页、本目录JSON和资源归档都是人工记录，不覆盖这个动态文件。动态文件中的`max_parallel_gpus=1`、`cpus_per_gpu=24`仍来自旧计划元数据；不能把它当成迁移后真实资源。实际分配以用户终端的3×(1GPU/12CPU/64G)记录为准，本次仅记录这个差异，没有热修改监控程序。

最初批量提交失败时，原START没有走到启动monitor的代码；早期滚动补交脚本只在全部66项完成后调用汇总monitor。因此“设计了上传”不代表最初每一分钟都在回传。现在的连续提交才是恢复同步的证据。聊天中两次误查其他仓库所得“未上传”结论不适用于research3。

完整模型、checkpoint、逐会话预测和报告先保存于学校持久共享目录。GitHub主要同步训练进度、完成数和部分汇总；没有证据证明所有模型文件都已复制到GitHub或对象存储。本次文档提交也不是checkpoint异地备份。历史ETA只作为当时假设，不作为完成承诺或最终科研结果。

## 6. 不改变运行状态的交接检查

以下仅是可选只读定位命令，无须为了本次归档在服务器执行：

```bash
squeue -r -u "$USER" -o '%.20i %.12P %.10T %.6C %.26j %.35R'
tail -n 40 /public/home/zhuziyuan/paper3-submit-three40/switch.log
```

作业号会变化。以当前队列、任务键、原checkpoint与`result.json`对应关系定位，不以历史jobID作为永久身份。无需重复安装依赖、整套pytest、pilot或全库校验；CPU申请从24降到12并不会把PyTorch/MKL线程自动改为12，原代码`torch.set_num_threads(4)`、OMP/MKL4、OpenBLAS1仍在，具体提速效果未做受控基准。

## 7. 证据位置与归档范围

[证据摘录JSON](records/PAPER3_SHARED_CHAT_EVIDENCE_20261024.json)保存原始提交错误、用户`scontrol`/`squeue`关键输出、第一轮逐步计数、任务表保存/跳过规则和本次GitHub固定提交。提取编号仅用于归档定位，不表示ChatGPT界面可见的消息序号。资源型号、节点总规格、部分迁移原因来自聊天截图解读的地方已经另列证据等级，不把助手预测当作实测。

既有[2026-10-24资源快照](HPC_RUN_STATUS_20261024_ZH.md)及`status/HPC_ALLOCATION_SNAPSHOT_20261024.json`保留原观测时间；本次将README、两轮导航和概览JSON的“第二轮未提交/24CPU128G”过时描述统一为已运行的40GB三单卡配置。科学设计、训练代码、安装包、校验哈希和动态状态文件不随本文修改。
