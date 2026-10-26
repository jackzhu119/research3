# R10 优化续跑：工程交付

本包是独立的探索性协议修订。它读取已完成的原15项机制证据，采用参与者配对EA增益进行开发选型，随后准备固定的165项探索任务。原研究包、15项结果及其收据保持原样。该修订不能把原规则的STOP诊断改写成PASS，也不能作为独立确认结果。

## 一行入口及明确动作

自包含入口为 `delivery/START-PAPER3-R10-OPTIMIZED.py`。下载时按交付记录验证入口SHA256；安装无需pip、Git克隆或训练数据下载。

- 默认或 `--install`：仅安装到 `~/paper3-round10-optimized/<payload SHA前20位>`。
- `--validate`：显式执行有限CPU软件自检。
- `--continue`：读取原证据、写新修订与计划、准备运行目录；不提交作业。
- `--continue --submit`：完成上述准备后，启动脱离终端的控制器、四个单卡Slurm作业，以及互相独立的报告/A/B/监控进程。终端可以关闭。
- `--start`：`--continue` 的兼容别名。
- `--status --plan PATH`：查看当前科学任务、调度作业、错误和上传状态。
- `--resume --plan PATH --submit`：从新计划断点继续；若操作者修复了已隔离问题，可显式追加 `-- --retry-blocked`。
- `--export --plan PATH`：单独恢复报告与双通道上传，不重新运行科学任务。
- `--gpu-precheck --registry PATH`：只有显式调用才执行有限GPU预检；本次交付未执行。

入口支持 `--python PATH` 指向学校已有Python环境，默认使用 `/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python`。`--root` 默认学校Paper3根目录。`--source-plan` 可明确指定原计划位置；来源的固定SHA与recipe仍须匹配已核实的原机制计划。`--output` 可选择新的续跑目录，不能覆盖原机制输出。

安装后也可直接执行：

```bash
"$P" -I -S -B -u "$PACKAGE/START.py" continue --root /public/home/zhuziyuan/projects/paper3 --submit
```

其中 `$P` 是已有Python绝对路径，`$PACKAGE` 是安装回执中记录的新包路径。正式提交只发生在显式 `--submit` 下。

## 旧证据如何安全复用

`paper3r10/continuation.py` 不用新执行代码的 `load_plan` 去加载旧包。它单独核验原计划文件SHA、完整recipe、15份result/completion任务身份与result SHA、原registry、精简会话CSV、报告及准确任务覆盖。旧包物理源码无需重新执行或重新哈希，预测大文件与EEG不做重复扫描。

选型仅读取原meta-validation数据；目标holdout、Pan和Weibo结果不参与重新排序。原证据的精简副本和验证记录保存于新输出的 `reports/legacy_evidence/`，修订记录为 `reports/development_amendment.json`，包括原/新代码身份、来源计划/报告SHA、修订规则及原规则重算诊断。

原seed20261018 reference直接按旧seal的 `reference.json` 和 `reference.npz` 精确字节导入新输出，并封存新recipe的结果。这个新元数据节点不重新打开源模型、catalog或重建几何。原14项机制评估不会进入新执行列表；原15项科学工作无需重算。

默认新目录为 `runs/r10_formal_exploratory_optimized/`，计划为其中的 `plan.json`。重复调用读取同一新计划，不重新选型。局部准备失败后可以直接重试；已有修订科学字段一致时保留最初的完整字节，不因时间戳变化要求删除目录。

## 正式探索任务与调度

165项任务包括3个reference节点及18种方法×3个种子×3个探索队列。预算曲线保持4、8、12、16、20，使用原B20固定未来query集合。资源为 **4个作业，每个1 GPU、16 CPU、64G，分区hpc_gpu**；最大时间沿用5天3小时。四个槽位共同领取一个固定任务队列。

每个分配内科学进程长期复用，避免每个任务重新导入Torch；源模型/catalog缓存保持冻结、角色隔离和身份绑定。CuBLAS确定性环境在数值导入前固定，累计和的CPU路径及dtype处理沿用已测试修复。SIGUSR1、时间片结束、运行目录STOP请求通过检查点继续；普通任务错误只隔离该任务，错误CUDA上下文只重启对应科学子进程。

每任务有限失败重试后隔离；共享启动失败和结构性sbatch配置错误有持久上限，避免空转提交。调度回执不明确时先观察原任务标记，不重复创建作业。本包没有取消旧作业的路径。

历史强基线若缺少已注册artifact，标记该任务不可用，保留其身份及错误原因，其余任务继续。不会用随机权重、弱基线或错误模型替代。缺基线意味着对应比较和整套结果报告保持PARTIAL，不能宣称完整165比较已经验证。

## 双通道与状态

通道A增量发布完整参与者/会话CSV、统计、配置、修订及证据；通道B独立归档预测、拟合状态和完整验证轨迹，保存SHA及远端回执。网络失败不改变训练状态或触发重训。

结果前缀仍是 `results/paper3/round-10/stages/formal_exploratory/<新recipe前16位>/`，新旧recipe分离。R10原mechanism索引保持，stage INDEX采用非force CAS合并。操作状态在 `paper3-r10-progress` 分支对应新stage/recipe目录；科学完成、报告完成及A/B上传完成分别记录，旧上传快照不能代替当前结果指针。

本次软件验证只使用CPU、合成/元数据夹具和原证据读取；未占用学校GPU，未提交或取消学校作业。正式EEG结果需要用户执行显式提交命令后才能产生。
