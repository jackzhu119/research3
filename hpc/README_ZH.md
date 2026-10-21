# 论文 3：学校 Slurm 部署

本入口复用学校已有 Python 环境，不安装 PyTorch、Docker 或便携 Python。保留原 RunPod 入口、科研协议、数值实现、数据划分、随机种子、训练预算及科研门禁。`test / prepare / pilot / train / backup` 均由 `sbatch` 调度；不要在 `login1` 直接执行 `hpc/run_stage.py` 或原训练入口。

代码已在 Codex 进行软件测试；这不代表已连接学校、完成学校兼容验证、启动真实训练或上传真实结果。

## 共享文件系统导入延迟与失败链恢复

学校作业 767434 的真实启动回执显示：纯导入等待 35–138 秒而 CPU 只消耗约 1–3 秒，传统模型模块导入超过 180 秒；原始 pytest 尚未开始。部署入口现在先在 Slurm 计算节点准备现有环境的逐文件 SHA 校验缓存，使用真实本地挂载及空间检查，保留原 Python 和库版本，完整保留 CUDA/`.libs` 布局。最多 8 个部署读写线程，不超过已申请 CPU；科学线程仍为 1。CA、R2 客户端、配置和凭证不进入缓存。

冻结项目代码只生成节点本地字节码，原 `__file__`、源码路径及哈希不变；`-B` 仍可读取这些字节码。原始测试期限、180 秒启动探针上限和科研 fit 预算均不改。启动探针逐模块持久化并把安全摘要写入 Slurm `.out`，原 pytest 全集和数据门仍须通过。

已失败测试的 `afterok` 子作业可能永久待运行。显式恢复时使用 `hpc/recover_failed_chain.py --failed-test-job 767434 --execute`：按原始回执、账户、UID、工作目录、依赖和调度器状态核对，只取消确认从未启动的旧待运行依赖；保留全部原始回执、缓存、claim、预算和 checkpoint。独立恢复证明完整后，`hpc/start.py --execute --staged-data` 才允许提交新链；未知或曾运行的作业仍保持阻塞。固定恢复启动包可以顺序完成更新、上述恢复与新链提交；不要自行删除旧记录或使用按用户名批量取消。

Slurm 20.11 的数据库读取代码会把部分已取消、原开始时间未知的作业显示为 `Start=End`。有原始取消意图时，恢复入口对这一格式要求同一作业已经绑定并确认 `PENDING`、未知开始时间、运行时长为零且重启次数为零；仅有取消时间戳、实际运行时间及身份不符仍会阻塞。原意图和旧证明不重写，恢复可以在取消后中断的地方继续。实现依据为 [Slurm 20.11.8 原始代码](https://github.com/SchedMD/slurm/blob/slurm-20-11-8-1/src/plugins/accounting_storage/mysql/as_mysql_jobacct_process.c#L703-L707)。

两个历史下游作业均已取消而没有本地取消意图时，入口额外读取两次完整 `sacct --duplicates` 作业和 step 记录，创建独立的“未获资源分配”证明，不补造取消意图。它要求原失败 CPU 测试确实有实际资源、节点和 batch/extern 步骤作为采集能力对照；两个下游必须实际 `AllocNodes=0`、`AllocTRES` 为空、无已分配节点、无任何 step、零运行时长、无本地执行痕迹、没有重复或重排队历史歧义，并且已不在本用户队列中。观测不一致、资源字段缺失或没有对照均保持阻塞。Slurm 的 `AllocCPUS` 和 `NNodes` 可能显示申请量，不能用它们代替实际分配证据；`Reason` 也只作信息记录，不宣称确定了取消来源。此路径只保留和核验历史，不执行 `scancel`。

恢复阻塞时可执行 `"$P" hpc/recover_failed_chain.py --root "$PWD" --failed-test-job 767434 --diagnose`，只读取已绑定的原始回执和调度状态，不取消、不提交、不创建锁或恢复证明；输出仅包含经过验证的状态字段。复制命令时只复制代码块，不复制终端提示符、`>`、下载进度或 JSON 输出。

学校 glibc 2.17 和 Bash 4.2 的兼容入口保留原有原子性要求：Git 元数据初始化优先使用 libc `renameat2`，缺少该符号时只在已识别的 Linux LP64 ABI 上调用同一内核 `RENAME_NOREPLACE` 操作；内核或文件系统不支持时继续阻塞。传输临时文件通过 `O_NOFOLLOW` 打开并验证 inode 后执行 `fchmod`，保留完整 GET/SHA、硬链接无覆盖发布和回读。Bash 默认计划模式可处理 `set -u` 下的空参数数组。

CPU 作业运行模拟联网主机测试时，只有该测试夹具隔离继承的 `SLURM_JOB_ID`；生产 `fetch` 仍拒绝计算节点下载，专门的拒绝测试保留。原科研线程、随机性、fit 预算、测试期限和全部测试入口不变。

## GitHub 下载不稳定时的代码更新

固定代码更新包可通过 `hpc/code_download.py` 下载：它复用系统及既有 R2 SDK 的 CA，分块保存进度，有界重试时使用 HTTP Range 续传，并在完整长度与 SHA-256 核验后才发布文件。重复执行复用已核验的包。该工具不安装依赖、不访问科研数据、不调用 Git 或 Slurm；错误包和证书失败不会被当成成功。更新包支持单独手动传到学校后离线使用。

有既有 HPC Git 历史时，使用基于真实历史提交的固定增量 Git bundle，避免重新下载完整仓库历史。必须先执行 `git bundle verify`，再通过本地文件 fetch 和 `merge --ff-only` 更新；缺少真实基础提交或本地代码存在冲突时停止，保留数据、checkpoint 和旧回执。固定恢复包的启动脚本完成本地代码更新后直接运行 `hpc/start.py --execute --staged-data`，不再重复进入 Git HTTPS 取回。完整科学软件和数据门禁仍由原 Slurm 链执行。

## 已传完数据、前台核查中断：交给 Slurm 继续

如果五个固定对象及缓存已传到学校，前台进程在最后核查期间退出，保留所有缓存和旧回执，在 `login1` 执行：

```bash
cd /public/home/zhuziyuan/projects/paper3 && bash hpc/START.sh --execute --staged-data
```

该入口更新部署代码，检查固定包存在及大小，然后提交原有 `test → prepare → train` 的 `afterok` 链。**提交端不下载、解包或扫描全部数据 SHA，不将数据声明为 ready。** 完整包校验、真实历史审计和全部本地数据校验由计算节点中的 CPU 作业执行；`prepare` 缺文件、SHA 不符或测试失败时，不放行 GPU 训练。训练开始前仍重新检查原软件和数据门禁。计算节点不需要 R2 外网访问或下载凭证。

这只将部署验证迁入 Slurm，不更改算法、数据划分、科学预算或随机性设置。CPU 仍按各阶段配置申请，SHA 读取线程不超过已有分配和原部署上限；科学计算线程要求保持原样。保留旧 `FETCHING_FROZEN_DATA` 回执，不伪造完成状态。若既有提交意图结果未知、存在活动数据进程或已有训练链，入口仍拒绝重复提交。

收到三个真实作业号及 `SLURM_CHAIN_SUBMITTED_NOT_TRAINING_COMPLETE` 后，已提交作业不再依赖 SSH 会话。测试、准备或训练中的实际缺口以对应作业日志为准，不等同于科研完成。已有缓存未全部补齐时，CPU 准备会安全阻塞，需在获准联网节点继续原有恢复，不得跳过验证。该模式无 R2 调用，结果仍沿原本地 checkpoint/交接队列和完整 GET/SHA 备份流程保存。

解包、索引收据验证和文件 SHA 核查现在分别显示实际阶段和已处理/总文件数；计数只代表完成相应检查，不代表科学训练或结果备份完成。

## 一条命令更新并提交

在学校允许联网的登录/传输节点执行以下一条命令。它更新已有分支、只取回冻结授权输入并完整校验，然后提交 `test → prepare → train` 的 `afterok` 链。测试或准备失败时训练不会开始；取回数据未通过时一个作业都不提交。默认训练申请 1 张 A100、24 CPU、128 GiB、24 小时，原科研预算不变。更新前保留本地修改、checkpoint 和 claim，不强制覆盖。

```bash
p3_boot=$(mktemp /tmp/paper3-start.XXXXXX) && curl -fL --retry 3 --connect-timeout 15 --max-time 180 https://raw.githubusercontent.com/jackzhu119/research3/research/paper3-slurm-hpc-20261021/hpc/START.sh -o "$p3_boot" && bash "$p3_boot" --execute --network-host
```

先下载为文件再执行，以保留终端的隐藏凭证输入。已有环境变量可直接复用；否则仅在提示处输入，不把凭证加入上述命令。默认 Git 取回最多三次，处理临时 `Empty reply`；已有本地分支直接切换并快进。若存在同项目活动作业或提交结果不确定，入口报告原作业号并停止重复提交。输入取回阶段断线可重跑同一命令，已通过 SHA 的文件复用；实际 `sbatch` 作业提交后不依赖 SSH。数据取回尚在前台时请等待返回作业号，勿将进度字节或 HEAD 200 当作数据就绪。

已有新版本代码时也可运行 `"$P" hpc/start.py --execute --network-host`；默认不加 `--execute` 仅显示计划。末尾可加 `--cpus 16 --mem-gib 64 --gpus 1 --time 12:00:00` 手动修改训练资源；测试与准备仍用各自配置。冻结数值实现、种子、科学线程和 fit 预算不随资源参数改变。

新的 HPC 数据入口把 GET 读取、临时文件写入、原子发布和最终 SHA 回读分开记录，显示正文进度与安全 `failure_detail`。仅明确瞬时网络错误做有界重试，真实正文 SHA 不符保持失败。使用已核验普通临时文件的 POSIX 无覆盖发布，避开旧 glibc 的路径 chmod 可选标志；共享文件系统不支持输入缓存的目录 fsync 时如实记录，不宣称崩溃持久化。checkpoint、预算账本和备份交接的持久性要求不变。原 RunPod 传输模块不改。

## 1. 更新代码和固定数据

下列是逐步操作方式；已使用上面的一键入口时不要再重复手工提交。在 `login1` 的现有项目目录执行：

```bash
cd /public/home/zhuziyuan/projects/paper3
git fetch origin refs/heads/research/paper3-slurm-hpc-20261021:refs/remotes/origin/research/paper3-slurm-hpc-20261021
git checkout research/paper3-slurm-hpc-20261021
git merge --ff-only origin/research/paper3-slurm-hpc-20261021
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
```

以后更新同一分支：

```bash
git fetch origin refs/heads/research/paper3-slurm-hpc-20261021:refs/remotes/origin/research/paper3-slurm-hpc-20261021
git checkout research/paper3-slurm-hpc-20261021
git merge --ff-only origin/research/paper3-slurm-hpc-20261021
```

使用旧 Git 支持的 `checkout` 和 `merge --ff-only`；不要 `reset --hard`、清理 `runs/` 或删除 claim 文件。若本地有已跟踪修改，先保存并处理，勿强制覆盖。

Git 仓库只含代码和冻结索引，不含完整科研缓存。固定包统一放在项目 `.hpc/packages/`，保留项目 `cache/`、`runs/`、`.paper3-hpc/`。若数据未转移到学校，先在**学校允许联网的登录/传输节点或用户自己的联网机器**运行以下取回命令；同一项目需可见，跨机器时须手动转移下载包及缓存：

```bash
"$P" hpc/stage_data.py fetch --root "$PWD" --execute --network-host
```

该命令只按固定对象、原收据和 SHA-256 完整 GET 取回已授权包及冻结 W3 缓存（含原内部留出缓存的完整性核验），不授予内部科学评估权限、不启动训练、不下载封存外测 query、不安装 SDK。优先复用已有 `R2_ENDPOINT`、`R2_BUCKET`、`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY` 配置；交互提示时按隐藏输入填写，勿把凭证写进命令、Git 或日志。`--network-host` 是操作者确认该机器获准联网的显式标志，不会自动连接学校服务器。

离线检查：

```bash
"$P" hpc/stage_data.py check --root "$PWD"
```

缺失或 SHA 不符时保持阻塞。历史审计由原固定包恢复并逐文件验证，不能用空文件代替；只有原包不够时仍须补齐冻结数值缓存，不能以审计恢复代替数据就绪。

## 2. 先提交 CPU 验证

```bash
"$P" hpc/submit.py test
```

该命令先执行 `sbatch --test-only`，通过后再真实提交 CPU 作业，返回作业号。CPU 测试恢复真实审计、核对冻结代码身份并运行原测试；没有科学 fit。测试成功后提交：

```bash
"$P" hpc/submit.py prepare
```

`prepare` 在计算节点离线恢复与核验冻结数据，不进行 source 训练。不要在数据门失败时提交 `pilot/train`。本阶段请将 `stage-test.json`、`stage-prepare.json` 和失败日志反馈给维护者，再决定正式训练。

默认申请量如下；物理节点大小不代表账户配额。

| 阶段 | 分区 | CPU | 内存 | GPU | 时限 |
|---|---|---:|---:|---:|---:|
| test | cyhpc_1 | 16 | 64 GiB | 0 | 2 小时 |
| prepare | cyhpc_1 | 24 | 128 GiB | 0 | 12 小时 |
| pilot | hpc_gpu1 | 24 | 128 GiB | 1 | 8 小时 |
| train | hpc_gpu1 | 24 | 128 GiB | 1 | 24 小时 |
| backup | cyhpc_1 | 8 | 32 GiB | 0 | 4 小时 |

学校测试通过并由用户决定启动后，分别使用：

```bash
"$P" hpc/submit.py pilot
"$P" hpc/submit.py train
```

不要同时提交这两个命令。`pilot` 仅允许原工程 pilot；若已有真实 pilot，仅验证其实际收据和 checkpoint，不越过到下一个 fit。`train` 沿原主线和独立 source 扩展顺序执行，原有科研门禁仍生效。作业由 Slurm 持有，SSH 断开不会终止已提交作业。

`train` 从全新、已通过准备门的项目开始时，会在同一作业内先运行原 pilot，再进入正式阶段。若先单独运行 `pilot`，随后 `train` 仍须通过原同设备身份核验；不能假设两次 Slurm 分配一定得到同一张 A100。

## 3. 日志、资源和恢复

可另开一个 SSH 窗口，在同一项目目录运行只读面板：

```bash
"$P" hpc/monitor.py --root "$PWD"
```

每 10 秒读取本账号 Paper3 的 Slurm 状态、数据回执、当前活动阶段及资源采样。CPU 使用量同时显示核数和占申请 CPU 的百分比；GPU 利用率是设备整体值，可能包含共享设备上的其他作业。本进程显存与 GPU 设备利用率分开标注。首次、旧版或过期采样显示不可用，已结束作业的峰值不冒充当前值。没有运行中的计算作业时不会显示登录节点的 `top`/`nvidia-smi` 作为训练占用率。`Ctrl+C` 只关闭面板。脚本不启动训练、下载、`srun` 或远程登录；可用 `--once` 输出一次。

如果数据取回报告 `R2_GET_FAILED`，说明该次入口在作业提交前已停止。此错误本身无法区分网络、权限或对象不存在；在学校允许联网的节点运行：

```bash
"$P" hpc/r2_diagnose.py --root "$PWD" --network-host
```

该诊断仅向固定五个包对象发送 HEAD 请求，输出白名单 HTTP/服务错误码或网络类别，不下载正文、不重试训练、不保存 endpoint/bucket/密钥或原始异常消息。仅诊断使用较短连接等待；原 GET、PUT、完整 SHA 回读与科研预算不变。诊断回执保存在 `.paper3-hpc/diagnostics/`，HEAD 成功不等于完整 GET/SHA 验证成功。

将 `JOBID` 替换为实际作业号，将日志中的 `test` 替换为实际阶段：

```bash
squeue -u zhuziyuan
tail -f .paper3-hpc/logs/test-JOBID.out
tail -f .paper3-hpc/jobs/JOBID/pytest.log
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
```

各作业的 `.paper3-hpc/jobs/JOBID/` 保存环境、阶段回执、资源观测及资源建议；训练细节见 `campaign.log` 和两个扩展 arm 的日志。`resources.live.json` 是观测状态，`resources.json`、`resource_recommendations.json` 是退出后观测和建议，均不代表科研完成或 R2 上传。可补充学校计费数据：

```bash
"$P" hpc/resources.py accounting --job-id JOBID --output "$PWD/.paper3-hpc/jobs/JOBID/slurm_accounting.json"
```

CPU 使用率、峰值内存、所属 GPU 进程显存与耗时用于生成建议；GPU 设备整体利用率可能包括同节点其他作业。学校配额未知时不自动扩大申请，不自动重提作业。`hpc/resources.json` 的 `recommendation_limits` 可在确认学校配额后填写。资源可逐次手动调整，先查看命令再提交：

`hpc/resources.json` 是可修改的运行资源配置，不属于冻结的数值实现身份；线程上限仍由作业脚本和环境门禁固定为 1。修改已跟踪配置后，GitHub 自动发布会要求先手动提交该变更；使用下列命令行参数调整则无需修改仓库文件。

```bash
"$P" hpc/submit.py pilot --cpus 16 --mem-gib 64 --gpus 1 --time 08:00:00 --dry-run
"$P" hpc/submit.py pilot --cpus 16 --mem-gib 64 --gpus 1 --time 08:00:00
```

`--account`、`--partition`、`--python`、`--root` 也可覆盖配置。默认不占整节点，未验证多 GPU 加速，增加 `--gpus` 不会自动改为分布式训练。所有科学 BLAS/OpenMP 线程仍固定为 1；申请更多 CPU 不授权改变随机性、并发科学 fit、预算或数据划分。Slurm 时限是行政限制，不延长科研预算。

时限信号通知原控制器停止，保存流程沿原 checkpoint/claim 进行。确认旧作业已经退出后，在**同一项目目录和同一环境**重新提交相同阶段即可尝试恢复：

```bash
"$P" hpc/submit.py train
```

不要删除 checkpoint、claim、ledger 或修改身份信息来强行通过恢复门。原严格恢复会核验运行时；GPU UUID、环境或代码变化时可能拒绝精确恢复，应保留原文件并反馈真实报错。进程被强制终止时只以实际存在且通过验证的 checkpoint 为准，不宣称未落盘步骤已保存。

## 4. 离线计算节点与备份交接

训练退出或受控暂停后，程序对原生产者允许的真实结果和 checkpoint 加锁、完整 SHA 校验并复制到项目 `.paper3-hpc/handoffs/<内容SHA>/payload/`，建立持久化交接队列。原 checkpoint 不删除，队列与 checkpoint 均须留在持久共享存储。若磁盘不足、仍有活动控制器或校验失败，保留原文件并报告阻塞。建议预留当前结果集一份额外副本空间；每个不同快照可能额外占空间。

队列包含实际存在的恢复关键元数据：live `campaign_progress.json`、claim、累计 wall ledger、checkpoint、进度引用的旧原 manifest、本次真实结果集合 manifest，以及扩展结果的原 manifest。旧进度及其引用文件保持原字节；新 partial checkpoint 不会令其被重写。若被进度引用的原 manifest 缺失或 SHA 不符，保持阻塞，不伪造状态、不删除 claim。收集清单和本地复制均不代表上传成功；只有 R2 完整 GET 正文及 SHA-256 全部通过才报告该批已验证。

离线状态为 `PENDING_EXTERNAL_BACKUP_NOT_UPLOADED`，不能视作已备份。待作业退出后，在**学校允许联网且可见该项目的登录/传输节点**显式运行：

```bash
"$P" -m hpc.handoff --root "$PWD" --flush --publish-code
```

这条命令不 fit：R2 继续复用原上传和**完整 GET 正文 SHA-256 回读**，每次尝试保留原回执；GitHub 只推送现有已提交且通过代码白名单检查的 HEAD，并回读远端 commit，不自动 `git add`、不提交数据或密钥。默认发布到 `research/paper3-hpc-code-<commit前12位>`；可加 `--branch research/指定分支`，已有不同 commit 的远端分支会被保留而非强推。GitHub 失败不会撤销已验证的 R2 备份证据。

R2 或 GitHub 失败后重复同一交接命令即可，仅重试备份/代码发布，不重训。需要指定一批结果时追加 `--queue-id <64位内容SHA>`。成功以队列 `status.json`、`attempts/*/R2_receipt.json` 的完整 GET 校验事实为准；网络不通或未完成正文回读始终保持待交接，不删除本地 checkpoint。

只有确认**计算节点允许出网、已有 R2/GitHub 配置并符合学校权限**时，才可另行提交 `"$P" hpc/submit.py backup`；无需出网的默认训练流程不依赖它。没有联网许可时应保持离线队列，使用学校认可的传输方式交接，不能绕过访问限制。
