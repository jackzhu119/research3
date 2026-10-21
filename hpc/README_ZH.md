# 论文 3：学校 Slurm 部署

本入口复用学校已有 Python 环境，不安装 PyTorch、Docker 或便携 Python。保留原 RunPod 入口、科研协议、数值实现、数据划分、随机种子、训练预算及科研门禁。`test / prepare / pilot / train / backup` 均由 `sbatch` 调度；不要在 `login1` 直接执行 `hpc/run_stage.py` 或原训练入口。

代码已在 Codex 进行软件测试；这不代表已连接学校、完成学校兼容验证、启动真实训练或上传真实结果。请先运行 CPU 测试和数据准备，反馈学校日志后再决定是否提交 GPU 作业。

## 1. 更新代码和固定数据

在 `login1` 的现有项目目录执行。以下首次创建本地研究分支；若此分支已存在，用下方更新命令。

```bash
cd /public/home/zhuziyuan/projects/paper3
git fetch origin research/paper3-slurm-hpc-20261021
git checkout -b research/paper3-slurm-hpc-20261021 FETCH_HEAD
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
```

以后更新同一分支：

```bash
git fetch origin research/paper3-slurm-hpc-20261021
git checkout research/paper3-slurm-hpc-20261021
git merge --ff-only FETCH_HEAD
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
