# 固定执行包的本地启动

本说明随本轮固定软件包使用。只有最终 `bootstrap_index.json`、cache-only 闭包及真实 R2 回读验收回执一并发布后，才表示 CPU 准备验收完成。当前软件测试、离线安装和数据清单不构成任何 GPU 训练完成或实验优势的证明。

用户本人先开启已有云服务器，将固定包校验 SHA256 后解压到一个新的论文3目录。以后一直在同一目录恢复；不要复制到新目录、删除 claim、更换输出路径或重置预算来重训。脚本不调用 RunPod、SSH、浏览器或实例管理接口。

固定包内必须有：已冻结的 Python/论文3代码和元数据、全部选定 R2 receipt/index、CPython 3.12.15 原归档、以及已校验的八个最小 SDK 离线 wheel。完整 EEG 原档不进入部署恢复；110 个完整依赖 wheel 和数值缓存从已有 R2 按 exact paths 恢复。既有 R2 环境变量由用户原配置提供，不能粘贴到命令、文件、聊天或日志。

```bash
cd /你的固定论文3目录
python3 tools/bootstrap_data_plan.py --index audit/data_plan_20261020/final_deployment/bootstrap_index.json
```

此默认命令只核对固定 index 和包内引导文件，不联网、不安装、不训练。若最终包把 index 放在其他位置，以其回执中的实际路径替换上面的路径。

先准备软件和缓存：

```bash
python3 tools/bootstrap_data_plan.py --index audit/data_plan_20261020/final_deployment/bootstrap_index.json --execute
```

这一步先用标准库核验、解压原 portable Python，创建包内独立环境，以 `--no-index` 安装已有八个 SDK wheel；随后由原有两套 IO 策略恢复各自 exact paths，并逐文件完整 GET/SHA256 核对。已有且 SHA 相同的文件直接本地复核，不重复下载；不一致文件会停止，不覆盖。最后从本地 110 个 wheel 安装固定的 100 个 GPU profile 依赖，运行 `pip check` 和导入检查。Linux x86_64 宿主必须有足够空间容纳缓存、全部 wheel、约 7 GiB 解包依赖以及临时空间；空间不足会在开始网络恢复前停止。

恢复过程和安装日志在 `.paper3-deployment/`。恢复中断后执行同一条命令；它会保留已核验文件和已有回执，继续同一份 index。完整 GET 证明在 `.paper3-deployment/restore_receipts/`；完整本地复核快照在 `.paper3-deployment/restore_complete/`。已安装环境不等于 GPU 驱动兼容性已通过，后者由受控 pilot 实测。

用户开机后，明确开始科学流程：

```bash
python3 tools/bootstrap_data_plan.py --index audit/data_plan_20261020/final_deployment/bootstrap_index.json --execute --launch-training --user-started-gpu
```

顺序为：主线唯一受控 source GPU 恢复 pilot，主线 f0/强传统对照/Rref/元学习/开发选择及一次性内部检验；主线真实 14 方法、三 seed 开发冻结通过后，分别运行 `BASE4_PLUS_YANG` 和 `BASE4_PLUS_DREYER` 的独立 f0/Rref/强对照/元学习/开发冻结。两支路不合并成默认六库，也不共享伪造的 fit 或工程计时。每个 producer 都保留其固定 claim、累计时间、调用预算和可恢复检查点；未通过的数据门、时延门或数值恢复门会停止后续阶段。

启动器在任何训练进程导入数值库前统一设置 OMP、OpenBLAS、MKL、NumExpr、VECLIB 的线程环境为 1，保证主线与新增支路的工程时延条件一致。source producer 自身冻结的内部线程设置保持原规则。它不自动训练未冻结的 CAR3/Ma 实验，不解封任何外测 query，不进入 W7 外测流程。

主线日志在 `.paper3-deployment/main_campaign.log`，新增支路日志分别在同目录对应 arm 的 `.log`。日志不是完成回执；科研完成状态以各 producer 的真实 receipt 和验证器为准。失败时保留全部 claim、检查点、结果和预算；先解决报告的最小缺口，再从原目录继续，不能换目录重复 fit。

## 训练后的结果备份

启动器只做本地恢复、安装与明确授权的科学执行，不把“生成 manifest”冒充 R2 上传。当前准备阶段科学 fit 为 0，尚无真实训练结果可上传。

主线停止或完成后，`runs/training_campaign_v1/backup_manifest.json` 由主线 `finally` 生成，列出实际存在的有限结果、claim 和检查点。先审查该 manifest，再使用原备份工具：

```bash
.paper3-deployment/venv/bin/python tools/r2_backup.py --manifest runs/training_campaign_v1/backup_manifest.json --prefix paper3/results/实际唯一快照编号/main/ --receipt audit/deployment_result_snapshots/实际唯一快照编号_main_R2.json --execute
```

原工具对每个对象执行条件 PUT 和完整 GET/SHA256；只有 receipt 的 `status=verified`、每个对象的 `complete_body_read=true` 且 SHA 匹配才算完成。manifest 随运行状态变化，下一快照必须用新的唯一 prefix，不能覆盖已有快照。

每个新增 f0 还会生成自身的 `runs/source_extensions/<arm>/seed-<实际seed>/backup_manifest.json`。新增支路整体可在所有相应 worker 停止后生成有限本地快照：

```bash
.paper3-deployment/venv/bin/python tools/bootstrap_data_plan.py --snapshot-extension-results
.paper3-deployment/venv/bin/python tools/bootstrap_data_plan.py --snapshot-extension-results --execute
```

默认只列出计划；第二条只写不可变的 `audit/deployment_result_snapshots/<实际SHA>_source_extensions.json`，不联网、不训练。它仅包含两条 canonical source arm 的实际 `.json/.npy/.pt` 结果和对应 f0 claim，拒绝私密文件/符号链接；若已有 worker 持锁则停止。把其输出的 `manifest_path` 传给原 `r2_backup.py`，使用另一个唯一 prefix 和 receipt 路径，即可同样完成 PUT + 完整 GET/SHA。失败/部分结果也应保留，不能为了得到完成回执而重置 fit。

需要额外独立回读时，以对应 verified R2 receipt 的明确路径选择调用 `tools/r2_restore.py --receipt ... --root 新的空目标目录 --report ... --paths 明确文件路径 ... --execute`；数值 source/development 数据若使用扩展 IO profile，则使用 `tools/r2_source_extensions.py restore` 同策略的 exact paths。不要对含原档证据的混合 receipt 使用 `--all-approved`。
