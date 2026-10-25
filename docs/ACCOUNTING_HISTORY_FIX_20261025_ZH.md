# 第五、六轮历史分区回执修复

## 用户现场证据

北京时间2026-10-25 12:15左右的日志确认：第一轮`HANDOFF READY`，772076/772077/772078已release并因Resources/Priority等待学校调度；第一轮旧任务770043已不在当前队列。第五、六轮仍有六份held新任务，旧772012/772013已不在当前队列，管理器重复报告`old_accounting_owner_or_partition_mismatch`。

## 根因与更改

旧单节点作业的marker按科学recipe与输出目录确定，不包含partition。原hpc_gpu1的a1与后来hpc_gpu的a2共用marker。先前迁移保留a1为`FINISHED / RESOURCE_MIGRATION_HANDOFF`历史attempt，而旧核验对sacct返回的所有历史行都要求partition=hpc_gpu，误拒绝了合法的旧终态记录。

修复后的正常查询只请求持久化作业ID；仅未知ACK提交意图需要按持久化名称查询受日期约束的历史。owner/marker/name先核验，终态历史保留原partition（含空/null）及来源，不要求重写成当前分区。活跃作业和当前队列仍按其真实分区处理；历史终态不能代替当前待交接作业的终态证明，也不能解除未知提交意图。

本次相对上一修复包的生产代码差异仅`round56/migrate.py`；第一轮代码与接管器均未改变。保留原计划、科学参数、结果、checkpoint、runtime、六个新作业号及旧不可变安装目录。

## login1单行执行

```bash
( set -e; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-parallel-downloads"; mkdir -p "$D"; H=6c27d840b443bd20e88ee31b20a4019e048ccb7e90e332d4640284b7de6a60f6; F="$D/REPAIR-HANDOFF-${H:0:12}.py"; R=3d9ea6d666e7ad976f1c5dcdc4eb3de9f85d0c9c; Q=operations/parallel_20261025/online/REPAIR-PAPER3-HANDOFF.py; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); if [ -f "$A" ]; then C=(--cacert "$A"); fi; T=""; trap 'rm -f "${T:-}"' EXIT; until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do T=$(mktemp "$D/.download.XXXXXX"); for U in "https://raw.githubusercontent.com/jackzhu119/research3/$R/$Q" "https://api.github.com/repos/jackzhu119/research3/contents/$Q?ref=$R"; do if curl -4 "${C[@]}" -fL --retry 2 --retry-delay 2 --connect-timeout 10 --max-time 90 -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" && printf '%s  %s\n' "$H" "$T" | sha256sum -c -; then mv -f "$T" "$F"; break; fi; done; if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then rm -f "$T"; printf 'DOWNLOAD_RETRY\n'; sleep 3; fi; done; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; cp "$F" "$D/START-PAPER3-PARALLEL.py"; "$P" -I -S -B -u "$F" --repair --execute )
```

第一轮应返回`OBSERVE_ONLY_HANDOFF_ALREADY_READY`，不会替换其管理器。第五、六轮返回`FIXED_CONTROLLERS_STARTED_EXISTING_JOBS_PRESERVED`后，由原生控制器核实旧作业终态、释放锁并release原六个held任务。无需手动取消/重新提交，也不会跑全量pytest、pilot、数据树核验或重新安装环境。

下载或GitHub暂时不可达会使用备用入口并重试；所有训练输出和第2/3/7轮上传器保持原位置及进程。放行后的Resources、Priority、AssocGrpGRES属于学校资源/配额排队，不能由本修复改变。

## 验证

第五、六轮70项测试通过（迁移部分21项），覆盖：历史hpc_gpu1 a1取消→hpc_gpu a2取消→放行原新作业；空历史分区；未知ACK与comment缺失；错误owner/marker；活跃旧分区；缺少当前作业终态不得用历史替代。复用未改变第一轮109项与接管器12项验证证据。

实际发布文件已通过raw/API双路下载与SHA256、21个安装文件逐一核验，三轮prepare保持既有科学文件字节及mtime不变。学校执行这一版修复须以用户命令回执和最新队列为准，本页不提前声称训练已经开始。
