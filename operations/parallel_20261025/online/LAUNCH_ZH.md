> 最新：第一轮已完成交接并正常排队；第五、六轮历史分区核验已修复。[保留现有六个作业的接管指令](../ACCOUNTING_HISTORY_FIX_ZH.md)。

> 2026-10-25 修复：学校排队作业的 `NumNodes=1-1` 已按单节点正确解析。已经提交772070–772078的用户，请先使用 [原作业接管修复指令](../NODE_RANGE_FIX_ZH.md)，保留原九个作业，不重新提交。

# 三个轮次的启动指令

在 login1 按顺序执行三条。第一条会下载校验共用启动器，然后启动第一轮迁移；第五、六轮使用同一份已校验启动器。也提供各轮独立的 `RUN-ROUND1.sh`、`RUN-ROUND5.sh`、`RUN-ROUND6.sh`（包含自己的下载重试）。

## 第1轮：hpc_gpu / A100 PCIe40GB，3个单卡作业，各12CPU，内存43G/43G/42G

```bash
(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-parallel-downloads"
mkdir -p "$D"
F="$D/START-PAPER3-PARALLEL.py"
H=6c27d840b443bd20e88ee31b20a4019e048ccb7e90e332d4640284b7de6a60f6
R=3d9ea6d666e7ad976f1c5dcdc4eb3de9f85d0c9c
Q=operations/parallel_20261025/online/START-PAPER3-PARALLEL.py
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); if [ -f "$A" ]; then C=(--cacert "$A"); fi
T=""
trap 'rm -f "${T:-}"' EXIT
until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1
 do
  T=$(mktemp "$D/.download.XXXXXX")
  for U in "https://raw.githubusercontent.com/jackzhu119/research3/$R/$Q" "https://api.github.com/repos/jackzhu119/research3/contents/$Q?ref=$R"
  do
    if curl -4 "${C[@]}" -fL --retry 2 --retry-delay 2 --connect-timeout 10 --max-time 90 -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" && printf '%s  %s\n' "$H" "$T" | sha256sum -c -
    then mv -f "$T" "$F"; break; fi
  done
  if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
    rm -f "$T"; printf 'DOWNLOAD_RETRY: 原训练保持运行，重试下载\n'; sleep 3
  fi
 done
printf '%s  %s\n' "$H" "$F" | sha256sum -c -
"$P" -I -S -B -u "$F" --round 1 --execute
)
```

## 第5轮：hpc_gpu1 / A100 SXM4 80GB，3个单卡作业，各12CPU/64G

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/paper3-parallel-downloads/START-PAPER3-PARALLEL.py" --round 5 --execute
```

## 第6轮：hpc_gpu1 / A100 SXM4 80GB，3个单卡作业，各12CPU/64G

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/paper3-parallel-downloads/START-PAPER3-PARALLEL.py" --round 6 --execute
```

三个替代作业入队确认之前，旧任务继续保留；之后由程序保存断点/取消旧pending并自动release新任务。无需手动scancel。

第一轮迁移边界：`runs/training_campaign_parallel_v1/migration-progress-snapshot.json`；运行日志为该目录`controller.log`。第五、六轮日志为各原输出`runtime/three_jobs/controller.log`。

`BACKGROUND_STARTED`只表示后台服务启动；实际作业ID、排队状态和handoff proof见各轮提交回执、progress.json及自动GitHub监控。学校QoS不允许同时排入三份替代作业时，程序保留旧任务并等待重试。
