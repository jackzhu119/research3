# 第四、五、六轮分开迁移启动

三个包已完整发布，并实际验证raw HTTP206续传、API完整下载、SHA、缓存禁网复用、下载回来的安装器离线准备与旧计划/完成结果/断点/会话保留。R4/R5/R6本地测试分别198/266/347项通过；尚未执行学校真实GPU或取消学校作业。

每个指令只处理对应轮次：hpc_gpu A100-PCIe-40GB，1节点1作业3GPU，总36CPU、共享128G（每槽12CPU）；三个指令总共最多三个作业，是否同时获得资源由学校排队决定。既有plan、原recipe、完成结果与checkpoint继续保留，学校启动不重跑全量test/pilot/全数据树扫描。

执行顺序是完整下载/校验相应包→MIGRATE --submit→该轮旧controller/allocation保存断点→仅取消同用户且精确marker匹配的旧PENDING任务，若已RUNNING则只请求USR1保存→等待旧作业终止证据→新hpc_gpu单job提交。不会取消其他轮次。网络失败保留分片重试，下载未完成时不进入旧作业交接。

| 轮次 | 原任务数 | 自动汇总分支 | 独立路径 |
|---|---:|---|---|
| 第四轮 | 204 | paper3-r4-progress | monitoring/paper3-round4/status.json |
| 第五轮 | 105 | paper3-r5-progress | monitoring/paper3-round5/status.json |
| 第六轮 | 198 | paper3-r6-progress | monitoring/paper3-round6/status.json |

R4/R5复用已有学校缓存；R6的Pan2023预处理缓存约23.4MB随包携带，计算节点不取原MAT或安装SciPy/h5py。原安装目录须保留以供旧计划路径和私有GitHub配置复用。监控失败与训练独立，完整原始预测、CSV和checkpoint先存学校各轮runs目录。

## 第4轮

在login1复制执行下面完整块。后续续跑仍用该轮同一块。

```bash
(
set -e
R=4
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-hpcgpu-downloads"
mkdir -p "$D"
F="$D/START-HPCGPU-MIGRATIONS.py"
H=988abd811a930eaf6c346711d19888feee86a5af02c5273c7e64ee4bd757920a
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
  T=$(mktemp "$D/.bootstrap.XXXXXX")
  trap 'rm -f "${T:-}"' EXIT
  until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
    for U in \
      'https://raw.githubusercontent.com/jackzhu119/research3/1d3e4860fc8a138e7be2717dae75ed55a0657e47/operations/hpc_gpu_migrations_20261024/online/START-HPCGPU-MIGRATIONS.py' \
      'https://api.github.com/repos/jackzhu119/research3/contents/operations/hpc_gpu_migrations_20261024/online/START-HPCGPU-MIGRATIONS.py?ref=1d3e4860fc8a138e7be2717dae75ed55a0657e47'
    do
      if curl -4 "${C[@]}" -fL --retry 1 --retry-delay 2 --connect-timeout 10 --max-time 60 \
        -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" &&
        printf '%s  %s\n' "$H" "$T" | sha256sum -c -
      then
        mv -f "$T" "$F"
        break
      fi
    done
    if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then sleep 5; fi
  done
fi
"$P" -I -S -B -u "$F" --round "$R" --download-dir "$D" --submit
)
```

## 第5轮

在login1复制执行下面完整块。后续续跑仍用该轮同一块。

```bash
(
set -e
R=5
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-hpcgpu-downloads"
mkdir -p "$D"
F="$D/START-HPCGPU-MIGRATIONS.py"
H=988abd811a930eaf6c346711d19888feee86a5af02c5273c7e64ee4bd757920a
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
  T=$(mktemp "$D/.bootstrap.XXXXXX")
  trap 'rm -f "${T:-}"' EXIT
  until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
    for U in \
      'https://raw.githubusercontent.com/jackzhu119/research3/1d3e4860fc8a138e7be2717dae75ed55a0657e47/operations/hpc_gpu_migrations_20261024/online/START-HPCGPU-MIGRATIONS.py' \
      'https://api.github.com/repos/jackzhu119/research3/contents/operations/hpc_gpu_migrations_20261024/online/START-HPCGPU-MIGRATIONS.py?ref=1d3e4860fc8a138e7be2717dae75ed55a0657e47'
    do
      if curl -4 "${C[@]}" -fL --retry 1 --retry-delay 2 --connect-timeout 10 --max-time 60 \
        -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" &&
        printf '%s  %s\n' "$H" "$T" | sha256sum -c -
      then
        mv -f "$T" "$F"
        break
      fi
    done
    if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then sleep 5; fi
  done
fi
"$P" -I -S -B -u "$F" --round "$R" --download-dir "$D" --submit
)
```

## 第6轮

在login1复制执行下面完整块。后续续跑仍用该轮同一块。

```bash
(
set -e
R=6
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-hpcgpu-downloads"
mkdir -p "$D"
F="$D/START-HPCGPU-MIGRATIONS.py"
H=988abd811a930eaf6c346711d19888feee86a5af02c5273c7e64ee4bd757920a
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
  T=$(mktemp "$D/.bootstrap.XXXXXX")
  trap 'rm -f "${T:-}"' EXIT
  until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
    for U in \
      'https://raw.githubusercontent.com/jackzhu119/research3/1d3e4860fc8a138e7be2717dae75ed55a0657e47/operations/hpc_gpu_migrations_20261024/online/START-HPCGPU-MIGRATIONS.py' \
      'https://api.github.com/repos/jackzhu119/research3/contents/operations/hpc_gpu_migrations_20261024/online/START-HPCGPU-MIGRATIONS.py?ref=1d3e4860fc8a138e7be2717dae75ed55a0657e47'
    do
      if curl -4 "${C[@]}" -fL --retry 1 --retry-delay 2 --connect-timeout 10 --max-time 60 \
        -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" &&
        printf '%s  %s\n' "$H" "$T" | sha256sum -c -
      then
        mv -f "$T" "$F"
        break
      fi
    done
    if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then sleep 5; fi
  done
fi
"$P" -I -S -B -u "$F" --round "$R" --download-dir "$D" --submit
)
```

迁移状态和日志在各轮结果目录runtime/hpc_gpu_migration/progress.json、migration.log。MIGRATION_PROCESS_STARTED只表示后台交接启动；原controller.log出现新作业号表示Slurm接受，节点logs/train-作业号.out出现TRAIN_PROCESS_STARTED才代表节点开始。科学效果与学校GPU实跑不由本地交付验证保证。
