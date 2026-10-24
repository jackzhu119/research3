# 第六轮线上启动与续跑

已实际验证：固定raw包HTTP206续传、API备用下载、完整缓存断网复用、线上bootstrap到已安装实际入口prepare，以及禁网状态下全部28会话/五预算读取。学校A100三卡实跑尚未现场执行。

计算节点不下载原MAT；23.4MB已处理Pan缓存在24.66MB单文件里。bootstrap保留分片持续重试，不设整次传输90秒上限。正常完成后相同命令复用包、结果和断点。资源不足时控制器等待，不重跑全量test/pilot。

在login1完整复制以下整块；带submit会按用户既有授权启动一个三卡作业的后台控制器。

```bash
(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-followup-downloads"
mkdir -p "$D"
F="$D/DOWNLOAD-AND-START-R6.py"
H=f4fc457acef0fa0f479dd1db95bbf250a40276618e844e6261d585399c0fb140
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
trap 'rm -f "${T:-}"' EXIT
while ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
  T=$(mktemp "$D/.r6-bootstrap.XXXXXX")
  for U in \
    'https://raw.githubusercontent.com/jackzhu119/research3/d5973a6fa7aa3d6a1bc3892b209eaa908fae8c0e/experiments/followup_20261024_r6/online/DOWNLOAD-AND-START-R6.py' \
    'https://api.github.com/repos/jackzhu119/research3/contents/experiments/followup_20261024_r6/online/DOWNLOAD-AND-START-R6.py?ref=d5973a6fa7aa3d6a1bc3892b209eaa908fae8c0e'
  do
    if curl -4 "${C[@]}" -fL --retry 2 --retry-delay 2 \
      --connect-timeout 10 --max-time 60 \
      -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" &&
      printf '%s  %s\n' "$H" "$T" | sha256sum -c -; then
      mv -f "$T" "$F"
      break
    fi
  done
  rm -f "$T"
  if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
    printf '启动器连接暂不可用，5秒后重试；尚未提交作业。\n'
    sleep 5
  fi
done
"$P" -I -S -B -u "$F" \
  --partition hpc_gpu1 --gpus 3 --cpus 12 --memory-total 164G --submit
)
```

看到PREPARED是计划生成，DETACHED_PROCESSES_STARTED是login控制器启动；job_id是提交记录。只有TRAIN_PROCESS_STARTED及outer/session计数增长才说明节点实际计算。完整启动脚本也存为[EXECUTE.sh](EXECUTE.sh)。

主输出：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r6`；GitHub：`paper3-r6-progress:monitoring/paper3-round6/status.json`。完整checkpoint/逐会话结果先存在学校共享目录。
