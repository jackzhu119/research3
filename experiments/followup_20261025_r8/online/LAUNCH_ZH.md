# 第八轮启动

版本固定到 `1ef2739faee1e0fa0d6832225481d003e2050547`。在学校 login1 完整复制下面一整行；正常运行中的其他轮次不受影响。

```bash
( set -e; F=/public/home/zhuziyuan/paper3-round8-downloads/START-PAPER3-R8-ONLINE.py; H=37d423669e6dcaf3786223a3341b3f1cfbda677d727963331d79c382c78160f1; mkdir -p "$(dirname "$F")"; C=(); [ ! -f /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem ] || C=(--cacert /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem); T="$F.part"; while ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do for U in https://raw.githubusercontent.com/jackzhu119/research3/1ef2739faee1e0fa0d6832225481d003e2050547/experiments/followup_20261025_r8/online/START-PAPER3-R8-ONLINE.py 'https://api.github.com/repos/jackzhu119/research3/contents/experiments/followup_20261025_r8/online/START-PAPER3-R8-ONLINE.py?ref=1ef2739faee1e0fa0d6832225481d003e2050547'; do RC=0; curl -4 "${C[@]}" -fL -C - --connect-timeout 10 --max-time 180 -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" || RC=$?; if [ "$RC" -eq 33 ] || [ "$RC" -eq 36 ] || [ "$RC" -eq 22 ]; then rm -f "$T"; fi; if [ "$RC" -eq 0 ]; then if printf '%s  %s\n' "$H" "$T" | sha256sum -c -; then mv -f "$T" "$F"; break; else rm -f "$T"; fi; fi; done; if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then printf 'DOWNLOAD_RETRY: partial download retained; retrying\n'; sleep 3; fi; done; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$F" --submit )
```

资源固定为 `hpc_gpu` 队列的三个独立单卡作业，每个16核CPU、64G内存；总计3GPU、48核CPU、192G内存。

下载可以断点续传，raw地址失败后尝试GitHub API，网络失败会继续重试；确认完整文件后才安装和启动。包内含已准备的Pan与Weibo裁剪缓存，计算节点不再下载数据或安装依赖。学校已有Stieger缓存和源模型在读取时复用。

`ROUND_8_BACKGROUND_PROCESSES_STARTED` 说明登录节点控制器已启动；实际提交的三个作业号在 `runs/followup_20261025_r8/runtime/three_jobs/scheduler-submissions/`。`TRAIN_PROCESS_STARTED` 表示计算节点脚本开始，训练更新和会话计数才表示科学计算进展。配额不足时控制器等待重试。

以后续跑使用同一条命令，采用已有作业及检查点。旧轮次不会被取消。全量测试和原始数据树扫描不进入学校启动流程。

GitHub进度：`paper3-r8-progress` 分支的 `monitoring/paper3-round8/three-jobs-status.json`。结果摘要：`paper3-results/results/paper3/round-08/`。完整预测、检查点和原始结果先留存在学校目录；GitHub网络失败不影响训练。

本地已按Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6进行CPU兼容验证；下载、安装、三作业调度模拟的证据见 `online/download-proof.json`。未声称学校GPU现场执行或不会遇到节点/队列故障。
