# 第七轮线上启动：hpc_gpu 单节点三卡

最终资源为 **hpc_gpu / A100-PCIe-40GB，1节点、1作业、3GPU、总36CPU、共享128G**。`--cpus 12` 是每槽12核；128G是整作业主机内存。分区汇总空闲卡数不保证单节点当下满足全部资源，资源不足正常排队。

源码固定提交：`e804bf9febcc6e95d4e230f3a9869073440f0d38`。启动器固定提交：`0ecb1bbac0e6365625f401e71604e4ddec6baf41`；下载包SHA256：`72dc52eefc18c343c08f794f8f0cbc77a541df077c644376b393189973d1ae0d`，24746248bytes，内含Pan2023已备缓存。

在login1复制执行下面完整块；这会准备273项并启动后台监控/单作业控制器，学校接受提交后会打印作业号，节点日志出现`TRAIN_PROCESS_STARTED`才说明批脚本开始。

```bash
(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-r7-downloads"
mkdir -p "$D"
F="$D/DOWNLOAD-AND-START-R7.py"
H=61566627c1c155ef40d42c4c71089dc09355a46121837b7d31731e8aa722d092
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
  T=$(mktemp "$D/.bootstrap.XXXXXX")
  trap 'rm -f "${T:-}"' EXIT
  until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
    for U in \
      'https://raw.githubusercontent.com/jackzhu119/research3/0ecb1bbac0e6365625f401e71604e4ddec6baf41/experiments/followup_20261024_r7/online/DOWNLOAD-AND-START-R7.py' \
      'https://api.github.com/repos/jackzhu119/research3/contents/experiments/followup_20261024_r7/online/DOWNLOAD-AND-START-R7.py?ref=0ecb1bbac0e6365625f401e71604e4ddec6baf41'
    do
      if curl -4 "${C[@]}" -fL --retry 1 --retry-delay 2 \
        --connect-timeout 10 --max-time 60 \
        -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" &&
        printf '%s  %s\n' "$H" "$T" | sha256sum -c -
      then
        mv -f "$T" "$F"
        break
      fi
    done
    if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
      printf 'Download will retry in 10 seconds; Ctrl+C preserves the training bundle cache.\n'
      sleep 10
    fi
  done
fi
"$P" -I -S -B -u "$F" --download-dir "$D" --partition hpc_gpu --gpus 3 --cpus 12 --memory-total 128G --submit
)
```

同一块也用于续跑。完成项、字典双代断点和已保存会话复用；计算节点不下载原数据或安装依赖，不重跑全量test/pilot/全树数据扫描，无累计12小时上限。单次Slurm时限到期前保存，由控制器对账续交；明确主动取消保持停止。

GitHub每180秒尝试同步`paper3-r7-progress:monitoring/paper3-round7/status.json`；同步失败训练继续。既有本地私有配置自动发现，包中无凭据。完整模型断点、概率/log概率及CSV先存学校`runs/followup_20261024_r7`。

273项是15字典、72内部核心、72敏感性、21参考、93重复Pan评价；不是273次从头训练EEGNet。文献、公式、组件消融及24项Holm见[设计](../DESIGN_ZH.md)、[文献](../LITERATURE.md)、[统计](../STATISTICS.md)。Pan复用第六轮同一14人28会话，属于探索性重复外部验证。

发布侧已实测raw HTTP206断点续传、API完整下载和缓存禁网复用；下载回来的最终包已在禁止socket/DNS与所有子进程条件下实际安装、准备273项并逐位比对28会话×5预算；四个隔离entry入口可启动。526项本地软件测试通过，另有下载/安装针对性检查。学校GPU现场运行尚未验证，没有提交学校作业或生成本轮识别效果。证据见本目录JSON。
