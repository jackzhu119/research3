# 第一轮完整运行修复

固定代码提交：`13a80ad8886297cba8a49f1fcb8f3f244487989c`；入口 SHA256：`14c3638a6705150837b7f6604990cb58d03fe13cf16c43667f92a12ffb65e6c0`。

在 login1 执行下列一整行。仅保留原三槽位，旧动作以原 SIGINT 路径断点退出；不取消现有 allocation、不清空结果。

```bash
( set -e; trap 'exit 130' INT TERM; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-r1-full-repair-downloads"; F="$D/REPAIR-PAPER3-R1-COMPLETE.py"; H=14c3638a6705150837b7f6604990cb58d03fe13cf16c43667f92a12ffb65e6c0; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); for T in 1 2 3; do if printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then break; fi; for U in https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/13a80ad8886297cba8a49f1fcb8f3f244487989c/operations/round1_complete_repair_20261027/delivery/REPAIR-PAPER3-R1-COMPLETE.py https://raw.githubusercontent.com/jackzhu119/research3/13a80ad8886297cba8a49f1fcb8f3f244487989c/operations/round1_complete_repair_20261027/delivery/REPAIR-PAPER3-R1-COMPLETE.py; do if curl -4 "${C[@]}" -fL --connect-timeout 8 --speed-time 12 --speed-limit 128 --max-time 90 "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; done; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; "$P" -I -S -B -u "$F" --repair --submit --refresh-current )
```

默认仅安装：去掉末尾 `--repair --submit --refresh-current`。只读检查：改为 `--check`。

后台状态：

```bash
W="$HOME/paper3-r1-full-repair/work"
cat "$W/STARTED.json"
tail -n 12 "$W/controller.log"
cat "$W/controller-status.json"
```

`R1_REPAIR_CONTROLLER_READY` 仅确认CPU控制器已成功观察原队列；实际并行槽位见 `parallel_policy`，实际动作完成见原 `receipt_complete: true`。`STARTUP_PENDING` 表示后台还在启动，可关闭终端但未确认恢复成功。原生 -11 具体根因未由现有日志证明，不能承诺环境永不再发生原生故障。
