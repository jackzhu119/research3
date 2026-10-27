# 学校 R11 I/O 恢复

入口固定提交：`850fd6e26adf425139402edc07a6faab3582f94b`；SHA256：`6969225b737616dd51199fa747d8223ff3b7be894f833a859b7ac5d8cbe6e731`。

在 login1 执行下列一整行。下载后核对完整 SHA，再显式恢复并允许失败 allocation 补位；不取消仍运行的作业。

```bash
( set -e; trap 'exit 130' INT TERM; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-r11-io-repair-downloads"; F="$D/REPAIR-PAPER3-R11-IO.py"; H=6969225b737616dd51199fa747d8223ff3b7be894f833a859b7ac5d8cbe6e731; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); for T in 1 2 3; do if printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then break; fi; for U in https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/850fd6e26adf425139402edc07a6faab3582f94b/operations/round11_io_repair_20261027/delivery/REPAIR-PAPER3-R11-IO.py https://raw.githubusercontent.com/jackzhu119/research3/850fd6e26adf425139402edc07a6faab3582f94b/operations/round11_io_repair_20261027/delivery/REPAIR-PAPER3-R11-IO.py; do if curl -4 "${C[@]}" -fL --connect-timeout 8 --speed-time 12 --speed-limit 128 --max-time 90 "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; done; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; "$P" -I -S -B -u "$F" --repair --submit )
```

仅安装：去掉末尾 `--repair --submit`。只读检查：改为 `--check`。

后台状态与日志：

```bash
W="$HOME/paper3-r11-io-repair/work/97e4cadb17278532"
cat "$W/STARTED.json"
tail -n 12 "$W/controller.log"
```

`IO_REPAIR_CONTROLLER_READY` 为完成首次成功队列观察；`STARTUP_PENDING` 是后台仍在启动，均不等于四张 GPU 已获分配。科学结果复用及新作业 ID 见 controller-status.json。现存旧 worker 保持运行，新补位 worker 才使用修复入口。
