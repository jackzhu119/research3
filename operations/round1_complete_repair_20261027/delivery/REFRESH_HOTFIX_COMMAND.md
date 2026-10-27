# 第一轮停止自动刷新修正

固定代码提交：`9ef67dfcbd6825c3396deb324a26a91981940918`；入口 SHA256：`f2f202c1330296e45defed4f6bf4d3e3e095b6623a4012a1b3ef938d71c2be0d`；大小：35516字节。

已部署原完整修复的用户在 login1 执行一整行。只更新后台管理代码，不取消、重提或打断当前三个科学作业。它停止身份确认的旧管理进程，清理已识别的旧刷新管理步骤，在独立目录启动无自动刷新的控制器；原固定包、科学代码、完成回执和检查点保持不变。

```bash
( set -e; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-r1-full-repair-downloads"; F="$D/HOTFIX-PAPER3-R1-REFRESH.py"; H=f2f202c1330296e45defed4f6bf4d3e3e095b6623a4012a1b3ef938d71c2be0d; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then for U in https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/9ef67dfcbd6825c3396deb324a26a91981940918/operations/round1_complete_repair_20261027/delivery/HOTFIX-PAPER3-R1-REFRESH.py https://raw.githubusercontent.com/jackzhu119/research3/9ef67dfcbd6825c3396deb324a26a91981940918/operations/round1_complete_repair_20261027/delivery/HOTFIX-PAPER3-R1-REFRESH.py; do if curl -4 "${C[@]}" -fL --connect-timeout 8 --max-time 90 "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; fi; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; "$P" -I -S -B -u "$F" --apply )
```

`R1_REFRESH_HOTFIX_CONTROLLER_READY` 表示新管理进程已成功观察原队列；`R1_REFRESH_HOTFIX_ALREADY_RUNNING` 且 `controller_ready: true` 表示已经完成过更新。`STARTUP_PENDING` 只表示后台启动尚未确认，不能当成训练恢复成功。重复执行同一命令可恢复有记录的管理启动失败，不会再发送科学子进程退出信号。

学校记录：`~/paper3-r1-full-repair/work/refresh-hotfix-applied.json`，后台日志仍为同目录 `controller.log`，应出现 `R1_AUTOMATIC_CHILD_REFRESH_DISABLED`。历史 `refresh-results/` 不再写入。

验证：151项CPU测试通过，包含真实管理进程交接和新管理进程启动失败后的重试，独立模拟计算持续推进且checkpoint/receipt/scheduler字节不变；实际生成入口经隔离Python导入、原包manifest核对和SHA验证。详情见 `proof/REFRESH_HOTFIX_VALIDATION.json`。本地未使用学校GPU，原生 `-11` 根因仍未证明，不能把测试通过等同于未来永不中断。
