# 学校R11在线修复指令

入口固定到提交 `b35636f8d7a8ab1e29b9924227f9dae4303dd78d`，SHA256 `0a8dd0776381f4bf268d384afde2b4b93027e814fa43284ba389aa881a2b9c8f`，约65KB。这个包仅恢复控制器和论文上传管理进程，原科学包、计划、模型、checkpoint、任务状态及已完成结果保留。

在login1粘贴下面**一整行**：

```bash
( set -e; trap 'exit 130' INT TERM; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-r11-repair-downloads"; F="$D/REPAIR-PAPER3-R11-OPERATIONS.py"; H=0a8dd0776381f4bf268d384afde2b4b93027e814fa43284ba389aa881a2b9c8f; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); for T in 1 2 3; do if printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then break; fi; for U in https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/b35636f8d7a8ab1e29b9924227f9dae4303dd78d/operations/round11_live_repair_20261026/delivery/REPAIR-PAPER3-R11-OPERATIONS.py https://raw.githubusercontent.com/jackzhu119/research3/b35636f8d7a8ab1e29b9924227f9dae4303dd78d/operations/round11_live_repair_20261026/delivery/REPAIR-PAPER3-R11-OPERATIONS.py; do if curl -4 "${C[@]}" -fL --connect-timeout 8 --speed-time 12 --speed-limit 128 --max-time 90 "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; done; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; "$P" -I -S -B -u "$F" --repair --submit )
```

末尾 `--repair --submit` 是显式恢复及调度补位授权。先观察原四作业与既有提交记录，已RUNNING/PENDING的作业不会重复提交。只有经UID、真实Python入口/子命令、原计划及PID启动身份核验的旧论文上传进程会接收SIGTERM；科学worker、Slurm作业、完整归档、报告及监控进程均保留。

恢复进程后台运行，可以关闭终端。`OPERATIONAL_REPAIR_READY` 表示真实管理进程已应答；`STARTUP_PENDING` 表示仍在启动或等旧上传锁，不能据此宣称恢复成功。`PARTIAL_REPAIR`/`STARTUP_FAILED` 会明确给出组件状态与日志位置，不要取消训练。重复执行会复用现有修复进程，等待中的进程也不会被重复创建。

## 只安装、检查与正式恢复

- 默认只安装：把最后 `"$P" -I -S -B -u "$F" --repair --submit` 改成 `"$P" -I -S -B -u "$F"`。
- 安装并只读检查原运行状态：最后改为 `"$P" -I -S -B -u "$F" --check`。
- 正式恢复：保留 `--repair --submit`。无submit不会启动允许新allocation补位的控制器。

默认计划为 `/public/home/zhuziyuan/projects/paper3/runs/r11_campaign/plan.json`，它与原准备计划的SHA绑定。安装在 `~/paper3-r11-operational-repair/<payload>/`；日志和就绪记录在 `~/paper3-r11-operational-repair/work/97e4cadb17278532/`。

## 检查恢复结果

```bash
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
F="$HOME/paper3-r11-repair-downloads/REPAIR-PAPER3-R11-OPERATIONS.py"
"$P" -I -S -B -u "$F" --check
# 读取修复进程日志；不停止任何任务
R="$HOME/paper3-r11-operational-repair/work/97e4cadb17278532"
tail -n 12 "$R/controller.log"
tail -n 12 "$R/paper.log"
```

`controller-ready.json`包含exact PID、recipe、启动nonce及首轮调度观察；`paper-ready.json`仅在原paper锁已持有且修补已安装后出现。就绪不代表远端上传已经完成。A远端成功以原R11 `PAPER_DATA_LATEST.json`的任务覆盖数和完整GET/SHA回执为准；B继续使用原始二进制归档。

A中的binary-base64目录明确保存base64表示及REPRESENTATIONS.json，按原始路径、字节数和SHA严格解码可恢复原NPZ；CSV/JSON原字节保留。无模型重训、算法改动或科研指标重算。

发布前34项CPU回归、低内存真实子进程复现、原调度器四作业观察零重复提交、二进制严格回还及默认安装17文件SHA验证通过。见`INSTALL_VALIDATION.json`与`../proof/FINAL_VERIFICATION.json`。这不声称学校端已应用或网络已完成恢复，实际结果由上述回执及后续GitHub监控确认。
