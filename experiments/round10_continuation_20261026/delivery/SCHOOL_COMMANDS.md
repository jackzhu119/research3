# R10优化续跑：完整下载、安装及正式提交

原机制15项保留，复用其选型数据和seed18参考原字节，接续新的165项探索任务。资源4个独立单卡作业，每个16CPU/64G。固定代码提交 `7f04b4c4905b1cbfe51febd477c1944c19c1a6e0`，入口SHA256 `ce9ce5026ac571a625bcc538a10ab5a78eaeb4d39538441a1799ff0388be1f66`。

在学校login1粘贴这一整行。末尾显式 `--continue --submit` 会完成下载、安装、原证据一次性精简核验、修订选型、准备新计划及正式提交。不是重新执行15项机制实验。显示 `DETACHED_CONTROLLER_STARTED` 后可以关闭终端；GPU何时分配由学校队列决定。

```bash
( set -e; trap 'exit 130' INT TERM; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-r10-optimized-downloads"; F="$D/START-PAPER3-R10-OPTIMIZED.py"; H=ce9ce5026ac571a625bcc538a10ab5a78eaeb4d39538441a1799ff0388be1f66; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); for T in 1 2 3; do if printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then break; fi; for U in https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/7f04b4c4905b1cbfe51febd477c1944c19c1a6e0/experiments/round10_continuation_20261026/delivery/START-PAPER3-R10-OPTIMIZED.py https://raw.githubusercontent.com/jackzhu119/research3/7f04b4c4905b1cbfe51febd477c1944c19c1a6e0/experiments/round10_continuation_20261026/delivery/START-PAPER3-R10-OPTIMIZED.py; do if curl -4 "${C[@]}" -fL --connect-timeout 8 --speed-time 20 --speed-limit 128 --max-time 300 "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; done; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; "$P" -I -S -B -u "$F" --continue --submit )
```

包约3.2MB；先尝试代理，后尝试固定GitHub地址，下载SHA匹配才执行。每地址有300秒传输上限，三轮后失败会退出，可重跑同一行。重复继续复用已安装包、已生成计划及已完成科学任务，不重复选型或新建重复作业。

## 安装、准备、验证、提交

- 只安装：把最后 `"$P" -I -S -B -u "$F" --continue --submit` 改成 `"$P" -I -S -B -u "$F"`。
- 安装并准备：最后改为 `"$P" -I -S -B -u "$F" --continue`，不提交。
- 显式CPU软件诊断：最后改为 `"$P" -I -S -B -u "$F" --validate`；正式启动不重复全套测试。
- 正式继续：保留 `--continue --submit`，无需另行运行freeze或手动切换阶段。

默认原计划 `ROOT/runs/r10_mechanism/plan.json`；`ROOT=/public/home/zhuziyuan/projects/paper3`。续跑新目录 `ROOT/runs/r10_formal_exploratory_optimized/`，计划为其中`plan.json`，安装到 `~/paper3-round10-optimized/<payload>/`。不覆盖原机制或R11。原计划文件SHA、结果seal及选型所需compactCSV只核验一次，不重复扫描完整EEG/预测和上游合同。

## 状态、恢复及上传

```bash
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
F="$HOME/paper3-r10-optimized-downloads/START-PAPER3-R10-OPTIMIZED.py"
PLAN=/public/home/zhuziyuan/projects/paper3/runs/r10_formal_exploratory_optimized/plan.json
"$P" -I -S -B -u "$F" --status --plan "$PLAN"
# allocation结束或断点继续；只在显式submit下启动
"$P" -I -S -B -u "$F" --resume --plan "$PLAN" --submit
# 独立恢复双通道，不重训
"$P" -I -S -B -u "$F" --export --plan "$PLAN"
```

报告与A/B独立后台，网络失败不停止训练或重训。结果在`paper3-results/results/paper3/round-10/stages/formal_exploratory/<新recipe>/`，原mechanism条目保留；监控回传`paper3-r10-progress`。历史基线缺少已冻结artifact时单独隔离对应方法/seed，固定比较保留NOT_READY，其余任务继续；不能把部分覆盖写成全部165科学成功。

验证：291项CPU回归，224文件SHA读回，实际installerCPU自检，安装API夹具165项准备/幂等恢复，3seed×18方法native生成信号预测/metrics/seal。证明见`INSTALL_VALIDATION.json`及`../research/CONTINUATION_FINAL_VERIFICATION.json`。尚未在学校CUDA执行新165实验，未宣称新方法优于EA。本修订是在原开发结果已见后登记，后续旧holdout/Pan/Weibo仍是探索性。
