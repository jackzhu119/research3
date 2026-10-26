# R11 学校完整下载、安装与启动

以下入口固定到已测试代码提交 `ee34ed45f2bed4c7bcf022d9144343164b6af032`，SHA256 为 `d8e347f25026c29863be514a3f306e1821ca63f3dc6781c741055dc1a83571c2`。不依赖事先安装文件，不需要 git clone、pip 安装或下载 EEG 数据。

在学校 login1 原样粘贴下面**一整行**。末尾显式 `--start --submit` 会安装、生成 10,270 项 campaign 并启动四个独立单卡 Slurm 作业；每个 16 CPU / 64G。显示 `DETACHED_CONTROLLER_STARTED` 后可关闭终端。它表示后台控制器已应答，实际四卡分配和排队原因看 status。

```bash
( set -e; trap 'exit 130' INT TERM; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-r11-downloads"; F="$D/START-PAPER3-R11.py"; H=d8e347f25026c29863be514a3f306e1821ca63f3dc6781c741055dc1a83571c2; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); for T in 1 2 3; do if printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then break; fi; for U in https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/ee34ed45f2bed4c7bcf022d9144343164b6af032/experiments/round11_20261026/delivery/START-PAPER3-R11.py https://raw.githubusercontent.com/jackzhu119/research3/ee34ed45f2bed4c7bcf022d9144343164b6af032/experiments/round11_20261026/delivery/START-PAPER3-R11.py; do if curl -4 "${C[@]}" -fL --connect-timeout 8 --speed-time 12 --speed-limit 128 --max-time 90 "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; done; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; "$P" -I -S -B -u "$F" --start --submit )
```

下载先尝试 GitHub 代理，再试原始 GitHub，每个 URL 最多90秒，共三轮；成功SHA验证才安装。网络暂不可达可重跑同一行；已安装、已生成计划及已完成任务会复用。下载失败不会改动任何旧轮次。

## 安装、验证、正式提交的区别

- 只安装：将上面最后的 `"$P" -I -S -B -u "$F" --start --submit` 改成 `"$P" -I -S -B -u "$F"`。默认不运行测试、不准备计划、不提交。
- 可选 CPU 软件验证：同一位置改成 `"$P" -I -S -B -u "$F" --validate`。这是显式诊断，不是每次训练前置门槛。
- 安装并准备：同一位置改成 `"$P" -I -S -B -u "$F" --start`，不会提交。
- 正式运行：保留 `--start --submit`。本次研发和发布没有在学校执行它，没有申请GPU或取消旧作业。

默认项目根目录 `/public/home/zhuziyuan/projects/paper3`；计划 `runs/round11_preparation/plan-campaign.json`；科学输出 `runs/r11_campaign/`；包 `~/paper3-round11/<payload>/`。

## 查看进度与恢复

安装后运行：

```bash
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
F="$HOME/paper3-r11-downloads/START-PAPER3-R11.py"
PLAN=/public/home/zhuziyuan/projects/paper3/runs/round11_preparation/plan-campaign.json
"$P" -I -S -B -u "$F" --status --plan "$PLAN"
# allocation结束/退出后显式恢复；已完成科学任务复用
"$P" -I -S -B -u "$F" --resume --submit --plan "$PLAN"
```

论文数据优先上传 `paper3-results/results/paper3/round-11/`；完整预测、算子和轨迹独立增量归档；监控 `paper3-r11-progress/monitoring/paper3-round11/status.json`。网络失败不会终止训练或让已完成任务重训。

软件验证：286项CPU测试、实际离线安装、206文件SHA读回、生成信号native格式流程和故障回放。证明见 `INSTALL_VALIDATION.json`、`../research/FINAL_VERIFICATION.json`。学校CUDA与R11真实EEG性能尚未实测，未宣称超过EA或保证永不发生外部故障。
