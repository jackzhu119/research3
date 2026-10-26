# 学校R10固定版本操作

资源为4个独立作业，各1GPU/16CPU/68G。以下下载使用固定源码提交，SHA锁定入口；默认仅安装，不导入Torch、不申请GPU、不提交Slurm，失败给出错误而不无限等待。

## 1. 一行安装（login1可执行）

```bash
( set -e; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-r10-downloads"; F="$D/START-PAPER3-R10.py"; H=37c899dc1f6d52af7e0258166777cdeb8d835561f8396629430e4237189abb22; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then for U in https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/a2f1d75a3cee62d225d6ab224d972dafebe153cc/experiments/round10_20261026/delivery/START-PAPER3-R10.py https://raw.githubusercontent.com/jackzhu119/research3/a2f1d75a3cee62d225d6ab224d972dafebe153cc/experiments/round10_20261026/delivery/START-PAPER3-R10.py; do if curl -4 "${C[@]}" -fL --connect-timeout 8 --speed-time 12 --speed-limit 128 --max-time 90 "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; fi; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; "$P" -I -S -B -u "$F" )
```

入口包含全部代码、测试、设计文档和紧凑证据，无需pip、git clone或下载原始数据。安装成功显示`R10_INSTALL_COMPLETE`和`R10_CODE_READY_NO_EXECUTION`。

## 2. 可选一次CPU验证

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/paper3-r10-downloads/START-PAPER3-R10.py" --validate
```

这不是正式运行的强制前置条件，运行时不自动重复全套测试。GPU预检只有已有allocation内显式调用，不在登录节点隐式申请GPU。

## 3. 准备15项开发机制实验（不提交）

```bash
( set -e; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; S=$("$P" -I -S -B -c 'import json,pathlib; print(pathlib.Path(json.loads((pathlib.Path.home()/"paper3-round10/CURRENT.json").read_text())["package"])/"START.py")'); "$P" -I -S -B -u "$S" plan --stage mechanism )
```

成功为`PREPARED_NOT_SUBMITTED`、15项任务、4作业、各16CPU/68G。只查已有冻结收据和小型元数据；不下载数据，不修改旧作业或结果。

## 4. 显式正式提交，随后可关闭终端

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/paper3-r10-downloads/START-PAPER3-R10.py" --submit --plan /public/home/zhuziyuan/projects/paper3/runs/round10_preparation/plan-mechanism.json
```

`DETACHED_CONTROLLER_STARTED`表示后台控制器已启动，实际sbatch确认随后记录在runtime中。控制器、CPU报告、双通道上传与monitor均脱离终端；网络失败不影响训练。学校队列决定实际并发数。完成后按[实现文档](../R10_IMPLEMENTATION.md)冻结开发结果，再分别显式提交165项正式探索及新数据57项确认；不会直接展开数千任务。

## 5. 状态与断点继续

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/paper3-r10-downloads/START-PAPER3-R10.py" --status --plan /public/home/zhuziyuan/projects/paper3/runs/round10_preparation/plan-mechanism.json
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/paper3-r10-downloads/START-PAPER3-R10.py" --resume --submit --plan /public/home/zhuziyuan/projects/paper3/runs/round10_preparation/plan-mechanism.json
```

恢复复用完成会话和封存预测，不重新跑已完成训练。首次真实学校GPU/EEG执行尚未发生，不能把本包CPU验证当作实际算法提升。
