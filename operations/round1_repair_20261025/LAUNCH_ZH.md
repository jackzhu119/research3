# 第一轮：应用修复并断点续排

只处理第一轮，继续hpc_gpu分区和原三独立单卡资源。第六轮不迁移、不修改。

在login1执行下面完整一行：

```bash
( set -e; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-r1-fix-downloads"; F="$D/FIX-PAPER3-R1.py"; H=b61ba1f22308ca1ab21e3f6624dcca764a08f5a87cee32b6610acdfe5334dd92; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); while ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do for U in https://raw.githubusercontent.com/jackzhu119/research3/f4d9fc0b4d20687fbd5faa2b17d427897033bd4a/operations/round1_repair_20261025/FIX-PAPER3-R1.py 'https://api.github.com/repos/jackzhu119/research3/contents/operations/round1_repair_20261025/FIX-PAPER3-R1.py?ref=f4d9fc0b4d20687fbd5faa2b17d427897033bd4a'; do if curl -4 "${C[@]}" -fL --retry 2 --retry-delay 2 --connect-timeout 10 --max-time 60 -H 'Accept: application/vnd.github.raw+json' "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then printf 'DOWNLOAD_RETRY\n'; sleep 3; fi; done; "$P" -I -S -B -u "$F" --apply --restart-workers )
```

先备份并修正真实使用的运行包，再复制到独立目录并确认新控制器接管。旧运行作业请求USR1保存退出，旧排队作业被替换，仍用原三槽调度状态续交。旧350断点和已完成源模型/工程/筛选结果沿用。

R1_WRAPPER_FIX_APPLIED确认封装已修复；R1_STANDALONE_CONTROLLER_STARTED或REUSED确认独立控制器已接管；R1_OLD_WORKER_SWITCH_REQUESTED确认已请求第一轮旧作业保存退出或替换。它们不等于GPU续训完成；新作业开始后应看到实际A100_SXM4_40GB、核验阶段和RESUMED/更新计数。

原生首次输入核验仍需时间，同一动作重复核验会复用成功结果。以后使用本入口，旧固定迁移bootstrap会重新安装旧版包。

[完整修复说明](README_ZH.md) · [实际GPU证据](ACTUAL_GPU_EVIDENCE.json)。
