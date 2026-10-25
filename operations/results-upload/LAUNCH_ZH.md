# 第二、三、七轮完整结果上传启动命令

在学校 login1 执行下面整段。脚本下载有raw/API备用入口，校验后后台启动；自动使用原有GitHub凭据。

```bash
(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-results-upload"
mkdir -p "$D"
H=a42192b5f5ff658544906fce4604199a19413c793cdcc46f3881b5e040100bda
F="$D/UPLOAD-${H:0:12}.py"
R=7957830dae6f082484db376d3c9cb813016edd36
Q=operations/results-upload/UPLOAD-PAPER3-RESULTS.py
if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
  T=$(mktemp "$D/.download.XXXXXX")
  trap 'rm -f "${T:-}"' EXIT
  A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
  C=(); if [ -f "$A" ]; then C=(--cacert "$A"); fi
  for U in "https://raw.githubusercontent.com/jackzhu119/research3/$R/$Q" "https://api.github.com/repos/jackzhu119/research3/contents/$Q?ref=$R"
  do
    if curl -4 "${C[@]}" -fL --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" && printf '%s  %s\n' "$H" "$T" | sha256sum -c -
    then
      mv -f "$T" "$F"
      break
    fi
  done
fi
printf '%s  %s\n' "$H" "$F" | sha256sum -c -
"$P" -I -S -B -u "$F" --start --rounds 2 3 7
)
```

查看进度：

```bash
tail -n 40 "$HOME/paper3-results-upload/upload.log"
```

看到 `ALL_REQUESTED_ROUNDS_UPLOADED` 才代表三轮全部上传并验证完成。中断后执行同一条启动命令；完整分片复用，失败分片重传。原来的各轮监控分支及快照继续保留。

结果入口：[按轮分类的结果目录](https://github.com/jackzhu119/research3/tree/paper3-results/results/paper3)。每轮 `previous/` 是旧回传，`final/` 是完整上传；`LATEST.json` 是该轮完成索引。
