# 第四轮可选Weibo外部验证

核心204任务不等待外部数据。建议核心完成后运行独立21项外测，仍为1节点1作业3GPU。复用第三轮已公开核验的8.4MB缓存，不重新下载4.34GB原MAT。

```bash
(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-round4-data"
mkdir -p "$D"
A="$D/weibo2014_lr_clinic11_v1.tar.gz"
CA="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$CA" ] || C=(--cacert "$CA")
curl -4 "${C[@]}" -fL --retry 3 --connect-timeout 10 --max-time 300 'https://raw.githubusercontent.com/jackzhu119/research3/45d4e031065a266179d8e25bd053068062f17322/data/paper3_round3/weibo2014_lr_clinic11_v1.tar.gz' -o "$A"
printf '%s  %s\n' '253bfe3e09e19bd72a456474f426273c37e4ee20f2cb1dd38f441c4ca9a4a34b' "$A" | sha256sum -c -
tar -xzf "$A" -C "$D"
"$P" -I -S -B -u "$HOME/paper3-round4/4e8f440ae0c016b94186/START.py" --external-manifest "$D/weibo/manifest.json" --basis-run /public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r4 --partition hpc_gpu1 --gpus 3 --cpus 16 --memory 64G --submit
)
```

发布顺序LR前缀稳健性，不是已确认真实采集前20条的前瞻验证；详细许可/身份边界保留在EXTERNAL_DATA_NOTES.md与compatibility/weibo_*。主机内存按作业192G申请，64G为每槽预算。
