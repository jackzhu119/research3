# 可选Weibo发布缓存与迁移

核心174任务不依赖外部缓存。建议核心完成后，再运行独立18项外部评价；它们读本轮完成字典，不重训源模型。以下命令先下载固定提交的8.4MB衍生缓存并校验，再解包，最后提交外测。

```bash
(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-round3-data"
mkdir -p "$D"
A="$D/weibo2014_lr_clinic11_v1.tar.gz"
CA="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$CA" ] || C=(--cacert "$CA")
curl -4 "${C[@]}" -fL --retry 3 --connect-timeout 10 --max-time 300 'https://raw.githubusercontent.com/jackzhu119/research3/45d4e031065a266179d8e25bd053068062f17322/data/paper3_round3/weibo2014_lr_clinic11_v1.tar.gz' -o "$A"
printf '%s  %s\n' '253bfe3e09e19bd72a456474f426273c37e4ee20f2cb1dd38f441c4ca9a4a34b' "$A" | sha256sum -c -
tar -xzf "$A" -C "$D"
"$P" -I -S -B -u "$HOME/paper3-round3/b86c0e0c6e9ea076c4fc/START.py" --external-manifest "$D/weibo/manifest.json" --basis-run /public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r3 --gpus 3 --cpus 12 --submit
)
```

数据许可CC0-1.0，原DOI10.7910/DVN/27306，10人/1580LR/B20支持200/固定查询1380。只能解释为发布顺序LR前缀的离线稳健性；真实物理采集前20条及run时序未知，不是新的前瞻确认。该外测代码使用单独输出目录与GitHub `weibo-status.json`，不会覆盖核心status。
