# 第五轮可选外部评价

核心105任务不依赖下载外部缓存。完成后可用同一冻结Weibo缓存运行21项跨数据集评价；仍为hpc_gpu1单节点单作业3GPU、总36CPU164G。下面只取约8.4MB已核验衍生缓存，不重新下载原MAT。

```bash
(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-round5-data"
mkdir -p "$D"
A="$D/weibo2014_lr_clinic11_v1.tar.gz"
CA="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$CA" ] || C=(--cacert "$CA")
curl -4 "${C[@]}" -fL --retry 3 --connect-timeout 10 --max-time 300 'https://raw.githubusercontent.com/jackzhu119/research3/45d4e031065a266179d8e25bd053068062f17322/data/paper3_round3/weibo2014_lr_clinic11_v1.tar.gz' -o "$A"
printf '%s  %s\n' '253bfe3e09e19bd72a456474f426273c37e4ee20f2cb1dd38f441c4ca9a4a34b' "$A" | sha256sum -c -
tar -xzf "$A" -C "$D"
"$P" -I -S -B -u "$HOME/paper3-round5/ba5fab6bfa849d9ecaeb/START.py" --external-manifest "$D/weibo/manifest.json" --basis-run /public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5 --partition hpc_gpu1 --gpus 3 --cpus 12 --memory-total 164G --submit
)
```

发布顺序LR子任务稳健性，真实物理采集时序未知，不作为前瞻确认。来源与CC0许可记录在EXTERNAL_DATA_NOTES.md及compatibility。第五轮不会写前四轮目录；核验下载只在独立外部准备步骤，不进入核心启动链。
