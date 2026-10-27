#!/usr/bin/env bash
set -euo pipefail
M12_PY="${PAPER3_M12_PYTHON:-/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python}"
[ -x "$M12_PY" ] || { echo "找不到学校 Python：$M12_PY" >&2; exit 1; }
M12_FILE="$HOME/START-PAPER3-M12.py"
M12_TMP="$(mktemp "$HOME/.paper3-m12.XXXXXX")"
trap 'rm -f "$M12_TMP"' EXIT
M12_CA=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem
M12_TLS=()
if [ -r "$M12_CA" ]; then M12_TLS=(--cacert "$M12_CA"); fi
for M12_URL in \
  "https://raw.githubusercontent.com/jackzhu119/research3/653dc25a2f09ed11361fe71d795f4fd31a47f1ba/experiments/controlled_mechanism_20261027/delivery/START-PAPER3-M12.py" \
  "https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/653dc25a2f09ed11361fe71d795f4fd31a47f1ba/experiments/controlled_mechanism_20261027/delivery/START-PAPER3-M12.py"
do
  if curl -fSL --retry 2 --connect-timeout 15 --max-time 180 "${M12_TLS[@]}" -o "$M12_TMP" "$M12_URL" &&
     printf '%s  %s\n' '22c1ee59a978b43a04edeef338c1a79cb002ee31eeb7f3ce3ca20146d469c886' "$M12_TMP" | sha256sum -c -
  then
    mv "$M12_TMP" "$M12_FILE"
    "$M12_PY" -I -S -B -u "$M12_FILE" --prepare --submit
    exit 0
  fi
done
echo '下载或校验失败，尚未提交作业。' >&2
exit 1
