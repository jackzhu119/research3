#!/usr/bin/env bash
set -e
trap 'exit 130' INT TERM
P=${PAPER3_R9_PYTHON:-/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python}
F=${PAPER3_R9_ENTRY:-"$HOME/paper3-round9-downloads/START-PAPER3-R9-SMALL.py"}
H=44729fa3fb09d480ec09dae088113a48cec28e8a7cbfcb0bb4170f6354a60068
B=https://raw.githubusercontent.com/jackzhu119/research3/52a01cc94ac623dd5f0d4439e7b579694f04f1c8/experiments/followup_20261025_r9/delivery_small/START-PAPER3-R9-SMALL.py
Q=https://raw.githubusercontent.com/jackzhu119/research3/7689e2801dbafc7f5d55725f4a0edad031bff740/experiments/followup_20261025_r9/delivery_small/paper3-r9-derived-cache.zip
S=44db84f4313b7cb9458e163f1d9b49dc5559eab3f72dcc267f8d2c1962ef317b
mkdir -p "$(dirname "$F")"
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
while ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
  for M in https://gh-proxy.com https://ghfast.top; do
    printf 'R9_SMALL_DOWNLOAD_TRY %s (307623 bytes)\n' "$M"
    if curl -4 "${C[@]}" -fL --connect-timeout 6 --speed-time 8 --speed-limit 128 --max-time 60 "$M/$B" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then
      mv -f "$F.part" "$F"
      break
    fi
  done
  if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
    printf 'R9_DOWNLOAD_RETRY; Ctrl+C exits this download only\n'
    sleep 2
  fi
done
if [ "$#" -eq 0 ]; then set -- --submit; fi
"$P" -I -S -B -u "$F" "$@" --data-archive "https://gh-proxy.com/$Q#sha256=$S" --data-archive "https://ghfast.top/$Q#sha256=$S"
