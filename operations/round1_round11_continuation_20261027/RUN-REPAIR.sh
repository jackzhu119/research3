#!/usr/bin/env bash
set -euo pipefail
printf '开始下载 R1/R11 续跑修复程序，现有作业继续运行。\n'
R1_PY="${PAPER3_R1_PYTHON:-/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python}"
[ -x "$R1_PY" ] || { echo '找不到学校 Python'; exit 1; }
R1_FILE="$HOME/FIX-PAPER3-R1-R11-CONTINUATION.py"
R1_SHA=5f7822b585b5b4190533f77bef08bcfa96bcf15e5ec467ff3daf644d266e4358
R1_TEMP="$(mktemp "$HOME/.r1-continuation.XXXXXX")"
trap 'rm -f "$R1_TEMP"' EXIT
R1_CA=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem
R1_TLS=()
if [ -r "$R1_CA" ]; then R1_TLS=(--cacert "$R1_CA"); fi
r1_verified() { [ -f "$1" ] && printf '%s  %s\n' "$R1_SHA" "$1" | sha256sum --status -c -; }
R1_RAW=https://raw.githubusercontent.com/jackzhu119/research3/2e4cd33b56b19d72a5e8a9723e4ab60b650aeb6f/operations/round1_round11_continuation_20261027/delivery/FIX-PAPER3-R1-R11-CONTINUATION.py
if ! r1_verified "$R1_FILE"; then
  for R1_URL in https://api.github.com/repos/jackzhu119/research3/git/blobs/1d41b57f34e449fdf6bfdd2698f84daede967b2b "https://gh-proxy.com/$R1_RAW"; do
    printf '尝试线路：%s\n' "$R1_URL"
    if curl -q -fSL --retry 0 --connect-timeout 8 --max-time 45 --speed-time 15 --speed-limit 256 "${R1_TLS[@]}" -H 'Accept: application/vnd.github.raw+json' -o "$R1_TEMP" "$R1_URL" && r1_verified "$R1_TEMP"; then
      mv "$R1_TEMP" "$R1_FILE"
      break
    fi
  done
fi
r1_verified "$R1_FILE" || { echo '下载或 SHA256 校验失败，未执行修复。'; exit 1; }
printf 'SHA256 校验通过，开始逐项检查并安装续跑钩子。\n'
"$R1_PY" -I -S -B -u "$R1_FILE" --apply
