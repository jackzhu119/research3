#!/usr/bin/env bash
set -euo pipefail
R11_PY="${PAPER3_R11_PYTHON:-/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python}"
[ -x "$R11_PY" ] || { echo "找不到学校 Python：$R11_PY" >&2; exit 1; }
R11_F="$HOME/REPAIR-PAPER3-R11-MANAGEMENT.py"
R11_SHA=547e26fb478599cb432b392ab59be4b76bdcbcffbc07d807922dfccafc97037d
R11_T="$(mktemp "$HOME/.r11-management.XXXXXX")"
trap 'rm -f "$R11_T"' EXIT
R11_CA=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem
R11_TLS=()
if [ -r "$R11_CA" ]; then R11_TLS=(--cacert "$R11_CA"); fi
r11_verified() {
  [ -f "$1" ] && printf '%s  %s\n' "$R11_SHA" "$1" | sha256sum --status -c -
}
R11_RAW=https://raw.githubusercontent.com/jackzhu119/research3/a6aab55eded2c7e45bacf7a5e582afe130fe6879/operations/round11_management_recovery_20261027/delivery/REPAIR-PAPER3-R11-MANAGEMENT.py
if ! r11_verified "$R11_F"; then
  for R11_URL in \
    https://api.github.com/repos/jackzhu119/research3/git/blobs/54e4204b03ad30e779117afa419bdb846bb72e0b \
    "https://gh-proxy.com/$R11_RAW" \
    "$R11_RAW"
  do
    printf '\n尝试下载：%s\n' "$R11_URL"
    if curl -q -fSL --retry 0 --connect-timeout 8 --max-time 45 \
      --speed-time 15 --speed-limit 256 "${R11_TLS[@]}" \
      -H 'Accept: application/vnd.github.raw+json' -o "$R11_T" "$R11_URL" &&
      r11_verified "$R11_T"
    then
      mv "$R11_T" "$R11_F"
      break
    fi
  done
fi
r11_verified "$R11_F" || { echo '下载或校验失败，未执行修复。' >&2; exit 1; }
echo 'SHA256 校验通过，开始应用修复。'
"$R11_PY" -I -S -B -u "$R11_F" --apply --submit
