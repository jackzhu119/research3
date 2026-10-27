#!/usr/bin/env bash
set -euo pipefail
R11_PY="${PAPER3_R11_PYTHON:-/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python}"
[ -x "$R11_PY" ] || { echo "找不到学校 Python：$R11_PY" >&2; exit 1; }
R11_T="$(mktemp "$HOME/.r11-management.XXXXXX")"
trap 'rm -f "$R11_T"' EXIT
R11_F="$HOME/REPAIR-PAPER3-R11-MANAGEMENT.py"
R11_CA=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem
R11_TLS=()
if [ -r "$R11_CA" ]; then R11_TLS=(--cacert "$R11_CA"); fi
R11_U="https://raw.githubusercontent.com/jackzhu119/research3/a6aab55eded2c7e45bacf7a5e582afe130fe6879/operations/round11_management_recovery_20261027/delivery/REPAIR-PAPER3-R11-MANAGEMENT.py"
for R11_LINK in "$R11_U" "https://gh-proxy.com/$R11_U"; do
  if curl -fSL --retry 2 --connect-timeout 15 --max-time 180 "${R11_TLS[@]}" -o "$R11_T" "$R11_LINK" &&
     printf '%s  %s\n' '547e26fb478599cb432b392ab59be4b76bdcbcffbc07d807922dfccafc97037d' "$R11_T" | sha256sum -c -
  then
    mv "$R11_T" "$R11_F"
    "$R11_PY" -I -S -B -u "$R11_F" --apply --submit
    exit 0
  fi
done
echo '下载或校验失败，未应用修复。' >&2
exit 1
