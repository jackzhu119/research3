#!/usr/bin/env bash
set -e
trap 'exit 130' INT TERM
P=${PAPER3_FIX_PYTHON:-/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python}
D=${PAPER3_FIX_DOWNLOADS:-"$HOME/paper3-r19-fix-downloads"}
mkdir -p "$D"
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=()
[ ! -f "$A" ] || C=(--cacert "$A")
FIX_ARGS=("$@")

apply_fix() {
  local NAME=$1 H=$2 PRIMARY=$3 MIRROR=$4 F
  F="$D/$NAME"
  while ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
    for U in "$PRIMARY" "$MIRROR"; do
      if curl -4 "${C[@]}" -fL --connect-timeout 6 --speed-time 8 --speed-limit 128 --max-time 40 "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then
        mv -f "$F.part" "$F"
        break
      fi
    done
    if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
      printf 'REPAIR_DOWNLOAD_RETRY %s\n' "$NAME"
      sleep 2
    fi
  done
  "$P" -I -S -B -u "$F" "${FIX_ARGS[@]}"
}

apply_fix FIX-PAPER3-R9-CUMSUM.py 4f1245f4f5c05fbbbf06474fa9e5e40916ec19ce3d134dc2f8dae3c6e656bb54 https://paste.rs/lhcCa https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/df88cc0e0608b8f1462b0c6339808017d8667c87/operations/round9_cumsum_repair_20261026/FIX-PAPER3-R9-CUMSUM.py
apply_fix FIX-PAPER3-R1-STARTUP-CACHE.py e22fe043de5e7a979fa4bdcacb3acadd4e5b5c40ba862c29df704e8253f3641e https://paste.rs/IipNg https://gh-proxy.com/https://raw.githubusercontent.com/jackzhu119/research3/a983de7eb1c831b0ba1802f126e56f4fb6e2f4de/operations/round1_startup_cache_repair_20261026/FIX-PAPER3-R1-STARTUP-CACHE.py
printf 'R1_R9_REPAIRS_INSTALLED; existing jobs continue and future child processes adopt corrections.\n'
