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

apply_fix FIX-PAPER3-R9-CUMSUM.py __R9_SHA__ __R9_PRIMARY__ __R9_MIRROR__
apply_fix FIX-PAPER3-R1-STARTUP-CACHE.py __R1_SHA__ __R1_PRIMARY__ __R1_MIRROR__
printf 'R1_R9_REPAIRS_INSTALLED; existing jobs continue and future child processes adopt corrections.\n'
