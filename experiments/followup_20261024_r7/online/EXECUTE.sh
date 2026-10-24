#!/bin/bash
(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-r7-downloads"
mkdir -p "$D"
F="$D/DOWNLOAD-AND-START-R7.py"
H=61566627c1c155ef40d42c4c71089dc09355a46121837b7d31731e8aa722d092
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
  T=$(mktemp "$D/.bootstrap.XXXXXX")
  trap 'rm -f "${T:-}"' EXIT
  until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
    for U in \
      'https://raw.githubusercontent.com/jackzhu119/research3/0ecb1bbac0e6365625f401e71604e4ddec6baf41/experiments/followup_20261024_r7/online/DOWNLOAD-AND-START-R7.py' \
      'https://api.github.com/repos/jackzhu119/research3/contents/experiments/followup_20261024_r7/online/DOWNLOAD-AND-START-R7.py?ref=0ecb1bbac0e6365625f401e71604e4ddec6baf41'
    do
      if curl -4 "${C[@]}" -fL --retry 1 --retry-delay 2 \
        --connect-timeout 10 --max-time 60 \
        -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" &&
        printf '%s  %s\n' "$H" "$T" | sha256sum -c -
      then
        mv -f "$T" "$F"
        break
      fi
    done
    if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
      printf 'Download will retry in 10 seconds; Ctrl+C preserves the training bundle cache.\n'
      sleep 10
    fi
  done
fi
"$P" -I -S -B -u "$F" --download-dir "$D" --partition hpc_gpu --gpus 3 --cpus 12 --memory-total 128G --submit
)
