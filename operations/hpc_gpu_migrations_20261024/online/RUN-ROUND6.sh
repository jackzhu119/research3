#!/bin/bash
(
set -e
R=6
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-hpcgpu-downloads"
mkdir -p "$D"
F="$D/START-HPCGPU-MIGRATIONS.py"
H=988abd811a930eaf6c346711d19888feee86a5af02c5273c7e64ee4bd757920a
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
  T=$(mktemp "$D/.bootstrap.XXXXXX")
  trap 'rm -f "${T:-}"' EXIT
  until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
    for U in \
      'https://raw.githubusercontent.com/jackzhu119/research3/1d3e4860fc8a138e7be2717dae75ed55a0657e47/operations/hpc_gpu_migrations_20261024/online/START-HPCGPU-MIGRATIONS.py' \
      'https://api.github.com/repos/jackzhu119/research3/contents/operations/hpc_gpu_migrations_20261024/online/START-HPCGPU-MIGRATIONS.py?ref=1d3e4860fc8a138e7be2717dae75ed55a0657e47'
    do
      if curl -4 "${C[@]}" -fL --retry 1 --retry-delay 2 --connect-timeout 10 --max-time 60 \
        -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" &&
        printf '%s  %s\n' "$H" "$T" | sha256sum -c -
      then
        mv -f "$T" "$F"
        break
      fi
    done
    if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then sleep 5; fi
  done
fi
"$P" -I -S -B -u "$F" --round "$R" --download-dir "$D" --submit
)
