(
set -e
P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
D="$HOME/paper3-followup-downloads"
mkdir -p "$D"
F="$D/DOWNLOAD-AND-START-R6.py"
H=f4fc457acef0fa0f479dd1db95bbf250a40276618e844e6261d585399c0fb140
A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"
C=(); [ ! -f "$A" ] || C=(--cacert "$A")
trap 'rm -f "${T:-}"' EXIT
while ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do
  T=$(mktemp "$D/.r6-bootstrap.XXXXXX")
  for U in \
    'https://raw.githubusercontent.com/jackzhu119/research3/d5973a6fa7aa3d6a1bc3892b209eaa908fae8c0e/experiments/followup_20261024_r6/online/DOWNLOAD-AND-START-R6.py' \
    'https://api.github.com/repos/jackzhu119/research3/contents/experiments/followup_20261024_r6/online/DOWNLOAD-AND-START-R6.py?ref=d5973a6fa7aa3d6a1bc3892b209eaa908fae8c0e'
  do
    if curl -4 "${C[@]}" -fL --retry 2 --retry-delay 2 \
      --connect-timeout 10 --max-time 60 \
      -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" &&
      printf '%s  %s\n' "$H" "$T" | sha256sum -c -; then
      mv -f "$T" "$F"
      break
    fi
  done
  rm -f "$T"
  if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then
    printf '启动器连接暂不可用，5秒后重试；尚未提交作业。\n'
    sleep 5
  fi
done
"$P" -I -S -B -u "$F" \
  --partition hpc_gpu1 --gpus 3 --cpus 12 --memory-total 164G --submit
)
