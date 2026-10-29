P="$HOME/.venvs/paper3-hpc-cu121-offline/bin/python"
F="$HOME/START-PAPER3-R1-R11-RESULTS.py"
C=35d898613313d222a607e51a3312ae311ee881c3
D=operations/round1_round11_completed_results_export_20261029/START-PAPER3-R1-R11-RESULTS.py
H=75e6ba5fba3dac2456fcd5a09fe92059d39078f3481e8462d317801177645eea
CA="$HOME/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem"
TLS=()
if [ -r "$CA" ]; then TLS=(--cacert "$CA"); fi
T=$(mktemp "$HOME/.p3-r1r11-upload.XXXXXX")
OK=0
P3_UPLOAD_RC=1
for U in "https://raw.githubusercontent.com/jackzhu119/research3/$C/$D" "https://cdn.jsdelivr.net/gh/jackzhu119/research3@$C/$D" "https://api.github.com/repos/jackzhu119/research3/contents/$D?ref=$C"; do
  printf '\n正在下载：%s\n' "$U"
  if curl -fsSL --connect-timeout 6 --max-time 25 --max-filesize 1048576 -H 'Accept: application/vnd.github.raw+json' "${TLS[@]}" -o "$T" "$U" && printf '%s  %s\n' "$H" "$T" | sha256sum -c -; then
    if mv "$T" "$F"; then OK=1; fi
    break
  fi
done
if [ "$OK" -eq 1 ]; then
  "$P" -I -S -B -u "$F" --start --refresh-r1-paper
  P3_UPLOAD_RC=$?
else
  rm -f "$T"
  echo '下载或校验失败，未执行启动。'
fi

printf "\n第8轮上传状态与日志末尾（只读）\n"
for F in "$HOME/paper3-missing-results-upload/full8/status.json" "$HOME/paper3-missing-results-upload/full8/upload.log"; do
  printf "\n%s\n" "$F"
  if [ -r "$F" ]; then timeout -k 1s 6s tail -c 4000 "$F"; else printf "未找到这个上传日志\n"; fi
done
exit "$P3_UPLOAD_RC"
