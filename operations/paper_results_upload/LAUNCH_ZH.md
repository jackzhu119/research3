# 第2、3、7轮论文报告优先上传

在学校login1复制执行以下一整行。上传器转入后台，完整输出备份继续独立运行；本入口只上传已有报告与完整表格，不申请GPU、不重新统计。

```bash
( set -e; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-paper-data-downloads"; F="$D/START-PAPER3-PAPER-DATA.py"; H=a7bc54c925ad6d4dbce5ab5cb06390043a83b9724a6b1cb41f36b4d6ced27494; mkdir -p "$D"; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); [ ! -f "$A" ] || C=(--cacert "$A"); while ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do for U in https://raw.githubusercontent.com/jackzhu119/research3/0eb552bc361542c3ae7ac3712dfa780808f04759/operations/paper_results_upload/START-PAPER3-PAPER-DATA.py 'https://api.github.com/repos/jackzhu119/research3/contents/operations/paper_results_upload/START-PAPER3-PAPER-DATA.py?ref=0eb552bc361542c3ae7ac3712dfa780808f04759'; do if curl -4 "${C[@]}" -fL --retry 2 --retry-delay 2 --connect-timeout 10 --max-time 60 -H 'Accept: application/vnd.github.raw+json' "$U" -o "$F.part" && printf '%s  %s\n' "$H" "$F.part" | sha256sum -c -; then mv -f "$F.part" "$F"; break; fi; done; if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then printf 'DOWNLOAD_RETRY\n'; sleep 3; fi; done; "$P" -I -S -B -u "$F" --start --rounds 2 3 7 )
```

看到PAPER_UPLOADER_STARTED只表示后台进程启动。日志出现PAPER_REPORTS_UPLOADED才表示对应轮次文件已发布。

查看日志：

```bash
tail -n 30 /public/home/zhuziyuan/paper3-paper-data-upload/upload.log
```

后续Codex先读取paper3-results分支 results/paper3/round-02、round-03、round-07 中的PAPER_DATA_ZH.md与PAPER_DATA_LATEST.json，再按清单读完整CSV或分片。论文数据入口在学校上传前尚不存在。

[内容范围和限制](README_ZH.md) · [现有摘要与缺失数据审计](RESULT_REQUIREMENTS_ZH.md)。
