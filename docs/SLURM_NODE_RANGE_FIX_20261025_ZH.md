# 修复学校 Slurm 单节点范围格式并接管现有九个作业

## 已确认原因与影响

学校 Slurm 20.11 将 held 单节点显示为 `NumNodes=1-1`，同时 `TRES=node=1`、`NumCPUs=12`。旧交接解析器只接受字符串 `1`，将合法请求误判为资源变化，导致新作业继续hold而旧作业未交接。此为程序兼容性错误。

修复同时支持 `1`、`1-1`，仍拒绝`1-2`、`2`、`0`。三轮的分区、CPU、内存、GPU数及任务归属保持原请求；错误日志记录实际字段，第一轮GitHub监控能在没有progress文件时识别真实`paper3r1p-`作业。

| 轮次 | 已有替代jobID | 目的分区 | 资源 |
|---|---|---|---|
| 1 | 772076、772077、772078 | hpc_gpu / A100 PCIe40GB | 各1GPU/12CPU，内存43G、43G、42G |
| 5 | 772070、772071、772072 | hpc_gpu1 / A100 SXM80GB | 各1GPU/12CPU/64G |
| 6 | 772073、772074、772075 | hpc_gpu1 / A100 SXM80GB | 各1GPU/12CPU/64G |

## login1执行一条单行命令

```bash
( set -e; P=/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python; D="$HOME/paper3-parallel-downloads"; mkdir -p "$D"; H=1090856b7a22a21250d04f1c09c788a20e9112a35a90f2cd92e218c62a2c5413; F="$D/REPAIR-NODES-${H:0:12}.py"; R=6315d2fa96b0b541009a7492b92fcd9841f91b2a; Q=operations/parallel_20261025/online/REPAIR-PAPER3-NODES.py; A="${P%/bin/python}/lib/python3.12/site-packages/botocore/cacert.pem"; C=(); if [ -f "$A" ]; then C=(--cacert "$A"); fi; T=""; trap 'rm -f "${T:-}"' EXIT; until printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do T=$(mktemp "$D/.download.XXXXXX"); for U in "https://raw.githubusercontent.com/jackzhu119/research3/$R/$Q" "https://api.github.com/repos/jackzhu119/research3/contents/$Q?ref=$R"; do if curl -4 "${C[@]}" -fL --retry 2 --retry-delay 2 --connect-timeout 10 --max-time 90 -H 'Accept: application/vnd.github.raw+json' "$U" -o "$T" && printf '%s  %s\n' "$H" "$T" | sha256sum -c -; then mv -f "$T" "$F"; break; fi; done; if ! printf '%s  %s\n' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then rm -f "$T"; printf 'DOWNLOAD_RETRY\n'; sleep 3; fi; done; printf '%s  %s\n' "$H" "$F" | sha256sum -c -; cp "$F" "$D/START-PAPER3-PARALLEL.py"; "$P" -I -S -B -u "$F" --repair --execute )
```

安装新独立目录后，复核现场已有三份held作业，替换本轮login侧旧交接controller与监控。保存原启动记录和调度快照，使用原runtime、marker、slot、attempt与jobID接管原九个作业。旧fa00安装保持，因为已提交作业的worker argv仍引用它；新安装用于管理器及未来续跑任务。

接管程序不直接提交、取消或放行Slurm作业。修复后的原控制器按原顺序保存旧训练checkpoint、等待其退出、保存第一轮迁移边界，随后release现有新作业。已完成结果不重新训练，不运行全量pytest、pilot、数据树SHA或依赖安装。累计12小时上限没有恢复。第4轮、第2/3/7轮上传进程不属于替换对象。

看到 `FIXED_CONTROLLERS_STARTED_EXISTING_JOBS_PRESERVED` 表示对应轮次的管理器替换完成；最终放行和实际GPU运行以队列和原控制器日志为准。若学校GPU配额导致`AssocGrpGRES`，属于等待资源。

## 验证与证据

本地109项R1、60项R5/6、12项接管测试全部通过，包含本次真实三份scontrol输出，以及真实隔离子进程的PID/锁替换。实际发布文件经raw/API双路下载、SHA256、21个安装模块逐一验证，三轮prepare保持科学文件字节和mtime不变。本页没有声称学校GPU修复已经执行。

## 第2/3/7轮上传情况（用户日志北京时间11:55:37）

上传进程645884仍存在，第二轮`PACK_PROGRESS`持续增长：31573/152481个文件（20.7%），压缩输出170427199字节。已完成5个32MiB本地分片，第6片正在写入；`status.json`尚未写入是初始打包阶段的实现行为。三轮均无`UPLOADED`回执，第二轮尚在打包，第三与第七轮按原顺序等待。

打包完成后原上传器会上传并验证Release分片，最后写每轮完整`LATEST.json`和本地`round-NN-UPLOADED.json`；当前不能把本地压缩分片当作远端备份完成。

继续查看（只读）：

```bash
timeout -k 2s 10s tail -n 15 "$HOME/paper3-results-upload/upload.log"
```
