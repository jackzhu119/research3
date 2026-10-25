# 第二、三、七轮完整结果上传

此上传器读取学校服务器现成结果，按轮推送到 `jackzhu119/research3` 的 `paper3-results` 分支。它不导入实验程序，不运行训练、pytest、pilot 或统计汇总，也不调用 Slurm。

## 仓库结果分区

- `results/paper3/round-NN/previous/<commit>/`：以前监控自动回传的原始快照及来源校验。原分支和历史继续保留。
- `results/paper3/round-NN/final/<snapshot_id>/`：本次完整结果的计划、已有报告/CSV、任务索引和逐文件清单。
- `results/paper3/round-NN/LATEST.json`：所有分片上传并校验、Git索引提交后才出现的完整上传入口。
- GitHub Release `paper3-results-rNN-<snapshot_id>`：完整科学输出的32 MiB tar.gz分片。

每轮完整归档包含所有计划任务的逐试次预测、标签、概率、验证历史、选择文件及checkpoint。原始EEG、轮次目录之外的共享源模型、安装环境、操作日志、私密配置和符号链接不属于此归档；排除项写入清单。

## 使用

线上粘贴命令见 [LAUNCH_ZH.md](LAUNCH_ZH.md)。已有脚本时可运行：

```bash
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u UPLOAD-PAPER3-RESULTS.py --start --rounds 2 3 7
```

默认读取 `/public/home/zhuziyuan/projects/paper3`，工作目录 `~/paper3-results-upload`。自动发现原安装中已有的 `private/github.json`；只使用token，目标分支固定为 `paper3-results`。凭据不写入结果。

程序后台运行，退出SSH不依赖终端保持连接。日志：

```bash
tail -n 40 "$HOME/paper3-results-upload/upload.log"
```

`UPLOAD_BACKGROUND_STARTED` 仅代表后台进程已启动。`ROUND_UPLOAD_COMPLETE` 代表对应轮次已上传；`ALL_REQUESTED_ROUNDS_UPLOADED` 才代表三轮都已上传。同一启动命令可重用，上传器有独立锁；已校验的完整分片会复用，失败分片重新上传。GitHub资产上传没有通用字节级Range续传。

## 报告与完整性

完整性依据本次计划下逐个任务的 `result.json`，不会用延迟的监控计数替代。已有报告/CSV逐文件冻结，允许后台监控继续改写它们。如果已有报告仍标 `PARTIAL`，清单会保留该原状态；全部任务输出依然完整上传，后续可据原始预测重新统计，上传器不会现场重算数小时。

上传器仅处理第2、3、7轮。第1、4轮训练和第5、6轮队列不被修改。

## 下载还原

```bash
python -I -S -B -u RESTORE-PAPER3-RESULTS.py --round 2 --destination "$HOME/paper3-restored"
```

将2替换为3或7。工具按索引下载分片、验证SHA256、流式解压到独立快照目录，再验证每个文件。支持 `--snapshot` 读取历史快照。它不会覆盖原训练目录。

## 验证

见 [VALIDATION.json](VALIDATION.json) 和测试源文件。已验证分片丢失响应、重复上传复用、SHA校验、并发Git更新保留其他轮次、上传失败不发布LATEST、源文件字节/mtime不变，以及真实GitHub软件测试资产上传。软件验证不等于已经访问学校服务器或已经上传真实实验数据。
