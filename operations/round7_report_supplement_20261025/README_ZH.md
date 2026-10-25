# 第七轮总报告定向补传

第二、三、七轮的快速通道已经成功传回计划和完整 CSV。第七轮的 `reports/report.json` 超过旧工具的 64 MiB 单文件限额，被跳过；此工具仅补传这一份总报告。

它不读取逐试次任务目录、不导入 NumPy/PyTorch、不重算报告、不提交、取消或修改任何训练作业，也不重启正在慢速打包的全量归档。凭证复用学校已有 `private/github.json`，不会写入报告。

## 远端位置

- `paper3-results` 分支：`results/paper3/round-07/REPORT_JSON_LATEST.json`
- 可读入口：`results/paper3/round-07/REPORT_JSON_ZH.md`
- 快照：`results/paper3/round-07/report-json/<原始文件SHA256>/`
- 既有快速通道 `PAPER_DATA_LATEST.json` 和完整归档 `LATEST.json` 保持各自含义，不互相覆盖。

报告不设任意大小上限。读取时每次约 1 MiB，分片保持 UTF-8 边界，每片 SHA256 和偏移、原始整文件 SHA256 都保存在 `manifest.json`。各片是原始 JSON 文本的片段，按编号直接拼接、不要加换行，即可恢复原始字节。

GitHub 单个 blob 约 1 MiB，避免 100 MiB 单文件限制。总报告按文本分片进入仓库；模型、逐试次原始输出等大体积完整归档仍走原有 Release 通道。

## 断点与日志

启动命令安装纯标准库包后，在登录节点后台运行。关闭终端不会停止上传。网络错误在上传进程中重试，不影响训练。再执行同一启动命令会识别已有上传进程；中断后重启会复用稳定报告副本和已上传 Git blob，避免重发已成功的分片。

本地独立工作目录：`$HOME/paper3-r7-report-json-upload`。

- `upload.log`：逐片上传和重试日志。
- `status.json`：上传字节、分片及状态。
- `R7-REPORT-JSON-UPLOADED.json`：最终提交回执；只有所有分片与索引发布成功才生成。
- `source-cache`：稳定的原报告分片缓存；源文件复制期间变化会重新获取，不会发布混合版本。

## 后续 Codex 如何读取

先读 `REPORT_JSON_LATEST.json`，再取其 `manifest`，用 `structural_index.fields` 定位感兴趣的顶层字段和对应分片。顶层定位是轻量结构扫描，不宣称重新验证了 JSON 语法或科研结论。

需要完整报告时，用此目录的 `restore_report.py`，将 manifest URL 和发布回执里的 commit SHA 传入 `--manifest` 与 `--ref`；恢复文件时验证每片和整文件 SHA256。已有 CSV 用于常规论文表格分析，总报告保留嵌套科研信息。上传成功不等于完整科学输出归档完成。

## 验证范围

已测试真实超过 64 MiB 的 UTF-8 JSON 精确恢复、分片边界、断网后已传 blob 复用、源文件变化、缓存损坏、旧恢复目标保护。仅执行单文件扫描，未在学校服务器现场运行上传，也未宣称学校 GPU 训练成功。
