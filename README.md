# Paper3：协议导入前置检查

**阶段门：BLOCKED。尚未取得配套 ZIP，未读取协议，W0 未通过。**

本分支 `research/paper3-protocol-v2-w0` 只记录原包可用之前已实际执行的前置检查，不是新算法实现、训练包或实验结果。W0–W9 的任务、方法公式、数据清单及阶段门必须从完整 v2.0 协议导入后确定。

## 已核实

- 工作区为 `/workspace/research3`；`origin` 为 `jackzhu119/research3`。首次检查无提交，远端无 refs。
- 当前环境网络策略为 enforced，状态 running/connected，观测为 current。
- 既有 `R2_ENDPOINT`、`R2_BUCKET`、`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY` 均存在且 ready。一次 `HeadBucket` 返回 HTTP 200，见 `audit/r2_connectivity.json`。
- 未打印凭据值，未复制密钥，未新增凭据。未列举、读取或上传 R2 对象；HeadBucket 成功不代表数据已存在或对象读写已获验证。

## 当前缺口

Library ID `libfile_5f1cbeccc8ec81918b0f27ec5dcdd7ed` 只有用户提供的元数据。本会话无可调用的 Library/File Service 下载工具；Sediment `download_file` 不支持该类型 ID。`/workspace/library-files` 与共享下载目录均为空，也无可调用的浏览器工具。

需要该 ZIP 的实际可读本地文件或受支持下载入口，无需提供凭据。用户给定的大小、SHA256、52 份 manifest 文件计数保存在 `provenance/handoff.expected.json`，均尚未独立核验；包内 DOCX 与附件一致也仅为用户声明。

## 固定前置检查包

仅在已安装 Python 和预先提供的既有环境变量下执行：

```bash
python -m pip install -r requirements-preflight.txt
python tools/check_r2.py --output audit/r2_connectivity_deployment.json
python tools/verify_handoff_archive.py /actual/path/Paper3_Work_Handoff_20261018.zip --output audit/archive_integrity.json
```

归档校验器仅验证指定 ZIP 的大小、整体 SHA256 和 ZIP 目录安全性，不打开成员载荷、不解压、不执行包内内容、不声称完成 52 份文件 SHA 核验或协议阅读。缺少原包时返回失败。

原包可用后的顺序：整体 ZIP 核验 → 安全导入 → 逐一核验 manifest 的 52 份文件及 SHA → 读取完整协议与配套代码 → 建立 W0–W9 阶段门记录 → 按协议执行真实适配和最小 pilot。若缺原数据或代码，记录缺口；不得用合成或旧缓存结果替代。

## 执行边界

- `legacy_baseline_v1` 只可按协议作为旧基线，不能当作新算法。
- 不提前打开封存外测，不跳过阶段门。
- RunPod `k3az08k9ji2d39` 由 dot 唯一浏览器操作者协调；本包不连接 RunPod，也不启动远端训练。
- 不复用已清理的论文1原始/处理缓存或旧 NPZ；论文1/2 原代码与上传配置保持不动。
- 不发邮件、不投稿、不新建付费实例。

尚未开始真实适配、pilot 或训练；尚未上传和回读。部署此前置检查包不能改变这些状态。
