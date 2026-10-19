# PhysioNet 原始注释与传输修订

本次修订处理实际源文件格式和传输，不增加方法、窗口候选或模型拟合。原始 52 份协议、505 个 canonical 对象、冻结的 v2.1 输入及 `source_native.py` 保持原字节。旧审计继续使用它实际执行的读取器。

实际 S100R08（官方 SHA256 为 `8371eec06d0b1ff62d10e016124e3bc2db411ab6764fea572fb2d1f25a6afd97`）为 128 Hz。12 个 LR 注释中有 11 个起点不在整数采样点；原 TAL 的最后任务结束于 123.025 秒，实际记录结束于 123 秒，MNE 已将该注释时长裁短。直接使用 MNE 的裁后值会丢失这项来源信息。

版本 `physionet_original_tal_inward_v1` 使用原 EDF+ TAL 十进制 token 的精确有理数。对原 onset `t`、duration `d`、下一注释起点 `u` 和记录终点 `T`，边界固定为：

```text
start = ceil(t × fs)
stop  = floor(min(t + d, u, T) × fs)
context ⊆ [start, stop)
```

保留原 token、MNE 注释值、记录/下一事件裁剪、向内量化损失及原 annotation index。UID 仍由原始身份字段计算；不补点、不前读任务、不把时长不足事件重新编号。真实数值缓存必须用同一 revision 重新解析并逐行核对元数据。新增 reader、dispatcher 及其共享组件的实际文件 SHA 均进入缓存和 pilot 的来源绑定；未知或更改的 revision 会拒绝执行。

官方数据页明确提供 `s3://physionet-open/eegmmidb/1.0.0/`。可选的匿名 HTTPS 传输只映射原 327 个源 LR 对象，并用所保留的官方 `SHA256SUMS` 核整个文件。HEAD、ETag 或 Range 尾段均不作为整文件 SHA 证明；R2 凭据不会用于公开源站。

旧失败尝试的 state、HEAD、partial 和计费记录先保全。显式重试产生含 revision 的新审计路径及真实三对象 R2 回读证明，再更新 canonical state 索引。原累计流量和墙钟预算继续有效。封存外测与 GPU 执行权限不受此修订影响。
