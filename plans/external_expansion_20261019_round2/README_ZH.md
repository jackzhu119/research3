# 论文3外部验证扩展：第二轮资格准备

本轮将登记身份从 44 条扩展到 62 条。主 11 通道次级锁定候选由 8 个增至 9 个，新增 BBCI_IV1 的 b、g 两名真实左右手 MI 参与者；原研究其余真人 a、f 是左手/脚，c、d、e 为人工数据，均不计入该 LR 候选。9 个候选的报告人数合计 146 只用于检索规模说明，尚未经本协议资格、波形 QC 和跨库生物身份去重，不能作为独立有效 N。所有新增外测实际合格人数、trial 和 payload 字节数仍为 null。 BBCI 只有两名 LR 候选，结果主要补充描述性跨设备复现，不能单独承载稳定的人群推断；不能用 trial、session 或 pipeline seed 扩大参与者 N。

原协议 52 文件、获准 v2.1 的 11 通道输入、HS-Stroke v6 的 P 对 B* / B20 主检验、source/development 分配和原固定 pilot 包保持原样。`legacy_baseline_v1` 只保留历史基线身份。新增外测 EEG、query、逐 trial 标签向量、预测均未读取；没有科研 fit 或 RunPod/GPU 操作。

## 本轮加入的实质内容

| 条目 | 已核实的贡献 | 资格限制 |
| --- | --- | --- |
| BBCI_IV1 | 官方原生 1000 Hz metadata：b/g 为 LR MI，全部 clinic11 电极具名；进入第 9 个主输入次级候选 | 仅用严格 Range 获取元信息，无连续信号或 trial 标签；单位、原始时序、B20/QC、先前接触和身份仍待门后核验 |
| Chang2025 | 原作者 Fig.9 证明完整 11 电极，替换仅依靠“59 通道”的名义依据 | 不增加新人数；原档 header/顺序、4 个记录的真实边界尚未核验 |
| Leeuwis2021 | 55 人原仓库/论文、16 个具名电极、校准与反馈 run 的真实结构；列为独立 CAR3 优先资格候选 | 缺 8 个 clinic11 电极；原单位、原 CSV 列顺序及 native clock 待核，不能并入主11模型 |
| Kodera2023 / MILimbEEG | 保留 29 / 60 人研究作为三通道时序条件储备 | 类别分文件或多 record 的全局采集顺序未证明；不把名义人数或理论 B20 可行写成合格人数 |
| 9 个临床检索条目 | 补充到 2026 年的卒中、SCI 与其它康复研究检索，并保留排除依据 | 本轮没有新的临床队列通过 clinic11 + 左右手 MI + 访问/时钟门，不能将同手动作、ME、NIRS 或 ECoG 混入 EEG LRMI |

新旧候选、重复版本、访问阻塞及排除理由全部保留在 `registry.json` 和来源审计内。Grosse 原生 chanlocs 仍是数字标签，不能依坐标猜电极；Shin2022 的少数具名传感器不等于完整通道映射；IMUMIA 公开 5 人不等于已获得全库 244 人。UET175_2025 的身份保留并补上协议别名 UET175，分类改为访问阻塞。

## 独立三通道数据准备

已在 Codex CPU 上从本论文四 source 的已核验 pre-RMS 缓存，实际派生 C3/Cz/C4：224 个 provider ID、28,830 条 trial。它与主 source 是同一批人、同一批 trial，新增独立参与者数为 0；不复用论文1旧 NPZ。904 份输入在执行前后 SHA 相同，1,344 份派生文件独立重读，标签、UID 和重算输出通过。40 项针对性数学、路径、哈希和数据门禁测试通过。

独立合同为 `contracts/sensitivity_car3_v1.json`；缓存 manifest SHA256：`001d1e9c3810140c8b71b367c223bf61204c83e569971eb88de1cdffac7d9492`。合同采用共同参考单极 CAR3；BCIIV2b 的双极测量不自动准入。此派生继承原 11 通道 QC，不声称完成新的原生 3 通道幅度资格审计；数学等价仅在既存 float32 输入与新输出的舍入精度内成立。

**三通道分支的 `training_ready:false`，GPU 训练入口尚未实现。** d=2 下 full/symmetric/skew 维数为 3/2/1，旧 r=4/8 配置不可搬用；未来必须独立完成 source3 的 f0、Rref、G、强对照、秩配置和开发选择。详见 `three_channel_sensitivity.json` 和 `THREE_CHANNEL_ZH.md`。该分支不阻塞原 READY 包的主11受控工程 pilot，也不能被传给其训练入口。

## 冻结与执行顺序

1. 用户本人手动开机后，继续使用原 `paper3-controlled-pilot-ready-b5fb2b424951eb16.tar.gz` 和其 READY 启动说明。先离线依赖恢复、source 缓存 GET/SHA、doctor 和门检；只运行唯一一次受控工程 pilot，随后备份真实结果。GPU/驱动兼容性仍需开机实测。
2. 按原 W3–W6 完成开发、强对照和模型选择。新增外测不参与 f0、元训练、参考统计或超参数选择。
3. W7 先封存算法与所有选择规则，再受控执行预先规定的 native 资格审计。任何外测预测前，再封存实际队列和比较族。保留未通过、未运行队列及原因；对完整冻结族按方案作 Holm 校正，不依据结果筛选队列。
4. 三通道研究单独满足其开发、模型和权限门后才可训练/检验；本次仅完成其 source 数值准备和 prospective 方案。

## 补充包与备份

本版补充包与原 READY 包分别解压，不覆盖原清单或旧回执。补充包携带候选登记、公开元证据、CPU 转换实现及审计、三通道缓存的显式 R2 恢复回执；三通道数组独立存储，不重复塞入软件包。以最终外层 HANDOFF/BACKUP_INDEX 中的实际状态为准，计划 manifest 本身不代表上传成功。

补充包解压后进入其根目录，执行：

```bash
sha256sum -c ADDON_MANIFEST.sha256
python -B tools/check_external_registry.py --root . --registry plans/external_expansion_20261019_round2/registry.json
```

正常元数据门检应返回 0，并报告 `evaluation_ready:false`；增加 `--require-evaluation-ready` 应返回 1。它不会下载或解封任何数据。三通道缓存恢复只能使用随包的已验证 `R2_receipt.json` 配合原 READY 包的 `tools/r2_restore.py`，恢复后核 manifest 及逐文件 SHA；禁止把 source 派生数量当成新增验证人数。
