# 交接合同与轻量检查器

`WORK_CONTRACT.json` 是待实施合同模板；当前 `status` 为 `planning_only`。具体参与者 ID、选定窗口、最终超参数、B*、实际 EEG schema 证明和模型产物尚不存在，保留为 `null` 或空列表。`work_contract.schema.json` 只描述主要 JSON 结构；`tools/check_work_contract.py` 额外核查部分研究角色、预算、身份和哈希规则，两者都不能替代真实数据和算法审查。

本轮实际可运行的命令，从交接包根目录执行：

```bash
python tools/check_work_contract.py
python tools/check_work_contract.py --report tools/work_contract_check.json
python tools/check_work_contract.py --require-frozen
```

第一个命令只读 JSON，不下载数据、不载入 EEG 数组、不导入模型、不训练，也不写文件。第二个命令另外写出 JSON 检查报告。第三个命令对当前模板应返回退出码 1，因为计划尚未冻结；这不是工具故障。默认检查通过只表示已选择的元数据规则没有冲突，不表示数据合格、模型正确、研究结果有效或外测已经获准启动。

未来 Work 实现完整流水线并完成对应审计后，可更新合同与 registry。不得为了使检查器通过而填造事实。准备冻结时至少需要：

- 在候选窗口筛选之前固定的 Stieger 原始合格 roster、40/11/11 split 及证据。`eligible_subject_ids` 指基于原始身份、通道和任务资格确定的名单，不是按最终窗口是否足长重新挑选的人；窗口 QC 不替换参与者。若人数不足 62，按协议留下 `reduced_roster_amendment`。
- 仅依据上述 meta-train 组时长元数据得到的 `window_audit_receipt`。它包含 `decision_pool_role: "meta_train"`，以及 `long`、`short` 各自的 `participant_mean_coverage` 和 `participant_mean_abs_class_exclusion_gap`。后二者是参与者等权平均的保留率，以及每人左右类排除率差绝对值的等权平均，取值为 0–1。最终选择按协议先长后短应用 0.80 / 0.10 门槛。原始逐人测量、分母和排除表由真正的审计文件保存；检查器只检查这些摘要数值。
- 最终 r/ρ、B*、其仅由 meta-validation 产生的选择记录，以及主要临床 query 的独立 acquisition-order 证据。
- 真实 EEG schema 审计及其产物。登记页的受试者总数、论文报告的 trial 数或本轮小元数据校验不能代替实际 `n_eligible_*_verified`。
- 完整源模型、Rref、G、对照、split、信号合同、分析计划和协议文件的相对路径与 SHA-256。`freeze.artifacts` 中每项包含 `path`、`sha256`、`role`；需要的角色为 `protocol`、`methods`、`data_registry`、`split`、`signal_contract`、`reference`、`f0_bundle`、`generator_bundle`、`comparators`、`analysis`。文件列表是显式清单，检查器不递归猜测哪些文件应该包含在内。

`freeze.protocol_freeze_sha256` 的计算约定为：复制整个合同，将这一个字段设为 `null`，以 UTF-8、键排序、无多余空格、禁止 NaN 的 canonical JSON 序列化，再求 SHA-256。`check_work_contract.py` 中 `contract_digest()` 实现这个约定。任何冻结后的事实或参数变更都需要新的审计和冻结版本，不能保留旧哈希。

只有在相应文件已经真实存在时，以下未来命令才可能通过：

```bash
python tools/check_work_contract.py --require-frozen --verify-artifacts --artifact-root .
```

`--verify-artifacts` 只读取显式清单中的文件字节并核 SHA-256；拒绝绝对路径、向上越界和解析后越出根目录的符号链接。它不解释 EEG、checkpoint 或 Markdown 内容，不能识别虚假元数据、伪造的独立身份或隐蔽的数据泄漏。必须结合 W0–W9 的真实审计与验收；普通哈希也不能证明未经授权的人从未接触过目标标签。

`p3work` 的 reader、plan、cache、meta、adapt 和 evaluate 等入口仍是 `WORK_TASKS.md` 中的待实现接口；本交付不存在能够运行这些新科研任务的命令。旧 `legacy_baseline_v1` 的 CLI 也不能代替新方法实现。
