> 历史资料：以下是切换到学校HPC之前或早期恢复阶段的原文；其中RunPod入口、未训练状态、12小时预算和重复测试要求不作为当前操作指令。当前请看[实验导航](https://github.com/jackzhu119/research3)。

# Paper3 protocol implementation

本仓库在独立研究分支实现论文3协议及可恢复科研流程。原始协议 ZIP 的52份 manifest 文件已逐一 SHA256 核验；`protocol/legacy_baseline_v1` 保留为旧基线，不作为新算法。实验输入以已获授权的 v2.1 信号合同和冻结角色为准。

## 当前数据与软件验收

source4、Stieger、Yang、Dreyer、Ma 和 CAR3 数值缓存均已完成原生解析、缓存校验及原生产 R2 完整 GET/SHA256 验证。各数据的实际覆盖、有效 trial 数量、单位限制和用途见[真实缓存与训练分工](../plans/data_plan_20261019/VERIFIED_DATA_ZH.md)。provider ID 不代表已核实的跨库独立自然人总数；CAR3 来自相同 source4 trial，不增加独立 N。

最终 CPU 软件回归实际通过 **3641项测试**，耗时557.99秒；197项软件输入（196份 Python 文件及 `pyproject.toml`）在真实退出后全部复核 SHA/bytes。软件 acceptance、完整 stdout 和原117/189份历史清单分别保留。当前软件验收不表示真实模型已拟合。

Stieger 全62人598会话的独立质量审计实际通过，固定40/11/11人、396/97/105会话；全部训练开发和验证会话具备真实 B20/future-query资格，没有跨分组完全相同信号。内部105会话只核缓存 SHA，没有调用内部 episode 或进行内部科学检验。最终数据契约见 `contracts/data_plan_final_validation_20261020.json`；部署回读与固定包验收另以最终交付回执为准。

此阶段科研 fit、GPU训练、内部科学检验及科研性能结果均为0。用户本人开启已有云服务器后，按固定包的[启动说明](../plans/data_plan_20261019/STARTUP_ZH.md)恢复缓存与离线依赖，先运行唯一受控 GPU 恢复 pilot。真实检查点、预测/数值恢复、预算和阶段门通过后，才继续主线及两条独立 source 增强支路。脚本不启动或重启 RunPod，不创建实例。

## 科研边界

主线使用固定四source、11通道合同及 Stieger 40/11/11。Yang 和 Dreyer 各自与 source4 组成一条独立五source支路，不合并为默认六库。Dreyer 在论文1已有暴露，不能作新盲法外测。比较方法共享各支路真实 f0/Rref、seed、校准/未来query与选择预算；不能把来源变化单独解释成人数增加的因果效应。

Ma 分组未冻结，不进入 fit；原删除 trial 的采集位置未知，不能还原原始付费 B20。完整三通道 Rref/G/方法比较有独立门禁。HS、Wang、Leeuwis、MILimbEEG 等封存外测遵循 W7 权限，当前未下载或解封其 query。未完成的科学参数保持 null，软件示例不作为估计或训练结果。

受控 source pilot 比较100次连续更新与50次更新后保存/重载再更新50次，总计200次；之后只对固定source训练UID重放预测及指标。任何失败保留原 claim、预算、检查点及结果，不能换目录重复 fit。GPU驱动兼容性及数值恢复等待用户开机后实际检验。

## 交付与启动

正式固定包必须同时具有冻结 index、cache-only manifest、archive PUT/full-GET回执、archive独立完整回读与成员SHA证明、全部 index 对象独立完整 GET/SHA256证明、fresh目录只读启动验收及 late-control 完整 GET回执。历史 preparation checkpoint 是可恢复快照，不是训练就绪包。

用户将最终固定包校验SHA并解压至新的论文3目录，保留原有R2环境变量，然后执行：

```bash
python3 tools/bootstrap_data_plan.py --index audit/data_plan_20261020/final_deployment/bootstrap_index.json
```

默认只读；用户开机并决定开始受控pilot及后续阶段后：

```bash
python3 tools/bootstrap_data_plan.py --index audit/data_plan_20261020/final_deployment/bootstrap_index.json --execute --launch-training --user-started-gpu
```

固定包携带原协议 ZIP、新数据方案 DOCX、代码/元数据、CPython3.12.15原归档及最小SDK wheel；完整固定依赖与数值缓存从已有R2按 exact paths恢复。100个固定依赖已完成独立CPU离线安装、版本/pip check及CPU导入验收；这不证明GPU兼容。启动器不自动训练未冻结的Ma/CAR3或进入W7外测。

## 仓库与隐私

GitHub仓库为PUBLIC。公开分支保留代码、授权导入协议和汇总验收；新增EEG、逐trial清单、原生catalog、R2回执及批处理状态仅保存在本地与用户已有R2。固定包不包含完整原始EEG、旧NPZ、论文1/2配置、旧权重或凭据。

- `protocol/`：原始研究规格和52文件校验来源。
- `p3work/`：原生适配、方法、可恢复训练、真实episode及预测存档。
- `contracts/input_contract.json`：冻结的v2.1信号合同。
- `status/tasks.json`：已完成准备门与待完成科学阶段。
- `plans/data_plan_20261019/`：数据分工、限制和固定启动说明。
- `p3work/methods/STATUS.md`：方法软件实现范围，W5科学验证仍待实际执行。

R2工具默认dry-run，只有显式 `--execute` 才使用已有配置，按明确manifest执行条件PUT和完整GET/SHA256；不扫描或复制密钥。已完成的数据协调器不应重启。运行结果只有实际上传并回读通过后才报告已备份。
