# 第三轮：有限前缀证据下可辨识的 Lie 空间校准

归档日期：2026-10-24。本文按已发布的代码、实验矩阵和GitHub回传记录分类；设计和软件验证不等于识别收益已经得到证明。

返回[五轮总目录](README_ZH.md)；相邻记录：[第二轮](ROUND2_ZH.md)、[第四轮](ROUND4_ZH.md)。

## 本轮要回答什么

论文主线保持为：冻结源EEGNet，只用目标会话最前20个已获取试次校准少量空间参数，再预测同一会话的未来查询。
第三轮问的是：20个标签能否区分空间变化与残差分类头变化；证据不足时，怎样收缩没有充分支持的空间移动？

- 解析切线和更充分的求解，是否优于旧10步SGD，而非仅仅增加了优化预算？
- 分类头已经能解释的支持预测变化被扣除后，剩余信息是否足以决定空间半径？
- 对真实支持预测加入源KL保护，是否减少负迁移，又会不会妨碍必要纠错？
- 与新学习器匹配的元训练字典，是否在相同训练先验下提供额外收益？

完整科学设计：[DESIGN_ZH.md](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/experiments/followup_20261024_r3/DESIGN_ZH.md)。

## 数据和实验合同

- 复用已有源EEGNet、匹配种子的源参考协方差和缓存，不重新训练整套源网络。
- 输入保持clinic11通道、11通道CAR、128Hz、8–30Hz及既有时间窗；不更换预处理来解释算法收益。
- Stieger2021沿用冻结参与者清单，预期40人元训练、11人元验证、11人内部留出，以manifest实际身份为准。
- 种子为20261018、20261019、20261020；种子不是额外独立受试者。
- QC失败仍消耗前20个采集位置，不从第21条以后补足有效样本。
- B4/8/12/16/20都截取同一B20采集前缀，并评价同一个B20之后的未来查询集。
- 未来查询的输入和标签均不参与目标会话的校准、空间半径或保护比例选择。

## 数学算法：已实现

1. **解析身份切空间。** 将CAR后的10维空间变化表示为99维无迹Lie生成元，在低秩字典中拟合空间参数；解析导数包含白化、RMS和冻结网络的输入导数。
2. **局部凸代理。** 在固定身份切线中建立带空间与残差头先验的平衡逻辑目标，使用Newton/GGN小系统求解。
3. **Schur头消元。** 以 `I = Gθθ − Gθa Gaa⁻¹ Gaθ` 描述头竞争后的空间曲率；空间球约束激活后，重新求给定空间端点的条件头。
4. **信息半径。** 令 `d_eff = tr{I(I+τI)⁻¹}`，使用 `ρS = ρ sqrt(d_eff/r)`；主信息尺度为1，固定先验、半径和求解预算。
5. **真实预测保护。** 沿身份点到完整候选更新回溯，在真实指数变换后的支持预测上要求源到新预测的平均Bernoulli KL不超过ε；主ε=.01。
6. **匹配字典元学习。** 在实际非线性未来查询损失上训练低秩字典，匹配新支持学习器，并保留相同头先验的SGD字典对照。
7. **更贵的参考。** `relinearized`反复重算真实网络Jacobian/GGN，仅作为固定字典评价参考，不称为已支持的高阶外层训练器。

信息半径是依赖先验的局部证据规则；小Fisher也可能来自自信但错误的源模型，不能解释为“不需要适应”。
KL只约束已观察支持分布上的平均预测变化，不保证未来准确率不下降或每个样本都不发生负翻转。

算法实现：[paper3r3/algorithm.py](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/experiments/followup_20261024_r3/paper3r3/algorithm.py)。
推导与文献：[math_review.md](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/experiments/followup_20261024_r3/math_review.md)、[LITERATURE_AND_DATA_REVIEW.md](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/experiments/followup_20261024_r3/LITERATURE_AND_DATA_REVIEW.md)。

## 精确实验规模

| 分类 | 组成 | 逻辑任务数 |
|---|---|---:|
| 旧字典训练或匹配复用 | legacy × r2/r4/r8 × 3种子 | 9 |
| 匹配先验字典 | matched_sgd × r4 × 3种子 | 3 |
| 新字典 | profiled / information / protected × 3种子 | 9 |
| 旧字典核心评价 | 11适应器及独立original_sgd10 × 3种子 | 36 |
| 新字典交叉评价 | 3字典 × 4适应器 × 3种子 | 36 |
| 匹配SGD字典评价 | 2适应器 × 3种子 | 6 |
| 敏感性评价 | 秩/半径24、头先验18、迭代18、KL6、信息尺度6 | 72 |
| 支持EA参考 | 3源模型种子 | 3 |
| **核心合计** | **21字典＋78核心评价＋72敏感性＋3参考** | **174** |
| 可选Weibo外部评价 | 6固定方法组 × 3种子 | 另18 |

174是逻辑任务数，不是174次完整元训练；最多21项字典fit，完整匹配的旧字典可以只读复用。
各字典最多1000次outer更新，每次4个episode；每50步元验证，每10步保存断点。
核心78项覆盖五种预算；敏感性主要评价B20，不在内部留出上挑选最优超参数。
六种B20压力为支持错标10%/20%、查询CAR对称增益.10/.25、查询反对称旋转.10/.25。
压力嵌于6方法组×3种子，共108个条件；空间压力只改变校准后的查询，支持输入保持不变。

机器可读依据：[EXPERIMENT_MATRIX.json](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/experiments/followup_20261024_r3/EXPERIMENT_MATRIX.json)。

## 已实现与后续提议的边界

本轮实现了解析切线、Schur约束求解、信息收缩、支持KL保护、匹配字典训练及预先固定的评价矩阵。
Weibo18项是可选独立评价阶段，不能因为缓存已准备就记为已经跑完。
真实在线采集顺序的独立队列、临床前瞻确认和新的人口风险保证均不属于已完成工作。
第四轮的OOF堆叠、第五轮的旋转—增益耦合也不应倒写成第三轮已实现的贡献。

## 软件与部署验证

- 精确版本CPU环境：Python3.12.3、PyTorch2.5.1+cu121、NumPy2.2.6；131项软件测试通过，0失败。
- 覆盖真实CompactEEGNet解析切线、约束边界梯度、完整信息半径元梯度、Schur一致性与换基等变。
- 覆盖合成元训练→隔离worker评价→报告、损坏新断点回退、QoS等待、丢失提交回执对账及独立任务失败。
- 本地验证主机没有GPU；测试不是学校A100性能或EEG准确率提升证据。

依据：[VALIDATION.json](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/experiments/followup_20261024_r3/VALIDATION.json)。该文件的“未提交”字段是发布时验证范围，实时作业状态另见回传快照。

## 已观察状态：带时间的快照

GitHub回传 `updated_utc = 2026-10-24T10:55:27Z`，即北京时间18:55:27：

- **6/174逻辑任务COMPLETE**；其中3个legacy-r4字典、3个seed20261018旧字典适应器评价完成。
- 控制器状态RUNNING；另2项ACTIVE、112项NEW、54项WAITING_INPUT。
- 报告PARTIAL，主要比较族未齐，`scientific_completion_claimed=false`。
- ACTIVE是控制器任务状态；判断某个作业此刻是否在计算，还需结合Slurm及实际步数/会话数增长。

状态证据：[monitoring/paper3-round3/status.json](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/monitoring/paper3-round3/status.json)。链接后续会更新；本文数字只对应上述时间，不冒充实时查询。

## 学校资源、续跑与结果位置

学校分区为 **hpc_gpu**，最多3个独立单GPU作业，每作业12CPU、64G内存；按既有40GB GPU配置设计，不要求同节点，不是DDP。
`hpc_gpu`分区与节点名gpu1/gpu2及`hpc_gpu1`分区应分别记录，不能混淆。
源码安装：`~/paper3-round3/<包hash>/`；结果：`/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r3`。
外部结果独立后缀为 `_r3_weibo`；不覆盖前两轮目录或取消旧作业。
字典每10步保存含随机状态的双代checkpoint；评价逐预算、条件、参与者和会话原子保存，只补缺失项。
没有累计12小时限制；单次Slurm默认5天3小时，提前300秒请求保存，退出75后控制器续交。
学校启动不重跑全量pytest、pilot或21,262文件SHA扫描，也不在作业中重装依赖。
独立任务失败会保留错误，其他可运行任务继续；明确用户取消或STOP不会无限自动复活。
GitHub每180秒尝试同步到 `paper3-r3-progress:monitoring/paper3-round3/status.json`，外部用独立 `weibo-status.json`。
网络回传失败不停止训练；全量checkpoint、逐会话预测、`reports/report.json`和CSV先保存在学校本地。

操作依据：[RUNBOOK_ZH.md](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/experiments/followup_20261024_r3/RUNBOOK_ZH.md)。

## 可以和不可以下的结论

主终点为B20、clean、内部留出参与者等权BA；10项预先固定的主要比较在全部所需三种子及会话网格齐备后统一Holm。
会话、试次、fold和种子均不是额外独立参与者；报告BA百分点、参与者配对区间与纠错/负翻转，而不只挑平均准确率。
内部队列此前已进入研究流程，本轮属于探索性内部复验，不能重新称为首次盲测。
Weibo是发布顺序LR子任务稳健性，真实物理采集前缀/run未知，不等同严格在线前瞻确认。
目前快照不足以宣布信息半径、KL保护或新字典显著优于旧方法；完整负结果和回退也应保留。

统计依据：[STATISTICS.md](https://github.com/jackzhu119/research3/blob/paper3-r3-progress/experiments/followup_20261024_r3/STATISTICS.md)。

