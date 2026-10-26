# R10 强基线实现审计

`paper3r10/baselines.py` 仅接受 acquired-support 拟合能力，查询输入只进入已拟合对象的 `predict_proba(query_x)`，查询标签不存在于拟合/预测签名。`baseline_registry.json` 可在不导入 Torch 的计划/验证流程中读取。

## 与历史算法的关系

- 冻结源模型直接调用原始 `source_logits`。
- Support-EA 明确调用 R7 原始 `fit_reference(..., 'support_EA')`。它不读取任务配置的 `variant`，避免 R4 已发现的错误映射到 `head_newton`。这里的 EA 是历史实验中每试验单位迹、.05 收缩后映射到训练源参考的具体实现；不泛称所有文献 EA。
- Head-SGD10 与 R7 原始头部参考逐元素一致；SGD100 保持相同优化器、物理系数先验及全头容量。bias/rank4 是明确标注的 R10 受控头容量对照：使用 R7 已有的 acquired-support 冻结特征 PCA 投影器与相同 SGD100，保留原始 feature_dim+1 先验归一化。它们不冒充 R7 的联合时空学习器。
- SGD10/100 使用 R3 原始非线性空间+头更新；R3 保护方法使用 `information_protected`；R4 使用原始 blocked OOF、one-SE 选择与 simplex-Brier stacking。默认强拟合配置采用历史主比较 rank4、radius=.15、head_l2=.1。可在冻结计划中显式传入经开发验证选定的配置，不允许目标 query 调参。
- R8 代表方法调用原始 fixed fiber/bounded transport，需训练前20采集位置的合法 class-reference。
- R9 代表方法调用原始普通 Bernoulli 似然 GP（不是把 balanced BCE 偷换进 GP），使用 training-only whitening 与 原meta_validation固定的 raw-RBF / EA-linear anchored-Laplace 配置（raw代表方法按最新完整补档更新，不用目标query选参）。EA+GP 首先调用真实 Support-EA。

原始源码位于 `paper3r10/vendor/native/`。`vendor/SOURCE_BASELINES.json` 固定每个源文件的 GitHub commit、blob SHA、SHA256 与本地复制 SHA256。唯一修改是已部署的 R9 CUDA deterministic cumsum 元数据修复：累加移至 CPU；GP 拟合与预测结构不变。源算法相对导入保持原名，旧实验文件没有被修改。

## 必需的实际学习产物

SGD、R3、R4 需要相同种子/source/data 身份的真实训练空间字典。R8 需要真实 meta_train class-reference。R9 需要真实 raw/EA 对应 reference/config（`mean,whitener,lengthscale,provenance,diagnostics`，配置由 meta_validation 固定）。数据层验证文件字节、SHA、种子、source 与输入身份；拟合层再次验证角色与配置。缺少任何产物立即明确报错，绝不以随机/零字典或其他弱方法冒充强基线。

允许复用历史预测仅限新 worker 验证完整目标、支持/query UID、数据、模型、种子与方法配置身份相同的情况。历史 R4 错误 EA 标签不可作为 EA 缓存复用。

## 已执行验证

本地 CPU、合成冻结网络、11通道CAR信号：14个测试全部通过（初次0.718秒；不含真实学校数据/GPU）。验证包括真实 EA 原生数值一致与标签不变性、完整头部原生一致性、低容量子空间/先验、R3 SGD10/100 及保护原生一致性、R4 全网格 OOF 原生一致性、R8 fiber 原生一致性、R9 raw/EA GP 原生一致性、查询分批不改变拟合、冻结源参数/BN状态、单类回退、缺失产物阻止及派发错配阻止。

运行：

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_baselines.py -v
```

这些验证证明软件对原始算法的重用与协议约束；不是新 R10 方法已经在 EEG 上提升的证据，也不是学校 GPU 已预检完成。
