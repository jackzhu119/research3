# R10 学校元数据自动发现

`paper3r10.registry` 仅用 Python 标准库，不导入 Torch/Numpy，不占 GPU，不操作 Slurm。它读取固定历史运行位置的 plan/launch、小结果回执以及已选小模型文件的 SHA，不扫描预测目录或 EEG 数组。

接口：

```python
from paper3r10.registry import build_registry, write_registry, validate_registry
registry = build_registry('/public/home/zhuziyuan/projects/paper3')
descriptor = write_registry(registry)
check = validate_registry(registry, stage='mechanism', seeds=[20261018])
```

`write_registry(root, ...)` 也可直接先构建再写入。默认唯一写入目录为 `root/runs/round10_preparation/`；指定旧轮次目录会报错。安装/准备/验证不会提交新作业。

自动解析：

- 3个种子的冻结 F0 source receipt 和训练 reference receipt；
- W3 `status/tasks.json` 声明的 Stieger 缓存manifest、参与者划分文件及三个精确 catalog 身份；不打开 manifest 声明的 X；
- R3/R4 主比较任务声明的 legacy 字典依赖，选择规则来自 task.basis_key，绝不按 holdout 分数挑字典；
- R8 完成的 meta_train-first20 class-reference；
- R9 完成的 raw/EA feature-reference，以及原 meta_validation 选择的 raw-RBF / EA-linear anchored-Laplace 配置；
- R9 plan 显式声明的 Pan/Weibo manifest，允许其在已部署的 R9 包中以绝对路径存在。

计划查找限制为每个固定旧运行目录中的 `plan.json`、`runtime/plan.json`、`runtime/three_jobs/{plan,backend-plan}.json` 及相应 `launch.json`。仅继续跟随这些 launch 显式声明的 plan 引用；不全盘 glob。结果文件须 status=COMPLETE、task/recipe/选择排除 holdout 一致、selected checkpoint SHA 一致。R3/R4 回执未单列 recipe_sha256；这里复算它们原始 identity(task,data,source,recipe,basis,training)，不是放过 recipe。已用上传的21个真实 R4 basis 回执验证该重算完全一致。

`baseline_artifacts["20261018"][method]` 为数据加载器约定的扁平 reference 键。每个 `.pt` descriptor 带 `field` 路径，只在后续科学 worker 里使用 `torch.load(weights_only=True)` 提取原始 b/class_reference/reference。注册表构建时不反序列化模型。provenance 保留精确 source/data/task 身份及 result/plan 文件 descriptor。开发标签与目标标签明确区分：B 的元训练及 config 的元验证可合法读取开发 query；target_query_labels_used/target_future_queries_read 均为 false。R8/R9 prefix-only reference 进一步要求从未读开发未来 query。

缺少合法历史产物会列入 `missing_artifacts`。机制阶段检查自己的source/cache；若三种子的raw与EA参考齐备则复用，否则按固定meta_train前缀生成参考，缺少可复用参考本身不会阻断这一生成路径；正式阶段进一步按所请求 baseline 方法检查产物。验证会在 GPU 分配之前重新核对选定小文件的字节和 SHA，给出具体缺失或变化路径。不会因此重启旧作业，也不会用随机/零字典或其他弱方法代替缺失产物。

显式 override 可作为补充；根目录/schema 必须一致，提供的 descriptor 都会核对实际字节/SHA。19个标准库测试覆盖合法发现、原始 R3 identity、外部绝对manifest、launch引用、相对checkpoint路径、错误/变化SHA、开发/目标标签语义、禁止递归扫描及仅写新目录。学校真实目录仍须由用户安装后运行准备/验证命令核实，合成目录测试不冒充学校执行结果。

重复准备得到相同科学输入时，保留已有注册表字节与创建时间，防止单纯观察时间变化导致 plan 输入 SHA 改变和重跑。
