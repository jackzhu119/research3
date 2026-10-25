# Paper3 第四、五、六轮 hpc_gpu 资源迁移

每轮独立申请一个节点、一个作业、三张 A100-PCIe-40GB，总36CPU、128G共享内存。

| 轮次 | 原计划 | 迁移源码和说明 | GitHub自动进度 |
|---|---:|---|---|
| 第四轮 | 204 | [第四轮](operations/hpc_gpu_migrations_20261024/round4/MIGRATION_LAUNCH_ZH.md) | paper3-r4-progress:monitoring/paper3-round4/status.json |
| 第五轮 | 105 | [第五轮](operations/hpc_gpu_migrations_20261024/round5/MIGRATION_LAUNCH_ZH.md) | paper3-r5-progress:monitoring/paper3-round5/status.json |
| 第六轮 | 198 | [第六轮](operations/hpc_gpu_migrations_20261024/round6/MIGRATION_LAUNCH_ZH.md) | paper3-r6-progress:monitoring/paper3-round6/status.json |

先完整下载并校验相应固定版本包，用户执行对应轮次--submit后才处理其旧排队或checkpoint交接。原计划、已完成结果、模型、逐会话数据和断点保留。当前发布不代表学校作业已取消或提交。固定三套启动指令和实际下载证据在本分支operations/hpc_gpu_migrations_20261024/online/发布。
