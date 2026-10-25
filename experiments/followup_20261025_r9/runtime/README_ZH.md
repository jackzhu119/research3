# 第九轮运行与双通道上传

本轮独立输出 `runs/followup_20261025_r9`，不迁移或取消其它轮次。沿用学校 `hpc_gpu` 分区，三个独立作业，每作业一个节点、一张分配到的 A100 40GB GPU、16核CPU、64G主机内存，总3卡/48核/192G。资源组标注PCIe并不用于拒绝真实的SXM4-40GB名称；科学worker记录实际CUDA设备。各作业只使用自己的 `cuda:0`，不把逻辑slot编号映射为物理卡号。

顶层启动器生成本轮plan后调用 `runtime/START.py --plan PLAN --package PACKAGE --python EXISTING_PYTHON --submit`。控制器只维护本轮 `paper3r9p-*` 三个长驻单卡作业，任务完成即领取下一项。重复启动采用已存在的确切作业与flock，不重复训练；配额拒绝退避重试，配额排队不当作训练失败。学校每作业5天3小时的时限前5分钟保存断点，控制器续接同一任务，不设置累计12小时上限。

节点不下载数据、不安装依赖、不运行pytest/pilot或完整数据树验证。现成Python3.12/Torch2.5.1+cu121/NumPy2.2.6环境与随包小型外部数据继续使用；只读取当前任务需要的输入。停止信号让科学worker保存，COMPLETE与checkpoint以科学任务身份复用，日志从第一行输出ROUND_9_PARALLEL_PROCESS_STARTED。三个作业是否同时获配由学校配额决定。

`START.py` 启动四个相互独立的后台进程：controller、monitor、paper、archive。上传缺凭据、GitHub失败、网络超时、报告生成错误均不停止或重训任务。报告构建在独立子进程，完成任务数变化后最多每10分钟生成一次，最后一项完成及时生成最终报告。

- 实时进度：`paper3-r9-progress:monitoring/paper3-round9/three-jobs-status.json`，约180秒一次。
- 通道A：`paper3-results:results/paper3/round-09/PAPER_DATA_LATEST.json`。浅读本轮plan、报告与报告同时生成的完整 `session_metrics.csv`、`participant_metrics.csv`，保留参与者/会话配对身份，CSV分片完整行、重复表头，所有源文件另存精确字节分片与SHA256；没有64MiB文件跳过限制、不扫描逐试次树。不重算统计。报告尚未完成时明确部分范围。
- 通道B：`paper3-results:results/paper3/round-09/ARCHIVE_PROGRESS.json`。每个COMPLETE任务在其原生锁释放、尾写完成后，按回执身份只遍历一次并增量上传，包含任务内预测、验证轨迹、模型/优化器/RNG断点。每个完成任务聚合成一个确定性tar.gz，再拆32MiB Release块；每个原文件均保留tar路径、字节数、SHA256，断网复用已打包和已验证块，不为每个小文件创建asset。本轮所有任务与已归档回执身份吻合，且最终报告的recipe/完成数/总数吻合本计划后才发布 `LATEST.json`。不包含原始EEG、共享源模型、venv、运行日志和凭据；本地文件始终保留。

分开日志为 `runtime/three_jobs/controller.log`、`monitor.log`、`paper.log`、`archive.log`；上传状态/缓存为 `upload-paper/` 与 `upload-archive/`。GitHub完成显示与科学完成、完整备份完成分别记账。训练过程保存的当前任务断点留在学校；通道B默认在任务完成后归档该任务，不把尚未完成的任务包装成完整备份。

精确文件恢复：按文件manifest中parts的byte_offset排序；通道A每块base64解码后拼接，通道B按groupmanifest中index顺序下载并校验Release分块，拼接成tar.gz，解包后逐文件校验bytes和SHA256。CSV行视图只用于直接分析，原始文件格式以精确字节分片为准。

本地模拟Slurm、真实隔离Python子进程与USR1、网络失败隔离、>64MiB字节恢复、增量不重扫、CSV多行记录均有定向测试。尚未在学校真实GPU上执行第九轮，不能把本地兼容检查写成已经现场训练通过。

最终定向验证：学校同版Python3.12.3、`-I -S`模式下36项运行时测试、16项双通道测试与9项传输测试均通过；学校GPU未执行。

完整通道采用每完成任务/当前报告快照分别聚合，任务结果清单只列分组索引；不会等待所有轮次打成单一巨大包。正常成功上传路径缓存Release资产清单，POST结果不明时强制重新查询以避免重复块。

Release按科学任务key的SHA256首字节模64分桶，避免1548个任务集中到单一Release碰到资产数限制；分组manifest记录实际下载URL，分块按SHA缓存，可引用已验证的相同字节资产。
