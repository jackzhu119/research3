R1 / R11 已完成结果后台补传（2026-10-29）

入口：START-PAPER3-R1-R11-RESULTS.py --start
状态：START-PAPER3-R1-R11-RESULTS.py --check
Python：/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python
运行参数：-I -S -B -u

用途
先上传两个轮次已经存在的原始论文证据（回执、指标、表格、报告），然后上传所有真实完成输出的完整文件，包括模型/断点、预测、数组、配置和训练结果日志。不会重算实验或报告。原始文件保留。
不安装依赖，不导入Torch，不申请GPU，不调用sbatch/scancel，不恢复R1/R11训练，不移除科学任务STOP。

后台运行
--start 会启动四个独立后台通道：r1-paper、r11-paper、r1-full、r11-full。
看见 R1_R11_RESULT_EXPORT_BACKGROUND_STARTED 且 terminal_can_close:true 后可以关闭终端/电脑。后台上传依赖学校login1继续运行；login1重启后重新执行--start，复用已经校验通过的上传记录。
网络失败自动重试，无累计次数或累计运行时限。--start重复执行不会复制已在运行的四个通道。
每个完整输出组在新Python进程中处理，避免长进程积累内存；分片上传后做远端完整读取SHA256校验，成功后仅清理上传程序生成的临时分片。
状态及日志：$HOME/paper3-r1-r11-results-upload/<通道>/status.json、upload.log。
仅停止这次补传可创建 $HOME/paper3-r1-r11-results-upload/STOP；不影响训练，重新--start只清除此上传专用STOP。

R11与已有上传的衔接
论文通道复用原paper-upload状态。原full上传继续运行直到两个轮次的论文优先阶段就绪。
接手R11 full时，只按原process.json、完整命令行、UID、/proc启动时间确认原CPU上传进程；先TERM，12秒后仍未退出才对同一PID使用KILL。不会向进程组、GPU任务或训练管理器发信号。
补传与旧程序使用相同full-upload.lock、groups.json、source-captures.json和原始任务归档证明，直接复用已回传分组。
未识别的锁持有者保持原样，补传显示等待；不重复写入。

完整性的含义
R1：逐个真实完成回执核对所属输出，单独披露整轮完成状态。已有观测245个完成动作不等于整轮完成；缺失/损坏的已声明完成回执会列为待解决，不假称齐全。
R11：现有3190个完成任务与计划10270个任务分开计数。补传齐当前3190个结果不会标为10270项实验已全部完成。
两个轮次的未完成任务不启动、不生成空结果。
不归档凭证、原始EEG输入、虚拟环境、共享缓存或调度管理目录。R11专用runtime/logs中的Slurm作业日志额外保留为现有快照，不充当完成证明。已完成科学输出内部的实际结果文件与日志按各原生归档契约保留。原始科学数据字节不重写。

仓库位置（jackzhu119/research3，paper3-results分支）
R1论文：results/paper3/round-01/PAPER_DATA_LATEST.json
R1完整：results/paper3/round-01/COMPLETED_OUTPUT_LATEST.json
R11论文：results/paper3/round-11/COMPLETED_PAPER_LATEST.json
R11完整：results/paper3/round-11/COMPLETED_FULL_LATEST.json 与 results/paper3/round-11/BACKFILL_LATEST.json
索引包含不可变manifest及GitHub Release分片URL、SHA256和大小。大文件在Release分片中，不要求出现在GitHub代码文件列表。

兼容范围
Python3.12标准库；沿用学校已安装的paper3r11原生运营模块及已有GitHub凭证和CA证书。无pip，无测试计算作业。
离线代码以内容摘要命名安装；重复启动先校验自身10个代码文件，不扫描全项目。不会修改现有不可变科研代码包。
--check读取状态文件，不启动上传、不扫描科研目录。

授权与执行边界
这段代码尚需用户在学校login1执行；提供下载链接不能证明学校上传已经启动。以四通道状态和远端校验指针确认完成。
