第一轮资源改动之外的独立结果补传（2026-10-28）

根据固定仓库证据 AUDIT.json：
- 第8轮2835/2835完成，论文数据33文件/545,964,970字节已完整回读验证；缺完整科学输出归档入口。
- 第12轮78任务完成、139项按预定科研规则未执行。完整通道已有78完成任务；公开论文指针停53，本地上传进度已74。补传会沿用学校已验证的缓存，补齐论文通道至所有实际完成任务，不把未执行139项伪装完成。
- 第13轮96/96任务完成；论文报告2文件已上传，旧observer每轮仅传4个完整文件，完整会话覆盖仍有积压。本工具补传完成任务目录的结果、逐会话预测和拟合断点。
- 第2、3、4、5、6、7、9、10轮已有完整归档，不重复。

执行方式：
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/START-PAPER3-MISSING-RESULTS.py" --start

返回 MISSING_RESULTS_BACKGROUND_STARTED 且 terminal_can_close=true 后，可关闭终端、断开电脑。
启动3个独立会话后台CPU进程：paper、full8、full13。两个完整补传进程等待优先论文阶段完成；原第8/13轮已验证论文数据复用；如旧指针缺失或不兼容，主动上传本地已有报告、CSV与完成任务摘要，不重新计算科学结果。第12轮论文通道追平后开始完整补传。
默认项目 /public/home/zhuziyuan/projects/paper3，默认第13轮plan为 runs/r13_update_validity/plan.json；如自行另设过输出，使用 --r13-plan <实际plan绝对路径>。

查看：
/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u "$HOME/START-PAPER3-MISSING-RESULTS.py" --check
后台目录：$HOME/paper3-missing-results-upload
日志：paper/upload.log、full8/upload.log、full13/upload.log
状态：各通道status.json与heartbeat.json；返回后台启动并不代表已上传完成。

完整结果按每个完成任务生成确定性tar.gz、32MiB分片，使用64个确定性GitHub Release桶规避单Release资产数上限。每片完整远端GET后验SHA256，每个源文件也有归档SHA。小回执一次冻结后复用；每完成一个task即持久记账，使用新进程处理下一task，避免内存随数千任务累积。已核验的本程序临时归档片可删除以节省本地空间，原科学文件不删除。

目的地：
- results/paper3/MISSING_RESULTS_PAPER_BACKFILL_20261028.json（优先论文阶段回执）
- results/paper3/round-08/LATEST.json 和 BACKFILL_LATEST.json（第8轮全部已声明任务输出）
- results/paper3/round-13/BACKFILL_LATEST.json（第13轮完成任务科学输出）
- 每个pointer引用manifest；manifest.groups引用逐task清单；按每组parts.index顺序拼接为tar.gz，逐源文件大小/SHA校验。

只用Python3.12标准库，不安装依赖、导入Torch、请求GPU、调用Slurm或发送训练进程信号。网络和凭证错误只在后台重试；新进程读取已有GitHub凭证，不记录token/header/body。
“所有文件”范围为本轮完成任务科学输出、计划和报告，含模型/优化器/随机数断点（如果该任务产生过）。保留已完成结果原文件里的预测、评价字段和真值；原始EEG、独立R13 evaluation-labels.npz、轮次外共享输入模型、环境、凭证、运行锁及日志不进入公开结果包。

本地验证见 VALIDATION.json：真实自解压隔离入口、后台独立会话、重复启动防重复、toy归档恢复SHA、模拟GitHub发布和私密路径拒绝。没有登录学校运行、没有以测试替代实际上传完成。
