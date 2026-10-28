第一轮单卡续跑 + 已完成轮次优先论文补传

下载并执行 START-PAPER3-R1-AND-UPLOAD.py --apply。
这是包含两个离线单文件程序的组合入口，不在学校安装依赖或运行测试作业。

第一部分：原第1轮继续，总共1 GPU / 4 CPU / 64G，分区沿用hpc_gpu。
精确交接旧CPU管理器和旧监控，保留原科学代码、任务身份、已有输出和断点。
如还有旧计算子进程则先完成当前动作，再启单卡；用户本次学校回传显示R1已无Slurm作业、claims为空。
主线完成后两条原扩展臂仍由唯一slot0继续，阶段和科学身份保留。
管理观察/上传各为独立有界子进程；管理90秒过短超时已改为900秒元数据周期且显示心跳。
strongest-control/freeze首次原生校验保留，重复成功验证仅当前动作复用；依赖变化拒绝发布。
实时进度和已有结果摘要自动上传，网络失败不停止计算。

第二部分：独立补传已跑完但远端结果仍有缺口的轮次。
先追平第12轮论文数据并确认/补传第8、13轮已有论文文件；然后补传第8、13轮完整科学输出。
第12轮已有78个完成任务完整输出，不重复；139项预定规则跳过不当作已跑实验。
完整输出含任务结果、预测、轨迹和checkpoint以及计划/report/catalog/CSV；
R8普通评估真值字段保留。原始EEG、环境、凭证、独立R13 evaluation-labels.npz输入包不在归档范围。
32MiB GitHub Release分片，完整远端GET/SHA核验后记账，断网重试、断终端继续。

关闭终端：看到R1_SINGLE_GPU_BACKGROUND_STARTED与MISSING_RESULTS_BACKGROUND_STARTED，
并且上传组件 terminal_can_close:true 后即可断开电脑；它们证明后台启动，不代表已分配GPU或上传完成。

只读检查：现有Python执行本入口 --check。
日志：
  $HOME/paper3-r1-single-gpu/work/controller.log
  $HOME/paper3-r1-single-gpu/work/observer.log
  $HOME/paper3-r1-single-gpu/work/results-observer.log
  $HOME/paper3-missing-results-upload/paper/upload.log
  $HOME/paper3-missing-results-upload/full8/upload.log
  $HOME/paper3-missing-results-upload/full13/upload.log

可单独执行 --uploads-only 或 --r1-only；默认同时处理两部分。
--apply明确恢复第一轮，会移除此前的R1停止标记；查看状态请使用--check。

开发验证：第一轮84项软件测试、补传17项软件测试、组合入口2项测试，
包括真实隔离Python入口、原科学worker→子进程→结果链路、原选择器输出一致、脱终端存活和模拟Slurm/GitHub。
没有从开发环境连接学校执行真实CUDA计算或替学校完成全量上传；现场状态以后续日志与远端完成指针为准。
