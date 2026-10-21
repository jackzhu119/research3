# 论文3：直接从 GitHub 启动

在本人已经开机的 RunPod Linux GPU 服务器上，新开 Terminal 执行：

```bash
cd /workspace
git clone --single-branch --branch research/paper3-online-20261021 https://github.com/jackzhu119/research3.git paper3-online
bash /workspace/paper3-online/RUN_ON_RUNPOD.sh
```

如果 `/workspace/paper3-online` 已经存在，直接执行最后一行即可，不需要再次 clone。

本入口自动查找已有论文3进程及原工作锁。有正在运行的任务时直接接入其 tmux 窗口，新增只读状态面板，不重新启动、不重新解包。上方输入窗口保持焦点，凭证只在真实提示出现后输入。

确实没有任务时，入口会自动从本仓库固定 GitHub Release 下载 17.7 MB 的代码/SDK 启动 ZIP，完整核对 SHA256，安全补齐尚未存在的文件，创建 tmux 并沿用原 START。无需手工下载、上传或解压 ZIP。发布包的固定身份是 `5bb056d39254854707e91c4eb2ce2a4cc6e8e922a9337a52f3daea08b0fb5489`（17,717,251 字节）。

原入口仍按顺序隐藏输入 GitHub Token、R2 Account ID/Endpoint、Bucket、Access Key ID、Secret Access Key。GitHub Token 须具有目标仓库 Contents 读写权限，R2 凭证须具有保存固定数据包的已有桶的对象读写权限。凭证不进入命令参数、URL、文件或日志。

GitHub 保存代码、启动器、公开依赖 wheel 和不含凭证的发布清单。研究数据、固定缓存和实际结果继续使用已有 Cloudflare R2，逐对象完整 GET/SHA256 核验。

## 当前任务只读检查

这条命令不会下载数据、启动训练或中断任务，每10秒刷新文件数量、控制进程CPU活动和GPU数值状态：

```bash
bash /workspace/paper3-online/RUN_ON_RUNPOD.sh --status
```

只查看一次：

```bash
bash /workspace/paper3-online/RUN_ON_RUNPOD.sh --status --once
```

按 Ctrl+C 只退出该检查。准备阶段 GPU 空闲是正常现象；解包期间进度 JSON 可能只在整包完成后更新，请结合落盘文件数和进程 CPU 活动判断。单次计数不变不能认定失败，GPU占用也不能证明科学训练完成。

## 原科学门禁

原197份软件输入及固定数据闭包不变。原入口恢复数据与离线依赖后检查CUDA，先完成200 update的受控source pilot及中断恢复验证，再推进主线14方法、3 seeds。主线真实冻结后才推进Yang/Dreyer两条独立source扩展臂；Ma不训练，CAR3不增加独立受试者，W7外测query保持封存，`legacy_baseline_v1`是历史基线。

训练入口返回或暂停并释放锁后，原程序自动备份已实际产生的有限结果，并完整GET/SHA256回读。备份失败不改变训练原状态。只重试备份时执行：

```bash
bash /workspace/Paper3_OneClick/START.sh --backup-only
```

GPU驱动、CUDA兼容性、实际训练和恢复效果以服务器上的真实回执为准。本次发布不操作RunPod API，不创建、启动或重启实例，也不将CPU测试称为训练完成。
