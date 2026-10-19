# 论文3固定包：受控 source pilot 启动说明

此说明供未来交付的真实 READY 包使用，不表示当前 W2 或 READY 已完成。完整命令以**包内 `STARTUP.md`** 为准。正式科研 fits 尚未发生，W7 尚未冻结；外测 query 保持封存。

1. 确认实际交付状态为 `READY_WAITING_USER_GPU_BOOT`，包的 SHA256 与随附校验文件一致，且 `BUILD.json` 中 `ready_for_user_boot_pilot` 为 `true`。标有 CHECKPOINT 的包不能开训。解压后进入 `paper3/`，先核验 `EXECUTION_MANIFEST.sha256`。
2. 仅由用户本人手动开机。使用 CPython 3.12、Linux x86_64 和全新虚拟环境；驱动与固定 CUDA12.8 profile 的实际兼容性、GPU 恢复和计时均须开机后实测。工具不会启动或重启 RunPod。
3. 按 `STARTUP.md` 顺序准备：从 `bootstrap_wheels` 离线安装 R2 工具；用已有 R2 环境变量恢复 `audit/r2_dependencies_receipt.json` 与 `audit/r2_dependency_extras_receipt.json` 覆盖的固定 110 个 wheel；随后用 `--no-index` 按 `requirements-gpu.lock` 安装，运行 `pip check`，并将完整 `pip list --format=json` 保存至 `audit/gpu_resolved_dependencies.json`。不复制 `.env`、密钥或论文1/2配置。
4. 数据只恢复 `audit/source_full/cache_backup_receipt.json` 所列 source 缓存，完成 GET/字节数/SHA256 核验；无需恢复 raw EEG 或 Stieger episode 缓存。运行 `python -m p3work doctor` 和 `python tools/check_pilot_readiness.py --require-ready`。所有准备命令成功、真实 W1/W2 通过后才能进入 pilot。

用户手动开机且上述检查通过后，只执行一次：

```bash
python tools/run_controlled_pilot.py --root . --output runs/controlled_source_pilot_seed20261018 --execute --user-started-gpu
```

这是 source 工程检查：100 次连续更新，对比独立的 50 次更新、保存/重载、再更新 50 次，共 200 次；固定 source 预测重放不增加更新。它不启动后续正式科研 fits。

attempt 返回后，无论成功或失败，单独执行备份；不要用 `&&` 让 pilot 的非零退出跳过备份：

```bash
python tools/r2_backup.py --manifest runs/controlled_source_pilot_seed20261018/engineering_backup_manifest.json --root . --prefix paper3/source-engineering-pilot/seed20261018/ --receipt audit/r2_source_engineering_pilot_receipt.json --execute
```

pilot 状态以 `runs/controlled_source_pilot_seed20261018/engineering_receipt.json` 为准；备份只有实际执行且 R2 回执为 `status: verified`，才能报告上传和完整 GET/SHA256 回读成功。备份成功不会改变 pilot 的失败状态。

始终保留 `runs/controlled_pilot_job_claim.json`，不删除 claim、不换目录重跑。若早期门禁或硬中断未生成结果 manifest，保留现有产物并报告缺口，不能为补备份证据重复训练。
