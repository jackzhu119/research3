"""Run the existing R1 worker, keeping metadata fixes out of numerical kernels."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def option(argv, name):
    if argv.count(name) != 1:
        raise ValueError('EXACT_NATIVE_OPTION_REQUIRED: '+name)
    return argv[argv.index(name)+1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def patch_parent(worker, package):
    from single_gpu import atomic
    import receipt_projection
    worker.atomic_json = atomic
    original_bridge = worker._bridge
    def management_bridge(path):
        bridge = original_bridge(path)
        if not getattr(bridge, '_single_gpu_metadata_projection', False):
            receipt_projection.install(bridge)
            bridge._single_gpu_metadata_projection = True
        return bridge
    worker._bridge = management_bridge
    original_run_child = worker.run_child
    def run_child(command, **kwargs):
        command = list(command)
        native = str(Path(package).resolve()/'worker.py')
        if command.count(native) != 1 or '--child' not in command:
            raise ValueError('EXACT_NATIVE_ACTION_CHILD_COMMAND_REQUIRED')
        command[command.index(native)] = str(HERE/'worker_runner.py')
        return original_run_child(command, **kwargs)
    worker.run_child = run_child
    def progress(root, action):
        # Monitoring must not load a multi-megabyte scientific receipt in a thread.
        folder = Path(root)/Path(action['expected_receipt']).parent
        rows = []
        for name in ('progress.json', 'execution_progress.json'):
            path = folder/name
            try:
                with path.open('rb') as handle:
                    data = handle.read(2*1024*1024+1)
                if len(data) > 2*1024*1024:
                    continue
                value = json.loads(data)
                if isinstance(value, dict):
                    rows.append(dict(file=str(path.relative_to(root)), values={
                        k:value[k] for k in worker.PROGRESS_KEYS if k in value}))
            except (OSError, ValueError):
                continue
        return rows
    worker.progress_for_action = progress


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    package = Path(option(argv, '--package')).resolve()
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    worker = load('_paper3_r1_retained_single_gpu_worker', package/'worker.py')
    if '--child' in argv:
        import finalization_cache
        import action_startup
        home = Path(os.environ.get('HOME', str(Path.home())))
        cache_path = home/'paper3-r1-full-repair/2bfd664251ee5d1bcaf5/verified_context_cache.py'
        retained_cache = load('_r1_single_gpu_retained_verified_cache', cache_path)
        finalization_cache.install(action_startup, retained_cache, Path(option(argv, '--root')).resolve())
    else:
        patch_parent(worker, package)
    return worker.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
