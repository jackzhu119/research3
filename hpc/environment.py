"""Checks for the existing school environment; never installs software."""
from __future__ import annotations

import hashlib
import ast
import importlib
import json
import os
from pathlib import Path
import platform
import sys

THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


class EnvironmentError(ValueError):
    pass


def verify_code(root):
    root = Path(root)
    source = root / 'hpc/scientific_code.json'
    value = json.loads(source.read_text())
    if value.get('schema_version') != 'paper3-hpc-software-overlay-v1':
        raise EnvironmentError('frozen_software_overlay_required')
    changed = value['deployment_overrides']
    allowed = {'p3work/training_campaign.py', 'p3work/source_extension_downstream.py',
               'tools/run_training_campaign.py', 'tools/run_source_extension_downstream.py',
               'tests/test_training_campaign.py', 'tests/test_data_plan_deploy_manifest_v2.py',
               'tests/test_audit_completed_stieger_w3.py', 'tests/test_audit_completed_stieger_w3_v2.py',
               'tools/run_controlled_pilot.py', 'p3work/source_training.py'}
    if not set(changed) <= allowed:
        raise EnvironmentError('scientific_kernel_override_forbidden')
    for row in value['baseline_files'] + value.get('deployment_support_files', []):
        relative = row['path']
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise EnvironmentError('unsafe_software_identity_path')
        path = root / relative
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise EnvironmentError('software_symlink_forbidden')
        expected = changed.get(relative, row)
        raw = path.read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise EnvironmentError('frozen_software_SHA256_mismatch:' + relative)
    compatibility = {'tools/run_controlled_pilot.py', 'p3work/source_training.py'} & set(changed)
    guards = value.get('deployment_only_ast_guards', {})
    if compatibility != set(guards):
        raise EnvironmentError('deployment_only_scientific_AST_guards_required')
    for relative in sorted(compatibility):
        guard = guards[relative]
        tree = ast.parse((root / relative).read_bytes())
        functions = set(guard['deployment_functions'])
        expected = ({'_resolved_dependencies', '_finalize_result_manifests', 'execute_pilot'}
                    if relative == 'tools/run_controlled_pilot.py' else {'_verify_pilot_result_manifests'})
        if functions != expected:
            raise EnvironmentError('deployment_function_allowlist_changed')
        untouched = [node for node in tree.body
                     if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in functions]
        raw = '\n'.join(ast.dump(node, include_attributes=False) for node in untouched).encode()
        if hashlib.sha256(raw).hexdigest() != guard['untouched_module_AST_sha256']:
            raise EnvironmentError('scientific_AST_changed_outside_deployment_functions:' + relative)
        if relative == 'tools/run_controlled_pilot.py':
            execution = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute_pilot')
            body = next(node for node in execution.body if isinstance(node, ast.Try)).body
            start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == 'deadline' for target in node.targets))
            raw = '\n'.join(ast.dump(node, include_attributes=False) for node in body[start:]).encode()
            if hashlib.sha256(raw).hexdigest() != guard['unchanged_200_update_and_exact_resume_AST_sha256']:
                raise EnvironmentError('original_pilot_update_or_exact_resume_algorithm_changed')
    return {'manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'baseline_files_verified': len(value['baseline_files']),
            'deployment_support_files_verified': len(value.get('deployment_support_files', [])),
            'controller_and_test_overrides': sorted(changed), 'numerical_kernel_overrides': 0}


def inspect_environment(*, gpu=False):
    """Import only inside an allocated job. CPU probes never fit a model."""
    if sys.version_info[:2] != (3, 12):
        raise EnvironmentError('existing_Python_3_12_required')
    if any(os.environ.get(key) != '1' for key in THREADS):
        raise EnvironmentError('original_scientific_thread_caps_must_equal_one')
    versions = {}
    for name in ('numpy', 'scipy', 'sklearn', 'torch', 'pytest'):
        try:
            module = importlib.import_module(name)
        except ImportError:
            raise EnvironmentError('existing_environment_missing_module:' + name) from None
        versions[name] = str(module.__version__)
    torch = sys.modules['torch']
    # Verify the APIs used by the unchanged checkpoint/precision implementation.
    if not all(hasattr(torch, name) for name in
               ('load', 'save', 'use_deterministic_algorithms', 'set_float32_matmul_precision')):
        raise EnvironmentError('installed_torch_missing_required_API')
    import inspect
    if 'weights_only' not in inspect.signature(torch.load).parameters:
        raise EnvironmentError('installed_torch_missing_weights_only_load')
    if torch.is_autocast_enabled('cpu'):
        raise EnvironmentError('unexpected_CPU_autocast_enabled')
    result = {'schema_version': 'paper3-hpc-existing-environment-v1',
              'python': platform.python_version(), 'python_executable': sys.executable,
              'libc': list(platform.libc_ver()), 'modules': versions,
              'existing_environment_reused': True, 'packages_installed': 0,
              'scientific_threads': {key: os.environ[key] for key in THREADS},
              'numerical_fit_performed': False, 'CUDA_probe_performed': False}
    if gpu:
        from tools.train_source import cuda_runtime
        runtime = cuda_runtime()  # Original deterministic CUDA/runtime identity.
        # A tiny device operation is a deployment check, never a fit or pilot.
        probe = torch.ones(1, device='cuda:0') + 1
        torch.cuda.synchronize()
        if probe.item() != 2:
            raise EnvironmentError('allocated_CUDA_operation_failed')
        result.update(CUDA_probe_performed=True, runtime=runtime)
    return result
