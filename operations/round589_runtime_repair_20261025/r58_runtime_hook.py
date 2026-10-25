"""Narrow startup corrections without changing a saved scientific recipe.

Only the named Round 5/8/9 worker import is compiled with the exact corrections
below. Native source files, plans, task identities and checkpoints stay intact.
This module uses only the standard library; install() runs before Torch import.
"""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys


VERSION = 'r5-r8-r9-runtime-fix-v1'
with open(__file__, 'rb') as _hook_file:
    HOOK_SHA256 = hashlib.sha256(_hook_file.read()).hexdigest()
OLD_R5_CONFIGURATION = '''    variant = values.setdefault("variant", task.get("variant", "tangent_profiled"))
'''
NEW_R5_CONFIGURATION = '''    variant = values.setdefault("variant", task.get("variant", "tangent_profiled"))
    if task.get("variant") == "support_EA":
        variant = "support_EA"
'''
OLD_BALANCED_LOSS = '''def balanced_loss(logits, y):
    weights = logits.new_empty(len(y))
    for c in (0,1):
        mask = y == c
        if not bool(mask.any()):
            raise ValueError('Both query classes are required')
        weights[mask] = .5 / mask.sum()
    return (weights * F.cross_entropy(logits, y, reduction='none')).sum()
'''
NEW_BALANCED_LOSS = '''def balanced_class_weights(y, dtype):
    """Give each class half the loss mass, using the consumer's precision.

    Integer division otherwise creates a tensor in the process default float
    dtype. CUDA indexed assignment requires that tensor to match the target.
    """
    weights = torch.empty(len(y), dtype=dtype, device=y.device)
    for c in (0,1):
        mask = y == c
        if not bool(mask.any()):
            raise ValueError('Both query classes are required')
        weights[mask] = .5 / mask.sum().to(dtype=dtype)
    return weights


def balanced_loss(logits, y):
    weights = balanced_class_weights(y, logits.dtype)
    return (weights * F.cross_entropy(logits, y, reduction='none')).sum()
'''
OLD_TRAIN_WEIGHTS = '''        weights = torch.empty(len(y), device=y.device, dtype=torch.float64)
        for c in (0,1):
            weights[y==c] = .5 / (y==c).sum()
'''
NEW_TRAIN_WEIGHTS = '''        weights = balanced_class_weights(y, torch.float64)
'''


def corrected_source(fullname, source):
    """Apply only exact declared spans; accept a repeated application."""
    if fullname not in ('paper3r5.worker', 'paper3r8.worker', 'paper3r9.worker'):
        return source, []
    pairs = ([('support_EA_reference_dispatch', OLD_R5_CONFIGURATION, NEW_R5_CONFIGURATION)]
             if fullname == 'paper3r5.worker' else
             [('balanced_loss', OLD_BALANCED_LOSS, NEW_BALANCED_LOSS)])
    if fullname == 'paper3r8.worker':
        pairs.append(('train_step_class_weights', OLD_TRAIN_WEIGHTS, NEW_TRAIN_WEIGHTS))
    applied = []
    for label, old, new in pairs:
        if source.count(new) == 1 and source.replace(new, '', 1).count(old) == 0:
            continue
        if source.count(old) == 1 and source.count(new) == 0:
            source = source.replace(old, new, 1)
            applied.append(label)
        else:
            raise RuntimeError('Unexpected native %s span for %s' % (label, fullname))
    return source, applied


class CorrectedWorkerLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        raw = self.get_data(self.path)
        source = raw.decode('utf-8')
        corrected, changes = corrected_source(fullname, source)
        before = hashlib.sha256(raw).hexdigest()
        after = hashlib.sha256(corrected.encode('utf-8')).hexdigest()
        # Compile the corrected copy directly; never create a changed native
        # worker file or a pyc cache that could be mistaken for native bytes.
        code = compile(corrected, self.path, 'exec', dont_inherit=True)
        self.fix_receipt = {'version': VERSION, 'module': fullname,
            'original_file': self.path, 'source_sha256': before,
            'executed_source_sha256': after, 'corrections': changes,
            'runtime_hook_sha256': HOOK_SHA256,
            'torch_already_imported_at_compile': 'torch' in sys.modules,
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'native_source_file_changed': False}
        print(json.dumps({'event': 'RUNTIME_WORKER_CORRECTION',
                          **self.fix_receipt}, sort_keys=True), flush=True)
        return code

    def exec_module(self, module):
        super().exec_module(module)
        module.__paper3_runtime_correction__ = dict(self.fix_receipt)
        native_put = module.put
        receipt = dict(self.fix_receipt)

        def put_with_provenance(path, value):
            # Add operational provenance only to inspectable task receipts.
            # Identity, task declarations and checkpoint pointers are untouched.
            if (os.path.basename(os.fspath(path)) in ('result.json', 'runtime.json', 'error.json')
                    and isinstance(value, dict)):
                value = dict(value, runtime_corrections=receipt)
            return native_put(path, value)

        module.put = put_with_provenance


class WorkerCorrectionFinder(importlib.abc.MetaPathFinder):
    paper3_runtime_correction = VERSION

    def __init__(self):
        self.targets = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            return None
        spec.loader = CorrectedWorkerLoader(fullname, spec.origin)
        return spec


def install(roundmodule):
    """Install before the original round entry imports its scientific worker."""
    if roundmodule == 'paper3r5':
        previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if previous not in (':4096:8', ':16:8'):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print(json.dumps({'event': 'R5_DETERMINISTIC_CUBLAS_READY',
            'value': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'torch_already_imported': 'torch' in sys.modules}, sort_keys=True), flush=True)
    if roundmodule not in ('paper3r5', 'paper3r8', 'paper3r9'):
        return {'roundmodule': roundmodule, 'version': VERSION, 'corrections': []}
    finder = next((item for item in sys.meta_path
        if getattr(item, 'paper3_runtime_correction', None) == VERSION), None)
    if finder is None:
        finder = WorkerCorrectionFinder()
        sys.meta_path.insert(0, finder)
    finder.targets.add(roundmodule + '.worker')
    return {'roundmodule': roundmodule, 'version': VERSION,
            'target': roundmodule + '.worker'}
