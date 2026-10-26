"""Round 9 compatibility corrections compiled only in the worker process.

Scientific source files and saved experiment identities stay byte-for-byte intact.
Only weighted-median cumulative metadata is computed on the CPU; model fitting
and prediction continue on the allocated GPU with strict determinism enabled.
"""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys


VERSION = 'paper3-r9-cumsum-fix-v2'
with open(__file__, 'rb') as _stream:
    HOOK_SHA256 = hashlib.sha256(_stream.read()).hexdigest()

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
    """Give each class half the loss mass in the consumer's precision."""
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
OLD_CUMSUM = '        cumulative=torch.cumsum(pairweights[order],0)\n'
NEW_CUMSUM = '        cumulative=torch.cumsum(pairweights[order].detach().cpu(),0).to(pairweights.device)\n'


def corrected_source(fullname, source):
    pairs = {
        'paper3r9.worker': [('balanced_loss', OLD_BALANCED_LOSS, NEW_BALANCED_LOSS)],
        'paper3r9.gp': [('weighted_median_deterministic_cpu_cumsum', OLD_CUMSUM, NEW_CUMSUM)],
    }.get(fullname, [])
    applied = []
    for label, old, new in pairs:
        if source.count(new) == 1 and source.replace(new, '', 1).count(old) == 0:
            continue
        if source.count(old) != 1 or source.count(new) != 0:
            raise RuntimeError('Unexpected native %s span for %s' % (label, fullname))
        source = source.replace(old, new, 1)
        applied.append(label)
    return source, applied


class CorrectedModuleLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        raw = self.get_data(self.path)
        corrected, changes = corrected_source(fullname, raw.decode('utf-8'))
        self.fix_receipt = {
            'version': VERSION, 'module': fullname, 'original_file': self.path,
            'source_sha256': hashlib.sha256(raw).hexdigest(),
            'executed_source_sha256': hashlib.sha256(corrected.encode()).hexdigest(),
            'corrections': changes, 'runtime_hook_sha256': HOOK_SHA256,
            'native_source_file_changed': False,
            'torch_already_imported_at_compile': 'torch' in sys.modules,
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        }
        print(json.dumps({'event': 'R9_RUNTIME_CORRECTION', **self.fix_receipt},
                         sort_keys=True), flush=True)
        return compile(corrected, self.path, 'exec', dont_inherit=True)

    def exec_module(self, module):
        super().exec_module(module)
        receipt = dict(self.fix_receipt)
        module.__paper3_runtime_correction__ = receipt
        if module.__name__ != 'paper3r9.worker':
            return
        gp = sys.modules.get('paper3r9.gp')
        dependency = getattr(gp, '__paper3_runtime_correction__', None)
        if dependency is None or dependency.get('version') != VERSION:
            raise RuntimeError('Round 9 GP correction was not installed before worker import')
        receipt['dependency_corrections'] = [dict(dependency)]
        native_put = module.put

        def put_with_provenance(path, value):
            if (os.path.basename(os.fspath(path)) in ('result.json', 'runtime.json', 'error.json')
                    and isinstance(value, dict)):
                value = dict(value, runtime_corrections=receipt)
            return native_put(path, value)

        module.put = put_with_provenance


class CorrectionFinder(importlib.abc.MetaPathFinder):
    paper3_runtime_correction = VERSION
    targets = frozenset(('paper3r9.worker', 'paper3r9.gp'))

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            return None
        spec.loader = CorrectedModuleLoader(fullname, spec.origin)
        return spec


def install(roundmodule='paper3r9'):
    if roundmodule != 'paper3r9':
        raise ValueError('This correction is only for Round 9')
    loaded = [name for name in CorrectionFinder.targets if name in sys.modules]
    if loaded and not all(getattr(sys.modules[name], '__paper3_runtime_correction__', {}).get('version') == VERSION
                          for name in loaded):
        raise RuntimeError('Install Round 9 correction before native worker/GP import')
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if previous not in (':4096:8', ':16:8'):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if not any(getattr(item, 'paper3_runtime_correction', None) == VERSION for item in sys.meta_path):
        sys.meta_path.insert(0, CorrectionFinder())
    print(json.dumps({'event': 'R9_DETERMINISTIC_CUMSUM_READY', 'version': VERSION,
                      'runtime_hook_sha256': HOOK_SHA256,
                      'cumulative_weights_device': 'cpu', 'global_determinism_disabled': False,
                      'torch_already_imported': 'torch' in sys.modules}, sort_keys=True), flush=True)
    return {'version': VERSION, 'targets': sorted(CorrectionFinder.targets)}
