"""Reapply the active R1 operational runtime in newly spawned owned children.

Python spawn preserves environment and import paths, but not the parent's
installed transfer/comparison functions. Use the same bridge as the current
allocation, without editing native code or changing scientific identities.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


MARKER = 'paper3-r1-owned-stage-bootstrap-v1'


def _f0_stage_identity(root, run, identity, actual):
    """Keep the existing arm claim lineage; actual source execution is separate."""
    historical = run.identity['runtime']
    if historical == actual:
        return identity
    if identity.get('runtime') != actual:
        raise ValueError('Source-stage actual runtime observation changed')
    import runtime_transfer
    claim = Path(root) / ('runs/source_extension_downstream/' + identity['arm'] + '/resources/claim.json')
    refs = [{'path': str(claim), 'sha256': hashlib.sha256(claim.read_bytes()).hexdigest()}] if claim.is_file() else []
    runtime_transfer.validate_transfer(root, historical, actual,
        'downstream:source_f0_arm_stage_lineage', refs)
    return dict(identity, runtime=copy.deepcopy(historical))


def _install_f0_lineage(root):
    """Amend only the missed administrative F0 dispatch identity comparison."""
    from p3work import source_extension_downstream as downstream
    import runtime_transfer
    if getattr(downstream, '_paper3_f0_stage_lineage_root', None) == str(root):
        return True
    path = Path(root) / 'p3work/source_extension_downstream.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != runtime_transfer.SOURCE['p3work/source_extension_downstream.py']:
        raise ValueError('Original source-extension stage implementation changed')
    tree = ast.parse(raw)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare_f0')
    anchor = ast.dump(ast.parse('run=ArmRun(root,catalog,runtime)').body[0], include_attributes=False)
    matches = [index for index,node in enumerate(definition.body)
               if ast.dump(node,include_attributes=False) == anchor]
    if len(matches) != 1:
        raise ValueError('Original source F0 arm-runtime dispatch anchor changed')
    definition.body.insert(matches[0]+1,
        ast.parse('identity = _paper3_f0_stage_identity(root, run, identity, runtime)').body[0])
    downstream.__dict__['_paper3_f0_stage_identity'] = _f0_stage_identity
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])), str(path), 'exec'), downstream.__dict__)
    downstream._paper3_f0_stage_lineage_root = str(root)
    return True


def _operational_module(package, name):
    path = package / (name + '.py')
    current = sys.modules.get(name)
    if current is not None and Path(current.__file__).resolve() == path.resolve():
        return current
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Active R1 operational module is unreadable: ' + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_owned_stage(root):
    root = Path(root).resolve()
    path = root / 'runs/training_campaign_parallel_v1/launch.json'
    # Older unsplit campaigns remain independent of the later R1 migration.
    if not path.is_file():
        return {'status': 'NOT_APPLICABLE_NO_PARALLEL_LAUNCH'}
    launch = json.loads(path.read_text())
    package = Path(launch['package']).resolve()
    payload = Path(launch.get('payload_manifest') or os.environ['PAPER3_CONTINUOUS_REPAIR_MANIFEST']).resolve()
    direct = Path(launch['direct_package']).resolve()
    seconds = int(launch.get('allocation_seconds', os.environ.get('PAPER3_HPC_WALL_TIME_SECONDS', 529200)))
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    # These are external operational modules, not native experiment modules.
    # Bind them to the current recorded standalone package rather than an older
    # combined installer that may still be present on the shared filesystem.
    for name in ('audited_runtime', 'hardware_comparison'):
        _operational_module(package, name)
    bridge = _operational_module(package, 'kernel_bridge')
    bridge.activate_runtime(root, payload, direct_package=direct, allocation_seconds=seconds)
    f0_lineage = _install_f0_lineage(root)
    receipt = {'status': 'R1_OWNED_CHILD_RUNTIME_BOOTSTRAPPED', 'version': MARKER,
        'root': str(root), 'package': str(package), 'payload_manifest': str(payload),
        'bridge_sha256': hashlib.sha256((package / 'kernel_bridge.py').read_bytes()).hexdigest(),
        'bootstrap_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'authorized_a100_transfer_installed': bool(getattr(sys.modules.get('runtime_transfer'),
            '_paper3_authorized_a100_transfer', False)),
        'development_comparison_installed': bool(getattr(sys.modules.get('p3work.development_evaluation'),
            '_paper3_parallel_hardware_comparison', False)),
        'source_f0_stage_lineage_installed': f0_lineage,
        'native_source_changed': False, 'checkpoint_changed': False,
        'actual_gpu_properties_replaced': False}
    print(json.dumps({'event': receipt['status'], **receipt}, sort_keys=True), flush=True)
    return receipt


def corrected_recovery_source(source):
    """Return an importable corrected copy of the two existing spawn targets."""
    marker = '    # PAPER3_R1_OWNED_STAGE_RUNTIME_BOOTSTRAP_V1\n'
    call = (marker + '    from r1_spawn_bootstrap import bootstrap_owned_stage\n'
            '    bootstrap_owned_stage(root)\n')
    targets = [
        '    from p3work import training_campaign as campaign\n'
        '    install(campaign, root)\n'
        '    return campaign._direct_original_stage_worker(\n',
        '    from p3work import source_extension_downstream as downstream\n'
        '    original_worker = downstream._owned_worker\n',
    ]
    changed = []
    for index, anchor in enumerate(targets):
        if call + anchor in source:
            continue
        if source.count(anchor) != 1:
            raise ValueError('Unexpected original R1 owned-stage entrypoint: ' + str(index))
        source = source.replace(anchor, call + anchor, 1)
        changed.append(('stage_worker', 'extension_stage_worker')[index])
    compile(source, 'campaign_recovery.py', 'exec')
    return source, changed
