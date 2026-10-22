"""Deployment compatibility cannot silently amend numerical implementations."""
import hashlib
import json
from pathlib import Path

import pytest

from hpc import environment

ROOT = Path(__file__).resolve().parents[1]
FILES = ('tools/run_controlled_pilot.py', 'p3work/source_training.py')


@pytest.fixture
def guarded(tmp_path):
    original = json.loads((ROOT / 'hpc/scientific_code.json').read_text())
    manifest = {'schema_version': 'paper3-hpc-software-overlay-v1', 'baseline_files': [],
                'deployment_support_files': [], 'deployment_overrides': {},
                'deployment_only_ast_guards': original['deployment_only_ast_guards']}
    for relative in FILES:
        raw = (ROOT / relative).read_bytes()
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        row = {'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        manifest['baseline_files'].append(row)
        manifest['deployment_overrides'][relative] = dict(row)
    (tmp_path / 'hpc').mkdir()
    (tmp_path / 'hpc/scientific_code.json').write_text(json.dumps(manifest))
    return tmp_path


def amend_deployment_hash(root, relative, old, new):
    path = root / relative
    raw = path.read_text(); assert old in raw
    path.write_text(raw.replace(old, new, 1))
    manifest_path = root / 'hpc/scientific_code.json'
    manifest = json.loads(manifest_path.read_text())
    raw = path.read_bytes()
    manifest['deployment_overrides'][relative].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest_path.write_text(json.dumps(manifest))


def test_registered_environment_adapters_leave_numerical_AST_intact(guarded):
    assert environment.verify_code(guarded)['numerical_kernel_overrides'] == 0


def test_updated_file_hash_cannot_authorize_changed_learning_rate(guarded):
    amend_deployment_hash(guarded, FILES[0], 'LEARNING_RATE = 0.001', 'LEARNING_RATE = 0.01')
    with pytest.raises(environment.EnvironmentError, match='scientific_AST_changed'):
        environment.verify_code(guarded)


def test_deployment_entrypoint_exception_cannot_change_the_200_update_body(guarded):
    amend_deployment_hash(guarded, FILES[0],
                         '_updates(cache, model, optimizer, sampler, 0, 100, deadline=deadline',
                         '_updates(cache, model, optimizer, sampler, 0, 101, deadline=deadline')
    with pytest.raises(environment.EnvironmentError, match='original_pilot_update_or_exact_resume_algorithm_changed'):
        environment.verify_code(guarded)


def test_updated_file_hash_cannot_weaken_the_exact_resume_verifier(guarded):
    amend_deployment_hash(guarded, FILES[1],
                         'receipt.get("actual_completed_updates") != 200',
                         'receipt.get("actual_completed_updates") != 100')
    with pytest.raises(environment.EnvironmentError, match='scientific_AST_changed'):
        environment.verify_code(guarded)


def test_compatibility_requires_the_original_algorithm_guards(guarded):
    path = guarded / 'hpc/scientific_code.json'; value = json.loads(path.read_text())
    del value['deployment_only_ast_guards']; path.write_text(json.dumps(value))
    with pytest.raises(environment.EnvironmentError, match='scientific_AST_guards_required'):
        environment.verify_code(guarded)


def test_deployment_function_allowlist_cannot_exempt_update_kernel(guarded):
    path = guarded / 'hpc/scientific_code.json'; value = json.loads(path.read_text())
    value['deployment_only_ast_guards'][FILES[0]]['deployment_functions'].append('_updates')
    path.write_text(json.dumps(value))
    with pytest.raises(environment.EnvironmentError, match='deployment_function_allowlist_changed'):
        environment.verify_code(guarded)
