"""Resource migration preserves the already prepared numerical experiment."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import pytest

from paper3r6 import plan as science, submit
from paper3r6.io import put, read

PACKAGE = Path(__file__).resolve().parents[1]


def test_migration_keeps_every_scientific_source_and_recipe_unchanged():
    identity = read(PACKAGE / 'MIGRATION_SOURCE_IDENTITY.json')
    assert identity['scientific_changes'] is False
    assert science._code() == identity['scientific_code_sha256']
    assert hashlib.sha256((PACKAGE / 'paper3r6/plan.py').read_bytes()).hexdigest() == identity['plan_declaration_sha256']
    current = science.make_plan('/unused/project', '/unused/output')
    assert current['recipe_sha256'] == identity['recipe_sha256']
    assert len(current['tasks']) == identity['tasks'] == 198
    assert current['counts'] == identity['counts']
    assert science._digest(science._identity_spec(current)['tasks']) == identity['normalized_tasks_sha256']


def prior_installation(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    output = root / 'runs/followup_20261024_r6'
    old_package = tmp_path / 'original-installed-package'
    old_manifest = old_package / 'data/pan2023/manifest.json'
    old_manifest.parent.mkdir(parents=True)
    old_manifest.write_bytes((PACKAGE / 'data/pan2023/manifest.json').read_bytes())
    code = science._code()
    monkeypatch.setattr(science, '_code', lambda: code)
    monkeypatch.setattr(science, 'PACKAGE', old_package)
    previous = science.make_plan(root, output)
    previous['prepared_at'] = 1792837924.0
    put(output / 'plan.json', previous)
    monkeypatch.setattr(science, 'PACKAGE', PACKAGE)
    monkeypatch.setattr(submit, 'discover_credentials', lambda *_: None)
    return root, output, previous


def test_adopts_original_task_paths_and_preserves_plan_results_and_checkpoint_bytes(tmp_path, monkeypatch, capsys):
    root, output, previous = prior_installation(tmp_path, monkeypatch)
    task = previous['tasks'][0]
    put(output / task['key'] / 'result.json', {'status': 'COMPLETE'})
    put(output / task['key'] / 'latest.json', {'path': 'state-0.pt', 'outer_update': 470})
    checkpoint = output / task['key'] / 'state-0.pt'
    checkpoint.write_bytes(b'opaque original checkpoint: preparation must not load it')
    retained = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    regenerated = science.make_plan(root, output)
    old_pan = next(t for t in previous['tasks'] if t['family'] == 'transfer')
    new_pan = next(t for t in regenerated['tasks'] if t['family'] == 'transfer')
    assert old_pan['external_manifest'] != new_pan['external_manifest']
    assert previous['recipe_sha256'] == regenerated['recipe_sha256']
    assert submit.main(['--root', str(root), '--python', sys.executable]) == 0
    assert all(path.read_bytes() == value for path, value in retained.items())
    adopted = read(output / 'plan.json')
    assert adopted == previous
    launch = read(output / 'runtime/launch.json')
    assert launch['partition'] == 'hpc_gpu'
    assert launch['cpus'] == 12 and launch['slots'] == 3 and launch['memory_total'] == '128G'
    prepared = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert prepared['tasks'] == 198 and prepared['slurm_jobs'] == 1
    assert prepared['cpus_per_job'] == 36 and prepared['host_memory_per_job'] == '128G'
    # The unchanged production worker binds the original task locator. Keeping
    # that locator is required even though preparation's recipe is path stable.
    from paper3r6.worker import _identity
    assert _identity(previous, old_pan, {}, {}) == _identity(adopted, old_pan, {}, {})
    assert _identity(previous, old_pan, {}, {}) != _identity(regenerated, new_pan, {}, {})


@pytest.mark.parametrize('change', ['config', 'requires', 'seed', 'missing_manifest'])
def test_only_content_bound_locations_may_differ_between_prepared_tasks(change):
    left = [{'key': 'task', 'seed': 20261018, 'config': {'rank': 4}, 'requires': ['basis'],
             'external_manifest': '/old/manifest.json'}]
    right = copy.deepcopy(left)
    right[0]['external_manifest'] = '/new/manifest.json'
    assert submit._same_prepared_tasks(left, right)
    if change == 'config':
        right[0]['config']['rank'] = 8
    elif change == 'requires':
        right[0]['requires'] = ['other-basis']
    elif change == 'seed':
        right[0]['seed'] += 1
    else:
        right[0].pop('external_manifest')
    assert not submit._same_prepared_tasks(left, right)


def test_live_same_resource_controller_retains_original_launch_package(tmp_path, monkeypatch, capsys):
    root, output, _ = prior_installation(tmp_path, monkeypatch)
    args = ['--root', str(root), '--python', sys.executable]
    assert submit.main(args) == 0
    launch_path = output / 'runtime/launch.json'
    launch = read(launch_path)
    launch['package'] = '/old/installed-runtime'
    put(launch_path, launch)
    old_bytes = launch_path.read_bytes()
    with (output / 'runtime/controller.lock').open('a') as live_controller:
        fcntl.flock(live_controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert submit.main(args) == 0
    assert launch_path.read_bytes() == old_bytes
    prepared = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert prepared['existing_controller_reused']


def test_live_old_partition_controller_requires_handoff_without_altering_plan(tmp_path, monkeypatch):
    root, output, _ = prior_installation(tmp_path, monkeypatch)
    args = ['--root', str(root), '--python', sys.executable]
    assert submit.main(args) == 0
    launch_path = output / 'runtime/launch.json'
    old_launch = read(launch_path)
    old_launch.update(partition='hpc_gpu1', memory_total='164G')
    put(launch_path, old_launch)
    retained = {path: path.read_bytes() for path in (output / 'plan.json', launch_path)}
    with (output / 'runtime/controller.lock').open('a') as live_controller:
        fcntl.flock(live_controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match='stop that controller before changing its resources'):
            submit.main(args)
    assert all(path.read_bytes() == value for path, value in retained.items())
