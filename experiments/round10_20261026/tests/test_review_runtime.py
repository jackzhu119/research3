"""Independent engineering review regressions; every scheduler RPC is mocked."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper3r10 import selection, study
from paper3r10.platform import common
from paper3r10.platform import runtime, uploads
from paper3r10.platform.scheduler import ThreeJobScheduler
from paper3r10.statistics import SECONDARY_ABLATION_FAMILY
from test_platform_contract import complete, plan_fixture
from test_study_cli import prepared


def scheduler_fixture(tmp_path, *, terminal_code=1, reject=None):
    path, _ = plan_fixture(tmp_path)
    context = runtime.context(path)
    commands, allocations = [], []
    now = [1000.]

    def runner(command, **_):
        commands.append(command)
        if command[0] == 'sbatch':
            if reject:
                return SimpleNamespace(returncode=1, stdout='', stderr=reject)
            job = str(100 + len(allocations))
            name = next(item.split('=', 1)[1] for item in command if item.startswith('--job-name='))
            allocations.append((job, name))
            return SimpleNamespace(returncode=0, stdout=job + '\n', stderr='')
        if command[0] == 'sacct':
            rows = [f'{job}|FAILED|{terminal_code}:0|{scheduler.marker}|{name}|'
                    for job, name in allocations]
            return SimpleNamespace(returncode=0, stdout='\n'.join(rows), stderr='')
        assert command[0] == 'squeue'
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    scheduler = ThreeJobScheduler(context['runtime'], context['launch'], runner=runner, clock=lambda: now[0])
    return scheduler, commands, allocations, now


def test_boot_failure_before_task_claim_has_a_finite_allocation_budget(tmp_path):
    scheduler, commands, allocations, now = scheduler_fixture(tmp_path)
    for _ in range(20):
        scheduler.step()
        now[0] += 1000
    issued = len(allocations)
    assert issued <= 12, 'Four slots get at most three failed allocations each before quarantine'
    for _ in range(5):
        scheduler.step()
        now[0] += 1000
    assert len(allocations) == issued
    assert not any(command[0] == 'scancel' for command in commands)


def test_invalid_partition_submission_is_bounded_without_touching_other_jobs(tmp_path):
    scheduler, commands, _, now = scheduler_fixture(
        tmp_path, reject='sbatch: error: Batch job submission failed: Invalid partition name specified')
    for _ in range(20):
        scheduler.step()
        now[0] += 1000
    issued = sum(command[0] == 'sbatch' for command in commands)
    assert issued <= 3, 'A structural resource/account configuration error requires repair, not endless submission'
    assert all(command[0] in ('squeue', 'sacct', 'sbatch') for command in commands)


def test_checkpoint75_replacements_do_not_consume_failure_quarantine_budget(tmp_path):
    scheduler, _, allocations, now = scheduler_fixture(tmp_path, terminal_code=75)
    for _ in range(8):
        scheduler.step()
        now[0] += 1000
    assert len(allocations) >= 12
    prior = len(allocations)
    for _ in range(8):
        scheduler.step()
        now[0] += 1000
    assert len(allocations) > prior


def test_stop_before_claim_preserves_state_and_does_not_spawn_science(tmp_path):
    path, plan = plan_fixture(tmp_path)
    state = runtime.runtime_dir(plan)
    state.mkdir(parents=True)
    (state / 'STOP').write_text('explicit fixture stop\n')
    checkpoint = Path(plan['output']) / 'retained-checkpoint.bin'
    checkpoint.write_bytes(b'original scientific checkpoint')

    def forbidden(*_, **__):
        raise AssertionError('Paused stage must not start a scientific child')

    assert runtime.worker_loop(path, 0, device='cpu', popen=forbidden) == 75
    assert checkpoint.read_bytes() == b'original scientific checkpoint'
    assert not (state / 'tasks.json').exists()


def actual_freeze_contract(prepared, monkeypatch, scientific_scope='REAL_DATA'):
    root, package, registry, registry_path, descriptor = prepared
    development = study.build_plan(root, descriptor, package=package)
    if scientific_scope == 'SOFTWARE_TEST_ONLY':
        development['synthetic_fixture'] = True
        development = common.sign_plan(development)
    plan_path = study.write_plan(development, root / 'development-plan.json')
    decision = {
        'status': 'PASS', 'decision_identity': 'd' * 64, 'source_result_receipts': [],
        'scientific_scope': scientific_scope, 'study_registry_sha256': descriptor['sha256'],
        'reference_strategy': development['reference_strategy'],
        'selected_candidate_id': 'synthetic-contract-fixture',
        'selected_configuration': study.default_configuration(),
        'comparison_family': study.PRIMARY_FAMILY,
        'secondary_ablation_family': SECONDARY_ABLATION_FAMILY,
        'selection_participant_ids': ['select0', 'select1', 'select2', 'select3'],
        'check_participant_ids': ['check0', 'check1', 'check2', 'check3'],
        'gates': {'fixture_only': True},
    }
    monkeypatch.setattr(selection, 'development_decision', lambda _: decision)
    monkeypatch.setattr(selection, 'load_completed', lambda _: (development, [], {}, []))
    freeze = selection.freeze_development(plan_path, root / 'actual-freeze.json')
    return root, package, registry, registry_path, descriptor, freeze


@pytest.mark.parametrize('mutation', ('code', 'registry'))
def test_actual_freeze_writer_and_formal_plan_reader_bind_identical_inputs(prepared, monkeypatch, mutation):
    root, package, registry, registry_path, descriptor, freeze = actual_freeze_contract(prepared, monkeypatch)
    study.build_plan(root, descriptor, stage='formal_exploratory', package=package, freeze=freeze)
    if mutation == 'code':
        (package / 'paper3r10/algorithm.py').write_text('# changed since actual development freeze\n')
    else:
        registry['review_test_note'] = 'changed since actual development freeze'
        common.atomic(registry_path, registry)
        descriptor = study.describe(registry_path, root)
    with pytest.raises(ValueError, match='[Cc]ode changed|registry'):
        study.build_plan(root, descriptor, stage='formal_exploratory', package=package, freeze=freeze)


@pytest.mark.parametrize('mutation', ('code', 'registry'))
def test_runtime_rechecks_freeze_identity_after_formal_plan_is_resigned(prepared, monkeypatch, mutation):
    root, package, registry, registry_path, descriptor, freeze = actual_freeze_contract(prepared, monkeypatch)
    formal = study.build_plan(root, descriptor, stage='formal_exploratory', package=package, freeze=freeze)
    if mutation == 'code':
        (package / 'paper3r10/algorithm.py').write_text('# changed after formal planning\n')
        formal['code_manifest'] = study.production_manifest(package)
    else:
        registry['review_test_note'] = 'changed after formal planning'
        common.atomic(registry_path, registry)
        formal['study_registry'] = study.describe(registry_path, root)
        formal['study_registry_sha256'] = formal['study_registry']['sha256']
    path = root / 'resigned-formal-plan.json'
    common.atomic(path, common.sign_plan(formal))
    with pytest.raises(ValueError, match='[Cc]ode|registry|freeze|frozen'):
        common.load_plan(path)


def test_software_fixture_freeze_cannot_authorize_a_real_formal_stage(prepared, monkeypatch):
    root, package, _, _, descriptor, freeze = actual_freeze_contract(
        prepared, monkeypatch, scientific_scope='SOFTWARE_TEST_ONLY')
    with pytest.raises(ValueError, match='scope|[Ss]ynthetic|[Ss]oftware|REAL_DATA|formal work is stopped'):
        plan = study.build_plan(root, descriptor, stage='formal_exploratory', package=package, freeze=freeze)
        path = study.write_plan(plan, root / 'unsafe-formal.json')
        common.load_plan(path)


def test_development_method_config_override_cannot_freeze_a_different_config(tmp_path):
    path, plan = plan_fixture(tmp_path)
    plan['tasks'][0]['method_config'] = {'method': 'different-scientific-configuration'}
    common.atomic(path, common.sign_plan(plan))
    with pytest.raises(ValueError, match='method_config|configuration'):
        common.load_plan(path)


@pytest.mark.parametrize('stale_keys', ([], ['different-task-same-count']))
def test_same_recipe_report_with_wrong_completed_task_coverage_stays_partial(tmp_path, stale_keys):
    path, plan = plan_fixture(tmp_path)
    runtime.prepare(path)
    complete(plan, plan['tasks'][0])
    reports = Path(plan['output']) / 'reports'
    report = {
        'status': 'COMPLETE', 'recipe_sha256': plan['recipe_sha256'],
        'coverage': {'total_tasks': 1, 'sealed_complete_tasks': len(stale_keys),
                     'complete_task_keys': stale_keys,
                     'complete_task_keys_sha256': common.digest(stale_keys)},
    }
    common.atomic(reports / 'report.json', report)
    common.atomic(reports / 'report-status.json', {
        'status': 'COMPLETE', 'recipe_sha256': plan['recipe_sha256'],
        'report_sha256': common.sha(reports / 'report.json'),
        'total_tasks': 1, 'sealed_complete_tasks': len(stale_keys),
        'complete_task_keys_sha256': common.digest(stale_keys),
    })
    _, _, metadata = uploads.paper_collection(path, tmp_path / 'paper-work')
    assert metadata['collection_status'] == 'PARTIAL'
    assert metadata['required_missing']


def test_late_mechanism_upload_preserves_formal_pointer_and_all_channel_evidence():
    formal = {'stage': 'formal_exploratory', 'recipe_sha256': 'f' * 64}
    development = {'stage': 'mechanism', 'recipe_sha256': 'a' * 64}
    first = uploads.merge_stage_index({}, formal, 'paper', {'manifest': 'formal-paper.json'})
    with_full = uploads.merge_stage_index(first, formal, 'full', {'manifest': 'formal-full.json'})
    delayed = uploads.merge_stage_index(with_full, development, 'paper', {'manifest': 'old-mechanism.json'})
    assert delayed['active_stage'] == 'formal_exploratory'
    assert delayed['active_recipe_sha256'] == formal['recipe_sha256']
    assert delayed['stages']['formal_exploratory']['f' * 64]['channels'] == {
        'paper': {'manifest': 'formal-paper.json'}, 'full': {'manifest': 'formal-full.json'}}
    assert delayed['stages']['mechanism']['a' * 64]['channels']['paper']['manifest'] == 'old-mechanism.json'
    assert uploads.stage_prefix(formal) != uploads.stage_prefix(development)


def test_stage_index_cas_conflict_reloads_and_preserves_concurrent_formal_publication():
    formal = {'stage': 'formal_exploratory', 'recipe_sha256': 'f' * 64}
    development = {'stage': 'mechanism', 'recipe_sha256': 'a' * 64}
    index_path = uploads.PREFIX + '/INDEX.json'

    class LocalGitHub:
        prefix = '/repos/local-fixture'
        branch = 'paper3-results'

        def __init__(self):
            self.head, self.remote = 'head0', {}
            self.blobs, self.trees, self.commits = {}, {}, {}
            self.read_refs, self.delays, self.patches = [], [], []

        def ensure_branch(self):
            pass

        def _delay(self, attempt):
            self.delays.append(attempt)

        def _blob(self, data):
            identity = hashlib.sha256(data).hexdigest()
            self.blobs[identity] = data
            return identity

        def api(self, path, data=None, method='GET'):
            if '/git/ref/heads/' in path:
                return {'object': {'sha': self.head}}
            if '/contents/' in path:
                self.read_refs.append(path.split('?ref=')[1])
                return {'content': base64.b64encode(common.encoded(self.remote)).decode()}
            if '/git/trees' in path:
                identity = 'tree' + str(len(self.trees))
                self.trees[identity] = data['tree']
                return {'sha': identity}
            if method == 'POST' and '/git/commits' in path:
                identity = 'commit' + str(len(self.commits))
                self.commits[identity] = data
                return {'sha': identity}
            if method == 'GET' and '/git/commits/' in path:
                return {'tree': {'sha': 'base-tree-' + self.head}}
            assert method == 'PATCH' and '/git/refs/heads/' in path
            self.patches.append(data)
            if len(self.patches) == 1:
                self.head = 'head1'
                self.remote = uploads.merge_stage_index({}, formal, 'full', {'manifest': 'concurrent-formal.json'})
                raise uploads.ApiError(409)
            commit = self.commits[data['sha']]
            assert commit['parents'] == ['head1']
            entry = next(item for item in self.trees[commit['tree']] if item['path'] == index_path)
            self.remote = json.loads(self.blobs[entry['sha']])
            self.head = data['sha']
            return {}

    github = LocalGitHub()
    uploads.publish_stage_index(github, development, 'paper', {'manifest': 'late-mechanism.json'})
    assert github.read_refs == ['head0', 'head1'] and github.delays == [1]
    assert all(patch['force'] is False for patch in github.patches)
    assert github.remote['active_stage'] == 'formal_exploratory'
    assert github.remote['stages']['formal_exploratory']['f' * 64]['channels']['full']['manifest'] == 'concurrent-formal.json'
    assert github.remote['stages']['mechanism']['a' * 64]['channels']['paper']['manifest'] == 'late-mechanism.json'
