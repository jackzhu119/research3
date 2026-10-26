"""Independent engineering review regressions; every scheduler RPC is mocked."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper3r11.platform import common
from paper3r11.platform import runtime, uploads
from paper3r11.platform.scheduler import ThreeJobScheduler
from test_platform_contract import complete, plan_fixture


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


@pytest.mark.parametrize('field', ('source_sha256','registry_sha256','configuration_sha256'))
def test_signed_campaign_identity_cannot_change_without_new_recipe(tmp_path,field):
    path,plan=plan_fixture(tmp_path);plan['stage']='campaign';plan=common.sign_plan(plan)
    common.atomic(path,plan)
    plan['tasks'][0].setdefault('input_identities',{})[field]='changed-without-resigning'
    common.atomic(path,plan)
    with pytest.raises(ValueError,match='new recipe'):common.load_plan(path)


def test_valid_signed_campaign_needs_no_foreign_runtime_profile_or_freeze(tmp_path):
    path,plan=plan_fixture(tmp_path);plan['stage']='campaign'
    plan['historical_runtime_metadata']={'GPU_UUID':'different-legal-device','capacity':'40G_vs80G'}
    common.atomic(path,common.sign_plan(plan))
    _,loaded=common.load_plan(path)
    assert 'freeze' not in loaded and loaded['stage']=='campaign'


@pytest.mark.parametrize('family,role', [('operator_train','meta_validation'),('selection','meta_train')])
def test_campaign_train_and_selection_roles_preserve_no_leak_boundary(tmp_path,family,role):
    path,plan=plan_fixture(tmp_path);plan['stage']='campaign'
    plan['tasks'][0].update(family=family,role=role)
    common.atomic(path,common.sign_plan(plan))
    with pytest.raises(ValueError,match='participants'):common.load_plan(path)


def test_confirmatory_plan_allows_development_selection_but_not_target_selection(tmp_path):
    path,plan=plan_fixture(tmp_path);plan.update(stage='confirmatory',synthetic_fixture=False)
    plan['tasks'][0].update(family='selection',role='meta_validation',phase='development')
    common.atomic(path,common.sign_plan(plan));common.load_plan(path)
    plan['tasks'][0]['phase']='confirmatory';common.atomic(path,common.sign_plan(plan))
    with pytest.raises(ValueError,match='independent'):common.load_plan(path)


def test_monitoring_metadata_read_does_not_repeat_code_file_hashes(tmp_path,monkeypatch):
    path,_=plan_fixture(tmp_path)
    monkeypatch.setattr(common,'sha',lambda *_:(_ for _ in ()).throw(AssertionError('No monitor code hash scan')))
    common.load_plan(path,validate_science_stage=False)


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


def test_late_development_upload_preserves_formal_pointer_and_all_channel_evidence():
    formal = {'stage': 'exploratory', 'recipe_sha256': 'f' * 64}
    development = {'stage': 'development', 'recipe_sha256': 'a' * 64}
    first = uploads.merge_stage_index({}, formal, 'paper', {'manifest': 'formal-paper.json'})
    with_full = uploads.merge_stage_index(first, formal, 'full', {'manifest': 'formal-full.json'})
    delayed = uploads.merge_stage_index(with_full, development, 'paper', {'manifest': 'old-development.json'})
    assert delayed['active_stage'] == 'exploratory'
    assert delayed['active_recipe_sha256'] == formal['recipe_sha256']
    assert delayed['stages']['exploratory']['f' * 64]['channels'] == {
        'paper': {'manifest': 'formal-paper.json'}, 'full': {'manifest': 'formal-full.json'}}
    assert delayed['stages']['development']['a' * 64]['channels']['paper']['manifest'] == 'old-development.json'
    assert uploads.stage_prefix(formal) != uploads.stage_prefix(development)


def test_stage_index_cas_conflict_reloads_and_preserves_concurrent_formal_publication():
    formal = {'stage': 'exploratory', 'recipe_sha256': 'f' * 64}
    development = {'stage': 'development', 'recipe_sha256': 'a' * 64}
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
    uploads.publish_stage_index(github, development, 'paper', {'manifest': 'late-development.json'})
    assert github.read_refs == ['head0', 'head1'] and github.delays == [1]
    assert all(patch['force'] is False for patch in github.patches)
    assert github.remote['active_stage'] == 'exploratory'
    assert github.remote['stages']['exploratory']['f' * 64]['channels']['full']['manifest'] == 'concurrent-formal.json'
    assert github.remote['stages']['development']['a' * 64]['channels']['paper']['manifest'] == 'late-development.json'
