"""Exercise the standalone bootstrap with fake Git/Python; no real scheduling."""
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'hpc/START.sh'


@pytest.fixture
def fixture(tmp_path):
    project = tmp_path / 'project'
    (project / 'hpc').mkdir(parents=True)
    (project / 'hpc/start.py').write_text('# only a software double\n')
    binary = tmp_path / 'bin'
    binary.mkdir()
    git = binary / 'git'
    git.write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$CALLS"
case "$*" in
 'rev-parse --show-toplevel') pwd -P ;;
 *'fetch --no-tags'*)
  count=0
  if [ -f "$COUNT" ]; then read -r count < "$COUNT"; fi
  count=$((count+1)); echo "$count" > "$COUNT"
  [ "$count" -gt "${FAIL_FETCHES:-0}" ] ;;
 'show-ref --verify --quiet'*) [ "${BRANCH_EXISTS:-yes}" = yes ] ;;
 checkout*) exit "${CHECKOUT_EXIT:-0}" ;;
 merge*) exit 0 ;;
 *) exit 2 ;;
esac
''')
    python = binary / 'python'
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" > "$STARTED"\n')
    git.chmod(0o700); python.chmod(0o700)
    env = dict(os.environ, PATH=str(binary) + ':' + os.environ.get('PATH', ''),
               CALLS=str(tmp_path / 'calls'), COUNT=str(tmp_path / 'count'),
               STARTED=str(tmp_path / 'started'), PAPER3_HPC_GIT_RETRY_DELAY='0')
    def run(*args, **changes):
        return subprocess.run(['bash', str(SCRIPT), '--root', str(project),
            '--python', str(python), *args], env=dict(env, **changes),
            capture_output=True, text=True, timeout=5)
    return tmp_path, run


@pytest.mark.parametrize('args', [(), ('--execute', '--dry-run', '--network-host')])
def test_plan_never_fetches_or_starts_program(fixture, args):
    root, run = fixture
    assert run(*args).returncode == 0
    assert not (root / 'count').exists() and not (root / 'started').exists()


def test_existing_branch_and_transient_empty_reply_retry_are_safe(fixture):
    root, run = fixture
    result = run('--execute', '--network-host', FAIL_FETCHES='2')
    assert result.returncode == 0, result.stderr
    assert (root / 'count').read_text().strip() == '3'
    calls = (root / 'calls').read_text()
    assert 'refs/heads/research/paper3-slurm-hpc-20261021:refs/remotes/origin/' in calls
    assert 'checkout research/paper3-slurm-hpc-20261021' in calls
    assert 'checkout --no-track -b' not in calls and 'reset' not in calls
    assert '--execute --network-host' in (root / 'started').read_text()


def test_missing_branch_created_once_with_tracking(fixture):
    root, run = fixture
    assert run('--execute', '--network-host', BRANCH_EXISTS='no').returncode == 0
    assert 'checkout --no-track -b research/paper3-slurm-hpc-20261021 refs/remotes/origin/' in (root / 'calls').read_text()


def test_staged_mode_updates_code_and_forwards_without_requiring_r2_network_flag(fixture):
    root, run = fixture
    assert run('--execute', '--staged-data').returncode == 0
    forwarded = (root / 'started').read_text()
    assert '--staged-data' in forwarded and '--network-host' not in forwarded


@pytest.mark.parametrize('changes', [{'FAIL_FETCHES': '99'}, {'CHECKOUT_EXIT': '1'}])
def test_git_failure_does_not_continue_to_data_or_sbatch(fixture, changes):
    root, run = fixture
    assert run('--execute', '--network-host', **changes).returncode != 0
    assert not (root / 'started').exists()
