#!/usr/bin/env python3
"""Attach or launch the existing frozen Paper3 bundle; never manage a Pod."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

ZIP_NAME = 'Paper3_OneClick_20261021.zip'
ZIP_SHA = '5bb056d39254854707e91c4eb2ce2a4cc6e8e922a9337a52f3daea08b0fb5489'
ZIP_BYTES = 17717251
TOP = 'Paper3_OneClick'
WORK = 'work_603d299351fa6605'
RELEASE = '603d299351fa6605aa0bc258bbbe2c421b2048e5becbeb33521e119a62f20fd7'
MONITOR_SOURCE = None
FETCH_SOURCE = None
ONLINE_REPO = 'jackzhu119/research3'
ONLINE_TAG = 'paper3-online-20261021-v2'
ONLINE_ASSET_COMMIT = '4e4fa1504e01bf2d04875c9ce04d543f3248f826'


class AutoError(ValueError):
    pass


def need(test, code):
    if not test:
        raise AutoError(code)


def checked_path(path):
    path = Path(os.path.abspath(path))
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINK_PATH_REFUSED')
    return path


def run(command):
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, check=False)
    need(result.returncode == 0, 'LOCAL_TOOL_COMMAND_FAILED')
    return result.stdout.strip()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def active_jobs(workspace, proc_root=Path('/proc')):
    """Exact argv names plus cwd scope; never inspect process environments."""
    jobs = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = [os.fsdecode(a) for a in (entry / 'cmdline').read_bytes().split(b'\0') if a]
            cwd = (entry / 'cwd').resolve(strict=True)
            for arg in args[1:]:
                name = Path(arg).name
                candidate = Path(os.path.abspath(Path(arg) if Path(arg).is_absolute() else cwd / arg))
                launcher = name == 'paper3_one_click.py' and candidate.parent.name == TOP
                core = name in {'bootstrap_data_plan.py', 'run_training_campaign.py',
                                'run_source_extension_downstream.py'} and \
                       ('paper3_data_plan_603d299351fa6605' in candidate.parts)
                inside = name == 'PAPER3_AUTO_RUN.py' and '--inside' in args
                if (launcher or core or inside) and candidate.is_relative_to(workspace):
                    jobs.append({'pid': int(entry.name), 'bundle': candidate.parent if launcher else None})
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
    return jobs


def ancestor_ids(pid, proc_root=Path('/proc')):
    result = set()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            lines = (proc_root / str(pid) / 'status').read_text().splitlines()
            pid = int(next(row.split()[1] for row in lines if row.startswith('PPid:')))
        except (OSError, ValueError, StopIteration):
            break
    return result


def pane_for_jobs(jobs, panes, proc_root=Path('/proc')):
    for job in jobs:
        ancestors = ancestor_ids(job['pid'], proc_root)
        for pane in panes:
            if pane['pid'] in ancestors:
                return pane
    return None


def list_panes():
    result = subprocess.run(['tmux', 'list-panes', '-a', '-F', '#{session_id}\t#{window_id}\t#{pane_id}\t#{pane_pid}'],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    panes = []
    for line in result.stdout.splitlines():
        fields = line.split('\t')
        if len(fields) == 4 and fields[3].isdigit():
            panes.append(dict(zip(('session', 'window', 'pane', 'pid'), fields[:3] + [int(fields[3])])))
    return panes


@contextmanager
def short_manager_lock(workspace):
    path = checked_path(workspace / '.paper3-auto-manager.lock')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AutoError('ANOTHER_AUTO_ENTRY_IS_CHECKING_PLEASE_RETRY') from None
        yield


def original_locks_busy(bundle):
    root = bundle / WORK
    project = root / 'release' / 'paper3_data_plan_603d299351fa6605'
    paths = [root / 'launcher.owner.lock', project / '.paper3-deployment/owner.lock',
             project / 'runs/.training_campaign_v1.lock']
    for path in paths:
        checked_path(path)
        if not path.exists():
            continue
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        with os.fdopen(fd, 'a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    return False


def ensure_bundle(workspace):
    """Use the pinned ZIP; compare existing members, never overwrite work."""
    source = checked_path(workspace / ZIP_NAME)
    if not source.exists():
        download_bundle(workspace)
    need(source.is_file(), 'FIXED_ONLINE_BUNDLE_NOT_AVAILABLE')
    print('正在核验已有启动 ZIP（17.7 MB）……', flush=True)
    need(source.stat().st_size == ZIP_BYTES and file_sha(source) == ZIP_SHA,
         'ORIGINAL_ZIP_SHA_MISMATCH')
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        need(len(names) == len(set(names)), 'ZIP_DUPLICATE_MEMBER')
        for member in members:
            parts = member.filename.split('/')
            need(parts[0] == TOP and all(p not in {'', '.', '..'} for p in parts)
                 and '\\' not in member.filename and not member.is_dir()
                 and stat.S_IFMT(member.external_attr >> 16) in {0, stat.S_IFREG}, 'ZIP_MEMBER_REFUSED')
            target = checked_path(workspace / member.filename)
            payload = archive.read(member)
            if target.exists():
                need(target.is_file() and target.stat().st_size == len(payload)
                     and file_sha(target) == hashlib.sha256(payload).hexdigest(),
                     'EXISTING_LAUNCHER_FILE_DIFFERS_PRESERVED')
                continue
            write_new_file(target, payload)
    print('启动包已核验；已有工作目录、缓存和断点均保留。', flush=True)
    return checked_path(workspace / TOP)


def download_bundle(workspace):
    need(isinstance(FETCH_SOURCE, str), 'USE_BUILT_ONLINE_SINGLE_FILE_HELPER')
    print('正在从 GitHub 固定版本自动下载启动器与SDK（17.7 MB）……', flush=True)
    namespace = {'__name__': 'paper3_online_release_fetch'}
    exec(compile(FETCH_SOURCE, 'embedded_online_fetch.py', 'exec'), namespace)
    try:
        return namespace['fetch_pinned_file'](ONLINE_REPO, ONLINE_ASSET_COMMIT, 'launcher_assets/' + ZIP_NAME,
                ZIP_SHA, ZIP_BYTES, checked_path(workspace / ZIP_NAME))
    except Exception as exc:
        code = getattr(exc, 'code', 'GITHUB_FIXED_RELEASE_DOWNLOAD_FAILED')
        raise AutoError(code) from None


def write_new_file(target, payload):
    target = checked_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.paper3-auto-copy-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


def install_helpers(workspace):
    source = Path(__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    root = checked_path(workspace / '.paper3-auto-tools' / digest[:16])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    need(isinstance(MONITOR_SOURCE, str), 'USE_BUILT_SINGLE_FILE_HELPER')
    for name, payload in [('PAPER3_AUTO_RUN.py', source), ('paper3_status.py', MONITOR_SOURCE.encode())]:
        path = checked_path(root / name)
        if path.exists():
            need(path.is_file() and path.read_bytes() == payload, 'AUTO_HELPER_COPY_DIFFERS')
        else:
            write_new_file(path, payload)
    return root


def add_status_pane(pane, helpers, bundle):
    marker = subprocess.run(['tmux', 'show-options', '-wqv', '-t', pane['window'], '@paper3_status_pane'],
                            capture_output=True, text=True, check=False).stdout.strip()
    current = {row['pane']: row for row in list_panes()}
    command = shlex.join([sys.executable, '-B', str(helpers / 'paper3_status.py'), '--bundle', str(bundle)])
    if marker in current:
        if marker == pane['pane']:
            return
        # Upgrade only an identified read-only monitor, never the input/fit pane.
        try:
            pid = current[marker]['pid']
            args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            paths = [Path(os.fsdecode(arg)) for arg in args[1:] if arg]
            script = next((path for path in paths if path.suffix == '.py'), None)
            owned = (script is not None and script.name == 'paper3_status.py'
                     and '.paper3-auto-tools' in script.parts and b'-c' not in args and b'-m' not in args)
            current_version = subprocess.run(['tmux', 'show-options', '-wqv', '-t', pane['window'],
                                               '@paper3_status_version'], capture_output=True, text=True).stdout.strip()
            if owned and current_version != helpers.name:
                run(['tmux', 'respawn-pane', '-k', '-t', marker, command])
                run(['tmux', 'set-option', '-w', '-t', pane['window'], '@paper3_status_version', helpers.name])
        except (OSError, KeyError, ValueError):
            pass
        return
    result = subprocess.run(['tmux', 'split-window', '-d', '-v', '-l', '9', '-t', pane['pane'],
                             '-P', '-F', '#{pane_id}', command], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        run(['tmux', 'set-option', '-w', '-t', pane['window'], '@paper3_status_pane', result.stdout.strip()])
        run(['tmux', 'set-option', '-w', '-t', pane['window'], '@paper3_status_version', helpers.name])
    else:
        print('终端空间不足以分屏；已保留原任务并接入其窗口。', flush=True)


def create_job(helpers, workspace, bundle, repo):
    command = shlex.join([sys.executable, '-B', str(helpers / 'PAPER3_AUTO_RUN.py'), '--inside',
                          '--workspace', str(workspace), '--bundle', str(bundle), '--repo', repo])
    sessions = subprocess.run(['tmux', 'has-session', '-t', '=paper3'], capture_output=True).returncode
    action = ['new-window', '-d', '-t', 'paper3:', '-n', 'train'] if sessions == 0 else \
             ['new-session', '-d', '-s', 'paper3', '-n', 'train']
    text = run(['tmux', *action, '-P', '-F', '#{session_id}\t#{window_id}\t#{pane_id}\t#{pane_pid}', command])
    fields = text.split('\t')
    need(len(fields) == 4 and fields[3].isdigit(), 'TMUX_NEW_PANE_NOT_IDENTIFIED')
    pane = dict(zip(('session', 'window', 'pane', 'pid'), fields[:3] + [int(fields[3])]))
    run(['tmux', 'set-option', '-p', '-t', pane['pane'], '@paper3_launcher_bundle', str(bundle)])
    return pane


def reserved_pane(bundle):
    for pane in list_panes():
        value = subprocess.run(['tmux', 'show-options', '-pqv', '-t', pane['pane'], '@paper3_launcher_bundle'],
                               capture_output=True, text=True, check=False).stdout.strip()
        if value == str(bundle):
            return pane
    return None


def attach(pane):
    run(['tmux', 'select-window', '-t', pane['window']])
    run(['tmux', 'select-pane', '-t', pane['pane']])
    if os.environ.get('TMUX'):
        run(['tmux', 'switch-client', '-t', pane['session']])
        return 0
    os.execvp('tmux', ['tmux', 'attach-session', '-t', pane['session']])


def inside(args):
    print('论文3自动入口：准备进度见下方状态面板；上方出现凭证提示后逐项输入。', flush=True)
    print('正在沿用固定训练入口：GitHub → R2 校验 → 受控 pilot → 科学门禁 → 结果备份。', flush=True)
    result = subprocess.run(['bash', str(args.bundle / 'START.sh'), '--repo', args.repo], cwd=args.bundle)
    print('原启动入口返回码：' + str(result.returncode) + '。完成状态以真实 producer 和 R2 回执为准。', flush=True)
    try:
        input('按 Enter 关闭这个窗口；已有数据、回执和断点保留。')
    except (EOFError, KeyboardInterrupt):
        pass
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description='论文3自动接入/启动，不创建、开机或重启云实例。')
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--repo', default='jackzhu119/research3')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        need(sys.platform.startswith('linux') and sys.version_info >= (3, 10), 'LINUX_PYTHON310_REQUIRED')
        args.workspace = checked_path(args.workspace)
        need(args.workspace.is_dir(), 'WORKSPACE_DIRECTORY_MISSING')
        if args.inside:
            return inside(args)
        need(sys.stdin.isatty(), 'RUN_IN_RUNPOD_TERMINAL')
        missing_tools = [name for name in ('tmux', 'git') if shutil.which(name) is None]
        if missing_tools:
            need(os.geteuid() == 0 and shutil.which('apt-get'), 'REQUIRED_TOOLS_MISSING_NO_AUTOMATIC_INSTALLER')
            print('正在自动安装启动工具：' + ', '.join(missing_tools) + '……', flush=True)
            for command in [['apt-get', 'update'], ['apt-get', 'install', '-y', *missing_tools]]:
                subprocess.run(command, check=True, env={**os.environ, 'DEBIAN_FRONTEND': 'noninteractive'})
        with short_manager_lock(args.workspace):
            helpers = install_helpers(args.workspace)
            jobs = active_jobs(args.workspace)
            monitor_only = False
            if jobs:
                print('发现已有论文3进程；直接接入，不重复启动或重新解压。', flush=True)
                pane = pane_for_jobs(jobs, list_panes())
                bundle = next((job['bundle'] for job in jobs if job['bundle']), args.workspace / TOP)
                if pane is None:
                    print('现有任务位于 tmux 外，保持运行；这里只显示状态，请在原终端输入凭证。', flush=True)
                    # No restart or attempt to move an already-running process.
                    monitor_only = True
            else:
                bundle = checked_path(args.workspace / TOP)
                pane = reserved_pane(bundle)
                if pane is None:
                    need(not original_locks_busy(bundle), 'EXISTING_TRAINING_LOCK_BUSY_NO_DUPLICATE_STARTED')
                    # Recheck all processes under the short manager lock before starting.
                    need(not active_jobs(args.workspace), 'TASK_APPEARED_DURING_CHECK_RETRY_TO_ATTACH')
                    bundle = ensure_bundle(args.workspace)
                    pane = create_job(helpers, args.workspace, bundle, args.repo)
            if not monitor_only:
                add_status_pane(pane, helpers, bundle)
        # Do not let the long-lived tmux client inherit the short manager lock.
        if monitor_only:
            return subprocess.call([sys.executable, '-B', str(helpers / 'paper3_status.py'), '--bundle', str(bundle)])
        return attach(pane)
    except KeyboardInterrupt:
        print('已退出自动入口；未停止或重启已有训练任务。')
        return 130
    except Exception as exc:
        code = str(exc) if isinstance(exc, AutoError) else type(exc).__name__
        print('自动入口未完成：' + code + '。原任务和工作目录保留。')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
