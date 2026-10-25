import os,json
from pathlib import Path

def upload_read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default

def upload_discover_token(explicit=None):
    for name in ('PAPER3_GITHUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
        token = os.environ.get(name, '').strip()
        if token:
            return token
    candidates = []
    for candidate in (explicit, os.environ.get('PAPER3_GITHUB_CREDENTIALS')):
        if candidate:
            candidates.append(Path(candidate).expanduser())
    candidates.append(Path.home()/'.config/paper3/github.json')
    for name in ('paper3-followup', 'paper3-round3', 'paper3-round4', 'paper3-round5',
                 'paper3-round6', 'paper3-round7', 'paper3-round4-hpcgpu',
                 'paper3-round5-hpcgpu', 'paper3-round6-hpcgpu'):
        base = Path.home()/name
        candidates.append(base/'private/github.json')
        candidates.extend(sorted(base.glob('*/private/github.json')))
    for path in candidates:
        value = upload_read(path, {})
        token = value.get('token') if isinstance(value, dict) else None
        if isinstance(token, str) and token.strip():
            return token.strip()
    raise RuntimeError('未找到现有GitHub凭证；可用 --credentials 指定原有 private/github.json。凭证不进入归档。')
