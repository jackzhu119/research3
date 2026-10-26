"""Read existing upload credentials without copying them into result bundles."""
import json
import os
from pathlib import Path


def upload_discover_token(explicit=None):
    for name in ('PAPER3_GITHUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
        token = os.environ.get(name, '').strip()
        if token:
            return token
    candidates = []
    for value in (explicit, os.environ.get('PAPER3_GITHUB_CREDENTIALS')):
        if value:
            candidates.append(Path(value).expanduser())
    candidates.append(Path.home()/'.config/paper3/github.json')
    for name in ('paper3-followup', 'paper3-round3', 'paper3-round4', 'paper3-round5',
                 'paper3-round6', 'paper3-round7', 'paper3-round8', 'paper3-round9',
                 'paper3-round9-thin', 'paper3-round4-hpcgpu', 'paper3-round5-hpcgpu',
                 'paper3-round6-hpcgpu'):
        base = Path.home()/name
        candidates.append(base/'private/github.json')
        candidates.extend(sorted(base.glob('*/private/github.json')))
    for path in candidates:
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        token = value.get('token') if isinstance(value, dict) else None
        if isinstance(token, str) and token.strip():
            return token.strip()
    raise RuntimeError('EXISTING_GITHUB_CREDENTIAL_NOT_FOUND')
