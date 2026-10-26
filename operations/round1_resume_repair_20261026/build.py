"""Build one offline stdlib installer from independently audited helper bytes."""
import base64
import hashlib
import json
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parent
OLD = Path('/workspace/paper3-r1-hardware-fix')
BOOT = Path('/workspace/paper3-future-audit-20261025/r1/r1_spawn_bootstrap.py')
DIGEST = Path('/workspace/paper3-r1-startup-cache-repair/r1_digest_cache.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_payload():
    helpers = {name: (ROOT / name).read_text() for name in
               ('r1_clock_recovery.py', 'r1_driver_compat.py')}
    helpers['node_refresh.py'] = (OLD / 'node_refresh.py').read_text()
    for name, source in helpers.items():
        compile(source, name, 'exec')
    return {'schema_version': 'paper3-r1-resume-delivery-payload-v1',
            'helpers': helpers,
            'lineage_sha256': {
                'action_startup.py': sha((OLD / 'action_startup.py').read_bytes()),
                'audited_runtime.py': sha((OLD / 'audited_runtime.py').read_bytes()),
                'r1_spawn_bootstrap.py': sha(BOOT.read_bytes())},
            'existing_digest_helper_sha256': sha(DIGEST.read_bytes()),
            'startup_install': ['r1_driver_compat.py', 'r1_clock_recovery.py'],
            'bootstrap_install': ['r1_driver_compat.py', 'r1_clock_recovery.py'],
            'runtime_install': ['r1_driver_compat.py'],
            'observed_driver_pair': ['535.104.12', '550.54.14'],
            'scope': 'Operational cross-node clock and explicitly observed driver compatibility only; '
                     'native source, models, splits, seeds and scientific limits remain unchanged.'}


def main():
    payload = make_payload()
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()
    source = (ROOT / 'delivery_template.py').read_text()
    source = source.replace('__PAYLOAD_B85__', base64.b85encode(zlib.compress(raw, 9)).decode())
    source = source.replace('__PAYLOAD_SHA256__', sha(raw))
    encoded = source.encode()
    compile(encoded, 'FIX-PAPER3-R1-RESUME.py', 'exec')
    (ROOT / 'FIX-PAPER3-R1-RESUME.py').write_bytes(encoded)
    record = {'schema_version': 'paper3-r1-resume-installer-package-v3',
              'entry': 'FIX-PAPER3-R1-RESUME.py', 'bytes': len(encoded),
              'sha256': sha(encoded), 'payload_sha256': sha(raw),
              'helpers': {name: sha(value.encode()) for name, value in payload['helpers'].items()},
              'known_operational_source_lineage': payload['lineage_sha256'],
              'default_changes_jobs': False, 'optional_flag': '--refresh-pretraining',
              'school_jobs_changed_by_assistant': 0}
    (ROOT / 'PACKAGE.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
