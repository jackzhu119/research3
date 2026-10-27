"""Build a frozen stdlib-only R11 management recovery package."""
from pathlib import Path
import base64
import hashlib
import io
import json
import stat
import zipfile

ROOT = Path(__file__).resolve().parent
REQUIRED = ('entry.py', 'supervisor.py', 'processes.py', 'environment.py',
            'controller.py', 'state_io.py', 'upload_fix.py', 'report_fix.py',
            'legacy_controller.py', 'atomic_io.py', 'resume_io.py', 'paper_fix.py',
            'build_delivery.py', 'installer_template.py')
FORBIDDEN = {'private', '.git', '.ssh', 'secrets', 'credentials', '__pycache__'}
BAD_NAMES = {'github.json', 'credentials.json', '.env', 'manifest.json'}
SCHEMA = 'paper3-r11-management-recovery-package-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(root=ROOT):
    root = Path(root).resolve()
    for name in REQUIRED:
        if not (root/name).is_file() or (root/name).is_symlink():
            raise ValueError('MANAGEMENT_REPAIR_SOURCE_MISSING: ' + name)
    paths = [p for p in root.iterdir() if p.is_file() and not p.is_symlink()
             and p.suffix.lower() in {'.py', '.md', '.json'}
             and p.name.lower() not in BAD_NAMES]
    for folder in ('tests', 'proof'):
        for path in (root/folder).rglob('*'):
            if (path.is_file() and not path.is_symlink()
                    and path.suffix.lower() in {'.py', '.md', '.json'}
                    and not any(part.lower() in FORBIDDEN for part in path.relative_to(root).parts)
                    and path.name.lower() not in BAD_NAMES):
                paths.append(path)
    values = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > 4*1024*1024:
            raise ValueError('MANAGEMENT_REPAIR_SOURCE_TOO_LARGE: ' + relative)
        data = path.read_bytes()
        if path.suffix == '.py':
            compile(data, relative, 'exec')
        values[relative] = data
    if sum(map(len, values.values())) > 16*1024*1024:
        raise ValueError('MANAGEMENT_REPAIR_CODE_SIZE_LIMIT')
    manifest = dict(schema_version=SCHEMA,
        scope='Operational code/tests/proof only; no EEG, original scientific package, plans, results, checkpoints or credentials',
        entry='entry.py', files={name:dict(bytes=len(data), sha256=sha(data))
                                 for name, data in values.items()})
    metadata = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    values['MANIFEST.json'] = metadata
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(values.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    entry = (root/'installer_template.py').read_text().replace(
        '__PAYLOAD_B85__', base64.b85encode(payload).decode()).replace(
        '__PAYLOAD_SHA256__', sha(payload)).replace('__MANIFEST_SHA256__', sha(metadata)).encode()
    compile(entry, 'REPAIR-PAPER3-R11-MANAGEMENT.py', 'exec')
    delivery = root/'delivery'; delivery.mkdir(exist_ok=True)
    (delivery/'REPAIR-PAPER3-R11-MANAGEMENT.py').write_bytes(entry)
    (delivery/'paper3-r11-management-recovery.zip').write_bytes(payload)
    (delivery/'MANIFEST.json').write_bytes(metadata)
    record = dict(schema_version='paper3-r11-management-recovery-delivery-v1',
        entry='REPAIR-PAPER3-R11-MANAGEMENT.py', entry_sha256=sha(entry), entry_bytes=len(entry),
        payload_sha256=sha(payload), payload_bytes=len(payload), manifest_sha256=sha(metadata),
        files=len(manifest['files']), default='install_only_no_scientific_action',
        activation='--apply --submit', actual_school_operations=0,
        immutable_original_science=True, existing_four_slots_preserved=True)
    (delivery/'PACKAGE.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
