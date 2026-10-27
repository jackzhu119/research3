"""Build an offline code-only mechanism-study package and standalone installer."""
from pathlib import Path
import base64
import hashlib
import io
import json
import stat
import zipfile

ROOT = Path(__file__).resolve().parent
FORBIDDEN = {'private', '.git', '.ssh', 'credentials', 'secrets', '__pycache__'}
CODE_SUFFIXES = {'.py', '.json', '.md'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(root=ROOT):
    root = Path(root)
    paths = []
    for name in ('build_delivery.py', 'installer_template.py'):
        if not (root / name).is_file():
            raise ValueError('DELIVERY_BUILD_SOURCE_MISSING: ' + name)
        paths.append(root / name)
    if (root / 'audit_design.py').is_file():
        paths.append(root / 'audit_design.py')
    for path in root.iterdir():
        if (path.is_file() and not path.is_symlink() and path.suffix.lower() in {'.md', '.json', '.csv'}
                and path.name.lower() not in {'github.json', 'credentials.json', 'manifest.json'}
                and path.stat().st_size <= 2 * 1024 * 1024):
            paths.append(path)
    # These are compact, public, already-analysed evidence tables, not school
    # EEG/model data. Keep their original byte representation (including CSV
    # CRLF) so the design's offline provenance audit remains independently
    # runnable from the installed package.
    evidence_manifest = root / 'evidence/MANIFEST.json'
    if evidence_manifest.is_file():
        inventory = json.loads(evidence_manifest.read_bytes())
        if not isinstance(inventory.get('files'), dict):
            raise ValueError('DESIGN_EVIDENCE_INVENTORY_INVALID')
        evidence_total = 0
        paths.append(evidence_manifest)
        for name, row in inventory['files'].items():
            relative = Path(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                    or relative.suffix.lower() not in {'.csv', '.json', '.md'}
                    or any(part.lower() in FORBIDDEN for part in relative.parts)
                    or relative.name.lower() in {'github.json', 'credentials.json', '.env'}):
                raise ValueError('UNSAFE_DESIGN_EVIDENCE_PATH')
            path = root / 'evidence' / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError('DESIGN_EVIDENCE_FILE_MISSING: ' + name)
            raw = path.read_bytes()
            if len(raw) != row.get('bytes') or sha(raw) != row.get('sha256'):
                raise ValueError('DESIGN_EVIDENCE_SHA256_DIFFERS: ' + name)
            evidence_total += len(raw)
            paths.append(path)
        if evidence_total > 2 * 1024 * 1024:
            raise ValueError('COMPACT_DESIGN_EVIDENCE_SIZE_LIMIT')
    package = root / 'paper3m12'
    if not (package / 'entry.py').is_file():
        raise ValueError('MECHANISM_DISPATCH_ENTRY_MISSING')
    for folder in ('paper3m12', 'tests', 'proof'):
        for path in (root / folder).rglob('*'):
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in CODE_SUFFIXES:
                continue
            relative = path.relative_to(root)
            if (any(part.lower() in FORBIDDEN for part in relative.parts)
                    or path.name.lower() in {'github.json', 'credentials.json', '.env'}):
                continue
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError('PACKAGE_CODE_FILE_TOO_LARGE: ' + relative.as_posix())
            paths.append(path)
    values = {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(set(paths))}
    for name, data in values.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    if sum(map(len, values.values())) > 64 * 1024 * 1024:
        raise ValueError('PACKAGE_CODE_SIZE_LIMIT')
    manifest = dict(schema_version='paper3-mechanism12-package-v1',
                    scope='Frozen code, tests, protocol and compact public evidence; no EEG, credentials, model or optimizer checkpoints',
                    entry='paper3m12/entry.py',
                    files={name: dict(bytes=len(data), sha256=sha(data)) for name, data in values.items()})
    manifest_data = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    values['MANIFEST.json'] = manifest_data
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(values.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = memory.getvalue()
    entry = (root / 'installer_template.py').read_text().replace(
        '__PAYLOAD_B85__', base64.b85encode(payload).decode()).replace(
        '__PAYLOAD_SHA256__', sha(payload)).replace('__MANIFEST_SHA256__', sha(manifest_data)).encode()
    compile(entry, 'START-PAPER3-M12.py', 'exec')
    delivery = root / 'delivery'
    delivery.mkdir(exist_ok=True)
    (delivery / 'START-PAPER3-M12.py').write_bytes(entry)
    (delivery / 'paper3-mechanism12.zip').write_bytes(payload)
    (delivery / 'MANIFEST.json').write_bytes(manifest_data)
    record = dict(schema_version='paper3-mechanism12-delivery-v1', entry='START-PAPER3-M12.py',
                  entry_sha256=sha(entry), entry_bytes=len(entry),
                  payload_sha256=sha(payload), payload_bytes=len(payload),
                  manifest_sha256=sha(manifest_data), files=len(manifest['files']),
                  default='install_only_no_gpu_or_slurm', activation='--prepare --submit',
                  resources=dict(workers=4, gpus_per_worker=1, cpus_per_worker=16, memory_per_worker='64G'),
                  actual_school_operations=0)
    (delivery / 'PACKAGE.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
