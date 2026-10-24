#!/usr/bin/env python3
"""Build the deterministic, public, source-only Round 5 single-file installer.

Run after source/tests/validation records are final. The generated PACKAGE.json
pins the exact downloadable wrapper hash; publication URLs can be added in a
separate PUBLICATION.json without rebuilding or embedding any credentials.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import textwrap
import zipfile

PACKAGE = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "START.py", "train.sbatch", "paper3r5/__init__.py", "paper3r5/entry.py",
    "paper3r5/submit.py", "paper3r5/runtime.py", "paper3r5/monitor.py",
    "paper3r5/plan.py", "paper3r5/worker.py", "paper3r5/algorithm.py", "paper3r5/core.py",
    "paper3r5/allocation.py",
    "paper3r5/baseline.py", "paper3r5/data.py", "paper3r5/diagnostics.py",
    "paper3r5/external.py", "paper3r5/report.py", "paper3r5/io.py",
    "p3work/__init__.py", "p3work/adapt.py", "p3work/contracts.py",
    "p3work/f0_bridge.py", "p3work/spatial_geometry.py", "p3work/splits.py",
    "p3work/window_amendment.py",
)
SECRET_NAME = re.compile(r"private|secret|credential|password|api[-_]?key|access[-_]?key|(?:^|[-_.])token(?:[-_.]|$)|^github\.json$|^online_key$", re.I)
SECRET_VALUE = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|AKIA[A-Z0-9]{16})")


INSTALLER = r'''#!/usr/bin/env python3
"""Pinned-source Round 5 installer. Stdlib only; Python 3.12/Linux 3.10 compatible."""
import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

PAYLOAD_SHA256 = __PAYLOAD_SHA256__
FILE_HASHES = __FILE_HASHES__
REQUIRED_FILES = __REQUIRED_FILES__
PAYLOAD_B85 = (
__PAYLOAD_B85__
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_manifest():
    return {"format_version": 1, "payload_sha256": PAYLOAD_SHA256, "files": FILE_HASHES}


def verify_install(destination):
    receipt = destination / "INSTALL_MANIFEST.json"
    if not destination.is_dir() or destination.is_symlink() or receipt.is_symlink():
        raise RuntimeError("Existing installation is not a regular package directory")
    try:
        manifest = json.loads(receipt.read_text())
    except (OSError, ValueError):
        raise RuntimeError("Existing installation has no valid install manifest") from None
    if manifest != _expected_manifest():
        raise RuntimeError("Existing installation manifest does not match this pinned payload")
    for name, expected in FILE_HASHES.items():
        path = destination / name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if destination in parent.parents):
            raise RuntimeError("Installed file has an unexpected symlink: " + name)
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError("Installed file hash mismatch; refusing to overwrite: " + name)
    return destination


def install(base=None):
    base = (Path(base).expanduser() if base is not None else Path.home() / "paper3-round5").resolve()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = base / PAYLOAD_SHA256[:20]
    # flock plus ordinary same-filesystem rename works on the school kernel.
    # No renameat2/RENAME_NOREPLACE syscall or recent Linux primitive is used.
    with (base / (".install-" + PAYLOAD_SHA256[:20] + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            return verify_install(destination)
        packed = base64.b85decode(PAYLOAD_B85)
        if hashlib.sha256(packed).hexdigest() != PAYLOAD_SHA256:
            raise RuntimeError("Embedded payload hash mismatch")
        stage = Path(tempfile.mkdtemp(prefix="." + PAYLOAD_SHA256[:20] + ".tmp-", dir=str(base)))
        try:
            with zipfile.ZipFile(io.BytesIO(packed)) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(FILE_HASHES):
                    raise RuntimeError("Payload member list differs from its pinned file manifest")
                if any(name not in FILE_HASHES for name in REQUIRED_FILES):
                    raise RuntimeError("Pinned payload is missing required numerical/runtime source")
                if sum(info.file_size for info in archive.infolist()) > 64 * 1024 * 1024:
                    raise RuntimeError("Unexpectedly large source payload")
                for name in names:
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                        raise RuntimeError("Unsafe payload member path")
                    content = archive.read(name)
                    if hashlib.sha256(content).hexdigest() != FILE_HASHES[name]:
                        raise RuntimeError("Embedded source hash mismatch: " + name)
                    path = stage / name
                    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    with path.open("xb") as handle:
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    path.chmod(0o700 if name == "train.sbatch" else 0o600)
            receipt = stage / "INSTALL_MANIFEST.json"
            with receipt.open("x") as handle:
                json.dump(_expected_manifest(), handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Every cooperative installer holds the same payload-specific lock.
            # An unexpected existing destination is verified, never removed.
            if destination.exists() or destination.is_symlink():
                return verify_install(destination)
            os.rename(str(stage), str(destination))
            directory = os.open(str(base), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return verify_install(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Other arguments are forwarded to the installed START.py, including --root, --python, --gpus, --cpus, --credentials and --submit.")
    parser.add_argument("--install-base", help="Override ~/paper3-round5 (useful for isolated local checks)")
    parser.add_argument("--install-only", action="store_true", help="Verify/unpack the package without preparing or submitting")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the plan without submitting jobs")
    args, forwarded = parser.parse_known_args(argv)
    if args.prepare_only and "--submit" in forwarded:
        parser.error("--prepare-only cannot be combined with --submit")
    destination = install(args.install_base)
    print(json.dumps({"status": "PACKAGE_READY", "package": str(destination), "payload_sha256": PAYLOAD_SHA256}, sort_keys=True), flush=True)
    if args.install_only:
        return 0
    command = [sys.executable, "-I", "-S", "-B", "-u", str(destination / "START.py"), *forwarded]
    return subprocess.call(command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print("INSTALLATION_ERROR: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
'''


def _selected(relative):
    parts = relative.parts
    if any(part.startswith(".") or part == "__pycache__" or SECRET_NAME.search(part) for part in parts):
        return False
    if len(parts) == 1:
        # Raw provider/download inventories are available in the working archive;
        # retain concise public provenance/validation records in the delivery.
        if relative.name.startswith("eye_bci_") or relative.name.endswith("_datacite_metadata.json") or relative.name == "weibo_nemar_v103_manifest.json":
            return False
        return relative.name == "train.sbatch" or relative.suffix in (".py", ".md", ".json") or (relative.name.startswith("requirements") and relative.suffix == ".txt")
    if parts[0] in ("paper3r5", "p3work", "tests"):
        return relative.suffix == ".py"
    if parts[0] == "compatibility":
        return relative.suffix in (".json", ".txt", ".md")
    return False


def collect_sources(package, excluded=()):
    package = Path(package).resolve()
    excluded = {Path(path).resolve() for path in excluded}
    files = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if not _selected(relative) or path.resolve() in excluded:
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != package.parent):
            raise ValueError("Selected source must not be a symlink: " + relative.as_posix())
        if not path.is_file():
            continue
        content = path.read_bytes()
        if SECRET_VALUE.search(content):
            raise ValueError("Credential-shaped content in selected source: " + relative.as_posix())
        files[relative.as_posix()] = content
    missing = sorted(set(REQUIRED_FILES) - set(files))
    if missing:
        raise ValueError("Required source files missing: " + ", ".join(missing))
    if sum(map(len, files.values())) > 64 * 1024 * 1024:
        raise ValueError("Source-only payload unexpectedly exceeds 64 MiB")
    return files


def build(package=PACKAGE, output=None, manifest=None):
    package = Path(package).resolve()
    output = Path(output).resolve() if output is not None else package / "online/START-PAPER3-R5-ONLINE.py"
    manifest = Path(manifest).resolve() if manifest is not None else output.parent / "PACKAGE.json"
    files = collect_sources(package, excluded=(output, manifest))
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100600 << 16)
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    payload = stream.getvalue()
    payload_sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b85encode(payload).decode("ascii")
    wrapper = INSTALLER.replace("__PAYLOAD_SHA256__", json.dumps(payload_sha))
    wrapper = wrapper.replace("__FILE_HASHES__", json.dumps(hashes, sort_keys=True, indent=2))
    wrapper = wrapper.replace("__REQUIRED_FILES__", repr(list(REQUIRED_FILES)))
    wrapper = wrapper.replace("__PAYLOAD_B85__", "\n".join("    " + repr(chunk) for chunk in textwrap.wrap(encoded, 100)))
    encoded_wrapper = wrapper.encode("utf-8")
    # Compile without importing or executing the generated installer.
    compile(wrapper, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded_wrapper)
    result = {"format_version": 1, "wrapper_name": output.name,
              "wrapper_sha256": hashlib.sha256(encoded_wrapper).hexdigest(),
              "wrapper_bytes": len(encoded_wrapper), "payload_sha256": payload_sha,
              "payload_bytes": len(payload), "source_bytes": sum(map(len, files.values())),
              "install_directory_name": payload_sha[:20], "file_count": len(files),
              "files": hashes, "required_files": list(REQUIRED_FILES),
              "built_utc": datetime.now(timezone.utc).isoformat(),
              "contains_credentials": False, "contains_original_data": False,
              "source_only": True, "default_submission": False}
    manifest.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(PACKAGE))
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    result = build(args.package, args.output, args.manifest)
    print(json.dumps({key: value for key, value in result.items() if key not in ("files", "required_files")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
