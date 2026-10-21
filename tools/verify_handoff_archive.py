#!/usr/bin/env python3
"""Verify ZIP digest and directory only; never read member payloads or execute code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def inspect(path, expected):
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "library_id": expected["library_id"],
        "status": "failed",
        "archive_integrity_verified": False,
        "manifest_verified": False,
        "manifest_files_verified": 0,
        "protocol_read": False,
        "extracted": False,
        "zip_member_payloads_read": False,
    }
    if not path.is_file():
        report["reason"] = "archive_not_available"
        return report
    actual_bytes = path.stat().st_size
    report["actual_bytes"] = actual_bytes
    if actual_bytes != expected["expected_bytes"]:
        report["reason"] = "size_mismatch"
        return report
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    report["actual_sha256"] = digest.hexdigest()
    if report["actual_sha256"] != expected["expected_sha256"]:
        report["reason"] = "sha256_mismatch"
        return report
    try:
        with zipfile.ZipFile(path) as archive:
            names = set()
            for entry in archive.infolist():
                member = PurePosixPath(entry.filename)
                if (member.is_absolute() or ".." in member.parts or
                        "\\" in entry.filename or entry.filename in names or
                        (member.parts and ":" in member.parts[0]) or
                        stat.S_ISLNK(entry.external_attr >> 16)):
                    report["reason"] = "unsafe_or_duplicate_archive_member"
                    return report
                names.add(entry.filename)
            report["archive_member_count"] = len(names)
    except zipfile.BadZipFile:
        report["reason"] = "invalid_zip"
        return report
    report.update(status="archive_verified_only", archive_integrity_verified=True,
                  reason="manifest_and_full_protocol_still_require_import_and_review")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected", type=Path, default=Path(__file__).resolve().parents[1] / "provenance/handoff.expected.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.archive, json.loads(args.expected.read_text()))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["archive_integrity_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
