#!/usr/bin/env python3
"""Explicit CPU/login preparation of optional public EEG; workers never fetch data.

Supports ``python -I -S prepare_weibo.py`` without executing site .pth files.
"""
import argparse
from pathlib import Path
import sys
import sysconfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Existing Paper3 project containing contracts/input_contract.json")
    parser.add_argument("--output", required=True, type=Path, help="New Round3 data directory; never an old cache")
    parser.add_argument("--raw", type=Path, help="Local original-MAT directory (default OUTPUT_PARENT/raw)")
    parser.add_argument("--subjects", nargs="+", default=[str(i) for i in range(1, 11)], help="Subject numbers, space or comma separated")
    parser.add_argument("--source-manifest", type=Path, default=Path(__file__).resolve().parent / "WEIBO_DOWNLOAD_MANIFEST.json")
    parser.add_argument("--download", action="store_true", help="Explicitly fetch missing pinned public MAT files during preparation")
    parser.add_argument("--delete-raw-after-prepare", action="store_true",
                        help="Remove only newly downloaded verified raw files after their cache is committed; preserve preexisting raw files")
    args = parser.parse_args(argv)
    try:
        subjects = [int(s) for group in args.subjects for s in group.split(",")]
    except ValueError:
        parser.error("Subjects must be numbers 1..10")
    if not subjects or any(s not in range(1, 11) for s in subjects):
        parser.error("Subjects must be numbers 1..10")
    package = Path(__file__).resolve().parent
    sys.path.insert(0, str(package))
    sys.path.append(str(args.root.resolve()))
    version = "python%d.%d" % sys.version_info[:2]
    executable = Path(sys.executable).absolute()
    candidates = [executable.parent.parent / lib / version / "site-packages" for lib in ("lib", "lib64")]
    candidates += [Path(sysconfig.get_paths()[k]) for k in ("purelib", "platlib")]
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))
    from paper3r6.external import prepare
    prepare(args.root, args.output, args.source_manifest, subjects,
            args.raw or args.output.parent / "raw", download=args.download,
            delete_raw_after_prepare=args.delete_raw_after_prepare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
