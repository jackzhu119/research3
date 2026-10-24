#!/usr/bin/env python3
"""Explicit CPU preparation of pinned Pan2023; workers consume prebuilt cache."""
import argparse
from pathlib import Path
import sys
import sysconfig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--source-manifest", type=Path, default=Path(__file__).resolve().with_name("PAN2023_DOWNLOAD_MANIFEST.json"))
    p.add_argument("--download", action="store_true")
    args = p.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.append(str(args.root.resolve()))
    version = "python%d.%d" % sys.version_info[:2]
    for x in [Path(sys.executable).absolute().parent.parent / lib / version / "site-packages" for lib in ("lib", "lib64")]:
        if x.is_dir():sys.path.append(str(x))
    for key in ("purelib", "platlib"):
        x = sysconfig.get_paths()[key]
        if x not in sys.path:sys.path.append(x)
    from paper3r7.pan2023 import prepare
    prepare(args.root, args.output, args.source_manifest, args.raw, download=args.download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
