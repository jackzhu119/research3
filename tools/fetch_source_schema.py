#!/usr/bin/env python3
"""Fetch only pre-registered source schema specimens, never a holdout dataset.

This does not produce epochs, processed EEG, training caches, or model fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request


SPECIMENS = {
    "BNCI2014_001": {
        "provider_subject": "1", "session_id": "T", "file": "A01T.mat",
        "url": "https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat",
        "expected_bytes": 42806453,
        "data_doi": None, "version": "001-2014, unversioned native MAT",
        "selected_part": "six native MI runs; only labels 1=left, 2=right",
    },
    "Cho2017": {
        "provider_subject": "1", "session_id": "provider_single_visit", "file": "s01.mat",
        "url": "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/s01.mat",
        "expected_bytes": 203446256,
        "data_doi": "10.5524/100295", "version": "GigaDB DOI snapshot retrieved in receipt",
        "selected_part": "imagery_left/right and imagery_event; not motor execution or non-task arrays",
    },
    "Lee2019_MI": {
        "provider_subject": "1", "session_id": "1", "file": "sess01_subj01_EEG_MI.mat",
        "url": "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/session1/s1/sess01_subj01_EEG_MI.mat",
        "expected_bytes": 609007751,
        "data_doi": "10.5524/100542", "version": "GigaDB DOI snapshot retrieved in receipt",
        "selected_part": "EEG_MI_train only; EEG_MI_test is in provider container but not parsed",
    },
    "PhysionetMI": {
        "provider_subject": "1", "session_id": "provider_single_visit", "file": "S001R04.edf",
        "url": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf",
        "expected_bytes": 2596896,
        "data_doi": "10.13026/C28G6P", "version": "1.0.0",
        "selected_part": "original run 4, explicit left/right hand MI; no executed or hands/feet runs",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/source_schema"))
    parser.add_argument("--receipt", type=Path, default=Path("audit/source_schema/raw_inventory.json"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECIMENS), default=list(SPECIMENS))
    parser.add_argument("--max-total-bytes", type=int, default=1500000000)
    args = parser.parse_args()
    selected = [SPECIMENS[name] for name in args.datasets]
    if sum(item["expected_bytes"] for item in selected) > args.max_total_bytes:
        raise SystemExit("Pre-registered source schema download exceeds explicit byte budget")
    old = json.loads(args.receipt.read_text()) if args.receipt.exists() else {"files": []}
    by_dataset = {row["dataset"]: row for row in old["files"]}
    for name in args.datasets:
        spec = SPECIMENS[name]
        path = args.raw_root / name / spec["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        reused = path.exists()
        headers = None
        if not reused:
            temporary = path.with_name(path.name + ".partial")
            with urllib.request.urlopen(spec["url"], timeout=120) as response, temporary.open("wb") as out:
                if response.status != 200:
                    raise RuntimeError("Only complete HTTP 200 raw downloads are accepted")
                headers = {key: response.headers.get(key) for key in
                           ("Content-Length", "ETag", "Last-Modified", "Content-Type")}
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > spec["expected_bytes"]:
                        raise RuntimeError("Official specimen grew; review version instead of silently replacing")
                    out.write(block)
            if temporary.stat().st_size != spec["expected_bytes"]:
                raise RuntimeError("Incomplete official source specimen")
            temporary.replace(path)
        if path.stat().st_size != spec["expected_bytes"]:
            raise RuntimeError("Existing raw specimen byte size changed")
        digest = sha256(path)
        if name in by_dataset and by_dataset[name]["sha256"] != digest:
            raise RuntimeError("Existing raw specimen SHA256 differs from retained receipt")
        row = dict(spec, dataset=name, role="source", path=str(path.resolve()),
                   project_relative_path=str(path),
                   bytes=path.stat().st_size, sha256=digest,
                   retrieved_at_utc=by_dataset.get(name, {}).get("retrieved_at_utc", datetime.now(timezone.utc).isoformat()),
                   last_verified_at_utc=datetime.now(timezone.utc).isoformat(),
                   response_headers=headers or by_dataset.get(name, {}).get("response_headers"),
                   http_status=200,
                   provider_sha256_verified=by_dataset.get(name, {}).get("provider_sha256_verified", False),
                   provider_sha256_evidence=by_dataset.get(name, {}).get("provider_sha256_evidence"),
                   scope="native schema specimen; no processed cache; not all subjects")
        by_dataset[name] = row
        receipt = {"schema_version": "paper3-source-schema-1", "status": "downloaded_schema_specimens_only",
                   "download_budget_bytes": args.max_total_bytes, "files": list(by_dataset.values()),
                   "no_external_holdout_access": True, "training_started": False}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"{name}: {row['bytes']} bytes SHA256 {digest}", flush=True)


if __name__ == "__main__":
    main()
