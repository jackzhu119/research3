#!/usr/bin/env python3
"""Check existing R2 bindings with HeadBucket; never list/read/write objects."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = ("R2_ENDPOINT", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "HeadBucket",
        "variables_present": {name: bool(os.environ.get(name)) for name in names},
        "credential_values_recorded": False,
        "uploads_attempted": False,
        "object_reads_attempted": False,
    }
    if not all(report["variables_present"].values()):
        report["status"] = "blocked_missing_variables"
    else:
        try:
            import boto3
            from botocore.config import Config
            client = boto3.client(
                "s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                config=Config(connect_timeout=10, read_timeout=15,
                              retries={"max_attempts": 1}),
            )
            response = client.head_bucket(Bucket=os.environ["R2_BUCKET"])
            report.update(status="pass", http_status=response["ResponseMetadata"]["HTTPStatusCode"])
        except Exception as exc:
            # Exception messages may contain endpoints or request data. Record only
            # the exception class and HTTP status, never str(exc) or a traceback.
            report.update(status="failed", error_type=type(exc).__name__)
            response = getattr(exc, "response", {})
            http_status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if http_status is not None:
                report["http_status"] = http_status
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
