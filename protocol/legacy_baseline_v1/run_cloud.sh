#!/usr/bin/env bash
set -Eeuo pipefail
P3_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
P3_ENV_DIR="${P3_ENV_DIR:-$P3_HERE/.venv}"
P3_ROOT="${P3_ROOT:-/workspace/paper3_source_data}"
P3_DEVICE="${P3_DEVICE:-cpu}"
cd "$P3_HERE"
if [[ "${1:-}" != '--execute' ]]; then
  "$P3_ENV_DIR/bin/python" -m p3 --root "$P3_ROOT" doctor
  "$P3_ENV_DIR/bin/python" -m p3 --root "$P3_ROOT" prep
  "$P3_ENV_DIR/bin/python" -m p3 --root "$P3_ROOT" train --device "$P3_DEVICE"
  echo 'Review above, then use bash run_cloud.sh --execute for source preparation + LODO training.'
  exit 0
fi
"$P3_ENV_DIR/bin/python" -m p3 --root "$P3_ROOT" prep --execute
"$P3_ENV_DIR/bin/python" -m p3 --root "$P3_ROOT" train --device "$P3_DEVICE" --execute
echo 'Source LODO jobs finished. Final all-source fitting is a separate explicit train --final --execute command.'
echo 'No final holdout evaluation or cloud-server shutdown/upload was performed.'
