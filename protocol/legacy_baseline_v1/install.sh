#!/usr/bin/env bash
set -Eeuo pipefail
P3_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
P3_ENV_DIR="${P3_ENV_DIR:-$P3_HERE/.venv}"
P3_BACKEND="${1:-cpu}"
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 for this pinned package"'
case "$P3_BACKEND" in
  cpu) P3_TORCH_INDEX=https://download.pytorch.org/whl/cpu ;;
  cu128) P3_TORCH_INDEX=https://download.pytorch.org/whl/cu128 ;;
  *) echo 'Usage: bash install.sh cpu|cu128' >&2; exit 2 ;;
esac
python3 -m venv "$P3_ENV_DIR"
"$P3_ENV_DIR/bin/python" -m pip install -r "$P3_HERE/requirements.txt"
"$P3_ENV_DIR/bin/python" -m pip install torch==2.8.0 --index-url "$P3_TORCH_INDEX"
"$P3_ENV_DIR/bin/python" -m pip check
"$P3_ENV_DIR/bin/python" -m pip freeze > "$P3_HERE/requirements-resolved-$P3_BACKEND.txt"
cd "$P3_HERE"
"$P3_ENV_DIR/bin/python" -m p3 doctor > "$P3_HERE/installed-environment-$P3_BACKEND.json"
echo 'Installation complete. No EEG was downloaded and no training started.'
echo "Python: $P3_ENV_DIR/bin/python"
