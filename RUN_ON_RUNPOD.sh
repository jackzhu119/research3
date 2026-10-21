#!/usr/bin/env bash
set +x
set -eu
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ "${1:-}" = "--status" ]; then
    shift
    exec python3 -B CHECK_PAPER3.py "$@"
fi
exec python3 -B PAPER3_AUTO_RUN.py "$@"
