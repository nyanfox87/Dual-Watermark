#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-0.0.0.0}"
PORT="${2:-7869}"
PYTHON_BIN="${PYTHON_BIN:-/home/project/Documents/EditGuard/envs/editguard/bin/python}"

"$PYTHON_BIN" app.py --host "$HOST" --port "$PORT"
