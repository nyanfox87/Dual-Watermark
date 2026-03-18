#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-0.0.0.0}"
PORT="${2:-7867}"
PYTHON_BIN="${PYTHON_BIN:-/home/project/Documents/EditGuard/envs/editguard/bin/python}"

"$PYTHON_BIN" -m uvicorn api:app --host "$HOST" --port "$PORT"
