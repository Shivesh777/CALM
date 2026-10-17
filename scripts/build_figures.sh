#!/usr/bin/env bash
# Re-export the data behind the figures from experiments/results and render them (needs npm and playwright).
set -euo pipefail
PY="${PYTHON:-$(command -v python || command -v python3)}"
cd "$(dirname "$0")/../figures"
[ -d node_modules ] || npm install
"$PY" export_data.py
"$PY" render.py
