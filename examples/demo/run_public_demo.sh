#!/usr/bin/env bash
# Public MILIEU review demos: figure replay + infer-only (no training).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python examples/demo/plot_fig2b_demo.py
python examples/demo/run_infer_demo.py

echo "Public demo finished (plot PNG + infer score JSON)."
