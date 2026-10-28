#!/usr/bin/env bash
# End-to-end dry run on synthetic traces (CPU only). Outputs -> outputs/synthetic/
set -euo pipefail
cd "$(dirname "$0")"
python scripts/01_collect_traces.py --synthetic
python scripts/02_diagnose.py --synthetic
python scripts/03_fit_estimators.py --synthetic
python scripts/04_compare_trees.py --synthetic --set compare.eval_steps=2000
