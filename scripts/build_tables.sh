#!/usr/bin/env bash
# Rebuild every table of the paper from the per-run results in experiments/results (CPU, a few minutes).
set -euo pipefail
PY="${PYTHON:-$(command -v python || command -v python3)}"
cd "$(dirname "$0")/../experiments"
mkdir -p tables
"$PY" paper_tab_dev_tmlr.py --out tables/tab_dev_tmlr.tex                     # Table 1: 75 development settings
"$PY" paper_tab_main_tmlr.py tables/tab_main_tmlr.tex                         # Table 2: main results (90 runs)
"$PY" paper_tab_confirm2.py --r1 $(cat results/stats/r1_dirs.txt) --r2 $(cat results/stats/r2_dirs.txt) \
    --out tables/tab_confirm2_v2.tex --out1 tables/tab_confirm_v2.tex \
    --summary-out tables/tab_confirm_rounds_v2.tex                             # Table 3 and the per-run tables
"$PY" paper_tab_ablation_v2.py                                                # full ablation table
"$PY" paper_tab_ablation_tmlr.py                                              # compact ablation table
"$PY" paper_tabs_v2.py                                                        # Proceed / DynaME protocols, RLinear
"$PY" paper_tab_seeds_dm_v2.py                                                # per-seed significance
"$PY" paper_tab_finance_v2.py                                                 # finance
echo "tables written to experiments/tables/"
