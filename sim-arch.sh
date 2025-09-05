#!/usr/bin/env bash
set -euo pipefail

echo "== Adaptive Orbital D2D: Repro =="

# Python & deps
if [ ! -d "venv" ]; then python3 -m venv venv; fi
source venv/bin/activate
python -m pip install -U pip >/dev/null
pip install -r requirements.txt >/dev/null

# Folders
mkdir -p results figures

# 1) Simulation
python main_simulation.py

# 2) Figures (saved to figures/)
python generate_figures.py

# 3) Analysis
python analyze_results.py

echo "== Done. See: results/*, figures/figure_*.pdf/.png, statistical_report.txt =="
