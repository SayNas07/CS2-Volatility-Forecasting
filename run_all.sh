#!/usr/bin/env bash
# Full pipeline. Requires STEAM_LOGIN_SECRET to be set.
# Skips stage 13 (robustness, 1-2 hours) - run that separately.
set -euo pipefail

: "${STEAM_LOGIN_SECRET:?Set STEAM_LOGIN_SECRET to your steamLoginSecure cookie}"

python steam_pull.py --discover
python steam_pull.py --pull
python steam_pull.py --screen
python clean_panel.py
python stylised_facts.py
python fit_all.py
python ar_check.py
python forecast.py
python evaluate.py
python lambda_check.py
python mcs_test.py
python shock_analysis.py
python make_figures.py

echo "Done. Outputs in data/ and figures/."
echo "Run 'python robustness.py' separately - it takes 1-2 hours."
