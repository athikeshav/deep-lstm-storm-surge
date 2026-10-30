#!/usr/bin/env bash
# 07 - Launch the Streamlit dashboard with replay mode.
set -euo pipefail
cd "$(dirname "$0")/.."
streamlit run dashboard/app.py --server.port 8501 --server.headless true
