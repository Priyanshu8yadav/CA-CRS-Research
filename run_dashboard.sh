#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  CA-CRS+ Dashboard Launcher
#  Run from the CA-CRS-Research project root:
#      bash run_dashboard.sh
# ─────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  🚨  CA-CRS+ Crowd Safety Command Dashboard"
echo "  ──────────────────────────────────────────"
echo "  Starting Streamlit server…"
echo ""

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Launch
streamlit run dashboard/app.py \
    --server.headless false \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.backgroundColor "#0a0e1a" \
    --theme.primaryColor "#6366f1" \
    --theme.secondaryBackgroundColor "#0f1628" \
    --theme.textColor "#e2e8f0"
