# ─────────────────────────────────────────────────────────────────
#  CA-CRS+ Dashboard Launcher
#  Run from the CA-CRS-Research project root in PowerShell:
#      .\run_dashboard.ps1
# ─────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "  🚨  CA-CRS+ Crowd Safety Command Dashboard" -ForegroundColor Cyan
Write-Host "  ──────────────────────────────────────────"
Write-Host "  Starting Streamlit server..."
Write-Host ""

# Activate venv if present
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

# Launch Streamlit
streamlit run dashboard/app.py `
    --server.headless false `
    --server.port 8501 `
    --browser.gatherUsageStats false `
    --theme.base dark `
    --theme.backgroundColor "#0a0e1a" `
    --theme.primaryColor "#6366f1" `
    --theme.secondaryBackgroundColor "#0f1628" `
    --theme.textColor "#e2e8f0"
