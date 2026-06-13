@echo off
rem ─────────────────────────────────────────────────────────────────
rem  CA-CRS+ Dashboard Launcher
rem  Run from the CA-CRS-Research project root in Command Prompt:
rem      run_dashboard.bat
rem ─────────────────────────────────────────────────────────────────

echo.
echo   🚨  CA-CRS+ Crowd Safety Command Dashboard
echo   ──────────────────────────────────────────
echo   Starting Streamlit server...
echo.

rem Activate venv if present
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

rem Launch Streamlit
streamlit run dashboard/app.py ^
    --server.headless false ^
    --server.port 8501 ^
    --browser.gatherUsageStats false ^
    --theme.base dark ^
    --theme.backgroundColor "#0a0e1a" ^
    --theme.primaryColor "#6366f1" ^
    --theme.secondaryBackgroundColor "#0f1628" ^
    --theme.textColor "#e2e8f0"
