@echo off
title Grim's Sloppa Studio - Upload Add-on Setup
cd /d "%~dp0"

echo ===================================================
echo  Upload add-on setup (SheetSage2: ~6 GB download, ~9 GB on disk)
echo  Installs into addons\sheetsage2. Delete that folder to remove it.
echo  You can also run this from the studio's Upload tab.
echo ===================================================

if not exist ".venv\Scripts\python.exe" (
    echo Run run_yue_studio.bat once first to set up the studio itself.
    goto END
)

.venv\Scripts\python.exe upload_addon.py install
if errorlevel 1 (
    echo.
    echo Setup stopped. Run this again to resume where it left off.
    goto END
)
echo.
echo Done. Restart the studio (or reload its page) to see the Upload tab.

:END
pause
