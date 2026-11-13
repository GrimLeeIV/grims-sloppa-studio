@echo off
setlocal enabledelayedexpansion
title Grim's Sloppa Studio
cd /d "%~dp0"

rem Bump when setup steps change: an install with an older marker reruns setup.
set "SETUP_VERSION=2"
set "MARKER=.venv\studio_setup_v%SETUP_VERSION%.ok"
set "VPY=.venv\Scripts\python.exe"

echo ===================================================
echo  Grim's Sloppa Studio
echo ===================================================

if exist "%MARKER%" goto LAUNCH_APP

rem ---------------------------------------------------------------------------
rem Setup (first run, or resuming one that stopped). Every step is skipped when
rem already done, so running this again after a failure picks up where it left.
rem ---------------------------------------------------------------------------
if exist "%VPY%" goto INSTALL_TORCH

echo [1/5] Finding Python 3.11 or 3.12...
set "PY="
for %%V in (3.11 3.12) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    rem No Python yet: install 3.11 for this user with winget ^(built into Windows 10/11^).
    where winget >nul 2>&1 && (
        echo Python 3.11 isn't installed. Installing it now ^(about 30 MB^)...
        winget install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
        if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe""
    )
)
if not defined PY (
    echo ERROR: The studio needs Python 3.11 ^(recommended^) or 3.12.
    echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run this again.
    goto ERROR_PAUSE
)
echo Using: !PY!
!PY! -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create the virtual environment ^(.venv^).
    goto ERROR_PAUSE
)
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check -q

:INSTALL_TORCH
echo [2/5] PyTorch 2.5.1 with CUDA 12.1 ^(about 2.5 GB the first time^)...
"%VPY%" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.5.1') else 1)" >nul 2>&1
if errorlevel 1 (
    "%VPY%" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 goto ERROR_PAUSE
)

echo [3/5] Studio requirements...
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check -q
if errorlevel 1 goto ERROR_PAUSE

echo [4/5] YuE2 runtime...
"%VPY%" setup_helpers.py fetch-yue
if errorlevel 1 goto ERROR_PAUSE
"%VPY%" -m pip install -e "./third_party/YuE" --no-deps --disable-pip-version-check -q
if errorlevel 1 goto ERROR_PAUSE

echo [5/5] Checking the install...
"%VPY%" -c "import torch, gradio, transformers, demucs, yue2" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Some packages failed to import. Run this again to retry the setup.
    goto ERROR_PAUSE
)
echo ok> "%MARKER%"
echo Setup complete.

:LAUNCH_APP
if defined STUDIO_SETUP_ONLY exit /b 0
echo Launching... ^(the studio window opens when it is ready^)
"%VPY%" app.py
if errorlevel 1 goto ERROR_PAUSE
rem Closed normally (or restarted itself): nothing to read, so no pause.
exit /b 0

:ERROR_PAUSE
echo ===================================================
echo  Something went wrong. Scroll up for the reason.
echo  Running this file again resumes the setup.
echo ===================================================

:END
pause
