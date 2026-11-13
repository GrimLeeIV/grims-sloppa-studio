"""Build the one-file installer: "Installer - share this"/Install Grim's Sloppa Studio.bat

    python build_installer.py

The installer is a small batch file with the studio's code packed inside it
(a base64 zip after its last line). Run from any empty folder, it unpacks the
code into "Core Files", adds an "Outputs" folder and the "Grim's Sloppa
Studio.bat" launcher, then hands over to that launcher (which sets everything
else up on first run) and deletes itself. Nothing else needs hosting.
"""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
# Installed, the installer lands next to "Core Files" in a folder that says what it's for.
OUT_DIR = (PROJECT_DIR.parent / "Installer - share this" if PROJECT_DIR.name == "Core Files"
           else PROJECT_DIR / "dist")
OUTPUT = OUT_DIR / "Install Grim's Sloppa Studio.bat"
# The studio's own files; everything else (.venv, models, add-ons, outputs) is made on install.
PACKED = ["*.py", "requirements.txt", "run_yue_studio.bat", "setup_upload_addon.bat", ".gitignore", "README.md",
          "assets/**/*"]
LEFT_OUT = {"comfy_api_bridge.py"}  # the old ComfyUI bridge; the studio doesn't use it

INSTALLER = r"""@echo off
title Grim's Sloppa Studio - Install
cd /d "%~dp0"
set "GSS_SELF=%~f0"
set "GSS_LAUNCH=%~dp0Grim's Sloppa Studio.bat"
echo ===================================================
echo  Grim's Sloppa Studio - installer
echo  Installs into: %~dp0
echo ===================================================
if exist "Core Files\app.py" (
    echo The studio is already installed here. Updating its code ^(your settings, models and music stay^)...
)
echo Unpacking the studio...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$t = [IO.File]::ReadAllText($env:GSS_SELF); $i = $t.LastIndexOf('::' + 'PAYLOAD::'); $zip = Join-Path $env:TEMP ('gss_' + [guid]::NewGuid() + '.zip'); [IO.File]::WriteAllBytes($zip, [Convert]::FromBase64String($t.Substring($i + 11).Trim())); Expand-Archive -Force $zip 'Core Files'; Remove-Item $zip"
if errorlevel 1 (
    echo ERROR: Could not unpack the studio. Make sure this folder isn't read-only, then run this again.
    pause
    exit /b 1
)
if not exist "Outputs" mkdir "Outputs"
> "%GSS_LAUNCH%" echo @echo off
>> "%GSS_LAUNCH%" echo rem Starts Grim's Sloppa Studio ^(and finishes its setup the first time^).
>> "%GSS_LAUNCH%" echo rem Everything it needs lives in "Core Files"; your music goes to "Outputs".
>> "%GSS_LAUNCH%" echo call "%%~dp0Core Files\run_yue_studio.bat"
echo Unpacked. Next: setting up and opening the studio.
echo This installer now replaces itself with "Grim's Sloppa Studio.bat" - use that from now on.
echo.
(goto) 2>nul & del "%GSS_SELF%" & call "%GSS_LAUNCH%"
exit /b 0
"""


def build() -> Path:
    files = sorted({path for pattern in PACKED for path in PROJECT_DIR.glob(pattern)
                    if path.is_file() and path.name not in LEFT_OUT and "__pycache__" not in path.parts})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(PROJECT_DIR).as_posix())
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    lines = [payload[i:i + 120] for i in range(0, len(payload), 120)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = INSTALLER.replace("\n", "\r\n") + "::PAYLOAD::\r\n" + "\r\n".join(lines) + "\r\n"
    OUTPUT.write_bytes(text.encode("ascii"))
    print(f"Packed {len(files)} files -> {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return OUTPUT


if __name__ == "__main__":
    build()
