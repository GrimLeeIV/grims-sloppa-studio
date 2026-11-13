"""Upload add-on: cover (and continue) your own audio via SheetSage2.

YuE2 cannot read audio directly: the tokenizer that turns audio into its song
tokens is not public. The supported route is to transcribe the recording into
a melody-only ABC score with SheetSage2 (same lab), then have YuE2 perform that
score in a new style. The result follows your melody; the sound is new.

SheetSage2 pins a different PyTorch/Transformers/NumPy stack, so it is an
optional add-on with its own environment under ``addons/sheetsage2``:

    addons/sheetsage2/
        .venv/          Python 3.11 + PyTorch 2.8 (CUDA 12.6) + its requirements
        model/          SheetSage2 snapshot, pinned to a reviewed revision
        hf_cache/       its MERT-v2 parent (and any other HF downloads)
        install.log     installer output
        ready.json      written when the install has been verified

Deleting that folder removes the add-on completely. The installer is resumable:
every step is skipped once done, so a failed or interrupted install can simply
be run again (from the studio's Upload tab, or ``setup_upload_addon.bat``).

SheetSage2's weights are licensed CC BY-NC 4.0 (non-commercial).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import prompt_tags

PROJECT_DIR = Path(__file__).resolve().parent
ADDON_DIR = PROJECT_DIR / "addons" / "sheetsage2"
VENV_DIR = ADDON_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe" if os.name == "nt" else VENV_DIR / "bin" / "python"
MODEL_DIR = ADDON_DIR / "model"
HF_HOME = ADDON_DIR / "hf_cache"
LOG_FILE = ADDON_DIR / "install.log"
READY_FILE = ADDON_DIR / "ready.json"
RUNNER = PROJECT_DIR / "sheetsage2_runner.py"

SHEETSAGE2_REPO = "m-a-p/SheetSage2"
# Pinned: loading runs the repository's own Python code (trust_remote_code).
SHEETSAGE2_REVISION = "b83c3d79a6d98bcffaf8f9ed112c90aa887887bc"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
DOWNLOAD_ESTIMATE = "about 6 GB to download, 9 GB on disk"

# Longest stretch of an upload that is transcribed. Covers plus a continuation
# stay inside YuE2's song-token budget.
MAX_UPLOAD_SECONDS = 240.0
UPLOAD_FORMATS = [".mp3", ".wav", ".flac", ".ogg"]

INSTALL_STEPS = 6
_INSTALL_PROCESS: Optional[subprocess.Popen] = None


# ---------------------------------------------------------------------------
# Status (studio side)
# ---------------------------------------------------------------------------
def is_ready() -> bool:
    return READY_FILE.is_file() and VENV_PYTHON.is_file()


def is_installing() -> bool:
    return _INSTALL_PROCESS is not None and _INSTALL_PROCESS.poll() is None


def start_install() -> None:
    """Run the installer in the background, logging to ``install.log``.

    Calling it again while an install is running just leaves it running.
    """
    global _INSTALL_PROCESS
    if is_installing():
        return
    ADDON_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _INSTALL_PROCESS = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "install"],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_DIR),
        creationflags=flags,
    )


def install_progress() -> Tuple[str, str]:
    """``(current step, last log line)`` for the setup panel."""
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", ""
    step = next((line for line in reversed(lines) if line.startswith("STEP ")), "")
    last = next((line.strip() for line in reversed(lines) if line.strip()), "")
    return step, last[-200:]


# ---------------------------------------------------------------------------
# Installer (runs as ``python upload_addon.py install``)
# ---------------------------------------------------------------------------
def _step(number: int, text: str) -> None:
    print(f"STEP {number}/{INSTALL_STEPS}: {text}", flush=True)


def _run(command: List[str], **kwargs: Any) -> None:
    print("> " + " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def _addon_env(offline: bool = False) -> Dict[str, str]:
    """Environment for add-on processes: its own HF cache, no user site."""
    env = dict(os.environ)
    env["HF_HOME"] = str(HF_HOME)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def _venv_check(code: str) -> bool:
    if not VENV_PYTHON.is_file():
        return False
    result = subprocess.run([str(VENV_PYTHON), "-I", "-c", code], capture_output=True, env=_addon_env())
    return result.returncode == 0


def _base_python() -> List[str]:
    """A Python 3.10/3.11 interpreter to build the add-on environment from."""
    candidates = [[getattr(sys, "_base_executable", sys.executable)], ["py", "-3.11"], ["py", "-3.10"]]
    for command in candidates:
        try:
            result = subprocess.run(
                command + ["-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                capture_output=True, text=True,
            )
        except OSError:
            continue
        if result.returncode == 0 and result.stdout.strip() in ("3.10", "3.11"):
            return command
    raise RuntimeError("SheetSage2 needs Python 3.10 or 3.11; install Python 3.11 from python.org and retry.")


def install() -> None:
    pip = [str(VENV_PYTHON), "-I", "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check"]
    env = _addon_env()

    _step(1, "Creating the add-on environment")
    if not VENV_PYTHON.is_file():
        _run(_base_python() + ["-m", "venv", str(VENV_DIR)])

    _step(2, "Installing PyTorch 2.8 with CUDA 12.6 (the largest download, ~3 GB)")
    if not _venv_check("import torch, torchaudio; assert torch.__version__.startswith('2.8.0')"):
        _run(pip + ["torch==2.8.0", "torchaudio==2.8.0", "--index-url", TORCH_INDEX], env=env)

    _step(3, "Downloading SheetSage2")
    revision_file = MODEL_DIR / ".studio_revision"
    if not (revision_file.is_file() and revision_file.read_text().strip() == SHEETSAGE2_REVISION):
        _run(pip + ["huggingface-hub==0.36.0"], env=env)
        _run(
            [str(VENV_PYTHON), "-I", "-c",
             "import sys; from huggingface_hub import snapshot_download; "
             "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])",
             SHEETSAGE2_REPO, SHEETSAGE2_REVISION, str(MODEL_DIR)],
            env=env,
        )
        revision_file.write_text(SHEETSAGE2_REVISION)

    _step(4, "Installing SheetSage2's requirements")
    if not _venv_check("import transformers, mir_eval, pretty_midi, soundfile; "
                       "assert transformers.__version__ == '4.45.2'"):
        _run(pip + ["-r", str(MODEL_DIR / "requirements.txt"), "soundfile"], env=env)

    _step(5, "Fetching the MERT-v2 encoder (~2.4 GB) and loading the model once")
    _run([str(VENV_PYTHON), "-I", str(RUNNER), "--model", str(MODEL_DIR), "--prefetch"], env=env)

    _step(6, "Finishing up")
    READY_FILE.write_text(
        json.dumps({"revision": SHEETSAGE2_REVISION, "installed": time.strftime("%Y-%m-%d %H:%M:%S")}),
        encoding="utf-8",
    )
    print("DONE: the Upload add-on is ready.", flush=True)


# ---------------------------------------------------------------------------
# Transcription + cover score preparation (studio side)
# ---------------------------------------------------------------------------
def transcribe(audio_path: str, work_dir: Optional[str] = None,
               max_seconds: float = MAX_UPLOAD_SECONDS) -> Dict[str, Any]:
    """Transcribe an upload to a melody-only ABC score in the add-on process.

    Returns the runner's summary (``abc``, ``seconds``, ``warnings`` ...) and
    raises with SheetSage2's own reason when no usable score came out.
    """
    if not is_ready():
        raise RuntimeError("The Upload add-on isn't installed yet; set it up from the Upload tab first.")
    source = Path(audio_path)
    if source.suffix.lower() not in UPLOAD_FORMATS:
        raise ValueError(f"'{source.suffix}' files aren't supported; use {', '.join(UPLOAD_FORMATS)}.")
    temp_root = PROJECT_DIR / ".temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(work_dir or tempfile.mkdtemp(prefix="sheetsage2_", dir=str(temp_root)))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = subprocess.run(
            [str(VENV_PYTHON), "-I", str(RUNNER), "--model", str(MODEL_DIR), str(source),
             "--output", str(out_dir), "--max-seconds", str(min(float(max_seconds), MAX_UPLOAD_SECONDS))],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=_addon_env(offline=True), creationflags=flags, timeout=1800,
        )
        summary_file = out_dir / "studio_result.json"
        if not summary_file.is_file():
            tail = (process.stderr or process.stdout or "").strip().splitlines()[-1:] or ["no output"]
            raise RuntimeError(f"SheetSage2 failed: {tail[0]}")
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    finally:
        if work_dir is None:  # our scratch folder: the summary already holds the score
            shutil.rmtree(out_dir, ignore_errors=True)
    if not summary.get("ok"):
        raise RuntimeError(f"SheetSage2 couldn't build a score from this audio: {summary.get('abc_error')}")
    print(f"[Upload] Transcribed {source.name}: {summary['seconds']:.0f}s, "
          f"peak GPU {summary.get('peak_gpu_mib', 0):.0f} MiB, warnings: {summary.get('warnings')}")
    return summary


def prepare_cover(abc: str, style: str, lyrics: str) -> Tuple[str, str, str]:
    """Turn a transcribed score + the user's prompt into a YuE2 cover request.

    With lyrics, the melody-only score is used as-is and the vocal line is
    sung. Without (empty or "[instrumental]"), this follows upstream's
    instrumental-cover recipe: the vocal melody moves onto the instrumental
    voice so nothing is left to sing, the style asks for no vocals, and the
    lyrics become the score's section tags. Returns ``(abc, style, lyrics)``.
    """
    if not prompt_tags.is_instrumental(lyrics):
        return abc, style, lyrics
    converted = prompt_tags.to_instrumental_score(abc, keep_chords=False)
    return converted, prompt_tags.instrumental_style(style), prompt_tags.section_tags(converted)


if __name__ == "__main__":
    if sys.argv[1:] == ["install"]:
        try:
            install()
        except Exception as exc:
            print(f"FAILED: {exc}", flush=True)
            sys.exit(1)
    else:
        print("usage: python upload_addon.py install")
        sys.exit(2)
