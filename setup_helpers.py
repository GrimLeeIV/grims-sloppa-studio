"""Setup steps run by run_yue_studio.bat (stdlib only, so they work in a bare venv).

    python setup_helpers.py fetch-yue     # third_party/YuE at the pinned commit
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
YUE_DIR = PROJECT_DIR / "third_party" / "YuE"
YUE_REPO = "https://github.com/multimodal-art-projection/YuE"
# The commit the studio's engine shims are tested against. Bump deliberately.
YUE_COMMIT = "1647252d68b70cbead046ffd3f7caf036dde47ee"


def fetch_yue() -> None:
    """Put the YuE2 runtime in third_party/YuE at YUE_COMMIT.

    Uses git when it's installed; otherwise downloads GitHub's zip of that
    exact commit, so git isn't a requirement. An existing checkout is left
    alone (its package folder is what matters).
    """
    if (YUE_DIR / "src" / "yue2").is_dir():
        print("YuE2 runtime already present.")
        return
    YUE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        print(f"Cloning YuE2 ({YUE_COMMIT[:7]}) with git...")
        subprocess.run(["git", "clone", YUE_REPO + ".git", str(YUE_DIR)], check=True)
        subprocess.run(["git", "-C", str(YUE_DIR), "checkout", "--quiet", YUE_COMMIT], check=True)
        return
    print(f"git not found; downloading YuE2 ({YUE_COMMIT[:7]}) as a zip...")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "yue.zip"
        urllib.request.urlretrieve(f"{YUE_REPO}/archive/{YUE_COMMIT}.zip", archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp)
        shutil.move(str(Path(tmp) / f"YuE-{YUE_COMMIT}"), str(YUE_DIR))
    print("YuE2 runtime downloaded.")


if __name__ == "__main__":
    commands = {"fetch-yue": fetch_yue}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        print("usage: python setup_helpers.py " + "|".join(commands))
        sys.exit(2)
    try:
        commands[sys.argv[1]]()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
