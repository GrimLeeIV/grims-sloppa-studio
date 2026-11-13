"""Startup system check: what this machine can run, in plain words.

Printed to the console at launch, and the warnings are shown in the studio
itself, so someone on a different machine learns *why* things will be slow
or fail before they wait on a generation.

YuE2's needs (measured on the studio's own pipeline): an NVIDIA GPU with
BF16 support (RTX 30-series or newer), about 12 GB of VRAM (peak ~10 GiB),
and ~25 GB of disk for the environment and model weights.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, List

PROJECT_DIR = Path(__file__).resolve().parent
MIN_VRAM_GIB = 11.5      # 12 GB cards report ~11.6-11.9 GiB usable
RESIDENT_VRAM_GIB = 14.0  # below this the decoder isn't kept resident
MIN_FREE_DISK_GB = 15.0

WARNINGS: List[str] = []


def _gpu() -> dict:
    try:
        import torch
    except Exception:
        return {"torch": None}
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}
    if info["available"]:
        props = torch.cuda.get_device_properties(0)
        info.update(name=props.name, vram=props.total_memory / 2**30, capability=(props.major, props.minor))
    return info


def report(engine: Any = None) -> List[str]:
    """Print the summary; returns (and keeps) the warnings for the UI."""
    WARNINGS.clear()
    gpu = _gpu()
    print(f"[System] Python {sys.version.split()[0]} · torch {gpu.get('torch')} · CUDA build {gpu.get('cuda')}")
    if not gpu.get("torch"):
        WARNINGS.append("PyTorch isn't installed correctly. Run Grim's Sloppa Studio.bat again to repair the setup.")
    elif not gpu.get("available"):
        WARNINGS.append(
            "No usable NVIDIA GPU was found, so everything would run on the CPU and take a very long time. "
            "YuE2 needs an NVIDIA RTX 30-series or newer with about 12 GB of VRAM. If you have one, update "
            "the NVIDIA driver (CUDA 12.1 needs driver 531 or newer)."
        )
    else:
        major, minor = gpu["capability"]
        print(f"[System] GPU {gpu['name']} · {gpu['vram']:.1f} GiB · compute {major}.{minor}")
        if (major, minor) < (8, 0):
            WARNINGS.append(
                f"{gpu['name']} doesn't support BF16, which YuE2 requires; the model will fail to load. "
                "An RTX 30-series or newer is needed."
            )
        elif gpu["vram"] < MIN_VRAM_GIB:
            WARNINGS.append(
                f"{gpu['name']} has {gpu['vram']:.0f} GB of VRAM; YuE2 needs about 12 GB, so expect "
                "out-of-memory errors (shorter Seconds and Batch 1 help a little)."
            )
        elif gpu["vram"] < RESIDENT_VRAM_GIB:
            print("[System] Under 14 GB of VRAM: the audio decoder loads per track to fit (a little slower).")
    free = shutil.disk_usage(PROJECT_DIR).free / 1e9
    print(f"[System] Free disk on the studio's drive: {free:.0f} GB")
    if free < MIN_FREE_DISK_GB:
        WARNINGS.append(f"Only {free:.0f} GB of disk is free on the studio's drive; model downloads need ~15 GB.")
    for warning in WARNINGS:
        print(f"[System] WARNING: {warning}")
    return list(WARNINGS)
