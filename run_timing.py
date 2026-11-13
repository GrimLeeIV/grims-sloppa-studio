"""Rough run-time estimates for the canvas's rendering cards.

A first guess comes from this machine's GPU (its SM count against the RTX
4080 SUPER the base rates were measured on) and how busy that GPU already is.
Every finished run then teaches a per-kind correction factor, kept in
``.temp/timing.json``, so estimates settle in on this machine after a few
runs. The UI shows the estimate padded by ``PAD`` so runs usually finish
early.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

STORE = Path(__file__).resolve().parent / ".temp" / "timing.json"
PAD = 1.5
REFERENCE_SMS = 80  # RTX 4080 SUPER
COLD_LOAD_SECONDS = 45  # loading the model before the first run

# (fixed seconds, seconds per second of audio) per item on the reference GPU.
BASE: Dict[str, tuple] = {
    "generate": (12.0, 1.2),
    "extend": (6.0, 1.0),
    "redux": (25.0, 1.3),
    "stems": (4.0, 0.12),
}
INSTRUMENTAL = 1.7  # the vocal guard's checks and rewrites

_gpu_scale: Optional[float] = None


def _factors() -> Dict[str, float]:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def gpu_scale() -> float:
    """How much slower than the reference GPU this machine is (sampled once)."""
    global _gpu_scale
    if _gpu_scale is None:
        try:
            import torch

            if torch.cuda.is_available():
                sms = torch.cuda.get_device_properties(0).multi_processor_count
                _gpu_scale = max(0.5, min(REFERENCE_SMS / max(sms, 1), 4.0))
            else:
                _gpu_scale = 20.0
        except Exception:
            _gpu_scale = 1.0
    return _gpu_scale


def load_scale() -> float:
    """Slowdown from other work already on the GPU (games, other AI apps)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        busy = float(out.split()[0])
    except Exception:
        return 1.0
    return 1.0 + busy / 100 if busy > 15 else 1.0


def estimate(kind: str, audio_seconds: float, *, instrumental: bool = False, cold: bool = False) -> float:
    """Expected seconds for one item (unpadded), learned correction included."""
    fixed, rate = BASE[kind]
    seconds = fixed + rate * max(audio_seconds, 0.0)
    if instrumental and kind != "stems":
        seconds *= INSTRUMENTAL
    seconds *= gpu_scale() * load_scale() * _factors().get(kind, 1.0)
    return seconds + (COLD_LOAD_SECONDS * gpu_scale() if cold else 0.0)


def shown(seconds: float) -> int:
    """The estimate the UI shows (padded)."""
    return int(seconds * PAD)


def record(kind: str, predicted: float, actual: float) -> None:
    """Fold a finished run into the kind's correction factor."""
    if predicted <= 0 or actual <= 0:
        return
    factors = _factors()
    old = factors.get(kind, 1.0)
    factors[kind] = round(max(0.3, min(old * (0.6 + 0.4 * actual / predicted), 5.0)), 3)
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(factors), encoding="utf-8")
    except Exception:
        pass
