"""Stem separation for finished tracks (Demucs).

YuE2 renders a single stereo mix, so stems come from a separation model run
over the finished MP3. Demucs is local and open source; weights download from
Hugging Face on first use (into the same cache as the YuE2 weights). Because it
works on any audio file, older tracks without token sidecars can be split too.

Separation is not perfect: dense mixes leave some bleed between stems.
"""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

# UI label -> Demucs model. htdemucs_ft is a bag of four fine-tuned models
# (slower, best 4-stem quality); htdemucs_6s adds guitar and piano.
STEM_MODES = {
    "4 stems": "htdemucs_ft",
    "6 stems": "htdemucs_6s",
}
DEFAULT_STEM_MODE = "4 stems"
# Display order for the result cards; models only emit the stems they know.
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]
MAX_STEMS = len(STEM_ORDER)

_SEPARATORS: Dict[str, Any] = {}


def available() -> bool:
    """True when Demucs is installed (run_yue_studio.bat installs it)."""
    return importlib.util.find_spec("demucs") is not None


def stems_dir(audio_path: Any) -> Path:
    """Stems live in a ``<track>_stems`` folder next to the track."""
    path = Path(audio_path)
    return path.with_name(path.stem + "_stems")


def _separator(model_name: str) -> Any:
    """Load each Demucs model once and keep it for later splits."""
    separator = _SEPARATORS.get(model_name)
    if separator is None:
        import torch
        from demucs.api import Separator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Stems] Loading Demucs '{model_name}' on {device} (first use downloads the weights)")
        separator = Separator(model=model_name, device=device, progress=False)
        _SEPARATORS[model_name] = separator
    return separator


# Vocal detection for the instrumental vocal guard. Calibrated on real YuE2
# output: instrumental passages give a vocals/mix energy ratio of 0.00-0.02,
# sung passages 0.3-0.6. Windows are 1 s; a passage counts when most of a
# few consecutive windows are over the threshold and the mix isn't silent.
VOCAL_MODEL = "htdemucs"  # single fast model; detection doesn't need the bag
VOCAL_RATIO = 0.15
VOCAL_RUN = 4
VOCAL_HITS = 3


def vocal_activity(audio: Any, rate: int) -> Any:
    """Per-second vocals/mix energy ratio of a [samples, channels] waveform.

    Seconds where the mix is near-silent read 0, so fades don't trigger.
    The caller holds the GPU lock.
    """
    import numpy as np
    import torch

    separator = _separator(VOCAL_MODEL)
    wav = torch.from_numpy(np.ascontiguousarray(np.asarray(audio, dtype="float32").T))
    with torch.inference_mode():
        _mix, stems = separator.separate_tensor(wav, rate)
    vocals = stems["vocals"].mean(0).cpu().numpy()
    mix = sum(stems.values()).mean(0).cpu().numpy()
    hop = separator.samplerate
    count = max(1, len(vocals) // hop)
    voc_rms = np.array([np.sqrt(np.mean(vocals[i * hop:(i + 1) * hop] ** 2)) for i in range(count)])
    mix_rms = np.array([np.sqrt(np.mean(mix[i * hop:(i + 1) * hop] ** 2)) for i in range(count)])
    audible = mix_rms > 0.2 * max(float(np.median(mix_rms)), 1e-6)
    return np.where(audible, voc_rms / np.maximum(mix_rms, 1e-6), 0.0)


def vocal_passages(audio: Any, rate: int, *, from_second: float = 0.0) -> list:
    """Sung passages as [(start, end)] whole seconds, end exclusive, from from_second on.

    A passage starts where most of a few consecutive seconds are over the
    threshold and ends after three quiet seconds in a row.
    """
    hits = vocal_activity(audio, rate) > VOCAL_RATIO
    passages, second = [], int(from_second)
    while second < len(hits):
        needed = max(2, min(VOCAL_HITS, len(hits) - second))
        if hits[second] and hits[second:second + VOCAL_RUN].sum() >= needed:
            end = second + 1
            while end < len(hits) and hits[end:end + 3].any():
                end += 1
            passages.append((second, end))
            second = end
        else:
            second += 1
    return passages


def first_vocal_second(audio: Any, rate: int, *, from_second: float = 0.0) -> Optional[int]:
    """Second at which a sung passage starts (at or after from_second), or None."""
    passages = vocal_passages(audio, rate, from_second=from_second)
    return passages[0][0] if passages else None


def separate(audio_path: str, mode: str = DEFAULT_STEM_MODE, *, lock: Optional[Any] = None) -> Dict[str, Any]:
    """Split a track into stems, saved as MP3s in ``<track>_stems``.

    ``lock`` is the engine's GPU lock, so a split never runs alongside a
    generation. Returns ``{"dir": <folder>, "files": {stem: path}}`` with the
    stems in display order.
    """
    import soundfile as sf
    import torch
    from demucs.api import save_audio

    model_name = STEM_MODES.get(mode, STEM_MODES[DEFAULT_STEM_MODE])
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"'{source}' does not exist.")

    data, rate = sf.read(str(source), always_2d=True, dtype="float32")
    wav = torch.from_numpy(data.T.copy())  # [channels, samples]
    with lock if lock is not None else nullcontext():
        separator = _separator(model_name)
        with torch.inference_mode():
            _mix, stems = separator.separate_tensor(wav, rate)

    out_dir = stems_dir(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    for name in sorted(stems, key=lambda s: STEM_ORDER.index(s) if s in STEM_ORDER else len(STEM_ORDER)):
        path = out_dir / f"{source.stem}_{name}.mp3"
        # "clamp" keeps every stem at its true level, so the stems still sum to
        # the mix; the default "rescale" would re-level each one separately.
        save_audio(stems[name].cpu(), str(path), samplerate=separator.samplerate, clip="clamp")
        files[name] = str(path.resolve())
    return {"dir": str(out_dir.resolve()), "files": files}
