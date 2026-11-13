"""Listen to any song and describe it in YuE2 style tags.

Drop in a file and get back the tags a YuE2 style line is made of: genre,
vocal, instruments, mood, production, key and tempo. Tags pick a lane, not an
exact sound; for a close match, also give YuE2 the song itself as a reference
with the upload add-on.

* Genre, instruments, mood and production come from CLAP (LAION's
  larger_clap_music_and_speech, Apache-2.0, about 780 MB, downloaded on first use into
  models/clap-music). It scores the music against the studio's own tag
  vocabulary (prompt_tags), so every suggestion is a tag YuE2 understands.
* Whether there's singing comes from Demucs (already used for stems); the
  vocal stem is then scored on its own for male / female and its character.
* Tempo (onset autocorrelation) and key (chroma against Krumhansl's key
  profiles) are measured directly, no model needed.

Everything runs on this computer; CLAP runs on the CPU so it never competes
with a loaded YuE2 for graphics memory.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import prompt_tags

PROJECT_DIR = Path(__file__).resolve().parent
CLAP_REPO = "laion/larger_clap_music_and_speech"  # the music-only checkpoint has a broken text tower
CLAP_DIR = PROJECT_DIR / "models" / "clap-music"
CLAP_RATE = 48000
WINDOW_SECONDS = 10      # CLAP hears 10 s at a time
EXCERPT_SECONDS = 30     # how much of the song gets listened to closely
VOCAL_SHARE = 0.2        # share of the excerpt's seconds that must be sung to count as a vocal song

_CLAP: Dict[str, Any] = {}
_CLAP_LOCK = threading.Lock()
_TEXT_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Vocabulary: the studio's own tags, grouped
# ---------------------------------------------------------------------------
def _unique(items) -> List[str]:
    seen, out = set(), []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


BASIC_INSTRUMENTS = ["acoustic guitar", "clean electric guitar", "distorted electric guitar", "piano", "synthesizer",
                     "drum machine", "live drums", "bass guitar", "strings", "violin", "saxophone", "trumpet",
                     "organ", "flute", "harp", "accordion", "ukulele", "marimba"]
VOCABULARY: Dict[str, List[str]] = {
    "genre": list(prompt_tags.GENRES),
    "instruments": _unique([i for g in prompt_tags.GENRES.values() for i in g["instruments"]]
                           + prompt_tags.GENERAL_POOLS["instruments"] + BASIC_INSTRUMENTS),
    "mood": _unique([m for g in prompt_tags.GENRES.values() for m in g["mood"]]
                    + prompt_tags.GENERAL_POOLS["mood"]),
    "production": _unique([p for g in prompt_tags.GENRES.values() for p in g["production"]]
                          + prompt_tags.GENERAL_POOLS["production"]),
    "vocal": _unique([v for g in prompt_tags.GENRES.values() for v in g["vocal"]]),
}
# How each group is phrased to CLAP; it was trained on captions, not bare tags.
TEMPLATES = {
    "genre": "{} music",
    "instruments": "music featuring {}",
    "mood": "{} music",
    "production": "a song with {}",
    "vocal": "a {}",
}
PICKS = {"genre": 2, "instruments": 3, "mood": 2, "production": 1}
GENERIC = {"music", "sound", "drums", "vocal", "warm", "soft", "deep", "heavy"}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_audio(path: str) -> Tuple[Any, int]:
    """[samples, channels] float32 and the sample rate."""
    import numpy as np
    import soundfile as sf

    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        import torchaudio  # mp3/m4a and friends soundfile can't open

        wav, rate = torchaudio.load(str(path))
        audio = wav.numpy().T
    return np.ascontiguousarray(audio, dtype="float32"), int(rate)


def _resample(mono: Any, rate: int, target: int) -> Any:
    import torch
    import torchaudio.functional as F

    if rate == target:
        return mono
    return F.resample(torch.from_numpy(mono), rate, target).numpy()


def excerpt(audio: Any, rate: int, seconds: float = EXCERPT_SECONDS) -> Any:
    """The part of a song worth listening to: past the intro, around 40% in."""
    length = len(audio)
    span = int(seconds * rate)
    if length <= span:
        return audio
    start = min(max(0, int(length * 0.4) - span // 2), length - span)
    return audio[start:start + span]


# ---------------------------------------------------------------------------
# Tempo and key (no model)
# ---------------------------------------------------------------------------
def estimate_tempo(mono: Any, rate: int) -> Optional[int]:
    """Beats per minute from the autocorrelation of spectral flux, nudged towards 120."""
    import numpy as np
    import torch

    wav = torch.from_numpy(_resample(mono, rate, 22050))
    if len(wav) < 22050 * 6:
        return None
    hop = 512
    spec = torch.stft(wav, 2048, hop, window=torch.hann_window(2048), return_complex=True).abs()
    flux = torch.relu(torch.diff(torch.log1p(spec * 100), dim=1)).sum(0).numpy()
    flux = flux - np.convolve(flux, np.ones(16) / 16, mode="same")
    flux = np.maximum(flux, 0)
    fps = 22050 / hop
    corr = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    lags = np.arange(int(fps * 60 / 200), int(fps * 60 / 60) + 1)
    bpm = 60 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpm / 120) / 0.7) ** 2)
    score = corr[lags] * prior
    best = int(np.argmax(score))
    if 0 < best < len(score) - 1:  # parabolic refinement between lags
        a, b, c = score[best - 1:best + 2]
        curve = a - 2 * b + c
        shift = 0.5 * (a - c) / curve if curve < 0 else 0.0
        return int(round(60 * fps / (lags[best] + shift)))
    return int(round(bpm[best]))


NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def estimate_key(mono: Any, rate: int) -> Optional[str]:
    """"A minor"-style key from a chroma profile (Krumhansl-Schmuckler)."""
    import numpy as np
    import torch

    wav = torch.from_numpy(_resample(mono, rate, 22050))
    if len(wav) < 22050 * 4:
        return None
    n_fft = 8192
    spec = torch.stft(wav, n_fft, 2048, window=torch.hann_window(n_fft), return_complex=True).abs().mean(1).numpy()
    freqs = np.arange(len(spec)) * 22050 / n_fft
    keep = (freqs > 55) & (freqs < 2000)
    pitch = np.round(12 * np.log2(freqs[keep] / 440.0) + 69).astype(int) % 12
    chroma = np.bincount(pitch, weights=np.sqrt(spec[keep]), minlength=12)
    if chroma.sum() <= 0:
        return None
    best, name = -2.0, None
    for tonic in range(12):
        for profile, mode in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
            r = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            if r > best:
                best, name = r, f"{NOTES[tonic]} {mode}"
    return name


# ---------------------------------------------------------------------------
# CLAP
# ---------------------------------------------------------------------------
def clap_ready() -> bool:
    weights = any(CLAP_DIR.glob("*.bin")) or any(CLAP_DIR.glob("*.safetensors"))
    return (CLAP_DIR / "config.json").exists() and weights


def _clap() -> Tuple[Any, Any]:
    with _CLAP_LOCK:
        if "model" not in _CLAP:
            from transformers import ClapModel, ClapProcessor

            if not clap_ready():
                from huggingface_hub import snapshot_download

                print("[Interpret] Downloading the CLAP listening model (about 780 MB, first use only)...")
                snapshot_download(CLAP_REPO, local_dir=str(CLAP_DIR), allow_patterns=["*.json", "*.txt", "*.bin"])
            _CLAP["processor"] = ClapProcessor.from_pretrained(str(CLAP_DIR))
            _CLAP["model"] = ClapModel.from_pretrained(str(CLAP_DIR)).eval()
        return _CLAP["model"], _CLAP["processor"]


def _text_embeddings(phrases: List[str]) -> Any:
    import torch

    missing = [p for p in phrases if p not in _TEXT_CACHE]
    if missing:
        model, processor = _clap()
        inputs = processor(text=missing, return_tensors="pt", padding=True)
        with torch.inference_mode():
            emb = torch.nn.functional.normalize(model.get_text_features(**inputs), dim=-1)
        _TEXT_CACHE.update(zip(missing, emb))
    return torch.stack([_TEXT_CACHE[p] for p in phrases])


def _audio_embeddings(mono: Any, rate: int, *, spread: int = 0) -> Any:
    """CLAP embeddings of 10 s windows: back to back, or `spread` of them across 10-90% of the clip."""
    import numpy as np
    import torch

    model, processor = _clap()
    wav = _resample(mono, rate, CLAP_RATE)
    span = WINDOW_SECONDS * CLAP_RATE
    if spread and len(wav) > span * 2:
        starts = np.linspace(0.1 * len(wav), max(0.1 * len(wav), 0.9 * len(wav) - span), spread).astype(int)
    else:
        starts = range(0, max(1, len(wav) - span // 2), span)
    windows = [wav[s:s + span] for s in starts] or [wav]
    inputs = processor(audios=windows, sampling_rate=CLAP_RATE, return_tensors="pt")
    with torch.inference_mode():
        return torch.nn.functional.normalize(model.get_audio_features(**inputs), dim=-1)


def _rank(audio_embs: Any, group: str, options: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    """Options of a group, best first, with CLAP's probability averaged over the windows."""
    import torch

    model, _processor = _clap()
    options = options or VOCABULARY[group]
    text = _text_embeddings([TEMPLATES[group].format(o) for o in options])
    probs = torch.softmax(audio_embs @ text.T * model.logit_scale_a.exp(), dim=-1).mean(0).tolist()
    return sorted(zip(options, probs), key=lambda pair: -pair[1])


def _distinct(ranked: List[Tuple[str, float]], count: int) -> List[str]:
    """The top picks, skipping ones that repeat a word already chosen ("soft piano" after "piano")."""
    chosen: List[str] = []
    words: set = set()
    for tag, _p in ranked:
        own = {w for w in tag.lower().replace("-", " ").split() if w not in GENERIC and len(w) > 2}
        if own & words:
            continue
        chosen.append(tag)
        words |= own
        if len(chosen) == count:
            break
    return chosen


# ---------------------------------------------------------------------------
# The whole read
# ---------------------------------------------------------------------------
def _vocals(clip: Any, rate: int, lock: Optional[Any]) -> Tuple[Any, Any, int]:
    """Demucs on the excerpt: (vocal stem, everything-but-drums, stem rate).

    Uses the GPU when it's free; while a song is rendering it runs on the CPU
    instead (slower, but nothing waits for the render to finish).
    """
    import numpy as np
    import torch

    import stem_splitter

    wav = torch.from_numpy(np.ascontiguousarray(clip.T))
    if lock is None or lock.acquire(blocking=False):
        try:
            separator = stem_splitter._separator(stem_splitter.VOCAL_MODEL)
            with torch.inference_mode():
                _mix, stems = separator.separate_tensor(wav, rate)
        finally:
            if lock is not None:
                lock.release()
    else:
        if "cpu_separator" not in _CLAP:
            from demucs.api import Separator

            _CLAP["cpu_separator"] = Separator(model=stem_splitter.VOCAL_MODEL, device="cpu", progress=False)
        separator = _CLAP["cpu_separator"]
        with torch.inference_mode():
            _mix, stems = separator.separate_tensor(wav, rate)
    vocals = stems["vocals"].mean(0).cpu().numpy()
    tonal = sum(v for k, v in stems.items() if k != "drums").mean(0).cpu().numpy()
    return vocals, tonal, separator.samplerate


def _sung_share(vocals: Any, tonal: Any, drums_too: Any, rate: int) -> float:
    """Share of seconds where the vocal stem carries a real part of the mix."""
    import numpy as np

    import stem_splitter

    count = max(1, len(vocals) // rate)
    rms = lambda x, i: float(np.sqrt(np.mean(x[i * rate:(i + 1) * rate] ** 2)))  # noqa: E731
    mix_rms = np.array([rms(drums_too, i) for i in range(count)])
    voc_rms = np.array([rms(vocals, i) for i in range(count)])
    audible = mix_rms > 0.2 * max(float(np.median(mix_rms)), 1e-6)
    ratio = np.where(audible, voc_rms / np.maximum(mix_rms, 1e-6), 0.0)
    return float((ratio > stem_splitter.VOCAL_RATIO).mean())


def interpret(path: str, *, separator_lock: Optional[Any] = None, language: str = "English") -> Dict[str, Any]:
    """Describe a song as YuE2 tag groups.

    Returns {"genre", "vocal", "instruments", "mood", "production": [tags],
    "more": {group: [runner-up tags]}, "key": str|None, "bpm": int|None,
    "instrumental": bool, "style": the suggested style line}.
    """
    import stem_splitter

    audio, rate = load_audio(path)
    mono = audio.mean(1)
    clip = excerpt(audio, rate)

    vocal_stem = None
    instrumental = False
    key = None
    if stem_splitter.available():
        vocals, tonal, stem_rate = _vocals(clip, rate, separator_lock)
        mix = _resample(clip.mean(1), rate, stem_rate)[:len(vocals)]
        instrumental = _sung_share(vocals, tonal, mix, stem_rate) < VOCAL_SHARE
        if not instrumental:
            vocal_stem = (vocals, stem_rate)
        # Key from everything but the drums, which only smear the chroma.
        key = estimate_key(tonal, stem_rate)
    else:
        key = estimate_key(clip.mean(1), rate)

    music = _audio_embeddings(mono, rate, spread=6)
    result: Dict[str, Any] = {"more": {}}
    for group, count in PICKS.items():
        ranked = _rank(music, group)
        result[group] = _distinct(ranked, count)
        result["more"][group] = [tag for tag, _p in ranked if tag not in result[group]][:4]
    # A second genre only when it's nearly as likely as the first.
    genres = _rank(music, "genre")
    if len(genres) > 1 and genres[1][1] < 0.4 * genres[0][1]:
        result["more"]["genre"].insert(0, result["genre"].pop())

    result["vocal"] = []
    if vocal_stem is not None:
        voice = _audio_embeddings(*vocal_stem)
        gender = _rank(voice, "vocal", ["male vocal", "female vocal"])[0][0].split()[0]
        other = "female" if gender == "male" else "male"
        # Language-flavoured voices ("breathy Portuguese vocal") would be guesses; leave language to the user.
        fitting = [v for v in VOCABULARY["vocal"] if other not in v.lower().split()
                   and not any(lang.lower() in v.lower() for lang in prompt_tags.LANGUAGES + ["Portuguese"])]
        ranked = _rank(voice, "vocal", fitting)
        best = ranked[0][0]
        if gender not in best.lower().split():
            best = best.replace("vocal", f"{gender} vocal", 1) if "vocal" in best else f"{best}, {gender} vocal"
        result["vocal"] = [best]
        result["more"]["vocal"] = [tag for tag, _p in ranked[1:4]]

    result["key"] = key
    result["bpm"] = estimate_tempo(mono, rate)
    result["instrumental"] = instrumental
    result["style"] = style_line(result, language)
    return result


def style_line(result: Dict[str, Any], language: str = "English") -> str:
    """The tags in YuE2's order: language, genre, vocal, instruments, mood, production, key, tempo.

    Instrumentals get no language and no "no vocals" wording; the studio's
    Instrumental switch adds that.
    """
    tags: List[str] = []
    if not result.get("instrumental") and language:
        tags.append(language)
    tags += result.get("genre", []) + result.get("vocal", []) + result.get("instruments", [])
    tags += result.get("mood", []) + result.get("production", [])
    if result.get("key"):
        tags.append(result["key"])
    if result.get("bpm"):
        tags.append(f"{result['bpm']} BPM")
    return ", ".join(tags)


if __name__ == "__main__":
    import json
    import sys
    import time

    for name in sys.argv[1:]:
        started = time.time()
        print(name)
        print(json.dumps(interpret(name), indent=1))
        print(f"({time.time() - started:.1f}s)")
