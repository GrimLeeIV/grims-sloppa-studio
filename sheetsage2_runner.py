"""SheetSage2 transcription, run inside the Upload add-on's own environment.

SheetSage2 pins PyTorch 2.8 / Transformers 4.45 / NumPy 1.24, which conflict
with the studio's own stack, so it lives in ``addons/sheetsage2/.venv`` and the
studio runs this script there (see ``upload_addon.transcribe``). The two
processes exchange files only.

Audio is decoded here with soundfile and handed over as an array, which skips
SheetSage2's FFmpeg-based file reader entirely: no FFmpeg install is needed.

    python sheetsage2_runner.py --model <dir> --prefetch
    python sheetsage2_runner.py --model <dir> song.mp3 --output <dir> [--max-seconds N]

Writes ``studio_result.json`` into ``--output``:
``{"ok", "abc", "abc_error", "warnings", "seconds", "peak_gpu_mib"}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="?", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--prefetch", action="store_true",
                        help="Load the model once (fetching its MERT-v2 parent) and exit")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True).eval().to(device)
    if args.prefetch:
        print(json.dumps({"ok": True, "device": device}), flush=True)
        return 0
    if args.audio is None or args.output is None:
        parser.error("audio and --output are required unless --prefetch is given")

    import soundfile as sf

    data, rate = sf.read(str(args.audio), always_2d=True, dtype="float32")
    seconds = len(data) / rate
    if args.max_seconds:
        seconds = min(seconds, args.max_seconds)
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {"ok": False, "abc": None, "abc_error": None, "warnings": [], "seconds": seconds}
    try:
        result = model.transcribe(
            data.T,  # SheetSage2 wants [channels, samples]
            str(args.output),
            sampling_rate=rate,
            dtype="bf16" if device == "cuda" else "fp32",
            max_seconds=args.max_seconds,
            melody_only=True,
        )
    except RuntimeError as error:
        # A melody-only ABC that cannot be built raises with the partial result attached.
        partial = getattr(error, "result", None) or {}
        summary.update(abc_error=partial.get("abc_error") or str(error), warnings=partial.get("warnings", []))
    else:
        summary.update(
            ok=bool(result.get("abc")),
            abc=result.get("abc"),
            abc_error=result.get("abc_error"),
            warnings=list(result.get("warnings") or []),
        )
    if device == "cuda":
        summary["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
    (args.output / "studio_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
