# Grim's Sloppa Studio

A desktop music studio for [YuE2](https://github.com/multimodal-art-projection/YuE): write a style and lyrics,
generate songs, then extend them, split them into stems, or turn your own recording into a new performance.
Drag across a track's waveform to redo just that part. Drop any song on **Interpret** to get Style tags (genre, voice, instruments, mood, key, tempo) that describe it.
Runs locally on an NVIDIA GPU, with an optional phone remote over your Wi-Fi.

## Install

1. Download **Install Grim's Sloppa Studio.bat** and put it in an empty folder.
2. Double-click it. It unpacks the studio, installs Python if needed, then downloads PyTorch and the YuE2 model
   (about 15 GB in all, the first time only).
3. From then on, start the studio with **Grim's Sloppa Studio.bat**.

The folder ends up as:

```
Core Files/                 the studio, its Python environment and models
Outputs/                    your music, in a folder per day
Grim's Sloppa Studio.bat    starts the studio
```

**Needs:** Windows 10 or 11 and an NVIDIA GPU with 12 GB or more of graphics memory (16 GB or more recommended).
With several GPUs, batches are shared between them.

## From source

Clone this repository into a folder named `Core Files`, then run `run_yue_studio.bat` inside it (or use the
installer, built with `python build_installer.py`).

## Credits

Built by Claude (Anthropic), directed by Grim Lee. Music model: YuE2 by M-A-P and HKUST. Built on Gradio,
PyTorch, Demucs, SheetSage2, CLAP (LAION), pywebview, Pillow and qrcode. See the studio's Info panel for licences.

## Licence

The studio's own code is MIT licensed (see `LICENSE`). The models and libraries it downloads keep their own
licences: the YuE2 weights in particular are fetched from their publisher and are not part of this repository.
