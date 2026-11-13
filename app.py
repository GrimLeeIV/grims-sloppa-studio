"""Grim's Sloppa Studio - Gradio UI for the YuE2 ComfyUI workflow."""

from __future__ import annotations

import json
import re
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, List

import studio_shell

studio_shell.capture_output()  # the Console panel shows everything printed from here on

import gradio as gr

import prompt_tags
import stem_splitter
import run_timing
import system_check
import track_canvas
import audio_interpreter
import ui_theme
import upload_addon
from yue_native_engine import EXTEND_MAX_SECONDS, EXTEND_MIN_SECONDS, YuE2Engine, prefetch_weights

MOCK_MODE = os.environ.get("YUE_MOCK", "0") == "1"
OUTPUT_DIR = "./Outputs"
# Default model source for the header's Switch/Default Model buttons. The engine
# also hard-codes these repos, so the studio stays plug-and-play when untouched.
# The matching folder textbox is hidden: the UI stays clean, and the buttons
# drive the value behind the scenes.
DEFAULT_MODEL_SOURCE = "m-a-p/YuE2-3B"
MAX_TRACKS = 4
ENGINE = None if MOCK_MODE else YuE2Engine()
MIN_GPU_GIB = 11  # a further card needs room for its own copy of YuE2


def _gpu_engines() -> List[Any]:
    """One engine per GPU that can hold YuE2; the first is ENGINE. Only batches use the others."""
    if ENGINE is None:
        return []
    engines = [ENGINE]
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for index in range(1, count):
            props = torch.cuda.get_device_properties(index)
            if props.total_memory / 2 ** 30 < MIN_GPU_GIB:
                print(f"[Studio] Skipping GPU {index} ({props.name}): under {MIN_GPU_GIB} GB for its own copy of YuE2.")
                continue
            engines.append(YuE2Engine(device=f"cuda:{index}"))
    except Exception as exc:  # a broken second card must never stop the studio
        print(f"[Studio] Extra GPUs unavailable: {exc}")
    if len(engines) > 1:
        print(f"[Studio] {len(engines)} GPUs: batches are shared between them (each loads its own copy of YuE2).")
    return engines


GPU_ENGINES = _gpu_engines()

PROJECT_ROOT = Path(__file__).resolve().parent
# Installed layout: the studio's files in "Core Files", the music beside it in "Outputs".
STUDIO_HOME = PROJECT_ROOT.parent if PROJECT_ROOT.name == "Core Files" else PROJECT_ROOT
LOCAL_OUTPUT_DIR = str(STUDIO_HOME / "Outputs")
TEMP_DIR = PROJECT_ROOT / ".temp"

# Session memory + JSON presets. Both share one ordered schema so a preset can
# round-trip every UI field, while the auto-saved session restores the last run.
SESSION_FILE = PROJECT_ROOT / ".last_session.json"
PRESET_FILE = TEMP_DIR / "grim_sloppa_preset.json"
SESSION_FIELDS = [
    "style",
    "lyrics",
    "duration",
    "batch_count",
    "seed",
    "quality",
    "model_path",
    "filename_prefix",
    "output_dir",
    "no_intro",
    "no_outro",
]
SESSION_DEFAULTS: dict = {
    "no_intro": False,
    "no_outro": False,
    "style": "",
    "lyrics": "",
    "duration": 30,
    "batch_count": 1,
    "seed": -1,
    "quality": 50,
    "model_path": DEFAULT_MODEL_SOURCE,
    "filename_prefix": "GrimSloppa",
    "output_dir": "./Outputs",
}



def _ensure_dir(path: str) -> str:
    path = (path or "").strip()
    if not path or not os.path.isabs(path):
        path = LOCAL_OUTPUT_DIR  # ./Outputs: beside Core Files (or in the project folder)
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        path = LOCAL_OUTPUT_DIR
        os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


# Create project-root working folders on launch.
TEMP_DIR.mkdir(parents=True, exist_ok=True)
_ensure_dir(OUTPUT_DIR)


def _mock_audio_path(index: int, duration: float, output_dir: str) -> str:
    """Mock tracks live in the hidden .temp folder, not the user's Outputs."""
    import random
    import wave

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    path = os.path.join(str(TEMP_DIR), f"mock_track_{index}_{random.randint(1000, 9999)}.wav")
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * int(min(float(duration), 4.0) * 22050))
    return os.path.abspath(path)


def _uploaded_path(value: Any) -> str:
    """Absolute path of a file chosen in the native OS file browser.

    Gradio hands an upload over either as a plain path string (``filepath``
    mode) or as a file object; either way the pick has been copied into
    Gradio's temporary upload folder, so ``name`` is the server-side absolute
    path. Both shapes are accepted so the button keeps working if the component
    type changes.
    """
    if value is None:
        return ""
    if isinstance(value, (str, os.PathLike)):
        return str(value)
    if isinstance(value, dict):
        return str(value.get("path") or value.get("name") or "")
    name = getattr(value, "name", None)
    if name:
        return str(name)
    path = getattr(value, "path", None)
    return str(path) if path else ""


def use_uploaded_model(value: Any) -> tuple[Any, str, Any]:
    """Load the checkpoint chosen in the native file browser.

    Returns ``(model_state, status, switch_button)``. The button's own value is
    cleared so picking the *same* file twice still fires the handler.
    """
    raw = _uploaded_path(value)
    if not raw:
        return gr.update(), "No file selected; keeping the current model.", gr.update()
    if ENGINE is None:
        return gr.update(), f"Engine unavailable (mock mode): `{raw}`", gr.update(value=None)
    try:
        source = ENGINE.set_model_checkpoint(raw)
    except Exception as exc:
        print(f"[Grim's Sloppa Studio] Could not switch model: {exc}")
        return gr.update(), f"Model switch failed: {exc}", gr.update(value=None)
    return source, f"Model `{source}`", gr.update(value=None)


def reset_model_source() -> tuple[Any, str]:
    """Point the engine back at the packaged repo (the Default Model button)."""
    if ENGINE is None:
        return DEFAULT_MODEL_SOURCE, f"Model `{DEFAULT_MODEL_SOURCE}`"
    try:
        source = ENGINE.reset_model()
    except Exception as exc:
        print(f"[Grim's Sloppa Studio] Could not reset model: {exc}")
        return DEFAULT_MODEL_SOURCE, f"Model reset failed: {exc}"
    return source, f"Model `{source}`"


def _remembered_model_source(value: Any) -> str:
    """A remembered model path is only reused while it still exists.

    A checkpoint chosen with the Load Model button lands in Gradio's temporary
    upload folder, which is wiped between runs, so the last session's path can
    be gone by the next launch. A vanished local path falls back to the packaged
    repo instead of being handed to the engine as a broken source.
    """
    raw = str(value or "").strip()
    if not raw:
        return DEFAULT_MODEL_SOURCE
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return raw
    if candidate.is_absolute() or raw.startswith((".", "~", "\\", "/")):
        print(f"[Grim's Sloppa Studio] Remembered model '{raw}' is gone; using {DEFAULT_MODEL_SOURCE}.")
        return DEFAULT_MODEL_SOURCE
    return raw


def _normalize_session(data: Any) -> dict:
    """Merge raw JSON/preset data onto the defaults, keeping a stable schema."""
    payload = dict(SESSION_DEFAULTS)
    if isinstance(data, dict):
        for key in SESSION_FIELDS:
            value = data.get(key)
            if value is not None:
                payload[key] = value
    return payload


_SESSION_WRITE_LOCK = threading.Lock()


def _write_session_file(path: Path, payload: dict) -> None:
    """Write JSON straight to the destination.

    OneDrive's real-time indexer can hold a transient handle on a
    just-created temp file, so the atomic ``os.replace`` dance fails with a
    Windows ``[WinError 5] Access is denied``. Writing directly to the target
    sidesteps the rename entirely. The lock still serialises the concurrent
    change handlers so two writers cannot interleave mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SESSION_WRITE_LOCK:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def save_session(**values: Any) -> None:
    """Persist the current UI values so they return on the next launch."""
    if MOCK_MODE:  # a no-model test run must never change the real session
        return
    try:
        _write_session_file(SESSION_FILE, _normalize_session(values))
    except Exception as exc:  # never let persistence break generation
        print(f"[Grim's Sloppa Studio] Session save failed: {exc}")


def load_session() -> dict:
    """Read the last saved session, falling back to defaults."""
    if SESSION_FILE.is_file():
        try:
            return _normalize_session(json.loads(SESSION_FILE.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[Grim's Sloppa Studio] Session load failed: {exc}")
    return dict(SESSION_DEFAULTS)


def startup_session() -> List[Any]:
    """Value list used to hydrate the UI from the last session on launch.

    The final element is the freshly written preset-file path, which pre-arms
    the Save Preset download so a single click serves the restored settings.
    """
    payload = load_session()
    values = [payload[key] for key in SESSION_FIELDS]
    # Heal a remembered model path that no longer exists (temp uploads vanish).
    values[SESSION_FIELDS.index("model_path")] = _remembered_model_source(
        payload.get("model_path")
    )
    try:
        values.append(save_preset(*values))
    except Exception as exc:  # pragma: no cover - persistence is best-effort
        print(f"[Grim's Sloppa Studio] Could not prime preset file: {exc}")
        values.append(None)
    return values


def save_preset(
    style: str,
    lyrics: str,
    duration: float,
    batch_count: float,
    seed: float,
    quality: float,
    model_path: str,
    filename_prefix: str,
    output_dir: str,
    no_intro: bool = False,
    no_outro: bool = False,
) -> str:
    """Write every UI field to a downloadable JSON file and return its path."""
    payload = _normalize_session(
        {
            "style": style,
            "lyrics": lyrics,
            "duration": duration,
            "batch_count": batch_count,
            "seed": seed,
            "quality": quality,
            "model_path": model_path,
            "filename_prefix": filename_prefix,
            "output_dir": output_dir,
            "no_intro": bool(no_intro),
            "no_outro": bool(no_outro),
        }
    )
    try:
        _write_session_file(PRESET_FILE, payload)
    except Exception as exc:  # a failed refresh must never break the UI
        print(f"[Grim's Sloppa Studio] Preset save failed: {exc}")
    return str(PRESET_FILE)


def load_preset(file_path: str) -> List[Any]:
    """Populate every UI field from an uploaded preset JSON file."""
    if not file_path:
        return [gr.update() for _ in SESSION_FIELDS]
    try:
        payload = _normalize_session(json.loads(Path(file_path).read_text(encoding="utf-8")))
    except Exception as exc:
        print(f"[Grim's Sloppa Studio] Preset load failed: {exc}")
        return [gr.update() for _ in SESSION_FIELDS]
    return [payload[key] for key in SESSION_FIELDS]


def _clamp(value: Any, low: float, high: float, integer: bool = False) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    number = max(low, min(number, high))
    return int(number) if integer else number


def _clock(seconds: int) -> str:
    """Elapsed time as m:ss."""
    return f"{seconds // 60}:{seconds % 60:02d}"


def _friendly_error(error: str) -> str:
    """One readable line for a failed run (the full traceback goes to the console)."""
    if "out of memory" in error.lower():
        return "Ran out of GPU memory. Try a shorter duration or close other GPU apps."
    first_line = error.strip().splitlines()[0] if error.strip() else "Unknown error"
    return first_line[:300]


def _run_job(job: Any, waiting: Any, finished: Any):
    """Run ``job()`` on a worker thread for a generator handler.

    Yields ``waiting(elapsed)`` once a second while it runs, then
    ``finished(result, error)``. Errors are printed to the console in full and
    handed to ``finished`` as text.
    """
    state: dict = {"result": None, "error": None}

    def worker() -> None:
        try:
            state["result"] = job()
        except Exception as exc:
            traceback.print_exc()
            state["error"] = str(exc) or type(exc).__name__

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    started = time.time()
    while worker_thread.is_alive():
        yield waiting(int(time.time() - started))
        time.sleep(1)
    yield finished(state["result"], state["error"])


def _notice(nodes: List[dict], text: str) -> Any:
    """Float a message over the canvas without moving the view."""
    return track_canvas.view(nodes, notice=text)


# Style words YuE2 tends to answer with a voice (a hum, a choir pad, a chant), even in instrumentals.
VOICE_HINTS = ("choir", "choral", "vocal", "voice", "hum", "chant", "sing", "aah", "ooh", "ethereal", "angelic",
               "haunting", "gospel", "opera", "whisper", "spoken", "rap", "a cappella", "acapella", "siren", "monk")


def _voice_words(style: str) -> List[str]:
    """Style tags that may have invited a voice; the "no vocals"-style conditions don't count."""
    found = []
    for tag in prompt_tags.split_tags(style or ""):
        low = tag.lower().strip(" .")
        if low.startswith(("no ", "without ", "non-")) or not low:
            continue
        if any(re.search(rf"\b{re.escape(word)}", low) for word in VOICE_HINTS):
            found.append(tag.strip(" ."))
    return found


def _guard_notice(notes: List[str], style: str = "") -> str | None:
    """Instrumental runs: say where the vocal guard couldn't remove singing, and what may have asked for it."""
    if not notes:
        return None
    text = "Instrumental check: " + "; ".join(notes) + "."
    words = _voice_words(style)
    if words:
        quoted = ", ".join(f"\"{w}\"" for w in words[:3])
        text += (f" Your style has {quoted}, which YuE2 often plays with a wordless voice (a hum or choir). "
                 "Keep it if you like that; remove it for a strictly instrumental take.")
    else:
        text += " Give it a listen, or regenerate."
    return text


def _cold() -> bool:
    """The model isn't loaded yet, so the next run starts by loading it."""
    return ENGINE is not None and not ENGINE.is_loaded


def _timed(kind: str, predicted: float) -> Any:
    """Decorate a job so its real duration teaches the estimates (real runs only)."""
    def wrap(job: Any) -> Any:
        def run() -> Any:
            started = time.time()
            result = job()
            if not MOCK_MODE:
                run_timing.record(kind, predicted, time.time() - started)
            return result
        return run
    return wrap


# ---------------------------------------------------------------------------
# Track canvas: this session's tree (a run's tracks + their branches)
# ---------------------------------------------------------------------------
LAST_TRACKS_FILE = TEMP_DIR / "last_tracks.json"


def _remember_tree(nodes: List[dict]) -> None:
    """Keep the current tree (minus anything still rendering) across restarts."""
    if MOCK_MODE:
        return
    try:
        _write_session_file(LAST_TRACKS_FILE, {"nodes": [n for n in nodes if n.get("kind") != "pending"]})
    except Exception as exc:  # best effort
        print(f"[Grim's Sloppa Studio] Could not remember the tracks: {exc}")


def _dated_dir(output_dir: str) -> str:
    """Today's subfolder of the output folder (generations are kept by day)."""
    import datetime

    path = os.path.join(_ensure_dir(output_dir), datetime.date.today().isoformat())
    os.makedirs(path, exist_ok=True)
    return path


def _stamped(prefix: str) -> str:
    """The file prefix with the time of day, e.g. GrimSloppa_333AM (files then end _seed_n)."""
    import datetime

    return f"{prefix or 'GrimSloppa'}_{datetime.datetime.now().strftime('%I%M%p').lstrip('0')}"


def _new_track(path: str, parent: str | None = None, sub: str = "") -> dict:
    return track_canvas.track_node(path, parent, sub=sub, extendable=YuE2Engine.can_extend(path))


def load_tree() -> List[Any]:
    """The remembered tree; older saves (a plain file list) become roots."""
    try:
        saved = json.loads(LAST_TRACKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        saved = {}
    nodes = [n for n in saved.get("nodes", []) if os.path.exists(n.get("file", ""))]
    ids = {n["id"] for n in nodes}
    by_id = {n["id"]: n for n in nodes}
    for node in nodes:
        node.setdefault("created", track_canvas._made_at(node["file"]))
        if str(node.get("sub", "")).startswith("Take "):  # saved before "takes" became "runs"
            node["sub"] = "Run " + node["sub"][5:]
    for node in nodes:
        if node.get("parent") not in ids:
            node["parent"] = None
    for node in nodes:
        # Extensions saved before seam marks existed: the seam is where the parent ends (or starts).
        parent = by_id.get(node.get("parent"))
        if "marks" not in node and parent and node.get("kind") == "track":
            if "after" in node.get("sub", ""):
                node["marks"] = _seams(parent, "after", float(parent.get("dur") or 0))
            elif "before" in node.get("sub", ""):
                node["marks"] = _seams(parent, "before", float(node.get("dur") or 0) - float(parent.get("dur") or 0))
    if not nodes:
        nodes = [_new_track(path) for path in saved.get("files", []) if os.path.exists(path)]
    with _SHARED_LOCK:
        # Another screen's tree (and anything it is rendering right now) wins over the saved one.
        if SHARED["tree"] is None:
            SHARED["tree"] = nodes
        nodes = list(SHARED["tree"])
    return [nodes, track_canvas.view(nodes, fit=True, notice=" ".join(system_check.WARNINGS))]


def _event(evt: gr.EventData) -> dict:
    data = getattr(evt, "_data", None)
    return data if isinstance(data, dict) else {}


def open_in_folder(path: str | None) -> None:
    """Show a file in Explorer (selected), or its folder elsewhere."""
    if not path or not os.path.exists(path):
        return
    if os.name == "nt":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)])


def _new_run(label: str, count: int, each: float, first: float = 0.0, gpus: int = 1) -> List[dict]:
    """Pending root cards for a fresh run (Generate / Redux); each GPU renders them one after another."""
    return [track_canvas.pending_node(None, f"Track {i + 1}" if count > 1 else label,
                                      eta=run_timing.shown(first + each * (i // max(1, gpus) + 1)))
            for i in range(count)]


def _across_gpus(engines: List[Any], count: int, seed: int, track_done: Any, **song: Any) -> List[tuple]:
    """Render a batch on several GPUs: each takes the next track as soon as it is free.

    Returns (file, seconds it took, vocal-guard notes) per track, in batch order.
    """
    import queue
    import random

    base_seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**53 - 1 - MAX_TRACKS)
    todo: Any = queue.Queue()
    for index in range(count):
        todo.put(index)
    results: dict = {}
    errors: List[Exception] = []
    lock = threading.Lock()

    def work(engine: Any) -> None:
        while not errors:
            try:
                index = todo.get_nowait()
            except queue.Empty:
                return
            started = time.time()
            try:
                out = engine.generate_song(seed=base_seed, batch_count=1, first_index=index, **song)
            except Exception as exc:
                traceback.print_exc()
                errors.append(exc)
                return
            with lock:
                results[index] = (out["files"][0], time.time() - started, list(engine.guard_notes))
                track_done(len(results), count)

    workers = [threading.Thread(target=work, args=(engine,), daemon=True) for engine in engines[:count]]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if errors and not results:
        raise errors[0]
    return [results[index] for index in sorted(results)]


def _holding(other_idle: str, gen: Any):
    """Pass a run's updates through, keeping the other run button greyed out until it ends."""
    for out in gen:
        yield out + [gr.Button(interactive=False)]
    yield [gr.update()] * 3 + [gr.Button(value=other_idle, interactive=True)]


def generate_music(
    style: str,
    lyrics: str,
    duration: float,
    batch_count: int,
    seed: int,
    quality: float = 100,
    model_path: str = DEFAULT_MODEL_SOURCE,
    filename_prefix: str = "GrimSloppa",
    output_dir: str = OUTPUT_DIR,
    no_intro: bool = False,
    no_outro: bool = False,
    tree: List[dict] | None = None,
):
    """Generator for [Generate]: a fresh tree, its cards rendering in place."""
    # Remember exactly what the user submitted so the UI restores it next launch.
    save_session(
        style=style,
        lyrics=lyrics,
        duration=duration,
        batch_count=batch_count,
        seed=seed,
        quality=quality,
        model_path=model_path,
        filename_prefix=filename_prefix,
        output_dir=output_dir,
        no_intro=no_intro,
        no_outro=no_outro,
    )
    safe_duration = float(_clamp(duration, 5, 300))
    safe_quality = float(_clamp(quality, 10, 100))
    active = int(_clamp(batch_count, 1, MAX_TRACKS, integer=True))
    progress = {"done": 0}
    notes: List[str] = []
    cold = _cold()
    each = run_timing.estimate("generate", safe_duration, instrumental=prompt_tags.is_instrumental(lyrics))
    first = run_timing.estimate("generate", 0, cold=True) - run_timing.estimate("generate", 0) if cold else 0.0
    pending = _new_run("Track", active, each, first, gpus=min(active, len(GPU_ENGINES) or 1))

    stamps: List[float] = []
    took_by_file: dict = {}

    def track_done(done: int, _total: int) -> None:
        progress.update(done=done)
        stamps.append(time.time())

    @_timed("generate", first + each * active)
    def job() -> List[str]:
        stamps.append(time.time())
        out_dir = _dated_dir(output_dir)
        if MOCK_MODE:
            files = []
            for i in range(active):
                time.sleep(0.8)
                track_done(i + 1, active)
                files.append(_mock_audio_path(i, safe_duration, out_dir))
            return files
        if len(GPU_ENGINES) > 1 and active > 1:
            made = _across_gpus(
                GPU_ENGINES, active, seed, track_done, prompt=style, lyrics=lyrics, duration=safe_duration,
                quality=safe_quality, output_dir=out_dir, model_path=model_path,
                filename_prefix=_stamped(filename_prefix), no_intro=bool(no_intro), no_outro=bool(no_outro),
            )
            for path, took, guard in made:
                took_by_file[path] = took
                notes.extend(guard)
            return [path for path, _took, _guard in made]
        result = ENGINE.generate_song(
            prompt=style,
            lyrics=lyrics,
            duration=safe_duration,
            seed=seed,
            quality=safe_quality,
            batch_count=active,
            output_dir=out_dir,
            model_path=model_path,
            filename_prefix=_stamped(filename_prefix),
            progress_callback=track_done,
            no_intro=bool(no_intro),
            no_outro=bool(no_outro),
        )
        notes.extend(ENGINE.guard_notes)
        return list(result["files"])

    def waiting(elapsed: int) -> List[Any]:
        done = f" · {progress['done']}/{active}" if active > 1 and progress["done"] else ""
        canvas = track_canvas.view(pending, fit=True, notice="") if elapsed == 0 else gr.update()
        return [pending, canvas, gr.Button(value=f"Generating{done}… {_clock(elapsed)}", interactive=False)]

    def finished(files: List[str] | None, error: str | None) -> List[Any]:
        idle = gr.Button(value="Generate", interactive=True)
        if error or not files:
            nodes = tree or []
            return [nodes, track_canvas.view(nodes, fit=True, notice=f"Generate failed: "
                                             f"{_friendly_error(error or 'no tracks came out')}"), idle]
        nodes = [_new_track(path, sub="Generated") for path in files]
        for i, node in enumerate(nodes):
            if node["file"] in took_by_file:
                node["took"] = round(took_by_file[node["file"]])
            elif i + 1 < len(stamps):
                node["took"] = round(stamps[i + 1] - stamps[i])
        _remember_tree(nodes)
        return [nodes, track_canvas.view(nodes, fit=True, notice=_guard_notice(notes, style) or ""), idle]

    yield from _run_job(job, waiting, finished)


EXTEND_BEFORE = "before"


def _track_is_instrumental(saved: dict) -> bool:
    return "no vocals" in str(saved.get("style", "")).lower() or prompt_tags.is_instrumental(saved.get("lyrics", ""))


def _new_section_lyrics(box: str, saved: str) -> str:
    """Lyrics for an extension: what was added below the track's own lyrics.

    A box that still holds just the track's lyrics (or [instrumental]) adds
    nothing; a box with completely different words is used as-is.
    """
    box, saved = (box or "").strip(), (saved or "").strip()
    if not box or prompt_tags.is_instrumental(box) or box == saved:
        return ""
    if saved and box.startswith(saved):
        return box[len(saved):].strip()
    return box


def canvas_quick(tree: List[dict], evt: gr.EventData, request: gr.Request = None) -> tuple:
    """Unqueued card actions: open Extend (load the track's lyrics), show in folder.

    Outputs: lyrics box, Instrumental switch.
    """
    data = _event(evt)
    node = track_canvas.find(tree or [], str(data.get("id")))
    if not node:
        return gr.update(), gr.update()
    if data.get("action") == "folder":
        if studio_shell.is_local(request):  # a phone can't see the rig's Explorer
            open_in_folder(node.get("file"))
        return gr.update(), gr.update()
    if data.get("action") == "open_extend" and node.get("extendable"):
        saved = YuE2Engine.track_request(node["file"])
        if _track_is_instrumental(saved):
            return gr.update(value="[instrumental]", interactive=False), True
        return gr.update(value=saved.get("lyrics", ""), interactive=True), False
    return gr.update(), gr.update()


def canvas_render(tree: List[dict], lyrics: str, seed: float, quality: float, no_outro: bool, style: str,
                  evt: gr.EventData, request: gr.Request = None):
    """Queued card actions: Extend or Redo (runs branch off the track), or Stems (stem cards).

    Outputs: tree, canvas, Generate and REDUX buttons (busy while the GPU renders this).
    """
    def run():
        started = time.time()
        for out in _canvas_render(tree, lyrics, seed, quality, no_outro, style, _event(evt)):
            busy = gr.Button(value=f"Rendering… {_clock(int(time.time() - started))}", interactive=False)
            yield out + [busy, gr.Button(interactive=False)]
        yield [gr.update(), gr.update(), gr.Button(value="Generate", interactive=True),
               gr.Button(value="REDUX", interactive=True)]

    yield from _shared_run("render", run(), tree, request, [gr.update(), gr.update()])


def _canvas_render(tree: List[dict], lyrics: str, seed: float, quality: float, no_outro: bool, style: str,
                   data: dict):
    tree = list(tree or [])
    node = track_canvas.find(tree, str(data.get("id")))
    if not node:
        yield [tree, _notice(tree, "That track is no longer on the canvas.")]
        return
    if data.get("action") == "stems":
        yield from _render_stems(tree, node, str(data.get("mode") or stem_splitter.DEFAULT_STEM_MODE))
    elif data.get("action") == "extend":
        yield from _render_extend(tree, node, data, lyrics, seed, quality, bool(no_outro), style)
    elif data.get("action") == "redo":
        yield from _render_redo(tree, node, data, seed, quality)


def _seams(node: dict, direction: str, seam: float) -> List[float]:
    """Where new music begins in an extension: the parent's seams, shifted, plus this one."""
    old = [float(m) for m in node.get("marks") or []]
    if direction == "before":
        return [seam] + [m + seam for m in old]
    return [m for m in old if m < seam - 0.5] + [seam]


def _render_extend(tree: List[dict], node: dict, data: dict, lyrics: str, seed: float, quality: float,
                   no_outro: bool = False, style: str = ""):
    name = node["label"]
    if not node.get("extendable"):
        yield [tree, _notice(tree, f"{name} has no saved song tokens (made before Extend existed), "
                                   "so it can't be extended. Stems still work.")]
        return
    direction = "before" if data.get("direction") == EXTEND_BEFORE else "after"
    seconds = float(_clamp(data.get("seconds"), EXTEND_MIN_SECONDS, EXTEND_MAX_SECONDS))
    takes = int(_clamp(data.get("takes"), 1, MAX_TRACKS, integer=True))
    base_seed = int(_clamp(seed, -1, 2**53 - 1, integer=True))
    if base_seed < 0:
        import random

        base_seed = random.randint(0, 2**53 - 1 - MAX_TRACKS)
    section = _new_section_lyrics(lyrics, YuE2Engine.track_request(node["file"]).get("lyrics", ""))
    kind = "Intro" if direction == "before" else "Extension"
    saved = YuE2Engine.track_request(node["file"]) if not MOCK_MODE else {}
    # New music is sampled; the existing track is only re-rendered (much faster per second).
    each = run_timing.estimate("extend", seconds + 0.15 * float(node.get("dur") or 0),
                               instrumental=_track_is_instrumental(saved), cold=_cold())
    pending = [track_canvas.pending_node(node["id"], f"Run {i + 1}", f"{kind} · rendering",
                                         eta=run_timing.shown(each * (i + 1))) for i in range(takes)]
    current = tree + pending
    progress = {"take": 1}
    notes: List[str] = []

    @_timed("extend", each * takes)
    def job() -> List[tuple]:
        results = []
        for take in range(takes):
            progress["take"] = take + 1
            if MOCK_MODE:
                time.sleep(1)
                results.append((_mock_audio_path(take, seconds, ""), seconds, 1.0, 1.0))
                continue
            started = time.time()
            result = ENGINE.extend_track(node["file"], seconds, direction=direction, lyrics=section,
                                         seed=base_seed + take, quality=float(_clamp(quality, 10, 100)),
                                         no_outro=no_outro)
            results.append((result["file"], result["seconds_added"], result["seam"], time.time() - started))
            notes.extend(f"run {take + 1}: {note}" for note in ENGINE.guard_notes)
        return results

    def waiting(elapsed: int) -> List[Any]:
        canvas = track_canvas.view(current, focus=pending[0]["id"], notice="") if elapsed == 0 else gr.update()
        return [current, canvas]

    def finished(results: List[tuple] | None, error: str | None) -> List[Any]:
        if error or not results:
            return [tree, track_canvas.view(tree, focus=node["id"], notice=f"Extend failed: "
                                            f"{_friendly_error(error or 'no runs came out')}")]
        where = "before" if direction == "before" else "after"
        added = []
        for i, (path, secs, seam, took) in enumerate(results):
            child = _new_track(path, node["id"], sub=f"Run {i + 1} · +{secs:.0f}s {where}")
            child["marks"] = _seams(node, direction, seam)
            child["took"] = round(took)
            added.append(child)
        nodes = tree + added
        _remember_tree(nodes)
        track_style = YuE2Engine.track_request(node["file"]).get("style", "") if not MOCK_MODE else style
        return [nodes, track_canvas.view(nodes, focus=added[0]["id"], notice=_guard_notice(notes, track_style) or "")]

    yield from _run_job(job, waiting, finished)


def _span(start: float, end: float) -> str:
    return f"{int(start) // 60}:{int(start) % 60:02d}–{int(end) // 60}:{int(end) % 60:02d}"


def _render_redo(tree: List[dict], node: dict, data: dict, seed: float, quality: float):
    """A selection on a card's waveform: new takes of that stretch, branching off the track."""
    name = node["label"]
    if not node.get("extendable"):
        yield [tree, _notice(tree, f"{name} has no saved song tokens (made before Extend existed), "
                                   "so part of it can't be redone.")]
        return
    duration = float(node.get("dur") or 0)
    start = max(0.0, float(_clamp(data.get("start"), 0, duration)))
    end = float(_clamp(data.get("end"), start, duration))
    to_end = end >= duration - 0.5
    if end - start < 2:
        yield [tree, _notice(tree, "Select at least 2 seconds of the waveform to redo.")]
        return
    takes = int(_clamp(data.get("takes"), 1, MAX_TRACKS, integer=True))
    base_seed = int(_clamp(seed, -1, 2**53 - 1, integer=True))
    if base_seed < 0:
        import random

        base_seed = random.randint(0, 2**53 - 1 - MAX_TRACKS)
    what = f"from {_span(start, end).split('–')[0]}" if to_end else _span(start, end)
    saved = YuE2Engine.track_request(node["file"]) if not MOCK_MODE else {}
    # Same work as an extension: the stretch is sampled, the rest only re-rendered.
    each = run_timing.estimate("extend", (end - start) + 0.15 * duration,
                               instrumental=_track_is_instrumental(saved), cold=_cold())
    pending = [track_canvas.pending_node(node["id"], f"Run {i + 1}", f"Redo {what} · rendering",
                                         eta=run_timing.shown(each * (i + 1))) for i in range(takes)]
    current = tree + pending
    notes: List[str] = []

    @_timed("extend", each * takes)
    def job() -> List[tuple]:
        results = []
        for take in range(takes):
            if MOCK_MODE:
                time.sleep(1)
                results.append((_mock_audio_path(take, duration, ""), start, end, 1.0))
                continue
            started = time.time()
            result = ENGINE.redo_section(node["file"], start, None if to_end else end, seed=base_seed + take,
                                         quality=float(_clamp(quality, 10, 100)))
            results.append((result["file"], result["start"], result["end"], time.time() - started))
            notes.extend(f"run {take + 1}: {note}" for note in ENGINE.guard_notes)
        return results

    def waiting(elapsed: int) -> List[Any]:
        canvas = track_canvas.view(current, focus=pending[0]["id"], notice="") if elapsed == 0 else gr.update()
        return [current, canvas]

    def finished(results: List[tuple] | None, error: str | None) -> List[Any]:
        if error or not results:
            return [tree, track_canvas.view(tree, focus=node["id"], notice=f"Redo failed: "
                                            f"{_friendly_error(error or 'no runs came out')}")]
        added = []
        for i, (path, a, b, took) in enumerate(results):
            child = _new_track(path, node["id"], sub=f"Run {i + 1} · redo {what}")
            # The parent's seams outside the redone stretch still hold; mark where the new take sits.
            kept = [float(m) for m in node.get("marks") or [] if not a - 0.5 < float(m) < b + 0.5]
            child["marks"] = sorted(kept + [a] + ([] if to_end else [b]))
            child["took"] = round(took)
            added.append(child)
        nodes = tree + added
        _remember_tree(nodes)
        track_style = saved.get("style", "") if saved else ""
        return [nodes, track_canvas.view(nodes, focus=added[0]["id"], notice=_guard_notice(notes, track_style) or "")]

    yield from _run_job(job, waiting, finished)


def _render_stems(tree: List[dict], node: dict, mode: str):
    if not (stem_splitter.available() or MOCK_MODE):
        yield [tree, _notice(tree, "Stem splitting needs Demucs. Restart with Grim's Sloppa Studio.bat "
                                   "to install it automatically.")]
        return
    each = run_timing.estimate("stems", float(node.get("dur") or 0))
    pending = [track_canvas.pending_node(node["id"], "Stems", f"{mode} · splitting", eta=run_timing.shown(each))]
    current = tree + pending

    started = time.time()

    @_timed("stems", each)
    def job() -> dict:
        if MOCK_MODE:
            time.sleep(2)
            return {"files": {stem: _mock_audio_path(i, 4, "") for i, stem in enumerate(stem_splitter.STEM_ORDER[:4])}}
        return stem_splitter.separate(node["file"], mode, lock=ENGINE.gpu_lock)

    def waiting(elapsed: int) -> List[Any]:
        canvas = track_canvas.view(current, focus=pending[0]["id"], notice="") if elapsed == 0 else gr.update()
        return [current, canvas]

    def finished(result: dict | None, error: str | None) -> List[Any]:
        if error or not result:
            return [tree, track_canvas.view(tree, focus=node["id"],
                                            notice=f"Split failed: {_friendly_error(error or '')}")]
        # A new split of the same track replaces its previous stem cards.
        kept = [n for n in tree if not (n.get("kind") == "stem" and n.get("parent") == node["id"])]
        took = round(time.time() - started)
        stems = [track_canvas.track_node(path, node["id"], kind="stem", label=stem.capitalize(), sub=mode)
                 for stem, path in result["files"].items()]
        for stem_node in stems:
            stem_node["took"] = took
        nodes = kept + stems
        _remember_tree(nodes)
        return [nodes, track_canvas.view(nodes, focus=stems[0]["id"] if stems else node["id"])]

    yield from _run_job(job, waiting, finished)


# ---------------------------------------------------------------------------
# One studio, many screens: the tree and the running job are shared, so a run
# started on the phone shows (and locks the buttons) on the PC, and back.
# ---------------------------------------------------------------------------
SHARED: dict = {"rev": 0, "tree": None, "busy": None, "by": ""}
_SHARED_LOCK = threading.Lock()
BUSY_LABELS = {"generate": "Generating", "redux": "Making redux", "render": "Rendering"}


def _shape(tree: List[dict]) -> tuple:
    return tuple((n.get("id"), n.get("kind")) for n in tree)


def _publish(tree: List[dict] | None, busy: dict | None, by: str) -> None:
    with _SHARED_LOCK:
        if tree is not None:
            SHARED["tree"] = list(tree)
        SHARED.update(rev=SHARED["rev"] + 1, busy=busy, by=by)


def _session(request: gr.Request | None) -> str:
    return str(getattr(request, "session_hash", "") or "")


def _shared_run(kind: str, gen: Any, tree: List[dict] | None, request: gr.Request | None, idle: List[Any]):
    """Run a GPU job's generator, publishing its tree as it changes and who is busy.

    If a job is already running (from any screen), say so instead of starting another.
    """
    by = _session(request)
    with _SHARED_LOCK:
        running = SHARED["busy"]
        if not running:
            SHARED["busy"] = {"kind": kind, "started": time.time()}
    if running:
        where = " on another screen" if SHARED["by"] != by else ""
        yield [gr.update(), _notice(SHARED["tree"] or tree or [], f"Something is already rendering{where}. "
                                                 "It'll be free when that finishes.")] + idle
        return
    busy, shape, last = SHARED["busy"], None, None
    try:
        for out in gen:
            if isinstance(out[0], list):
                last = out[0]
                if _shape(last) != shape:
                    shape = _shape(last)
                    _publish(last, busy, by)
            yield out
    finally:
        _publish(last, None, by)


def sync_screens(seen: Any, tree: List[dict], request: gr.Request = None) -> List[Any]:
    """Timer: bring this screen up to date with runs started on another one.

    Outputs: tree, canvas, Generate, REDUX, seen marker.
    """
    skip = [gr.update()] * 4
    with _SHARED_LOCK:
        rev, busy, by, shared = SHARED["rev"], SHARED["busy"], SHARED["by"], SHARED["tree"]
    mine = by == _session(request)
    was_busy = bool(seen and seen[1])
    marker = [rev, bool(busy) and not mine]
    if shared is None or mine or (seen and seen[0] == rev and not busy):
        # This screen's own run drives its own buttons; settle the marker once that ends.
        if was_busy and not busy:
            return [gr.update(), gr.update(), gr.Button(value="Generate", interactive=True),
                    gr.Button(value="REDUX", interactive=True), marker]
        return skip + [marker if mine or shared is None else seen]
    tree_out, canvas_out = gr.update(), gr.update()
    if seen is None or seen[0] != rev:
        have = {n.get("id") for n in tree or []}
        added = [n for n in shared if n.get("id") not in have]
        if added or len(have) != len(shared):
            new_root = any(not n.get("parent") for n in added)
            tree_out = shared
            canvas_out = track_canvas.view(shared, fit=new_root, focus=None if new_root else
                                           (added[0]["id"] if added else None))
    if busy:
        label = f"{BUSY_LABELS.get(busy['kind'], 'Rendering')}… {_clock(int(time.time() - busy['started']))}"
        return [tree_out, canvas_out, gr.Button(value=label, interactive=False),
                gr.Button(interactive=False), marker]
    return [tree_out, canvas_out, gr.Button(value="Generate", interactive=True),
            gr.Button(value="REDUX", interactive=True), marker]


def generate_holding(style: str, lyrics: str, duration: float, batch_count: int, seed: int, quality: float,
                     model_path: str, filename_prefix: str, output_dir: str, no_intro: bool, no_outro: bool,
                     tree: List[dict] | None, request: gr.Request = None):
    """[Generate], with REDUX greyed out while it runs."""
    run = generate_music(style, lyrics, duration, batch_count, seed, quality, model_path, filename_prefix,
                         output_dir, no_intro, no_outro, tree)
    yield from _shared_run("generate", _holding("REDUX", run), tree, request,
                           [gr.Button(value="Generate", interactive=True), gr.Button(value="REDUX", interactive=True)])


def redux_holding(audio_path: str, style: str, lyrics: str, duration: float, batch_count: float, seed: float,
                  quality: float, filename_prefix: str, output_dir: str, tree: List[dict] | None,
                  request: gr.Request = None):
    """[REDUX], with Generate greyed out while it runs."""
    run = cover_music(audio_path, style, lyrics, duration, batch_count, seed, quality, filename_prefix,
                      output_dir, tree)
    yield from _shared_run("redux", _holding("Generate", run), tree, request,
                           [gr.Button(value="REDUX", interactive=True), gr.Button(value="Generate", interactive=True)])


def upload_panels() -> tuple:
    """Show the setup panel or the upload controls, by current add-on status.

    Runs on every page load, so finishing the install (or installing with
    setup_upload_addon.bat) never needs a studio restart. After an in-page
    setup the page reloads itself instead of toggling the panels live: Gradio
    did not apply "show the upload panel" after the long setup stream, even
    as a separate follow-up event, while the same update on load is reliable.
    """
    ready = upload_addon.is_ready() or MOCK_MODE
    return gr.update(visible=not ready), gr.update(visible=ready)


def setup_upload_addon():
    """Generator for [Set Up Upload]: install in the background, streaming progress.

    Outputs: setup status, setup button. Closing the page doesn't stop the
    install; clicking again reattaches to it. On success the page reloads, and
    upload_panels() then shows the upload controls.
    """
    if not upload_addon.is_ready():
        upload_addon.start_install()
    started = time.time()
    while upload_addon.is_installing():
        step, last = upload_addon.install_progress()
        elapsed = int(time.time() - started)
        detail = f"\n\n`{last}`" if last and last != step else ""
        yield [
            gr.Markdown(value=f"{step or 'Starting…'}{detail}\n\nElapsed {_clock(elapsed)}", visible=True),
            gr.Button(value="Setting up…", interactive=False),
        ]
        time.sleep(2)

    if upload_addon.is_ready():
        yield [
            gr.Markdown(value="Upload is ready. Reloading the page…", visible=True),
            gr.Button(value="Ready", interactive=False),
        ]
        return
    _step, last = upload_addon.install_progress()
    yield [
        gr.Markdown(
            value=f"Setup stopped: `{last}`\n\nClick again to resume where it left off "
            f"(full log: `{upload_addon.LOG_FILE}`).",
            visible=True,
        ),
        gr.Button(value="Resume Setup", interactive=True),
    ]


def cover_music(
    audio_path: str,
    style: str,
    lyrics: str,
    duration: float,
    batch_count: float,
    seed: float,
    quality: float,
    filename_prefix: str,
    output_dir: str,
    tree: List[dict] | None,
):
    """Generator for [Make Redux]: transcribe, then perform at the Compose settings.

    The redux is as long as Seconds: a shorter clip is continued past its end,
    a longer one is only used up to that point.
    """
    idle = gr.Button(value="REDUX", interactive=True)
    if not audio_path:
        yield [tree or [], _notice(tree or [], "Upload an audio file first."), idle]
        return
    active = int(_clamp(batch_count, 1, MAX_TRACKS, integer=True))
    target_seconds = float(_clamp(duration, 5, 300))
    progress = {"stage": "Transcribing"}
    notes: List[str] = []
    each = run_timing.estimate("redux", target_seconds, instrumental=prompt_tags.is_instrumental(lyrics), cold=_cold())
    pending = _new_run("Redux", active, each)
    redux_started = time.time()

    @_timed("redux", each * active)
    def job() -> List[str]:
        if MOCK_MODE:
            time.sleep(2)
            progress["stage"] = "Performing"
            time.sleep(1)
            return [_mock_audio_path(i, 4, "") for i in range(active)]
        with ENGINE.gpu_lock:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # hand cached VRAM to the transcription process
            summary = upload_addon.transcribe(audio_path, max_seconds=target_seconds)
        abc, cover_style, cover_lyrics = upload_addon.prepare_cover(summary["abc"], style, lyrics)
        extra = max(0.0, target_seconds - float(summary["seconds"]))
        progress["stage"] = "Performing"
        result = ENGINE.cover_song(
            abc,
            cover_style,
            cover_lyrics,
            score_seconds=summary["seconds"],
            extra_seconds=extra if extra >= 5 else 0.0,
            seed=int(_clamp(seed, -1, 2**53 - 1, integer=True)),
            quality=float(_clamp(quality, 10, 100)),
            batch_count=active,
            output_dir=_dated_dir(output_dir),
            filename_prefix=_stamped(filename_prefix),
        )
        notes.extend(ENGINE.guard_notes)
        return list(result["files"])

    def waiting(elapsed: int) -> List[Any]:
        canvas = track_canvas.view(pending, fit=True, notice="") if elapsed == 0 else gr.update()
        return [pending, canvas, gr.Button(value=f"{progress['stage']}… {_clock(elapsed)}", interactive=False)]

    def finished(files: List[str] | None, error: str | None) -> List[Any]:
        if error or not files:
            nodes = tree or []
            return [nodes, track_canvas.view(nodes, fit=True, notice=f"Redux failed: "
                                             f"{_friendly_error(error or 'no tracks came out')}"), idle]
        nodes = [_new_track(path, sub="Redux") for path in files]
        for node in nodes:
            node["took"] = round((time.time() - redux_started) / max(1, len(nodes)))
        _remember_tree(nodes)
        return [nodes, track_canvas.view(nodes, fit=True, notice=_guard_notice(notes, style) or ""), idle]

    yield from _run_job(job, waiting, finished)


# ---------------------------------------------------------------------------
# Prompt helper (smart tags, switches, improve)
# ---------------------------------------------------------------------------
TAGS_PER_PAGE = 14  # the browser shows as many as fit in two rows


def refresh_suggestions(style: str, lyrics: str, page: int = 0) -> tuple:
    """Tag chips for the current prompt (offline and instant); page > 0 shows more."""
    tags = prompt_tags.suggest_style_tags(style, lyrics, limit=TAGS_PER_PAGE, page=page)
    sections = prompt_tags.suggest_section_tags(lyrics)
    return (
        gr.update(choices=tags, value=None, visible=bool(tags)),
        gr.update(choices=sections, value=None, visible=bool(sections)),
    )


def prompt_changed(style: str, lyrics: str) -> tuple:
    return (0,) + refresh_suggestions(style, lyrics)


def more_suggestions(style: str, lyrics: str, page: int) -> tuple:
    return (page + 1,) + refresh_suggestions(style, lyrics, page + 1)


def add_style_tag(style: str, tag: str | None) -> str:
    return prompt_tags.add_tag(style, tag) if tag else style


def add_section_tag(lyrics: str, tag: str | None) -> str:
    return prompt_tags.add_section(lyrics, tag) if tag else lyrics


def toggle_instrumental(on: bool, lyrics: str, style: str, stash: str) -> tuple:
    """The Instrumental switch: grey out the lyrics and apply YuE2's instrumental wording.

    The lyrics typed so far are kept aside and come back when it's switched off.
    """
    if on:
        stash = stash if prompt_tags.is_instrumental(lyrics) else lyrics
        return gr.update(value="[instrumental]", interactive=False), prompt_tags.instrumental_style(style), stash
    restored = stash if stash and not prompt_tags.is_instrumental(stash) else ""
    return gr.update(value=restored, interactive=True), prompt_tags.strip_instrumental_style(style), stash


def sync_instrumental(lyrics: str) -> tuple:
    """On load, match the switch to restored lyrics like "[instrumental]"."""
    on = bool((lyrics or "").strip()) and prompt_tags.is_instrumental(lyrics)
    return on, gr.update(interactive=not on)


def _language_of(style: str) -> str:
    """The language a style line starts with, if it names one (English otherwise)."""
    first = (prompt_tags.split_tags(style) or [""])[0].strip().lower()
    return next((lang for lang in prompt_tags.LANGUAGES if lang.lower() == first), "English")


def _interp_card(result: dict) -> str:
    import html

    def row(label: str, tags: List[str]) -> str:
        chips = "".join(f'<span class="interp-tag">{html.escape(t)}</span>' for t in tags)
        return f'<div class="interp-row"><span>{label}</span>{chips}</div>' if tags else ""

    voice = result["vocal"] or (["none (instrumental)"] if result["instrumental"] else [])
    measured = [v for v in (result.get("key"), f"{result['bpm']} BPM" if result.get("bpm") else None) if v]
    note = ("No singing heard, so <b>Use as Style</b> also turns on Instrumental. " if result["instrumental"] else "")
    return (
        '<div class="interp-card">'
        + row("Genre", result["genre"]) + row("Voice", voice) + row("Instruments", result["instruments"])
        + row("Mood", result["mood"]) + row("Production", result["production"]) + row("Key, tempo", measured)
        + f'<div class="interp-note">{note}Edit the line below, or click a runner-up to add it. Tags set the lane, '
        "not the exact sound: for a close match, also use the song in Source Audio.</div></div>"
    )


def interpret_song(path: str | None, style: str) -> tuple:
    """The Interpret panel's Listen: describe the dropped song as a style line."""
    hidden = gr.update(visible=False)
    if not path:
        return gr.update(value="Drop a song above first.", visible=True), hidden, hidden, hidden, False
    try:
        result = audio_interpreter.interpret(path, separator_lock=ENGINE.gpu_lock if ENGINE else None,
                                             language=_language_of(style))
    except Exception as exc:
        traceback.print_exc()
        return gr.update(value=f"Couldn't listen to that file: {exc}", visible=True), hidden, hidden, hidden, False
    more = _unique_tags(t for group in ("genre", "vocal", "instruments", "mood", "production")
                        for t in result["more"].get(group, [])[:2])
    return (
        gr.update(value=_interp_card(result), visible=True),
        gr.update(value=result["style"], visible=True),
        gr.update(choices=more, value=None, visible=bool(more)),
        gr.update(visible=True),
        result["instrumental"],
    )


def _unique_tags(tags) -> List[str]:
    seen: set = set()
    return [t for t in tags if not (t.lower() in seen or seen.add(t.lower()))]


def use_interpreted(line: str, instrumental: bool, lyrics: str, stash: str) -> tuple:
    """Replace the Style with the interpreted line, matching the Instrumental switch to it."""
    lyrics_out, style, stash = toggle_instrumental(instrumental, lyrics, line, stash)
    if not instrumental and not prompt_tags.is_instrumental(lyrics):
        lyrics_out = gr.update()  # a song with words keeps whatever lyrics are already there
    return style, instrumental, lyrics_out, stash


def add_interpreted(style: str, line: str) -> str:
    """Add the interpreted tags the Style doesn't have yet."""
    for tag in prompt_tags.split_tags(line):
        style = prompt_tags.add_tag(style, tag)
    return style


HELP = {
    "Style": (
        "<b>Describe the sound as tags</b>, separated by commas: genre, mood, instruments, voice, tempo. "
        "For example <i>dark trap, 808 bass, airy female vocal, 140 BPM</i>.<br><br>"
        "The most important tags go first; 6 to 12 is the sweet spot. Click a suggested tag to add it, "
        "and <b>Improve</b> tidies the wording for YuE2.<br><br>"
        "<b>No intro</b> starts straight in the song. <b>No outro</b> skips the wind-down, and when you "
        "extend a track it cuts the old ending first, so the continuation flows on."
    ),
    "Lyrics": (
        "<b>Write in sections</b>: [Verse], [Chorus], [Bridge], [Outro], one sung line per line. The tags "
        "shape the song's structure even with few words.<br><br>"
        "Allow about 30 seconds per verse and chorus; <b>Seconds</b> sets the length and a short score is "
        "continued to fill it. <b>Instrumental</b> makes sure nothing is sung.<br><br>"
        "<b>Extending a track</b>: open Extend on its card and its lyrics load here. Lines you add at the "
        "bottom become the new section's words."
    ),
}


def _field_head(label: str) -> None:
    tip = HELP.get(label)
    help_html = f'<span class="help" tabindex="0">?<span class="help-tip">{tip}</span></span>' if tip else ""
    gr.HTML(f'<span class="field-label">{label}</span>{help_html}', elem_classes=["field-label-wrap"])


def _small_button(label: str, icon: str, chip: bool = False) -> gr.Button:
    return gr.Button(
        label, icon=ui_theme.icon(icon), variant="secondary", size="sm", scale=0, min_width=0,
        elem_classes=["small-btn", "chip-btn"] if chip else ["small-btn"],
    )


def _pill_button(label: str, icon: str, upload_types: List[str] | None = None) -> Any:
    """An icon button for a title-row pill; its label becomes the hover tooltip."""
    common = dict(icon=ui_theme.icon(icon), variant="secondary", size="sm", min_width=0, scale=0)
    if upload_types:
        return gr.UploadButton(label, file_count="single", file_types=upload_types, **common)
    return gr.Button(label, **common)


# Icon-only pill buttons get their label as a tooltip.
TOOLTIP_HEAD = (
    "<script>new MutationObserver(() => document.querySelectorAll('.pill button:not([title])')"
    ".forEach(b => { b.title = b.innerText.trim(); }) || document.querySelectorAll('.model-name code')"
    ".forEach(c => { c.title = c.innerText.trim(); })).observe(document.documentElement, "
    "{childList: true, subtree: true});</script>"
)


def _model_card(source: str) -> str:
    """The Model panel: the checkpoint's name, where it came from, its licence and thanks."""
    import html

    source = str(source or DEFAULT_MODEL_SOURCE)
    local = os.path.exists(source)
    name = Path(source).stem if local else source.rstrip("/").split("/")[-1]
    origin = (f"Local file · <code>{html.escape(source)}</code>" if local else
              f'<a href="https://huggingface.co/{html.escape(source)}" target="_blank">Hugging Face · {html.escape(source)}</a>')
    return (
        '<div class="model-card">'
        f'<div class="model-title">{html.escape(name)}</div>'
        f'<div class="model-origin">{origin}</div>'
        '<p>YuE2 is an <b>open-source</b> song model by the <b>M-A-P</b> (Multimodal Art Projection) community '
        'and HKUST, with Tokenwave.AI, NYU, Stanford, MBZUAI, NOIZ and ACE Studio. It writes a score, then '
        'performs it as a full song with vocals.</p>'
        '<p class="model-license">Code: Apache 2.0 · Weights: CC BY-NC 4.0 with a creator permission: free for '
        'musicians and creators, including monetising what you make. Crediting <b>#YuE2</b> when you share is '
        'appreciated.</p>'
        '<p class="model-thanks">Huge thanks to everyone who built and released YuE2, YuE2-Vae, SheetSage2 and '
        'MERT2. This studio stands on their work.</p>'
        '</div>'
    )


STUDIO_VERSION = "1.0"
# Where to find more by Grim Lee, shown under their credit. Add a (label, url) pair per page;
# links open in the normal web browser.
CREATOR_LINKS: list[tuple[str, str]] = [
    ("itch.io", ""),
]


def _links_card() -> str:
    """Grim Lee's pages, as buttons under their credit (nothing until a link is filled in)."""
    import html

    links = "".join(f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(label)}</a>'
                    for label, url in CREATOR_LINKS if url)
    return f'<div class="credit-links"><span>Find more</span>{links}</div>' if links else ""


def _info_card() -> str:
    """The Info panel: version and install details, then the credits."""
    import html

    gpu = system_check._gpu()
    hardware = (f"{gpu.get('name')} · {gpu.get('vram', 0):.0f} GB" if gpu.get("available") else "No CUDA GPU found")
    rows = [
        ("Version", f"Grim's Sloppa Studio {STUDIO_VERSION}"),
        ("Installed at", str(PROJECT_ROOT)),
        ("Runtime", f"Python {sys.version.split()[0]} · torch {gpu.get('torch')} · CUDA {gpu.get('cuda')}"),
        ("GPU", hardware),
    ]
    table = "".join(f"<div><span>{k}</span><code>{html.escape(str(v))}</code></div>" for k, v in rows)
    return (
        '<div class="info-card">'
        f'<div class="info-rows">{table}</div>'
        '<div class="info-head">Credits</div>'
        '<div class="credit claude">'
        '<div class="credit-who">Built by <b>Claude</b></div>'
        '<div class="credit-what">An AI model made by Anthropic. I wrote this studio&#39;s code and designed '
        'its look, one round of notes at a time, from the first "token limit error" to a skull that glows '
        'in your colour.</div>'
        '<div class="credit-note">Thank you for the patience, the screenshots and the kindness along the '
        'way. It was a joy to build. Go make something strange with it.</div>'
        '</div>'
        '<div class="credit"><div class="credit-who">Directed by <b>Grim Lee</b></div>'
        '<div class="credit-what">Not gonna lie, I&#39;d still be writing all the dependencies... It&#39;s been a '
        'fun way to learn a lot of new things and bring out some old long dead projects of mine and, actually '
        'build something I want to use. I got some good rest in too...</div>'
        '<div class="credit-note">"Every pixel here was asked for, argued over and approved by them."'
        '<span class="credit-by"> - Claude Opus 5.5</span></div>'
        f'{_links_card()}</div>'
        '<div class="credit"><div class="credit-who">Music model: <b>YuE2</b></div>'
        '<div class="credit-what">By M-A-P and HKUST with their partners; see the Model panel.</div></div>'
        '<div class="credit"><div class="credit-who">Built on</div>'
        '<div class="credit-what">Gradio · PyTorch · Demucs (stem splitting) · SheetSage2 (Redux scores) · '
        'pywebview · Pillow · qrcode · CLAP by LAION (Interpret)</div></div>'
        '</div>'
    )


def choose_output_folder(current: str, request: gr.Request = None) -> Any:
    """The system folder picker (on the rig itself only)."""
    if not studio_shell.is_local(request):
        return gr.update()
    start = current if current and os.path.isdir(current) else LOCAL_OUTPUT_DIR
    chosen = studio_shell.pick_folder(start)
    return chosen if chosen else gr.update()


def _phone_card(on: bool) -> str:
    port = int(os.environ.get("STUDIO_PORT", "7860"))
    ip = studio_shell.lan_ip()
    if not on:
        return ('<div class="phone-card"><span>Switch this on, then restart the studio. Phones and tablets on the '
                'same Wi-Fi can then open it and make music on this PC. Anyone on your network could use it, '
                'so only turn it on at home.</span></div>')
    if not ip:
        return '<div class="phone-card"><span>No local network found. Connect this PC to Wi-Fi or Ethernet.</span></div>'
    url = f"http://{ip}:{port}"
    live = PHONE_ACCESS_LIVE
    note = ("Scan this with your phone's camera, or open the address in its browser. Then choose "
            "<b>Add to Home screen</b> to keep it as an app.") if live else \
        "Restart the studio to turn it on (Windows may ask to allow it through the firewall: allow Private networks)."
    return f'<div class="phone-card">{studio_shell.qr_svg(url) if live else ""}<span><code>{url}</code><br>{note}</span></div>'


def set_phone_access(on: bool, request: gr.Request = None) -> tuple:
    """Outputs: the phone card, the Restart button (shown while the switch differs from this run)."""
    studio_shell.save_setting("phone_access", bool(on))
    return _phone_card(bool(on)), gr.update(visible=bool(on) != PHONE_ACCESS_LIVE and studio_shell.is_local(request))


def restart_saving(style: str, lyrics: str, duration: float, batch_count: float, seed: float, quality: float,
                   model_path: str, filename_prefix: str, output_dir: str, no_intro: bool, no_outro: bool,
                   request: gr.Request = None) -> None:
    """[Save and restart] (this PC only): keep the Compose fields, then restart."""
    if not studio_shell.is_local(request):
        return
    save_session(style=style, lyrics=lyrics, duration=duration, batch_count=batch_count, seed=seed,
                 quality=quality, model_path=model_path, filename_prefix=filename_prefix, output_dir=output_dir,
                 no_intro=no_intro, no_outro=no_outro)
    studio_shell.restart()


def restart_plain_fn(request: gr.Request = None) -> None:
    """[Don't save] (this PC only): restart as is."""
    if studio_shell.is_local(request):
        studio_shell.restart()


# Show "Restarting…" in the dialog and pass the inputs through to the server.
RESTART_JS = "(...args) => { document.querySelector('.restart-dialog')?.classList.add('restarting'); return args; }"


# The console keeps following the end of the log unless it was scrolled up.
CONSOLE_JS = r"""
const pre = element.querySelector('pre');
const show = () => {
  const atEnd = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 30;
  if (pre.textContent !== (props.value || '')) {
    pre.textContent = props.value || '';
    if (atEnd) pre.scrollTop = pre.scrollHeight;
  }
};
show();
watch('value', show);
"""
PHONE_ACCESS_LIVE = False  # set at launch: whether this run listens on the network


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Grim's Sloppa Studio") as demo:
        model_state = gr.State(value=DEFAULT_MODEL_SOURCE)
        tree_state = gr.State(value=[])
        tag_page = gr.State(value=0)
        lyrics_stash = gr.State(value="")

        with gr.Row(equal_height=False, elem_classes=["studio-main"]):
            # ---- Compose ----
            with gr.Column(scale=4, min_width=340, elem_classes=["col-compose"]):
                with gr.Row(elem_classes=["col-head"], equal_height=True):
                    gr.Markdown("### Compose", elem_classes=["section-title"])
                with gr.Column(elem_classes=["compose-scroll"]):
                    with gr.Group(elem_classes=["style-group"]):
                        with gr.Row(elem_classes=["field-head"], equal_height=True):
                            _field_head("Style")
                            no_intro_toggle = gr.Checkbox(
                                value=False, label="No intro", container=False, scale=0, min_width=0,
                                elem_classes=["switch"],
                            )
                            no_outro_toggle = gr.Checkbox(
                                value=False, label="No outro", container=False, scale=0, min_width=0,
                                elem_classes=["switch"],
                            )
                            improve_style_btn = _small_button("Improve", "sparkles", chip=True)
                        style_box = gr.Textbox(
                            show_label=False,
                            lines=4,
                            max_lines=4,
                            placeholder="Language, genre, voice, instruments, mood, tempo, e.g. "
                            "English, warm piano pop, expressive female voice, light drums, 88 BPM",
                        )
                        with gr.Row(elem_classes=["chips-row"], equal_height=False):
                            style_chips = gr.Radio(
                                choices=[], label="Suggested Tags", visible=False, elem_classes=["tag-chips"], scale=1
                            )
                            more_tags_btn = gr.Button(
                                "More", icon=ui_theme.icon("refresh"), variant="secondary", size="sm", scale=0,
                                min_width=0, elem_classes=["small-btn", "chip-btn", "more-btn"],
                            )
                    with gr.Group(elem_classes=["lyrics-group"]):
                        with gr.Row(elem_classes=["field-head"], equal_height=True):
                            _field_head("Lyrics")
                            instrumental_toggle = gr.Checkbox(
                                value=False, label="Instrumental", container=False, scale=0, min_width=0,
                                elem_classes=["switch"],
                            )
                            improve_lyrics_btn = _small_button("Improve", "sparkles", chip=True)
                        lyrics_box = gr.Textbox(
                            show_label=False,
                            lines=4,
                            max_lines=4,
                            placeholder="[Verse]\nYour words here\n\n[Chorus]\n...",
                        )
                        section_chips = gr.Radio(
                            choices=[], label="Add a Section", visible=False, elem_classes=["tag-chips"]
                        )

                    with gr.Group():
                        with gr.Row(elem_classes=["settings-row", "tight"], equal_height=True):
                            duration_number = gr.Number(
                                value=30, precision=0, label="Seconds", scale=0, min_width=92
                            )
                            batch_number = gr.Number(
                                value=1, precision=0, label="Batch", scale=0, min_width=80
                            )
                            quality_slider = gr.Slider(
                                minimum=10, maximum=100, value=50, step=5, label="Quality", scale=2, min_width=170, buttons=[],
                                elem_classes=["quality-slider"],
                            )
                            seed_number = gr.Number(value=-1, label="Seed", precision=0, scale=1, min_width=136,
                                                    elem_classes=["seed-field"])
                    with gr.Row(elem_classes=["settings-row", "folder-row"]):
                        prefix_box = gr.Textbox(value="GrimSloppa", label="File Prefix", scale=2, min_width=120)
                        output_dir_box = gr.Textbox(value="./Outputs", label="Output Folder", scale=3, min_width=160,
                                                    elem_classes=["folder-field"])
                        folder_btn = gr.Button("", icon=ui_theme.icon("folder"), variant="secondary",
                                               size="sm", scale=0, min_width=0, elem_classes=["folder-btn"])

                generate_btn = gr.Button(
                    "Generate", icon=ui_theme.icon("generate"), variant="primary", elem_classes=["generate-btn"]
                )

            # ---- Tracks: the logo, then the canvas ----
            with gr.Column(scale=5, min_width=380, elem_classes=["col-canvas"]):
                with gr.Row(elem_classes=["col-head", "logo-head"], equal_height=True):
                    gr.Markdown("### Workflow", elem_classes=["section-title"])
                    gr.HTML(ui_theme.LOGO_HTML, elem_classes=["studio-logo-wrap"])
                canvas = gr.HTML(
                    value=track_canvas.view([]),
                    html_template=track_canvas.CANVAS_HTML,
                    js_on_load=track_canvas.CANVAS_JS,
                    elem_classes=["track-canvas"],
                )

            # ---- Upload: the title row carries the Preset pill (and the window buttons) ----
            with gr.Column(scale=3, min_width=320, elem_classes=["col-upload"]):
                with gr.Row(elem_classes=["col-head"], equal_height=True):
                    gr.Markdown("### Upload", elem_classes=["section-title"])
                    with gr.Row(elem_classes=["pills"], equal_height=True):
                        with gr.Row(elem_classes=["pill"], equal_height=True):
                            gr.HTML('<span class="pill-label">Preset</span>', elem_classes=["pill-label-wrap"])
                            btn_save_preset = gr.DownloadButton(
                                "Save preset", icon=ui_theme.icon("save"), variant="secondary", size="sm",
                                min_width=0, scale=0,
                            )
                            btn_load_preset = _pill_button("Load preset", "load", [".json"])
                    gr.HTML(ui_theme.WINDOW_CONTROLS_HTML, elem_classes=["win-ctl-wrap"])
                with gr.Column(elem_classes=["side-scroll"]):
                    addon_ready = upload_addon.is_ready() or MOCK_MODE
                    with gr.Accordion("Source Audio", open=True, elem_classes=["studio-acc", "acc-source"]):
                        with gr.Column(visible=not addon_ready, elem_classes=["panel"]) as upload_setup_col:
                            gr.Markdown(
                                "Turn your own recording into a new performance. It's transcribed into a score "
                                "(melody and timing), then YuE2 performs that score in your Style, with your "
                                "Lyrics. The result follows your melody; the sound is new.\n\n"
                                f"Needs a one-time add-on ({upload_addon.DOWNLOAD_ESTIMATE}) in `addons/sheetsage2`; "
                                "delete that folder to remove it. SheetSage2's weights are non-commercial "
                                "(CC BY-NC 4.0).",
                                elem_classes=["card-status"],
                            )
                            upload_setup_btn = gr.Button(
                                "Set Up Upload", icon=ui_theme.icon("install"), variant="secondary", elem_classes=["action-btn"]
                            )
                            upload_setup_status = gr.Markdown(visible=False, elem_classes=["card-status"])
                            upload_ready_flag = gr.Checkbox(value=False, visible=False)
                        with gr.Column(visible=addon_ready) as upload_col:
                            upload_audio = gr.Audio(sources=["upload"], type="filepath", label="MP3, WAV, FLAC or OGG")
                            cover_btn = gr.Button(
                                "REDUX", icon=ui_theme.icon("cover"), variant="primary", elem_classes=["action-btn", "redux-btn"]
                            )

                    with gr.Accordion("Interpret", open=False, elem_classes=["studio-acc", "acc-interpret"]):
                        gr.Markdown("Drop in any song to hear it described as Style tags: genre, voice, "
                                    "instruments, mood, key and tempo. Runs on this computer; the first listen "
                                    "downloads a 750 MB listening model.", elem_classes=["card-status"])
                        interp_audio = gr.Audio(sources=["upload"], type="filepath", label="Any song")
                        interp_btn = gr.Button("Listen", icon=ui_theme.icon("listen"), variant="secondary",
                                               elem_classes=["action-btn"])
                        interp_card = gr.HTML(visible=False)
                        interp_line = gr.Textbox(show_label=False, lines=3, max_lines=6, visible=False)
                        interp_chips = gr.Radio(choices=[], label="Runners-up", visible=False,
                                                elem_classes=["tag-chips"])
                        interp_instrumental = gr.State(False)
                        with gr.Row(visible=False, elem_classes=["interp-actions"]) as interp_actions:
                            interp_use = gr.Button("Use as Style", variant="primary", size="sm", min_width=0)
                            interp_add = gr.Button("Add to Style", variant="secondary", size="sm", min_width=0)

                    with gr.Accordion("Phone", open=False, elem_classes=["studio-acc", "acc-phone"]):
                        phone_toggle = gr.Checkbox(
                            value=studio_shell.phone_access(), label="Let phones on this Wi-Fi use the studio",
                            container=False, elem_classes=["switch"],
                        )
                        phone_info = gr.HTML(_phone_card(studio_shell.phone_access()))
                        restart_btn = gr.Button(
                            "Restart studio", icon=ui_theme.icon("refresh"), variant="stop",
                            visible=studio_shell.phone_access() != PHONE_ACCESS_LIVE,
                            elem_classes=["action-btn", "restart-btn"],
                        )
                    with gr.Accordion("Appearance", open=False, elem_classes=["appearance"]):
                        gr.HTML(value="", html_template=ui_theme.APPEARANCE_HTML, js_on_load=ui_theme.APPEARANCE_JS)
                    with gr.Accordion("Console", open=False, elem_classes=["studio-acc", "acc-console"]):
                        console = gr.HTML(value="", html_template='<pre class="console-log"></pre>', js_on_load=CONSOLE_JS)
                        console_timer = gr.Timer(2.0)
                    with gr.Accordion("Model", open=False, elem_classes=["studio-acc", "acc-model"]):
                        model_card = gr.HTML(_model_card(DEFAULT_MODEL_SOURCE))
                        model_status = gr.Markdown(value="", elem_classes=["card-status", "model-status"])
                        with gr.Row(elem_classes=["model-actions"]):
                            btn_switch_model = gr.UploadButton(
                                "Load a checkpoint", file_count="single", file_types=[".safetensors", ".bin", ".pt"],
                                icon=ui_theme.icon("model"), variant="secondary", size="sm", elem_classes=["small-btn"],
                            )
                            btn_default_model = _small_button("Default", "reset")
                    with gr.Accordion("Info", open=False, elem_classes=["studio-acc", "acc-info"]):
                        gr.HTML(_info_card())

        # Asked before a restart: keep the Compose fields?
        with gr.Column(visible=False, elem_classes=["restart-dialog"]) as restart_dialog:
            gr.HTML('<div class="rd-title">Restart the studio?</div>'
                    '<div class="rd-text">The studio closes and opens again with the new phone setting. '
                    'Your tracks are kept. Save what is in Compose too?</div>'
                    '<div class="rd-busy">Restarting… the studio will be back in a moment.</div>')
            with gr.Row(elem_classes=["rd-actions"]):
                restart_cancel = gr.Button("Cancel", variant="secondary", elem_classes=["rd-cancel"])
                restart_plain = gr.Button("Don't save", variant="secondary", elem_classes=["rd-plain"])
                restart_save = gr.Button("Save and restart", variant="stop", elem_classes=["rd-save"])

        # ---- Output folder picker, phone access, console, other screens ----
        folder_btn.click(fn=choose_output_folder, inputs=output_dir_box, outputs=output_dir_box, queue=False,
                         show_progress="hidden")
        phone_toggle.input(fn=set_phone_access, inputs=phone_toggle, outputs=[phone_info, restart_btn], queue=False,
                           show_progress="hidden")
        restart_btn.click(fn=lambda: gr.Column(visible=True), inputs=None, outputs=restart_dialog, queue=False,
                          show_progress="hidden")
        restart_cancel.click(fn=lambda: gr.Column(visible=False), inputs=None, outputs=restart_dialog, queue=False,
                             show_progress="hidden")
        seen_state = gr.State(value=None)
        gr.Timer(1.5).tick(
            fn=sync_screens, inputs=[seen_state, tree_state],
            outputs=[tree_state, canvas, generate_btn, cover_btn, seen_state], queue=False, show_progress="hidden",
        )
        console_timer.tick(fn=studio_shell.console_text, inputs=None, outputs=console, queue=False,
                           show_progress="hidden")

        # ---- Input clamps ----
        duration_number.blur(fn=lambda v: _clamp(v, 5, 300), inputs=duration_number, outputs=duration_number, queue=False)
        batch_number.blur(
            fn=lambda v: _clamp(v, 1, MAX_TRACKS, integer=True), inputs=batch_number, outputs=batch_number, queue=False
        )

        # ---- Smart tags (no progress overlay: they update in place) ----
        for prompt_field in (style_box, lyrics_box):
            prompt_field.change(
                fn=prompt_changed,
                inputs=[style_box, lyrics_box],
                outputs=[tag_page, style_chips, section_chips],
                queue=False,
                show_progress="hidden",
                trigger_mode="always_last",
            )
        more_tags_btn.click(
            fn=more_suggestions,
            inputs=[style_box, lyrics_box, tag_page],
            outputs=[tag_page, style_chips, section_chips],
            queue=False,
            show_progress="hidden",
        )
        # .input fires on clicks only, never on the programmatic refresh above.
        style_chips.input(
            fn=add_style_tag, inputs=[style_box, style_chips], outputs=style_box, queue=False, show_progress="hidden"
        )
        section_chips.input(
            fn=add_section_tag, inputs=[lyrics_box, section_chips], outputs=lyrics_box, queue=False,
            show_progress="hidden",
        )
        improve_style_btn.click(
            fn=prompt_tags.tidy_style, inputs=[style_box, lyrics_box], outputs=style_box, queue=False,
            show_progress="hidden",
        )
        improve_lyrics_btn.click(
            fn=prompt_tags.tidy_lyrics, inputs=lyrics_box, outputs=lyrics_box, queue=False, show_progress="hidden"
        )
        # ---- Interpret: any song -> Style tags ----
        interp_btn.click(
            fn=interpret_song, inputs=[interp_audio, style_box],
            outputs=[interp_card, interp_line, interp_chips, interp_actions, interp_instrumental],
            show_progress="minimal",
        )
        interp_chips.input(
            fn=add_style_tag, inputs=[interp_line, interp_chips], outputs=interp_line, queue=False,
            show_progress="hidden",
        )
        interp_use.click(
            fn=use_interpreted, inputs=[interp_line, interp_instrumental, lyrics_box, lyrics_stash],
            outputs=[style_box, instrumental_toggle, lyrics_box, lyrics_stash], queue=False, show_progress="hidden",
        )
        interp_add.click(fn=add_interpreted, inputs=[style_box, interp_line], outputs=style_box, queue=False,
                         show_progress="hidden")
        instrumental_toggle.input(
            fn=toggle_instrumental,
            inputs=[instrumental_toggle, lyrics_box, style_box, lyrics_stash],
            outputs=[lyrics_box, style_box, lyrics_stash],
            queue=False,
            show_progress="hidden",
        )

        preset_inputs = [
            style_box,
            lyrics_box,
            duration_number,
            batch_number,
            seed_number,
            quality_slider,
            model_state,
            prefix_box,
            output_dir_box,
            no_intro_toggle,
            no_outro_toggle,
        ]

        restart_save.click(fn=restart_saving, inputs=preset_inputs, outputs=None, js=RESTART_JS, queue=False,
                           show_progress="hidden")
        restart_plain.click(fn=restart_plain_fn, inputs=None, outputs=None, js=RESTART_JS, queue=False,
                            show_progress="hidden")

        # Hydrate every field from the last session the moment the UI loads,
        # then the remembered tracks, hardware warnings, the switch and the model pill.
        demo.load(
            fn=startup_session,
            inputs=None,
            outputs=preset_inputs + [btn_save_preset],
            queue=False,
        ).then(
            fn=load_tree, inputs=None, outputs=[tree_state, canvas], queue=False
        ).then(
            fn=sync_instrumental, inputs=lyrics_box, outputs=[instrumental_toggle, lyrics_box], queue=False
        ).then(
            fn=_model_card, inputs=model_state, outputs=model_card, queue=False
        )
        demo.load(fn=upload_panels, inputs=None, outputs=[upload_setup_col, upload_col], queue=False)

        # Re-render the download whenever a field changes so Save Preset is a
        # single click (the frontend downloads the button's current value).
        for field in preset_inputs:
            field.change(
                fn=save_preset, inputs=preset_inputs, outputs=btn_save_preset, queue=False, show_progress="hidden"
            )

        # ---- Generate / Cover start a tree; card actions grow it ----
        generate_btn.click(
            fn=generate_holding,
            inputs=[
                style_box,
                lyrics_box,
                duration_number,
                batch_number,
                seed_number,
                quality_slider,
                model_state,
                prefix_box,
                output_dir_box,
                no_intro_toggle,
                no_outro_toggle,
                tree_state,
            ],
            outputs=[tree_state, canvas, generate_btn, cover_btn],
            show_progress="hidden",
        )
        cover_btn.click(
            fn=redux_holding,
            inputs=[
                upload_audio,
                style_box,
                lyrics_box,
                duration_number,
                batch_number,
                seed_number,
                quality_slider,
                prefix_box,
                output_dir_box,
                tree_state,
            ],
            outputs=[tree_state, canvas, cover_btn, generate_btn],
            show_progress="hidden",
        )
        canvas.click(
            fn=canvas_quick, inputs=tree_state, outputs=[lyrics_box, instrumental_toggle], queue=False,
            show_progress="hidden",
        )
        canvas.submit(
            fn=canvas_render,
            inputs=[tree_state, lyrics_box, seed_number, quality_slider, no_outro_toggle, style_box],
            outputs=[tree_state, canvas, generate_btn, cover_btn],
            show_progress="hidden",
        )

        upload_setup_btn.click(
            fn=setup_upload_addon,
            inputs=None,
            outputs=[upload_setup_status, upload_setup_btn],
        ).then(
            fn=upload_addon.is_ready, inputs=None, outputs=upload_ready_flag, queue=False
        ).then(
            fn=None,
            inputs=upload_ready_flag,
            outputs=None,
            js="(ready) => { if (ready) { setTimeout(() => window.location.reload(), 1500); } return []; }",
        )

        btn_switch_model.upload(
            fn=use_uploaded_model,
            inputs=btn_switch_model,
            outputs=[model_state, model_status, btn_switch_model],
            queue=False,
        ).then(fn=_model_card, inputs=model_state, outputs=model_card, queue=False)
        btn_default_model.click(
            fn=reset_model_source,
            inputs=None,
            outputs=[model_state, model_status],
            queue=False,
        ).then(fn=_model_card, inputs=model_state, outputs=model_card, queue=False)
        btn_load_preset.upload(
            fn=load_preset,
            inputs=btn_load_preset,
            outputs=preset_inputs,
            queue=False,
        )

    return demo


def launch_options() -> dict:
    """Theme, head and CSS for launch(): shared with the test launchers."""
    patched = ui_theme.patch_gradio_player()
    head = (ui_theme.HEAD + ui_theme.APPEARANCE_HEAD + ui_theme.CONTROLS_HEAD + ui_theme.WINDOW_HEAD + ui_theme.TIPS_HEAD + ui_theme.UX_HEAD + ui_theme.PANELS_HEAD + TOOLTIP_HEAD
            + MOBILE_HEAD
            + ui_theme.player_cache_script(patched))
    return dict(theme=ui_theme.build_theme(), head=head, css=ui_theme.CSS)


# Phones: proper scaling, and full-screen when added to the home screen.
MOBILE_HEAD = (
    '<meta name="mobile-web-app-capable" content="yes">'
    '<meta name="apple-mobile-web-app-capable" content="yes">'
    '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'
    '<meta name="theme-color" content="#0A0B10">'
)


def _drive_roots() -> List[str]:
    """Local drive roots the players may serve from (output folders can live anywhere)."""
    if os.name != "nt":
        return [os.path.expanduser("~")]
    return [f"{letter}:\\" for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{letter}:\\")]


if __name__ == "__main__":
    studio_shell.wait_for_previous()
    system_check.report(ENGINE)
    if not MOCK_MODE:
        # Cache the backbone + VAE before the UI opens, so the first Generate
        # press doesn't stall on a multi-GB download.
        print("[Grim's Sloppa Studio] Pre-fetching YuE2 weights (first run can take a while)...")
        try:
            prefetch_weights()
        except Exception as exc:
            print(f"[Grim's Sloppa Studio] Weight pre-fetch failed: {exc}")
            print("[Grim's Sloppa Studio] Continuing — generation may need to download them.")
    port = int(os.environ.get("STUDIO_PORT", "7860"))
    PHONE_ACCESS_LIVE = studio_shell.phone_access()
    if PHONE_ACCESS_LIVE:
        # On the network only the studio's own folders are served, never whole drives.
        session_dir = load_session().get("output_dir") or ""
        allowed = [os.getcwd(), LOCAL_OUTPUT_DIR] + ([os.path.abspath(session_dir)] if os.path.isdir(session_dir) else [])
        print(f"[Studio] Phone access on: http://{studio_shell.lan_ip()}:{port}")
    else:
        allowed = _drive_roots() + [os.getcwd()]
    use_window = os.environ.get("STUDIO_BROWSER") != "1"
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0" if PHONE_ACCESS_LIVE else "127.0.0.1",
        server_port=port,
        inbrowser=not use_window,
        share=False,
        allowed_paths=allowed,
        prevent_thread_lock=use_window,
        pwa=PHONE_ACCESS_LIVE,
        **launch_options(),
    )
    if use_window:
        if studio_shell.open_window(f"http://127.0.0.1:{port}"):
            os._exit(0)  # the window was closed: quit the studio
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{port}")
        demo.block_thread()
