"""Native PyTorch engine for YuE2 (replaces the ComfyUI API bridge).

Architecture
------------
    app.py (Gradio)
        -> YuE2Engine.generate_song()        # end-to-end, batch-aware
             -> generate_abc()               # stage 1: symbolic plan (ABC)
             -> generate_audio_tokens()      # stage 2: music/codec tokens
             -> decode_audio()               # stage 3: tokens -> waveform
        -> YuE2Engine.extend_track()         # continue a track (or add an intro)

The backbone and VAE are resolved from the pre-fetched Hugging Face cache
(``prefetch_weights()``) and handed to ``YuE2Pipeline`` as local directories,
which keeps the model resident on the GPU in bfloat16 inside the ~16 GB VRAM
budget. The model repos are hard-coded, so the UI is fully plug-and-play.

ABC score planning is hardcoded on for every run (``ABC_MODE = "auto"`` ->
chord-annotated ``cot="full"``), and every sampling loop runs inside
``torch.inference_mode()`` on bfloat16 weights for peak throughput.

Two further optimizations keep a generation press fast:

* **No repeat weight verification.** Resolution is offline-first (the cache
  ``prefetch_weights()`` fills is looked up, never re-negotiated) and the
  safetensors digests are memoised in a sidecar file, so the ~7 GB identity scan
  runs once ever instead of on every launch. See ``install_hash_cache()``.
* **Decode stays resident.** The VAE audio decoder is loaded into VRAM once at
  startup and kept there, replacing the per-track offload/reload shuffle that
  ``YuE2Pipeline.decode()`` performs. See ``_install_resident_decoder()``.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import threading
import time
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# Transformers base classes (imported eagerly, before any model/codec load)
# --------------------------------------------------------------------------
# YuE2 ships custom modeling code (modeling_yue2.py) whose classes subclass the
# transformers base types below. transformers' dynamic module loader evaluates
# that file in a fresh namespace and resolves these names there, so they must
# already be imported into this module's namespace. Without this, loading can
# fail with `Could not import module 'PreTrainedModel'`.
import prompt_tags
import transformers
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

try:  # optional so the module imports without a GPU stack
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# --------------------------------------------------------------------------
# Paths / defaults
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = str(PROJECT_DIR / "models")
DEFAULT_TEMP_DIR = str(PROJECT_DIR / ".temp")   # hidden scratch space
DEFAULT_OUTPUT_DIR = str((PROJECT_DIR.parent if PROJECT_DIR.name == "Core Files" else PROJECT_DIR) / "Outputs")
DEFAULT_CHECKPOINT = r"C:\Comfy\models\checkpoints\yue2_3b_int8_convrot.safetensors"
DEFAULT_VAE_ID = "m-a-p/YuE2-Vae"
DEFAULT_MODEL_ID = "m-a-p/YuE2-3B"

# INT8 "convrot" checkpoint published for the ComfyUI build.
INT8_REPO_ID = os.environ.get("YUE2_INT8_REPO", "Comfy-Org/YuE2")
INT8_FILENAME = os.environ.get("YUE2_INT8_FILE", "checkpoints/yue2_3b_int8_convrot.safetensors")

# Small config/tokenizer files needed to instantiate the architecture.
CONFIG_PATTERNS = [
    "config.json",
    "generation_config.json",
    "yue2_generation_config.json",
    "qwen.tiktoken",
]

SAMPLE_RATE = 48_000
CHANNELS = 2
# Every rendered track gets a sidecar with its request, ABC score IDs and song
# (codec) tokens, saved next to the MP3. extend_track() continues from it.
TOKENS_SUFFIX = ".yue2tokens.npz"

# Songs land within this many seconds of the requested length: shorter scores
# are continued, and the song-token stage may only stop inside this window.
LENGTH_TOLERANCE_SECONDS = 25

# Instrumental vocal guard: regenerations allowed per take.
VOCAL_GUARD_RETRIES = 3
# Classifier-free guidance for instrumentals: pushes the song tokens toward the
# style text ("Instrumental ... no vocals"). A/B on three seeds of a prompt that
# kept drawing vocals: 39 s sung at the default 1.0, 22 s at 1.5, for ~50% more
# song-token time. Upstream calls CFG an experiment; here it is measured.
INSTRUMENTAL_CFG = 1.5

# Length of one extension, in seconds.
EXTEND_MIN_SECONDS = 10.0
EXTEND_MAX_SECONDS = 120.0

# Duration -> semantic token bounds. The YuE2 VAE decodes one semantic token per
# `downsampling_ratio` audio samples at `sample_rate` (modeling_vae.YuE2VAEConfig:
# 48000 / 1920), i.e. 25 codec tokens per second. Measured on-device:
# 250 tokens -> 10.0s of audio. Duration (s) becomes a max_tokens bound on the
# semantic sampler, which is where a "song length" lives in the native protocol.
TOKENS_PER_SECOND = 25
MIN_SEMANTIC_TOKENS = 200
MAX_SEMANTIC_TOKENS = 9000

# Quality % (10-100) maps linearly onto the audio sampler's ODE steps (10-50).
MIN_QUALITY = 10.0
ODE_STEPS_MIN = 10
ODE_STEPS_MAX = 50

# Generation quality ceiling. `generate_song()` defaults to it, so a caller that
# passes nothing (or an out-of-range value) still gets peak-quality output.
MAX_QUALITY = 100.0

# ABC (symbolic score) planning is hardcoded ON for every generation: the
# chord-annotated plan hands the semantic sampler the richest musical context,
# which is what produces the best audio. The studio therefore exposes no toggle
# for it. "auto" is the studio's word for "always plan the whole score", and it
# resolves to the protocol's chord-annotated "full" instruction.
ABC_MODE = "auto"
ABC_MODE_RESOLUTION = {
    "auto": "full",
    "full": "full",
    "melody": "melody",
    "off": "off",
}

# Weight-digest cache: the file lives in the hidden .temp scratch folder next to
# the session/preset JSON files.
HASH_CACHE_FILENAME = "yue2_sha256_cache.json"

# Staging folder for checkpoints picked with [ ðŸ“‚ Switch Model ]. Deliberately
# *outside* the project: the project lives in OneDrive, which would try to
# sync/upload a multi-GB weight file and reading it back through the sync engine
# is an order of magnitude slower than a plain local read. The system temp
# folder is on the same volume as the Hugging Face cache, so staging can
# hardlink instead of copying.
CHECKPOINT_STAGE_DIR = str(Path(tempfile.gettempdir()) / "yue2_checkpoints")

# Keep at least this much VRAM free before pinning the audio decoder there. The
# backbone is already resident when this is checked, so the reserve covers the
# decode activations for a full-length song.
DECODER_VRAM_RESERVE_GIB = 2.0

# Decode tile size while the backbone stays resident. Upstream picks 1024 when it
# has the GPU to itself (it offloads the backbone first); with the ~7 GiB
# backbone still in VRAM, 1024-frame tiles push the decode peak to ~12.3 GiB.
# 512 is upstream's own small-budget setting and roughly halves that spike.
RESIDENT_VAE_CORE_FRAMES = 512
# Below this much VRAM the decoder is loaded per track instead (12 GB cards).
RESIDENT_DECODER_MIN_GIB = 14.0


def _wants_no_vocals(style: str) -> bool:
    """True for instrumental requests (the recipe's "no vocals" condition)."""
    return "no vocals" in (style or "").lower()


def _join_lyrics(existing: str, section: str, *, before: bool) -> str:
    """Add a new section's lyrics to a song's lyrics.

    An "[instrumental]"-only song has no words to keep, so the new section's
    lyrics replace that tag instead of contradicting it.
    """
    section = section.strip()
    existing = existing.strip()
    if not existing or (existing.startswith("[") and existing.endswith("]") and "instrumental" in existing.lower()):
        return section + "\n"
    parts = [section, existing] if before else [existing, section]
    return "\n".join(parts) + "\n"


def resolve_abc_mode() -> str:
    """Return the protocol ``cot`` instruction for the hardcoded ABC mode.

    ABC planning cannot be switched off from outside the engine: whatever
    ``ABC_MODE`` is set to, an unrecognised value falls back to the
    chord-annotated "full" plan so maximum-quality structure still runs.
    """
    return ABC_MODE_RESOLUTION.get(str(ABC_MODE).strip().lower(), "full")


def purge_hf_dynamic_modules() -> None:
    """Delete the Hugging Face `modules` cache before loading custom code.

    transformers' dynamic module loader caches compiled copies of
    `modeling_yue2.py`; a stale cache triggers `Could not import module
    'PreTrainedModel'`. Clearing it forces a clean re-evaluation.
    """
    candidates = {
        Path.home() / ".cache" / "huggingface" / "modules",
        Path(tempfile.gettempdir()) / "huggingface" / "modules",
    }
    for env in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"):
        value = os.environ.get(env)
        if value:
            base = Path(value)
            candidates.add(base / "modules")
            candidates.add(base)
    for path in candidates:
        if path.name == "modules" and path.is_dir():
            try:
                shutil.rmtree(path, ignore_errors=True)
                print(f"[YuE2] Cleared HF dynamic module cache: {path}")
            except Exception:
                pass


def patch_metadata_guard() -> bool:
    """Work around transformers <= 4.46 crashing on metadata-less safetensors.

    ``m-a-p/YuE2-3B``'s ``model.safetensors`` ships without a ``__metadata__``
    block, and transformers 4.46.3 reads that block unguarded::

        File "transformers/modeling_utils.py", line 3997, in from_pretrained
            if metadata.get("format") == "pt":
        AttributeError: 'NoneType' object has no attribute 'get'

    Later releases treat a missing block as a plain PyTorch checkpoint. This
    shim applies the same rule to the two ``safe_open`` call sites inside
    ``transformers.modeling_utils`` (the weights loaders only), and is a no-op
    on any transformers version that already guards the call.

    Returns True when the shim is active.
    """
    try:
        from transformers import modeling_utils
    except Exception:  # pragma: no cover - transformers unavailable
        return False
    if getattr(modeling_utils, "_yue2_metadata_shim", False):
        return True

    original = getattr(modeling_utils, "safe_open", None)
    if original is None or getattr(original, "_yue2_shim", False):
        return False

    class _Handle:
        """Forwards to the real handle, reporting empty metadata as PyTorch."""

        __slots__ = ("_handle",)

        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def metadata(self) -> Dict[str, str]:
            value = self._handle.metadata()
            return {"format": "pt"} if value is None else value

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    class _SafeOpen:
        """Context-manager proxy around ``safetensors.safe_open``."""

        __slots__ = ("_inner",)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = original(*args, **kwargs)

        def __enter__(self) -> "_Handle":
            return _Handle(self._inner.__enter__())

        def __exit__(self, *exc: Any) -> Any:
            return self._inner.__exit__(*exc)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    def _safe_open(*args: Any, **kwargs: Any) -> "_SafeOpen":
        return _SafeOpen(*args, **kwargs)

    setattr(_safe_open, "_yue2_shim", True)
    modeling_utils.safe_open = _safe_open
    modeling_utils._yue2_metadata_shim = True
    print("[YuE2] Applied safetensors metadata shim (transformers compatibility).")
    return True


def _flash_attention_supported() -> bool:
    """Probe whether this torch build truly has CUDA FlashAttention.

    ``torch.ops.aten._flash_attention_forward`` is *registered* in stock
    CUDA wheels even when the build was not compiled with
    ``USE_FLASH_ATTENTION``, where calling it raises ``RuntimeError:
    USE_FLASH_ATTENTION was not enabled for build.`` Only a real (tiny)
    call tells the two builds apart, so probe the exact shape ``GraphAR``
    uses: one branch, one query, one key-value head group.
    """
    if torch is None or not torch.cuda.is_available():
        return False
    op = getattr(torch.ops.aten, "_flash_attention_forward", None)
    if op is None:
        return False
    try:
        heads, head_dim = 8, 128
        query = torch.randn(1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(2, heads, head_dim, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(2, heads, head_dim, device="cuda", dtype=torch.bfloat16)
        cum_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        cum_k = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        seqused_k = torch.tensor([1], dtype=torch.int32, device="cuda")
        op(query, key, value, cum_q, cum_k, 1, 2, 0.0, False, False, seqused_k=seqused_k)
        return True
    except Exception:
        return False


def patch_flash_attention() -> bool:
    """Route YuE2's CUDA-graph decode around an unusable FlashAttention.

    ``GraphAR`` auto-detects FlashAttention with
    ``hasattr(torch.ops.aten, "_flash_attention_forward")``. The CUDA 12.1
    wheels register that op without compiling it in, so auto-detection picks
    "flash" and generation dies with ``RuntimeError: USE_FLASH_ATTENTION was
    not enabled for build.``

    When the one-off probe fails, ``GraphAR``'s default backend is switched to
    public SDPA, which the same wheels *can* capture into a CUDA graph (their
    cuDNN SDPA entrypoint is rejected during capture). CUDA graphs are kept, so
    the decode stays graph-accelerated.

    Returns True when the shim is active (or was not needed).
    """
    try:
        from yue2 import cuda_graph
    except Exception:  # pragma: no cover - package unavailable
        return False
    if getattr(cuda_graph.GraphAR, "_yue2_backend_shim", False):
        return True
    if _flash_attention_supported():
        return False  # the real op works: nothing to patch

    original = cuda_graph.GraphAR

    class _GraphAR(original):
        """``GraphAR`` with a CUDA-graph-capturable default backend."""

        def __init__(self, model: Any, prefixes: Any, max_tokens: Any, *,
                     capture: bool = True, attention_backend: str = "auto",
                     fuse_projections: bool = False) -> None:
            if attention_backend == "auto":
                attention_backend = "sdpa"
            super().__init__(
                model, prefixes, max_tokens, capture=capture,
                attention_backend=attention_backend, fuse_projections=fuse_projections,
            )

    setattr(_GraphAR, "_yue2_backend_shim", True)
    cuda_graph.GraphAR = _GraphAR
    print("[YuE2] FlashAttention unavailable in this torch build; CUDA-graph decode will use public SDPA.")
    return True


def _fused_gqa_supported() -> bool:
    """Probe whether a fused SDPA kernel accepts grouped-query heads directly.

    YuE2 has 16 query heads over 8 key/value heads. On builds without
    FlashAttention (the Windows CUDA 12.1 wheels), the memory-efficient kernel
    refuses mismatched head counts ("both fused kernels require query, key and
    value to have the same num_heads"), so SDPA silently drops to the math
    kernel, which materialises the full ``heads x N x N`` score matrix.
    """
    if torch is None or not torch.cuda.is_available():
        return True  # nothing to fix off-GPU
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        query = torch.randn(1, 16, 8, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                query, key, key, is_causal=True, enable_gqa=True
            )
        return True
    except Exception:
        return False


def patch_gqa_attention() -> bool:
    """Keep long-sequence attention on a memory-efficient kernel.

    Without this, every multi-token attention call (the ABC/semantic prefills
    and the acoustic NAR stage) runs on the math kernel. For a prompt plus a
    4k-token ABC score plus a few thousand song tokens, its score matrix alone
    is several GiB per layer on top of the ~6.8 GiB backbone. That is the
    out-of-memory failure that hits long songs, and hits shorter ones whenever
    the sampled ABC score happens to run long. Quality (ODE steps) does not
    change it.

    The fix expands K/V to the query head count before attention, which the
    memory-efficient kernel accepts: O(N) memory instead of O(N^2), with the
    same result. Single-token decode steps are left alone, since their math
    kernel cost is tiny.

    Returns True when the shim is active (or was not needed).
    """
    try:
        from yue2 import modeling_yue2, nar
    except Exception:  # pragma: no cover - package unavailable
        return False
    if getattr(modeling_yue2, "_yue2_gqa_shim", False):
        return True
    if _fused_gqa_supported():
        return False  # fused kernels handle GQA natively: nothing to patch

    original_sdpa = modeling_yue2.sdpa
    original_attention = nar.attention

    def sdpa(query: Any, key: Any, value: Any, *, attn_mask: Any = None, is_causal: bool = False) -> Any:
        # [batch, heads, tokens, dim]
        if query.device.type == "cuda" and query.shape[2] > 1 and query.shape[1] != key.shape[1]:
            groups = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        return original_sdpa(query, key, value, attn_mask=attn_mask, is_causal=is_causal)

    def attention(q: Any, k: Any, v: Any, **kwargs: Any) -> Any:
        # [tokens, heads, dim]
        if q.device.type == "cuda" and q.shape[0] > 1 and q.shape[1] != k.shape[1]:
            groups = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        return original_attention(q, k, v, **kwargs)

    modeling_yue2.sdpa = sdpa
    nar.attention = attention
    modeling_yue2._yue2_gqa_shim = True
    print("[YuE2] Grouped-query attention routed to the memory-efficient kernel (long songs fit in VRAM).")
    return True


# --------------------------------------------------------------------------
# Weight-digest cache (skips the multi-GB verification scan on every launch)
# --------------------------------------------------------------------------
# ``yue2.storage.model_identity()`` re-reads and re-hashes every safetensors
# shard to record the run identity. The backbone is ~7 GB, so that scan costs
# 7-25 s of pure I/O per launch and finds nothing new once the pre-fetch has
# succeeded. Digests are immutable for a given (path, size, mtime), so they are
# memoised in a small sidecar JSON and reused.
_HASH_CACHE: Dict[str, Dict[str, Any]] = {}
_HASH_CACHE_DIRTY = False


def hash_cache_path() -> Path:
    """Path of the sidecar digest cache (inside the hidden ``.temp`` folder)."""
    return Path(DEFAULT_TEMP_DIR) / HASH_CACHE_FILENAME


def _load_hash_cache() -> Dict[str, Dict[str, Any]]:
    try:
        payload = json.loads(hash_cache_path().read_text(encoding="utf-8"))
    except Exception:  # missing/corrupt cache: rebuild it
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    return entries if isinstance(entries, dict) else {}


def flush_hash_cache() -> None:
    """Persist newly computed digests. No-op when nothing changed."""
    global _HASH_CACHE_DIRTY

    if not _HASH_CACHE_DIRTY:
        return
    path = hash_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written straight to the destination: OneDrive's indexer can hold a
        # transient handle on freshly created temp files, which breaks the
        # atomic os.replace dance.
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "entries": _HASH_CACHE}, handle, indent=2)
    except Exception as exc:  # pragma: no cover - cache is best effort
        print(f"[YuE2] Could not persist the digest cache: {exc}")
    else:
        _HASH_CACHE_DIRTY = False


def _digest_stamp(path: Path) -> Optional[str]:
    """Cheap fingerprint of a file: size + mtime with nanosecond resolution."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _cached_digest(path: Path, *, original: Callable[[Any], str]) -> str:
    """Return a file's sha256, reusing the cache when size+mtime still match."""
    global _HASH_CACHE_DIRTY

    try:
        key = str(path.resolve())
    except OSError:
        return original(path)

    stamp = _digest_stamp(path)
    if stamp is not None:
        entry = _HASH_CACHE.get(key)
        if isinstance(entry, dict) and entry.get("stamp") == stamp:
            digest = entry.get("sha256")
            if isinstance(digest, str) and digest:
                return digest

    digest = original(path)
    if stamp is not None:
        _HASH_CACHE[key] = {"stamp": stamp, "sha256": digest}
        _HASH_CACHE_DIRTY = True
    return digest


def install_hash_cache(enabled: bool = True) -> bool:
    """Route ``yue2.storage.sha256_file`` through the sidecar digest cache.

    ``model_identity(verify=False)`` skips the digest *comparison* but still
    computes every digest for the run identity, so turning verification off does
    not by itself remove the scan. Memoising the digests is what removes the
    per-launch re-read, and it keeps the recorded identity byte-for-byte
    identical to a full scan.

    Call with ``enabled=False`` to restore the pristine always-rehash
    implementation. Returns True when the cache is in force.
    """
    global _HASH_CACHE, _HASH_CACHE_DIRTY

    try:
        from yue2 import storage
    except Exception:  # pragma: no cover - package not installed yet
        return False

    pristine = getattr(storage, "_yue2_pristine_sha256_file", None)
    if not enabled:
        if pristine is not None:
            storage.sha256_file = pristine
            delattr(storage, "_yue2_pristine_sha256_file")
        return False

    if pristine is None:
        pristine = storage.sha256_file
        setattr(storage, "_yue2_pristine_sha256_file", pristine)
    if not _HASH_CACHE:
        _HASH_CACHE = _load_hash_cache()
        _HASH_CACHE_DIRTY = False

    def sha256_file(path: Any) -> str:
        return _cached_digest(Path(path), original=pristine)

    storage.sha256_file = sha256_file
    return True


def warm_digest_cache(directories: Iterable[str], *, label: str = "weights") -> Dict[str, Any]:
    """Hash cached model files once so later launches can skip the scan.

    Only files whose size/mtime changed are re-read; everything else is served
    from the sidecar. Returns ``{"files", "hashed", "reused", "seconds"}``.
    """
    from yue2 import storage

    install_hash_cache(True)

    targets: List[Path] = []
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            continue
        targets.extend(sorted(root.glob("*.safetensors")))
        config = root / "config.json"
        if config.is_file():
            targets.append(config)

    report: Dict[str, Any] = {"files": len(targets), "hashed": 0, "reused": 0, "seconds": 0.0}
    if not targets:
        return report

    print(f"[YuE2] Checking {label} digests (one-time scan; later launches reuse it)", flush=True)
    start = time.perf_counter()
    for target in targets:
        stamp = _digest_stamp(target)
        entry = _HASH_CACHE.get(str(target.resolve()))
        if stamp is not None and isinstance(entry, dict) and entry.get("stamp") == stamp:
            report["reused"] += 1
            continue
        storage.sha256_file(target)
        report["hashed"] += 1
        size_gib = target.stat().st_size / 2 ** 30
        print(f"[YuE2]   {target.name} ({size_gib:.2f} GiB) hashed", flush=True)

    flush_hash_cache()
    report["seconds"] = time.perf_counter() - start
    print(
        f"[YuE2] {label} digests ready: {report['files']} files, "
        f"{report['hashed']} hashed, {report['reused']} reused "
        f"({report['seconds']:.1f}s)",
        flush=True,
    )
    return report


def _resolve_cached_dir(
    repo_id: str,
    *,
    allow_download: bool,
    what: str,
    token: Optional[str] = None,
) -> str:
    """Resolve a hub repo id to its local snapshot, preferring the local cache.

    ``prefetch_weights()`` has already filled the cache, so the offline resolve
    is a pure cache lookup: no hub round-trip and no re-download prompt on every
    load. The network is only touched when the snapshot is genuinely missing and
    ``allow_download`` permits it.
    """
    from yue2.storage import resolve_model

    try:
        return str(resolve_model(repo_id, local_files_only=True, token=token))
    except Exception as offline_error:
        if not allow_download:
            raise
        print(f"[YuE2] {what} not cached locally ({offline_error}); fetching {repo_id}")
        return str(resolve_model(repo_id, local_files_only=False, token=token))


def _safe_stem(path: Path) -> str:
    """A filesystem-safe staging folder name for a user-selected checkpoint."""
    stem = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in path.stem)
    return stem.strip("._")[:64] or "checkpoint"


# ComfyUI exports wrap the transformer under these prefixes and bundle the VAE in
# the same file; Hugging Face checkpoints use plain ``model.*`` / ``lm_head.*``.
COMFYUI_KEY_PREFIXES = ("model.diffusion_model.", "vae.", "cond_stage_model.")
UNSUPPORTED_QUANTIZATION = ("convrot", "int8")


def _read_safetensors_header(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse a safetensors header without loading a single tensor.

    Returns ``(metadata, tensors)``; both empty when the file cannot be parsed
    (for example a pickle ``.bin`` that was renamed to ``.safetensors``).
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return {}, {}
            length = int.from_bytes(raw, "little")
            if length <= 0 or length > 512 * 1024 * 1024:
                return {}, {}
            header = json.loads(handle.read(length).decode("utf-8"))
    except Exception:
        return {}, {}
    if not isinstance(header, dict):
        return {}, {}
    metadata = header.get("__metadata__")
    metadata = metadata if isinstance(metadata, dict) else {}
    return metadata, {key: value for key, value in header.items() if key != "__metadata__"}


def _reject_quantized_comfy_export(path: Path) -> None:
    """Fail fast, with instructions, on ComfyUI / convrot-quantised checkpoints.

    The native pipeline loads Hugging Face bf16 safetensors. A ComfyUI export
    uses a different key layout (``model.diffusion_model.*`` plus bundled
    ``vae.*`` tensors) *and* a quantised weight representation
    (``quantization: "convrot_int8"``), so it cannot be loaded as-is: it needs a
    conversion step first. Without this probe the problem surfaces as a wall of
    key-mismatch tracebacks in the middle of a load instead of one sentence.
    """
    metadata, tensors = _read_safetensors_header(path)
    quantization = str(metadata.get("quantization", "")).lower()
    comfy_keys = [name for name in tensors if name.startswith(COMFYUI_KEY_PREFIXES)]
    quantised = any(token in quantization for token in UNSUPPORTED_QUANTIZATION)
    if not comfy_keys and not quantised:
        return

    detail = []
    if quantization:
        detail.append(f"quantization={quantization!r}")
    if comfy_keys:
        detail.append(f"{len(comfy_keys)} 'model.diffusion_model.*'-style tensors")
    raise ValueError(
        f"'{path.name}' is a ComfyUI / quantised export ({', '.join(detail)}), not a Hugging "
        "Face bf16 checkpoint, so it cannot be loaded as-is â€” it needs a convrot-INT8 to "
        "HF-bf16 conversion step first. Pick an HF-format .safetensors, or press "
        "[ â†º Default Model ]."
    )


def _seed_hash_cache(source: Path, destination: Path) -> bool:
    """Copy a known digest onto an identical staged copy of the same file.

    Staging renames the weights, which would otherwise miss the cache and force
    a fresh multi-GB hash. The staged file holds the same bytes on disk (a
    hardlink, or a copy that preserved size and mtime), so the digest transfers
    directly. Returns True when the staged file inherited a cached digest.
    """
    global _HASH_CACHE_DIRTY

    try:
        entry = _HASH_CACHE.get(str(Path(os.path.realpath(source)).resolve()))
    except OSError:
        return False
    if not isinstance(entry, dict) or not entry.get("sha256"):
        return False
    stamp = _digest_stamp(destination)
    if stamp is None or stamp != entry.get("stamp"):
        return False  # not provably identical: hash it for real
    try:
        key = str(destination.resolve())
    except OSError:
        return False
    _HASH_CACHE[key] = {"stamp": stamp, "sha256": entry["sha256"]}
    _HASH_CACHE_DIRTY = True
    return True


def prefetch_weights(
    model: Optional[str] = None,
    vae: Optional[str] = None,
    *,
    allow_download: bool = True,
    token: Optional[str] = None,
    cache_digests: bool = True,
) -> Dict[str, str]:
    """Cache the YuE2 backbone + VAE locally before the UI launches.

    Uses the package's own resolver so the downloaded file set matches exactly
    what ``YuE2Pipeline`` will look for (config, tokenizer, weight shards and
    licences). ``huggingface_hub`` streams its own progress bars, so a first
    run shows a clear percentage instead of a silent multi-GB stall.

    When ``cache_digests`` is set (the default) the safetensors digests are
    computed here and stored in the sidecar cache, so the pipeline's integrity
    step on the first Generate press is a cache lookup rather than a 7 GB
    re-read. Already-known files are skipped.

    Returns ``{"backbone": <local dir>, "vae": <local dir>}``. Failures are
    reported and re-raised so the caller can decide whether to continue.
    """
    model = model or DEFAULT_MODEL_ID
    vae = vae or DEFAULT_VAE_ID
    resolved: Dict[str, str] = {}
    for label, repo in (("backbone", model), ("VAE", vae)):
        target = Path(repo).expanduser()
        if target.is_dir():
            print(f"[YuE2] Pre-fetch: {label} is a local folder, using {target}", flush=True)
            resolved[label] = str(target.resolve())
            continue
        print(f"[YuE2] Pre-fetch: {label} <- {repo}", flush=True)
        path = _resolve_cached_dir(
            str(repo),
            allow_download=allow_download,
            what=f"Pre-fetch {label}",
            token=token,
        )
        resolved[label] = path
        print(f"[YuE2] Pre-fetch: {label} ready at {path}", flush=True)

    if cache_digests:
        warm_digest_cache(
            [resolved[key] for key in ("backbone", "VAE") if key in resolved],
            label="Pre-fetch",
        )
    return resolved


_LOAD_LOCK = threading.Lock()


@dataclass
class EngineConfig:
    """Runtime knobs for one engine instance."""

    # Hard-coded HF sources: the UI no longer exposes a model picker.
    checkpoint_path: str = DEFAULT_MODEL_ID
    vae: str = DEFAULT_VAE_ID
    model_id: str = DEFAULT_MODEL_ID
    models_dir: str = DEFAULT_MODELS_DIR
    temp_dir: str = DEFAULT_TEMP_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    int8_repo: str = INT8_REPO_ID
    int8_filename: str = INT8_FILENAME
    device: str = "cuda"
    dtype: str = "bfloat16"
    # YuE2Pipeline knobs. `memory_budget_gib <= 0` auto-sizes to the GPU's VRAM.
    memory_budget_gib: float = 0.0
    quantization: str = "none"  # "none" or "fp8"
    # Weight integrity is verified once, at pre-fetch time (the digests it
    # produces are cached). Re-hashing ~7 GB on every launch finds nothing new,
    # so the pipeline's manifest comparison is off by default. Set True to force
    # a full scan + comparison on every load.
    verify_hashes: bool = False
    use_vllm: bool = False
    allow_download: bool = True
    # Keeps the backbone *and* the audio decoder resident in VRAM between runs.
    keep_in_vram: bool = True


@dataclass
class AbcPlan:
    abc_text: str
    abc_ids: List[int] = field(default_factory=list)
    seed: int = 0
    mode: str = "full"


@dataclass
class AudioTokens:
    tokens: Any = None
    frames: int = 0
    seed: int = 0
    duration: float = 0.0
    sample_rate: int = SAMPLE_RATE


class YuE2Engine:
    # A standalone checkpoint's extension decides the file name it must take
    # inside the staged folder, because that is where the loader looks for it.
    WEIGHT_SUFFIX_TARGETS = {".safetensors": "model.safetensors", ".bin": "pytorch_model.bin"}

    def __init__(self, config: Optional[EngineConfig] = None, **overrides: Any) -> None:
        # Clear any stale/corrupt Hugging Face dynamic-module cache on init so
        # YuE2's custom modeling code re-evaluates cleanly on every launch.
        purge_hf_dynamic_modules()
        # transformers <= 4.46 cannot read the YuE2 backbone's metadata-less
        # safetensors header; install the compatibility shim before any load.
        patch_metadata_guard()
        # The CUDA 12.1 wheels falsely advertise FlashAttention; pick a
        # CUDA-graph-safe attention backend up front.
        patch_flash_attention()
        # ...and without it, long prefills fall back to O(N^2) math attention.
        patch_gqa_attention()
        self.config = config or EngineConfig(**overrides)
        # Enforce CUDA placement explicitly. A backbone that quietly lives on the
        # CPU is a ~100x generation slowdown, so resolve the real device here,
        # from actual CUDA availability, instead of trusting the config default.
        if torch is None:  # pragma: no cover - no GPU stack installed
            self.device = None
            print("[YuE2] WARNING: torch is unavailable; CUDA acceleration is off.")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # A specific card ("cuda:1") for the extra engines of a multi-GPU studio.
            wanted = str(self.config.device)
            if self.device.type == "cuda" and wanted.startswith("cuda:") and                     int(wanted.split(":")[1]) < torch.cuda.device_count():
                self.device = torch.device(wanted)
            if self.device.type != "cuda":
                print("[YuE2] WARNING: CUDA is unavailable; generation will run on the CPU.")
        self.pipe = None
        self.model = None
        self.vae = None
        self.backbone_loaded = False
        self._quality = 100.0
        self._loaded_from: Optional[str] = None
        self._base_generation_config = None
        # Generate and Extend are separate UI events; only one may use the GPU.
        self.gpu_lock = threading.RLock()
        # Vocal-guard findings for the current run, reported back to the UI.
        self.guard_notes: List[str] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None or self.backbone_loaded

    def _torch_dtype(self) -> Any:
        if torch is None:
            return None
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            self.config.dtype, torch.bfloat16
        )

    def _inference_context(self) -> Any:
        """Disable autograd for generation.

        ``torch.inference_mode()`` skips gradient bookkeeping and tensor version
        counters, which is the single biggest CPU-side win during sampling.
        """
        if torch is None:  # pragma: no cover - GPU stack missing
            return nullcontext()
        stack = ExitStack()
        stack.enter_context(self._on_device())
        stack.enter_context(torch.inference_mode())
        return stack

    def _on_device(self) -> Any:
        """Make this engine's card the current one (on this thread) for code that says just "cuda"."""
        if torch is None or self.device is None or self.device.type != "cuda" or self.device.index is None:
            return nullcontext()
        return torch.cuda.device(self.device)

    def _to_device(self, value: Any) -> Any:
        """Recursively move tensors in a request/payload onto the engine device.

        Input tokens and other intermediate tensors must live on the same device
        as the backbone, or torch silently falls back to slow CPU copies.
        """
        if torch is None or self.device is None:
            return value
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            moved = [self._to_device(item) for item in value]
            return tuple(moved) if isinstance(value, tuple) else moved
        return value

    def load(self, model_path: Optional[str] = None) -> "YuE2Engine":
        """Ensure weights exist, then build the YuE2 generation pipeline.

        `model_path` overrides the configured source: a local Hugging Face model
        *directory*, a standalone ``.safetensors``/``.bin`` checkpoint (staged
        into a loadable folder first), or a hub repo id. Reloads if it changes.
        """
        if model_path:
            self.config.checkpoint_path = self._normalise_source(model_path)

        if self.pipe is not None and self._loaded_from == self.config.checkpoint_path:
            return self
        if self.pipe is not None:
            self.unload()

        # One load at a time: engines on other cards share the module cache it rebuilds.
        with _LOAD_LOCK, self._on_device():
            purge_hf_dynamic_modules()
            patch_metadata_guard()
            patch_flash_attention()
            patch_gqa_attention()
            self._load_pipe()
        self._loaded_from = self.config.checkpoint_path
        return self

    # ---- model switching (app.py header buttons) ---------------------- #
    def set_model_checkpoint(self, path: str) -> str:
        """Point the engine at a checkpoint file or HF folder chosen by the user.

        Accepts a standalone ``.safetensors``/``.bin`` file (staged into a
        loadable folder when it is first used) or an existing Hugging Face model
        directory. The loaded pipeline is dropped so the next generation picks
        the new weights up. Returns the resolved source string, and raises with a
        human-readable message when the selection cannot be used.
        """
        raw = str(path or "").strip().strip('"').strip("'")
        if not raw:
            raise ValueError("No checkpoint was selected.")

        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            if not (candidate / "config.json").is_file():
                raise ValueError(f"'{candidate}' is not a Hugging Face model folder (no config.json).")
            source = str(candidate.resolve())
        elif candidate.is_file():
            suffix = candidate.suffix.lower()
            if suffix not in self.WEIGHT_SUFFIX_TARGETS:
                supported = ", ".join(sorted(self.WEIGHT_SUFFIX_TARGETS))
                raise ValueError(
                    f"'{candidate.name}' is not a loadable weights format (supported: {supported})."
                )
            # abspath, not resolve(): a Hub snapshot's shard is a symlink whose
            # target has no extension, and the extension decides the staged name.
            if suffix == ".safetensors":
                _reject_quantized_comfy_export(candidate)
            source = os.path.abspath(str(candidate))
        else:
            raise FileNotFoundError(f"'{raw}' does not exist.")

        previous = self.config.checkpoint_path
        self.config.checkpoint_path = source
        if previous != source:
            self.unload()  # release VRAM; the new weights load on the next run
        print(f"[YuE2] Model source switched to {source}")
        return source

    def reset_model(self) -> str:
        """Point the engine back at the packaged model repo (Default Model)."""
        if self.config.checkpoint_path != DEFAULT_MODEL_ID:
            self.config.checkpoint_path = DEFAULT_MODEL_ID
            self.unload()
        print(f"[YuE2] Model source reset to {DEFAULT_MODEL_ID}")
        return self.config.checkpoint_path

    # ---- pipeline construction (weights + VAE + device placement) ----- #
    def _memory_budget_gib(self) -> float:
        """VRAM the pipeline may use; auto-sizes to the GPU when unset."""
        budget = float(self.config.memory_budget_gib or 0.0)
        if budget > 0:
            return budget
        if self.device is not None and self.device.type == "cuda":
            return torch.cuda.get_device_properties(self.device).total_memory / 2 ** 30
        return 24.0

    # ---- weight resolution (pre-fetched HF cache) --------------------- #
    @staticmethod
    def _normalise_source(value: Optional[str]) -> str:
        """Fall back to the packaged repo when a remembered path has vanished.

        The session file remembers the last model source. A checkpoint chosen
        with ``[ ðŸ“‚ Switch Model ]`` lands in Gradio's temporary upload folder,
        which is wiped between runs, so a remembered absolute path can be gone by
        the next launch. Anything that *looks* like a local path but no longer
        exists falls back to the default instead of failing the load.
        """
        raw = str(value or "").strip().strip('"').strip("'")
        if not raw:
            return DEFAULT_MODEL_ID
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return raw
        looks_local = candidate.is_absolute() or raw.startswith((".", "~", "\\", "/"))
        if looks_local:
            print(f"[YuE2] Remembered model source '{raw}' no longer exists; using {DEFAULT_MODEL_ID}.")
            return DEFAULT_MODEL_ID
        return raw

    def _stage_checkpoint_file(self, source: Path) -> Path:
        """Wrap a standalone weights file in a loadable Hugging Face folder.

        ``YuE2Pipeline`` needs a *directory*: ``config.json``, ``qwen.tiktoken``
        and the tensors have to sit together. A checkpoint picked with
        ``[ ðŸ“‚ Switch Model ]`` only carries tensors, so a staging folder is built
        under ``CHECKPOINT_STAGE_DIR`` (system temp, outside OneDrive) from the
        packaged architecture files, and the chosen weights are linked into it.
        The link is a hardlink when the volume allows it, so a multi-GB
        checkpoint is never duplicated on disk.
        """
        suffix = source.suffix.lower()
        target_name = self.WEIGHT_SUFFIX_TARGETS.get(suffix)
        if target_name is None:
            supported = ", ".join(sorted(self.WEIGHT_SUFFIX_TARGETS))
            raise ValueError(
                f"'{suffix or source.name}' is not a loadable weights format for this engine "
                f"(supported: {supported})."
            )
        if suffix == ".safetensors":
            # Also checked when the model is switched; repeated here so a path
            # set straight on EngineConfig fails the same clear way.
            _reject_quantized_comfy_export(source)

        stage = Path(CHECKPOINT_STAGE_DIR) / _safe_stem(source)
        stage.mkdir(parents=True, exist_ok=True)

        base = Path(
            _resolve_cached_dir(
                DEFAULT_MODEL_ID,
                allow_download=self.config.allow_download,
                what="Default architecture files",
            )
        )
        for name in (*CONFIG_PATTERNS, "modeling_yue2.py", "modeling_vae.py"):
            origin, destination = base / name, stage / name
            if origin.is_file() and not destination.exists():
                shutil.copyfile(origin, destination)

        # A leftover manifest or shard index would describe the *default* shards
        # rather than the selected weights, and would fail (or misdirect) the load.
        for stale in (
            "weights_manifest.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            (stage / stale).unlink(missing_ok=True)

        destination = stage / target_name
        for other in set(self.WEIGHT_SUFFIX_TARGETS.values()) - {target_name}:
            (stage / other).unlink(missing_ok=True)

        # Link the *real* file. On NTFS, os.link() hardlinks a source symlink
        # itself rather than its target, and a Hub shard is a relative symlink
        # that would then dangle from the staging folder.
        real = Path(os.path.realpath(source))
        size = real.stat().st_size
        if destination.is_symlink() or not destination.exists() or destination.stat().st_size != size:
            destination.unlink(missing_ok=True)
            try:
                os.link(real, destination)  # same volume: no second copy
                how = "hardlinked"
            except OSError:
                shutil.copy2(real, destination)
                how = "copied"
            print(f"[YuE2] {how} '{source.name}' into the staging folder.")
        if _seed_hash_cache(source, destination):
            flush_hash_cache()
            print("[YuE2] Staged checkpoint inherited its cached digest (no re-scan).")
        return stage

    def _resolve_model_dir(self) -> str:
        """Return the local Hugging Face snapshot directory for the backbone.

        Weights come from the cache ``prefetch_weights()`` fills, so this is an
        offline lookup whenever the snapshot is already on disk. A local folder
        is used as-is, a selected ``.safetensors`` file is staged first, and
        anything else is treated as a hub repo id (which lets the Switch Model
        button accept a different repo, not just a folder).
        """
        source = str(self.config.checkpoint_path or "").strip()
        candidate = Path(source).expanduser() if source else None
        if candidate is not None and candidate.is_dir() and (candidate / "config.json").is_file():
            print(f"[YuE2] Using local backbone directory {candidate}")
            return str(candidate)
        if candidate is not None and candidate.is_file():
            staged = self._stage_checkpoint_file(candidate)
            print(f"[YuE2] Using selected checkpoint {candidate.name} (staged at {staged})")
            return str(staged)
        is_folder = candidate is not None and candidate.is_dir()
        repo_id = str(self.config.model_id or DEFAULT_MODEL_ID) if (not source or is_folder) else source
        print(f"[YuE2] Resolving backbone from {repo_id}")
        return _resolve_cached_dir(
            repo_id, allow_download=self.config.allow_download, what="Backbone"
        )

    def _resolve_vae_dir(self) -> str:
        """Return the local Hugging Face snapshot directory for the VAE.

        The repo id must be resolved before it reaches ``YuE2Pipeline``: its
        constructor does ``Path(vae_dir)``, so a bare id becomes a *relative*
        folder that does not exist and the load dies with
        ``FileNotFoundError: No safetensors weights in m-a-p\\YuE2-Vae``.
        """
        vae = str(self.config.vae or DEFAULT_VAE_ID)
        path = Path(vae).expanduser()
        if path.is_dir():
            return str(path)
        print(f"[YuE2] Resolving VAE from {vae}")
        return _resolve_cached_dir(
            vae, allow_download=self.config.allow_download, what="VAE"
        )

    @staticmethod
    def _generation_config_for(model_dir: str) -> Any:
        """Load the checkpoint's own ``yue2_generation_config.json`` if present.

        ``YuE2Pipeline.__init__`` otherwise ignores it and silently uses the
        packaged defaults, which would discard the checkpoint's published
        sampler settings.
        """
        from yue2.protocol import GenerationConfig

        path = Path(model_dir) / "yue2_generation_config.json"
        if path.is_file():
            try:
                loaded = GenerationConfig.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                print(f"[YuE2] Ignoring unusable yue2_generation_config.json: {exc}")
            else:
                print("[YuE2] Using the checkpoint's published generation config.")
                return loaded
        return GenerationConfig()

    def _ensure_configs(self) -> None:
        """Guarantee non-None config objects before sampling touches them.

        The sampler reads attributes on the backbone config and the generation
        config. A checkpoint that omits either surfaces as
        ``'NoneType' object has no attribute 'get'``, so substitute the
        packaged defaults instead of raising mid-generation.
        """
        from yue2.modeling_yue2 import YuE2Config
        from yue2.protocol import GenerationConfig

        if self.pipe is None:
            return
        if getattr(self.pipe, "generation_config", None) is None:
            self.pipe.generation_config = GenerationConfig()
            print("[YuE2] Generation config missing -> default GenerationConfig().")
        model = getattr(self.pipe, "_model", None)
        if model is not None and getattr(model, "config", None) is None:
            model.config = YuE2Config()
            print("[YuE2] Backbone config missing -> default YuE2Config().")

    def _load_pipe(self) -> None:
        """Construct the YuE2Pipeline that owns loading, placement & sampling.

        ``YuE2Pipeline`` is the reviewed, complete implementation: it keeps the
        backbone resident on ``self.device`` in bfloat16, pins the cudnn/matmul
        flags, and enforces the VRAM budget. It is handed two *local
        directories* resolved from the pre-fetched cache, so no hub round-trip
        happens here.

        The backbone is materialised eagerly so the device log line below is
        truthful and an out-of-memory failure surfaces here at load, rather
        than in the middle of generation.
        """
        from yue2.pipeline import YuE2Pipeline

        # Digests are cached on disk, so the pipeline's integrity step is a
        # sidecar lookup instead of a 7 GB re-read. verify_hashes=True bypasses
        # the cache and re-hashes + compares everything, as before.
        install_hash_cache(enabled=not self.config.verify_hashes)

        model_dir = self._resolve_model_dir()
        vae_dir = self._resolve_vae_dir()

        kwargs = {
            "device": str(self.device) if self.device is not None else "auto",
            "memory_budget_gib": self._memory_budget_gib(),
            "backend": "vllm" if self.config.use_vllm else "torch",
            "quantization": self.config.quantization,
            "verify_hashes": self.config.verify_hashes,
            "progress": True,
        }
        if self._resident_decoder_fits():
            kwargs["vae_core_frames"] = RESIDENT_VAE_CORE_FRAMES
        self.pipe = YuE2Pipeline(
            model_dir,
            vae_dir,
            generation_config=self._generation_config_for(model_dir),
            **kwargs,
        )
        self._ensure_configs()

        with self._inference_context():  # force the lazy backbone load now
            self.pipe._load_model()
        self._ensure_configs()
        # Keep the shipped GenerationConfig as the per-request baseline, so the
        # Quality slider can override ode_steps without losing anything else.
        self._base_generation_config = self.pipe.generation_config
        print(f"[YuE2] Backbone model loaded onto {self.pipe.device} in bfloat16 precision.")
        flush_hash_cache()

        # Load the audio decoder once, here, and keep it in VRAM instead of
        # letting the pipeline shuffle it in and out around every track.
        if self._resident_decoder_fits():
            self._install_resident_decoder()

    # ---- audio decoder residency (one load, kept in VRAM) -------------- #
    def _resident_decoder_fits(self) -> bool:
        """Keep the decoder resident only on GPUs with room for it next to the backbone.

        Under ~14 GiB (12 GB cards) the backbone plus a resident decoder leaves
        too little headroom, so the pipeline's own per-track offload is used.
        """
        if not self.config.keep_in_vram or self.device is None or self.device.type != "cuda":
            return False
        total = torch.cuda.get_device_properties(self.device).total_memory / 2 ** 30
        return total >= RESIDENT_DECODER_MIN_GIB

    def _install_resident_decoder(self) -> bool:
        """Load the VAE audio decoder into VRAM once and keep it there.

        ``YuE2Pipeline.decode()`` is written for a one-model-at-a-time budget: on
        every call it offloads the ~6.8 GiB backbone to host RAM, pulls the VAE
        decoder up from the CPU, and pushes it back down afterwards. For a
        4-track batch that is several gigabytes shuffled back and forth over
        PCIe, per track.

        Both models fit the RTX 4080 SUPER's ~14 GiB working budget together, so
        the decoder is materialised here, pinned to ``self.device``, and the
        per-call offload is replaced with an otherwise identical resident decode
        (``_resident_decode``). Returns True when the decoder is resident.
        """
        if self.pipe is None or self.device is None:
            return False
        if getattr(self.pipe, "_yue2_resident_decoder", False):
            return True

        try:
            from yue2.modeling_vae import YuE2VAE

            if self.device.type == "cuda":
                free, _total = torch.cuda.mem_get_info(self.device)
                if free < DECODER_VRAM_RESERVE_GIB * 2 ** 30:
                    print(
                        f"[YuE2] Only {free / 2 ** 30:.1f} GiB VRAM free; leaving the audio "
                        "decoder on the CPU (per-track loads)."
                    )
                    return False

            if self.pipe._vae is None:
                self.pipe._vae = YuE2VAE.from_pretrained(
                    self.pipe.vae_dir, decoder_only=True, device="cpu", local_files_only=True
                )
            with self._inference_context():
                self.pipe._vae = self.pipe._vae.to(self.pipe.device).eval()
        except Exception as exc:
            print(f"[YuE2] Could not pin the audio decoder to VRAM ({exc}); using per-track loads.")
            return False
        self.pipe._yue2_upstream_decode = self.pipe.decode
        self.pipe._yue2_resident_decoder = True
        self.pipe.decode = self._resident_decode  # bound: no descriptor rebinding
        resident_gib = sum(p.numel() * p.element_size() for p in self.pipe._vae.parameters()) / 2 ** 30
        print(
            f"[YuE2] Audio decoder loaded into VRAM once ({resident_gib:.2f} GiB) and kept resident."
        )
        return True

    def _resident_decode(self, latents: Any, *, full: bool = False, vae: Any = None) -> Any:
        """``YuE2Pipeline.decode`` with the GPU-resident decoder left in place.

        Mirrors the upstream implementation (latent shaping, halo-cropped tiled
        decode, finiteness check, progress reporting) but skips the offload: the
        backbone and the decoder both stay on the GPU. Falls back to the
        untouched upstream method whenever the decoder is not resident or a
        specific ``vae`` was requested.
        """
        pipe = self.pipe
        upstream = getattr(pipe, "_yue2_upstream_decode", None) if pipe is not None else None
        if pipe is None or vae is not None or not getattr(pipe, "_yue2_resident_decoder", False):
            if upstream is None:
                raise RuntimeError("No resident decoder and no upstream decode() to fall back on.")
            return upstream(latents, full=full, vae=vae)

        model = pipe._vae
        z = torch.as_tensor(latents, dtype=torch.float32)
        if z.ndim == 2 and z.shape[1] == 64:
            z = z.T.unsqueeze(0)
        if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != 64:
            raise ValueError("Expected latents [T,64] or [1,64,T]")

        tiles = 1 if full else (z.shape[-1] + pipe.vae_core_frames - 1) // pipe.vae_core_frames
        with pipe._status("Decoding audio", total=tiles, unit="chunks") as status:
            report = (
                (lambda completed, total: status.update(completed, total=total))
                if pipe.progress
                else None
            )
            with self._inference_context():
                if full:
                    audio = model.decode(z.to(pipe.device)).cpu()
                    status.update(1)
                else:
                    audio = model.decode_tiled(
                        z,
                        core_frames=pipe.vae_core_frames,
                        halo_frames=16,
                        output_device="cpu",
                        on_progress=report,
                    )
                if not torch.isfinite(audio).all():
                    raise ValueError("VAE produced non-finite audio")
                return audio[0].float().clamp(-1, 1).T.contiguous().numpy()

    # ---- duration / quality -> sampler knobs -------------------------- #
    @staticmethod
    def _semantic_sampling(duration: float) -> Dict[str, int]:
        """Map Duration (s) onto the semantic sampler's token bounds.

        YuE2 renders ~50 codec tokens per second, and the sampler validates
        ``0 <= min_tokens <= max_tokens``, so both bounds are derived and
        clamped together.
        """
        seconds = max(5.0, min(float(duration), 300.0))
        max_tokens = int(round(seconds * TOKENS_PER_SECOND))
        max_tokens = max(1, min(max_tokens, MAX_SEMANTIC_TOKENS))
        # The song may end on its own within LENGTH_TOLERANCE_SECONDS of the
        # target, never sooner (it used to be allowed to stop after 8 s).
        floor = int((seconds - LENGTH_TOLERANCE_SECONDS) * TOKENS_PER_SECOND)
        min_tokens = min(max(MIN_SEMANTIC_TOKENS, floor), max_tokens)
        return {"max_tokens": max_tokens, "min_tokens": min_tokens}

    def _apply_ode_steps(self, quality: float) -> int:
        """Map Quality % (10-100) onto the ODE steps (10-50) and return them."""
        safe_quality = max(MIN_QUALITY, min(float(quality), 100.0))
        span = ODE_STEPS_MAX - ODE_STEPS_MIN
        ratio = (safe_quality - MIN_QUALITY) / (100.0 - MIN_QUALITY)
        steps = int(round(ODE_STEPS_MIN + ratio * span))
        steps = max(1, min(steps, ODE_STEPS_MAX))
        if self.pipe is not None:
            base = self._base_generation_config or self.pipe.generation_config
            try:
                self.pipe.generation_config = replace(base, ode_steps=steps)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[YuE2] Could not set ode_steps={steps}: {exc}")
        print(f"[YuE2] Quality {safe_quality:.0f}% -> {steps} ODE steps")
        return steps

    # ---- legacy INT8 / ComfyUI checkpoint path (superseded) ---------- #
    # load() now builds a YuE2Pipeline; the helpers below remain for the raw
    # convrot workflow and are no longer reached from load().
    def _local_candidates(self) -> List[Path]:
        return [
            Path(self.config.models_dir) / self.config.int8_filename,
            Path(self.config.checkpoint_path),
        ]

    def _ensure_weights(self) -> Tuple[Path, Path]:
        """Return (checkpoint, config_dir), downloading into ./models if needed."""
        custom = Path(self.config.checkpoint_path).expanduser()
        if custom.is_dir():
            # A full HF-format model directory (e.g. a BF16 checkpoint folder).
            return custom, custom

        models_dir = Path(self.config.models_dir).expanduser()
        models_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = next((p.expanduser() for p in self._local_candidates() if p.expanduser().is_file()), None)

        if checkpoint is None:
            if not self.config.allow_download:
                raise FileNotFoundError(
                    f"INT8 weights not found in {models_dir}. Expected {self.config.int8_filename}."
                )
            from huggingface_hub import hf_hub_download

            print(f"[YuE2] Downloading {self.config.int8_filename} from {self.config.int8_repo} ...")
            downloaded = hf_hub_download(
                repo_id=self.config.int8_repo,
                filename=self.config.int8_filename,
                local_dir=str(models_dir),
            )
            checkpoint = Path(downloaded)

        config_dir = models_dir / "YuE2-3B"
        if not (config_dir / "config.json").is_file():
            if self.config.allow_download:
                try:
                    from huggingface_hub import snapshot_download

                    print(f"[YuE2] Fetching model config/tokenizer from {self.config.model_id} ...")
                    snapshot_download(
                        repo_id=self.config.model_id,
                        local_dir=str(config_dir),
                        allow_patterns=CONFIG_PATTERNS,
                    )
                except Exception as exc:  # pragma: no cover - offline fallback
                    print(f"[YuE2] Config download failed ({exc}); falling back to hub reference.")
        if not (config_dir / "config.json").is_file():
            config_dir = Path(self.config.model_id)  # let transformers resolve from hub

        return checkpoint, config_dir

    # ---- backbone loading (INT8 convrot or BF16) --------------------- #
    def _load_backbone(self, weights_path: Path, config_dir: Path) -> None:
        import torch
        from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

        # Single source of truth for placement: whatever device the engine
        # resolved on construction (CUDA when available).
        device = self.device if self.device is not None else torch.device("cpu")

        if weights_path.is_dir():
            print(f"[YuE2] Loading backbone from model directory {weights_path}")
            model = YuE2ForCausalLM.from_pretrained(str(weights_path))
            model.to(device=device, dtype=self._torch_dtype())
            model.eval()
            self.model = model
            self.backbone_loaded = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[YuE2] Backbone model loaded onto {device} in bfloat16 precision.")
            return

        from safetensors.torch import load_file

        try:
            config = YuE2Config.from_pretrained(str(config_dir))
        except Exception:
            config = YuE2Config.from_pretrained(self.config.model_id)

        print(f"[YuE2] Loading backbone weights from {weights_path}")
        model = YuE2ForCausalLM(config)
        state = load_file(str(weights_path), device="cpu")
        state = self._dequantize_convrot(state)
        model.load_state_dict(state, strict=False)
        del state
        model.to(device=device, dtype=self._torch_dtype())
        model.eval()
        self.model = model
        self.backbone_loaded = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[YuE2] Backbone model loaded onto {device} in bfloat16 precision.")

    @staticmethod
    def _dequantize_convrot(state: Dict[str, Any]) -> Dict[str, Any]:
        """Dequantize INT8 convrot tensors back to floating point.

        ComfyUI's convrot checkpoints store int8 weights alongside a scale
        companion. Anything already floating point is passed through untouched.
        """
        import torch

        scales: Dict[str, Any] = {}
        for key, value in state.items():
            short = key.rsplit(".", 1)[-1]
            if short in {"weight_scale", "scales", "scale", "weight_scale_inv"}:
                base = key.rsplit(".", 1)[0]
                scales.setdefault(base + ".weight", value)

        out: Dict[str, Any] = {}
        for key, value in state.items():
            if value.dtype == torch.int8 and key in scales:
                scale = scales[key].float()
                weight = value.float() * scale
                out[key] = weight
            else:
                out[key] = value
        return out

    def unload(self) -> None:
        if self.pipe is not None:
            try:
                self.pipe.close()
            except Exception:  # pragma: no cover - cleanup is best effort
                pass
        self.pipe = None
        self.model = None
        self.vae = None
        self.backbone_loaded = False
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "YuE2Engine":
        return self.load()

    def __exit__(self, *exc: Any) -> None:
        if not self.config.keep_in_vram:
            self.unload()

    # ------------------------------------------------------------------ #
    # Stage 1 - symbolic plan
    # ------------------------------------------------------------------ #
    def generate_abc(
        self,
        prompt: str,
        lyrics: str,
        seed: int = -1,
        cfg_scale: Optional[float] = None,
    ) -> AbcPlan:
        """Stage 1: the full chord-annotated ABC score.

        ABC planning is hardcoded on (`ABC_MODE`) so full musical structure and
        chord planning run for every generation; callers cannot disable it.
        """
        seed = random.randint(0, 2**53 - 1) if seed is None or int(seed) < 0 else int(seed)
        cot = resolve_abc_mode()
        request = self._build_request(prompt, lyrics, seed=seed, cot=cot, cfg_scale=cfg_scale)
        self.load()
        with self._inference_context():
            plan = self.pipe.plan(request=request)
        return AbcPlan(plan.abc or "", list(plan.abc_ids), seed, cot)

    # ------------------------------------------------------------------ #
    # Stage 2 - music / codec tokens
    # ------------------------------------------------------------------ #
    def generate_audio_tokens(self, abc_data: AbcPlan | str, duration: float = 30.0, seed: int = -1) -> AudioTokens:
        seed = random.randint(0, 2**53 - 1) if seed is None or int(seed) < 0 else int(seed)
        safe_duration = max(5.0, min(float(duration), 300.0))
        self.load()
        try:
            from yue2.protocol import chunk_ranges  # noqa: F401
        except Exception as exc:  # pragma: no cover - wiring seam
            raise NotImplementedError(
                "Wire this to YuE2ForCausalLM.generate(...) with chunk_ranges()."
            ) from exc
        raise NotImplementedError(
            "Native token sampling is not implemented yet; use generate_song()."
        )

    # ------------------------------------------------------------------ #
    # Stage 3 - waveform decode
    # ------------------------------------------------------------------ #
    def decode_audio(self, audio_tokens: AudioTokens | Any) -> Any:
        self.load()
        tokens = getattr(audio_tokens, "tokens", audio_tokens)
        if tokens is None:
            raise ValueError("No audio tokens supplied to decode_audio().")
        latents = getattr(tokens, "latents", tokens)
        # The pipeline owns the VAE and its device placement.
        with self._inference_context():
            return self.pipe.decode(self._to_device(latents))

    # ------------------------------------------------------------------ #
    # End-to-end generation (public pipeline)
    # ------------------------------------------------------------------ #
    def generate_song(
        self,
        prompt: str,
        lyrics: str,
        duration: float = 30.0,
        seed: int = -1,
        quality: float = MAX_QUALITY,
        cfg_scale: Optional[float] = None,
        batch_count: int = 1,
        output_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        filename_prefix: str = "GrimSloppa",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        no_intro: bool = False,
        no_outro: bool = False,
        first_index: int = 0,
    ) -> Dict[str, Any]:
        """Generate `batch_count` songs and save them as files on disk.

        ``first_index`` numbers this call's songs within a larger batch (their
        seeds and file numbers continue from it), for batches split across GPUs.

        ``no_intro`` removes the planned score's intro section (and any
        [Intro] lyrics), so the song starts in its body; ``no_outro`` does the
        same for a closing outro, so the song doesn't wind down (handy before
        extending it).

        ABC score planning is hardcoded on (see ``ABC_MODE``), so full musical
        structure and chord planning always run. Quality defaults to
        ``MAX_QUALITY``; Duration and Quality only tune the sampler budgets.
        """
        safe_duration = max(5.0, min(float(duration), 300.0))
        safe_quality = max(MIN_QUALITY, min(float(quality), MAX_QUALITY))
        batch_count = max(1, int(batch_count))
        base_seed = random.randint(0, 2**53 - 1) if seed is None or int(seed) < 0 else int(seed)

        dest = self._output_dir(output_dir)

        with self.gpu_lock:
            self.guard_notes = []
            self.load(model_path=model_path)
            # Duration -> semantic token bounds; Quality -> audio sampler ODE steps.
            semantic_sampling = self._semantic_sampling(safe_duration)
            self._apply_ode_steps(safe_quality)
            # ABC planning is always the full chord-annotated plan.
            abc_mode = resolve_abc_mode()
            print(f"[YuE2] ABC score planning: '{ABC_MODE}' -> cot='{abc_mode}'")
            files: List[str] = []
            # The whole batch runs under inference_mode: no autograd graph is built
            # across the sampling loop, which is what makes generation fast.
            with self._inference_context():
                for index in range(batch_count):
                    item_seed = base_seed + first_index + index
                    request = self._build_request(
                        prompt, lyrics, seed=item_seed, cot=abc_mode, cfg_scale=cfg_scale, quality=safe_quality
                    )
                    instrumental = prompt_tags.is_instrumental(lyrics)
                    if instrumental:
                        request = self._instrumental_request(request, safe_duration, no_intro=no_intro, no_outro=no_outro)
                    else:
                        request = self._planned_request(request, safe_duration, no_intro, no_outro)
                    payload = asdict(request) if hasattr(request, "__dataclass_fields__") else dict(request)
                    song = self.pipe(  # type: ignore[misc]
                        **self._to_device(payload),
                        semantic_sampling=semantic_sampling,
                    )
                    plan = song.semantic.plan
                    codec, audio = song.semantic.tokens, song.audio
                    if instrumental:
                        codec, audio = self._vocal_guard(plan, codec, audio, item_seed)

                    out_path = self._unique_output(dest, f"{filename_prefix}_{item_seed}_{first_index + index + 1}")
                    self._write_mp3(audio, song.sample_rate, out_path)
                    self._save_track_tokens(out_path, plan.request, plan.abc_ids, codec)
                    files.append(str(out_path.resolve()))

                    if progress_callback:
                        progress_callback(index + 1, batch_count)

        return {
            "files": files,
            "seed": base_seed,
            "dir": str(dest),
            "duration": safe_duration,
            "quality": safe_quality,
            "sample_rate": SAMPLE_RATE,
        }

    def _vocal_guard(
        self, plan: Any, codec: List[int], audio: Any, seed: int, *, check_from: int = 0
    ) -> Tuple[List[int], Any]:
        """Regenerate sung passages out of an instrumental song.

        Neither the score (no vocal notes) nor the prompt ("no vocals") can stop
        YuE2's audio stage improvising a vocal; one appeared 2 minutes into a
        3.5-minute instrumental whose score had zero vocal notes. So the audio
        is checked with Demucs, and only the sung window (from 2 s before to
        3 s after) is regenerated with a new seed, forced to the same length,
        with the original music spliced back in after it. Regenerating
        everything to the end instead tended to give the model a fresh chance
        to sing later on. If singing survives every retry, the take with the
        fewest sung seconds is kept. ``check_from`` (in song tokens) leaves an
        earlier part untouched. Returns the (codec, audio) to keep.
        """
        import stem_splitter
        from yue2.pipeline import SemanticResult
        from yue2.protocol import CODEC_OFFSET, negative_prefix

        if not stem_splitter.available():
            return codec, audio
        pipe = self.pipe
        request = plan.request
        best = None
        for attempt in range(VOCAL_GUARD_RETRIES + 1):
            passages = stem_splitter.vocal_passages(audio, SAMPLE_RATE, from_second=check_from / TOKENS_PER_SECOND)
            sung = sum(end - start for start, end in passages)
            if best is None or sung < best[0]:
                best = (sung, codec, audio)
            if not passages:
                if attempt:
                    print(f"[YuE2] Vocal guard: clean after {attempt} fix(es).")
                return codec, audio
            if attempt == VOCAL_GUARD_RETRIES:
                break
            start, end = passages[0]
            cut = max(check_from, int((start - 2) * TOKENS_PER_SECOND))
            stop = min(len(codec), int((end + 3) * TOKENS_PER_SECOND))
            window = max(stop - cut, TOKENS_PER_SECOND)
            print(f"[YuE2] Vocal guard: singing at {start}-{end}s; rewriting "
                  f"{cut / TOKENS_PER_SECOND:.0f}-{stop / TOKENS_PER_SECOND:.0f}s.")
            context = [token + CODEC_OFFSET for token in codec[:cut]]
            negative = None
            if request.guidance != 1:
                negative = negative_prefix(request, pipe.tokenizer, plan.abc_ids) + context
            # Exact length inside the song, so the original music after it still fits.
            at_end = stop >= len(codec)
            sampling = replace(
                pipe.generation_config.semantic,
                min_tokens=int(window * 0.8) if at_end else window,
                max_tokens=window,
            )
            ids, _timing, _truncated = pipe._generate(
                plan.prefix + context, sampling, seed + 7919 * (attempt + 1), "semantic",
                negative=negative, cfg_scale=request.guidance, legacy_off=request.cot == "off",
            )
            codec = codec[:cut] + [int(token) - CODEC_OFFSET for token in ids] + codec[stop:]
            audio = pipe.decode(pipe.synthesize(SemanticResult(plan, codec, {}, False)))
        sung, codec, audio = best
        print(f"[YuE2] Vocal guard: kept the take with the least singing ({sung}s).")
        passages = stem_splitter.vocal_passages(audio, SAMPLE_RATE, from_second=check_from / TOKENS_PER_SECOND)
        if passages:
            spots = ", ".join(f"{s // 60}:{s % 60:02d}-{e // 60}:{e % 60:02d}" for s, e in passages[:3])
            self.guard_notes.append(f"possible singing left at {spots}")
        return codec, audio

    def _plan_score(self, request: Any, seconds: Optional[float], no_intro: bool,
                    no_outro: bool = False) -> Optional[str]:
        """Plan the score up front, then shape it: intro/outro removed, length trimmed.

        Returns the score to render, or None when the planner gave nothing.
        """
        plan = self.pipe.plan(request=request)
        if not plan.abc:
            return None
        score = prompt_tags.trim_truncated_score(plan.abc) if plan.truncated else plan.abc
        if no_intro:
            shorter = prompt_tags.drop_intro(score)
            print("[YuE2] No intro: " + ("removed the intro section." if shorter != score
                                         else "the score has no separate intro to remove."))
            score = shorter
        if no_outro:
            shorter = prompt_tags.drop_outro(score)
            print("[YuE2] No outro: " + ("removed the outro section." if shorter != score
                                         else "the score has no separate outro to remove."))
            score = shorter
        if seconds:
            score = self._lengthen_score(request, score, seconds)
            # A little past the song's length, so the music doesn't run out early.
            score = prompt_tags.trim_score_to_seconds(score, seconds * 1.1 + 10)
            if no_outro:  # the continuation may have written its own ending
                score = prompt_tags.drop_outro(score)
        return score

    def _lengthen_score(self, request: Any, score: str, seconds: float) -> str:
        """Continue a score that is clearly shorter than the requested length.

        The song follows its score and ends with it, so a short plan (short
        lyrics, or an intro removed) made songs end well before Seconds. The
        score is continued the way Extend does it, aiming a little past the
        target; songs then land within ~30 s of Seconds.
        """
        have = prompt_tags.score_seconds(score)
        if not have or have >= seconds - LENGTH_TOLERANCE_SECONDS:
            return score
        tokenizer = self.pipe.tokenizer
        ids = tokenizer.encode(score)
        wanted = round(len(ids) * (seconds * 1.1 - have) / have)
        reserve = int(seconds * 1.2 * TOKENS_PER_SECOND)
        added = self._continue_abc(replace(request, abc=None), ids, wanted, reserve, request.seed)
        longer = tokenizer.decode(ids + added)
        print(f"[YuE2] Score was {have:.0f}s for a {seconds:.0f}s song; continued to "
              f"{prompt_tags.score_seconds(longer):.0f}s.")
        return longer

    def _planned_request(self, request: Any, seconds: float, no_intro: bool, no_outro: bool = False) -> Any:
        """A sung song with its score planned and shaped up front (length, intro, outro)."""
        score = self._plan_score(request, seconds, no_intro, no_outro)
        if not score:
            return request
        lyrics = request.lyrics
        if no_intro:
            lyrics = prompt_tags.drop_intro_lyrics(lyrics) or lyrics
        if no_outro:
            lyrics = prompt_tags.drop_outro_lyrics(lyrics) or lyrics
        return replace(request, abc=score, lyrics=lyrics)

    def _instrumental_request(self, request: Any, seconds: Optional[float] = None, *, no_intro: bool = False,
                              no_outro: bool = False) -> Any:
        """Apply upstream's instrumental recipe to a request (no singing).

        A bare "[instrumental]" lyric only *asks* for no vocals; YuE2's planner
        can still write a vocal melody, which then gets sung. Upstream's
        instrumental workflow avoids that: plan the score with the instrumental
        style wording and placeholder section tags, move every vocal note onto
        the instrumental voice, then render that score with the section tags
        as lyrics. If the move can't be done exactly (upstream refuses rather
        than guess) the planned score is used as-is.
        """
        planning = replace(
            request,
            style=prompt_tags.instrumental_style(request.style),
            lyrics=prompt_tags.INSTRUMENTAL_PLAN_LYRICS,
            cfg_scale=request.cfg_scale if request.cfg_scale is not None else INSTRUMENTAL_CFG,
        )
        planned = self._plan_score(planning, seconds, no_intro, no_outro)
        if not planned:
            return planning
        try:
            score = prompt_tags.to_instrumental_score(planned, keep_chords=True)
        except Exception as exc:
            print(f"[YuE2] Instrumental: kept the planned score ({exc})")
            score = planned
        else:
            print("[YuE2] Instrumental: moved the vocal melody onto the instrumental voice.")
        return replace(planning, abc=score, lyrics=prompt_tags.section_tags(score))

    @staticmethod
    def _export_mp3(src_path: Path, dest_path: Path) -> str:
        """Convert a rendered artifact to MP3 (soundfile first, ffmpeg fallback)."""
        try:
            import soundfile as sf  # type: ignore

            data, rate = sf.read(str(src_path), always_2d=True)
            sf.write(str(dest_path), data, rate, format="MP3")
        except Exception as sf_exc:  # pragma: no cover - depends on libsndfile build
            try:
                import subprocess

                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(src_path),
                        "-codec:a", "libmp3lame", "-qscale:a", "2", str(dest_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            except Exception as ff_exc:
                raise RuntimeError(
                    f"MP3 export failed (soundfile: {sf_exc}; ffmpeg: {ff_exc}). "
                    "Install ffmpeg or an MP3-enabled libsndfile."
                ) from ff_exc
        return str(Path(dest_path).resolve())

    # ------------------------------------------------------------------ #
    # Extension (continue a track, or add an intro)
    # ------------------------------------------------------------------ #
    def extend_track(
        self,
        source: str,
        seconds: float = 30.0,
        *,
        direction: str = "after",
        lyrics: Optional[str] = None,
        seed: int = -1,
        quality: float = MAX_QUALITY,
        output_dir: Optional[str] = None,
        no_outro: bool = False,
    ) -> Dict[str, Any]:
        """Lengthen a generated track, working from its saved song tokens.

        ``direction="after"`` is a true continuation: the ABC score is continued
        from where it stopped, then the song tokens are continued from the last
        one, with the whole existing track as context. That is exactly the
        model's own generation process, resumed.

        ``direction="before"`` adds an intro. YuE2 only writes forward, so an
        intro cannot be conditioned on the music that follows it. Instead a new
        opening is sampled from the track's own prompt and score and placed in
        front. Rendering the joined sequence in one pass keeps the join
        acoustically smooth, but it is not musically planned.

        ``lyrics`` are words for the new section. They are added to the song's
        lyrics (after them for a continuation, before them for an intro), so
        the model writes the new section to sing them and later extensions
        keep them.

        The whole track is re-rendered with its original seed, so the existing
        part stays close to the source but is not bit-identical. The result is
        saved as a new MP3 (with its own token sidecar, so it can be extended
        again). ``no_outro`` (continuations only) first cuts the track's
        ending off, its outro section and/or a fade-out, so the new music
        carries the song on instead of starting over after it has wound down.

        Returns ``{"file", "seconds_added", "seam", "seed"}``; ``seam`` is where
        the new music meets the old, in seconds into the result.
        """
        from yue2.pipeline import SemanticResult, SymbolicPlan
        from yue2.protocol import CODEC_OFFSET, CONTEXT, negative_prefix, token_prefixes

        if direction not in ("after", "before"):
            raise ValueError("direction must be 'after' or 'before'")
        source_path = Path(source)
        request, abc_ids, codec = self._load_track_tokens(source_path)
        if lyrics and lyrics.strip():
            request = replace(request, lyrics=_join_lyrics(request.lyrics, lyrics, before=direction == "before"))
        seconds = max(EXTEND_MIN_SECONDS, min(float(seconds), EXTEND_MAX_SECONDS))
        new_tokens = int(round(seconds * TOKENS_PER_SECOND))
        seed = random.randint(0, 2**53 - 1) if seed is None or int(seed) < 0 else int(seed)
        dest = self._output_dir(output_dir) if output_dir else source_path.parent

        with self.gpu_lock:
            self.guard_notes = []
            self.load()
            self._apply_ode_steps(quality)
            pipe = self.pipe
            tokenizer = pipe.tokenizer
            with self._inference_context():
                if direction == "after" and no_outro:
                    codec, abc_ids = self._cut_outro(source_path, codec, abc_ids, tokenizer)
                if direction == "after" and abc_ids and request.cot != "off":
                    # New score tokens follow the track's own score-to-audio ratio.
                    wanted = round(new_tokens * len(abc_ids) / max(1, len(codec)))
                    abc_ids = abc_ids + self._continue_abc(
                        request, abc_ids, wanted, len(codec) + new_tokens, seed
                    )

                prefix = token_prefixes(request, tokenizer, abc_ids)
                negative = negative_prefix(request, tokenizer, abc_ids) if request.guidance != 1 else None
                if direction == "after":
                    # The existing song becomes the prompt for what comes next.
                    context_ids = [token + CODEC_OFFSET for token in codec]
                    prefix_ar = prefix + context_ids
                    negative_ar = negative + context_ids if negative is not None else None
                else:
                    prefix_ar, negative_ar = prefix, negative

                room = CONTEXT - max(len(prefix_ar), len(negative_ar or []))
                if new_tokens > room:
                    raise ValueError(
                        f"This track is too long to extend by {seconds:.0f}s; there is room for about "
                        f"{max(room, 0) // TOKENS_PER_SECOND}s more."
                    )
                # A continuation may end naturally in its last quarter; an intro
                # fills its full length because the original song follows it.
                min_tokens = int(new_tokens * 0.75) if direction == "after" else new_tokens
                sampling = replace(
                    pipe.generation_config.semantic, min_tokens=min_tokens, max_tokens=new_tokens
                )
                print(f"[YuE2] Extending '{source_path.name}' {direction} by up to {seconds:.0f}s")
                ids, _timing, _truncated = pipe._generate(
                    prefix_ar, sampling, seed, "semantic",
                    negative=negative_ar, cfg_scale=request.guidance, legacy_off=request.cot == "off",
                )
                added = [int(token) - CODEC_OFFSET for token in ids]
                combined = codec + added if direction == "after" else added + codec

                abc_text = tokenizer.decode(abc_ids) if abc_ids else None
                plan = SymbolicPlan(request, abc_text, abc_ids, prefix)
                latents = pipe.synthesize(SemanticResult(plan, combined, {}, False))
                audio = pipe.decode(latents)
                if direction == "after" and _wants_no_vocals(request.style):
                    # Only the new section is checked; the existing music stays.
                    combined, audio = self._vocal_guard(plan, combined, audio, seed, check_from=len(codec))
                    added = combined[len(codec):]

        suffix = "ext" if direction == "after" else "intro"
        out_path = self._unique_output(dest, f"{source_path.stem}_{suffix}")
        self._write_mp3(audio, SAMPLE_RATE, out_path)
        self._save_track_tokens(out_path, request, abc_ids, combined)
        seam = len(codec) if direction == "after" else len(added)
        return {
            "file": str(out_path.resolve()),
            "seconds_added": len(added) / TOKENS_PER_SECOND,
            "seam": seam / TOKENS_PER_SECOND,
            "seed": seed,
        }

    def redo_section(
        self,
        source: str,
        start: float,
        end: Optional[float] = None,
        *,
        seed: int = -1,
        quality: float = MAX_QUALITY,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write a new take of part of a generated track, from its saved song tokens.

        Everything before ``start`` is kept and becomes the prompt, exactly as
        in a continuation; the selected stretch is sampled again, forced to
        its original length, and the original music after ``end`` follows it.
        The joined tokens are rendered in one pass, so the join at ``end`` is
        acoustically smooth (the new part isn't planned around what follows,
        so now and then the change is audible; another run fixes that). The
        score is unchanged, so the section keeps its words and melody plan:
        it's a new performance of the same part. ``end`` of None (or within
        half a second of the end) redoes everything from ``start`` on.

        Returns ``{"file", "start", "end", "seed"}``, the redone stretch in
        seconds into the result.
        """
        from yue2.pipeline import SemanticResult, SymbolicPlan
        from yue2.protocol import CODEC_OFFSET, CONTEXT, negative_prefix, token_prefixes

        source_path = Path(source)
        request, abc_ids, codec = self._load_track_tokens(source_path)
        total = len(codec)
        cut = max(0, min(int(round(float(start) * TOKENS_PER_SECOND)), total - TOKENS_PER_SECOND))
        stop = total if end is None else int(round(float(end) * TOKENS_PER_SECOND))
        if stop >= total - TOKENS_PER_SECOND // 2:
            stop = total
        if stop - cut < 2 * TOKENS_PER_SECOND:
            raise ValueError("Select at least 2 seconds of the track to redo.")
        to_end = stop == total
        new_tokens = stop - cut
        seed = random.randint(0, 2**53 - 1) if seed is None or int(seed) < 0 else int(seed)
        dest = self._output_dir(output_dir) if output_dir else source_path.parent

        with self.gpu_lock:
            self.guard_notes = []
            self.load()
            self._apply_ode_steps(quality)
            pipe = self.pipe
            tokenizer = pipe.tokenizer
            with self._inference_context():
                prefix = token_prefixes(request, tokenizer, abc_ids)
                negative = negative_prefix(request, tokenizer, abc_ids) if request.guidance != 1 else None
                context_ids = [token + CODEC_OFFSET for token in codec[:cut]]
                prefix_ar = prefix + context_ids
                negative_ar = negative + context_ids if negative is not None else None
                if new_tokens > CONTEXT - max(len(prefix_ar), len(negative_ar or [])):
                    raise ValueError("This part of the track is too far in to redo; try a shorter track.")
                # Inside the song the new take must fill the gap exactly; at the end it may finish early.
                sampling = replace(pipe.generation_config.semantic,
                                   min_tokens=int(new_tokens * 0.75) if to_end else new_tokens, max_tokens=new_tokens)
                print(f"[YuE2] Redoing '{source_path.name}' from {cut / TOKENS_PER_SECOND:.1f}s"
                      + (" to the end" if to_end else f" to {stop / TOKENS_PER_SECOND:.1f}s"))
                ids, _timing, _truncated = pipe._generate(
                    prefix_ar, sampling, seed, "semantic",
                    negative=negative_ar, cfg_scale=request.guidance, legacy_off=request.cot == "off",
                )
                added = [int(token) - CODEC_OFFSET for token in ids]
                combined = codec[:cut] + added + codec[stop:]

                abc_text = tokenizer.decode(abc_ids) if abc_ids else None
                plan = SymbolicPlan(request, abc_text, abc_ids, prefix)
                audio = pipe.decode(pipe.synthesize(SemanticResult(plan, combined, {}, False)))
                if _wants_no_vocals(request.style):
                    combined, audio = self._vocal_guard(plan, combined, audio, seed, check_from=cut)

        out_path = self._unique_output(dest, f"{source_path.stem}_redo")
        self._write_mp3(audio, SAMPLE_RATE, out_path)
        self._save_track_tokens(out_path, request, abc_ids, combined)
        return {
            "file": str(out_path.resolve()),
            "start": cut / TOKENS_PER_SECOND,
            "end": (cut + len(added)) / TOKENS_PER_SECOND,
            "seed": seed,
        }

    @staticmethod
    def _cut_outro(source_path: Path, codec: List[int], abc_ids: List[int], tokenizer: Any) -> tuple:
        """Cut a track's ending (outro section, fade-out) before continuing it.

        The fade is found in the audio: the tail after the last second that is
        still at least ~-7 dB of the song's typical loudness. A trailing
        "% outro" section is found in the score and mapped onto the audio by
        the score-to-audio length ratio. The earlier of the two wins, but at
        most 40% of the track is cut. Returns ``(codec, abc_ids)``.
        """
        total = len(codec) / TOKENS_PER_SECOND
        cut = total
        try:
            import numpy as np
            import soundfile as sf

            data, rate = sf.read(str(source_path), always_2d=True, dtype="float32")
            mono = data.mean(axis=1)
            secs = len(mono) // rate
            rms = np.array([np.sqrt(np.mean(mono[i * rate:(i + 1) * rate] ** 2)) for i in range(secs)])
            loud = rms[rms > rms.max() * 0.05]
            if len(loud):
                floor = float(np.median(loud)) * 0.45
                last = int(np.nonzero(rms >= floor)[0][-1]) + 1 if (rms >= floor).any() else secs
                if secs - last >= 2:
                    cut = min(cut, float(last))
        except Exception as exc:
            print(f"[YuE2] No outro: couldn't read the audio for a fade ({exc}).")
        abc_text = tokenizer.decode(abc_ids) if abc_ids else ""
        outro_at = prompt_tags.outro_start_seconds(abc_text) if abc_text else 0.0
        score_len = prompt_tags.score_seconds(abc_text) if abc_text else 0.0
        if outro_at and score_len:
            ratio = max(0.5, min(total / score_len, 2.0))
            cut = min(cut, outro_at * ratio)
            abc_ids = tokenizer.encode(prompt_tags.drop_outro(abc_text))
        cut = max(cut, total * 0.6)
        keep = int(cut * TOKENS_PER_SECOND)
        if keep < len(codec):
            print(f"[YuE2] No outro: continuing from {cut:.0f}s of {total:.0f}s (ending cut).")
            codec = codec[:keep]
        else:
            print("[YuE2] No outro: the track has no outro or fade to cut.")
        return codec, abc_ids

    def _continue_abc(
        self, request: Any, abc_ids: List[int], wanted: int, reserve: int, seed: int
    ) -> List[int]:
        """Continue an ABC score so new music has a written plan.

        The ABC stage is autoregressive too, so resuming it from the existing
        score IDs keeps the opening bars identical. ``wanted`` is the number of
        new score tokens; ``reserve`` is how many song tokens must still fit in
        the context window after the score, and caps ``wanted``.
        """
        from yue2.protocol import CONTEXT, token_prefixes

        pipe = self.pipe
        # [EOD] + text + [ABC_START]; the score itself is supplied as IDs.
        base = token_prefixes(replace(request, abc=None), pipe.tokenizer)
        # Leave room for [ABC_END, MUSIC_START] and the song tokens.
        room = CONTEXT - (len(base) + len(abc_ids) + 2 + reserve)
        wanted = min(max(32, int(wanted)), room)
        if wanted < 16:
            return []
        sampling = replace(pipe.generation_config.abc, min_tokens=int(wanted * 0.75), max_tokens=wanted)
        ids, _timing, _truncated = pipe._generate(base + abc_ids, sampling, seed, "abc")
        return [int(token) for token in ids]

    def cover_song(
        self,
        abc: str,
        style: str,
        lyrics: str,
        *,
        score_seconds: float,
        extra_seconds: float = 0.0,
        seed: int = -1,
        quality: float = MAX_QUALITY,
        batch_count: int = 1,
        output_dir: Optional[str] = None,
        filename_prefix: str = "GrimSloppa",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Perform a supplied melody-only ABC score (a cover), optionally continued.

        This is the native YuE2 cover route: the score replaces the symbolic
        planner (``cot="melody"``) and the song is generated to follow it. With
        ``extra_seconds`` the score is first continued autoregressively, so the
        cover carries on past the end of the source. Each take is saved like a
        generated track (MP3 + token sidecar), so it can be extended or split.

        ``score_seconds`` is the source length; it sizes the song-token budget
        and the score continuation.
        """
        from yue2.protocol import SongRequest

        extra_seconds = max(0.0, min(float(extra_seconds), EXTEND_MAX_SECONDS))
        total_seconds = max(5.0, float(score_seconds) + extra_seconds)
        target = int(round(total_seconds * TOKENS_PER_SECOND))
        # Let the song follow the score to its end, with headroom either side.
        max_tokens = min(int(target * 1.15) + TOKENS_PER_SECOND * 5, MAX_SEMANTIC_TOKENS)
        semantic_sampling = {"max_tokens": max_tokens, "min_tokens": min(int(target * 0.8), max_tokens)}
        safe_quality = max(MIN_QUALITY, min(float(quality), MAX_QUALITY))
        base_seed = random.randint(0, 2**53 - 1) if seed is None or int(seed) < 0 else int(seed)
        dest = self._output_dir(output_dir)

        with self.gpu_lock:
            self.guard_notes = []
            self.load()
            self._apply_ode_steps(safe_quality)
            tokenizer = self.pipe.tokenizer
            source_ids = tokenizer.encode(abc)
            files: List[str] = []
            with self._inference_context():
                for index in range(max(1, int(batch_count))):
                    item_seed = base_seed + index
                    request = SongRequest(
                        style=style, lyrics=lyrics, cot="melody", seed=item_seed,
                        cfg_scale=INSTRUMENTAL_CFG if _wants_no_vocals(style) else None,
                    )
                    abc_ids = list(source_ids)
                    if extra_seconds:
                        wanted = round(len(abc_ids) * extra_seconds / max(1.0, float(score_seconds)))
                        abc_ids += self._continue_abc(request, abc_ids, wanted, max_tokens, item_seed)
                    song = self.pipe(
                        **asdict(replace(request, abc=tokenizer.decode(abc_ids))),
                        semantic_sampling=semantic_sampling,
                    )
                    plan = song.semantic.plan
                    codec, audio = song.semantic.tokens, song.audio
                    if _wants_no_vocals(style):
                        codec, audio = self._vocal_guard(plan, codec, audio, item_seed)
                    out_path = self._unique_output(dest, f"{filename_prefix}_cover_{item_seed}_{index + 1}")
                    self._write_mp3(audio, song.sample_rate, out_path)
                    self._save_track_tokens(out_path, plan.request, plan.abc_ids, codec)
                    files.append(str(out_path.resolve()))
                    if progress_callback:
                        progress_callback(index + 1, batch_count)
        return {"files": files, "seed": base_seed}

    # ---- track files (MP3 + token sidecar) ----------------------------- #
    @staticmethod
    def tokens_path(audio_path: Any) -> Path:
        """The token sidecar that sits next to a rendered track."""
        path = Path(audio_path)
        return path.with_name(path.stem + TOKENS_SUFFIX)

    @classmethod
    def can_extend(cls, audio_path: Any) -> bool:
        return cls.tokens_path(audio_path).is_file()

    @classmethod
    def track_request(cls, audio_path: Any) -> Dict[str, Any]:
        """A track's saved request (style, lyrics, ...) from its sidecar, or {}."""
        import numpy as np

        path = cls.tokens_path(audio_path)
        if not path.is_file():
            return {}
        try:
            with np.load(path, allow_pickle=False) as data:
                return json.loads(str(data["request"]))
        except Exception:
            return {}

    def _save_track_tokens(self, audio_path: Path, request: Any, abc_ids: Any, codec: Any) -> None:
        import numpy as np

        np.savez_compressed(
            self.tokens_path(audio_path),
            # The score is kept as IDs only; the request's ABC text would go
            # stale once the score is continued.
            request=np.array(json.dumps(replace(request, abc=None).to_dict())),
            abc_ids=np.asarray(list(abc_ids), dtype=np.int32),
            semantic=np.asarray(list(codec), dtype=np.int32),
        )

    def _load_track_tokens(self, audio_path: Path) -> Tuple[Any, List[int], List[int]]:
        import numpy as np
        from yue2.protocol import SongRequest

        path = self.tokens_path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"'{Path(audio_path).name}' has no saved song tokens ({path.name}), so it can't be "
                "extended. Only tracks generated by this version of the studio can be."
            )
        with np.load(path, allow_pickle=False) as data:
            request = SongRequest(**json.loads(str(data["request"])))
            return request, data["abc_ids"].tolist(), data["semantic"].tolist()

    def _output_dir(self, output_dir: Optional[str]) -> Path:
        dest = Path(output_dir).expanduser() if output_dir else Path(self.config.output_dir)
        if not dest.is_absolute():
            dest = PROJECT_DIR / dest
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    @staticmethod
    def _unique_output(dest: Path, stem: str) -> Path:
        out_path = dest / f"{stem}.mp3"
        counter = 1
        while out_path.exists():
            out_path = dest / f"{stem}_{counter}.mp3"
            counter += 1
        return out_path

    def _write_mp3(self, audio: Any, sample_rate: int, out_path: Path) -> None:
        """Render to FLAC in the hidden .temp folder, then export only the MP3."""
        import soundfile as sf  # type: ignore

        work_root = Path(self.config.temp_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="yue2_", dir=str(work_root)))
        try:
            flac = tmp / "audio.flac"
            sf.write(str(flac), audio, sample_rate, subtype="PCM_24")
            self._export_mp3(flac, out_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_request(
        self,
        prompt: str,
        lyrics: str,
        seed: int,
        cot: str,
        cfg_scale: Optional[float],
        quality: float = MAX_QUALITY,
    ) -> Any:
        # SongRequest is frozen; quality is carried on the engine and will be
        # mapped to Sampling/ode_steps once the native stage-2 sampler lands.
        self._quality = max(10.0, min(float(quality), 100.0))
        try:
            from yue2.protocol import SongRequest  # type: ignore

            return SongRequest(style=prompt, lyrics=lyrics, cot=cot, seed=int(seed), cfg_scale=cfg_scale)
        except ModuleNotFoundError:  # yue2 not installed yet: keep a plain mapping
            return {
                "style": prompt,
                "lyrics": lyrics,
                "cot": cot,
                "seed": int(seed),
                "cfg_scale": cfg_scale,
            }

    def save_tokens(self, tokens: AudioTokens | Any, path: str) -> str:
        import numpy as np

        array = getattr(tokens, "tokens", tokens)
        if torch is not None and hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        np.savez_compressed(path, tokens=array)
        return os.path.abspath(path)

    def load_tokens(self, path: str) -> Any:
        import numpy as np

        with np.load(path) as data:
            return data["tokens"]

    def save_waveform(self, waveform: Any, path: str, sample_rate: int = SAMPLE_RATE) -> str:
        import numpy as np
        import soundfile as sf  # type: ignore

        data = waveform
        if torch is not None and hasattr(data, "detach"):
            data = data.detach().cpu().float().numpy()
        data = np.asarray(data)
        if data.ndim == 2:
            data = data.T
        sf.write(path, data, sample_rate)
        return os.path.abspath(path)


__all__ = [
    "YuE2Engine",
    "EngineConfig",
    "AbcPlan",
    "AudioTokens",
    "prefetch_weights",
    "purge_hf_dynamic_modules",
    "patch_metadata_guard",
    "patch_flash_attention",
    "patch_gqa_attention",
    "install_hash_cache",
    "warm_digest_cache",
    "flush_hash_cache",
    "hash_cache_path",
    "CHECKPOINT_STAGE_DIR",
    "SAMPLE_RATE",
    "TOKENS_PER_SECOND",
    "TOKENS_SUFFIX",
    "EXTEND_MIN_SECONDS",
    "EXTEND_MAX_SECONDS",
    "ABC_MODE",
    "resolve_abc_mode",
    "MAX_QUALITY",
    "DEFAULT_MODEL_ID",
    "DEFAULT_VAE_ID",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_TEMP_DIR",
    "DEFAULT_OUTPUT_DIR",
]
