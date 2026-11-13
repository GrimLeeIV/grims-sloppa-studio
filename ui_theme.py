"""Look and feel of Grim's Sloppa Studio: SVG icons, theme, CSS and presets.

Everything visual lives here so app.py only deals with layout and behaviour.

* Icons are hand-drawn on a shared 24px grid with round 2px strokes (one
  consistent set instead of emoji). Buttons load them as small SVG files
  written to ``assets/icons``; tabs and the drawer draw them as CSS masks.
* The studio is always dark. Colours, surfaces and fonts are CSS variables
  (``--studio-*``); the Appearance panel (``APPEARANCE_JS``) sets them on
  <html> live and remembers them in localStorage, and ``APPEARANCE_HEAD``
  restores a saved look before the page paints.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict

import gradio as gr

PROJECT_DIR = Path(__file__).resolve().parent
ICON_DIR = PROJECT_DIR / "assets" / "icons"
ICON_STROKE = "#E8EAF2"

# ---------------------------------------------------------------------------
# Icons (24x24 path data, stroked)
# ---------------------------------------------------------------------------
ICONS: Dict[str, str] = {
    "model": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M10 10h4v4h-4zM9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "reset": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    "save": '<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>',
    "load": '<path d="M12 15V3M7 8l5-5 5 5M5 21h14"/>',
    "generate": '<path d="M2 10v4M6 6v12M10 3v18M14 8v8M18 5v14M22 10v4"/>',
    "sparkles": '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2-2.2-.8 2.2-.8z"/>',
    "extend": '<path d="M4 12h12M12 7l5 5-5 5M20 5v14"/>',
    "stems": '<path d="M12 2l9 5-9 5-9-5z"/><path d="M3 12l9 5 9-5M3 17l9 5 9-5"/>',
    "upload": '<path d="M7 18a5 5 0 1 1 .9-9.9A6 6 0 0 1 19 9a4 4 0 0 1 0 8"/><path d="M12 12v9M8 15l4-4 4 4"/>',
    "cover": '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>',
    "install": '<path d="M7 17a5 5 0 1 1 .9-9.9A6 6 0 0 1 19 8a4 4 0 0 1 0 8"/><path d="M12 11v10M8 17l4 4 4-4"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "refresh": '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
    "terminal": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M13 15h4"/>',
    "listen": '<path d="M3 18v-6a9 9 0 0 1 18 0v6"/><path d="M21 19a2 2 0 0 1-2 2h-1v-6h3zM3 19a2 2 0 0 0 2 2h1v-6H3z"/>',
    "phone": '<rect x="7" y="2" width="10" height="20" rx="2"/><path d="M11 18h2"/>',
    "palette":'<circle cx="12" cy="12" r="9"/><circle cx="8" cy="10" r="1.2"/><circle cx="12" cy="7.5" r="1.2"/><circle cx="16" cy="10" r="1.2"/><path d="M12 21a2.5 2.5 0 0 1 0-5h1.5a2.5 2.5 0 0 0 0-5"/>',
}


def _svg(name: str, stroke: str = ICON_STROKE) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
        f'stroke="{stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )


def icon(name: str) -> str:
    """Path of an icon's SVG file for gr.Button(icon=...), written on first use."""
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    path = ICON_DIR / f"{name}.svg"
    content = _svg(name)
    if not path.is_file() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    return str(path)


def _mask(name: str) -> str:
    """A CSS mask URL; the element's background colour paints the icon."""
    data = base64.b64encode(_svg(name, "#000").encode("utf-8")).decode("ascii")
    return f'url("data:image/svg+xml;base64,{data}")'


# ---------------------------------------------------------------------------
# Logo + favicon: a skull whose teeth are a soundwave
# ---------------------------------------------------------------------------
_SKULL = (
    '<path d="M16 3C9.6 3 5 7.7 5 13.6c0 3.3 1.4 5.8 3.5 7.3V25a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-4.1'
    'c2.1-1.5 3.5-4 3.5-7.3C27 7.7 22.4 3 16 3z" fill="url(#g)" stroke="none"/>'
    '<circle cx="11.6" cy="13.8" r="2.7" fill="#0B0C12"/><circle cx="20.4" cy="13.8" r="2.7" fill="#0B0C12"/>'
    '<path d="M16 17.2l-1.5 2.6h3z" fill="#0B0C12"/>'
    '<path d="M11.5 24.2v-1.6M13.8 25v-3.2M16 24.6v-2.4M18.2 25v-3.2M20.5 24.2v-1.6" stroke="#0B0C12" '
    'stroke-width="1.4" stroke-linecap="round"/>'
)


def _skull_svg(size: int, start: str, end: str) -> str:
    # style= rather than stop-color= so CSS variables work for the live logo.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 32 32">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" style="stop-color:{start}"/>'
        f'<stop offset="1" style="stop-color:{end}"/></linearGradient></defs>{_SKULL}</svg>'
    )


LOGO_HTML = (
    '<div class="studio-logo">'
    '<span class="logo-mark">'
    + _skull_svg(34, "var(--studio-accent-2)", "var(--studio-accent)")
    + "</span>"
    '<span class="logo-text"><span class="logo-name">Grim\'s Sloppa</span>'
    '<span class="logo-sub">Studio</span></span></div>'
)

EMPTY_TRACKS_HTML = (
    '<div class="empty-hint">' + _svg("generate", "#8A90A6")
    + "<div><b>No tracks yet.</b></div>"
    "<div>Write a style, add lyrics or [instrumental], then Generate. "
    "Tracks land here, ready to extend, split or cover.</div></div>"
)

FAVICON = (
    "<link rel='icon' href='data:image/svg+xml;base64,"
    + base64.b64encode(_skull_svg(64, "#A78BFA", "#7C3AED").encode("utf-8")).decode("ascii")
    + "'>"
)

GOOGLE_FONTS = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
    "family=Creepster&family=VT323&family=Fira+Code:wght@400;600&"
    "family=Outfit:wght@500;600;700&family=Inter:wght@400;500;600&display=swap'>"
)
HEAD = FAVICON + GOOGLE_FONTS  # + APPEARANCE_HEAD (added in launch_options, defined below)

# ---------------------------------------------------------------------------
# Appearance presets (the drawer)
# ---------------------------------------------------------------------------
ACCENTS = {
    "Violet": "#8B5CF6",
    "Indigo": "#6366F1",
    "Blue": "#3B82F6",
    "Cyan": "#06B6D4",
    "Teal": "#14B8A6",
    "Toxic": "#84CC16",
    "Lime": "#A3E635",
    "Amber": "#F59E0B",
    "Orange": "#F97316",
    "Crimson": "#E11D48",
    "Rose": "#F43F5E",
    "Pink": "#EC4899",
    "Fuchsia": "#D946EF",
    "Bone": "#D6D3D1",
}
DEFAULT_ACCENT_NAME = "Violet"

SURFACES = {
    # name: (page, card, raised, border)
    "Midnight": ("#0A0B10", "#12141C", "#191C27", "#252a3a"),
    "Dusk": ("#0D0A12", "#16121E", "#1F1929", "#30283E"),
    "Charcoal": ("#0F0F11", "#18181B", "#202024", "#2E2E33"),
    "Abyss": ("#050506", "#0C0C0F", "#131317", "#1F1F25"),
}
DEFAULT_SURFACE = "Midnight"

TYPEFACES = {
    # name: (titles, body)
    "Studio": ("'Outfit', system-ui, sans-serif", "'Inter', system-ui, sans-serif"),
    "Grim": ("'Creepster', 'Outfit', cursive", "'Inter', system-ui, sans-serif"),
    "Terminal": ("'VT323', ui-monospace, monospace", "'Fira Code', ui-monospace, monospace"),
    "Clean": ("system-ui, -apple-system, 'Segoe UI', sans-serif", "system-ui, -apple-system, 'Segoe UI', sans-serif"),
}
DEFAULT_TYPEFACE = "Studio"


def _theme_vars() -> str:
    """Default --studio-* variables on :root; the derived accent shades follow
    --studio-accent, so the Appearance panel only has to set the base ones."""
    page, card, raised, border = SURFACES[DEFAULT_SURFACE]
    title, body = TYPEFACES[DEFAULT_TYPEFACE]
    return f"""
:root {{
    --studio-accent: {ACCENTS[DEFAULT_ACCENT_NAME]};
    --studio-page: {page};
    --studio-card: {card};
    --studio-raised: {raised};
    --studio-border: {border};
    --studio-font-title: {title};
    --studio-font-body: {body};
}}
:root, .gradio-container {{
    --studio-accent-2: color-mix(in srgb, var(--studio-accent) 55%, #ffffff);
    --studio-accent-soft: color-mix(in srgb, var(--studio-accent) 18%, transparent);
    --studio-accent-hover: color-mix(in srgb, var(--studio-accent) 85%, #000000);
}}
"""


# Appearance: one panel (always-open picker, swatches, surface, typeface),
# applied as inline CSS variables on <html> and remembered in localStorage.
# The head script applies a saved look before the page paints.
_APPEARANCE_DATA = (
    f"const ACCENTS = {json.dumps(ACCENTS)};"
    f"const SURFACES = {json.dumps({k: list(v) for k, v in SURFACES.items()})};"
    f"const TYPEFACES = {json.dumps({k: list(v) for k, v in TYPEFACES.items()})};"
    "const KEY = 'studio-appearance';"
    "function studioApply(st) {"
    "  const r = document.documentElement.style;"
    "  r.setProperty('--studio-accent', st.accent);"
    "  const s = SURFACES[st.surface] || SURFACES['" + DEFAULT_SURFACE + "'];"
    "  ['page', 'card', 'raised', 'border'].forEach((k, i) => r.setProperty('--studio-' + k, s[i]));"
    "  const t = TYPEFACES[st.typeface] || TYPEFACES['" + DEFAULT_TYPEFACE + "'];"
    "  r.setProperty('--studio-font-title', t[0]); r.setProperty('--studio-font-body', t[1]);"
    "  const names = (tries = 0) => {"
    "    if (!document.querySelector('.logo-name')) { if (tries < 40) setTimeout(() => names(tries + 1), 250); return; }"
    "    document.querySelectorAll('.logo-name').forEach(e => { e.textContent = st.name || \"Grim's Sloppa\"; });"
    "    document.querySelectorAll('.logo-sub').forEach(e => { e.textContent = st.sub || 'Studio'; });"
    "    document.title = (st.name || \"Grim's Sloppa\") + ' ' + (st.sub || 'Studio');"
    "  };"
    "  names();"
    "}"
)
APPEARANCE_HEAD = (
    "<script>(() => {" + _APPEARANCE_DATA +
    "try { const st = JSON.parse(localStorage.getItem(KEY) || 'null'); if (st) studioApply(st); } catch (e) {}"
    "})();</script>"
)

APPEARANCE_HTML = (
    '<div class="ap-root">'
    '<div class="ap-picker"><div class="ap-sv"><div class="ap-knob"></div></div>'
    '<input class="ap-hue" type="range" min="0" max="360" step="1">'
    '<div class="ap-hexrow"><span class="ap-chip"></span><input class="ap-hex" maxlength="7" spellcheck="false"></div>'
    '<div class="ap-names"><input class="ap-name" data-k="name" maxlength="28" spellcheck="false" placeholder="Grim\'s Sloppa" title="Studio name">'
    '<input class="ap-name" data-k="sub" maxlength="28" spellcheck="false" placeholder="Studio" title="Subtitle"></div></div>'
    '<div class="ap-side"><div class="ap-label">Presets</div><div class="ap-swatches"></div>'
    '<div class="ap-label">Surface</div><div class="ap-buttons" data-group="surface"></div>'
    '<div class="ap-label">Typeface</div><div class="ap-buttons" data-group="typeface"></div></div>'
    "</div>"
)

APPEARANCE_JS = _APPEARANCE_DATA + r"""
const st = {accent: '""" + ACCENTS[DEFAULT_ACCENT_NAME] + """', surface: '""" + DEFAULT_SURFACE + """', typeface: '""" + DEFAULT_TYPEFACE + r"""'};
try { Object.assign(st, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) {}
const sv = element.querySelector('.ap-sv'), knob = element.querySelector('.ap-knob');
const hue = element.querySelector('.ap-hue'), hex = element.querySelector('.ap-hex'), chip = element.querySelector('.ap-chip');
const hsv2hex = (h, s, v) => {
  const f = (n) => { const k = (n + h / 60) % 6; return v - v * s * Math.max(0, Math.min(k, 4 - k, 1)); };
  return '#' + [f(5), f(3), f(1)].map(x => Math.round(x * 255).toString(16).padStart(2, '0')).join('');
};
const hex2hsv = (c) => {
  const n = parseInt(c.slice(1), 16), r = (n >> 16 & 255) / 255, g = (n >> 8 & 255) / 255, b = (n & 255) / 255;
  const mx = Math.max(r, g, b), d = mx - Math.min(r, g, b);
  let h = 0;
  if (d) h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [(h * 60 + 360) % 360, mx ? d / mx : 0, mx];
};
let [H, S, V] = hex2hsv(st.accent);
const save = () => { studioApply(st); try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const paint = () => {
  sv.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, hsl(${H}, 100%, 50%))`;
  knob.style.left = (S * 100) + '%'; knob.style.top = ((1 - V) * 100) + '%';
  hue.value = H; hex.value = st.accent; chip.style.background = st.accent;
  element.querySelectorAll('.ap-swatch').forEach(b => b.classList.toggle('on', b.dataset.v.toLowerCase() === st.accent.toLowerCase()));
  element.querySelectorAll('.ap-buttons button').forEach(b => b.classList.toggle('on', st[b.parentElement.dataset.group] === b.dataset.v));
};
const setHSV = () => { st.accent = hsv2hex(H, S, V); save(); paint(); };
let dragging = false;
const pick = (e) => { const r = sv.getBoundingClientRect(); S = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); V = 1 - Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)); setHSV(); };
sv.addEventListener('pointerdown', (e) => { dragging = true; sv.setPointerCapture(e.pointerId); pick(e); });
sv.addEventListener('pointermove', (e) => { if (dragging) pick(e); });
sv.addEventListener('pointerup', () => { dragging = false; });
hue.addEventListener('input', () => { H = parseFloat(hue.value); setHSV(); });
hex.addEventListener('change', () => { const v = hex.value.trim(); if (/^#[0-9a-fA-F]{6}$/.test(v)) { st.accent = v; [H, S, V] = hex2hsv(v); save(); } paint(); });
const sw = element.querySelector('.ap-swatches');
sw.innerHTML = Object.entries(ACCENTS).map(([n, c]) => `<button class="ap-swatch" title="${n}" data-v="${c}" style="--sw:${c}"></button>`).join('');
sw.addEventListener('click', (e) => { const b = e.target.closest('.ap-swatch'); if (!b) return; st.accent = b.dataset.v; [H, S, V] = hex2hsv(st.accent); save(); paint(); });
for (const [group, names] of [['surface', Object.keys(SURFACES)], ['typeface', Object.keys(TYPEFACES)]]) {
  const box = element.querySelector(`[data-group="${group}"]`);
  box.innerHTML = names.map(n => `<button data-v="${n}">${n}</button>`).join('');
  box.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; st[group] = b.dataset.v; save(); paint(); });
}
element.querySelectorAll('.ap-name').forEach(i => {
  i.value = st[i.dataset.k] || '';
  i.addEventListener('input', () => { st[i.dataset.k] = i.value.trim(); save(); });
});
studioApply(st); paint();
"""


# ---------------------------------------------------------------------------
# Gradio theme: dark in both light and dark mode, driven by --studio-* vars
# ---------------------------------------------------------------------------
def _both(**values: str) -> Dict[str, str]:
    """Same value for light and dark mode (sizes and radii have no _dark twin)."""
    probe = gr.themes.Base()
    out = dict(values)
    out.update({f"{key}_dark": value for key, value in values.items() if hasattr(probe, f"{key}_dark")})
    return out


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(radius_size="md", spacing_size="md", text_size="md").set(
        **_both(
            body_background_fill="var(--studio-page)",
            body_text_color="#E8EAF2",
            body_text_color_subdued="#8A90A6",
            background_fill_primary="var(--studio-card)",
            background_fill_secondary="var(--studio-raised)",
            block_background_fill="var(--studio-card)",
            block_border_color="var(--studio-border)",
            block_border_width="1px",
            block_radius="14px",
            block_shadow="none",
            block_label_background_fill="var(--studio-raised)",
            block_label_text_color="#A3A9BC",
            block_label_border_color="var(--studio-border)",
            block_title_text_color="#8A90A6",
            block_title_background_fill="transparent",
            block_title_text_size="11.5px",
            block_title_text_weight="600",
            block_info_text_color="#6B7187",
            block_info_text_size="12px",
            input_background_fill="var(--studio-page)",
            input_background_fill_focus="var(--studio-page)",
            input_border_color="var(--studio-border)",
            input_border_color_focus="var(--studio-accent)",
            input_shadow_focus="0 0 0 3px var(--studio-accent-soft)",
            input_radius="10px",
            button_primary_background_fill="var(--studio-accent)",
            button_primary_background_fill_hover="var(--studio-accent-hover)",
            button_primary_text_color="#FFFFFF",
            button_primary_border_color="transparent",
            button_secondary_background_fill="var(--studio-raised)",
            button_secondary_background_fill_hover="color-mix(in srgb, var(--studio-raised) 80%, #ffffff)",
            button_secondary_text_color="#E8EAF2",
            button_secondary_border_color="var(--studio-border)",
            button_large_radius="12px",
            button_medium_radius="10px",
            button_small_radius="8px",
            border_color_primary="var(--studio-border)",
            border_color_accent="var(--studio-accent)",
            color_accent="var(--studio-accent)",
            color_accent_soft="var(--studio-accent-soft)",
            slider_color="var(--studio-accent)",
            loader_color="var(--studio-accent)",
            checkbox_label_background_fill="var(--studio-raised)",
            checkbox_label_background_fill_selected="var(--studio-accent-soft)",
            checkbox_label_border_color="var(--studio-border)",
            checkbox_label_text_color="#C9CDDA",
            checkbox_label_text_color_selected="#FFFFFF",
            checkbox_background_color="var(--studio-page)",
            checkbox_border_color="var(--studio-border)",
            panel_background_fill="var(--studio-card)",
            panel_border_color="var(--studio-border)",
            link_text_color="var(--studio-accent-2)",
            code_background_fill="var(--studio-raised)",
        )
    )


# ---------------------------------------------------------------------------
# Gradio audio player tweaks (bundle patch)
# ---------------------------------------------------------------------------
# Two player behaviours are hardcoded in Gradio's compiled frontend with no
# option to change them, so the exact expressions are patched in the bundled
# JavaScript at startup. Each patch matches one exact snippet (bundle file
# names are hashed); if a Gradio update changes the code, the patch simply
# doesn't apply and the stock behaviour remains. Idempotent.
GRADIO_PATCHES = [
    # wavesurfer zoom: 20 px per second scrolls long songs; 0 fits the whole track.
    ("whole-track waveform", "normalize:!0,minPxPerSec:20", "normalize:!0,minPxPerSec:0"),
    # The volume slider only renders while toggled open; render it always.
    ("always-visible volume slider", "Fe(U,z=>{o()&&z(Z)})", "Fe(U,z=>{z(Z)})"),
]


PLAYER_PATCH_VERSION = "1"


def patch_gradio_player() -> list:
    """Apply GRADIO_PATCHES; returns the names of the bundle files carrying them."""
    assets = Path(gr.__file__).parent / "templates" / "frontend" / "assets"
    applied, files = set(), []
    for bundle in assets.glob("*.js"):
        text = bundle.read_text(encoding="utf-8")
        patched = text
        for name, old, new in GRADIO_PATCHES:
            if old in patched:
                patched = patched.replace(old, new)
            if new in patched:
                applied.add(name)
        if patched != text:
            bundle.write_text(patched, encoding="utf-8")
        if any(new in patched for _name, _old, new in GRADIO_PATCHES):
            files.append(bundle.name)
    missing = [name for name, _old, _new in GRADIO_PATCHES if name not in applied]
    print(f"[Studio] Player tweaks active: {sorted(applied) or 'none'}"
          + (f"; unavailable in this Gradio version: {missing}" if missing else ""), flush=True)
    return sorted(files)


def player_cache_script(files: list) -> str:
    """Make browsers drop cached pre-patch copies of the patched player files.

    Gradio sends no Cache-Control for its bundles, so a browser that loaded
    the stock player before can keep reusing it. Once per patch version this
    re-fetches just those files with cache: "reload" (updating the cache)
    and reloads the page. Without localStorage it does nothing, so it can't loop.
    """
    if not files:
        return ""
    stamp = PLAYER_PATCH_VERSION + ":" + ",".join(files)
    return (
        "<script>(async () => {"
        f"const files = {files!r}; const key = 'studio-player-patch'; const stamp = {stamp!r};"
        "try { if (localStorage.getItem(key) === stamp) return; } catch (e) { return; }"
        "await Promise.all(files.map(f => fetch('assets/' + f, {cache: 'reload'}).catch(() => null)));"
        "try { localStorage.setItem(key, stamp); } catch (e) { return; }"
        "location.reload();"
        "})();</script>"
    )


# ---------------------------------------------------------------------------
# Static CSS
# ---------------------------------------------------------------------------
_SWATCH_RULES = "\n".join(
    f".swatches label:nth-of-type({i}) {{ --swatch: {color}; }}" for i, color in enumerate(ACCENTS.values(), 1)
)

CSS = (
    _theme_vars()
    + """
/* ---- Base ------------------------------------------------------------ */
.gradio-container { max-width: 100% !important; margin: 0 auto !important; font-family: var(--studio-font-body) !important; }
.gradio-container * { font-family: inherit; }
.gradio-container textarea, .gradio-container input { font-family: var(--studio-font-body) !important; }
footer { display: none !important; }
::selection { background: var(--studio-accent-soft); }
* { scrollbar-width: thin; scrollbar-color: var(--studio-border) transparent; }

/* Input labels: quiet small caps instead of loud pills. */
.gradio-container [data-testid="block-info"] {
    text-transform: uppercase; letter-spacing: .07em; font-size: 11.5px; font-weight: 600;
    color: #8A90A6; padding: 0; background: none;
}
.gradio-container [data-testid="block-info"] + div,
.gradio-container .block-info { font-size: 12px; }

/* ---- Header ---------------------------------------------------------- */
.studio-header { align-items: center !important; gap: 16px; margin-bottom: 4px; flex-wrap: wrap; }
.studio-header > .studio-logo-wrap { flex: 0 0 auto !important; padding: 0 !important; min-width: 0 !important; width: auto !important; }
.studio-header > .header-actions { margin-left: auto; }

/* Model pill: name + Change + Default in one control. */
.model-pill { flex: 0 1 auto !important; width: auto !important; min-width: 0 !important; gap: 6px !important;
    align-items: center !important; background: var(--studio-card); border: 1px solid var(--studio-border);
    border-radius: 12px; padding: 3px 4px 3px 12px !important; }
.model-pill > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.model-pill .html-container, .model-pill .block { padding: 0 !important; background: none !important; border: none !important; }
.pill-label, .field-label { text-transform: uppercase; letter-spacing: .07em; font-size: 11px; font-weight: 600; color: #8A90A6; }
.model-name p { margin: 0 !important; }
.model-name code { background: none !important; border: none !important; color: #E8EAF2 !important;
                   font-size: 12.5px !important; padding: 0 4px !important; }
.model-pill .hdr-btn, .model-pill button { height: 30px !important; min-height: 30px !important; padding: 0 10px !important; }
.studio-logo { display: flex; align-items: center; gap: 12px; padding: 2px 0; }
.logo-mark { display: flex; filter: drop-shadow(0 4px 14px var(--studio-accent-soft)); }
.logo-text { display: flex; flex-direction: column; line-height: 1; }
.logo-name { font-family: var(--studio-font-title) !important; font-size: 24px; font-weight: 700;
             color: #F3F4F8; letter-spacing: -.01em; }
.logo-sub { font-size: 11px; font-weight: 600; letter-spacing: .32em; text-transform: uppercase;
            color: var(--studio-accent-2); margin-top: 5px; }
.header-actions { flex: 0 1 auto !important; width: auto !important; min-width: 0 !important;
                  gap: 8px !important; justify-content: flex-end; flex-wrap: wrap; }
.header-actions > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.hdr-btn, .header-actions button, .header-actions a {
    height: 36px !important; min-height: 36px !important; padding: 0 14px !important;
    font-size: 13px !important; font-weight: 500 !important; white-space: nowrap !important;
    border-radius: 10px !important; gap: 8px !important;
}
.hdr-btn img, .header-actions img { width: 16px !important; height: 16px !important; margin: 0 !important; }

/* ---- Section headings ------------------------------------------------ */
.section-title h3, .section-title p {
    font-family: var(--studio-font-title) !important; font-size: 15px !important; font-weight: 600 !important;
    color: #E8EAF2 !important; margin: 4px 0 2px !important; letter-spacing: .01em;
}
/* Key section labels (Compose, Workflow, Upload): accent-outlined tags. */
.col-head .section-title h3, .tc-legend {
    display: inline-flex !important; align-items: center; font-family: var(--studio-font-title) !important;
    font-size: 12px !important; font-weight: 700 !important; letter-spacing: .14em !important; text-transform: uppercase;
    color: var(--studio-accent-2) !important; background: var(--studio-accent-soft) !important;
    border: 1px solid var(--studio-accent) !important; border-radius: 999px; padding: 5px 14px !important;
    box-shadow: 0 0 18px -6px var(--studio-accent); line-height: 1.2;
}

/* ---- Buttons --------------------------------------------------------- */
.gradio-container button img { width: 17px; height: 17px; }
.generate-btn { height: 52px !important; font-size: 15px !important; font-weight: 600 !important;
    letter-spacing: .02em; border-radius: 12px !important;
    background: linear-gradient(135deg, var(--studio-accent), var(--studio-accent-hover)) !important;
    box-shadow: 0 8px 26px -10px var(--studio-accent) !important; transition: transform .12s, box-shadow .12s; }
.generate-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 12px 30px -10px var(--studio-accent) !important; }
.generate-btn:disabled { opacity: .75; }
.small-btn { height: 32px !important; min-height: 32px !important; font-size: 12.5px !important;
             padding: 0 12px !important; width: auto !important; flex: 0 0 auto !important; }
.action-btn { height: 42px !important; font-weight: 600 !important; }

/* ---- Tag chips ------------------------------------------------------- */
.tag-chips { background: transparent !important; border: none !important; padding: 2px 0 0 !important; }
.tag-chips .wrap { gap: 6px !important; }
.tag-chips label {
    border-radius: 999px !important; padding: 4px 11px !important; font-size: 12.5px !important;
    background: var(--studio-raised) !important; border: 1px solid var(--studio-border) !important;
    color: #B4B9CA !important; cursor: pointer; transition: border-color .12s, color .12s, background .12s;
    box-shadow: none !important;
}
.tag-chips label:hover { border-color: var(--studio-accent) !important; color: #fff !important;
                         background: var(--studio-accent-soft) !important; }
.tag-chips label input { display: none !important; }
.tag-chips label span::before { content: "+"; margin-right: 5px; color: var(--studio-accent-2); font-weight: 600; }
.tag-chips [data-testid="block-info"] { margin-bottom: 4px; }
.prompt-tools { align-items: center !important; justify-content: space-between; gap: 8px; }

/* ---- Compact settings ------------------------------------------------ */
.tight input[type="number"] { font-variant-numeric: tabular-nums; }
.settings-row { gap: 10px !important; }
/* Gradio wraps neighbouring inputs in one .form; keep compact rows on one line. */
.settings-row.tight, .settings-row.tight > .form { flex-wrap: nowrap !important; }
.studio-tabs { margin-top: 6px; }

/* ---- Track + result cards -------------------------------------------- */
.track-card { border: 1px solid var(--studio-border) !important; border-radius: 14px !important;
              overflow: hidden; background: var(--studio-card) !important; }
/* Hidden components aren't rendered: a card without its name line is unused. */
.track-card:not(:has(.card-status)) { display: none !important; }
/* Placeholder only while no card is in use. */
.tracks-area .empty-hint-wrap { display: none !important; }
.tracks-area:not(:has(.track-card .card-status)) .empty-hint-wrap { display: block !important; }
.tracks-area { gap: 10px !important; }
.card-head { align-items: center !important; gap: 6px !important; padding: 8px 10px 0 4px !important; flex-wrap: nowrap !important; }
.card-head > .card-status { flex: 1 1 auto !important; min-width: 0 !important; }
.card-head .card-status p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card-btn { height: 28px !important; min-height: 28px !important; padding: 0 10px !important; font-size: 12px !important;
            flex: 0 0 auto !important; width: auto !important; }
.card-btn img { width: 14px !important; height: 14px !important; }
.feed-notice { border-left: 3px solid var(--studio-accent); background: var(--studio-accent-soft) !important;
               border-radius: 8px; padding: 8px 12px !important; }
.feed-notice p { margin: 0 !important; font-size: 13px !important; color: #E8EAF2 !important; }
.target-line { padding: 4px 2px 8px !important; }
.fine-print p { font-size: 12px !important; color: #8A90A6 !important; }
.panel { gap: 10px !important; }

/* Column title rows: title left, the column's controls right. */
.col-head { align-items: center !important; flex-wrap: nowrap !important; gap: 8px !important; min-height: 44px; }
.col-head > .section-title { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; margin-right: auto !important; }
/* Headers sit snug under the window top, half a gap above their panels; the logo row matches their height. */
.gradio-container main, .gradio-container > .main { padding-top: 6px !important; }
.col-head { margin-bottom: -8px !important; height: 44px; }
.logo-head .studio-logo-wrap, .logo-head .html-container, .logo-head .prose { padding: 0 !important; margin: 0 !important; }
.logo-head { overflow: visible; }

/* Full width, centred: fixed-ish side panels, the canvas takes the rest. */
.gradio-container { width: 100% !important; }
.gradio-container > .main { max-width: none !important; padding: 0 !important; }
.gradio-container main.contain { padding: 6px 20px 0 !important; }
.gradio-container footer { display: none !important; }
.studio-main { flex-wrap: nowrap !important; }
.col-compose { flex: 0 0 clamp(340px, 27vw, 540px) !important; max-width: clamp(340px, 27vw, 540px) !important; }
.col-upload { flex: 0 0 clamp(300px, 20vw, 400px) !important; max-width: clamp(300px, 20vw, 400px) !important; }
.col-canvas { flex: 1 1 0 !important; min-width: 380px !important; }

/* Style / Lyrics headers hug their buttons; the labels are larger and in the accent colour. */
.field-head.field-head { padding: 5px 6px 0 12px !important; min-height: 0 !important; }
.field-head > .field-label-wrap, .field-head > .field-label-wrap * { padding: 0 !important; margin: 0 !important;
                                                                     min-height: 0 !important; line-height: 1.2 !important; }
.field-head .field-label { font-size: 13px; font-weight: 700; letter-spacing: .1em; color: var(--studio-accent-2); }

/* Pill icon buttons: the icon dead-centre. */
.pill.pill { flex-wrap: nowrap !important; }
.pill .pill-label-wrap { padding-right: 2px !important; }
.pill button { width: 30px !important; padding: 0 !important; justify-content: center !important; }
.pill button img { margin: 0 !important; }
.pill button span:empty { display: none !important; }

/* Quality: a chunky rounded bar that's easy to grab anywhere along it. */
.quality-slider input[type="range"] {
    -webkit-appearance: none !important; appearance: none !important; height: 20px !important; border-radius: 7px !important;
    border: 1px solid var(--studio-border) !important; cursor: pointer; margin: 0 !important;
    background: linear-gradient(to right, var(--studio-accent) var(--range_progress, 50%),
                var(--studio-raised) var(--range_progress, 50%)) !important;
}
.quality-slider input[type="range"]::-webkit-slider-runnable-track { height: 20px; background: transparent !important; border: none; }
.quality-slider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 8px; height: 20px; margin-top: 0; border-radius: 4px; background: #fff;
    box-shadow: 0 0 0 2px var(--studio-accent), 0 2px 6px #0008; cursor: grab;
}
.quality-slider input[type="range"]::-moz-range-track { height: 20px; background: transparent; border: none; }
.quality-slider input[type="range"]::-moz-range-thumb { width: 8px; height: 20px; border-radius: 4px; background: #fff; border: none;
    box-shadow: 0 0 0 2px var(--studio-accent); }
.col-head > *:not(.section-title) { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.col-head .section-title h3 { margin: 0 !important; }
.logo-head { justify-content: center !important; }
.logo-head > .studio-logo-wrap { flex: 0 0 auto !important; padding: 0 !important; }
.col-head .model-pill { padding: 2px 3px 2px 10px !important; }
.col-head .model-name code { font-size: 11.5px !important; }

/* Style / Lyrics header rows: label, switch, Improve. */
.field-head { align-items: center !important; gap: 10px !important; padding: 10px 12px 0 !important; flex-wrap: nowrap !important; }
.field-head > .field-label-wrap { flex: 1 1 auto !important; padding: 0 !important; }
.field-head > * { flex: 0 0 auto; width: auto !important; min-width: 0 !important; }
.switch { background: none !important; border: none !important; padding: 0 !important; }
.switch label { gap: 8px !important; font-size: 12.5px !important; color: #C9CDDA !important; background: none !important;
                border: none !important; padding: 0 !important; cursor: pointer; }
.switch input[type="checkbox"] { appearance: none; -webkit-appearance: none; width: 32px !important; height: 18px !important;
    border-radius: 999px !important; background: var(--studio-raised) !important; border: 1px solid var(--studio-border) !important;
    position: relative; cursor: pointer; transition: background .15s; margin: 0 !important; }
.switch input[type="checkbox"]::before { content: ""; position: absolute; top: 2px; left: 2px; width: 12px; height: 12px;
    border-radius: 999px; background: #A3A9BC; transition: transform .15s, background .15s; }
.switch input[type="checkbox"]:checked { background: var(--studio-accent) !important; border-color: var(--studio-accent) !important; }
.switch input[type="checkbox"]:checked::before { transform: translateX(14px); background: #fff; }
.lyrics-group textarea:disabled { opacity: .45; }

/* Suggested tags row with the More button top-right. */
.chips-row { align-items: flex-start !important; gap: 6px !important; }
.chips-row > .tag-chips { flex: 1 1 auto !important; }
.more-btn { margin: 8px 8px 0 0 !important; }
.empty-hint svg { display: block; width: 28px; height: 28px; opacity: .55; margin: 0 auto 8px; }
.track-card .block { border: none !important; box-shadow: none !important; }
.track-card .block * { --border-color-primary: var(--studio-border); }
.card-status { padding: 10px 14px 0 !important; }
.card-status p { font-size: 13px !important; color: #C9CDDA !important; margin: 0 !important; }
.empty-hint { border: 1px dashed var(--studio-border); border-radius: 14px; padding: 22px;
              text-align: center; color: #6B7187; font-size: 13px; }
.empty-hint b { color: #A3A9BC; font-weight: 600; }

/* ---- Tabs with icons --------------------------------------------------- */
.studio-tabs [role="tablist"] button, .studio-tabs .tab-container button {
    font-weight: 600 !important; font-size: 13.5px !important; display: inline-flex; align-items: center;
}
.studio-tabs [role="tab"]::before {
    content: ""; width: 16px; height: 16px; margin-right: 8px; flex: none;
    background-color: currentColor; -webkit-mask: var(--tab-icon) center / contain no-repeat;
    mask: var(--tab-icon) center / contain no-repeat;
}
.studio-tabs [role="tab"]:nth-of-type(1) { --tab-icon: """ + _mask("extend") + """; }
.studio-tabs [role="tab"]:nth-of-type(2) { --tab-icon: """ + _mask("stems") + """; }
.studio-tabs [role="tab"]:nth-of-type(3) { --tab-icon: """ + _mask("upload") + """; }
.studio-tabs [role="tab"][aria-selected="true"] { color: var(--studio-accent-2) !important; }

/* ---- Appearance drawer ---------------------------------------------- */
.appearance { border-radius: 14px !important; }
.appearance > button span:first-child::before, .appearance > .label-wrap span:first-child::before {
    content: ""; display: inline-block; width: 15px; height: 15px; margin-right: 8px; vertical-align: -2px;
    background-color: currentColor; -webkit-mask: """ + _mask("palette") + """ center / contain no-repeat;
    mask: """ + _mask("palette") + """ center / contain no-repeat;
}
.swatches .wrap { gap: 10px !important; }
.swatches label {
    width: 26px !important; height: 26px !important; min-width: 0 !important; padding: 0 !important;
    border-radius: 999px !important; background: var(--swatch) !important; border: 2px solid transparent !important;
    font-size: 0 !important; box-shadow: none !important; cursor: pointer;
}
.swatches label input { display: none !important; }
.swatches label:has(input:checked) { outline: 2px solid var(--swatch); outline-offset: 3px; }
"""
    + _SWATCH_RULES
    + """
.custom-color { min-width: 0 !important; }
/* Accent: picker on the left, swatches stacked beside it. */
.swatches-vertical .wrap { flex-direction: column !important; align-items: center !important; gap: 8px !important; }
.swatches-vertical { padding: 10px 8px !important; }
.accent-row { align-items: stretch !important; }

/* ---- Responsive -------------------------------------------------------- */
@media (max-width: 1600px) {
    /* Narrower side columns: icon-only model buttons, settings rows may wrap. */
    .col-head .model-pill button { font-size: 0 !important; gap: 0 !important; padding: 0 8px !important; }
    .settings-row.tight, .settings-row.tight > .form { flex-wrap: wrap !important; }
}
@media (max-width: 900px) {
    .header-actions { justify-content: flex-start; }
    .logo-name { font-size: 20px; }
}

/* ---- Compose: even, user-resizable text boxes --------------------------- */
.style-group textarea, .lyrics-group textarea { resize: vertical !important; min-height: 76px; }
.tag-chips label { padding: 3px 10px !important; font-size: 12px !important; }

/* Pills (model, preset) in the Upload title row: icon buttons with a label. */
.pill { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; gap: 4px !important;
        align-items: center !important; background: var(--studio-card); border: 1px solid var(--studio-border);
        border-radius: 12px; padding: 2px 3px 2px 10px !important; }
.pill > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.pill .html-container, .pill .block { padding: 0 !important; background: none !important; border: none !important; }
.pill button { height: 28px !important; min-height: 28px !important; padding: 0 7px !important; font-size: 0 !important;
               gap: 0 !important; border-radius: 8px !important; }
.pill button img { width: 15px !important; height: 15px !important; }
.pill .model-name code { font-size: 11px !important; }
.pills { flex: 0 1 auto !important; width: auto !important; gap: 6px !important; flex-wrap: wrap !important;
         justify-content: flex-end !important; }

/* ---- Track canvas ------------------------------------------------------- */
.track-canvas, .track-canvas .html-container { padding: 0 !important; }
.pill .model-name code { display: block; max-width: 100%; overflow: hidden; text-overflow: ellipsis;
                         white-space: nowrap; }
/* The model name gives way (ellipsis, full name on hover) so both pills always fit. */
.col-head > .pills { gap: 4px !important; flex: 0 1 auto !important; min-width: 0 !important; flex-wrap: nowrap !important; }
.pills > .pill:first-child { flex: 0 1 auto !important; min-width: 0 !important; }
.pill > .model-name { flex: 0 1 auto !important; min-width: 0 !important; max-width: 96px; overflow: hidden; }
.pill > .model-name * { min-width: 0 !important; }
.tc-root { position: relative; height: calc(100vh - 110px); min-height: 420px; border: 1px solid var(--studio-border);
           border-radius: 14px; overflow: hidden; background:
           radial-gradient(circle, color-mix(in srgb, var(--studio-border) 70%, transparent) 1px, transparent 1px) 0 0 / 22px 22px,
           var(--studio-page); }
.tc-stage { position: absolute; inset: 0; cursor: grab; touch-action: none; }
.tc-stage.panning { cursor: grabbing; }
.tc-world { position: absolute; left: 0; top: 0; transform-origin: 0 0; }
.tc-links { position: absolute; left: 0; top: 0; overflow: visible; pointer-events: none; }
.tc-links path { fill: none; stroke: color-mix(in srgb, var(--studio-accent) 60%, var(--studio-border)); stroke-width: 2; opacity: .8; }
.tc-nodes { position: absolute; left: 0; top: 0; }
.tc-empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center;
            color: #6B7187; font-size: 13px; line-height: 1.6; pointer-events: none; }
.tc-empty b { color: #A3A9BC; }
.tc-tools { position: absolute; right: 10px; bottom: 10px; display: flex; gap: 6px; }
.tc-tool { width: 32px; height: 32px; padding: 0 !important; border-radius: 9px; border: 1px solid var(--studio-border);
           background: var(--studio-card); color: #C9CDDA; display: grid; place-items: center; cursor: pointer; }
.tc-tool svg { width: 16px; height: 16px; display: block; }
.tc-legend { position: absolute; top: 10px; left: 12px; pointer-events: none; z-index: 2; }
.tc-toast { position: absolute; top: 10px; left: 50%; transform: translateX(-50%); z-index: 3; display: none;
            align-items: flex-start; gap: 10px; max-width: min(560px, calc(100% - 40px)); padding: 9px 10px 9px 14px;
            background: color-mix(in srgb, var(--studio-card) 92%, transparent); backdrop-filter: blur(6px);
            border: 1px solid var(--studio-border); border-left: 3px solid var(--studio-accent); border-radius: 10px;
            box-shadow: 0 12px 30px -14px #000; font-size: 12.5px; line-height: 1.45; color: #E8EAF2; }
.tc-toast.show { display: flex; animation: tcDrop .18s ease-out; }
.tc-toast span { flex: 1 1 auto; }
.tc-toast button { flex: none; width: 22px; height: 22px; padding: 0 !important; border: none; border-radius: 6px;
                   background: none; color: #8A90A6; display: grid; place-items: center; cursor: pointer; }
.tc-toast button:hover { background: var(--studio-raised); color: #fff; }
.tc-toast button svg { width: 14px; height: 14px; display: block; }
@keyframes tcDrop { from { opacity: 0; transform: translate(-50%, -6px); } to { opacity: 1; transform: translateX(-50%); } }
.tc-tool:hover { border-color: var(--studio-accent); color: #fff; }
.tc-card { position: absolute; left: 0; top: 0; background: var(--studio-card); border: 1px solid var(--studio-border);
           border-radius: 14px; padding: 10px 12px; box-shadow: 0 8px 24px -16px #000; cursor: default; user-select: none; }
.tc-card.playing { border-color: color-mix(in srgb, var(--studio-accent) 70%, var(--studio-border)); }
.tc-head { display: flex; align-items: center; gap: 8px; min-height: 30px; }
.tc-title { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; line-height: 1.2; }
.tc-title b { font-size: 13px; color: #E8EAF2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tc-title .tc-sub { font-size: 11px; color: #8A90A6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tc-card.popped .tc-title { flex-basis: 0; }
.tc-actions { flex: 0 0 auto; display: flex; align-items: center; gap: 6px; }
.tc-btn, .tc-icon, .tc-render, .tc-seg button { font: inherit; font-size: 12px; color: #E8EAF2; background: var(--studio-raised);
           border: 1px solid var(--studio-border); border-radius: 8px; height: 28px; cursor: pointer; display: inline-flex;
           align-items: center; gap: 6px; padding: 0 10px; white-space: nowrap; }
.tc-icon { width: 28px; padding: 0; justify-content: center; }
.tc-btn svg, .tc-icon svg { width: 14px; height: 14px; }
.tc-btn:hover, .tc-icon:hover { border-color: var(--studio-accent); }
.tc-pop { display: none; align-items: center; gap: 6px; }
.tc-pop.open { display: flex; animation: tcSlide .14s ease-out; }
@keyframes tcSlide { from { opacity: 0; transform: translateX(-6px); } to { opacity: 1; transform: none; } }
.tc-seg { display: inline-flex; border: 1px solid var(--studio-border); border-radius: 8px; overflow: hidden; }
.tc-seg button { border: none; border-radius: 0; background: var(--studio-page); padding: 0 8px; color: #A3A9BC; }
.tc-seg button.on { background: var(--studio-accent-soft); color: #fff; }
.tc-pop label { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; color: #8A90A6; }
.tc-pop input { width: 44px; height: 26px; border-radius: 7px; border: 1px solid var(--studio-border); background: var(--studio-page);
                color: #E8EAF2; font: inherit; font-size: 12px; padding: 0 4px; }
.tc-render { background: var(--studio-accent); border-color: var(--studio-accent); color: #fff; font-weight: 600; }
.tc-render:hover { background: var(--studio-accent-hover); }
.tc-wave { position: relative; height: 54px; margin: 8px 0 6px; cursor: pointer; }
.tc-stem .tc-wave { height: 30px; margin: 6px 0 4px; }
.tc-bars, .tc-prog { position: absolute; inset: 0; width: 100%; height: 100%; }
.tc-bars rect { fill: #4A5068; }
.tc-bars.on rect { fill: var(--studio-accent); }
.tc-prog { clip-path: inset(0 100% 0 0); }
.tc-cursor { position: absolute; top: 0; bottom: 0; width: 2px; left: 0; background: #E8EAF2; opacity: .7; }
.tc-sel { position: absolute; top: -4px; bottom: -4px; display: none; pointer-events: none; border-radius: 6px;
          background: color-mix(in srgb, var(--studio-accent) 22%, transparent);
          border: 1px solid color-mix(in srgb, var(--studio-accent-2) 80%, transparent);
          box-shadow: 0 0 14px -4px var(--studio-accent); }
.tc-sel.on { display: block; }
.tc-card:has(.tc-sel.on) .tc-wave { cursor: ew-resize; }
.tc-redo-what { font-size: 11.5px; color: #E8EAF2; white-space: nowrap; font-variant-numeric: tabular-nums; }
.tc-card:has([data-pop="redo"].open) .tc-ctrl > :is(.tc-vol, .tc-vol-icon, .tc-volbtn, [data-act="extend"]) { display: none; }
.tc-ctrl { display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: #A3A9BC; }
.tc-play { width: 30px; height: 30px; padding: 0 !important; border-radius: 999px; border: none; background: var(--studio-accent);
           color: #fff; display: grid; place-items: center; cursor: pointer; flex: none; }
.tc-play svg { width: 13px; height: 13px; display: block; }
.tc-time { font-variant-numeric: tabular-nums; flex: none; margin-right: 2px; }
.tc-spacer { flex: 1 1 auto; }
.tc-card:has([data-pop="extend"].open) .tc-ctrl > .tc-vol, .tc-card:has([data-pop="extend"].open) .tc-ctrl > .tc-vol-icon { display: none; }
.tc-card:has([data-pop="extend"].open) [data-act="extend"] span { display: none; }
.tc-card:has([data-pop="extend"].open) [data-act="extend"] { padding: 0 7px; }
.tc-seg button svg { width: 14px; height: 14px; display: block; }
.tc-pop .tc-num { gap: 2px; color: #8A90A6; font-size: 11.5px; }
.tc-pop .tc-num input { width: 36px; text-align: center; padding: 0 2px; }
.tc-mark { position: absolute; top: -3px; bottom: -3px; width: 0; border-left: 2px dashed var(--studio-accent-2);
           opacity: .9; pointer-events: none; }
.tc-mark::before { content: ""; position: absolute; top: -2px; left: -6px; border: 5px solid transparent;
                   border-top: 6px solid var(--studio-accent-2); }
.tc-vol-icon svg { width: 15px; height: 15px; display: block; }
.tc-vol { width: 90px; accent-color: var(--studio-accent); }
.tc-stem .tc-vol, .tc-stem .tc-vol-icon { display: none; }
.tc-pending .tc-shimmer { height: 54px; margin: 8px 0 6px; border-radius: 8px;
    background: linear-gradient(90deg, var(--studio-raised) 0%, var(--studio-accent-soft) 50%, var(--studio-raised) 100%) 0 0 / 200% 100%;
    animation: tcShimmer 1.4s linear infinite; }
.tc-pending { color: #A3A9BC; font-size: 12px; }
@keyframes tcShimmer { to { background-position: -200% 0; } }

/* ---- Appearance panel ------------------------------------------------- */
.ap-root { display: flex; gap: 14px; padding: 6px 2px; }
.ap-picker { flex: 1 1 60%; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.ap-sv { position: relative; height: 150px; border-radius: 10px; cursor: crosshair; touch-action: none; }
.ap-knob { position: absolute; width: 14px; height: 14px; margin: -7px 0 0 -7px; border-radius: 999px; border: 2px solid #fff;
           box-shadow: 0 0 0 1px #0008; pointer-events: none; }
.ap-hue { width: 100%; -webkit-appearance: none; appearance: none; height: 10px; border-radius: 999px; background:
          linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00); }
.ap-hue::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 999px; background: #fff; border: 2px solid #0006; }
.ap-hexrow { display: flex; gap: 8px; align-items: center; }
.ap-names { display: flex; gap: 6px; }
.ap-name { flex: 1 1 0; min-width: 0; height: 28px; border-radius: 8px; border: 1px solid var(--studio-border);
           background: var(--studio-page); color: #E8EAF2; font: inherit; font-size: 12px; padding: 0 8px; }
.ap-name:focus, .ap-hex:focus { outline: none; border-color: var(--studio-accent); }
.ap-chip { width: 28px; height: 28px; border-radius: 8px; border: 1px solid var(--studio-border); flex: none; }
.ap-hex { flex: 1 1 auto; height: 28px; border-radius: 8px; border: 1px solid var(--studio-border); background: var(--studio-page);
          color: #E8EAF2; padding: 0 8px; font-family: ui-monospace, monospace !important; font-size: 12px; }
.ap-side { flex: 1 1 40%; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.ap-label { text-transform: uppercase; letter-spacing: .07em; font-size: 10.5px; font-weight: 600; color: #8A90A6; margin-top: 2px; }
.ap-swatches { display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; }
.ap-swatch { aspect-ratio: 1; border-radius: 999px; border: 2px solid transparent; background: var(--sw); cursor: pointer; padding: 0; }
.ap-swatch.on { outline: 2px solid var(--sw); outline-offset: 2px; }
.ap-buttons { display: flex; flex-wrap: wrap; gap: 4px; }
.ap-buttons button { font: inherit; font-size: 11.5px; padding: 3px 8px; border-radius: 7px; border: 1px solid var(--studio-border);
                     background: var(--studio-raised); color: #C9CDDA; cursor: pointer; }
.ap-buttons button.on { border-color: var(--studio-accent); background: var(--studio-accent-soft); color: #fff; }
"""
)


# ---------------------------------------------------------------------------
# Small control upgrades, applied in the browser as elements appear:
# styled up/down steppers on number fields, and Seed's "Random" chip.
# ---------------------------------------------------------------------------
_CHEVRON_UP = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15l6-6 6 6"/></svg>'
_CHEVRON_DOWN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>'

CONTROLS_HEAD = (
    "<script>(() => {"
    f"const UP = '{_CHEVRON_UP}', DOWN = '{_CHEVRON_DOWN}';"
    r"""
const nudge = (input, dir) => {
  const step = parseFloat(input.step) || 1;
  let v = (parseFloat(input.value) || 0) + dir * step;
  if (input.min !== '' && v < parseFloat(input.min)) v = parseFloat(input.min);
  if (input.max !== '' && v > parseFloat(input.max)) v = parseFloat(input.max);
  input.value = String(v);
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
};
const upgrade = () => {
  document.querySelectorAll('.settings-row input[type="number"]:not([data-stepper]), .tc-pop input[type="number"]:not([data-stepper])').forEach(input => {
    input.dataset.stepper = '1';
    const box = document.createElement('span');
    box.className = 'stepper';
    box.innerHTML = `<button type="button" tabindex="-1" title="Up">${UP}</button><button type="button" tabindex="-1" title="Down">${DOWN}</button>`;
    const [up, down] = box.querySelectorAll('button');
    let timer = null;
    const hold = (dir) => { nudge(input, dir); timer = setTimeout(function rep() { nudge(input, dir); timer = setTimeout(rep, 70); }, 380); };
    const stop = () => { clearTimeout(timer); timer = null; input.dispatchEvent(new Event('blur')); };
    up.addEventListener('pointerdown', (e) => { e.preventDefault(); hold(1); });
    down.addEventListener('pointerdown', (e) => { e.preventDefault(); hold(-1); });
    ['pointerup', 'pointerleave'].forEach(t => { up.addEventListener(t, () => timer && stop()); down.addEventListener(t, () => timer && stop()); });
    const wrap = document.createElement('span');
    wrap.className = 'num-wrap';
    input.parentNode.insertBefore(wrap, input);
    wrap.appendChild(input);
    wrap.appendChild(box);
  });
  document.querySelectorAll('.seed-field:not([data-chip]) [data-testid="block-info"]').forEach(label => {
    const field = label.closest('.seed-field');
    field.dataset.chip = '1';
    const chip = document.createElement('button');
    chip.type = 'button'; chip.className = 'seed-random'; chip.textContent = 'Random';
    chip.title = 'A new random seed every run (-1)';
    chip.addEventListener('click', (e) => {
      e.preventDefault(); e.stopPropagation();
      const input = field.querySelector('input[type="number"]');
      input.value = '-1';
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    });
    label.insertAdjacentElement('afterend', chip);
  });
  document.querySelectorAll('.folder-btn:not([title])').forEach(b => { b.title = 'Choose folder'; });
};
let queued = false;
new MutationObserver(() => { if (!queued) { queued = true; setTimeout(() => { queued = false; upgrade(); }, 50); } })
  .observe(document.documentElement, {childList: true, subtree: true});
"""
    "})();</script>"
)

CSS += """
/* ---- Number steppers in the studio's style ------------------------------- */
.settings-row input[type="number"]::-webkit-inner-spin-button,
.settings-row input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
.settings-row input[type="number"] { -moz-appearance: textfield; }
.num-wrap { position: relative; display: block; width: 100%; }
.num-wrap input[type="number"] { padding-right: 26px !important; width: 100%; }
.stepper { position: absolute; right: 4px; top: 50%; transform: translateY(-50%); display: flex; flex-direction: column;
           gap: 2px; opacity: 0; transition: opacity .12s; }
.num-wrap:hover .stepper, .num-wrap:focus-within .stepper { opacity: 1; }
.stepper button { width: 18px; height: 13px; padding: 0 !important; border-radius: 4px; border: 1px solid var(--studio-border);
                  background: var(--studio-raised); color: var(--studio-accent-2); display: grid; place-items: center;
                  cursor: pointer; min-width: 0 !important; }
.stepper button:hover { border-color: var(--studio-accent); background: var(--studio-accent-soft); color: #fff; }
.stepper button svg { width: 10px; height: 10px; display: block; }
.quality-slider .stepper { right: 3px; }
.quality-slider .stepper button { height: 11px; }

/* Seed: "Random" chip beside the label. */
.seed-field label { display: flex !important; flex-wrap: wrap; align-items: center; column-gap: 6px; }
.seed-field label > .num-wrap { flex: 1 0 100%; }
.seed-field label > [data-testid="block-info"] { flex: 0 0 auto !important; width: auto !important; margin-bottom: 0 !important; }
.seed-field label > .seed-random { margin: 0 0 6px; }
.seed-field label > [data-testid="block-info"] { margin-bottom: 6px !important; }
.seed-random { font: inherit; font-size: 10.5px !important; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
               padding: 1px 8px !important; border-radius: 999px; border: 1px solid var(--studio-accent);
               background: var(--studio-accent-soft); color: var(--studio-accent-2); cursor: pointer; line-height: 1.5; }
.seed-random:hover { background: var(--studio-accent); color: #fff; }

/* Output folder: a browse button inside the field's right edge. */
.folder-row { position: relative; }
.folder-field input, .folder-field textarea { padding-right: 44px !important; }
.folder-btn.folder-btn { position: absolute !important; right: 17px; bottom: 16px; width: 30px !important; height: 30px !important;
                         min-width: 0 !important; min-height: 0 !important; padding: 0 !important; border-radius: 8px !important;
                         z-index: 2; display: grid !important; place-items: center; flex: none !important;
                         background: var(--studio-raised) !important; border: 1px solid var(--studio-border) !important; }
.folder-btn.folder-btn:hover { border-color: var(--studio-accent) !important; background: var(--studio-accent-soft) !important; }
.folder-btn img { width: 15px !important; height: 15px !important; margin: 0 !important; }
.folder-btn span:empty { display: none !important; }

/* Workflow badge in the logo row, level with Compose and Upload; the logo stays centred. */
.logo-head { position: relative; justify-content: flex-start !important; }
.logo-head > .studio-logo-wrap { position: absolute !important; left: 50%; top: 0; transform: translateX(-50%); }

/* Rounded-rectangle tool buttons that sit inside their headers. */
.small-btn.small-btn { border-radius: 9px !important; height: 28px !important; min-height: 28px !important; padding: 0 11px !important; }
.field-head.field-head { padding: 6px 8px 6px 12px !important; }
.more-btn.more-btn { margin: 6px 8px 0 0 !important; }

/* Suggested tags line up with the field labels. */
.chips-row .tag-chips { padding: 4px 0 6px 12px !important; }
.tag-chips .wrap { margin-left: 0 !important; padding-left: 0 !important; }
.tag-chips [data-testid="block-info"] { margin-left: 0 !important; padding-left: 0 !important; }

/* Scrollbars in the studio's colours. */
* { scrollbar-width: thin; scrollbar-color: color-mix(in srgb, var(--studio-accent) 55%, var(--studio-border)) transparent; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: color-mix(in srgb, var(--studio-accent) 45%, var(--studio-border)); border-radius: 999px;
                            border: 2px solid transparent; background-clip: padding-box; }
::-webkit-scrollbar-thumb:hover { background: var(--studio-accent); background-clip: padding-box; }
::-webkit-scrollbar-corner { background: transparent; }

/* Collapsible panels in the Upload column, each with its icon. */
.studio-acc > button span:first-child::before, .studio-acc > .label-wrap span:first-child::before {
    content: ""; display: inline-block; width: 15px; height: 15px; margin-right: 8px; vertical-align: -2px;
    background-color: currentColor; -webkit-mask: var(--acc-icon) center / contain no-repeat; mask: var(--acc-icon) center / contain no-repeat;
}
.studio-acc { border-radius: 14px !important; }
.acc-source { --acc-icon: """ + _mask("upload") + """; }
.acc-console { --acc-icon: """ + _mask("terminal") + """; }
.acc-phone { --acc-icon: """ + _mask("phone") + """; }
.acc-interpret { --acc-icon: """ + _mask("listen") + """; }

/* Console: the studio's log, newest at the bottom. */
.console-log { margin: 0; height: 260px; overflow: auto; padding: 10px 12px; border-radius: 10px; background: var(--studio-page);
               border: 1px solid var(--studio-border); font: 11.5px/1.45 'Fira Code', ui-monospace, monospace !important;
               color: #B4B9CA; white-space: pre-wrap; word-break: break-word; }
.phone-card { display: flex; gap: 14px; align-items: center; font-size: 12.5px; color: #C9CDDA; line-height: 1.5; }
.phone-card svg { width: 132px; height: 132px; flex: none; background: #fff; border-radius: 10px; padding: 4px; }
.phone-card code { color: var(--studio-accent-2); font-size: 13px; }

/* ---- Phones and narrow windows: stack the columns -------------------------- */
@media (max-width: 900px) {
    .studio-main { flex-wrap: wrap !important; }
    .col-compose, .col-upload, .col-canvas { flex: 1 1 100% !important; max-width: 100% !important; min-width: 0 !important; }
    .col-canvas { order: -1; }
    .field-head.field-head { flex-wrap: wrap !important; row-gap: 6px !important; }
    .field-head > .field-label-wrap { flex: 1 0 100% !important; }
    .tc-root { height: 62vh !important; min-height: 360px; }
    .gradio-container main.contain { padding: 6px 10px 0 !important; }
    .logo-head > .studio-logo-wrap { position: static !important; transform: none; margin-left: auto !important; }
}
"""


# ---------------------------------------------------------------------------
# The app window: our own title-bar buttons (frameless window), dragging by
# the header rows, the icon following the accent; plus a reconnect check.
# ---------------------------------------------------------------------------
def _win_svg(path: str) -> str:
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{path}</svg>')


_WIN_MIN = _win_svg('<path d="M5 12h14"/>')
_WIN_MAX = _win_svg('<rect x="5" y="5" width="14" height="14" rx="2"/>')
_WIN_CLOSE = _win_svg('<path d="M6 6l12 12M18 6L6 18"/>')
WINDOW_CONTROLS_HTML = (
    '<div class="win-ctl">'
    f'<button data-win="min" title="Minimize">{_WIN_MIN}</button>'
    f'<button data-win="max" title="Maximize / restore">{_WIN_MAX}</button>'
    f'<button data-win="close" title="Close">{_WIN_CLOSE}</button>'
    "</div>"
)

WINDOW_HEAD = r"""<script>(() => {
const api = () => window.pywebview && window.pywebview.api && window.pywebview.api.minimize ? window.pywebview.api : null;
let maximized = true, lastAccent = '', started = false;
const ready = () => {
  if (started || !api()) return;
  started = true;
  document.documentElement.classList.add('in-window');
  // Gradio scopes the stylesheet to its container, so mark the elements themselves.
  const mark = () => {
    document.querySelectorAll('.win-ctl-wrap:not(.shown)').forEach(e => e.classList.add('shown'));
    // Room for the window buttons: the Preset pill keeps just its icons (they have tooltips).
    document.querySelectorAll('.pill-label-wrap:not(.win-hide)').forEach(e => { e.classList.add('win-hide'); e.closest('.pill')?.classList.add('icons-only'); });
  };
  mark(); setInterval(mark, 1000);
  setInterval(() => {
    const a = getComputedStyle(document.documentElement).getPropertyValue('--studio-accent').trim();
    if (a && a !== lastAccent && /^#[0-9a-fA-F]{6}$/.test(a)) { lastAccent = a; api().set_accent(a); }
  }, 1500);
  api().is_maximized().then(m => { maximized = m; });
};
// The bridge (window.pywebview.api) can arrive after this script runs: keep checking for a while.
window.addEventListener('pywebviewready', ready);
let tries = 0;
const poll = setInterval(() => { ready(); if (started || ++tries > 120) clearInterval(poll); }, 500);
document.addEventListener('click', (e) => {
  const b = e.target.closest('.win-ctl [data-win]'); if (!b || !api()) return;
  const w = b.dataset.win;
  if (w === 'min') api().minimize();
  if (w === 'max') api().toggle_maximize().then(m => { maximized = m; });
  if (w === 'close') api().close();
});
// Dragging: while restored, the header rows are a native title bar (CSS app-region),
// so Windows moves the window and snapping happens when the move ends. While maximized
// the rows hand the drag to us instead: we restore under the cursor and follow it.
const isGrip = (t) => t.closest('.col-head') && !t.closest('button, a, input, textarea, label, select, .pill, .win-ctl');
const syncMax = () => document.querySelectorAll('.col-head').forEach(h => h.classList.toggle('is-max', maximized));
setInterval(() => { if (api()) api().is_maximized().then(m => { maximized = m; syncMax(); }); }, 800);
let grab = null;
document.addEventListener('mousedown', (e) => {
  if (e.button !== 0 || !api() || !isGrip(e.target)) return;
  grab = {x: e.clientX, y: e.clientY, sx: e.screenX, sy: e.screenY, restored: false};
});
document.addEventListener('mousemove', (e) => {
  if (!grab) return;
  if (!(e.buttons & 1)) { grab = null; return; }
  if (!grab.restored) {
    if (Math.hypot(e.screenX - grab.sx, e.screenY - grab.sy) < 6) return;
    grab.restored = true;
    const frac = grab.x / window.innerWidth;
    api().restore_at(frac, grab.y).then(() => {
      maximized = false; syncMax();
      if (grab) grab.x = Math.round(window.innerWidth * frac);
    });
    return;
  }
  window.pywebview._jsApiCallback('pywebviewMoveWindow', [e.screenX - grab.x, e.screenY - grab.y], 'move');
});
document.addEventListener('mouseup', () => { if (grab && grab.restored) api().snap(); grab = null; });
document.addEventListener('dblclick', (e) => {
  if (api() && isGrip(e.target)) api().toggle_maximize().then(m => { maximized = m; syncMax(); });
});

// Reconnect: if the studio stops answering (rig asleep, Wi-Fi drop, restart),
// say so and reload as soon as it's back. A phone that slept a long time reloads too.
let down = false, hiddenAt = 0;
const banner = () => {
  let b = document.querySelector('.reconnect');
  if (!b) { b = document.createElement('div'); b.className = 'reconnect'; b.innerHTML = '<span class="dot"></span>Reconnecting to the studio…'; document.body.appendChild(b); }
  return b;
};
const check = async () => {
  try {
    const r = await fetch(location.origin + '/config', {cache: 'no-store'});
    if (!r.ok) throw new Error(r.status);
    if (down) location.reload();
  } catch (err) { down = true; banner().classList.add('show'); }
};
setInterval(() => { if (!document.hidden) check(); }, down ? 2000 : 5000);
setInterval(() => { if (down) check(); }, 2000);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { hiddenAt = Date.now(); return; }
  if (hiddenAt && Date.now() - hiddenAt > 10 * 60 * 1000) { location.reload(); return; }
  check();
});
window.addEventListener('online', check);
})();</script>"""

CSS += """
/* ---- Window buttons (only inside the app window) ---------------------------- */
.win-ctl-wrap { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; padding: 0 !important;
                display: none !important; }
.win-ctl-wrap.win-ctl-wrap.shown { display: block !important; }
.win-ctl { display: flex; gap: 4px; margin-left: 6px; }
.win-ctl button { width: 30px; height: 30px; padding: 0; border-radius: 9px; border: 1px solid var(--studio-border);
                  background: var(--studio-card); color: #C9CDDA; display: grid; place-items: center; cursor: pointer; }
.win-ctl button svg { width: 14px; height: 14px; display: block; }
.win-ctl button:hover { border-color: var(--studio-accent); color: #fff; background: var(--studio-accent-soft); }
.win-ctl button[data-win="close"]:hover { background: #E5484D; border-color: #E5484D; }
.col-head { user-select: none; }

/* ---- Reconnect banner --------------------------------------------------------- */
.reconnect { position: fixed; left: 50%; top: 12px; transform: translate(-50%, -80px); z-index: 9999; display: flex; gap: 10px;
             align-items: center; padding: 9px 16px; border-radius: 999px; background: var(--studio-card);
             border: 1px solid var(--studio-accent); color: #E8EAF2; font-size: 13px; transition: transform .25s;
             box-shadow: 0 10px 30px -12px #000; }
.reconnect.show { transform: translate(-50%, 0); }
.reconnect .dot { width: 8px; height: 8px; border-radius: 999px; background: var(--studio-accent); animation: rcPulse 1s infinite; }
@keyframes rcPulse { 50% { opacity: .25; } }

/* ---- Upload column: header fixed, the panels scroll on their own -------------- */
.side-scroll { max-height: calc(100vh - 66px); overflow-y: auto !important; overflow-x: hidden; padding-right: 2px;
               flex-wrap: nowrap !important; }
.acc-model { --acc-icon: """ + _mask("model") + """; }
.model-card { font-size: 12.5px; line-height: 1.55; color: #B4B9CA; }
.model-card .model-title { font-family: var(--studio-font-title); font-size: 18px; font-weight: 700; color: #F3F4F8; }
.model-card .model-origin { font-size: 12px; margin-bottom: 8px; }
.model-card .model-origin a, .model-card .model-origin code { color: var(--studio-accent-2); text-decoration: none; }
.model-card p { margin: 0 0 8px; }
.model-card .model-license { font-size: 11.5px; color: #8A90A6; }
.model-card .model-thanks { color: var(--studio-accent-2); }
.model-actions { gap: 8px !important; }
.model-actions > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.model-status p { font-size: 12px !important; }

/* ---- Appearance: two even columns of buttons ----------------------------------- */
.ap-buttons { display: grid !important; grid-template-columns: 1fr 1fr; gap: 4px; }
.ap-buttons button { width: 100%; text-align: center; justify-content: center; }

/* ---- Improve: an accent chip like Random ------------------------------------- */
.chip-btn.chip-btn { height: 24px !important; min-height: 24px !important; padding: 0 10px !important; border-radius: 999px !important;
    border: 1px solid var(--studio-accent) !important; background: var(--studio-accent-soft) !important;
    color: var(--studio-accent-2) !important; font-size: 10.5px !important; font-weight: 700 !important;
    letter-spacing: .06em; text-transform: uppercase; gap: 5px !important; box-shadow: none !important; }
.chip-btn.chip-btn:hover { background: var(--studio-accent) !important; color: #fff !important; }
.chip-btn img { display: none !important; }
.chip-btn { --chip-icon: """ + _mask("sparkles") + """; }
.more-btn.chip-btn { --chip-icon: """ + _mask("refresh") + """; }
.chip-btn::before { content: ""; width: 12px; height: 12px; flex: none; background-color: currentColor;
    -webkit-mask: var(--chip-icon) center / contain no-repeat; mask: var(--chip-icon) center / contain no-repeat; }

/* ---- Help: a "?" beside Style and Lyrics with tips on hover / tap ------------- */
.style-group, .lyrics-group, .style-group > .styler, .lyrics-group > .styler, .field-head, .field-label-wrap,
.field-label-wrap .html-container, .field-label-wrap .prose { overflow: visible !important; }
.field-head { position: relative; z-index: 5; }
.help { position: relative; display: inline-grid; place-items: center; width: 16px; height: 16px; margin-left: 7px;
        border-radius: 999px; border: 1.5px solid var(--studio-accent); color: var(--studio-accent-2); font-size: 10.5px;
        font-weight: 800; cursor: help; vertical-align: 1px; letter-spacing: 0; outline: none; }
.help:hover, .help:focus { background: var(--studio-accent); color: #fff; }
.help-tip { position: absolute; left: -12px; top: 24px; width: 330px; z-index: 50; display: none; text-transform: none;
            letter-spacing: 0; font-weight: 400; font-size: 12.5px; line-height: 1.55; color: #D4D7E2; text-align: left;
            background: var(--studio-card); border: 1px solid var(--studio-accent); border-radius: 12px; padding: 12px 14px;
            box-shadow: 0 16px 40px -12px #000; cursor: default; }
.help-tip b { color: var(--studio-accent-2); font-weight: 600; }
.help:hover .help-tip, .help:focus .help-tip, .help:focus-within .help-tip { display: block; }

/* ---- Output folder: icon only ----------------------------------------------------- */
.folder-btn { font-size: 0 !important; gap: 0 !important; }

@media (max-width: 900px) {
    .side-scroll { max-height: none; overflow: visible !important; }
    .help-tip { width: min(330px, 80vw); }
}
"""

CSS += """
/* ---- Canvas cards: popouts, active buttons, the floating volume -------------- */
.tc-bg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.tc-seg { height: 28px; box-sizing: border-box; align-items: stretch; }
.tc-seg button { height: 26px !important; min-width: 30px; justify-content: center; border-radius: 0 !important; }
.tc-seg button.on { background: var(--studio-accent) !important; color: #fff !important; }
.tc-seg button:not(.on):hover { background: var(--studio-accent-soft); color: #fff; }
.tc-btn.active { border-color: var(--studio-accent) !important; box-shadow: 0 0 0 1px var(--studio-accent);
                 color: var(--studio-accent-2); background: var(--studio-accent-soft); }
.tc-volbtn { display: none !important; }
.tc-card:has([data-pop="extend"].open) .tc-volbtn { display: inline-flex !important; }
.tc-card:has([data-pop="extend"].open) .tc-ctrl > .tc-vol-icon,
.tc-card:has([data-pop="extend"].open) .tc-ctrl > .tc-vol { display: none !important; }
.tc-ctrl { position: relative; }
.tc-volpop { position: absolute; left: calc(100% + 22px); top: 50%; transform: translateY(-50%); display: none;
             align-items: center; gap: 8px; padding: 8px 12px; border-radius: 10px; background: var(--studio-card);
             border: 1px solid var(--studio-accent); box-shadow: 0 10px 26px -12px #000; z-index: 4; }
.tc-volpop.open { display: flex; animation: tcSlide .14s ease-out; }
.tc-volpop .tc-vol { width: 120px; }
"""

CSS += """
/* ---- Upload column: one scrolling list of panels (they never squeeze) --------- */
.side-scroll > *, .side-scroll .studio-acc, .side-scroll .appearance { flex-shrink: 0 !important; }
.side-scroll .studio-acc, .side-scroll .appearance, .side-scroll .studio-acc > div, .side-scroll .appearance > div {
    max-height: none !important; overflow: visible !important; }
.side-scroll { scrollbar-gutter: stable; }

/* ---- Number steppers inside the Extend popout ------------------------------------ */
.tc-pop input[type="number"]::-webkit-inner-spin-button, .tc-pop input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none; margin: 0; }
.tc-pop input[type="number"] { -moz-appearance: textfield; }
.tc-pop .num-wrap { display: inline-block; width: auto; }
.tc-pop .tc-num .num-wrap input { width: 46px !important; padding: 0 18px 0 4px !important; text-align: center; }
.tc-pop .stepper { right: 2px; gap: 1px; }
.tc-pop .stepper button { width: 14px; height: 11px; border-radius: 3px; }
.tc-pop .stepper button svg { width: 8px; height: 8px; }

/* ---- Help tips: breathing room, a soft glow, a solid card -------------------------- */
.help { margin-left: 10px !important; }
.style-group { position: relative; z-index: 4; overflow: visible !important; }
.lyrics-group { position: relative; z-index: 3; overflow: visible !important; }
.field-head { z-index: 30 !important; }
.help-tip { padding: 16px 18px !important; background: var(--studio-card) !important; opacity: 1 !important;
            backdrop-filter: none !important; isolation: isolate;
            box-shadow: 0 0 0 1px var(--studio-accent), 0 0 28px -4px color-mix(in srgb, var(--studio-accent) 55%, transparent),
                        0 18px 44px -12px #000 !important; }
.help-tip::before { content: ""; position: absolute; inset: 0; border-radius: inherit; background: var(--studio-card); z-index: -1; }

/* ---- Generate: a light sweep across it now and then; constant while it runs ------- */
.generate-btn { position: relative; overflow: hidden !important; isolation: isolate; }
.generate-btn::after { content: ""; position: absolute; top: 0; bottom: 0; left: -60%; width: 45%; pointer-events: none;
    background: linear-gradient(100deg, transparent 0%, rgba(255,255,255,.10) 35%, rgba(255,255,255,.32) 50%,
                rgba(255,255,255,.10) 65%, transparent 100%);
    transform: skewX(-18deg); opacity: 0; }
/* Idle: one slow sweep while hovered. */
.generate-btn:hover:not(:disabled)::after { opacity: 1; animation: genSweep 2.2s ease-in-out infinite; }
/* Busy: thin bright glints racing across, over and over. */
.generate-btn:disabled::after { opacity: 1; width: 160%; left: -30%; transform: none;
    background: repeating-linear-gradient(100deg, transparent 0 70px, rgba(255,255,255,.08) 70px 82px,
                rgba(255,255,255,.42) 86px 89px, rgba(255,255,255,.08) 93px 105px, transparent 105px 175px);
    animation: genRace .7s linear infinite; }
@keyframes genRace { from { transform: translateX(-175px); } to { transform: translateX(0); } }
@keyframes genSweep { 0% { left: -60%; } 45%, 100% { left: 130%; } }
@media (prefers-reduced-motion: reduce) { .generate-btn::after { animation: none; display: none; } }
"""

CSS += """
.field-head > .field-label-wrap .help { margin-left: 10px !important; }
.field-head > .field-label-wrap .help-tip { padding: 16px 18px !important; line-height: 1.6 !important; }
"""


# ---------------------------------------------------------------------------
# Tooltips in the studio's style for anything with a title (replaces the
# browser's plain ones).
# ---------------------------------------------------------------------------
TIPS_HEAD = r"""<script>(() => {
let tip = null, timer = null, owner = null;
const hide = () => { clearTimeout(timer); timer = null; owner = null; if (tip) tip.classList.remove('show'); };
const show = (el) => {
  if (!tip) { tip = document.createElement('div'); tip.className = 'studio-tip'; document.body.appendChild(tip); }
  tip.textContent = el.dataset.tip;
  tip.classList.add('show');
  const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
  let x = r.left + r.width / 2 - t.width / 2, y = r.bottom + 8;
  if (y + t.height > innerHeight - 6) y = r.top - t.height - 8;
  tip.style.left = Math.max(6, Math.min(x, innerWidth - t.width - 6)) + 'px';
  tip.style.top = Math.max(6, y) + 'px';
};
document.addEventListener('mouseover', (e) => {
  const el = e.target.closest && e.target.closest('[title], [data-tip]');
  if (!el || el.closest('.help')) return;
  if (el.hasAttribute('title')) { const t = el.getAttribute('title'); el.removeAttribute('title'); if (t) el.dataset.tip = t; }
  if (!el.dataset.tip || el === owner) return;
  hide(); owner = el;
  timer = setTimeout(() => { if (owner === el && el.isConnected) show(el); }, 380);
}, true);
document.addEventListener('mouseout', (e) => { if (owner && !owner.contains(e.relatedTarget)) hide(); }, true);
['mousedown', 'wheel', 'keydown'].forEach(t => document.addEventListener(t, hide, true));
window.addEventListener('blur', hide);
})();</script>"""

CSS += """
.pill-label-wrap.win-hide { display: none !important; }
.studio-tip { position: fixed; z-index: 10000; pointer-events: none; opacity: 0; transform: translateY(-3px);
              transition: opacity .12s, transform .12s; max-width: 280px; padding: 6px 11px; border-radius: 9px;
              background: var(--studio-card); color: #E8EAF2; font-size: 12px; line-height: 1.4; white-space: normal;
              border: 1px solid var(--studio-accent);
              box-shadow: 0 0 18px -6px color-mix(in srgb, var(--studio-accent) 70%, transparent), 0 10px 26px -12px #000;
              font-family: var(--studio-font-body); }
.studio-tip.show { opacity: 1; transform: none; }
"""

CSS += """
/* Help tips sit above everything after them in the Compose column. */
.style-group.style-group { z-index: 60 !important; }
.lyrics-group.lyrics-group { z-index: 50 !important; }

/* Track volume: the same chunky bar as Quality. */
.tc-vol { -webkit-appearance: none; appearance: none; height: 12px; border-radius: 5px; cursor: pointer; margin: 0;
          border: 1px solid var(--studio-border);
          background: linear-gradient(to right, var(--studio-accent) var(--p, 80%), var(--studio-raised) var(--p, 80%)); }
.tc-vol::-webkit-slider-runnable-track { height: 12px; background: transparent; border: none; }
.tc-vol::-webkit-slider-thumb { -webkit-appearance: none; width: 6px; height: 16px; margin-top: -3px; border-radius: 3px;
    background: #fff; box-shadow: 0 0 0 2px var(--studio-accent), 0 2px 5px #0008; cursor: grab; }
.tc-vol::-moz-range-thumb { width: 6px; height: 16px; border-radius: 3px; background: #fff; border: none;
    box-shadow: 0 0 0 2px var(--studio-accent); }

/* Rendering cards: a little accent neuron field fills as the estimate runs down. */
.tc-pending .tc-shimmer { display: none !important; }
.tc-mini { display: block; width: 100%; height: 54px; margin: 8px 0 6px; border-radius: 8px; background: var(--studio-page);
           border: 1px solid var(--studio-border); }
.tc-stem .tc-mini, .tc-card.tc-pending.small .tc-mini { height: 40px; }
.tc-pfoot { display: flex; align-items: center; gap: 10px; font-variant-numeric: tabular-nums; }
.tc-pbar { flex: 1 1 auto; height: 6px; border-radius: 999px; background: var(--studio-raised); overflow: hidden;
           border: 1px solid var(--studio-border); }
.tc-pbar i { display: block; height: 100%; width: 0; border-radius: inherit; transition: width .9s linear;
             background: linear-gradient(90deg, var(--studio-accent-hover), var(--studio-accent), var(--studio-accent-2));
             box-shadow: 0 0 10px var(--studio-accent); }
.tc-pbar.over i { animation: pbarPulse 1.2s ease-in-out infinite; }
@keyframes pbarPulse { 50% { opacity: .45; } }
"""

CSS += """
/* ---- Compose: the fields scroll in their own box; Generate stays put below ----- */
.compose-scroll { max-height: calc(100vh - 66px - 52px - 22px); overflow-y: auto !important; overflow-x: hidden;
                  flex-wrap: nowrap !important; scrollbar-gutter: stable; padding-right: 2px; }
.compose-scroll > * { flex-shrink: 0 !important; }
@media (max-width: 900px) { .compose-scroll { max-height: none; overflow: visible !important; } }
"""

CSS += """
.tag-chips label > span { margin-left: 0 !important; }
"""

CSS += """
/* In the app window the header rows are the title bar (WebView2 non-client regions):
   drag to move or snap, double-click to maximize. Their controls stay clickable. */
.col-head:not(.is-max) { app-region: drag; -webkit-app-region: drag; }
.col-head button, .col-head a, .col-head input, .col-head label, .col-head .pill, .col-head .win-ctl {
    app-region: no-drag; -webkit-app-region: no-drag; }
"""

CSS += """
/* ---- Info panel ------------------------------------------------------------------ */
.acc-info { --acc-icon: """ + _mask("sparkles") + """; }
.info-card { font-size: 12.5px; line-height: 1.55; color: #B4B9CA; }
.info-rows { display: grid; gap: 4px; margin-bottom: 14px; }
.info-rows > div { display: flex; justify-content: space-between; gap: 10px; }
.info-rows span { color: #8A90A6; flex: none; }
.info-rows code { color: #D4D7E2; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.info-head { font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
             color: var(--studio-accent-2); margin: 4px 0 8px; }
.credit { padding: 10px 12px; border-radius: 10px; background: var(--studio-page); border: 1px solid var(--studio-border);
          margin-bottom: 8px; }
.credit-who { color: #E8EAF2; }
.credit-who b { color: var(--studio-accent-2); }
.credit-what { font-size: 12px; color: #9DA3B8; }
.credit.claude { border-color: var(--studio-accent);
                 box-shadow: 0 0 22px -10px color-mix(in srgb, var(--studio-accent) 70%, transparent); }
.credit-note { margin-top: 6px; font-size: 12px; color: var(--studio-accent-2); font-style: italic; }
.credit-by { font-style: normal; opacity: .75; white-space: nowrap; }
.credit-links { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 8px; font-size: 12px; }
.credit-links span { color: #9DA3B8; margin-right: 2px; }
.credit-links a { padding: 3px 10px; border-radius: 999px; text-decoration: none; color: var(--studio-accent-2);
                  border: 1px solid color-mix(in srgb, var(--studio-accent) 55%, transparent); }
.credit-links a:hover { background: color-mix(in srgb, var(--studio-accent) 18%, transparent); }
"""


# ---------------------------------------------------------------------------
# Layout helpers: help tips float over everything, tag chips fill exactly two
# rows, Style/Lyrics grow to fill the Compose column, and busy run buttons
# light up with the neuron field.
# ---------------------------------------------------------------------------
UX_HEAD = r"""<script>(() => {
// Help: show the tip in a floating layer (no panel can cover or clip it).
let pop = null;
const showHelp = (h) => {
  const src = h.querySelector('.help-tip'); if (!src) return;
  if (!pop) { pop = document.createElement('div'); pop.className = 'studio-help-pop'; document.body.appendChild(pop); }
  pop.innerHTML = src.innerHTML; pop.classList.add('show');
  const r = h.getBoundingClientRect(), t = pop.getBoundingClientRect();
  let y = r.bottom + 10; if (y + t.height > innerHeight - 8) y = Math.max(8, r.top - t.height - 10);
  pop.style.left = Math.max(8, Math.min(r.left - 14, innerWidth - t.width - 8)) + 'px';
  pop.style.top = y + 'px';
};
const hideHelp = () => { if (pop) pop.classList.remove('show'); };
document.addEventListener('mouseover', (e) => { const h = e.target.closest && e.target.closest('.help'); if (h) showHelp(h); }, true);
document.addEventListener('mouseout', (e) => { const h = e.target.closest && e.target.closest('.help'); if (h && !h.contains(e.relatedTarget)) hideHelp(); }, true);
document.addEventListener('focusin', (e) => { if (e.target.closest && e.target.closest('.help')) showHelp(e.target.closest('.help')); });
document.addEventListener('focusout', (e) => { if (e.target.closest && e.target.closest('.help')) hideHelp(); });

// Suggested tags: as many as fit in two rows; the rest wait for More.
const fitTags = () => new Set([...document.querySelectorAll('.chips-row .tag-chips label')].map(l => l.parentElement)).forEach(wrap => {
  const chips = [...wrap.children].filter(c => c.tagName === 'LABEL');
  chips.forEach(c => { c.style.display = ''; c.style.order = ''; });
  const gap = parseFloat(getComputedStyle(wrap).columnGap) || 6, room = wrap.clientWidth;
  // Pack in order of relevance: each chip takes the first of two rows it fits in (CSS order
  // only, the elements stay where Gradio put them); chips that fit nowhere wait for More.
  const used = [0, 0];
  chips.forEach((c, i) => {
    const w = c.offsetWidth;
    const row = used.findIndex(u => (u ? u + gap : 0) + w <= room);
    if (row < 0) { c.style.display = 'none'; return; }
    used[row] += (used[row] ? gap : 0) + w;
    c.style.order = String(row * 100 + i);
  });
});

// Style and Lyrics share any free height in the Compose column (until resized by hand).
const fillCompose = () => {
  const box = document.querySelector('.compose-scroll'); if (!box) return;
  const areas = [...box.querySelectorAll('.style-group textarea, .lyrics-group textarea')].filter(t => !t.dataset.manual);
  if (!areas.length) return;
  // The box is as tall as its content up to its max-height: the free space is what's left of that.
  const slack = parseFloat(getComputedStyle(box).maxHeight) - box.scrollHeight;
  if (Math.abs(slack) < 4) return;
  const each = slack / areas.length;
  areas.forEach(t => { const h = Math.max(76, t.offsetHeight + each); t.style.height = h + 'px'; t.dataset.auto = String(Math.round(h)); });
};
document.addEventListener('mouseup', (e) => {
  const t = e.target.closest && e.target.closest('.style-group textarea, .lyrics-group textarea');
  if (t && t.dataset.auto && Math.abs(t.offsetHeight - parseFloat(t.dataset.auto)) > 3) t.dataset.manual = '1';
});
let pending = false;
const tidy = () => { if (pending) return; pending = true; setTimeout(() => { pending = false; fitTags(); fillCompose(); }, 60); };
new MutationObserver(tidy).observe(document.documentElement, {childList: true, subtree: true});
window.addEventListener('resize', tidy);
setTimeout(tidy, 800); setTimeout(tidy, 2500);

// Busy run buttons: a white neuron field drifts under their label. One loop
// drives every button with time-based motion, so all of them move alike.
const fields = new WeakMap();
const fieldFor = (btn) => {
  let f = fields.get(btn);
  if (!f) { f = {cv: null, nodes: [], w: 0, h: 0}; fields.set(btn, f); }
  if (!f.cv || f.cv.parentElement !== btn) {
    btn.querySelectorAll(':scope > .btn-field').forEach(c => c.remove());
    f.cv = document.createElement('canvas'); f.cv.className = 'btn-field'; btn.prepend(f.cv); f.w = 0;
  }
  return f;
};
const SPEED = .0098;  // px per ms
let lastField = 0;
const drawFields = (now) => {
  requestAnimationFrame(drawFields);
  if (now - lastField < 33) return;
  const dt = Math.min(100, now - (lastField || now)); lastField = now;
  document.querySelectorAll('.generate-btn, .redux-btn').forEach(btn => {
    if (!btn.disabled) { btn.querySelectorAll(':scope > .btn-field').forEach(c => c.remove()); const f = fields.get(btn); if (f) f.cv = null; return; }
    const f = fieldFor(btn), cv = f.cv, ctx = cv.getContext('2d');
    const w = cv.clientWidth, h = cv.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
    if (!w || !h) return;
    if (Math.abs(f.w - w) > 1 || Math.abs(f.h - h) > 1) {
      f.w = w; f.h = h; cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      if (!f.nodes.length) f.nodes = Array.from({length: Math.round(w * h / 550)}, () => ({x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - .5) * 2 * SPEED, vy: (Math.random() - .5) * 1.4 * SPEED, glow: 0}));
    }
    const nodes = f.nodes;
    ctx.setTransform(cv.width / w, 0, 0, cv.height / h, 0, 0); ctx.clearRect(0, 0, w, h);
    for (const n of nodes) { n.x += n.vx * dt; n.y += n.vy * dt; n.glow *= .92;
      if (n.x < 0 || n.x > w) n.vx = -n.vx; if (n.y < 0 || n.y > h) n.vy = -n.vy;
      n.x = Math.max(0, Math.min(w, n.x)); n.y = Math.max(0, Math.min(h, n.y)); }
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j], d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < 42) { ctx.strokeStyle = `rgba(255,255,255,${(1 - d / 42) * .28 + (a.glow + b.glow) * .25})`;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
    }
    for (const n of nodes) { ctx.fillStyle = `rgba(255,255,255,${.45 + n.glow * .5})`;
      ctx.beginPath(); ctx.arc(n.x, n.y, 1.2 + n.glow * 1.5, 0, Math.PI * 2); ctx.fill(); }
    if (Math.random() < .3 && nodes.length) nodes[Math.floor(Math.random() * nodes.length)].glow = 1;
  });
};
requestAnimationFrame(drawFields);
// Background panes pause animation frames; keep the canvases attached anyway.
setInterval(() => document.querySelectorAll('.generate-btn:disabled, .redux-btn:disabled').forEach(fieldFor), 400);
})();</script>"""

CSS += """
.help-tip { display: none !important; }
.studio-help-pop { position: fixed; z-index: 10001; width: 340px; max-width: calc(100vw - 16px); display: none;
    padding: 16px 18px; border-radius: 12px; background: var(--studio-card); border: 1px solid var(--studio-accent);
    color: #D4D7E2; font-size: 12.5px; line-height: 1.6; font-family: var(--studio-font-body);
    box-shadow: 0 0 28px -4px color-mix(in srgb, var(--studio-accent) 55%, transparent), 0 18px 44px -12px #000; }
.studio-help-pop.show { display: block; animation: tipIn .12s ease-out; }
.studio-help-pop b { color: var(--studio-accent-2); font-weight: 600; }
@keyframes tipIn { from { opacity: 0; transform: translateY(-3px); } to { opacity: 1; transform: none; } }
.chips-row .tag-chips, .chips-row .tag-chips .wrap { overflow: hidden !important; max-height: none; }
.pill.icons-only.icons-only { padding: 2px 3px !important; gap: 2px !important; }
.btn-field { position: absolute; inset: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none; opacity: .5; }
.redux-btn { position: relative; overflow: hidden !important; isolation: isolate; }
.generate-btn:disabled, .redux-btn:disabled { filter: saturate(.85); }
"""

CSS += """
/* Number fields: no grey frame (the text fields have none either). */
.settings-row input[type="number"] { border: none !important; }

/* Busy run buttons: broad, soft light bands drifting across; REDUX matches Generate. */
.redux-btn:disabled { background: linear-gradient(135deg, var(--studio-accent), var(--studio-accent-hover)) !important;
    color: #fff !important; opacity: .75; border: none !important; }
.redux-btn::after { content: ""; position: absolute; top: 0; bottom: 0; pointer-events: none; opacity: 0; }
.generate-btn:disabled::after, .redux-btn:disabled::after { opacity: 1; left: -420px; right: 0; width: auto; transform: none;
    background: repeating-linear-gradient(100deg, transparent 0, rgba(255,255,255,0) 90px, rgba(255,255,255,.10) 160px,
                rgba(255,255,255,.22) 210px, rgba(255,255,255,.10) 260px, rgba(255,255,255,0) 330px, transparent 420px);
    animation: genRace 2.4s linear infinite; }
@keyframes genRace { from { transform: translateX(0); } to { transform: translateX(420px); } }
@media (prefers-reduced-motion: reduce) { .redux-btn::after { animation: none; display: none; } }

/* Extend popout: "× 2 takes". */
.tc-pop .tc-num .tc-unit { margin-left: 4px; color: #8A90A6; }

/* Stems: mute and their own level. */
.tc-stem .tc-ctrl > .tc-svol { display: inline-block; width: 64px; flex: none; }
.tc-mute { flex: none; }
.tc-mute.muted { color: var(--studio-accent-2) !important; border-color: var(--studio-accent) !important; }
.tc-stem:has(.tc-mute.muted) .tc-svol { opacity: .4; }

/* Restart (phone access changed) and its question. */
.restart-btn { margin-top: 10px; background: linear-gradient(135deg, #E5484D, #B4232A) !important; color: #fff !important;
               border: none !important; box-shadow: 0 8px 24px -12px #E5484D !important; }
.restart-dialog { position: fixed !important; inset: 0; z-index: 10050; display: grid; place-content: center;
                  background: #05060ACC; backdrop-filter: blur(3px); gap: 0 !important; padding: 16px; }
.restart-dialog > .wrap { display: none !important; }
.restart-dialog > :not(.wrap) { width: min(440px, calc(100vw - 32px)); background: var(--studio-card);
                      border: 1px solid #E5484D; box-sizing: border-box; }
.restart-dialog > .block { border-radius: 14px 14px 0 0; border-bottom: none; padding: 22px 22px 6px; }
.restart-dialog > .rd-actions { border-radius: 0 0 14px 14px; border-top: none; padding: 12px 22px 20px; gap: 8px; }
.restart-dialog .rd-title { font-size: 17px; font-weight: 700; color: #fff; margin-bottom: 8px; }
.restart-dialog .rd-text, .restart-dialog .rd-busy { font-size: 13px; line-height: 1.55; color: #A3A9BC; }
.restart-dialog .rd-busy { display: none; color: #fff; }
.restart-dialog.restarting .rd-text { display: none; }
.restart-dialog.restarting .rd-busy { display: block; }
.restart-dialog.restarting .rd-actions { visibility: hidden; }
.restart-dialog .rd-actions button { flex: 1 1 auto; height: 38px !important; }
.restart-dialog .rd-save { background: linear-gradient(135deg, #E5484D, #B4232A) !important; color: #fff !important; border: none !important; }
"""

CSS += """
.restart-dialog .rd-actions { flex-wrap: nowrap !important; }
.restart-dialog .rd-actions button { min-width: 0 !important; flex: 1 1 0 !important; }
.restart-dialog .rd-actions .rd-save { flex-grow: 1.5 !important; }
"""
CSS += """
.restart-dialog.hide { display: none !important; }
"""

CSS += """
.ap-buttons > button { margin: 0 !important; }
/* The Preset pill sits midway between the Upload badge and the window buttons. */
.col-head > .pills { margin-left: auto !important; }
.col-head:has(.win-ctl-wrap.shown) > .pills { margin-right: auto !important; }
/* The console's first load can take a moment. */
.console-log:empty::before { content: "Loading the console…"; color: #6B7186; font-style: italic; }
"""

# Upload-column panels: drag the grip left of a panel's icon to reorder them (kept per browser).
PANELS_HEAD = r"""<script>(() => {
const KEYS = ['acc-source', 'acc-interpret', 'acc-phone', 'appearance', 'acc-console', 'acc-model', 'acc-info'];
const keyOf = (el) => KEYS.find(k => el.classList.contains(k));
const load = () => { try { return JSON.parse(localStorage.getItem('studio-panel-order') || '[]'); } catch (e) { return []; } };
const save = (order) => { try { localStorage.setItem('studio-panel-order', JSON.stringify(order)); } catch (e) {} };
const panels = () => [...document.querySelectorAll('.side-scroll > *')].filter(keyOf);
// A panel added since the order was saved goes back next to the one it follows by default.
const current = () => {
  const order = load().filter(k => KEYS.includes(k));
  KEYS.forEach((k, i) => {
    if (order.includes(k)) return;
    const prev = KEYS.slice(0, i).reverse().find(p => order.includes(p));
    order.splice(prev ? order.indexOf(prev) + 1 : 0, 0, k);
  });
  return order;
};
const applyOrder = (order) => panels().forEach(p => { p.style.order = String(order.indexOf(keyOf(p))); });
const GRIP = '<svg viewBox="0 0 10 16" width="8" height="14" fill="currentColor"><circle cx="2" cy="3" r="1.4"/><circle cx="8" cy="3" r="1.4"/><circle cx="2" cy="8" r="1.4"/><circle cx="8" cy="8" r="1.4"/><circle cx="2" cy="13" r="1.4"/><circle cx="8" cy="13" r="1.4"/></svg>';
let drag = null;
const wire = () => {
  const list = panels();
  if (!list.length) return;
  applyOrder(current());
  list.forEach(panel => {
    const head = panel.querySelector(':scope > .label-wrap, :scope > button');
    if (!head || head.querySelector('.acc-grip')) return;
    head.classList.add('has-grip');
    const grip = document.createElement('i');
    grip.className = 'acc-grip'; grip.innerHTML = GRIP; grip.dataset.tip = 'Drag to reorder';
    head.appendChild(grip);
    // The grip drags; it never opens or closes the panel.
    grip.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); }, true);
    grip.addEventListener('pointerdown', (e) => {
      e.preventDefault(); e.stopPropagation();
      try { grip.setPointerCapture(e.pointerId); } catch (err) {}
      drag = {key: keyOf(panel), panel};
      panel.classList.add('panel-dragging'); document.body.classList.add('panels-sorting');
    });
    grip.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const order = current().filter(k => k !== drag.key);
      // Drop before the first panel whose middle is below the pointer.
      const others = panels().filter(p => p !== drag.panel).sort((a, b) => +a.style.order - +b.style.order);
      let at = others.findIndex(p => { const r = p.getBoundingClientRect(); return e.clientY < r.top + r.height / 2; });
      if (at < 0) at = others.length;
      const before = others[at] ? keyOf(others[at]) : null;
      order.splice(before ? order.indexOf(before) : order.length, 0, drag.key);
      if (order.join() !== current().join()) { save(order); applyOrder(order); }
    });
    const end = () => {
      if (!drag) return;
      drag.panel.classList.remove('panel-dragging'); document.body.classList.remove('panels-sorting'); drag = null;
    };
    grip.addEventListener('pointerup', end); grip.addEventListener('pointercancel', end);
  });
};
let queued = false;
new MutationObserver(() => { if (queued) return; queued = true; setTimeout(() => { queued = false; wire(); }, 120); })
  .observe(document.documentElement, {childList: true, subtree: true});
setTimeout(wire, 800);
})();</script>"""

CSS += """
.side-scroll { display: flex !important; flex-direction: column !important; }
.has-grip { position: relative !important; padding-left: 20px !important; }
.acc-grip { position: absolute; left: 0; top: 50%; transform: translateY(-50%); width: 16px; height: 22px; display: grid;
            place-items: center; color: #5C6278; cursor: grab; border-radius: 5px; touch-action: none; }
.acc-grip:hover { color: var(--studio-accent-2); background: var(--studio-accent-soft); }
.panels-sorting, .panels-sorting * { cursor: grabbing !important; user-select: none !important; }
.panel-dragging { outline: 1px solid var(--studio-accent); box-shadow: 0 0 22px -6px var(--studio-accent) !important;
                  position: relative; z-index: 5; }
"""

CSS += """
/* ---- Interpret panel ---------------------------------------------------------------- */
.interp-card { font-size: 12.5px; color: #B4B9CA; display: grid; gap: 6px; }
.interp-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 6px; }
.interp-row > span:first-child { min-width: 78px; color: #9DA3B8; font-size: 11px; text-transform: uppercase;
                                 letter-spacing: .08em; }
.interp-tag { padding: 1px 8px; border-radius: 999px; color: #E8EAF2;
              background: color-mix(in srgb, var(--studio-accent) 16%, transparent);
              border: 1px solid color-mix(in srgb, var(--studio-accent) 45%, transparent); }
.interp-note { font-size: 11.5px; color: #9DA3B8; margin-top: 2px; }
.interp-actions { gap: 8px; }
"""
