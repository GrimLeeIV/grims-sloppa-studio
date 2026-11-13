"""Track canvas: the session's tracks as a pannable tree of playable cards.

Each Generate or Cover starts a new tree; extensions and stems branch off the
track they came from, joined by a line, so a composition's history stays
visible and any branch can be revisited. Built on gr.HTML: the cards, the
waveform players and pan/zoom are plain JS (``CANVAS_JS``), and card actions
come back to Python as events:

    click  {action: "open_extend" | "folder", id}            quick, unqueued
    submit {action: "extend", id, direction, seconds, takes}  renders
    submit {action: "stems", id, mode}

The tree is a flat list of nodes (``parent`` links them), sent as the
component's value. Each node carries its audio URL and ~140 waveform peaks,
so the browser draws waveforms without decoding audio.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

FILE_ROUTE = "/gradio_api/file="
PEAK_BARS = 140
_PEAK_CACHE: Dict[tuple, List[float]] = {}


def peaks(path: str, bars: int = PEAK_BARS) -> List[float]:
    """Normalized per-bar peak levels of an audio file (cached by path + mtime)."""
    try:
        stamp = (path, Path(path).stat().st_mtime_ns)
    except OSError:
        return []
    if stamp in _PEAK_CACHE:
        return _PEAK_CACHE[stamp]
    try:
        import numpy as np
        import soundfile as sf

        data, _rate = sf.read(path, always_2d=True, dtype="float32")
        mono = np.abs(data).mean(axis=1)
        chunks = np.array_split(mono, bars)
        levels = np.array([chunk.max() if len(chunk) else 0.0 for chunk in chunks])
        levels = levels / max(float(levels.max()), 1e-6)
        result = [round(float(v), 3) for v in levels]
    except Exception:
        result = []
    _PEAK_CACHE[stamp] = result
    return result


def duration(path: str) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def track_node(path: str, parent: Optional[str] = None, *, sub: str = "", kind: str = "track",
               label: Optional[str] = None, extendable: bool = False,
               marks: Optional[List[float]] = None) -> Dict[str, Any]:
    """A playable card. ``marks`` are seconds where an extension's new music begins."""
    return {
        "id": uuid.uuid4().hex[:10],
        "kind": kind,
        "parent": parent,
        "file": path,
        "url": FILE_ROUTE + quote(str(Path(path).resolve()), safe=":/\\"),
        "label": label or Path(path).stem,
        "sub": sub,
        "dur": round(duration(path), 2),
        "peaks": peaks(path, 60 if kind == "stem" else PEAK_BARS),
        "extendable": extendable,
        "marks": [round(float(m), 2) for m in (marks or [])],
        "created": _made_at(path),
    }


def _made_at(path: str) -> int:
    """When the file was written (ms since epoch), shown under the track's name."""
    try:
        return int(Path(path).stat().st_mtime * 1000)
    except OSError:
        return 0


def pending_node(parent: Optional[str], label: str, sub: str = "Rendering", eta: Optional[int] = None) -> Dict[str, Any]:
    """A rendering card; ``eta`` is the estimated seconds until it's done, shown beside its timer."""
    return {"id": uuid.uuid4().hex[:10], "kind": "pending", "parent": parent, "label": label, "sub": sub,
            "started": int(time.time() * 1000), "eta": eta}


def view(nodes: List[Dict[str, Any]], focus: Optional[str] = None, fit: bool = False,
         notice: Optional[str] = None) -> Dict[str, Any]:
    """The canvas value: nodes plus where to look (``focus`` id, or fit everything).

    ``notice`` floats a dismissable message over the canvas; "" clears it and
    None leaves whatever is showing.
    """
    return {"nodes": nodes, "focus": focus, "fit": fit, "notice": notice, "rev": time.time()}


def find(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    return next((node for node in nodes if node.get("id") == node_id), None)


ICON_PATHS = {
    "extend": '<path d="M4 12h12M12 7l5 5-5 5M20 5v14"/>',
    "before": '<path d="M20 12H8M12 7l-5 5 5 5M4 5v14"/>',
    "close": '<path d="M6 6l12 12M18 6L6 18"/>',
    "stems": '<path d="M12 2l9 5-9 5-9-5z"/><path d="M3 12l9 5 9-5M3 17l9 5 9-5"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "fit": '<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "play": '<path d="M6 4l13 8-13 8z" fill="currentColor"/>',
    "pause": '<path d="M7 4h4v16H7zM13 4h4v16h-4z" fill="currentColor" stroke="none"/>',
    "volume": '<path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16 9a4 4 0 0 1 0 6"/>',
    "muted": '<path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16.5 9.5l5 5M21.5 9.5l-5 5"/>',
    "close": '<path d="M6 6l12 12M18 6L6 18"/>',
}


def _svg(name: str) -> str:
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICON_PATHS[name]}</svg>')


CANVAS_HTML = (
    '<div class="tc-root"><canvas class="tc-bg"></canvas><div class="tc-stage"><div class="tc-world">'
    '<svg class="tc-links"></svg><div class="tc-nodes"></div></div></div>'
    f'<div class="tc-toast"><span></span><button title="Dismiss">{_svg("close")}</button></div>'
    '<div class="tc-empty"><b>No tracks yet.</b><br>Generate, or make a cover. Extend and Stems on a '
    'track branch off it here.</div>'
    '<div class="tc-tools">'
    f'<button class="tc-tool" data-tool="fit" title="Fit everything">{_svg("fit")}</button>'
    f'<button class="tc-tool" data-tool="in" title="Zoom in">{_svg("plus")}</button>'
    f'<button class="tc-tool" data-tool="out" title="Zoom out">{_svg("minus")}</button>'
    "</div></div>"
)

# The card/player behaviour. `element`, `props`, `trigger` and `watch` come from gr.HTML.
CANVAS_JS = r"""
const ICON = __ICONS__;
const stage = element.querySelector('.tc-stage');
const world = element.querySelector('.tc-world');
const nodesEl = element.querySelector('.tc-nodes');
const links = element.querySelector('.tc-links');
const empty = element.querySelector('.tc-empty');
const root = element.querySelector('.tc-root');
const toast = element.querySelector('.tc-toast');
const view = element.__view || (element.__view = {x: 40, y: 30, k: 1});
const cards = element.__cards || (element.__cards = new Map());
const W = 500, H = 158, SW = 300, SH = 92, GX = 80, GY = 22;
let volume = 0.8;
try { volume = parseFloat(localStorage.getItem('studio-volume') || '0.8'); } catch (e) {}

const fmt = (s) => { s = Math.max(0, Math.floor(s || 0)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
const apply = () => { world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.k})`; };
// Cards are measured once drawn, so tall cards (fonts, stems) never overlap.
const size = (n) => {
  const c = cards.get(n.id), h = c && c.card.offsetHeight;
  return [n.kind === 'stem' ? SW : W, h || (n.kind === 'stem' ? SH : H)];
};

function layout(nodes) {
  const byId = new Map(nodes.map(n => [n.id, n]));
  const kids = new Map();
  const roots = [];
  for (const n of nodes) {
    if (n.parent && byId.has(n.parent)) { if (!kids.has(n.parent)) kids.set(n.parent, []); kids.get(n.parent).push(n); }
    else roots.push(n);
  }
  const pos = new Map();
  let cursor = 0;
  const place = (n, depth, x) => {
    const [w, h] = size(n);
    const children = kids.get(n.id) || [];
    if (!children.length) { pos.set(n.id, {x, y: cursor, w, h}); cursor += h + GY; return; }
    const start = cursor;
    for (const c of children) place(c, depth + 1, x + w + GX);
    const first = pos.get(children[0].id), last = pos.get(children[children.length - 1].id);
    const mid = (first.y + last.y + last.h) / 2;
    const y = Math.max(start, mid - h / 2);
    pos.set(n.id, {x, y, w, h});
    cursor = Math.max(cursor, y + h + GY);
  };
  for (const r of roots) { place(r, 0, 0); cursor += GY; }
  return pos;
}

function bars(peaks, cls) {
  const n = Math.max(peaks.length, 1), w = 100 / n;
  let out = `<svg class="${cls}" viewBox="0 0 100 40" preserveAspectRatio="none">`;
  peaks.forEach((p, i) => { const h = Math.max(1.5, p * 38); out += `<rect x="${i * w + w * 0.18}" y="${(40 - h) / 2}" width="${w * 0.64}" height="${h}" rx="0.4"/>`; });
  return out + '</svg>';
}

function marks(n) {
  const d = n.dur || 0;
  return (n.marks || []).filter(m => d && m > 0.5 && m < d - 0.5)
    .map(m => `<div class="tc-mark" style="left:${(m / d * 100).toFixed(2)}%" title="New music starts at ${fmt(m)}"></div>`).join('');
}

function fitHeight() {
  const top = root.getBoundingClientRect().top + window.scrollY;
  root.style.height = Math.max(420, window.innerHeight - top - 14) + 'px';
}

function showNotice(text) {
  if (text === null || text === undefined) return;
  toast.querySelector('span').textContent = text;
  toast.classList.toggle('show', !!text);
}

function made(ms) {
  if (!ms) return '';
  const d = new Date(ms), now = new Date();
  const time = d.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  return d.toDateString() === now.toDateString() ? time : d.toLocaleDateString([], {month: 'short', day: 'numeric'}) + ', ' + time;
}

// A rendering card's little neuron field in the accent colour: the part up to the
// progress point is lit and firing, the rest waits dim; the front edge glows.
function miniField(cv, progress) {
  const ctx = cv.getContext('2d');
  let nodes = [], pulses = [], last = 0, w = 0, h = 0, dpr = 1, accent = '#8B5CF6';
  const css = () => { accent = getComputedStyle(document.documentElement).getPropertyValue('--studio-accent').trim() || accent; };
  const rgb = () => { const n = parseInt(accent.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
  const size = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 2); w = cv.clientWidth; h = cv.clientHeight;
    if (!w || !h) return;
    cv.width = w * dpr; cv.height = h * dpr;
    const count = Math.max(12, Math.round(w * h / 650));
    nodes = Array.from({length: count}, () => ({x: Math.random() * w, y: 4 + Math.random() * (h - 8),
      vx: (Math.random() - .5) * .2, vy: (Math.random() - .5) * .13, glow: 0}));
  };
  css(); size();
  const reach = 46;
  const frame = (now) => {
    if (!cv.isConnected) return;
    requestAnimationFrame(frame);
    if (document.hidden || now - last < 40) return;
    last = now;
    if (!w) { size(); if (!w) return; }
    if (Math.random() < .03) css();
    const [r, g, b] = rgb(), front = progress() * w;
    const lit = (x) => x <= front;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy; n.glow *= .93;
      if (n.x < 0 || n.x > w) n.vx *= -1; if (n.y < 3 || n.y > h - 3) n.vy *= -1;
    }
    // Ahead of the line: only faint, unconnected points waiting to be reached.
    ctx.fillStyle = 'rgba(127,139,201,.22)';
    for (const n of nodes) if (!lit(n.x)) { ctx.beginPath(); ctx.arc(n.x, n.y, 1.1, 0, Math.PI * 2); ctx.fill(); }
    // Behind it: the network, revealed by the line and never drawn past it.
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, front, h); ctx.clip();
    const glow = ctx.createLinearGradient(0, 0, front, 0);
    glow.addColorStop(0, `rgba(${r},${g},${b},.04)`); glow.addColorStop(1, `rgba(${r},${g},${b},.16)`);
    ctx.fillStyle = glow; ctx.fillRect(0, 0, front, h);
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], c = nodes[j], d = Math.hypot(a.x - c.x, a.y - c.y);
      if (d > reach || !lit(a.x) || !lit(c.x)) continue;
      ctx.strokeStyle = `rgba(${r},${g},${b},${(1 - d / reach) * .55 + (a.glow + c.glow) * .3})`;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(c.x, c.y); ctx.stroke();
    }
    for (const n of nodes) if (lit(n.x)) {
      ctx.fillStyle = `rgba(${r},${g},${b},${.7 + n.glow * .3})`;
      ctx.beginPath(); ctx.arc(n.x, n.y, 1.6 + n.glow * 1.6, 0, Math.PI * 2); ctx.fill();
    }
    if (Math.random() < .35) {
      const from = nodes.filter(n => lit(n.x));
      const a = from[Math.floor(Math.random() * from.length)];
      if (a) { const near = nodes.filter(o => o !== a && lit(o.x) && Math.hypot(o.x - a.x, o.y - a.y) < reach);
        if (near.length) pulses.push({a, b: near[Math.floor(Math.random() * near.length)], t: 0}); }
    }
    pulses = pulses.filter(p => {
      p.t += .09;
      if (p.t >= 1) { p.b.glow = 1; return false; }
      const x = p.a.x + (p.b.x - p.a.x) * p.t, y = p.a.y + (p.b.y - p.a.y) * p.t;
      ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.beginPath(); ctx.arc(x, y, 1.3, 0, Math.PI * 2); ctx.fill();
      return true;
    });
    // Every few seconds a soft glimmer runs through the lit part.
    const cycle = (now % 3600) / 1200;
    if (cycle < 1 && front > 20) {
      const gx = cycle * front;
      const shine = ctx.createLinearGradient(gx - 40, 0, gx + 40, 0);
      shine.addColorStop(0, `rgba(${r},${g},${b},0)`); shine.addColorStop(.5, `rgba(255,255,255,.14)`);
      shine.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.fillStyle = shine; ctx.fillRect(gx - 40, 0, 80, h);
    }
    ctx.restore();
    const edge = ctx.createLinearGradient(front - 10, 0, front + 1, 0);
    edge.addColorStop(0, `rgba(${r},${g},${b},0)`); edge.addColorStop(1, `rgba(${r},${g},${b},.9)`);
    ctx.fillStyle = edge; ctx.fillRect(front - 10, 0, 11, h);
  };
  requestAnimationFrame(frame);
}

function stopOthers(except) {
  for (const [, c] of cards) if (c.audio && c.audio !== except && !c.audio.paused) c.audio.pause();
}

function closePops(except) {
  element.querySelectorAll('.tc-pop.open').forEach(p => {
    if (p === except) return;
    p.classList.remove('open');
    const card = p.closest('.tc-card');
    card?.querySelector(`[data-act="${p.dataset.pop}"]`)?.classList.remove('active');
    card?.classList.remove('popped');
    if (p.dataset.pop === 'redo') card?.querySelector('.tc-sel')?.classList.remove('on');
  });
  element.querySelectorAll('.tc-volpop.open').forEach(v => v.classList.remove('open'));
}

function buildCard(n) {
  const card = document.createElement('div');
  card.className = 'tc-card tc-' + n.kind;
  card.dataset.id = n.id;
  if (n.kind === 'pending') {
    card.innerHTML = `<div class="tc-head"><div class="tc-title"><b></b><span class="tc-sub"></span></div></div>
      <div class="tc-pending"><canvas class="tc-mini"></canvas>
        <div class="tc-pfoot"><span class="tc-elapsed">0:00</span><div class="tc-pbar"><i></i></div><span class="tc-eta"></span></div></div>`;
    card.querySelector('b').textContent = n.label;
    card.querySelector('.tc-sub').textContent = n.sub || 'Rendering';
    const el = card.querySelector('.tc-elapsed'), etaEl = card.querySelector('.tc-eta');
    const bar = card.querySelector('.tc-pbar'), fill = bar.querySelector('i');
    if (n.eta) { etaEl.textContent = '~' + fmt(n.eta); etaEl.title = 'Estimated time, padded so it usually finishes early'; }
    else bar.style.visibility = 'hidden';
    const progress = () => n.eta ? Math.min(.97, (Date.now() - n.started) / 1000 / n.eta) : .5;
    const tick = () => {
      if (!card.isConnected) return;
      const secs = (Date.now() - n.started) / 1000;
      el.textContent = fmt(secs);
      fill.style.width = (progress() * 100).toFixed(1) + '%';
      bar.classList.toggle('over', !!n.eta && secs > n.eta);
      setTimeout(tick, 1000);
    };
    setTimeout(tick, 0);
    miniField(card.querySelector('.tc-mini'), progress);
    return {card};
  }
  const stem = n.kind === 'stem';
  card.innerHTML = `
    <div class="tc-head">
      <div class="tc-title"><b></b><span class="tc-sub"></span></div>
      <div class="tc-actions">
        <button class="tc-icon" data-act="folder" title="Show in folder">${ICON.folder}</button>
        ${stem ? '' : `
        <button class="tc-btn" data-act="stems">${ICON.stems}<span>Stems</span></button>
        <div class="tc-pop" data-pop="stems">
          <div class="tc-seg" data-field="mode"><button data-v="4 stems" class="on">4</button><button data-v="6 stems">6</button></div>
          <button class="tc-render" data-act="render-stems">Render</button>
        </div>`}
      </div>
    </div>
    <div class="tc-wave">${bars(n.peaks || [], 'tc-bars')}<div class="tc-prog">${bars(n.peaks || [], 'tc-bars on')}</div>${marks(n)}<div class="tc-sel"></div><div class="tc-cursor"></div></div>
    <div class="tc-ctrl">
      <button class="tc-play" title="Play">${ICON.play}</button>
      <span class="tc-time">0:00 / ${fmt(n.dur)}</span>
      ${stem ? '' : `
      <button class="tc-btn" data-act="extend" title="Extend">${ICON.extend}<span>Extend</span></button>
      <div class="tc-pop" data-pop="extend">
        <div class="tc-seg" data-field="direction"><button data-v="before" title="Add before (new intro)">${ICON.before}</button><button data-v="after" class="on" title="Add after (continue)">${ICON.extend}</button></div>
        <label class="tc-num" title="Seconds to add"><input type="number" data-field="seconds" value="30" min="10" max="120">s</label>
        <label class="tc-num" title="How many runs to render">×<input type="number" data-field="takes" value="1" min="1" max="4"><span class="tc-unit">runs</span></label>
        <button class="tc-render" data-act="render-extend">Render</button>
      </div>
      <div class="tc-pop tc-redo" data-pop="redo">
        <span class="tc-redo-what" title="A new take of the selected part; the rest of the track stays"></span>
        <label class="tc-num" title="How many runs to render">×<input type="number" data-field="takes" value="1" min="1" max="4"><span class="tc-unit">runs</span></label>
        <button class="tc-render" data-act="render-redo">Redo</button>
        <button class="tc-icon" data-act="clear-redo" title="Clear the selection">${ICON.close}</button>
      </div>`}
      <span class="tc-spacer"></span>
      ${stem ? `<button class="tc-icon tc-mute" data-act="mute" title="Mute">${ICON.volume}</button>
      <input type="range" class="tc-vol tc-svol" min="0" max="1" step="0.01" title="This stem's level">` : `
      <span class="tc-vol-icon">${ICON.volume}</span><input type="range" class="tc-vol" min="0" max="1" step="0.01">`}
      ${stem ? '' : `<button class="tc-icon tc-volbtn" data-act="vol" title="Volume">${ICON.volume}</button>
      <div class="tc-volpop"><span class="tc-vol-icon">${ICON.volume}</span><input type="range" class="tc-vol" min="0" max="1" step="0.01"></div>`}
    </div>`;
  card.querySelector('b').textContent = n.label;
  card.querySelector('.tc-sub').textContent = [n.sub, n.took ? 'took ' + fmt(n.took) : ''].filter(Boolean).join(' · ');
  if (n.created) card.querySelector('.tc-sub').title = 'Made ' + made(n.created);
  card.querySelector('b').title = n.label;
  const audio = new Audio();
  audio.preload = 'none';
  const mix = {level: 1};  // a stem's own level, on top of the studio volume
  audio.volume = volume;
  const prog = card.querySelector('.tc-prog'), cursorEl = card.querySelector('.tc-cursor');
  const timeEl = card.querySelector('.tc-time'), play = card.querySelector('.tc-play');
  const vols = [...card.querySelectorAll('.tc-vol:not(.tc-svol)')];
  vols.forEach(v => { v.value = volume; v.style.setProperty('--p', (volume * 100) + '%'); });
  const svol = card.querySelector('.tc-svol');
  if (svol) {
    svol.value = 1; svol.style.setProperty('--p', '100%');
    svol.addEventListener('input', (e) => {
      e.stopPropagation(); mix.level = parseFloat(svol.value);
      svol.style.setProperty('--p', (mix.level * 100) + '%'); audio.volume = volume * mix.level;
      if (audio.muted && mix.level > 0) card.querySelector('[data-act="mute"]').click();
    });
  }
  const ensure = () => { if (!audio.src) audio.src = n.url; };
  const paint = () => {
    const d = audio.duration || n.dur || 1, f = Math.min(1, (audio.currentTime || 0) / d);
    prog.style.clipPath = `inset(0 ${100 - f * 100}% 0 0)`;
    cursorEl.style.left = (f * 100) + '%';
    timeEl.textContent = fmt(audio.currentTime) + ' / ' + fmt(d);
  };
  audio.addEventListener('timeupdate', paint);
  audio.addEventListener('play', () => { play.innerHTML = ICON.pause; card.classList.add('playing'); });
  audio.addEventListener('pause', () => { play.innerHTML = ICON.play; card.classList.remove('playing'); });
  audio.addEventListener('ended', () => { audio.currentTime = 0; paint(); });
  play.addEventListener('click', (e) => { e.stopPropagation(); ensure(); if (audio.paused) { stopOthers(audio); audio.play(); } else audio.pause(); });
  // Drag across the waveform to select a stretch to redo; a plain click still seeks.
  const wave = card.querySelector('.tc-wave'), selEl = card.querySelector('.tc-sel');
  const sel = {a: 0, b: 0}; let picking = null, pickedAt = 0;
  const frac = (x) => { const r = wave.getBoundingClientRect(); return Math.max(0, Math.min(1, (x - r.left) / r.width)); };
  const showSel = (f0, f1) => { selEl.style.left = (f0 * 100) + '%'; selEl.style.width = ((f1 - f0) * 100) + '%'; selEl.classList.add('on'); };
  if (!stem && n.extendable) {
    wave.title = 'Click to play from here · drag across a part to redo it';
    wave.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      picking = {x: e.clientX, f: frac(e.clientX), id: e.pointerId, moved: false};
    });
    wave.addEventListener('pointermove', (e) => {
      if (!picking || e.pointerId !== picking.id) return;
      if (!picking.moved) {
        if (Math.abs(e.clientX - picking.x) < 6) return;
        picking.moved = true; closePops();
        try { wave.setPointerCapture(e.pointerId); } catch (err) {}
      }
      const f = frac(e.clientX); showSel(Math.min(picking.f, f), Math.max(picking.f, f));
    });
    const drop = (e) => {
      if (!picking || e.pointerId !== picking.id) return;
      const p = picking; picking = null;
      if (!p.moved) return;
      pickedAt = performance.now();
      const d = audio.duration || n.dur || 0, f = frac(e.clientX);
      let f0 = Math.min(p.f, f), f1 = Math.max(p.f, f);
      if (f1 > 0.985) f1 = 1;  // dragged to the end: redo everything from there
      sel.a = f0 * d; sel.b = f1 * d;
      if (e.type === 'pointercancel' || sel.b - sel.a < 2) { selEl.classList.remove('on'); return; }
      showSel(f0, f1);
      card.querySelector('.tc-redo-what').textContent = f1 === 1 ? `Redo from ${fmt(sel.a)}` : `Redo ${fmt(sel.a)}–${fmt(sel.b)}`;
      const pop = card.querySelector('[data-pop="redo"]');
      closePops(pop); pop.classList.add('open'); card.classList.add('popped');
      selEl.classList.add('on');
    };
    wave.addEventListener('pointerup', drop);
    wave.addEventListener('pointercancel', drop);
  }
  wave.addEventListener('click', (e) => {
    e.stopPropagation();
    if (performance.now() - pickedAt < 400) return;  // the end of a selection drag, not a seek
    ensure();
    const r = e.currentTarget.getBoundingClientRect();
    const seek = () => { audio.currentTime = ((e.clientX - r.left) / r.width) * (audio.duration || n.dur); paint(); };
    if (audio.readyState >= 1) seek(); else audio.addEventListener('loadedmetadata', seek, {once: true});
    if (audio.paused) { stopOthers(audio); audio.play(); }
  });
  vols.forEach(vol => vol.addEventListener('input', (e) => {
    e.stopPropagation(); volume = parseFloat(vol.value);
    for (const [, c] of cards) if (c.audio) {
      c.audio.volume = volume * (c.mix ? c.mix.level : 1);
      (c.vols || []).forEach(v => { if (v !== vol) v.value = volume; v.style.setProperty('--p', (volume * 100) + '%'); });
    }
    try { localStorage.setItem('studio-volume', String(volume)); } catch (err) {}
  }));
  card.querySelectorAll('.tc-seg').forEach(seg => seg.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return; e.stopPropagation();
    seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  }));
  card.querySelectorAll('[data-act]').forEach(btn => btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const act = btn.dataset.act;
    if (act === 'folder') { trigger('click', {action: 'folder', id: n.id}); return; }
    if (act === 'vol') { card.querySelector('.tc-volpop').classList.toggle('open'); return; }
    if (act === 'mute') {
      audio.muted = !audio.muted; btn.innerHTML = audio.muted ? ICON.muted : ICON.volume;
      btn.classList.toggle('muted', audio.muted); btn.title = audio.muted ? 'Unmute' : 'Mute';
      return;
    }
    if (act === 'extend' || act === 'stems') {
      const pop = card.querySelector(`[data-pop="${act}"]`);
      const opening = !pop.classList.contains('open');
      closePops(pop); pop.classList.toggle('open', opening);
      btn.classList.toggle('active', opening);
      if (!opening) card.querySelector('.tc-volpop')?.classList.remove('open');
      card.classList.toggle('popped', opening);
      if (opening && act === 'extend') trigger('click', {action: 'open_extend', id: n.id});
      return;
    }
    const pop = btn.closest('.tc-pop');
    const seg = (f) => pop.querySelector(`[data-field="${f}"] .on`)?.dataset.v;
    const num = (f) => parseFloat(pop.querySelector(`[data-field="${f}"]`)?.value || '0');
    closePops();
    if (act === 'render-extend') trigger('submit', {action: 'extend', id: n.id, direction: seg('direction'), seconds: num('seconds'), takes: num('takes')});
    if (act === 'render-stems') trigger('submit', {action: 'stems', id: n.id, mode: seg('mode')});
    if (act === 'render-redo') trigger('submit', {action: 'redo', id: n.id, start: sel.a, end: sel.b, takes: num('takes')});
  }));
  card.querySelectorAll('input').forEach(i => ['pointerdown', 'click', 'keydown', 'wheel'].forEach(t => i.addEventListener(t, ev => ev.stopPropagation())));
  card.querySelector('.tc-head').addEventListener('dblclick', () => focusOn(n.id));
  return {card, audio, vols, mix};
}

function focusOn(id, zoom) {
  const c = cards.get(id); if (!c || !c.pos) return;
  if (zoom) view.k = zoom;
  view.x = stage.clientWidth / 2 - (c.pos.x + c.pos.w / 2) * view.k;
  view.y = stage.clientHeight / 2 - (c.pos.y + c.pos.h / 2) * view.k;
  apply();
}

function fitAll() {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [, c] of cards) { if (!c.pos) continue; minX = Math.min(minX, c.pos.x); minY = Math.min(minY, c.pos.y); maxX = Math.max(maxX, c.pos.x + c.pos.w); maxY = Math.max(maxY, c.pos.y + c.pos.h); }
  if (minX === Infinity) return;
  const pad = 30, sw = stage.clientWidth || 600, sh = stage.clientHeight || 400;
  view.k = Math.min(1, (sw - pad * 2) / (maxX - minX), (sh - pad * 2) / (maxY - minY));
  view.x = (sw - (maxX - minX) * view.k) / 2 - minX * view.k;
  view.y = Math.max(pad, (sh - (maxY - minY) * view.k) / 2) - minY * view.k;
  apply();
}

function render(value) {
  const data = value && typeof value === 'object' ? value : {nodes: []};
  const nodes = data.nodes || [];
  if (element.__rev !== data.rev) { element.__rev = data.rev; showNotice(data.notice); }
  empty.style.display = nodes.length ? 'none' : 'flex';
  const seen = new Set();
  for (const n of nodes) {
    seen.add(n.id);
    const sig = n.kind + '|' + (n.file || n.label);
    let c = cards.get(n.id);
    if (c && c.sig !== sig) { if (c.audio) c.audio.pause(); c.card.remove(); c = null; }
    if (!c) {
      c = buildCard(n); c.sig = sig; cards.set(n.id, c);
      c.card.style.width = (n.kind === 'stem' ? SW : W) + 'px';
      nodesEl.appendChild(c.card);
    }
  }
  for (const [id, c] of cards) if (!seen.has(id)) { if (c.audio) c.audio.pause(); c.card.remove(); cards.delete(id); }
  const pos = layout(nodes);
  for (const n of nodes) {
    const c = cards.get(n.id);
    c.pos = pos.get(n.id);
    c.card.style.transform = `translate(${c.pos.x}px, ${c.pos.y}px)`;
  }
  let paths = '';
  for (const n of nodes) {
    if (!n.parent || !pos.has(n.parent)) continue;
    const a = pos.get(n.parent), b = pos.get(n.id);
    const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2, m = (x1 + x2) / 2;
    paths += `<path d="M${x1} ${y1} C ${m} ${y1}, ${m} ${y2}, ${x2} ${y2}"/>`;
  }
  links.innerHTML = paths;
  const bounds = [...pos.values()].reduce((m, p) => [Math.max(m[0], p.x + p.w), Math.max(m[1], p.y + p.h)], [0, 0]);
  links.setAttribute('width', bounds[0] + 10); links.setAttribute('height', bounds[1] + 10);
  if (data.fit || !element.__fitted) { element.__fitted = true; requestAnimationFrame(fitAll); }
  else if (data.focus && cards.has(data.focus)) requestAnimationFrame(() => focusOn(data.focus));
}

if (!element.__wired) {
  element.__wired = true;
  // Clicking anywhere outside an open popout closes it (and the volume pop).
  document.addEventListener('pointerdown', (e) => {
    if (!e.target.closest('.tc-pop, .tc-volpop, [data-act="extend"], [data-act="stems"], [data-act="vol"]')) closePops();
  }, true);
  synapses();
  toast.querySelector('button').addEventListener('click', () => toast.classList.remove('show'));
  window.addEventListener('resize', fitHeight);
  setTimeout(fitHeight, 300);
  let drag = null;
  // Two fingers anywhere on the canvas (cards too) pinch-zoom around their midpoint.
  const touches = new Map();
  let pinch = null;
  const spread = () => { const [a, b] = [...touches.values()]; return {d: Math.hypot(a.x - b.x, a.y - b.y) || 1, x: (a.x + b.x) / 2, y: (a.y + b.y) / 2}; };
  stage.addEventListener('pointerdown', (e) => {
    if (e.pointerType !== 'touch') return;
    touches.set(e.pointerId, {x: e.clientX, y: e.clientY});
    if (touches.size === 2) {
      drag = null; stage.classList.remove('panning');
      const s = spread(), r = stage.getBoundingClientRect();
      pinch = {d: s.d, k: view.k, wx: (s.x - r.left - view.x) / view.k, wy: (s.y - r.top - view.y) / view.k};
    }
  }, true);
  stage.addEventListener('pointermove', (e) => {
    if (!touches.has(e.pointerId)) return;
    touches.set(e.pointerId, {x: e.clientX, y: e.clientY});
    if (!pinch || touches.size < 2) return;
    e.preventDefault();
    const s = spread(), r = stage.getBoundingClientRect();
    view.k = Math.min(2, Math.max(0.25, pinch.k * s.d / pinch.d));
    view.x = s.x - r.left - pinch.wx * view.k; view.y = s.y - r.top - pinch.wy * view.k; apply();
  }, true);
  const lift = (e) => { touches.delete(e.pointerId); if (touches.size < 2) pinch = null; };
  stage.addEventListener('pointerup', lift, true);
  stage.addEventListener('pointercancel', lift, true);
  stage.addEventListener('pointerdown', (e) => {
    if (pinch || e.target.closest('.tc-card')) return;
    closePops(); element.querySelectorAll('.tc-card.popped').forEach(c => c.classList.remove('popped'));
    drag = {x: e.clientX - view.x, y: e.clientY - view.y}; stage.setPointerCapture(e.pointerId); stage.classList.add('panning');
  });
  stage.addEventListener('pointermove', (e) => { if (!drag || pinch) return; view.x = e.clientX - drag.x; view.y = e.clientY - drag.y; apply(); });
  stage.addEventListener('pointerup', () => { drag = null; stage.classList.remove('panning'); });
  stage.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = stage.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.min(2, Math.max(0.25, view.k * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
    view.x = mx - (mx - view.x) * (k / view.k); view.y = my - (my - view.y) * (k / view.k); view.k = k; apply();
  }, {passive: false});
  element.querySelectorAll('.tc-tool').forEach(b => b.addEventListener('click', () => {
    const t = b.dataset.tool;
    if (t === 'fit') return fitAll();
    const k = Math.min(2, Math.max(0.25, view.k * (t === 'in' ? 1.2 : 1 / 1.2)));
    const cx = stage.clientWidth / 2, cy = stage.clientHeight / 2;
    view.x = cx - (cx - view.x) * (k / view.k); view.y = cy - (cy - view.y) * (k / view.k); view.k = k; apply();
  }));
}
// The canvas backdrop: a slow, faint field of neurons. Links form between
// neighbours as they drift, and now and then one fires, a pulse travelling
// down its links and sometimes setting off the next.
function synapses() {
  const cv = element.querySelector('.tc-bg'), ctx = cv.getContext('2d');
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let w = 0, h = 0, dpr = 1, nodes = [], pulses = [], last = 0;
  const reach = 150;
  const resize = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const count = Math.min(170, Math.round(w * h / 8000));
    while (nodes.length < count) nodes.push({x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - .5) * .16, vy: (Math.random() - .5) * .16, r: .8 + Math.random() * 1.6, glow: 0});
    nodes.length = count;
  };
  new ResizeObserver(resize).observe(cv);
  // A soft blue-violet of its own (not the accent), so it sits calmly under any theme.
  const rgba = (a) => `rgba(127,139,201,${a})`, spark = (a) => `rgba(196,206,255,${a})`;
  const fire = (from) => {
    const near = nodes.filter(o => o !== from && Math.hypot(o.x - from.x, o.y - from.y) < reach);
    if (!near.length) return;
    const to = near[Math.floor(Math.random() * near.length)];
    pulses.push({a: from, b: to, t: 0, speed: .008 + Math.random() * .012});
  };
  const frame = (now) => {
    requestAnimationFrame(frame);
    if (document.hidden || now - last < 33) return;  // ~30 fps is plenty for a backdrop
    last = now;
    const px = (view.x * .12) % w, py = (view.y * .12) % h;  // a little parallax with panning
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    for (const n of nodes) {
      if (!still) { n.x += n.vx; n.y += n.vy; }
      if (n.x < -20) n.x += w + 40; if (n.x > w + 20) n.x -= w + 40;
      if (n.y < -20) n.y += h + 40; if (n.y > h + 20) n.y -= h + 40;
      n.glow *= .96;
    }
    const at = (n) => [((n.x + px) % w + w) % w, ((n.y + py) % h + h) % h];
    ctx.lineWidth = 1;
    for (let i = 0; i < nodes.length; i++) {
      const [ax, ay] = at(nodes[i]);
      for (let j = i + 1; j < nodes.length; j++) {
        const [bx, by] = at(nodes[j]);
        const d = Math.hypot(ax - bx, ay - by);
        if (d < reach) {
          ctx.strokeStyle = rgba((1 - d / reach) * .26 + (nodes[i].glow + nodes[j].glow) * .22);
          ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        }
      }
    }
    for (const n of nodes) {
      const [x, y] = at(n);
      ctx.fillStyle = n.glow > .05 ? spark(.45 + n.glow * .55) : rgba(.45);
      ctx.beginPath(); ctx.arc(x, y, n.r + n.glow * 2, 0, Math.PI * 2); ctx.fill();
    }
    if (!still && Math.random() < .06 && nodes.length) fire(nodes[Math.floor(Math.random() * nodes.length)]);
    pulses = pulses.filter(p => {
      p.t += p.speed;
      const [ax, ay] = at(p.a), [bx, by] = at(p.b);
      if (Math.hypot(ax - bx, ay - by) > reach * 1.4) return false;  // wrapped apart
      if (p.t >= 1) { p.b.glow = 1; if (Math.random() < .55 && pulses.length < 40) fire(p.b); return false; }
      const x = ax + (bx - ax) * p.t, y = ay + (by - ay) * p.t;
      const g = ctx.createRadialGradient(x, y, 0, x, y, 8);
      g.addColorStop(0, spark(.9)); g.addColorStop(1, spark(0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill();
      return true;
    });
  };
  resize();
  requestAnimationFrame(frame);
}

apply();
fitHeight();
render(props.value);
watch('value', () => render(props.value));
""".replace("__ICONS__", "{" + ", ".join(f"{name}: '{_svg(name)}'" for name in ICON_PATHS) + "}")
