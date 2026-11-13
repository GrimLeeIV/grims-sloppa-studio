"""The studio as a desktop app: its own window, an in-app console, phone access.

* ``capture_output()`` copies everything printed (stdout/stderr) into a ring
  buffer for the Console panel, and into ``.temp/studio.log``.
* ``open_window(url)`` shows the UI in a native window (WebView2 on Windows,
  via pywebview) and hides the console window behind it; closing the window
  quits the studio. Without pywebview it falls back to the browser.
* Phone access: when switched on (``settings``), the next start listens on
  the local network, and the Phone panel shows the address and a QR code.
"""

from __future__ import annotations

import collections
import json
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_DIR / ".temp"
LOG_FILE = TEMP_DIR / "studio.log"
SETTINGS_FILE = TEMP_DIR / "studio_settings.json"
LOG_LINES = 600

_lines: collections.deque = collections.deque(maxlen=LOG_LINES)
_lock = threading.Lock()
_partial = [""]


class _Tee:
    """A stream that writes through and keeps a copy of whole lines."""

    def __init__(self, stream: Any, log: Any) -> None:
        self._stream, self._log = stream, log

    def write(self, text: str) -> int:
        try:
            self._stream.write(text)
        except Exception:
            pass
        with _lock:
            chunk = _partial[0] + text.replace("\r\n", "\n")
            *done, _partial[0] = chunk.split("\n")
            for line in done:
                # Progress bars redraw with \r; keep only their final state.
                _lines.append(line.rsplit("\r", 1)[-1])
            if self._log:
                try:
                    self._log.write(text)
                    self._log.flush()
                except Exception:
                    pass
        return len(text)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def capture_output() -> None:
    """Start keeping printed output for the Console panel (idempotent)."""
    if isinstance(sys.stdout, _Tee):
        return
    log = None
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)


def console_text() -> str:
    with _lock:
        return "\n".join(list(_lines) + ([_partial[0]] if _partial[0] else []))


# ---- settings ---------------------------------------------------------------
def settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_setting(key: str, value: Any) -> None:
    data = settings()
    data[key] = value
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[Studio] Could not save the setting: {exc}")


def phone_access() -> bool:
    return bool(settings().get("phone_access"))


def lan_ip() -> Optional[str]:
    """This machine's address on the local network (no traffic is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except Exception:
        return None


def qr_svg(text: str) -> str:
    """A QR code as inline SVG (empty when the qrcode package is missing)."""
    try:
        import qrcode
        import qrcode.image.svg

        image = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
        return image.to_string(encoding="unicode")
    except Exception:
        return ""


def is_local(request: Any) -> bool:
    """Whether a request comes from this machine (not a phone on the network)."""
    host = getattr(getattr(request, "client", None), "host", None) if request is not None else None
    return host in (None, "127.0.0.1", "::1", "localhost")


def pick_folder(start: str = "") -> Optional[str]:
    """The system folder picker, on this machine; None when cancelled."""
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            chosen = filedialog.askdirectory(parent=root, initialdir=start or str(PROJECT_DIR),
                                             title="Choose the output folder")
        finally:
            root.destroy()
        return chosen or None
    except Exception as exc:
        print(f"[Studio] Folder picker unavailable: {exc}")
        return None


def restart() -> None:
    """Start a fresh copy of the studio (it waits for this one to exit), then quit."""
    import subprocess

    env = dict(os.environ, STUDIO_WAIT_PID=str(os.getpid()))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # its log is in the Console panel
    subprocess.Popen([sys.executable] + sys.argv, cwd=os.getcwd(), env=env, creationflags=flags)
    print("[Studio] Restarting…")
    threading.Timer(0.8, lambda: os._exit(0)).start()


def wait_for_previous() -> None:
    """After a restart: let the old studio exit (and free its port) before this one starts."""
    pid = int(os.environ.pop("STUDIO_WAIT_PID", "0") or 0)
    if not pid:
        return
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if handle:
            ctypes.windll.kernel32.WaitForSingleObject(handle, 20000)
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        import time

        for _ in range(100):
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.2)


# ---- the window ---------------------------------------------------------------
def _hide_console() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
    except Exception:
        pass


def skull_icon(accent: str = "#8B5CF6") -> Optional[Path]:
    """The skull logo as a Windows .ico in the given accent colour (drawn with Pillow)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    try:
        accent = accent if len(accent) == 7 and accent.startswith("#") else "#8B5CF6"
        rgb = tuple(int(accent[i:i + 2], 16) for i in (1, 3, 5))
        light = tuple(int(c + (255 - c) * 0.45) for c in rgb)
        n = 512  # drawn large, scaled down for each icon size
        k = n / 32
        grad = Image.new("RGBA", (n, n))
        px = grad.load()
        for y in range(n):
            for x in range(n):
                t = (x + y) / (2 * n)
                px[x, y] = tuple(int(light[i] + (rgb[i] - light[i]) * t) for i in range(3)) + (255,)
        mask = Image.new("L", (n, n), 0)
        d = ImageDraw.Draw(mask)
        d.ellipse([5 * k, 3 * k, 27 * k, 24.2 * k], fill=255)            # cranium
        d.rounded_rectangle([8.5 * k, 16 * k, 23.5 * k, 27 * k], radius=2 * k, fill=255)  # jaw
        skull = Image.composite(grad, Image.new("RGBA", (n, n), (0, 0, 0, 0)), mask)
        d = ImageDraw.Draw(skull)
        dark = (11, 12, 18, 255)
        for cx in (11.6, 20.4):
            d.ellipse([(cx - 2.7) * k, 11.1 * k, (cx + 2.7) * k, 16.5 * k], fill=dark)
        d.polygon([(16 * k, 17.2 * k), (14.5 * k, 19.8 * k), (17.5 * k, 19.8 * k)], fill=dark)
        for x, y1, y2 in ((11.5, 22.6, 24.2), (13.8, 21.8, 25), (16, 22.2, 24.6), (18.2, 21.8, 25), (20.5, 22.6, 24.2)):
            d.line([(x * k, y1 * k), (x * k, y2 * k)], fill=dark, width=int(1.4 * k))
            for y in (y1, y2):
                d.ellipse([(x - .7) * k, (y - .7) * k, (x + .7) * k, (y + .7) * k], fill=dark)
        path = TEMP_DIR / f"skull_{accent[1:].lower()}.ico"
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        skull.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
        return path
    except Exception as exc:
        print(f"[Studio] Couldn't draw the window icon: {exc}")
        return None


class WindowApi:
    """Called from the page (window.pywebview.api.*): the studio's own title-bar buttons."""

    def __init__(self) -> None:
        self._window: Any = None

    def _form(self) -> Any:
        return getattr(self._window, "native", None)

    def _invoke(self, fn: Any) -> None:
        form = self._form()
        if form is None:
            return
        from System import Action  # pythonnet, present with pywebview on Windows

        form.Invoke(Action(fn))

    def minimize(self) -> None:
        self._window.minimize()

    def toggle_maximize(self) -> bool:
        """Maximize to the work area (the taskbar stays visible), or restore."""
        state = {}

        def run() -> None:
            import System.Windows.Forms as WinForms

            form = self._form()
            if form.WindowState == WinForms.FormWindowState.Maximized:
                form.WindowState = WinForms.FormWindowState.Normal
                state["max"] = False
            else:
                form.MaximizedBounds = WinForms.Screen.FromHandle(form.Handle).WorkingArea
                form.WindowState = WinForms.FormWindowState.Maximized
                state["max"] = True

        self._invoke(run)
        return bool(state.get("max"))

    def is_maximized(self) -> bool:
        try:
            import System.Windows.Forms as WinForms

            return self._form().WindowState == WinForms.FormWindowState.Maximized
        except Exception:
            return False

    def close(self) -> None:
        self._window.destroy()

    def _snap(self) -> None:
        """Aero-Snap-like ending to a drag (frameless windows get none from Windows):
        cursor at the top edge maximizes, at the left/right edge fills that half."""
        import System.Windows.Forms as WinForms
        from System.Drawing import Rectangle

        form = self._form()
        pos = WinForms.Cursor.Position
        area = WinForms.Screen.FromPoint(pos).WorkingArea
        if form.WindowState != WinForms.FormWindowState.Normal:
            return
        if pos.Y <= area.Top + 4:
            form.MaximizedBounds = area
            form.WindowState = WinForms.FormWindowState.Maximized
        elif pos.X <= area.Left + 4:
            form.Bounds = Rectangle(area.Left, area.Top, area.Width // 2, area.Height)
        elif pos.X >= area.Right - 5:
            form.Bounds = Rectangle(area.Left + area.Width // 2, area.Top, area.Width - area.Width // 2, area.Height)

    def snap(self) -> None:
        """Called by the page when its own drag ends."""
        self._invoke(self._snap)

    def restore_at(self, fraction: float, grab_y: int) -> None:
        """Leave maximized for a drag: restore so the cursor keeps the same spot on the window."""
        def run() -> None:
            import System.Windows.Forms as WinForms
            from System.Drawing import Point

            form = self._form()
            if form.WindowState != WinForms.FormWindowState.Maximized:
                return
            form.WindowState = WinForms.FormWindowState.Normal
            pos = WinForms.Cursor.Position
            form.Location = Point(int(pos.X - form.Width * max(0.0, min(float(fraction), 1.0))),
                                  int(pos.Y - grab_y))

        self._invoke(run)

    def enable_drag_regions(self) -> bool:
        """Let the page mark title-bar areas (CSS app-region: drag). Windows then treats them
        as the real caption: dragging, Aero Snap, Snap Layouts and double-click maximize.
        Needs WebView2 SDK 1.0.2420+; applies after a reload."""
        state = {"ok": False}

        def run() -> None:
            import System.Windows.Forms as WinForms

            form = self._form()
            form.MaximizedBounds = WinForms.Screen.FromHandle(form.Handle).WorkingArea
            # Windows' own moves end here: snap like Aero would.
            from System import EventHandler

            form.ResizeEnd += EventHandler(lambda *_: self._snap())
            for control in form.Controls:
                core = getattr(control, "CoreWebView2", None)
                if core is not None:
                    core.Settings.IsNonClientRegionSupportEnabled = True
                    core.Reload()
                    state["ok"] = True

        try:
            self._invoke(run)
        except Exception as exc:
            print(f"[Studio] Native title-bar regions unavailable ({exc}); using the fallback drag.")
        return state["ok"]

    def add_taskbar_behaviour(self) -> None:
        """A frameless form lacks the minimize/maximize styles, so clicking its taskbar
        button does nothing; add them back (they draw nothing without a title bar)."""
        try:
            import ctypes

            hwnd = int(self._form().Handle.ToInt64())
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, -16)
            user32.SetWindowLongW(hwnd, -16, style | 0x00020000 | 0x00010000 | 0x00080000)
        except Exception as exc:
            print(f"[Studio] Couldn't enable taskbar minimize: {exc}")

    def set_accent(self, accent: str) -> None:
        """Repaint the taskbar/window icon in the current accent colour."""
        path = skull_icon(accent)
        if path is None or self._form() is None:
            return

        def run() -> None:
            from System.Drawing import Icon

            self._form().Icon = Icon(str(path))

        self._invoke(run)


def _app_id() -> None:
    """Group the window under its own taskbar entry (so it shows the studio's icon, not Python's)."""
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GrimSloppa.Studio")
        except Exception:
            pass


def open_window(url: str, title: str = "Grim's Sloppa Studio") -> bool:
    """Show the studio in its own frameless window until it's closed. False if that isn't possible."""
    if os.environ.get("STUDIO_BROWSER") == "1":
        return False
    try:
        import webview
    except Exception as exc:
        print(f"[Studio] No app window ({exc}); opening the browser instead.")
        return False
    try:
        _app_id()
        api = WindowApi()
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = False
        api._window = webview.create_window(
            title, url, js_api=api, width=1600, height=950, min_size=(900, 600), background_color="#0A0B10",
            text_select=True, frameless=True, easy_drag=False, shadow=True,
        )

        def started() -> None:
            _hide_console()
            api._window.events.shown.wait(15)
            api.add_taskbar_behaviour()
            api.set_accent("#8B5CF6")  # the page sends the saved accent once it loads
            api.toggle_maximize()
            api._window.events.loaded.wait(60)
            api.enable_drag_regions()

        storage = TEMP_DIR / "window"
        storage.mkdir(parents=True, exist_ok=True)
        # Not private: the window keeps its look (localStorage) between runs.
        webview.start(func=started, private_mode=False, storage_path=str(storage))
        return True
    except Exception as exc:
        print(f"[Studio] The app window failed ({exc}); opening the browser instead.")
        return False
