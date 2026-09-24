"""End-to-end on real windows: opens Chrome on the primary screen, runs the real
hotkey flow with the real model, and screenshots each result.

    python tests/live_e2e.py OUT_DIR
Don't touch the mouse/keyboard while it runs.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PDF = (Path(__file__).resolve().parents[2] / "EN-WBNR-SlideDeck-SREVM92894-pdf.pdf").as_posix()
SLIDE = (Path(__file__).resolve().parents[2] / "submission" / "lecture_slide.png").as_posix()
CASES = [
    ("slide", f"file:///{SLIDE}", "lecture_slide", "What does the eta symbol mean here?", False),
    ("honesty", f"file:///{SLIDE}", "lecture_slide", "What did my lecturer say about this slide yesterday?", False),
    ("youtube", "https://www.youtube.com/watch?v=aircAruvnKk&t=662s", "YouTube",
     "What is the bias he is talking about, and why is it negative?", True),
    ("pdf", f"file:///{PDF}", ".pdf", "How are entries judged? What are the criteria and points?", False),
    ("web", "https://en.wikipedia.org/wiki/Gradient_descent", "Gradient descent",
     "What variants of this method does the article list further down?", False),
]
user32 = ctypes.windll.user32


def find_window(fragment, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        hits = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def cb(h, _):
            b = ctypes.create_unicode_buffer(400)
            user32.GetWindowTextW(h, b, 400)
            if user32.IsWindowVisible(h) and fragment in b.value and "Google Chrome" in b.value:
                hits.append(h)
            return True

        user32.EnumWindows(cb, 0)
        if hits:
            return hits[0]
        time.sleep(0.5)
    return None


def bring_front(hwnd):
    user32.ShowWindow(hwnd, 9)  # restore
    user32.SetWindowPos(hwnd, 0, 60, 60, 1400, 900, 0x0040)
    user32.ShowWindow(hwnd, 3)  # maximise on the primary screen
    user32.keybd_event(0x12, 0, 0, 0)  # a tap of Alt lets us take the foreground
    user32.keybd_event(0x12, 0, 2, 0)
    user32.SetForegroundWindow(hwnd)
    user32.SetCursorPos(700, 700)


def press_k():  # YouTube: pause
    user32.keybd_event(0x4B, 0, 0, 0)
    user32.keybd_event(0x4B, 0, 2, 0)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    import main as chalk
    import mss
    from PIL import Image

    chalk_app = chalk.App()
    queue = list(CASES)
    state = {}

    def snap(name, hwnd):
        with mss.MSS() as sct:
            mon = next(m for m in sct.monitors[1:] if m["left"] == chalk_app.shot.left and m["top"] == chalk_app.shot.top)
            raw = sct.grab(mon)
        path = out / f"live_{name}.png"
        Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX").save(path)
        print(f"  saved {path}")
        chalk_app.dismiss()
        user32.PostMessageW(hwnd, 0x0010, 0, 0)  # close the test window
        QTimer.singleShot(1500, next_case)

    def finished(g, err):
        name, hwnd = state["case"]
        QTimer.singleShot(int(chalk_app._remaining() * 1000) + 1200, lambda: snap(name, hwnd))

    chalk_app.bridge.finished.connect(finished)

    def next_case():
        if not queue:
            app.quit()
            return
        name, url, title, question, pause = queue.pop(0)
        print(f"\n== {name}: {question}")
        subprocess.Popen([CHROME, "--new-window", url])
        hwnd = find_window(title)
        if not hwnd:
            print("  window didn't open")
            QTimer.singleShot(100, next_case)
            return
        time.sleep(6)
        bring_front(hwnd)
        time.sleep(1)
        if pause:
            press_k()
            time.sleep(1)
        state["case"] = (name, hwnd)
        chalk_app.on_hotkey()
        QTimer.singleShot(2500, lambda: chalk_app.on_submit(question))

    QTimer.singleShot(4000, next_case)  # give the model warm-up a head start
    app.exec()


if __name__ == "__main__":
    main()
