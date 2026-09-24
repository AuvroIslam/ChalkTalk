"""Drives the real overlay end to end and saves a screenshot of the result.

    python selftest.py out.png [--demo] [--primary] [--question "..."]

--primary moves the mouse to the primary monitor first (to test DPI scaling there).
"""
from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

import main as chalk


def run():
    out = sys.argv[1]
    question = sys.argv[sys.argv.index("--question") + 1] if "--question" in sys.argv else "explain this"
    if "--primary" in sys.argv:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2, like Qt
        ctypes.windll.user32.SetCursorPos(200, 200)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    chalk_app = chalk.App()

    def snap():
        import mss
        from PIL import Image

        with mss.MSS() as sct:  # the monitor the overlay is on, even if the mouse moved
            mon = next(m for m in sct.monitors[1:]
                       if m["left"] == chalk_app.shot.left and m["top"] == chalk_app.shot.top)
            raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        img.save(out)
        print("saved", out, img.size)
        app.quit()

    def submit():
        chalk_app.on_submit(question)
        QTimer.singleShot(int(float(__import__("os").environ.get("CHALK_SELFTEST_WAIT", "9")) * 1000), snap)

    QTimer.singleShot(1200, chalk_app.on_hotkey)
    QTimer.singleShot(2500, submit)
    app.exec()


if __name__ == "__main__":
    run()
