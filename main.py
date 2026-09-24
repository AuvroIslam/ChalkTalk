"""ChalkTalk: press the hotkey, ask about what's on screen, and watch it get
explained with circles, arrows, notes and sketches drawn right on top.

    python main.py          # uses Azure OpenAI on Microsoft Foundry (see .env.example)
    python main.py --demo   # no API key: draws a canned explanation to test the overlay
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication

import config
import context
import hotkey
import window
from capture import grab_monitor_under_cursor
from PIL import Image

from compose import Composer, warm_up
from layout import Scene
from ocr import ocr_lines
import voice
from overlay import Bar, Canvas

DEMO = "--demo" in sys.argv

# Video titles and questions can contain emoji or Bangla; never let logging crash on them.
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


class Bridge(QObject):
    """Carries events from worker threads to the Qt thread."""

    hotkey = Signal()
    voice_hotkey = Signal()
    heard = Signal(int, str, str)  # generation, transcript, error
    ready = Signal(int, object)  # generation, session
    action = Signal(int, dict)
    status = Signal(int, str)  # generation, "reading the transcript…" etc.
    finished = Signal(int, str)  # generation, error text ("" if fine)


class DemoSession:
    """Stands in for the model: explains the first real sentence on screen."""

    def __init__(self, scene: Scene, image, selection):
        from ocr import Box

        self.scene = scene
        view = selection if selection is not None else Box(0, 0, scene.w, scene.h)
        scene.set_view(view, int(view.w), int(view.h))

    def cancel(self):
        pass

    def ask(self, question, on_action):
        lines = [l for l in self.scene.visible_lines() if len(l.words) >= 3] or self.scene.visible_lines()
        if not lines:
            on_action({"op": "note", "text": "I can't see any text here."})
            return
        a = lines[len(lines) // 3]
        b = lines[min(len(lines) - 1, len(lines) // 3 + 3)]
        key = " ".join(w.text for w in a.words[:2])
        steps = [
            {"op": "circle", "target": a.id, "phrase": key, "color": "red", "say": f"Look at “{key}” first."},
            {"op": "note", "target": a.id, "phrase": key, "text": "This is the key idea: everything else builds on it.", "color": "red"},
            {"op": "underline", "target": b.id, "color": "blue"},
            {"op": "arrow", "from": a.id, "to": b.id, "label": "leads to", "color": "blue", "say": "And this is where it leads."},
            {"op": "diagram", "target": b.id, "title": "The idea", "nodes": ["Input", "Process", "Result"],
             "edges": [[0, 1, ""], [1, 2, ""], [2, 0, "repeat"]], "color": "purple"},
            {"op": "summary", "text": "First the idea, then what it leads to."},
        ]
        for s in steps:
            time.sleep(0.35)  # pretend to stream
            on_action(s)


class App:
    def __init__(self):
        self.bridge = Bridge()
        self.canvas = Canvas()
        self.bar = Bar()
        self.pill = self.bar  # older name, still used by the test drivers
        self.recorder = voice.Mic()
        self.answering = False
        self._listen_after_capture = False
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.gen = 0
        self.session = None
        self.composer: Composer | None = None
        self.shot = None
        self.scale = 1.0
        self.ocr_job: Future | None = None
        self.ctx_job: Future | None = None
        self.fg = None
        self.t_submit = 0.0
        self.first_action_seen = False

        self.bridge.hotkey.connect(self.on_hotkey)
        self.bridge.voice_hotkey.connect(self.on_voice_hotkey)
        self.bridge.ready.connect(self.on_ready)
        self.bridge.action.connect(self.on_action)
        self.bridge.finished.connect(self.on_finished)
        self.bridge.heard.connect(self.on_heard)
        self.bridge.status.connect(lambda g, s: g == self.gen and self.bar.say(s))
        self.bar.submitted.connect(self.on_submit)
        self.bar.cleared.connect(self.on_stop)
        self.bar.speak.connect(self.on_speak)
        self.canvas.cancelled.connect(self.dismiss)
        self.canvas.item_started.connect(self.bar.say)
        self.canvas.window_focus_request = lambda: (self.bar.raise_(), self.bar.activateWindow(), self.bar.edit.setFocus())
        self.bar.mascot.gaze_provider = self.canvas.pen_position  # eyes follow the pen while drawing
        self.recorder.level.connect(lambda v: setattr(self.bar.mascot, "level", v))
        self.recorder.finished.connect(self.on_recorded)
        self.recorder.failed.connect(self.on_record_failed)

        warm_up()
        self.pool.submit(ocr_lines, Image.new("RGB", (64, 32), "white"))  # load the OCR engine now, not on first use
        if not DEMO:
            self.pool.submit(self._warm_model)
        if config.STT_ENDPOINT:  # voice is real even in demo mode
            self.pool.submit(self._warm_voice)
        hotkey.listen(config.HOTKEY, self.bridge.hotkey.emit, hotkey_id=1)
        hotkey.listen(config.VOICE_HOTKEY, self.bridge.voice_hotkey.emit, hotkey_id=2)
        print(f"ChalkTalk ready{' (demo mode)' if DEMO else ''}. Press {config.HOTKEY} to type a question, "
              f"or {config.VOICE_HOTKEY} to just ask out loud.")

    # -- flow ------------------------------------------------------------------

    def on_hotkey(self):
        fg = window.foreground()
        ours = {int(self.canvas.winId()), int(self.bar.winId())}
        if fg.hwnd not in ours or self.fg is None:
            self.fg = fg  # the app the user is looking at (not our own bar)
        self._cancel()
        self.recorder.stop(deliver=False)
        self.canvas.dismiss()
        self.bar.hide()
        QTimer.singleShot(80, self.capture)  # let the overlay disappear before the screenshot

    # -- voice -------------------------------------------------------------------

    def on_voice_hotkey(self):
        """Hands-free: capture the screen and start listening at once."""
        if self.bar.isVisible() and self.canvas.mode == "ask":
            self.on_speak()
            return
        self._listen_after_capture = True
        self.on_hotkey()

    def on_speak(self):
        if self.recorder.active:
            self.recorder.stop()  # done talking: transcribe now
            return
        self.bar.listening(True)
        self.recorder.start()

    def on_recorded(self, wav: bytes):
        self.bar.listening(False)
        self.bar.thinking(True, "Got it, writing that down")
        g = self.gen

        def work():
            try:
                self.bridge.heard.emit(g, voice.transcribe(wav), "")
            except Exception as e:
                traceback.print_exc()
                self.bridge.heard.emit(g, "", f"{type(e).__name__}: {e}")

        self.pool.submit(work)

    def on_record_failed(self, msg: str):
        self.bar.listening(False)
        self._mood("sad", msg)

    def on_heard(self, g, text: str, error: str):
        if g != self.gen:
            return
        if error or not text:
            self._mood("sad", ("⚠️ " + error[:140]) if error else "I didn't catch that. Try again?")
            return
        print(f"  heard: {text!r}")
        if voice.command_for(text) == "stop":
            self.dismiss()
            return
        self.bar.edit.setText(text)
        self.on_submit(text)

    def _mood(self, mode: str, text: str):
        self.bar.thinking(False)
        self.bar.say(text)
        self.bar.mascot.set_mode(mode)
        if mode == "sad":
            QTimer.singleShot(2500, lambda: self.bar.mascot.mode == "sad" and self.bar.mascot.set_mode("idle"))

    @staticmethod
    def _warm_voice():
        try:
            voice.warm_up()
        except Exception as e:
            print(f"  voice warm-up failed (check CHALK_STT_* in .env): {type(e).__name__}: {e}")

    def on_stop(self):
        """Stop: ends a recording, else stops the answer, else closes."""
        if self.recorder.active:
            self.recorder.stop()
        elif self.answering:
            self._cancel()
            self.gen += 1
            self.answering = False
            self._mood("idle", "Stopped. Ask something else?")
        else:
            self.dismiss()

    def capture(self):
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        self.shot = grab_monitor_under_cursor()
        self.scale = screen.geometry().width() / self.shot.image.width
        self.gen += 1
        self.session = None
        self.ocr_job = self.pool.submit(ocr_lines, self.shot.image)  # runs while you type
        self.ctx_job = None if DEMO else self.pool.submit(context.detect, self.fg)  # so does this
        self.canvas.begin_ask(screen)
        self.bar.begin_ask(screen)
        if self._listen_after_capture:
            self._listen_after_capture = False
            self.on_speak()

    def on_submit(self, question: str):
        follow_up = self.session is not None
        if follow_up:
            self.session.cancel()
            self.gen += 1  # drop anything still arriving from the previous answer
        g = self.gen
        self.t_submit = time.perf_counter()
        self.first_action_seen = False
        if self.recorder.active:
            self.recorder.stop(deliver=False)
            self.bar.listening(False)
        self.answering = True
        selection = self.canvas.selection_box()
        pill_box = self.bar.box_on(self.canvas.screen()).pad(10)
        self.canvas.clear()
        self.canvas.begin_draw()
        self.bar.begin_draw(question)
        if follow_up:
            self.session.scene.reserved[:] = 0
            self.composer = Composer(self.session.scene)
        self.pool.submit(self._work, g, question, selection, follow_up, pill_box)

    def _work(self, g, question, selection, follow_up, pill_box):
        try:
            if not follow_up:
                lines = self.ocr_job.result()
                scene = Scene(self.shot.image, lines, self.scale)
                scene.reserve(pill_box, 3.0)
                if DEMO:
                    session = DemoSession(scene, self.shot.image, selection)
                else:
                    import planner

                    source = self._source()
                    session = planner.Session(scene, self.shot.image, selection, source,
                                              on_status=lambda s: self.bridge.status.emit(g, s))
                self.bridge.ready.emit(g, session)
            else:
                session = self.session
            session.ask(question, lambda a: self.bridge.action.emit(g, a))
            self.bridge.finished.emit(g, "")
        except Exception as e:  # show the problem in the bar instead of dying silently
            traceback.print_exc()
            self.bridge.finished.emit(g, f"{type(e).__name__}: {e}")

    @staticmethod
    def _warm_model():
        import planner

        t = time.perf_counter()
        try:
            planner.warm_up()
            print(f"  model {config.MODEL} warmed up in {time.perf_counter() - t:.1f} s")
        except Exception as e:
            print(f"  model warm-up failed (check .env): {type(e).__name__}: {e}")

    def _source(self):
        try:
            source = self.ctx_job.result(timeout=4)
        except Exception as e:  # context is a bonus; the screen alone still works
            print(f"  context detection failed: {e}")
            return None
        print(f"  context source: {source.kind if source else 'screen only'}"
              f"{' - ' + source.label if source else ''} ({self.fg.exe})")
        return source

    def on_ready(self, g, session):
        if g != self.gen:
            session.cancel()
            return
        self.session = session
        self.composer = Composer(session.scene)
        print(f"  scene ready in {(time.perf_counter() - self.t_submit) * 1000:.0f} ms "
              f"({len(session.scene.lines)} lines, {len(session.scene.regions)} regions)")

    def on_action(self, g, action):
        if g != self.gen or self.composer is None:
            return
        if not self.first_action_seen:
            self.first_action_seen = True
            self.bar.mascot.set_mode("drawing")
            print(f"  first drawing after {(time.perf_counter() - self.t_submit) * 1000:.0f} ms")
        try:
            items = self.composer.build(action)
        except Exception:
            traceback.print_exc()
            return
        if items:
            self.canvas.add(items)

    def on_finished(self, g, error):
        if g != self.gen:
            return
        self.answering = False
        print(f"  answer complete after {(time.perf_counter() - self.t_submit) * 1000:.0f} ms")
        if error:
            self._mood("sad", "⚠️ " + error[:160])
        elif not self.canvas.items:
            self._mood("sad", "I couldn't find anything to mark. Try selecting the area.")
        else:
            QTimer.singleShot(int(self._remaining() * 1000) + 300,
                              lambda: g == self.gen and self._mood("happy", "Ask a follow-up, or press Stop to clear."))

    def _remaining(self) -> float:
        now = time.monotonic()
        return max([s + i.duration - now for i, s in self.canvas.items] + [0])

    def _cancel(self):
        if self.session is not None:
            self.session.cancel()

    def dismiss(self):
        self._cancel()
        self.recorder.stop(deliver=False)
        self.answering = False
        self.gen += 1
        self.session = None
        self.canvas.dismiss()
        self.bar.hide()


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    chalk = App()
    app.aboutToQuit.connect(chalk.recorder.shutdown)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
