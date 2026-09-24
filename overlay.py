"""The transparent drawing layer over the screen, and the ChalkTalk bar."""
from __future__ import annotations

import time
from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal

import config
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from items import Item, pen_at
from mascot import Mascot
from ocr import Box

TOP_FLAGS = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
ACCENT = "#5EEAD4"  # chalk mint


class Canvas(QWidget):
    """Full-screen layer. In 'ask' mode it dims the screen and lets you drag a
    selection; in 'draw' mode it is click-through and plays the drawings."""

    item_started = Signal(str)
    step_started = Signal(str)  # a teaching step begins: speak and caption this
    idle = Signal()             # every step drawn and spoken
    cancelled = Signal()

    def __init__(self):
        super().__init__(None, TOP_FLAGS)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.mode = "hidden"
        self.items: list[tuple[Item, float]] = []
        self.steps: deque[tuple[list[Item], str]] = deque()  # waiting their turn, like a teacher's next point
        self.voice_busy = lambda: False  # replaced by the app: is the narrator still talking?
        self._next_ok_at = 0.0
        self._active = False
        self.announced: set[int] = set()
        self.selection: QRectF | None = None
        self._drag_from: QPointF | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.cancelled.emit)

    # -- modes ---------------------------------------------------------------

    def _show_on(self, screen, click_through: bool) -> None:
        self.hide()
        self.setWindowFlag(Qt.WindowTransparentForInput, click_through)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        self.show()

    def begin_ask(self, screen) -> None:
        self.clear()
        self.mode = "ask"
        self.selection = None
        self.setCursor(Qt.CrossCursor)
        self._show_on(screen, click_through=False)

    def begin_draw(self) -> None:
        self.mode = "draw"
        self._show_on(self.screen(), click_through=True)

    def dismiss(self) -> None:
        self.mode = "hidden"
        self.clear()
        self.hide()

    def clear(self) -> None:
        self.items.clear()
        self.steps.clear()
        self._active = False
        self.announced.clear()
        self.timer.stop()
        self.update()

    def stop_steps(self) -> None:
        """Stop teaching: keep what's drawn, drop what hasn't started."""
        self.steps.clear()

    def add_step(self, items: list[Item], say: str | None) -> None:
        """Queue one teaching step. It starts once the previous step is drawn AND spoken."""
        for it in items:
            it.duration = it.duration / max(0.1, config.DRAW_SPEED)  # a hand writing, not a flash
        self.steps.append((items, say or ""))
        self._active = True
        if not self.timer.isActive():
            self.timer.start()

    def pending(self) -> bool:
        now = time.monotonic()
        return bool(self.steps) or any(now < s + i.duration for i, s in self.items) or self.voice_busy()

    def selection_box(self) -> Box | None:
        s = self.selection
        if s is None or s.width() < 12 or s.height() < 12:
            return None
        return Box(s.x(), s.y(), s.width(), s.height())

    # -- drawing queue -------------------------------------------------------

    def add(self, items: list[Item]) -> None:
        now = time.monotonic()
        last_end = max((s + i.duration for i, s in self.items), default=now)
        start = max(now, last_end)
        for it in items:
            self.items.append((it, start))
            start += it.duration
        if not self.timer.isActive():
            self.timer.start()

    def pen_position(self) -> QPointF | None:
        """Where the 'pen' is right now (global coordinates), for the mascot's eyes."""
        now = time.monotonic()
        for it, start in self.items:
            if start <= now < start + it.duration:
                p = pen_at(it, (now - start) / it.duration)
                return QPointF(self.mapToGlobal(p.toPoint())) if p is not None else None
        return None

    def _tick(self) -> None:
        now = time.monotonic()
        drawing = any(now < s + i.duration for i, s in self.items)
        if self.steps and not drawing and not self.voice_busy() and now >= self._next_ok_at:
            items, say = self.steps.popleft()
            start = now
            for it in items:
                self.items.append((it, start))
                start += it.duration
            self._next_ok_at = start + 0.35  # a breath between steps
            if say:
                self.step_started.emit(say)
            drawing = True
        if not drawing and not self.steps and not self.voice_busy():
            self.timer.stop()
            if self._active:
                self._active = False
                self.idle.emit()
        self.update()

    # -- painting & mouse ----------------------------------------------------

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        if self.mode == "ask":
            p.fillRect(self.rect(), QColor(15, 20, 35, 60))
            if self.selection is not None:
                p.setCompositionMode(QPainter.CompositionMode_Source)
                p.fillRect(self.selection, QColor(0, 0, 0, 1))  # alpha 1 keeps it clickable
                p.setCompositionMode(QPainter.CompositionMode_SourceOver)
                p.setPen(QPen(QColor(ACCENT), 2, Qt.DashLine))
                p.drawRect(self.selection)
        now = time.monotonic()
        for it, start in self.items:
            if now >= start:
                it.paint(p, min(1.0, (now - start) / it.duration))
        p.end()

    def mousePressEvent(self, e):
        if self.mode == "ask" and e.button() == Qt.LeftButton:
            self._drag_from = e.position()
            self.selection = QRectF(self._drag_from, self._drag_from)
            self.update()

    def mouseMoveEvent(self, e):
        if self.mode == "ask" and self._drag_from is not None:
            self.selection = QRectF(self._drag_from, e.position()).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag_from = None
        if self.selection_box() is None:
            self.selection = None
        self.update()
        self.window_focus_request()

    def window_focus_request(self):
        pass  # replaced by the app to hand focus back to the question box


class Bar(QWidget):
    """The ChalkTalk bar at the top of the screen: mascot, your question, the
    tutor's line, a text box, Speak and Reply."""

    submitted = Signal(str)
    cleared = Signal()   # Stop
    speak = Signal()
    voice_toggled = Signal(bool)  # the teacher's voice on/off

    SHADOW = 14  # transparent margin around the card for its shadow

    def __init__(self):
        super().__init__(None, TOP_FLAGS)
        self.setAttribute(Qt.WA_TranslucentBackground)
        m = self.SHADOW
        outer = QVBoxLayout(self)
        outer.setContentsMargins(m + 18, m + 12, m + 16, m + 16)
        outer.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(12)
        self.mascot = Mascot()
        top.addWidget(self.mascot, 0, Qt.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(1)
        self.sub = QLabel("")
        self.sub.setStyleSheet("color:#A1A1AA; font: 600 14px 'Segoe UI';")
        self.main = QLabel("")
        self.main.setWordWrap(True)
        self.main.setStyleSheet("color:#FAFAFA; font: 600 19px 'Segoe UI';")
        texts.addWidget(self.sub)
        texts.addWidget(self.main)
        top.addLayout(texts, 1)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet(
            "QPushButton{background:transparent; color:#A1A1AA; border:none; font: 600 16px 'Segoe UI'; padding:4px 6px;}"
            " QPushButton:hover{color:#FAFAFA;}")
        self.stop_btn.clicked.connect(self.cleared.emit)
        self.voice_btn = QPushButton("🔊" if config.VOICE_OUT else "🔇")
        self.voice_btn.setToolTip("Voice on/off")
        self.voice_btn.setCursor(Qt.PointingHandCursor)
        self.voice_btn.setStyleSheet(
            "QPushButton{background:transparent; color:#A1A1AA; border:none; font: 16px 'Segoe UI Emoji'; padding:4px 6px;}"
            " QPushButton:hover{color:#FAFAFA;}")
        self._voice_on = config.VOICE_OUT
        self.voice_btn.clicked.connect(self._toggle_voice)
        # compact mode (while teaching): a small follow-up box lives in this row
        self.mini = QLineEdit()
        self.mini.setPlaceholderText("Ask a follow-up…")
        self.mini.setFixedWidth(230)
        self.mini.setStyleSheet(
            f"QLineEdit{{background:#0E1110; color:#FAFAFA; border:1.5px solid #3F4A46; border-radius:12px;"
            f" padding:4px 10px; font: 14px 'Segoe UI';}} QLineEdit:focus{{border-color:{ACCENT};}}")
        self.mini.returnPressed.connect(lambda: self.submitted.emit(self.mini.text()))
        self.mini.hide()
        top.addWidget(self.mini, 0, Qt.AlignVCenter)
        top.addWidget(self.voice_btn, 0, Qt.AlignTop)
        top.addWidget(self.stop_btn, 0, Qt.AlignTop)
        outer.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.compact = False
        self.spot: tuple[float, float] | None = None  # top-left of the card, logical, chosen per question
        self.edit = QLineEdit()
        self.edit.setMinimumHeight(46)
        self.edit.setStyleSheet(
            f"QLineEdit{{background:#0E1110; color:#FAFAFA; border:1.5px solid #3F4A46; border-radius:16px;"
            f" padding:6px 16px; font: 17px 'Segoe UI'; selection-background-color:{ACCENT}; selection-color:#0B0F0D;}}"
            f" QLineEdit:focus{{border-color:{ACCENT};}}")
        self.edit.returnPressed.connect(self._submit)
        row.addWidget(self.edit, 1)
        self.speak_btn = QPushButton("🎙 Speak")
        self.speak_btn.setCursor(Qt.PointingHandCursor)
        self._speak_style(False)
        self.speak_btn.clicked.connect(self.speak.emit)
        row.addWidget(self.speak_btn)
        self.reply_btn = QPushButton("Ask")
        self.reply_btn.setCursor(Qt.PointingHandCursor)
        self.reply_btn.setMinimumHeight(46)
        self.reply_btn.setStyleSheet(
            "QPushButton{background:#FAFAFA; color:#0B0F0D; border:none; border-radius:23px; padding:0 22px;"
            " font: 700 17px 'Segoe UI';} QPushButton:hover{background:#FFFFFF;} QPushButton:pressed{background:#D4D4D8;}")
        self.reply_btn.clicked.connect(self._submit)
        row.addWidget(self.reply_btn)
        outer.addLayout(row)

        # Starter prompts: one click asks a typical question for what's on screen.
        self.chips_row = QHBoxLayout()
        self.chips_row.setSpacing(8)
        self.chips: list[QPushButton] = []
        for _ in range(4):
            b = QPushButton("")
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                "QPushButton{background:#232826; color:#D4D4D8; border:1px solid #3F4A46; border-radius:14px;"
                " padding:5px 12px; font: 14px 'Segoe UI';}"
                f" QPushButton:hover{{border-color:{ACCENT}; color:#FFFFFF;}}")
            b.clicked.connect(lambda _=False, btn=b: self._starter(btn.text()))
            self.chips_row.addWidget(b)
            self.chips.append(b)
        self.chips_row.addStretch(1)
        outer.addLayout(self.chips_row)

        QShortcut(QKeySequence(Qt.Key_Escape), self, self.cleared.emit)
        self._dots = 0
        self._thinking_text = ""
        self._thinking = QTimer(self)
        self._thinking.setInterval(350)
        self._thinking.timeout.connect(self._tick)

    def _speak_style(self, listening: bool) -> None:
        color = ACCENT if listening else "#D4D4D8"
        self.speak_btn.setStyleSheet(
            f"QPushButton{{background:transparent; color:{color}; border:none; font: 600 17px 'Segoe UI'; padding:0 6px;}}"
            " QPushButton:hover{color:#FFFFFF;}")

    # -- look ---------------------------------------------------------------------

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        m = self.SHADOW
        card = QRectF(self.rect()).adjusted(m, m, -m, -m)
        for i in range(m, 0, -2):  # soft shadow
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(34 * (1 - i / m) ** 2)))
            p.drawRoundedRect(card.adjusted(-i, -i + 3, i, i + 3), 30 + i, 30 + i)
        p.setBrush(QColor(24, 26, 25, 246))
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        p.drawRoundedRect(card, 30, 30)
        p.end()

    FULL = (900, 196)     # asking: text box, Speak, Ask, starter prompts
    COMPACT = (640, 92)   # teaching: just Chalky, the caption, voice and Stop

    def size_for(self, screen, compact: bool) -> tuple[float, float]:
        w, h = self.COMPACT if compact else self.FULL
        return min(w, screen.geometry().width() - 40), h

    def box_on(self, screen) -> Box:
        """Where the card sits (logical coords on that screen) - drawings avoid it."""
        w, h = self.size_for(screen, self.compact)
        if self.spot is not None:
            return Box(self.spot[0], self.spot[1], w, h)
        return Box((screen.geometry().width() - w) / 2, 12, w, h)

    def set_compact(self, on: bool, screen=None, spot=None) -> None:
        self.compact = on
        for wdg in (self.edit, self.speak_btn, self.reply_btn):
            wdg.setVisible(not on)
        self.show_starters(False if on else self.show_starters_default)
        self.mini.hide()
        if spot is not None:
            self.spot = spot
        if screen is not None:
            self._place(screen)

    def show_followup(self) -> None:
        if self.compact:
            self.mini.clear()
            self.mini.show()
            self.adjustSize()

    show_starters_default = True

    def _place(self, screen) -> None:
        g = screen.geometry()
        b = self.box_on(screen)
        self.setScreen(screen)
        self.setFixedWidth(int(b.w) + 2 * self.SHADOW)
        self.adjustSize()
        self.move(int(g.x() + b.x - self.SHADOW), int(g.y() + b.y - self.SHADOW))

    def _refit(self) -> None:
        self.adjustSize()

    # -- states ---------------------------------------------------------------------

    def begin_ask(self, screen) -> None:
        self._thinking.stop()
        self.sub.setText("ChalkTalk")
        self.main.setText("What's confusing you?")
        self.edit.clear()
        self.edit.setPlaceholderText("Type a question, or press Speak… (drag on screen to select an area)")
        self.reply_btn.setText("Ask")
        self.mascot.set_mode("idle")
        self.set_compact(False)
        self.show_starters(True)
        self._place(screen)
        self.show()
        self.raise_()
        self.activateWindow()
        self.edit.setFocus()

    def begin_draw(self, question: str) -> None:
        self.sub.setText(question.strip() or "Explain this")
        self.edit.clear()
        self.edit.setPlaceholderText("Ask a follow-up…")
        self.reply_btn.setText("Reply")
        self.show_starters(False)
        self.thinking(True)
        self.show()
        self.raise_()

    def listening(self, on: bool) -> None:
        self._speak_style(on)
        self.speak_btn.setText("● Listening" if on else "🎙 Speak")
        if on:
            self._thinking.stop()
            self.main.setText("I'm listening…")
            self.mascot.set_mode("listening")
            self._refit()

    def thinking(self, on: bool, text: str = "Reading your screen") -> None:
        if on:
            self._thinking_text = text
            self._dots = 0
            self._tick()
            self._thinking.start()
            self.mascot.set_mode("thinking")
        else:
            self._thinking.stop()

    def _tick(self) -> None:
        self._dots = (self._dots + 1) % 4
        self.main.setText(self._thinking_text + "." * self._dots)

    def say(self, text: str) -> None:
        self._thinking.stop()
        self.main.setText(text)
        self._refit()

    def _submit(self) -> None:
        self.submitted.emit(self.edit.text() if not self.compact else self.mini.text())

    def _toggle_voice(self) -> None:
        self._voice_on = not self._voice_on
        self.voice_btn.setText("🔊" if self._voice_on else "🔇")
        self.voice_toggled.emit(self._voice_on)

    def set_starters(self, prompts: list[str]) -> None:
        for b, text in zip(self.chips, prompts + [""] * len(self.chips)):
            b.setText(text)
            b.setVisible(bool(text))
        self._refit()

    def show_starters(self, on: bool) -> None:
        for b in self.chips:
            b.setVisible(on and bool(b.text()))
        self._refit()

    def _starter(self, text: str) -> None:
        self.edit.setText(text)
        self.submitted.emit(text)
