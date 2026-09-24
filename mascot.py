"""Chalky: a little piece of chalk with eyes that lives in the bar.

It is never completely still (breathes, bobs, sways), blinks, looks at your
mouse while you type, glances around while thinking, follows the pen tip while
drawing (shedding a bit of chalk dust), and bounces when it's done.
"""
from __future__ import annotations

import math
import random
import time
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QCursor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

BODY_TOP, BODY_BOTTOM, OUTLINE = QColor("#E8FFF4"), QColor("#3DDC97"), QColor("#12925E")


class Mascot(QWidget):
    MODES = ("idle", "listening", "thinking", "drawing", "happy", "sad")

    def __init__(self, parent=None, size: int = 46):
        super().__init__(parent)
        self.setFixedSize(size, size + 6)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.mode = "idle"
        self.level = 0.0  # microphone loudness 0..1 while listening
        self.talking = False  # the teacher's voice is speaking: the mouth moves
        self.gaze_provider: Callable[[], QPointF | None] | None = None  # global point to look at
        self._t0 = time.monotonic()
        self._pupil = QPointF(0, 0)
        self._saccade = QPointF(0, 0)
        self._next_saccade = 0.0
        self._blink_at = self._t0 + random.uniform(1.5, 4)
        self._blinks_left = 0
        self._mode_since = self._t0
        self._dust: list[list[float]] = []  # x, y, vy, born
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.update)

    # -- control -----------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.mode = mode
            self._mode_since = time.monotonic()
            if mode == "happy":
                QTimer.singleShot(1600, lambda: self.mode == "happy" and self.set_mode("idle"))

    def showEvent(self, e):
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    # -- behaviour -------------------------------------------------------------------

    def _gaze_target(self, now: float) -> QPointF:
        """Where the pupils want to point, as an offset in -1..1."""
        if self.mode == "thinking":  # glance up, side to side
            side = -1 if int((now - self._mode_since) / 0.9) % 2 == 0 else 1
            return QPointF(0.7 * side, -0.8)
        target = self.gaze_provider() if self.gaze_provider else None
        if target is None and self.mode in ("idle", "listening", "happy"):
            target = QPointF(QCursor.pos())
        if target is not None and self.mode != "sad":
            me = QPointF(self.mapToGlobal(self.rect().center()))
            d = target - me
            dist = math.hypot(d.x(), d.y()) or 1
            reach = min(1.0, dist / 160)
            v = QPointF(d.x() / dist * reach, d.y() / dist * reach)
            if self.mode == "idle" and dist < 30 and now > self._next_saccade:  # mouse parked on us: look around
                self._saccade = QPointF(random.uniform(-1, 1), random.uniform(-0.6, 0.6))
                self._next_saccade = now + random.uniform(0.8, 2.2)
            return v if dist >= 30 or self.mode != "idle" else self._saccade
        if now > self._next_saccade:
            self._saccade = QPointF(random.uniform(-1, 1), random.uniform(-0.6, 0.6))
            self._next_saccade = now + random.uniform(0.8, 2.5)
        return self._saccade if self.mode != "sad" else QPointF(0, 0.8)

    def _blink(self, now: float) -> float:
        """Eye openness 0..1."""
        if now >= self._blink_at:
            phase = (now - self._blink_at) / 0.14
            if phase >= 1:
                if self._blinks_left == 0 and random.random() < 0.2:
                    self._blinks_left = 1
                if self._blinks_left:
                    self._blinks_left -= 1
                    self._blink_at = now + 0.12
                else:
                    self._blink_at = now + random.uniform(2.4, 5.5)
                return 1.0
            return 0.1 + 0.9 * abs(1 - 2 * phase)
        return 1.0

    # -- painting ----------------------------------------------------------------------

    def paintEvent(self, _):
        now = time.monotonic()
        t = now - self._t0
        mt = now - self._mode_since
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height() - 6
        cx, cy = w / 2, h / 2

        # motion per mood
        breathe = math.sin(t * 2.2)
        angle, dy, sx, sy = -12 + 5 * math.sin(t * 1.1), 1.5 * math.sin(t * 1.6), 1 - 0.02 * breathe, 1 + 0.03 * breathe
        if self.mode == "thinking":
            angle = -12 + 7 * math.sin(t * 2.4)
        elif self.mode == "drawing":
            angle = -18 + 9 * math.sin(t * 11)  # scribbling
            dy = 1.2 * math.sin(t * 22)
        elif self.mode == "happy":
            hop = abs(math.sin(mt * 7)) * max(0.0, 1 - mt / 1.4)
            dy, sy, sx = -6 * hop, 1 + 0.08 * hop, 1 - 0.06 * hop
            angle = -12 + 10 * math.sin(mt * 7) * max(0.0, 1 - mt / 1.4)
        elif self.mode == "sad":
            angle, dy = -24, 2.5
        elif self.mode == "listening":
            s = 1 + 0.12 * self.level
            sx, sy = sx * s, sy * s
            dy -= 3 * self.level

        # chalk dust while drawing
        if self.mode == "drawing" and (not self._dust or now - self._dust[-1][3] > 0.09):
            self._dust.append([cx + random.uniform(-4, 4), cy + 14, random.uniform(14, 26), now])
        self._dust = [d for d in self._dust if now - d[3] < 0.9]

        p.save()
        p.translate(cx, cy + dy)
        p.rotate(angle)
        p.scale(sx, sy)
        bw, bh = 22.0, 31.0
        body = QRectF(-bw / 2, -bh / 2, bw, bh)
        # soft shadow
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 60))
        p.drawRoundedRect(body.translated(1.5, 2.5), 9, 9)
        grad = QLinearGradient(0, -bh / 2, 0, bh / 2)
        grad.setColorAt(0, BODY_TOP)
        grad.setColorAt(1, BODY_BOTTOM)
        p.setBrush(QBrush(grad))
        p.setPen(QPen(OUTLINE, 1.3))
        p.drawRoundedRect(body, 9, 9)
        # worn chalk tip + a shine
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 150))
        p.drawEllipse(QRectF(-bw / 2 + 3, -bh / 2 + 3, 6, 9))
        p.setBrush(QColor(18, 146, 94, 60))
        p.drawRoundedRect(QRectF(-bw / 2, bh / 2 - 6, bw, 6), 3, 3)

        # face
        open_ = self._blink(now)
        tgt = self._gaze_target(now)
        self._pupil += (tgt - self._pupil) * 0.28  # quick, smooth eye movement
        ey = -4.0
        for ex in (-5.0, 5.0):
            if self.mode == "happy":
                arc = QPainterPath(QPointF(ex - 3, ey + 1))
                arc.quadTo(QPointF(ex, ey - 3.5), QPointF(ex + 3, ey + 1))
                p.setPen(QPen(QColor("#0F172A"), 1.8, Qt.SolidLine, Qt.RoundCap))
                p.setBrush(Qt.NoBrush)
                p.drawPath(arc)
                continue
            eh = 8.5 * open_ * (0.6 if self.mode == "sad" else 1.0) * (1.1 if self.mode == "listening" else 1.0)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("white"))
            p.drawEllipse(QPointF(ex, ey), 3.6, max(0.6, eh / 2))
            if open_ > 0.35:
                p.setBrush(QColor("#0F172A"))
                px = ex + self._pupil.x() * 1.6
                py = ey + self._pupil.y() * 1.9 * open_
                p.drawEllipse(QPointF(px, py), 2.0, 2.0 * min(1.0, open_ * 1.2))
                p.setBrush(QColor(255, 255, 255, 220))
                p.drawEllipse(QPointF(px + 0.7, py - 0.8), 0.6, 0.6)
        # cheeks
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 120, 150, 80))
        p.drawEllipse(QPointF(-7.5, 2.5), 2.4, 1.5)
        p.drawEllipse(QPointF(7.5, 2.5), 2.4, 1.5)
        # mouth
        p.setPen(QPen(QColor("#0F172A"), 1.4, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        mouth = QPainterPath()
        if self.talking and self.mode not in ("listening", "sad"):
            open_m = 0.6 + 1.9 * abs(math.sin(t * 11)) * (0.6 + 0.4 * math.sin(t * 3.7))
            p.setBrush(QColor("#0F172A"))
            p.drawEllipse(QPointF(0, 4.6), 2.3, max(0.5, open_m))
        elif self.mode == "listening":
            r = 1.2 + 2.2 * self.level
            p.setBrush(QColor("#0F172A"))
            p.drawEllipse(QPointF(0, 4.5), r * 0.9, r)
        elif self.mode == "thinking":
            mouth.moveTo(-2.5, 4.5)
            mouth.cubicTo(-1, 3.5, 1, 5.5, 2.5, 4.5)
            p.drawPath(mouth)
        elif self.mode == "sad":
            mouth.moveTo(-2.5, 5.5)
            mouth.quadTo(0, 3.5, 2.5, 5.5)
            p.drawPath(mouth)
        else:
            mouth.moveTo(-2.8, 3.8)
            mouth.quadTo(0, 6.5 if self.mode == "happy" else 5.6, 2.8, 3.8)
            p.drawPath(mouth)
        p.restore()

        for x, y, vy, born in self._dust:
            age = now - born
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(236, 255, 245, int(200 * (1 - age / 0.9))))
            p.drawEllipse(QPointF(x + math.sin(born * 50) * 3 * age, y + vy * age), 1.2, 1.2)
        p.end()
