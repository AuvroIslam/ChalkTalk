"""Animated, hand-drawn-looking marks. Each item paints itself at progress t (0..1)."""
from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen

from ocr import Box

PEN_W = 3.0


def _color(c: str, alpha: int = 255) -> QColor:
    q = QColor(c)
    q.setAlpha(alpha)
    return q


def _pen(c: str, width: float = PEN_W, alpha: int = 255) -> QPen:
    pen = QPen(_color(c, alpha), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _rect(b: Box) -> QRectF:
    return QRectF(b.x, b.y, b.w, b.h)


def _draw_partial(p: QPainter, pts: list[QPointF], t: float) -> None:
    """Draw the first fraction t of a polyline, measured by length."""
    if len(pts) < 2 or t <= 0:
        return
    p.setBrush(Qt.NoBrush)
    seg = [math.dist((a.x(), a.y()), (b.x(), b.y())) for a, b in zip(pts, pts[1:])]
    goal = sum(seg) * min(t, 1.0)
    path = QPainterPath(pts[0])
    for a, b, d in zip(pts, pts[1:], seg):
        if goal >= d:
            path.lineTo(b)
            goal -= d
        else:
            f = goal / d if d else 0
            path.lineTo(QPointF(a.x() + (b.x() - a.x()) * f, a.y() + (b.y() - a.y()) * f))
            break
    p.drawPath(path)


def _text_with_halo(p: QPainter, pos: QPointF, text: str, font: QFont, ink: str, halo: str | None) -> None:
    path = QPainterPath()
    path.addText(pos, font, text)
    if halo:
        p.strokePath(path, _pen(halo, 4.0, 230))
    p.fillPath(path, QBrush(_color(ink)))


class Item:
    duration = 0.4
    say: str | None = None

    def paint(self, p: QPainter, t: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class Circle(Item):
    duration = 0.45

    def __init__(self, box: Box, ink: str):
        self.ink = ink
        rnd = random.Random(int(box.x * 7 + box.y * 13))
        cx, cy = box.cx, box.cy
        # A superellipse hugs a line of text much tighter than an ellipse,
        # so the loop doesn't swallow the neighbouring words.
        rx, ry = box.w / 2 + 5 + box.h * 0.1, box.h / 2 + 5 + box.h * 0.15
        start = rnd.uniform(-2.6, -2.0)  # start top-left like a hand does
        phase = rnd.uniform(0, 6.28)
        self.pts = []
        n = 90
        for k in range(n + 1):
            f = k / n
            a = start + f * 2 * math.pi * 1.08
            c, s = math.cos(a), math.sin(a)
            ex = math.copysign(abs(c) ** 0.6, c)
            ey = math.copysign(abs(s) ** 0.6, s)
            wob = 1 + 0.03 * math.sin(3 * a + phase) + 0.06 * f
            self.pts.append(QPointF(cx + rx * wob * ex, cy + ry * wob * ey))

    def paint(self, p, t):
        p.setPen(_pen(self.ink))
        _draw_partial(p, self.pts, t)


class Underline(Item):
    duration = 0.3

    def __init__(self, box: Box, ink: str):
        self.ink = ink
        y = box.y2 + 3
        n = 24
        self.pts = [
            QPointF(box.x - 2 + (box.w + 4) * k / n, y + 1.2 * math.sin(k * 0.9) + 1.5 * k / n) for k in range(n + 1)
        ]

    def paint(self, p, t):
        p.setPen(_pen(self.ink))
        _draw_partial(p, self.pts, t)


class Highlight(Item):
    duration = 0.3

    def __init__(self, box: Box, color: str, alpha: int):
        self.box = box.pad(2)
        self.color = color
        self.alpha = alpha

    def paint(self, p, t):
        b = self.box
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(_color(self.color, self.alpha)))
        p.drawRoundedRect(QRectF(b.x, b.y, b.w * min(t, 1), b.h), 3, 3)


class Frame(Item):
    duration = 0.5

    def __init__(self, box: Box, ink: str):
        self.ink = ink
        b = box.pad(8)
        rnd = random.Random(int(b.x + b.y))
        corners = [(b.x, b.y), (b.x2, b.y), (b.x2, b.y2), (b.x, b.y2), (b.x, b.y)]
        self.pts = []
        for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
            for k in range(12):
                f = k / 12
                self.pts.append(QPointF(x0 + (x1 - x0) * f + rnd.uniform(-0.8, 0.8), y0 + (y1 - y0) * f + rnd.uniform(-0.8, 0.8)))
        self.pts.append(QPointF(b.x + 3, b.y - 1))

    def paint(self, p, t):
        p.setPen(_pen(self.ink, 2.5))
        _draw_partial(p, self.pts, t)


def _curve(a: QPointF, b: QPointF, bend: float, n: int = 32) -> list[QPointF]:
    dx, dy = b.x() - a.x(), b.y() - a.y()
    mx, my = (a.x() + b.x()) / 2 - dy * bend, (a.y() + b.y()) / 2 + dx * bend
    out = []
    for k in range(n + 1):
        f = k / n
        x = (1 - f) ** 2 * a.x() + 2 * (1 - f) * f * mx + f**2 * b.x()
        y = (1 - f) ** 2 * a.y() + 2 * (1 - f) * f * my + f**2 * b.y()
        out.append(QPointF(x, y))
    return out


def _arrow_head(p: QPainter, tip: QPointF, prev: QPointF, size: float = 11) -> None:
    ang = math.atan2(tip.y() - prev.y(), tip.x() - prev.x())
    for s in (-1, 1):
        a = ang + math.pi - s * 0.45
        p.drawLine(tip, QPointF(tip.x() + size * math.cos(a), tip.y() + size * math.sin(a)))


class Arrow(Item):
    duration = 0.45

    def __init__(self, a: QPointF, b: QPointF, ink: str, bend: float = 0.18, width: float = PEN_W):
        self.ink = ink
        self.width = width
        self.pts = _curve(a, b, bend)

    def mid(self) -> QPointF:
        return self.pts[len(self.pts) // 2]

    def paint(self, p, t):
        p.setPen(_pen(self.ink, self.width))
        _draw_partial(p, self.pts, t)
        if t >= 1:
            _arrow_head(p, self.pts[-1], self.pts[-4])


class Label(Item):
    """Small text (arrow labels). Always haloed so it reads on anything."""

    duration = 0.2

    def __init__(self, pos: QPointF, text: str, font: QFont, ink: str, halo: str):
        self.pos, self.text, self.font, self.ink, self.halo = pos, text, font, ink, halo

    def paint(self, p, t):
        n = len(self.text) if t >= 1 else max(1, int(len(self.text) * t))
        _text_with_halo(p, self.pos, self.text[:n], self.font, self.ink, self.halo)


class Badge(Item):
    duration = 0.2

    def __init__(self, center: QPointF, n: str, ink: str, font: QFont):
        self.c, self.n, self.ink, self.font = center, n, ink, font

    def paint(self, p, t):
        r = 12 * (0.6 + 0.4 * min(t, 1))
        p.setPen(_pen("#FFFFFF", 2))
        p.setBrush(QBrush(_color(self.ink)))
        p.drawEllipse(self.c, r, r)
        if t >= 1:
            p.setPen(_pen("#FFFFFF", 1))
            p.setFont(self.font)
            p.drawText(QRectF(self.c.x() - r, self.c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, self.n)


class Note(Item):
    """Handwritten text. On empty space it is written straight onto the screen
    with a thin halo; over busy content it sits on a sticky-note card."""

    def __init__(self, box: Box, lines: list[str], font: QFont, line_h: float, top: float, ink: str,
                 card: str | None, halo: str | None, leader: tuple[QPointF, QPointF] | None, pad: float):
        self.box, self.lines, self.font, self.line_h, self.top = box, lines, font, line_h, top
        self.ink, self.card, self.halo, self.leader, self.pad = ink, card, halo, leader, pad
        chars = sum(len(s) for s in lines)
        self.duration = min(1.1, 0.25 + chars * 0.012)

    def paint(self, p, t):
        if self.leader:
            a, b = self.leader
            p.setPen(_pen(self.ink, 2.0, 220))
            lt = min(1.0, t / 0.3)
            _draw_partial(p, _curve(a, b, 0.12, 20), lt)
            if lt >= 1:
                p.setBrush(QBrush(_color(self.ink)))
                p.drawEllipse(b, 3.2, 3.2)
        if self.card:
            r = _rect(self.box)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 45)))
            p.drawRoundedRect(r.translated(2, 3), 8, 8)
            p.setBrush(QBrush(_color(self.card, 245)))
            p.setPen(_pen(self.ink, 1.5, 200))
            p.drawRoundedRect(r, 8, 8)
        total = sum(len(s) for s in self.lines)
        show = int(total * min(1.0, max(0.0, (t - 0.1) / 0.9)) + 0.999)
        y = self.box.y + self.pad + self.top
        for s in self.lines:
            if show <= 0:
                break
            _text_with_halo(p, QPointF(self.box.x + self.pad, y), s[:show], self.font, self.ink,
                            None if self.card else self.halo)
            show -= len(s)
            y += self.line_h


class Group(Item):
    """Several items drawn one after another as a single step (e.g. a diagram)."""

    def __init__(self, parts: list[Item], backdrop: Box | None = None, card: str | None = None, ink: str = "#000"):
        self.parts, self.backdrop, self.card, self.ink = parts, backdrop, card, ink
        self.duration = sum(x.duration for x in parts) * 0.8 + 0.1

    def paint(self, p, t):
        if self.card and self.backdrop:
            r = _rect(self.backdrop)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 45)))
            p.drawRoundedRect(r.translated(2, 3), 10, 10)
            p.setBrush(QBrush(_color(self.card, 245)))
            p.setPen(_pen(self.ink, 1.5, 200))
            p.drawRoundedRect(r, 10, 10)
        total = sum(x.duration for x in self.parts)
        at = t * total
        for x in self.parts:
            if at <= 0:
                break
            x.paint(p, 1.0 if t >= 1 else min(1.0, at / x.duration))
            at -= x.duration


def pen_at(item: Item, t: float) -> QPointF | None:
    """Where the pen tip is while this item is being drawn (for the mascot's eyes)."""
    t = min(max(t, 0.0), 1.0)
    if isinstance(item, Group):
        total = sum(x.duration for x in item.parts) or 1
        at = t * total
        for x in item.parts:
            if at <= x.duration:
                return pen_at(x, at / x.duration)
            at -= x.duration
        return None
    pts = getattr(item, "pts", None)
    if pts:
        return pts[min(len(pts) - 1, int(t * (len(pts) - 1)))]
    box = getattr(item, "box", None)
    if box is not None:
        return QPointF(box.x + box.w * t, box.cy)
    for attr in ("pos", "c"):
        if hasattr(item, attr):
            return getattr(item, attr)
    return None


class RoundBox(Item):
    duration = 0.35

    def __init__(self, box: Box, ink: str, fill: str | None):
        self.box, self.ink, self.fill = box, ink, fill

    def paint(self, p, t):
        b = self.box
        p.setPen(_pen(self.ink, 2.2))
        p.setBrush(QBrush(_color(self.fill, 235)) if self.fill else Qt.NoBrush)
        r = QRectF(b.x, b.y, b.w, b.h)
        if t < 1:
            p.setOpacity(t)
        p.drawRoundedRect(r, 7, 7)
        p.setOpacity(1)
