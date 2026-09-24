"""Turns the model's drawing actions into placed, coloured items.

The model only says *what* to mark and explain. Everything about *where* (exact
pixels, free space, collision avoidance) and *how it stays readable* (colours,
halos, cards) is decided here from measured geometry.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF
from PySide6.QtGui import QFont, QFontMetricsF

import config
from items import Arrow, Badge, Circle, Frame, Group, Highlight, Item, Label, Note, RoundBox, Underline
from layout import Scene, hex_rgb, luminance
from ocr import Box

LIGHT_CARD, DARK_CARD = "#FFF6BF", "#1E2A3A"


def _font(px: int, bold: bool = False, family: str | None = None) -> QFont:
    f = QFont(family or config.NOTE_FONT)
    f.setPixelSize(px)
    f.setBold(bold)
    return f


def _metrics(fm: QFontMetricsF) -> tuple[float, float, float]:
    """(first baseline from top, line height, space below last baseline).
    Measured from cap/x-height because handwriting fonts like Segoe Print
    report a huge ascent that would space lines far apart."""
    top = fm.capHeight() * 1.15
    line_h = max(fm.capHeight() * 1.9, fm.xHeight() * 2.7)
    bottom = fm.descent() * 0.75
    return top, line_h, bottom


def _wrap(text: str, fm: QFontMetricsF, max_w: float) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            trial = f"{cur} {word}".strip()
            if cur and fm.horizontalAdvance(trial) > max_w:
                out.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            out.append(cur)
    return out or [""]


def _edge_point(b: Box, toward: QPointF, gap: float = 4) -> QPointF:
    """Where the ray from b's centre toward a point leaves the (padded) box."""
    b = b.pad(gap)
    dx, dy = toward.x() - b.cx, toward.y() - b.cy
    if dx == 0 and dy == 0:
        return QPointF(b.cx, b.cy)
    sx = (b.w / 2) / abs(dx) if dx else math.inf
    sy = (b.h / 2) / abs(dy) if dy else math.inf
    s = min(sx, sy)
    return QPointF(b.cx + dx * s, b.cy + dy * s)


def _center(b: Box) -> QPointF:
    return QPointF(b.cx, b.cy)


def warm_up() -> None:
    """Load the fonts once at startup (the first use costs ~200 ms each)."""
    from PySide6.QtGui import QPainterPath

    for f in (_font(config.NOTE_FONT_PX), _font(config.NOTE_FONT_PX + 1, True), _font(13, True, "Segoe UI")):
        QFontMetricsF(f).horizontalAdvance("warm up")
        QPainterPath().addText(0, 0, f, "warm up")


class Composer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.note_font = _font(config.NOTE_FONT_PX)
        self.small_font = _font(max(11, config.NOTE_FONT_PX - 3))
        self.node_font = _font(max(12, config.NOTE_FONT_PX - 2))
        self.badge_font = _font(13, True, "Segoe UI")
        self.last_ink_name = "red"
        self.marked: list[Box] = []
        self.skipped_duplicates = 0
        self.max_note_w = min(300.0, scene.w * 0.3)

    # -- helpers -------------------------------------------------------------

    def _ink_name(self, a: dict) -> str:
        name = a.get("color") or self.last_ink_name
        self.last_ink_name = name
        return name

    def _halo(self, b: Box) -> str:
        r, g, bl = self.scene.bg_color(b)
        return f"#{int(r):02X}{int(g):02X}{int(bl):02X}"

    def _card_for(self, b: Box) -> str:
        return DARK_CARD if self.scene.is_dark(b) else LIGHT_CARD

    def _ink_on_card(self, name: str, card: str) -> str:
        from layout import INKS

        on_light, on_dark = INKS.get(name.lower(), INKS["red"])
        return on_light if luminance(hex_rgb(card)) > 0.3 else on_dark

    # -- public ----------------------------------------------------------------

    def build(self, a: dict) -> list[Item]:
        op = str(a.get("op", "")).lower()
        fn = getattr(self, f"_op_{op}", None)
        if fn is None:
            return []
        items = fn(a) or []
        say = a.get("say")
        if items and isinstance(say, str) and say.strip():
            items[0].say = say.strip()
        return items

    # -- marks -----------------------------------------------------------------

    def _already_marked(self, b: Box) -> bool:
        """One emphasis per spot: a box + highlight + underline on the same line is just noise."""
        for m in self.marked:
            ix = max(0.0, min(b.x2, m.x2) - max(b.x, m.x))
            iy = max(0.0, min(b.y2, m.y2) - max(b.y, m.y))
            inter = ix * iy
            if inter and inter / min(b.w * b.h, m.w * m.h) > 0.6:
                self.skipped_duplicates += 1
                return True
        self.marked.append(b)
        return False

    def _mark(self, a: dict, kind):
        b = self.scene.resolve(a.get("target"), a.get("phrase"))
        if b is None or self._already_marked(b):
            return []
        name = self._ink_name(a)
        ink = self.scene.ink(name, b.pad(10))
        self.scene.reserve(b.pad(10), 0.6)
        return [kind(b, ink)]

    def _op_circle(self, a):
        return self._mark(a, Circle)

    def _op_underline(self, a):
        return self._mark(a, Underline)

    def _op_box(self, a):
        return self._mark(a, Frame)

    def _op_highlight(self, a):
        b = self.scene.resolve(a.get("target"), a.get("phrase"))
        if b is None or self._already_marked(b):
            return []
        dark = self.scene.is_dark(b)
        return [Highlight(b, "#FACC15" if dark else "#FFE14D", 80 if dark else 110)]

    def _op_number(self, a):
        b = self.scene.resolve(a.get("target"), a.get("phrase"))
        if b is None:
            return []
        ink = self.scene.ink(self._ink_name(a), b)
        c = QPointF(b.x - 16, b.cy) if b.x > 20 else QPointF(b.x + 12, b.y - 12)
        self.scene.reserve(Box(c.x() - 13, c.y() - 13, 26, 26), 1.0)
        return [Badge(c, str(a.get("n", "?"))[:2], ink, self.badge_font)]

    def _op_arrow(self, a):
        s = self.scene.resolve(a.get("from"), a.get("from_phrase"))
        e = self.scene.resolve(a.get("to"), a.get("to_phrase"))
        if s is None or e is None:
            return []
        ink = self.scene.ink(self._ink_name(a), s.union(e))
        p0 = _edge_point(s, _center(e), 6)
        p1 = _edge_point(e, _center(s), 6)
        items: list[Item] = [Arrow(p0, p1, ink)]
        label = a.get("label")
        if isinstance(label, str) and label.strip():
            fm = QFontMetricsF(self.small_font)
            top, _, bottom = _metrics(fm)
            lw = fm.horizontalAdvance(label)
            mid = items[0].mid()
            box, _ = self.scene.place(lw + 8, top + bottom + 6, Box(mid.x() - 2, mid.y() - 2, 4, 4))
            items.append(Label(QPointF(box.x + 4, box.y + 3 + top), label.strip(), self.small_font,
                               ink, self._halo(box)))
        return items

    # -- notes -----------------------------------------------------------------

    def _note(self, text: str, target: Box | None, name: str, font: QFont) -> Note:
        fm = QFontMetricsF(font)
        top, line_h, bottom = _metrics(fm)
        lines = _wrap(text, fm, self.max_note_w)
        pad = 8.0
        w = max(fm.horizontalAdvance(s) for s in lines) + 2 * pad
        h = top + line_h * (len(lines) - 1) + bottom + 2 * pad
        box, covers = self.scene.place(w, h, target)
        card = self._card_for(box) if covers else None
        ink = self._ink_on_card(name, card) if card else self.scene.ink(name, box)
        return Note(box, lines, font, line_h, top, ink, card, self._halo(box), self._leader(box, target), pad)

    def _leader(self, box: Box, target: Box | None):
        if target is None:
            return None
        gap = max(0.0, max(target.x - box.x2, box.x - target.x2), max(target.y - box.y2, box.y - target.y2))
        if gap <= 14:
            return None
        _, p, q = self.scene.leader(box, target)
        return QPointF(*p), QPointF(*q)

    def _op_note(self, a):
        text = str(a.get("text", "")).strip()
        if not text:
            return []
        target = self.scene.resolve(a.get("target"), a.get("phrase")) if a.get("target") or a.get("phrase") else None
        return [self._note(text, target, self._ink_name(a), self.note_font)]

    def _op_summary(self, a):
        text = str(a.get("text", "")).strip()
        if not text:
            return []
        return [self._note("★ " + text, None, a.get("color") or "green", _font(config.NOTE_FONT_PX + 1, True))]

    # -- diagrams --------------------------------------------------------------

    def _op_diagram(self, a):
        nodes = [str(n) for n in (a.get("nodes") or [])][:6]
        if not nodes:
            return []
        edges = [e for e in (a.get("edges") or []) if isinstance(e, list) and len(e) >= 2]
        if not edges:
            edges = [[i, i + 1, ""] for i in range(len(nodes) - 1)]
        title = str(a.get("title") or "").strip()
        name = self._ink_name(a)
        fm = QFontMetricsF(self.node_font)
        tfm = QFontMetricsF(self.note_font)
        n_top, n_line, n_bottom = _metrics(fm)
        t_top, t_line, t_bottom = _metrics(tfm)
        np_ = 8.0
        sizes = []
        for n in nodes:
            ls = _wrap(n, fm, 130)
            sizes.append((ls, max(fm.horizontalAdvance(s) for s in ls) + 2 * np_,
                          n_top + n_line * (len(ls) - 1) + n_bottom + 2 * np_))
        gap = 38.0
        sf = QFontMetricsF(self.small_font)
        # Between side-by-side nodes, leave room for the label on the arrow joining them.
        gaps = [gap] * max(0, len(sizes) - 1)
        for e in edges:
            try:
                i, j = int(e[0]), int(e[1])
            except (TypeError, ValueError, IndexError):
                continue
            if abs(i - j) == 1 and min(i, j) < len(gaps) and len(e) > 2 and e[2]:
                gaps[min(i, j)] = max(gaps[min(i, j)], sf.horizontalAdvance(str(e[2]).strip()) + 16)
        max_w = min(560.0, self.scene.w * 0.5)
        horiz = sum(s[1] for s in sizes) + sum(gaps) <= max_w
        has_back = any(abs(int(e[1]) - int(e[0])) != 1 for e in edges if str(e[0]).isdigit() or isinstance(e[0], int))
        pad = 12.0
        title_h = t_top + t_bottom + 10 if title else 0
        if horiz:
            inner_w = sum(s[1] for s in sizes) + sum(gaps)
            inner_h = max(s[2] for s in sizes) + (34 if has_back else 0)
        else:
            inner_w = max(s[1] for s in sizes) + (40 if has_back else 0) + \
                max([sf.horizontalAdvance(str(e[2])) for e in edges if len(e) > 2 and e[2]] + [0])
            inner_h = sum(s[2] for s in sizes) + gap * (len(sizes) - 1)
        w = max(inner_w, tfm.horizontalAdvance(title) if title else 0) + 2 * pad
        h = inner_h + title_h + 2 * pad

        target = self.scene.resolve(a.get("target"), a.get("phrase")) if a.get("target") else None
        box, covers = self.scene.place(w, h, target)
        card = self._card_for(box) if covers else None
        ink = self._ink_on_card(name, card) if card else self.scene.ink(name, box)
        fill = card or None
        halo = None if card else self._halo(box)

        parts: list[Item] = []
        if title:
            parts.append(Label(QPointF(box.x + pad, box.y + pad + t_top), title, self.note_font, ink, halo or card))
        rects: list[Box] = []
        x, y = box.x + pad, box.y + pad + title_h
        row_h = max(s[2] for s in sizes)
        for k, (ls, nw, nh) in enumerate(sizes):
            if horiz:
                rects.append(Box(x, y + (row_h - nh) / 2, nw, nh))
                x += nw + (gaps[k] if k < len(gaps) else gap)
            else:
                rects.append(Box(x, y, nw, nh))
                y += nh + gap
        node_items: list[Item] = []  # each node's box, then its text, node by node
        for (ls, nw, nh), r in zip(sizes, rects):
            node_items.append(RoundBox(r, ink, fill))
            for k, s in enumerate(ls):
                node_items.append(Label(QPointF(r.x + np_, r.y + np_ + n_top + k * n_line), s,
                                        self.node_font, ink, halo or card))
        edge_items = []
        for e in edges:
            try:
                i, j = int(e[0]), int(e[1])
            except (TypeError, ValueError):
                continue
            if not (0 <= i < len(rects) and 0 <= j < len(rects)) or i == j:
                continue
            ri, rj = rects[i], rects[j]
            if abs(i - j) == 1:
                p0, p1 = _edge_point(ri, _center(rj), 3), _edge_point(rj, _center(ri), 3)
                bend = 0.0
            else:  # loop back around the row/column, bulging a fixed ~24px outward
                if horiz:
                    p0, p1 = QPointF(ri.cx, ri.y2 + 3), QPointF(rj.cx, rj.y2 + 3)
                else:
                    p0, p1 = QPointF(ri.x2 + 3, ri.cy), QPointF(rj.x2 + 3, rj.cy)
                length = max(1.0, math.dist((p0.x(), p0.y()), (p1.x(), p1.y())))
                bend = 2 * 24 / length
                bend = -bend if (j < i) == horiz else bend
            arrow = Arrow(p0, p1, ink, bend, 2.4)
            edge_items.append(arrow)
            lab = str(e[2]).strip() if len(e) > 2 and e[2] else ""
            if lab:
                sf = QFontMetricsF(self.small_font)
                m = arrow.mid()
                if bend == 0:
                    pos = QPointF(m.x() - sf.horizontalAdvance(lab) / 2, m.y() - 6)
                elif horiz:
                    pos = QPointF(m.x() - sf.horizontalAdvance(lab) / 2, m.y() + sf.capHeight() + 4)
                else:
                    pos = QPointF(m.x() + 5, m.y() + sf.capHeight() / 2)
                edge_items.append(Label(pos, lab, self.small_font, ink, halo or card))
        # nodes first, then the arrows between them, so the idea builds up step by step
        parts += node_items + edge_items
        leader_items: list[Item] = []
        lead = self._leader(box, target)
        if lead is not None:
            leader_items.append(Arrow(lead[0], lead[1], ink, 0.12, 2.0))
        return [Group(leader_items + parts, box, card, ink)]
