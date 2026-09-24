"""Geometry of the captured screen: where things are, where empty space is,
and which ink colours stay readable.

Everything here is in *logical* overlay coordinates (the units Qt draws in), so
a 300%-scaled laptop screen and a 100% monitor behave the same.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ocr import Box, Line, Word

CELL = 6  # logical px per occupancy-grid cell


@dataclass
class Region:
    id: str
    box: Box


def _scale_line(line: Line, s: float) -> Line:
    return Line(line.id, line.text, line.box.scaled(s), [Word(w.text, w.box.scaled(s)) for w in line.words], line.kind)


def _reading_order(lines: list[Line]) -> list[Line]:
    """Sort top-to-bottom, left-to-right, and renumber L1..Ln."""
    lines = sorted(lines, key=lambda l: l.box.cy)
    rows: list[list[Line]] = []
    for ln in lines:
        if rows and abs(rows[-1][0].box.cy - ln.box.cy) < 0.5 * min(rows[-1][0].box.h, ln.box.h):
            rows[-1].append(ln)
        else:
            rows.append([ln])
    ordered = [ln for row in rows for ln in sorted(row, key=lambda l: l.box.x)]
    for i, ln in enumerate(ordered):
        ln.id = f"L{i + 1}"
    return ordered


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


# ---- colour -----------------------------------------------------------------

INKS = {  # (for light backgrounds, for dark backgrounds)
    "red": ("#D62828", "#FF7A7A"),
    "blue": ("#1D4ED8", "#7CB8FF"),
    "green": ("#137A3A", "#5BE08A"),
    "purple": ("#7C3AED", "#C9A7FF"),
    "orange": ("#C2410C", "#FFB067"),
}


def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def luminance(rgb) -> float:
    def ch(c: float) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a, b) -> float:
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


class Scene:
    def __init__(self, image: Image.Image, lines: list[Line], scale: float):
        """image: physical-pixel screenshot. lines: OCR result in physical px.
        scale: logical px per physical px (1/devicePixelRatio)."""
        self.image = image
        self.scale = scale
        self.w = image.width * scale
        self.h = image.height * scale
        self.lines = _reading_order([_scale_line(l, scale) for l in lines])
        self.line_by_id = {l.id: l for l in self.lines}
        self.gw = math.ceil(self.w / CELL)
        self.gh = math.ceil(self.h / CELL)

        gray = np.asarray(image.convert("L"))
        edges = cv2.Canny(gray, 40, 120)
        density = cv2.resize(edges.astype(np.float32) / 255, (self.gw, self.gh), interpolation=cv2.INTER_AREA)
        occ = (density > 0.015).astype(np.float32)
        for ln in self.lines:
            self._mark(occ, ln.box.pad(3), 1.0)
        self.occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
        self.reserved = np.zeros_like(self.occ)
        self.colors = np.asarray(image.convert("RGB").resize((self.gw, self.gh), Image.BOX)).astype(np.float32)

        self.regions = self._find_regions(edges)
        self.region_by_id = {r.id: r for r in self.regions}
        self.parts = self._find_parts(np.asarray(image.convert("RGB")))
        self.part_by_id = {p.id: p for p in self.parts}

        self.hidden: list[Box] = []  # covered by our own bar / the taskbar: never mark things there

        # Mapping between the image the model saw and logical coordinates.
        self.view = Box(0, 0, self.w, self.h)
        self.view_px = (self.w, self.h)

    # ---- grid helpers -----------------------------------------------------

    def _cells(self, b: Box):
        x0 = max(0, int(b.x // CELL))
        y0 = max(0, int(b.y // CELL))
        x1 = min(self.gw, math.ceil(b.x2 / CELL))
        y1 = min(self.gh, math.ceil(b.y2 / CELL))
        return y0, y1, x0, x1

    def _mark(self, grid: np.ndarray, b: Box, value: float) -> None:
        y0, y1, x0, x1 = self._cells(b)
        if y1 > y0 and x1 > x0:
            grid[y0:y1, x0:x1] = np.maximum(grid[y0:y1, x0:x1], value)

    def reserve(self, b: Box, weight: float = 1.0) -> None:
        """Mark space as used by a drawing so later notes avoid it."""
        self._mark(self.reserved, b, weight)

    def unreserve(self, b: Box) -> None:
        y0, y1, x0, x1 = self._cells(b)
        if y1 > y0 and x1 > x0:
            self.reserved[y0:y1, x0:x1] = 0

    def bg_color(self, b: Box) -> tuple[float, float, float]:
        y0, y1, x0, x1 = self._cells(b.pad(CELL))
        patch = self.colors[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            patch = self.colors.reshape(-1, 3)
        return tuple(np.median(patch, axis=0))  # type: ignore[return-value]

    def is_dark(self, b: Box) -> bool:
        return luminance(self.bg_color(b)) < 0.3

    def ink(self, name: str | None, b: Box) -> str:
        """An ink colour that contrasts with the background around b."""
        bg = self.bg_color(b)
        on_light, on_dark = INKS.get((name or "red").lower(), INKS["red"])
        first, second = (on_light, on_dark) if luminance(bg) > 0.3 else (on_dark, on_light)
        if contrast(hex_rgb(first), bg) >= 3 or contrast(hex_rgb(first), bg) >= contrast(hex_rgb(second), bg):
            return first
        return second

    # ---- regions (figures, images, charts) --------------------------------

    def _find_regions(self, edges: np.ndarray) -> list[Region]:
        mask = edges.copy()
        inv = 1 / self.scale
        for ln in self.lines:
            b = ln.box.pad(4).scaled(inv)
            mask[max(0, int(b.y)) : int(b.y2) + 1, max(0, int(b.x)) : int(b.x2) + 1] = 0
        k = max(3, int(14 * inv))
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = self.w * self.h
        found = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            b = Box(x, y, w, h).scaled(self.scale)
            if 0.01 * area <= b.w * b.h <= 0.8 * area and min(b.w, b.h) >= 40:
                found.append(b)
        found = sorted(found, key=lambda b: -b.w * b.h)[:10]
        found.sort(key=lambda b: (round(b.y / 40), b.x))
        return [Region(f"R{i + 1}", b) for i, b in enumerate(found)]

    def _find_parts(self, rgb: np.ndarray) -> list[Region]:
        """The separate marks inside a drawing (a whiteboard, chalk video, diagram):
        the pivot, the rod, the force arrow, a hand-written "F". OCR can't name these,
        so each gets an id the tutor can point at. Photos and thumbnails are skipped."""
        inv = 1 / self.scale
        found: list[Box] = []
        for r in self.regions:
            if min(r.box.w, r.box.h) < 150:
                continue  # toolbars and strips: their icons aren't a figure
            pb = r.box.scaled(inv)
            x0, y0, x1, y1 = int(pb.x), int(pb.y), int(pb.x2), int(pb.y2)
            patch = rgb[y0:y1, x0:x1].astype(np.int16)
            if patch.size == 0:
                continue
            bg = np.median(patch.reshape(-1, 3), axis=0)
            ink = np.abs(patch - bg).max(axis=2) > 70
            if ink.mean() > 0.15:
                continue  # busy all over: a photo or a thumbnail, not a drawing
            for ln in self.lines:  # text is already listed as lines
                b = ln.box.pad(3).scaled(inv)
                ink[max(0, int(b.y) - y0):max(0, int(b.y2) - y0 + 1), max(0, int(b.x) - x0):max(0, int(b.x2) - x0 + 1)] = False
            k = max(3, int(5 * inv))
            mask = cv2.dilate(ink.astype(np.uint8), np.ones((k, k), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            ra = (x1 - x0) * (y1 - y0)
            for i in range(1, n):
                x, y, w, h, _a = stats[i]
                b = Box(float(x + x0), float(y + y0), float(w), float(h)).scaled(self.scale)
                if max(b.w, b.h) < 18 or min(b.w, b.h) < 10 and max(b.w, b.h) < 45 or w * h > 0.5 * ra:
                    continue  # specks and dash fragments, or the frame itself
                k = int(min(6, round((b.w + b.h) / 120)))
                if k < 2:
                    found.append(b)
                    continue
                # One long connected stroke (a rod with its pivot and the force arrow) is
                # really several things: split it by position so each piece gets a name.
                ys, xs = np.nonzero((labels[y:y + h, x:x + w] == i) & ink[y:y + h, x:x + w])
                pts = np.column_stack([xs, ys]).astype(np.float32)[:: max(1, len(xs) // 4000)]
                if len(pts) < 4 * k:
                    found.append(b)
                    continue
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
                _, lab, _ = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
                for c in range(k):
                    q = pts[lab.ravel() == c]
                    if len(q):
                        qx, qy = q[:, 0].min(), q[:, 1].min()
                        found.append(Box(float(x + x0 + qx), float(y + y0 + qy), float(q[:, 0].max() - qx + 1),
                                         float(q[:, 1].max() - qy + 1)).scaled(self.scale).pad(3))
        found = sorted(found, key=lambda b: -b.w * b.h)[:16]
        found.sort(key=lambda b: (round(b.y / 40), b.x))
        return [Region(f"P{i + 1}", b) for i, b in enumerate(found)]

    # ---- the model's view -------------------------------------------------

    def set_view(self, view: Box, px_w: int, px_h: int) -> None:
        """The model sees `view` (logical) as an image of px_w x px_h pixels."""
        self.view = view
        self.view_px = (px_w, px_h)

    def to_model(self, b: Box) -> list[int]:
        sx = self.view_px[0] / self.view.w
        sy = self.view_px[1] / self.view.h
        return [round((b.x - self.view.x) * sx), round((b.y - self.view.y) * sy), round(b.w * sx), round(b.h * sy)]

    def from_model(self, xywh) -> Box:
        if not isinstance(xywh, (list, tuple)) or len(xywh) != 4:
            raise ValueError(f"box must be [x, y, w, h], got {xywh!r}")
        x, y, w, h = (float(v) for v in xywh)
        sx = self.view.w / self.view_px[0]
        sy = self.view.h / self.view_px[1]
        return Box(self.view.x + x * sx, self.view.y + y * sy, w * sx, h * sy)

    def visible_lines(self) -> list[Line]:
        v = self.view
        return [l for l in self.lines if l.box.cx >= v.x and l.box.cx <= v.x2 and l.box.cy >= v.y and l.box.cy <= v.y2]

    def visible_regions(self) -> list[Region]:
        v = self.view
        return [r for r in self.regions if r.box.cx >= v.x and r.box.cx <= v.x2 and r.box.cy >= v.y and r.box.cy <= v.y2]

    def visible_parts(self) -> list[Region]:
        v = self.view
        return [r for r in self.parts if r.box.cx >= v.x and r.box.cx <= v.x2 and r.box.cy >= v.y and r.box.cy <= v.y2]

    # ---- resolving what the model points at -------------------------------

    def is_hidden(self, b: Box) -> bool:
        return any(h.x <= b.cx <= h.x2 and h.y <= b.cy <= h.y2 for h in self.hidden)

    def _target_lines(self, target) -> tuple[list[Line], list[Box]]:
        if isinstance(target, list):
            lines, boxes = [], []
            for t in target:
                l2, b2 = self._target_lines(t)
                lines += l2
                boxes += b2
            return lines, boxes
        if isinstance(target, dict) and "box" in target:
            try:
                b = self.from_model(target["box"])
            except (TypeError, ValueError):
                return [], []  # malformed box from the model: ignore it rather than crash
            return [], [b] if b.w > 0 and b.h > 0 else []
        if isinstance(target, str):
            t = target.strip().upper()
            m = re.fullmatch(r"L(\d+)\s*[-–]\s*L?(\d+)", t)
            if m:
                a, b = sorted((int(m.group(1)), int(m.group(2))))
                lines = [self.line_by_id[f"L{i}"] for i in range(a, b + 1) if f"L{i}" in self.line_by_id]
                if len(lines) > 2:
                    # Ids run row by row across columns; keep the column(s) the range starts and ends in,
                    # so a box around a formula doesn't swallow the sidebar next to it.
                    ends = (lines[0].box, lines[-1].box)
                    lines = [l for l in lines if any(l.box.x < e.x2 and e.x < l.box.x2 for e in ends)]
                return lines, [l.box for l in lines]
            if t in self.line_by_id:
                return [self.line_by_id[t]], [self.line_by_id[t].box]
            if t in self.region_by_id:
                r = self.region_by_id[t].box
                inside = [l for l in self.lines if r.x <= l.box.cx <= r.x2 and r.y <= l.box.cy <= r.y2]
                return inside, [r]
            if t in self.part_by_id:
                return [], [self.part_by_id[t].box]
        return [], []

    def resolve(self, target=None, phrase: str | None = None) -> Box | None:
        lines, boxes = self._target_lines(target)
        if phrase:
            hit = self.find_phrase(phrase, lines) if lines else None
            if hit is None:
                # The model sometimes gets a line id wrong but the words right: search all text.
                # But when it points into a figure, the same words in a far-off title are a
                # different thing (the "why is it pointing at the title" bug): search nearby only.
                pool = self.visible_lines() or self.lines
                text_target = bool(lines) and not (isinstance(target, str) and target.strip().upper()[:1] in "RP")
                if boxes and not text_target:
                    near = boxes[0]
                    for b in boxes[1:]:
                        near = near.union(b)
                    m = max(40.0, 3 * max((l.box.h for l in lines), default=20.0))
                    pool = [l for l in pool if near.x - m <= l.box.cx <= near.x2 + m and near.y - m <= l.box.cy <= near.y2 + m]
                hit = self.find_phrase(phrase, pool)
            if hit is not None:
                return hit
        if not boxes:
            return None
        out = boxes[0]
        for b in boxes[1:]:
            out = out.union(b)
        return out

    def find_phrase(self, phrase: str, lines: list[Line]) -> Box | None:
        want = _norm(phrase)
        if not want:
            return None
        best_score, best = 0.0, None
        for ln in lines:
            norms = [_norm(w.text) for w in ln.words]
            exact_line = want in _norm(ln.text)
            for i in range(len(norms)):
                acc = ""
                for j in range(i, len(norms)):
                    acc += norms[j]
                    if len(acc) > len(want) * 1.6 + 3:
                        break
                    if exact_line and want in acc:
                        score = 1.0 + len(want) / max(len(acc), 1)  # exact, tightest span wins
                    elif abs(len(acc) - len(want)) <= max(3, len(want) // 2):
                        score = difflib.SequenceMatcher(None, acc, want).ratio()
                    else:
                        continue
                    if score > best_score:
                        best_score, best = score, (ln, i, j)
        if best is None or best_score < 0.72:
            return None
        ln, i, j = best
        out = ln.words[i].box
        for w in ln.words[i + 1 : j + 1]:
            out = out.union(w.box)
        return out

    # ---- finding space for notes -------------------------------------------

    def place(self, w: float, h: float, near: Box | None = None) -> tuple[Box, bool]:
        """Best spot for a w x h drawing: empty, on screen, close to `near`.
        Returns (box, covers_content)."""
        w = min(w, self.w - 2 * CELL)
        h = min(h, self.h - 2 * CELL)
        cw, ch = max(1, math.ceil(w / CELL)), max(1, math.ceil(h / CELL))
        # Covering content is OK-ish (the note gets a card); covering another drawing is not.
        cost = self.occ + self.reserved * 40
        if near is not None:
            cost = cost.copy()
            self._mark(cost, near.pad(4), 50.0)  # never cover what we're explaining
        ii = np.pad(cost, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = ii[ch:, cw:] - ii[:-ch, cw:] - ii[ch:, :-cw] + ii[:-ch, :-cw]

        ys = (np.arange(sums.shape[0]) * CELL)[:, None].astype(np.float32)
        xs = (np.arange(sums.shape[1]) * CELL)[None, :].astype(np.float32)
        score = sums * 50.0
        # keep one cell away from the screen edges
        edge = (ys < CELL) | (xs < CELL) | (ys + h > self.h - CELL) | (xs + w > self.w - CELL)
        score = score + edge * 1e7
        if near is not None:
            dx = np.maximum(0, np.maximum(near.x - (xs + w), xs - near.x2))
            dy = np.maximum(0, np.maximum(near.y - (ys + h), ys - near.y2))
            dist = np.hypot(dx, dy)
            score = score + dist + (dist < 8) * 30 + ((xs + w) <= near.x) * 25
        else:
            cx, cy = self.view.cx, self.view.y + self.view.h * 0.65
            score = score + 0.15 * np.hypot(xs + w / 2 - cx, ys + h / 2 - cy)

        if near is None:
            i, j = np.unravel_index(int(np.argmin(score)), score.shape)
        else:
            i, j = self._best_leader_spot(score, w, h, near)
        box = Box(float(j * CELL), float(i * CELL), w, h)
        covers = float(sums[i, j]) > 0.5
        self.reserve(box.pad(4), 1.0)
        return box, covers

    def _crossings(self, busy: np.ndarray, p: tuple[float, float], q: tuple[float, float]) -> int:
        n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / CELL))
        crossed = 0
        for f in np.linspace(0.1, 0.9, n):
            gx = int((p[0] + (q[0] - p[0]) * f) // CELL)
            gy = int((p[1] + (q[1] - p[1]) * f) // CELL)
            if 0 <= gy < self.gh and 0 <= gx < self.gw and busy[gy, gx]:
                crossed += 1
        return crossed

    def leader(self, note: Box, target: Box, busy: np.ndarray | None = None):
        """Cleanest pointer from a note to its target: tries attaching at the
        target's top, bottom, left and right. Returns (crossings, start, end)."""
        if busy is None:
            busy = (self.occ + self.reserved) > 0
        anchors = [(target.cx, target.y - 4), (target.cx, target.y2 + 4),
                   (target.x - 4, target.cy), (target.x2 + 4, target.cy)]
        best = None
        for qx, qy in anchors:
            px = min(max(qx, note.x), note.x2)
            py = min(max(qy, note.y), note.y2)
            c = self._crossings(busy, (px, py), (qx, qy))
            key = (c, math.hypot(qx - px, qy - py))
            if best is None or key < best[0]:
                best = (key, (px, py), (qx, qy))
        (c, _), p, q = best
        return c, p, q

    def _best_leader_spot(self, score: np.ndarray, w: float, h: float, near: Box) -> tuple[int, int]:
        """Among the best few spots, prefer one whose pointer line to the target
        doesn't cut through text or other drawings."""
        flat = score.ravel()
        k = min(150, flat.size)
        cand = np.argpartition(flat, k - 1)[:k]
        busy = (self.occ + self.reserved) > 0
        best, best_s = None, math.inf
        for idx in cand:
            i, j = divmod(int(idx), score.shape[1])
            crossed, _, _ = self.leader(Box(j * CELL, i * CELL, w, h), near, busy)
            s = float(flat[idx]) + crossed * 80
            if s < best_s:
                best, best_s = (i, j), s
        return best
