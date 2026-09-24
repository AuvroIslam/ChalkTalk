"""Live walkthroughs on content the prompt has never seen, graded against the right answer.

    python tests/eval_walkthroughs.py [out_dir] [case ...]

Every case is a freshly drawn picture (graphs of different shapes, letter and number
nodes, an array, an equation). The real model teaches it step by step, and the
result is checked in code: the distances it writes against Dijkstra, the order it
visits against BFS levels, and the final answer. Each case also saves a picture
of what was drawn.
"""
from __future__ import annotations

import heapq
import math
import re
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PySide6.QtCore import QRectF  # noqa: E402  (Qt before winrt)
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import planner  # noqa: E402
from compose import Composer  # noqa: E402
from layout import Scene  # noqa: E402
from ocr import ocr_lines  # noqa: E402


def draw_graph(nodes, edges, title, weighted=True, dark=False):
    bg, fg = ((30, 30, 36), (235, 235, 235)) if dark else ("white", "black")
    img = Image.new("RGB", (1920, 1080), bg)
    d = ImageDraw.Draw(img)
    d.text((80, 50), title, font=ImageFont.truetype("arialbd.ttf", 44), fill=fg)
    fn, fw = ImageFont.truetype("arialbd.ttf", 40), ImageFont.truetype("arial.ttf", 36)
    for u, v, w in edges:
        (x1, y1), (x2, y2) = nodes[u], nodes[v]
        d.line([(x1, y1), (x2, y2)], fill=fg, width=3)
        if weighted:  # the weight sits beside the middle of the edge
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            nx, ny = -(y2 - y1), x2 - x1
            k = 34 / math.hypot(nx, ny)
            d.text((mx + nx * k - 10, my + ny * k - 20), str(w), font=fw, fill=fg)
    for n, (x, y) in nodes.items():
        d.ellipse((x - 45, y - 45, x + 45, y + 45), outline=fg, width=3, fill=bg)
        tw = d.textlength(n, font=fn)
        d.text((x - tw / 2, y - 23), n, font=fn, fill=fg)
    return img


def text_frame(lines, size=64):
    img = Image.new("RGB", (1920, 1080), "white")
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((160, 160 + i * int(size * 1.8)), t, font=ImageFont.truetype("cambria.ttc", size), fill="black")
    return img


def dijkstra(edges, src):
    adj = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    dist, pq = {src: 0}, [(0, src)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u]:
            continue
        for v, w in adj[u]:
            if du + w < dist.get(v, 1e9):
                dist[v] = du + w
                heapq.heappush(pq, (dist[v], v))
    return dist, adj


def bfs_levels(edges, src):
    adj = {}
    for u, v, _ in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    lvl, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in lvl:
                lvl[v] = lvl[u] + 1
                q.append(v)
    return lvl, adj


def num(s):
    s = str(s)
    if "∞" in s or re.search(r"(?i)\binf", s):
        return math.inf
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def path_in(text, names):
    """A → B → C (or A-B-C, A, B, C) in the text, as a list of node names."""
    pat = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    best = []
    for m in re.finditer(rf"(?:\b(?:{pat})\b\s*(?:→|->|—|–|-|,|to)\s*)+\b(?:{pat})\b", text):
        seq = re.findall(rf"\b(?:{pat})\b", m.group(0))
        if len(seq) > len(best):
            best = seq
    return best


def check_dijkstra(ctx, src, dst):
    dist, adj = dijkstra(ctx["edges"], src)
    names = set(ctx["nodes"])
    problems = []
    final = {n: vals[-1] for n, vals in ctx["tags"].items() if vals}
    if final.get(src) != 0 and 0 not in ctx["tags"].get(src, []):
        problems.append(f"start {src} not tagged 0")
    settled = set(ctx.get("circled", []))
    for n, v in final.items():
        # A node settled (circled) must show its true distance. One still waiting when the search
        # stops at the target may keep a temporary value, but never one below the true distance.
        if v is None or v == math.inf:
            continue
        if v < dist[n] or n in settled and v != dist[n]:
            problems.append(f"{n}: wrote {v:g}, true distance {dist[n]}")
    if final.get(dst) != dist[dst]:
        problems.append(f"target {dst} ends at {final.get(dst)}, true {dist[dst]}")
    path = path_in(ctx["summary"], names)
    if not path or path[0] != src or path[-1] != dst:
        problems.append(f"no {src}→{dst} path in summary: {ctx['summary'][:80]!r}")
    else:
        w = {(u, v): c for u in adj for v, c in adj[u]}
        cost = sum(w.get((a, b), math.inf) for a, b in zip(path, path[1:]))
        if cost != dist[dst]:
            problems.append(f"summary path {'→'.join(path)} costs {cost}, shortest is {dist[dst]}")
    if not re.search(rf"\b{dist[dst]}\b", ctx["summary"]):
        problems.append(f"summary doesn't give the cost {dist[dst]}")
    return problems


def check_bfs(ctx, src):
    lvl, _ = bfs_levels(ctx["edges"], src)
    order = ctx["circled"]
    problems = []
    if set(order) != set(lvl):
        problems.append(f"visited {order}, missing {sorted(set(lvl) - set(order))}")
    if order and order[0] != src:
        problems.append(f"doesn't start at {src}")
    if any(lvl[a] > lvl[b] for a, b in zip(order, order[1:]) if a in lvl and b in lvl):
        problems.append(f"order {order} isn't level by level ({ {n: lvl[n] for n in order} })")
    return problems


def check_answer(pattern):
    def f(ctx):
        return [] if re.search(pattern, ctx["all_text"]) else [f"final answer {pattern!r} not found"]
    return f


G1 = dict(nodes={"P": (260, 560), "Q": (720, 300), "R": (720, 820), "S": (1200, 300), "T": (1200, 820), "U": (1650, 560)},
          edges=[("P", "Q", 4), ("P", "R", 2), ("Q", "R", 1), ("Q", "S", 5), ("R", "T", 8), ("R", "S", 9), ("S", "T", 2),
                 ("S", "U", 6), ("T", "U", 3)])
G2 = dict(nodes={"A": (240, 300), "B": (700, 200), "C": (1150, 260), "D": (1640, 420), "E": (260, 800), "F": (760, 620),
                 "G": (1240, 820)},
          edges=[("A", "B", 7), ("A", "E", 3), ("B", "C", 2), ("E", "F", 2), ("F", "B", 1), ("F", "G", 6), ("C", "D", 4),
                 ("G", "D", 1), ("C", "G", 3)])
G3 = dict(nodes={"1": (960, 220), "2": (560, 480), "3": (1360, 480), "4": (330, 820), "5": (780, 820), "6": (1180, 820),
                 "7": (1600, 820)},
          edges=[("1", "2", 0), ("1", "3", 0), ("2", "4", 0), ("2", "5", 0), ("3", "6", 0), ("3", "7", 0), ("5", "6", 0)])

CASES = {
    "dijkstra_PU": (lambda: draw_graph(G1["nodes"], G1["edges"], "Shortest paths"), G1,
                    "walk me through dijkstra from P to U, step by step", lambda c: check_dijkstra(c, "P", "U")),
    "dijkstra_AD_dark": (lambda: draw_graph(G2["nodes"], G2["edges"], "Weighted graph", dark=True), G2,
                         "show me how to get from A to D with dijkstra's algorithm", lambda c: check_dijkstra(c, "A", "D")),
    "bfs_numbers": (lambda: draw_graph(G3["nodes"], G3["edges"], "Breadth-first search", weighted=False), G3,
                    "run BFS from node 1 on this graph step by step", lambda c: check_bfs(c, "1")),
    "bubble_sort": (lambda: text_frame(["Bubble sort", "", "[ 5,  2,  8,  1,  9 ]"]), None,
                    "walk me through bubble sort on this array, step by step", check_answer(r"1,\s*2,\s*5,\s*8,\s*9")),
    "equation": (lambda: text_frame(["Solve for x:", "", "3x − 7 = 2x + 5"]), None,
                 "solve this step by step", check_answer(r"x\s*=\s*12\b")),
}


def render(img, scene, items, path):
    out = QImage(img.width, img.height, QImage.Format_ARGB32)
    qimg = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.drawImage(QRectF(0, 0, scene.w, scene.h), qimg)
    for it in items:
        it.paint(p, 1.0)
    p.end()
    out.save(str(path))


def run(name, out_dir):
    make, graph, question, check = CASES[name]
    img = make()
    scene = Scene(img, ocr_lines(img), 1.0)
    node_of = {l.id: l.text for l in scene.lines if graph and l.text in graph["nodes"]}
    comp = Composer(scene)
    ctx = dict(nodes=graph["nodes"] if graph else {}, edges=graph["edges"] if graph else [],
               tags={}, circled=[], summary="", all_text="")
    items, actions, landed = [], [], 0
    t0 = time.perf_counter()

    def on_action(a):
        nonlocal landed
        actions.append(a)
        got = comp.build(a)
        items.extend(got)
        landed += bool(got)
        ctx["all_text"] += " " + str(a.get("text", "")) + " " + str(a.get("say", ""))
        targets = a.get("target") if isinstance(a.get("target"), list) else [a.get("target")]
        for t in targets:
            n = node_of.get(str(t).upper()) if t else None
            if n and a.get("op") == "tag":
                ctx["tags"].setdefault(n, []).append(num(a.get("text")))
            if n and a.get("op") == "circle" and n not in ctx["circled"]:
                ctx["circled"].append(n)
        if a.get("op") == "summary":
            ctx["summary"] += " " + str(a.get("text", ""))

    session = planner.Session(scene, img, None, None)
    err = ""
    try:
        session.ask(question, on_action)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    took = time.perf_counter() - t0
    if graph:
        seen = set(node_of.values())
        missing = set(graph["nodes"]) - seen
    problems = ([err] if err else []) + ([f"OCR didn't find nodes {sorted(missing)}"] if graph and missing else [])
    problems += check(ctx)
    missed = len(actions) - landed - comp.skipped_duplicates
    if missed > 0:
        problems.append(f"{missed}/{len(actions)} steps drew nothing")
    unspoken = sum(1 for a in actions if not a.get("say"))
    render(img, scene, items, out_dir / f"walk_{name}.png")
    print(f"{'PASS' if not problems else 'FAIL'} {name:<18} {took:5.1f}s  steps={len(actions)} "
          f"spoken={len(actions) - unspoken}  tags={ {k: [('∞' if v == math.inf else v) for v in vs] for k, vs in ctx['tags'].items()} }")
    if ctx["summary"]:
        print(f"      summary: {ctx['summary'].strip()[:140]}")
    for p in problems:
        print(f"      - {p}")
    return not problems


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("walk_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:] or list(CASES)
    planner.warm_up()
    ok = sum(run(n, out_dir) for n in names)
    print(f"\n{ok}/{len(names)} walkthroughs correct")


if __name__ == "__main__":
    main()
