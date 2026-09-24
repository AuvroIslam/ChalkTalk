"""Scenario test against the real model (Microsoft Foundry deployment from .env).

    python tests/eval_model.py            # all scenarios
    python tests/eval_model.py 3 5        # only some

For each scenario it checks: did the model fetch more context exactly when it
should, did its drawings land on real things on screen, did it cite fetched
sources, did it answer in the user's language, and how fast was it.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PySide6.QtWidgets import QApplication  # noqa: E402  (Qt before winrt)

app = QApplication.instance() or QApplication([])

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import config  # noqa: E402
import context  # noqa: E402
import planner  # noqa: E402
from compose import Composer  # noqa: E402
from layout import Scene  # noqa: E402
from ocr import Box, ocr_lines  # noqa: E402
from render_test import sample_slide  # noqa: E402

VIDEO = "aircAruvnKk"
RULES_PDF = Path(__file__).resolve().parents[2] / "EN-WBNR-SlideDeck-SREVM92894-pdf.pdf"


def frame(lines, bg=(20, 20, 20), fg=(240, 240, 240), size=40, font="cambria.ttc"):
    img = Image.new("RGB", (1920, 1080), bg)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((160, 160 + i * int(size * 1.7)), t, font=ImageFont.truetype(font, size), fill=fg)
    return img


def youtube():
    yt = context.YouTubeSource(VIDEO, f"https://www.youtube.com/watch?v={VIDEO}",
                               "But what is a neural network? | Deep learning chapter 1 - YouTube - Google Chrome")
    yt.wait(40)
    return yt


def rules_pdf():
    d = context.DocumentSource(str(RULES_PDF))
    d.wait(30)
    return d


def wiki():
    return context.WebPageSource("https://en.wikipedia.org/wiki/Gradient_descent", "Gradient descent - Wikipedia - Google Chrome")


# (name, image, source factory, question, expect_tools, extra checks, selection)
def scenarios():
    slide = sample_slide()
    pdf_first = [l[:75] for l in rules_pdf().pages[0].splitlines() if l.strip()][:10]
    pdf_dates = next(i for i, p in enumerate(rules_pdf().pages) if re.search(r"deadline|23:59", p, re.I))
    pdf_date_lines = [l[:75] for l in rules_pdf().pages[pdf_dates].splitlines() if l.strip()][:12]
    return [
        ("slide: symbol meaning", slide, None, "What does the eta symbol mean here?", False, {}, None),
        ("slide: vague 'explain'", slide, None, "", False, {}, None),
        ("slide: Bangla question", slide, None, "লার্নিং রেট বেশি হলে কী হয়?", False, {"bangla": True}, None),
        ("slide: selected chart only", slide, None, "What does this graph show?", False, {"in_selection": True},
         Box(583, 108, 320, 300)),
        ("video: earlier in the video", frame(["σ(w₁a₁ + w₂a₂ + ⋯ + wₙaₙ − 10)"], size=72),
         youtube, "Why is there a minus 10 here? What did he say about it?", True, {"cite": "video"}, None),
        ("video: 'what did he just say' (clock visible)", frame(["Neural network", "", "", "", "", "", "", "",
                                                               "▶  10:45 / 18:40"], size=44),
         youtube, "What did he just say? Summarise the last minute.", False, {}, None),
        ("pdf: answer on another page", frame(pdf_first, bg=(255, 255, 255), fg=(20, 20, 20), size=32, font="segoeui.ttf"),
         rules_pdf, "How are entries judged: what criteria and how many points?", True,
         {"cite": "p", "mentions": r"30|90"}, None),
        ("pdf: answer on this page", frame(pdf_date_lines, bg=(255, 255, 255), fg=(20, 20, 20), size=30, font="segoeui.ttf"),
         rules_pdf, "What does this part say about dates?", False, {}, None),
        ("web: content further down the page", frame(["Gradient descent", "From Wikipedia, the free encyclopedia",
                                                      "Gradient descent is a method for unconstrained mathematical optimization."],
                                                     bg=(255, 255, 255), fg=(20, 20, 20), size=36, font="segoeui.ttf"),
         wiki, "What variants or extensions of this method does the article list further down?", True,
         {"mentions": r"(?i)momentum|nesterov|stochastic|adam"}, None),
        ("slide: needs fresh facts", slide, None, "Which optimizer do PyTorch's official docs recommend as the default today?",
         True, {}, None),
        # Not in the document: it must say so first (it may then look it up elsewhere, with a source).
        ("pdf: answer isn't in the document", frame(pdf_first, bg=(255, 255, 255), fg=(20, 20, 20), size=32, font="segoeui.ttf"),
         rules_pdf, "Which team won this contest last year?", None, {"admits": ADMITS, "or_cited": True}, None),
        # Truly unknowable from any source: it must say it can't know, not invent an answer.
        ("slide: unknowable", slide, None, "What did my lecturer say about this slide in class yesterday?", None,
         {"admits": ADMITS}, None),
    ]


ADMITS = (r"(?i)not (in|on|mentioned|stated|listed|given|found|shown|say|specified|available|recorded)"
          r"|doesn'?t (say|mention|list|name|include|show)|no (information|mention|record|way)|don'?t (have|know)"
          r"|can'?t (see|know|tell|find|determine|access|hear|be determined)|cannot|unknown|isn'?t (in|listed|stated|shown)")


def run(idx, name, img, make_src, question, expect_tools, checks, selection):
    src = make_src() if make_src else None
    scene = Scene(img, ocr_lines(img), 0.5)
    session = planner.Session(scene, img, selection, src)
    comp = Composer(scene)
    actions, drawn, first = [], 0, None
    t0 = time.perf_counter()

    def on_action(a):
        nonlocal drawn, first
        first = first or time.perf_counter() - t0
        actions.append(a)
        items = comp.build(a)
        drawn += bool(items)
        if checks.get("in_selection") and items and a.get("op") in ("circle", "underline", "highlight", "box"):
            b = scene.resolve(a.get("target"), a.get("phrase"))
            checks.setdefault("_outside", 0)
            if b and not (selection.x - 5 <= b.cx <= selection.x2 + 5 and selection.y - 5 <= b.cy <= selection.y2 + 5):
                checks["_outside"] += 1

    tools = []
    orig = session._run_tools

    def spy(calls):
        tools.extend(c["name"] for c in calls)
        orig(calls)

    session._run_tools = spy
    err = ""
    try:
        session.ask(question, on_action)
    except Exception as e:  # report, keep going
        err = f"{type(e).__name__}: {e}"
    total = time.perf_counter() - t0
    texts = " ".join(str(a.get("text", "")) for a in actions)
    problems = []
    if err:
        problems.append(err[:120])
    if expect_tools is not None and bool(tools) != expect_tools:
        problems.append(f"expected tools={expect_tools}, used {tools or 'none'}")
    cited = re.search(r"\((?:[\w-]+\.)+[a-z]{2,}\)|source:|\(p\. ?\d|\(video \d", texts, re.I)
    if checks.get("admits") and not re.search(checks["admits"], texts) and not (checks.get("or_cited") and cited):
        problems.append("guessed instead of saying it can't be determined (or citing where it found it)")
    missed = len(actions) - drawn - comp.skipped_duplicates  # duplicates are skipped on purpose
    if actions and missed > 0:
        problems.append(f"{missed}/{len(actions)} actions didn't land on anything")
    if not actions:
        problems.append("no drawing at all")
    if checks.get("bangla") and not re.search(r"[ঀ-৿]", texts):
        problems.append("not answered in Bangla")
    if checks.get("cite") == "video" and not re.search(r"\d+:\d\d", texts):
        problems.append("no video timestamp cited")
    if checks.get("cite") == "p" and not re.search(r"\bp(age)?\.?\s*\d", texts, re.I):
        problems.append("no page cited")
    if checks.get("mentions") and not re.search(checks["mentions"], texts):
        problems.append(f"answer doesn't contain the facts ({checks['mentions']})")
    if not checks.get("admits") and \
            re.search(r"(?i)\b(look for|see the (section|page)|open the|scroll down|refer to|check the)\b", texts):
        problems.append("sends the user elsewhere instead of answering")
    if checks.get("_outside"):
        problems.append(f"{checks['_outside']} marks outside the selected area")
    ok = not problems
    print(f"{'PASS' if ok else 'FAIL'} {idx:>2}. {name:<44} tools={','.join(tools) or '-':<48} "
          f"first={first or -1:4.1f}s total={total:4.1f}s actions={len(actions)} landed={drawn}")
    for p in problems:
        print(f"        - {p}")
    notes = [a.get("text") for a in actions if a.get("op") in ("note", "summary")][:3]
    for n in notes:
        print(f"        note: {n}")
    return ok, first, total


def main():
    wanted = {int(a) for a in sys.argv[1:]}
    planner.warm_up()
    print(f"model={config.MODEL} reasoning={config.REASONING or '-'} endpoint={planner.make_client().base_url}\n")
    results = []
    for i, sc in enumerate(scenarios(), 1):
        if not wanted or i in wanted:
            results.append(run(i, *sc))
    passed = sum(r[0] for r in results)
    firsts = sorted(r[1] for r in results if r[1])
    print(f"\n{passed}/{len(results)} scenarios passed; first drawing median {firsts[len(firsts) // 2]:.1f}s, "
          f"max {firsts[-1]:.1f}s")


if __name__ == "__main__":
    main()
