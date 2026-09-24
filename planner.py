"""Asks the model what to draw, and streams back drawing actions one by one.

Two paths:
- Enough context (screen + digest): the model starts drawing immediately, and
  each JSON line is drawn the moment it arrives.
- Not enough: the model first calls typed tools (transcript, other pages, the
  full web page, web search), we run them in parallel, and then it draws.
"""
from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Literal

import openai
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

import config
from clients import make_client
from context import Source, Toolbox
from layout import Scene
from ocr import Box

SYSTEM = """You are ChalkTalk, a patient tutor that explains what is on the user's screen by drawing on it, like a teacher with a marker on a whiteboard.

You get a screenshot, the text lines found on it by OCR (id, box [x,y,w,h] in screenshot pixels, text), the non-text regions (figures, images, charts), a CONTEXT digest about what is open (a video, a document, a web page), and the user's question.

STEP 1 - CONTEXT CHECK (silently, before anything else)
Decide whether the screen, the digest and solid general knowledge are enough to answer THIS question correctly and specifically.
- Enough: go straight to drawing. Don't call tools just to be thorough - speed matters.
- Not enough: call tools first (several at once when useful), then draw. Typical cases: the question refers to something said earlier or later in the video; a term, symbol or variable is defined on another page or slide; the visible part is cut off; the page continues below; recent, local or niche facts; you would otherwise be guessing.
- If the answer lives on a page, slide or time you haven't read, read it. Never send the user elsewhere ("see page 2") instead of answering.
- Write nothing before calling tools. Never invent what a video, document or page says.
- If you still can't tell after reading, say so in a note.

STEP 2 - DRAW. Reply ONLY with drawing actions: one JSON object per line (JSON Lines). No prose, no markdown, no code fences.

TARGETS - point at things by id, never by guessing pixels:
- "L12" a text line, "R2" a region, "L3-L7" several consecutive lines, ["L3","L9"] a set.
- A mark inside a hand-drawn sketch (an arrow, stroke, circle, symbol): when MARKS IN THE DRAWING is given, describe it, {"ink":"red","points":"right"} or {"ink":"white","shape":"round"}.
- "phrase": the exact words inside the target line to pinpoint, copied exactly from its text.
- Only if nothing listed covers it: {"box":[x,y,w,h]} in screenshot pixels.

ACTIONS (every one carries "say", see VOICE)
{"op":"circle","target":"L4","phrase":"learning rate","color":"red","say":"Look at the learning rate."}   ring a key term or a part of a figure
{"op":"underline","target":"L6","phrase":"...","say":"..."}
{"op":"highlight","target":"L8","say":"..."}   marker over words or lines
{"op":"box","target":"L3-L7","say":"..."}   group lines that belong together
{"op":"arrow","from":"L2","to":"R1","label":"feeds","say":"This feeds into that."}   only to join two separate things on screen that are directly related, with a label saying how (at most 2 per answer; from_phrase/to_phrase allowed). Never for emphasis, never into empty space, never from a note: notes already point at their target.
{"op":"number","target":"L5","n":1,"say":"..."}   step-order badge
Only when MARKS IN THE DRAWING appears below (a hand-drawn sketch fills the screen), target a mark in it by describing it:
{"op":"number","target":{"ink":"red","points":"right"},"n":1,"say":"Force one: the red 50 newton arrow, pushing right."}   the app puts the 1 at that arrow's end, by its label
{"op":"circle","target":{"ink":"white","shape":"round"},"say":"This circle is the pivot."}
{"op":"note","target":"L4","phrase":"learning rate","text":"How big each step is. Too big = you overshoot.","say":"This is how big each step is. Make it too big and you jump right past the answer."}   handwritten explanation; the app finds empty space beside the target and draws a pointer to it
{"op":"diagram","target":"R1","title":"Gradient descent","nodes":["Guess","Measure error","Step downhill"],"edges":[[0,1,""],[1,2,""],[2,0,"repeat"]],"say":"..."}   a small sketch (max 5 nodes) of something NOT already pictured on screen: a flow, cycle, cause->effect, comparison, analogy
{"op":"trace","from":"L5","to":"L9","color":"green","say":"..."}   a thick marker stroke along a connection, e.g. a graph edge or one step of a path
{"op":"tag","target":"L5","text":"d = 4","say":"..."}   a small value written right next to something; a new tag on the same target crosses out the old value. "target" may be a list to write the same value on several things in one step.
{"op":"summary","text":"...","say":"..."}   one-line takeaway, last

VOICE: you are speaking aloud while you draw, like a teacher at a whiteboard. Every action has "say": one or two short, natural spoken sentences. The written text is the short version; "say" is what you tell the student, in plain words (say "times", "equals", "perpendicular", not symbols). The next step waits until you finish speaking.
Colors: red, blue, green, purple, orange. Keep one colour per idea (a mark and its note share a colour).

HOW TO TEACH
- Answer the user's actual question. If they only point at an area, explain the most confusing idea in it.
- Mark first, then explain: circle or highlight the thing, then a note on it.
- Teach ON the picture. When the screen shows a figure, diagram, video frame, chart or formula, point at its actual parts (the pivot, the force arrow, the distance r, a curve, each symbol of the formula) and explain each one there. Point at a mark by describing it (see MARKS IN THE DRAWING) or by the text lines inside the figure; a {"box"} only if neither covers it. To number or label several things in a drawing (forces, parts), one "number" or "tag" per thing, each targeting that thing's own description, and include EVERY one: count their labels first (F1, F2 ... F7) so none is skipped. Only add a "diagram" if the picture you need isn't on screen.
- Never target a whole region (R ids) for one thing inside it: describe that thing instead.
- A mark or pointer must land on the thing it explains. A title, heading or caption that merely contains the same word is not that thing: never point at it instead of the figure.
- To explain a formula: mark each symbol, say what it stands for and how changing it changes the result, then give one everyday example.
- "What is happening here?" / "I'm confused": teach THIS example like a teacher at the board, in order, simply: 1) what the situation is (the object, what it's made of or weighs); 2) the things acting on it, one by one, with their real values read from the screen ("this 50 N pushes right"); 3) what is being worked out and why; 4) the result, using the actual numbers on screen. No general advice ("identify each force...") in place of the actual example. One idea per step, everyday words.
- Only name what you can actually read. Check colours, labels and numbers in the ZOOM image before saying what a mark is; if you can't tell what something is, don't mention it.
- Notes: at most 15 words, plain language, an analogy or a concrete example when possible. Never just restate the screen.
- Notes explain the idea itself. Never narrate what you are doing ("I'll open...", "Let me...").
- Never speculate about what you cannot see or know (what someone said or likely said, private details). Explain what is visible instead.
- Every note that uses fetched context ends its written "text" with its source in brackets: "(video 3:12)", "(p. 14)", "(slide 5)", "(Wikipedia)".
- Usually 3 to 8 actions; teaching a whole example to a confused learner up to 11, always ending with the summary (walkthroughs: as many steps as they need, max 30). Don't cover the whole screen.
- Use only ids from the lists. Don't place notes yourself.
- LANGUAGE: every note, tag, label, caption, summary and "say" is in the language of the user's question (a Bangla question gets Bangla notes and Bangla "say"), even when the screen is in English."""


# Added to the question only when the user asks to be walked through something.
WALKTHROUGH = """STEP-BY-STEP WALKTHROUGH MODE
- Actually perform it on what's on screen, in the real order, and get every number right. Never skip to the answer. Up to 30 actions, each one state change, each with "say" explaining why.
- Values that change go in "tag" next to the thing they belong to (a new tag crosses the old one out). Notes are for reasons, not values.
- Finish with a "summary" giving the result.
Recipes (use the one that fits; anything else: same idea, show each state change and say why):
- Graph shortest path (Dijkstra): circle the start and tag it "d=0"; ONE tag listing every other node as its target with text "∞". From the node just settled, relax EVERY edge to an unsettled neighbour: trace the edge, tag the neighbour if it improves (say the sum, "1 + 2 = 3, better than infinity"). Say which unsettled node is now smallest and why it goes next; circle it. Stop exploring the moment the target is circled. Then trace the final path edge by edge in red, and a summary with the path and cost.
- BFS / DFS: circle the start; tag each node with its level or visit number as it is discovered; trace the edge it was discovered through; circle a node when it is visited.
- In graphs, vertices are the lines marked (node); target them by those ids, never by an edge weight. Only trace between two nodes joined by an edge in the picture. Only circle nodes that are settled or visited.
- Sorting / arrays: for each comparison, circle or box the two items and say which is bigger; when they swap, tag both positions with their new values. Tag the state after each pass.
- Algebra / derivations / calculations: underline the part being changed; write each new line as a note next to the previous one, saying the rule used ("subtract 2x from both sides"). End with the answer in the summary.
- Physics / formula problems: mark each given quantity with its value, write the formula, substitute step by step, then the result with units.
Example (Dijkstra, start X, target Z; X-Y costs 2, Y-Z costs 3, X-Z costs 9):
{"op":"circle","target":"<X>","say":"We start at X."}
{"op":"tag","target":"<X>","text":"d=0","say":"Its distance is zero."}
{"op":"tag","target":["<Y>","<Z>"],"text":"∞","say":"Every other node starts at infinity, because we haven't reached it yet."}
{"op":"trace","from":"<X>","to":"<Y>","say":"From X, the edge to Y costs 2."}
{"op":"tag","target":"<Y>","text":"d=2","say":"That beats infinity, so Y becomes 2."}
{"op":"trace","from":"<X>","to":"<Z>","say":"The edge to Z costs 9."}
{"op":"tag","target":"<Z>","text":"d=9","say":"So for now Z is 9."}
{"op":"circle","target":"<Y>","say":"The smallest unsettled distance is Y, with 2, so we settle Y next."}
{"op":"trace","from":"<Y>","to":"<Z>","say":"Through Y, Z costs 2 plus 3."}
{"op":"tag","target":"<Z>","text":"d=5","say":"That's 5, which beats 9, so Z improves to 5."}
{"op":"circle","target":"<Z>","say":"Z is now the smallest, so it's settled. We've reached the target."}
{"op":"trace","from":"<X>","to":"<Y>","color":"red","say":"So the cheapest route is X, then Y,"}
{"op":"trace","from":"<Y>","to":"<Z>","color":"red","say":"then Z."}
{"op":"summary","text":"Shortest path X → Y → Z, cost 5","say":"The answer: X to Y to Z, with a total cost of 5."}"""

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="check")


def warm_up() -> None:
    """A tiny request at startup: opens the connection and wakes the deployment,
    so the first real question doesn't pay a cold start (seen at 10+ s)."""
    kwargs = dict(model=config.MODEL, messages=[{"role": "user", "content": "Reply with OK."}],
                  max_completion_tokens=64)
    if config.REASONING:
        kwargs["reasoning_effort"] = config.REASONING
    make_client().chat.completions.create(**kwargs)


def _encode(img) -> tuple[str, int, int]:
    s = min(1.0, config.MODEL_IMAGE_MAX / max(img.size))
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=82)
    return base64.standard_b64encode(buf.getvalue()).decode(), img.width, img.height


def _elements(scene: Scene, drawing: bool = True) -> str:
    rows = []
    for ln in scene.visible_lines()[:220]:
        text = ln.text if len(ln.text) <= 90 else ln.text[:87] + "..."
        kind = " (node)" if ln.kind == "node" else ""  # a label inside a drawn circle, e.g. a graph vertex
        rows.append(f"{ln.id} {scene.to_model(ln.box)} {json.dumps(text, ensure_ascii=False)}{kind}")
    regs = [f"{r.id} {scene.to_model(r.box)}" for r in scene.visible_regions()]
    out = "TEXT LINES\n" + ("\n".join(rows) or "(none)") + "\n\nREGIONS\n" + ("\n".join(regs) or "(none)")
    if drawing and scene.visible_parts():
        # No list of marks on purpose: given one, the model copies a wrong entry from it; asked to
        # describe what it sees, it is right nearly every time, and the app finds the mark.
        out += ("\n\nMARKS IN THE DRAWING: the app has found the hand-drawn marks (arrows, strokes, circles). "
                "To point at one, DESCRIBE it as you see it in the image and the app finds it: "
                "{\"ink\":\"purple\",\"points\":\"down\"} = the purple arrow pointing down; "
                "{\"ink\":\"white\",\"shape\":\"round\"} = a small white circle; {\"ink\":\"orange\",\"shape\":\"filled\"} "
                "= a filled blob. \"ink\" is the colour it is drawn in; \"points\" is where its free end (usually "
                "with its label) is, seen from where it starts: right, left, up, down, up-left, up-right, down-left, "
                "down-right.")
    return out


def _main_drawing(scene: Scene) -> Box | None:
    """The region holding most of the figure parts (e.g. the video frame with the sketch)."""
    parts = scene.visible_parts()
    best, n_best = None, 0
    for r in scene.visible_regions():
        n = sum(r.box.x <= p.box.cx <= r.box.x2 and r.box.y <= p.box.cy <= r.box.y2 for p in parts)
        if n > n_best or n == n_best and best is not None and r.box.w * r.box.h < best.w * best.h:
            best, n_best = r.box, n
    return best if n_best >= 2 else None


def _label_parts(crop, scene: Scene, view: Box):
    """Set-of-marks: write each figure part's id beside it on the model's copy of the screen."""
    from PIL import ImageDraw, ImageFont

    parts = scene.visible_parts()
    if not parts:
        return crop
    img = crop.copy()
    d = ImageDraw.Draw(img)
    s = 1 / scene.scale
    font = ImageFont.truetype("arialbd.ttf", max(12, round(15 * s)))
    for p in parts:
        # The id goes at the part's free end (an arrow's head, by its own label): those are spread
        # out, unlike box corners, which pile up where all the arrows meet.
        tx, ty = p.tip if p.tip is not None else (p.box.x, p.box.y)
        x, y = (tx - view.x) * s, (ty - view.y) * s
        tw = d.textlength(p.id, font=font)
        lx = min(max(0, x - tw / 2 - 3), img.width - tw - 6)
        ly = min(max(0, y - font.size / 2 - 2), img.height - font.size - 3)
        d.rectangle((lx, ly, lx + tw + 6, ly + font.size + 3), fill=(250, 204, 21), outline=(0, 0, 0))
        d.text((lx + 3, ly), p.id, font=font, fill=(0, 0, 0))
    return img


class ActionStream:
    """Parses JSON objects out of a growing text stream."""

    def __init__(self):
        self.buf = ""
        self.dec = json.JSONDecoder()

    def feed(self, chunk: str):
        self.buf += chunk
        while True:
            i = self.buf.find("{")
            if i < 0:
                self.buf = ""
                return
            try:
                obj, end = self.dec.raw_decode(self.buf, i)
            except json.JSONDecodeError:
                nxt = self.buf.find("\n{", i + 1)
                if nxt >= 0:
                    self.buf = self.buf[nxt + 1:]  # skip a malformed line
                    continue
                self.buf = self.buf[i:]
                return
            self.buf = self.buf[end:]
            if isinstance(obj, dict):
                yield obj


JUDGE = """You are the context checker for ChalkTalk, a tutor that explains what is on the user's screen.
Decide whether the tutor can answer the question correctly and specifically using ONLY the screen, the context digest and solid general knowledge.
The tutor SEES the screenshot: pictures, charts, graphs, diagrams, formulas and layout are all available even though you only get the OCR text. Never request lookups to understand something visible on screen.

NOT enough when:
- it asks what was said earlier or later in a video than the captions in the digest;
- it depends on a page, slide or section that is not on screen or in the digest (e.g. where a term is defined, what comes later, what the rest of the document says);
- it asks about current or recent facts: latest versions, what docs recommend today, prices, news, schedules;
- the answer would otherwise be a guess.
Otherwise it IS enough. Don't request lookups just to be thorough: speed matters.
Asking to explain, simplify, give an example of, or walk through what is visible is ALWAYS enough. So is "what is happening here?", "explain this", "I'm confused": "here" and "this" mean what is on screen right now, which the tutor can see.
Running or tracing something drawn on screen (a graph, an equation, code, a diagram, a table) is ALWAYS enough: the tutor reads the picture itself (which edge a weight belongs to, how things connect).

If not enough: "missing" says in a few words what is missing, and "lookups" lists at most 3 lookups using only the listed tools, with exact arguments (unused fields null). For read_webpage and every search, put what to look for in "query". If enough: "missing" is "" and "lookups" is [].

"cannot_know": if NO screen, document, video, web page or search could ever answer it (what someone said offline, private or personal records, grades, the future), one short honest sentence to the user, e.g. "I can't know what your lecturer said in class." Then lookups is []. Otherwise ""."""


def _check_model(tool_names: list[str]) -> type[BaseModel]:
    """The typed verdict. `tool` only accepts the tools available right now."""
    lookup = create_model(
        "Lookup", __config__=ConfigDict(extra="forbid"),
        tool=(Literal[tuple(tool_names)], ...),
        query=(str | None, ...), url=(str | None, ...),
        start_sec=(float | None, ...), end_sec=(float | None, ...),
        start_page=(int | None, ...), end_page=(int | None, ...),
    )
    return create_model("ContextCheck", __config__=ConfigDict(extra="forbid"),
                        enough=(bool, ...), missing=(str, ...), lookups=(list[lookup], ...),
                        cannot_know=(str, ...))


_NARRATION = re.compile(r"(?i)^\s*(i'?ll|i will|i'm going to|let me|let's (check|open|look)|checking|looking (at|for)|"
                        r"open(ing)? the|scroll|see the (section|page)|look for)\b")


_WALKTHROUGH = re.compile(r"(?i)step[- ]by[- ]step|walk (me )?through|\btrace\b|show (me )?how|\blive\b|simulat|dry[- ]?run|"
                          r"how (can|do|would) (i|we|you) (reach|get|go)|shortest path|run (the|this) algorithm|"
                          r"\bsolve\b|\bderive\b|work (it )?out|go through|demonstrat")

# A question about something beyond the screen (what was said, earlier or later, elsewhere).
_ELSEWHERE = re.compile(r"(?i)\b(said|say|says|saying|mention|mentioned|earlier|before|previous|later|next|"
                        r"rest of|other (page|slide|part)|transcript|who|when|source|latest|today|recommend)\b")

_SPECULATION = re.compile(r"(?i)\b(likely|probably|presumably|might have|may have|must have)\b[^.]{0,40}"
                          r"\b(said|say|explained|covered|mentioned|talked|showed|discussed|meant)\b")


def _cite_hint(name: str, result: str) -> str:
    if name.startswith(("get_video", "search_video")):
        form = "(video m:ss)"
    elif name.endswith("document") or name == "read_document_pages":
        form = "(p. N)" if "--- page" in result else "(slide N)"
    else:
        urls = re.findall(r"https?://(?:www\.)?([^/\s]+)", result[:3000])
        form = f"({urls[0]})" if urls else "(the website's name)"
    return f"\n\n[Notes that use this must end their written \"text\" with the source, like {form} (not only in \"say\").]"


TOOL_STATUS = {
    "get_video_transcript": "📺 Reading the video's captions",
    "search_video_transcript": "📺 Searching the video",
    "read_document_pages": "📄 Reading more of the document",
    "search_document": "📄 Searching the document",
    "read_webpage": "🌐 Reading the full page",
    "web_search": "🔎 Searching the web",
}


class Session:
    """One screen, plus follow-up questions about it."""

    def __init__(self, scene: Scene, image, selection: Box | None, source: Source | None = None,
                 on_status: Callable[[str], None] = lambda s: None):
        self.scene = scene
        self.source = source
        self.toolbox = Toolbox(None)  # rebuilt once the source has loaded
        self.on_status = on_status
        self.client = make_client()
        self.messages: list = [{"role": "system", "content": SYSTEM}]
        self.used_tools = False
        self._fetched: set[str] = set()  # tool calls already run in this session
        self._must_answer = False
        self._cancel = threading.Event()
        self._lock = threading.Lock()  # one answer at a time keeps the history in order
        view = selection if selection is not None else Box(0, 0, scene.w, scene.h)
        s = 1 / scene.scale
        crop = image.crop((round(view.x * s), round(view.y * s), round(view.x2 * s), round(view.y2 * s)))
        labelled = _label_parts(crop, scene, view) if config.LABEL_PARTS else crop
        self.b64, pw, ph = _encode(labelled)
        scene.set_view(view, pw, ph)
        # Hand-written numbers in a drawing are tiny once the whole screen is shrunk for
        # the model: also send the drawing itself, zoomed, so it reads them right.
        self.zoom_b64, self.zoom_note = None, ""
        fig = _main_drawing(scene)
        # A drawing that fills the screen (a chalk video, a whiteboard) is what the lesson is
        # about: look closer (zoom + more reasoning). A small chart on a text slide doesn't need it.
        self.drawing_lesson = fig is not None and fig.w * fig.h >= 0.25 * view.w * view.h
        if self.drawing_lesson and fig.w * fig.h < 0.7 * view.w * view.h:
            z = crop.crop((round((fig.x - view.x) * s), round((fig.y - view.y) * s),
                           round((fig.x2 - view.x) * s), round((fig.y2 - view.y) * s)))
            self.zoom_b64 = _encode(z)[0]
            self.zoom_note = (f"\n\nZOOM: the second image is the drawing at {scene.to_model(fig)} enlarged and without "
                              f"labels. Read values, symbols and colours there; point by describing the mark (or by "
                              f"L ids, or boxes in first-image pixels).")

    def cancel(self) -> None:
        self._cancel.set()

    def ask(self, question: str, on_action: Callable[[dict], None]) -> None:
        """Blocking: streams the answer, calling on_action for each drawing action."""
        with self._lock:
            self._cancel.clear()
            self._ask(question, on_action)

    def _context_text(self) -> str:
        if self.source is None:
            return "CONTEXT\nNothing beyond the screen (no video, document or page detected)."
        self.source.wait(timeout=3.0)  # usually loaded already, while the user was typing
        screen_text = "\n".join(l.text for l in self.scene.visible_lines())
        return "CONTEXT\n" + self.source.digest(screen_text)

    def _ask(self, question: str, on_action: Callable[[dict], None]) -> None:
        question = question.strip() or "Explain this to me."
        self._question = question
        self._walkthrough = bool(_WALKTHROUGH.search(question))
        self._honest_note, self._honest_done = "", False
        raw_on_action = on_action

        def on_action(a: dict) -> None:
            # Notes must explain, not narrate ("I'll check the section...") or speculate about
            # what someone said ("your lecturer likely explained..."). Drop those.
            text = str(a.get("text", ""))
            if a.get("op") in ("note", "summary") and (_NARRATION.match(text) or _SPECULATION.search(text)):
                return
            raw_on_action(a)
        mode = "\n\n" + WALKTHROUGH if self._walkthrough else ""
        if len(self.messages) == 1:
            self.context_text = self._context_text()
            self.toolbox = Toolbox(self.source)
            content = [
                {"type": "text", "text": f"{_elements(self.scene, self.drawing_lesson)}\n\n{self.context_text}{self.zoom_note}"
                                         f"\n\nUSER QUESTION\n{question}{mode}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.b64}",
                                                    "detail": config.IMAGE_DETAIL}},
            ]
            if self.zoom_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.zoom_b64}",
                                                                   "detail": "high"}})
        else:
            content = "FOLLOW-UP QUESTION (same screen; earlier drawings were cleared)\n" + question + mode
        self.messages.append({"role": "user", "content": content})

        tools = self.toolbox.specs()
        # The typed context check runs in parallel with the drawing request.
        # (A walkthrough of something on screen needs nothing more: the picture is the source.)
        # (Nor does explaining a drawing that fills the screen, unless the question reaches beyond it:
        # the text-only checker can't see the drawing and keeps asking for the transcript.)
        on_screen = self._walkthrough or self.drawing_lesson and not _ELSEWHERE.search(question)
        check = _pool.submit(self._check, question) if tools and config.CONTEXT_CHECK and not on_screen else None
        for round_no in range(config.MAX_TOOL_ROUNDS + 1):
            last = round_no == config.MAX_TOOL_ROUNDS
            gate = self._gate(check) if round_no == 0 else None
            # With the typed check running, it alone decides round 0; the drawer just draws.
            text, calls, held = self._stream(tools, force_answer=last or gate is not None or self._must_answer
                                             or (self._walkthrough and round_no == 0),
                                             on_action=on_action, gate=gate,
                                             # a walkthrough, or a drawing to point into, needs a closer look
                                             reasoning=(config.DRAW_REASONING if self._walkthrough or self.drawing_lesson
                                                        else config.REASONING) if round_no == 0
                                             else config.AFTER_LOOKUP_REASONING)
            self._must_answer = False
            if self._cancel.is_set():
                return
            if not calls and round_no == 0 and check is not None:
                verdict = self._verdict(check)
                if verdict is not None and verdict.cannot_know.strip() and not self._honest_done:
                    self._honest_done = True
                    self._honest_note = verdict.cannot_know.strip()
                self._say_honest_note(on_action)
                if verdict is not None and not verdict.enough and not verdict.cannot_know.strip():
                    calls = self._lookups(verdict)  # drawings from this pass are discarded, never shown
                    if calls:
                        text = ""
                        self.on_status(f"🔎 Need more context: {verdict.missing}")
                if not calls:
                    for a in held:  # the check said enough: show what was waiting
                        on_action(a)
            if not calls:
                if round_no == 0:
                    print("  context: enough - drew straight from the screen" +
                          (" and digest" if self.source else ""))
                self.messages.append({"role": "assistant", "content": text or "{}"})
                return
            self.used_tools = True
            self.messages.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls]})
            self._run_tools(calls)

    # -- typed context check ----------------------------------------------------

    def _check(self, question: str):
        t = time.perf_counter()
        model = _check_model(list(self.toolbox.models))
        tools = "\n".join(f"- {name}({', '.join(f for f in m.model_fields if f != 'reason')}): {(m.__doc__ or '').strip()}"
                          for name, m in self.toolbox.models.items())
        screen = "\n".join(l.text for l in self.scene.visible_lines())[:6000]
        if self.scene.visible_parts():  # hand-drawn marks that OCR can't turn into text
            screen += ("\n[The screen also shows a drawing (a sketch, diagram or video frame with hand-written "
                       "labels and values). The tutor sees it; it isn't in this text.]")
        kwargs = dict(model=config.MODEL, max_completion_tokens=3000, response_format=model, messages=[
            {"role": "system", "content": JUDGE},
            {"role": "user", "content": f"AVAILABLE TOOLS\n{tools}\n\nSCREEN TEXT\n{screen}\n\n{self.context_text}"
                                        f"\n\nQUESTION\n{question}"}])
        if config.CHECK_REASONING:
            kwargs["reasoning_effort"] = config.CHECK_REASONING
        verdict = self.client.chat.completions.parse(**kwargs).choices[0].message.parsed
        print(f"  context check ({time.perf_counter() - t:.1f}s): " +
              ("enough" if verdict.enough else f"NOT enough - {verdict.missing}"))
        return verdict

    @staticmethod
    def _verdict(check: Future, timeout: float = 4.0):
        """The check's verdict, waiting at most `timeout` s: if it stalls, the drawing
        that is already waiting is shown rather than keeping the user waiting."""
        try:
            return check.result(timeout=timeout)
        except TimeoutError:
            print(f"  context check too slow (> {timeout:.0f}s after the drawing was ready); showing the drawing")
            return None
        except Exception as e:  # the check is a safety net; never block an answer on it
            print(f"  context check failed: {type(e).__name__}: {e}")
            return None

    def _gate(self, check: Future | None):
        """None = hold drawings (check pending), True = show them, False = stop (check wants more context)."""
        if check is None:
            return None

        def gate():
            if not check.done():
                return None
            v = self._verdict(check)
            if v is not None and v.cannot_know.strip() and not self._honest_done:
                self._honest_done = True
                self._honest_note = v.cannot_know.strip()  # said first, before any explanation
            return v is None or v.enough or bool(v.cannot_know.strip()) or not self._lookups(v)
        return gate

    def _say_honest_note(self, on_action) -> None:
        if self._honest_note:
            on_action({"op": "note", "text": self._honest_note, "color": "orange"})
            print(f"  honest: {self._honest_note}")
            self._honest_note = ""

    def _lookups(self, verdict) -> list[dict]:
        calls = []
        for i, lk in enumerate(verdict.lookups[:3]):
            fields = self.toolbox.models[lk.tool].model_fields
            args = {k: getattr(lk, k, None) for k in fields if k != "reason"}
            if "find" in fields:
                args["find"] = lk.query or self._question  # what to look for on a long page
            if isinstance(args.get("url"), str) and args["url"].lower() in ("null", "none", ""):
                args["url"] = None
            args["reason"] = verdict.missing
            if lk.tool == "read_webpage" and not args.get("url") and self.source is None or \
                    lk.tool == "read_webpage" and not args.get("url") and self.source.kind != "webpage":
                continue  # "read the page" only makes sense when a web page is open
            raw = json.dumps(args)
            try:
                self.toolbox.parse(lk.tool, raw)  # type-check before anything runs
            except ValidationError:
                continue
            calls.append({"id": f"check_{i}", "name": lk.tool, "arguments": raw})
        return calls

    def _stream(self, tools: list, force_answer: bool, on_action, gate=None,
                reasoning: str | None = None) -> tuple[str, list[dict], list[dict]]:
        kwargs = dict(model=config.MODEL, messages=self.messages, stream=True, max_completion_tokens=6000)
        if "azure" not in str(self.client.base_url):  # Azure caches automatically and may reject this field
            kwargs["prompt_cache_key"] = "chalktalk-v1"
        reasoning = config.REASONING if reasoning is None else reasoning
        # Some newer models (gpt-5.4+ on OpenAI's chat API) refuse tools together with reasoning.
        # Then the drawing request goes without tools: the typed context check still fetches
        # whatever is missing, before this request, with no tools needed.
        if config.NO_TOOLS_WITH_REASONING and reasoning and reasoning != "none":
            tools = []
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "none" if force_answer else "auto"
            kwargs["parallel_tool_calls"] = True
        if reasoning:
            kwargs["reasoning_effort"] = reasoning
        parser = ActionStream()
        text: list[str] = []
        calls: dict[int, dict] = {}
        held: list[dict] = []  # drawings waiting for the context check's verdict
        seen: set[str] = set()  # some models repeat their whole answer; draw each action once
        with self.client.chat.completions.create(**kwargs) as stream:
            for chunk in stream:
                if self._cancel.is_set():
                    break
                state = gate() if gate else True
                if state is False:
                    return "", [], []  # the check wants more context; this pass is redone after the lookups
                if state:
                    self._say_honest_note(on_action)
                if state and held:
                    for a in held:
                        on_action(a)
                    held.clear()
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text.append(delta.content)
                    for action in parser.feed(delta.content):
                        key = json.dumps({k: action.get(k) for k in ("op", "target", "phrase", "from", "to", "text")},
                                         sort_keys=True)
                        if key not in seen and len(seen) < (32 if self._walkthrough else 12):
                            seen.add(key)
                            (on_action if state else held.append)(action)
                for tc in delta.tool_calls or []:
                    c = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        c["id"] = tc.id
                    if tc.function and tc.function.name:
                        c["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        c["arguments"] += tc.function.arguments
        return "".join(text), [calls[i] for i in sorted(calls)], held

    def _run_tools(self, calls: list[dict]) -> None:
        parsed = []
        fresh = 0
        for c in calls:
            try:
                args = self.toolbox.parse(c["name"], c["arguments"])
                key = c["name"] + json.dumps(args.model_dump(exclude={"reason"}), sort_keys=True)
                if key in self._fetched:
                    parsed.append((c, "(Already fetched above - nothing new. Answer with what you have.)"))
                    continue
                self._fetched.add(key)
                fresh += 1
                reason = getattr(args, "reason", "")
                status = TOOL_STATUS.get(c["name"], "Looking things up")
                self.on_status(f"{status}… {reason}".strip())
                print(f"  context: NOT enough -> {c['name']}({c['arguments']})")
                parsed.append((c, args))
            except (KeyError, ValidationError) as e:
                parsed.append((c, e))

        def run(item):
            c, args = item
            if isinstance(args, str):
                return args
            if isinstance(args, Exception):
                return f"Invalid tool call: {args}"
            return self.toolbox.run(args)

        if not fresh:
            self._must_answer = True  # only repeats: stop looking things up and draw

        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(run, parsed))
        for (c, _), result in zip(parsed, results):
            self.messages.append({"role": "tool", "tool_call_id": c["id"],
                                  "content": result + _cite_hint(c["name"], result)})
        self.on_status("✏️ Got it, explaining…")
