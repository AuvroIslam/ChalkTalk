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
- "phrase": the exact words inside the target line to pinpoint, copied exactly from its text.
- Only if nothing listed covers it: {"box":[x,y,w,h]} in screenshot pixels.

ACTIONS
{"op":"circle","target":"L4","phrase":"learning rate","color":"red"}   ring a key term or element
{"op":"underline","target":"L6","phrase":"..."}
{"op":"highlight","target":"L8"}   marker over words or lines
{"op":"box","target":"L3-L7"}   group lines that belong together
{"op":"arrow","from":"L2","to":"R1","label":"feeds"}   show a relationship or flow; from_phrase/to_phrase allowed
{"op":"number","target":"L5","n":1}   step-order badge
{"op":"note","target":"L4","phrase":"learning rate","text":"How big each step is. Too big = you overshoot."}   handwritten explanation; the app finds empty space beside the target and draws a pointer
{"op":"diagram","target":"R1","title":"Gradient descent","nodes":["Guess","Measure error","Step downhill"],"edges":[[0,1,""],[1,2,""],[2,0,"repeat"]]}   a small sketch (max 5 nodes) when a picture explains better than words: a flow, cycle, cause->effect, comparison, analogy
{"op":"summary","text":"..."}   one-line takeaway, last

Any action may add "say": one short spoken-style sentence shown as a caption while it is drawn.
Colors: red, blue, green, purple, orange. Keep one colour per idea (a mark and its note share a colour).

HOW TO TEACH
- Answer the user's actual question. If they only point at an area, explain the most confusing idea in it.
- Mark first, then explain: circle or highlight the thing, then a note on it.
- Notes: at most 15 words, plain language, an analogy or a concrete example when possible. Never just restate the screen.
- Notes explain the idea itself. Never narrate what you are doing ("I'll open...", "Let me...").
- Every note that uses fetched context ends with its source in brackets: "(video 3:12)", "(p. 14)", "(slide 5)", "(Wikipedia)".
- 3 to 8 actions in total, never more. Don't cover the whole screen.
- Use only ids from the lists. Don't place notes yourself.
- Write every note, label and caption in the language of the user's question (a Bangla question gets Bangla notes), even when the screen is in English."""


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


def _elements(scene: Scene) -> str:
    rows = []
    for ln in scene.visible_lines()[:220]:
        text = ln.text if len(ln.text) <= 90 else ln.text[:87] + "..."
        rows.append(f"{ln.id} {scene.to_model(ln.box)} {json.dumps(text, ensure_ascii=False)}")
    regs = [f"{r.id} {scene.to_model(r.box)}" for r in scene.visible_regions()]
    return "TEXT LINES\n" + ("\n".join(rows) or "(none)") + "\n\nREGIONS\n" + ("\n".join(regs) or "(none)")


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

If not enough: "missing" says in a few words what is missing, and "lookups" lists at most 3 lookups using only the listed tools, with exact arguments (unused fields null). For read_webpage and every search, put what to look for in "query". If enough: "missing" is "" and "lookups" is []."""


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
                        enough=(bool, ...), missing=(str, ...), lookups=(list[lookup], ...))


_NARRATION = re.compile(r"(?i)^\s*(i'?ll|i will|i'm going to|let me|let's (check|open|look)|checking|looking (at|for)|"
                        r"open(ing)? the|scroll|see the (section|page)|look for)\b")


def _cite_hint(name: str, result: str) -> str:
    if name.startswith(("get_video", "search_video")):
        form = "(video m:ss)"
    elif name.endswith("document") or name == "read_document_pages":
        form = "(p. N)" if "--- page" in result else "(slide N)"
    else:
        urls = re.findall(r"https?://(?:www\.)?([^/\s]+)", result[:3000])
        form = f"({urls[0]})" if urls else "(the website's name)"
    return f"\n\n[Notes that use this must end with the source, like {form}.]"


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
        self.b64, pw, ph = _encode(crop)
        scene.set_view(view, pw, ph)

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
        raw_on_action = on_action

        def on_action(a: dict) -> None:
            # Notes must explain, not narrate ("I'll check the section..."). Drop those.
            if a.get("op") in ("note", "summary") and _NARRATION.match(str(a.get("text", ""))):
                return
            raw_on_action(a)
        if len(self.messages) == 1:
            self.context_text = self._context_text()
            self.toolbox = Toolbox(self.source)
            content = [
                {"type": "text", "text": f"{_elements(self.scene)}\n\n{self.context_text}\n\nUSER QUESTION\n{question}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.b64}",
                                                    "detail": config.IMAGE_DETAIL}},
            ]
        else:
            content = "FOLLOW-UP QUESTION (same screen; earlier drawings were cleared)\n" + question
        self.messages.append({"role": "user", "content": content})

        tools = self.toolbox.specs()
        # The typed context check runs in parallel with the drawing request.
        check = _pool.submit(self._check, question) if tools and config.CONTEXT_CHECK else None
        for round_no in range(config.MAX_TOOL_ROUNDS + 1):
            last = round_no == config.MAX_TOOL_ROUNDS
            gate = self._gate(check) if round_no == 0 else None
            # With the typed check running, it alone decides round 0; the drawer just draws.
            text, calls, held = self._stream(tools, force_answer=last or gate is not None or self._must_answer,
                                             on_action=on_action, gate=gate,
                                             reasoning=config.REASONING if round_no == 0 else config.AFTER_LOOKUP_REASONING)
            self._must_answer = False
            if self._cancel.is_set():
                return
            if not calls and round_no == 0 and check is not None:
                verdict = self._verdict(check)
                if verdict is not None and not verdict.enough:
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
    def _verdict(check: Future):
        try:
            return check.result(timeout=20)
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
            return v is None or v.enough or not self._lookups(v)
        return gate

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
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "none" if force_answer else "auto"
            kwargs["parallel_tool_calls"] = True
        reasoning = config.REASONING if reasoning is None else reasoning
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
                        if key not in seen and len(seen) < 9:
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
