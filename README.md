# ChalkTalk

An AI tutor that explains whatever is on your screen by **drawing on it**: circling the key term, underlining, drawing arrows, writing handwritten notes in the empty space and sketching small diagrams. It works like a teacher with a marker on a whiteboard.

Press **Ctrl+Shift+Space** over a slide, a PDF, a YouTube video, code or a web page. Optionally drag around the part that confuses you, type your question and press Enter. You can ask follow-ups in the bar at the bottom.

**Or just talk:** press **Ctrl+G** (or click **Speak**) and ask out loud.
- It stops listening by itself when you pause, then transcribes your question with a speech-to-text model on Microsoft Foundry (English).
- Saying "stop", "clear the screen" or "never mind" closes it.

**The bar** sits at the top of the screen. It holds Chalky (a piece of chalk with eyes), your question, the tutor's current line, **Stop**, a text box, **Speak** and **Ask/Reply**. Chalky shows what's happening:
- **Always:** breathes, bobs, sways and blinks.
- **While you type:** watches your mouse.
- **While thinking:** glances around.
- **While you talk:** opens its mouth in time with your voice.
- **While drawing:** its eyes follow the pen tip and it sheds chalk dust.
- **At the end:** bounces when done, or droops if something went wrong.

**Stop** does the sensible thing: it ends a recording, otherwise stops an answer, otherwise closes.

**It teaches like a teacher.** Every step is spoken aloud (Windows voices, offline), and the next step only starts after the voice has finished. Strokes are drawn at a hand's pace, and 🔊 in the bar mutes the voice. While it teaches, the bar shrinks to a pill and gets out of the way of whatever it is explaining.

**Live walkthroughs.** Ask things like "show how I reach A to E" or "walk me through it", and ChalkTalk actually runs the algorithm on your screen, step by step:
- it circles each node as it's settled;
- it traces each edge it relaxes;
- it writes each distance next to its node, crossing out the old value when a better one is found;
- it finishes by drawing the final path in red.

Graph node letters and single-digit weights, which OCR normally skips, are recovered by reading them together on one strip, and each node is mapped to its whole circle.

Before answering, it checks whether it knows enough. If the screen is enough, it draws straight away. If not, it first reads more: the video's captions, other pages of the PDF, the whole web page, or a web search.

## Setup

```
pip install -r requirements.txt
copy .env.example .env      # add OPENAI_API_KEY (and FIRECRAWL_API_KEY for the web)
python main.py
```

- **Microsoft Foundry:** set `OPENAI_BASE_URL` to your Azure OpenAI endpoint and `CHALK_MODEL` to your deployment name. The code is the same.
- **No key:** `python main.py --demo` draws a canned explanation, so you can test the overlay.
- **OCR language pack:** Windows OCR needs one. English is normally installed. If it's missing, go to Settings > Time & language > Language > English > Language options > OCR.

## Getting enough context before explaining

As soon as you press the hotkey, ChalkTalk works out what's in front of you. It loads that in the background while you type:

| On screen | How it's recognised | Loaded up front | Tools the model can call |
|---|---|---|---|
| YouTube video | Browser address bar (UI Automation, exact URL) | Title, channel, description, the last 2.5 min of captions (position read from the player's `3:05 / 18:40`) | `get_video_transcript`, `search_video_transcript` |
| PDF / PPTX / DOCX / TXT on your device | `file:///` URL in the browser, or the window title of Acrobat, PowerPoint, Word, Sumatra, etc. (file found through Windows' Recent items) | Which page or slide is on screen (matched by text), plus the pages around it | `read_document_pages`, `search_document` |
| PDF opened from the web | `.pdf` URL | Downloaded, then handled like a local PDF | same |
| Any web page | Address bar | Title and URL | `read_webpage` |
| Anything | – | – | `web_search` (Firecrawl) |

**A typed context check decides.** Two requests start at the same time when you press Enter:
1. **The check:** a small text-only request returns a typed verdict, `ContextCheck {enough: bool, missing: str, lookups: [...]}` (a Pydantic model; `lookups[].tool` only accepts the tools available right now).
2. **The drawing request:** it starts drawing straight away, but its drawings are held until the verdict arrives.

What happens next depends on the verdict:
- **Enough:** the held drawings appear. There's no extra wait, because both ran in parallel.
- **Not enough:** the held drawings are thrown away unseen, and the lookups run in parallel. The bar shows what's happening, e.g. "🔎 Need more context: …". Then it draws with the new context, citing "(video 11:02)", "(p. 4)" or "(en.wikipedia.org)".

**Search by meaning:** transcripts, document passages and web-page sections are embedded (`text-embedding-3-small`) in the background while you type. A question about "variants" finds the sections called "Momentum" and "Extensions".

**Type safety:** every tool's arguments are also a strict Pydantic schema, validated before anything runs. Repeated lookups are cached, and notes that narrate ("I'll check…") are filtered out.

The console logs every decision, e.g. `context check (2.6s): NOT enough - …` followed by the lookups.

## Testing

```
python -m pytest tests/test_offline.py -q   # 32 tests: placement, DPI, colours, parser, typed tools, PDF/PPTX/DOCX, detection
python -m pytest tests/test_network.py -q   # YouTube, Firecrawl, a real Chrome window
python tests/eval_model.py                  # 10 real-model scenarios: right context decision, drawings land, facts, citations, language, speed
python tests/live_e2e.py out_dir            # real Chrome windows (YouTube, local PDF, Wikipedia) end to end, with screenshots
```

## How it meets the three drawing goals

**1. Fast response**
- Capture, OCR and context loading run while you're still typing.
- The model streams JSON Lines, and each action is drawn the moment it arrives.
- Low reasoning effort, a capped JPEG, and prompt caching.
- The console prints `first drawing after … ms`.

**2. Drawing in the right place**
- The model never guesses pixels. Windows OCR measures every line and word, and the model points at them by id, phrase or figure region.
- Correct on 100% to 300% display scaling and on multiple monitors.

**3. Readable writing**
- Notes go into empty space found with an occupancy map, avoiding the target, other notes and the bar.
- Pointer lines are routed around text.
- Ink colours are picked by contrast with the real background, text gets a halo, and there's a sticky-note card when no space is free.

## Files

| File | What it does |
|---|---|
| `main.py` | App flow: hotkey → capture + context → ask → stream → draw |
| `window.py` | Foreground app, window title, exact browser URL |
| `context.py` | Sources (YouTube, documents, web pages), digests, typed tools, Firecrawl |
| `planner.py` | Prompt, context check, tool loop, streaming parser (OpenAI) |
| `capture.py`, `ocr.py` | Screenshot of the monitor under the mouse; Windows OCR with word boxes |
| `layout.py` | Screen geometry: reading order, figures, free-space placement, colours, pointer routing |
| `compose.py`, `items.py` | Turns actions into placed, animated, hand-drawn marks |
| `overlay.py` | Transparent click-through drawing layer and the bottom bar |
| `render_test.py`, `selftest.py` | Offline render check; drives the real overlay and screenshots it |
