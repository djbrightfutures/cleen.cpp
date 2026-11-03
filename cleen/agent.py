"""cleen.cpp agent loop -- the thing that makes a model MOVE, not just talk.

A raw model answers once, in text. An agent THINKS -> ACTS -> SEES THE RESULT ->
THINKS AGAIN, until the job is done. That is the difference between "I can't open a
browser" and actually opening it. cleen is a router, so this loop lives ONCE here and
every model that routes through cleen inherits it -- the 27B, the 8B, a cloud burst,
all the same.

Design choices that make even a SMALL model reliable:
  * ONE tool per step, emitted as a single JSON block. An 8B can't juggle parallel tool
    calls, but it can emit `{"tool":"open_url","args":{"url":"youtube.com"}}`.
  * The loop -- not the model -- parses, executes, and feeds the result back. The model's
    only job is to name the next action; cleen does it.
  * Non-stream with full failover each step (router.complete), so a step is never dark.
  * Hard step cap + repeat-guard so a confused model can't spin forever.

Public:
    run(messages, model=None, mode="auto", fast=False, max_steps=None, allow=None, log=None)
      -> {"text": final_answer, "actions": [..], "steps": n, "route": label}
"""
import json
import re
import urllib.parse

from . import config, router, tools


def _tool_prompt(allow):
    return (
        "You are not a text-only assistant. You can CONTROL this Windows computer through "
        "tools -- open the browser, open apps, run shell commands, search the web, type, and "
        "see the screen. When an answer requires DOING something on the machine, DO IT with a "
        "tool instead of saying you can't.\n\n"
        "TOOLS:\n" + tools.spec_for_prompt(allow) + "\n\n"
        "HOW TO USE A TOOL: reply with ONE fenced block and nothing else:\n"
        "```tool\n"
        '{"tool": "<name>", "args": { ... }}\n'
        "```\n"
        "I run it and give you the result. Then either call another tool the same way, or -- "
        "when the task is done -- reply with your final answer in plain text (NO block). "
        "Use the fewest steps. If a tool returns an error, read it and try a different way. "
        "Open exactly what was asked, nothing extra."
    )


# A tool call = the first JSON object carrying a tool name. Accept a fenced ```tool/```json
# block, or a bare object, plus common key aliases a model might emit.
_FENCE = re.compile(r"```(?:tool|json)?\s*(\{.*?\})\s*```", re.S | re.I)
_BARE = re.compile(r"(\{(?:[^{}]|\{[^{}]*\})*\})", re.S)
_NAME_KEYS = ("tool", "name", "action", "function")
_ARG_KEYS = ("args", "arguments", "input", "parameters", "params")


def parse_tool_call(text):
    """Return (name, args) for the first tool call in `text`, or (None, None) if it's a
    final plain-text answer. Tolerant of fenced/bare JSON and key-name variants."""
    candidates = _FENCE.findall(text or "")
    if not candidates:
        candidates = _BARE.findall(text or "")
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str)), None)
        if not name:
            continue
        args = next((obj[k] for k in _ARG_KEYS if isinstance(obj.get(k), dict)), None)
        if args is None:
            # some models inline args as sibling keys: {"tool":"open_url","url":"..."}
            args = {k: v for k, v in obj.items() if k not in _NAME_KEYS and k not in _ARG_KEYS}
        return name, args
    return None, None


# --------------------------------------------------------------------------------
# Deterministic intent resolver -- the "Codex-grade" reliability layer.
#
# A weak local model will NARRATE a browser action ("I've opened YouTube and searched")
# without ever emitting a tool call -- it confuses "I should do X" with "I did X". So for
# the few HIGH-VALUE, single-step commands the owner actually gives (open a site, search
# YouTube, web search), cleen does NOT trust the model to compose the call: it recognizes
# the intent itself and fires the exact right open_url. The action becomes real + verified
# by the tool's own result, and can never be faked. Everything the resolver doesn't know
# still flows through the model loop (guarded below).
# --------------------------------------------------------------------------------
_YT_RESULTS = "https://www.youtube.com/results?search_query="
_GOOGLE = "https://www.google.com/search?q="
_DOMAIN = re.compile(r"\b([a-z0-9][\w-]*\.(?:com|net|org|io|tv|gov|edu|co|me|app|dev|ai|us|xyz))\b", re.I)
_SEARCH_LEAD = re.compile(r"^(?:search(?:\s+the\s+web)?(?:\s+for)?|google|look\s+up|lookup|web\s+search)\b", re.I)
_OPEN_LEAD = re.compile(r"^(?:open|go\s*to|goto|visit|pull\s+up|bring\s+up)\b", re.I)
_YT = re.compile(r"\byou\s?tube\b", re.I)

# A real COMMAND starts with an imperative. A question, a complaint, or a recap is
# CONVERSATION and must never be turned into an action -- the resolver used to fire on the mere
# PRESENCE of "youtube", so "u open edge, but then open youtube and i didnt tell u that lol"
# got dumped verbatim into a search query. Two gates now stand in front of every intent.
_CMD_LEAD = re.compile(
    r"^(?:open|go\s*to|goto|visit|pull\s+up|bring\s+up|launch|start|show\s+me|"
    r"search(?:\s+the\s+web)?(?:\s+for)?|google|look\s*up|lookup|web\s+search|find|"
    r"play|put\s+on|watch|listen\s+to|stream)\b", re.I)
# Feedback markers. These win even over a leading verb, because "open edge again" inside
# "lets start over, open edge again" is him RESETTING, and acting on a complaint is how the
# agent kept re-searching something he had already told it to stop.
_CHATTER = re.compile(
    r"\b(lol|haha|didn'?t|didnt|did\s+not|isn'?t|isnt|wasn'?t|aren'?t|"
    r"even\s+though|not\s+really|start\s+over|fix\s+it|"
    r"but\s+(?:u|you)\b|why\s+(?:u|you|did|is|isn)|"
    r"(?:u|you)\s+(?:wanted|said|told|open|opened|didnt|continue|continued)|"
    r"wrong|broken|dumb|stupid)\b", re.I)
# Checked only AFTER the two gates pass, so a bare "play" inside a confirmed command is safe.
_PLAY = re.compile(r"\b(?:play|put\s+on|listen\s+to|stream)\b", re.I)
# Part of "play the FIRST SONG" -- never part of the artist/subject being searched for.
_PLAY_FILLER = {"first", "1st", "second", "2nd", "top", "song", "songs", "track", "tracks",
                "video", "videos", "one", "result", "results", "tune", "it", "this", "that"}

# Command/filler words stripped when extracting the actual search subject. Deliberately does
# NOT drop ordinary query words (in, on, of, to, with, video) -- only verbs/connectors/fillers.
_STOP = {"search", "google", "find", "look", "lookup", "web", "open", "go", "goto", "visit",
         "pull", "bring", "launch", "up", "please", "can", "you", "u", "me", "i", "want",
         "then", "now", "some", "my", "browser",
         "play", "put", "watch", "listen", "stream"}
# Connectors trimmed only from the ENDS of the extracted subject (a dangling "on" from
# "... on youtube"), never from the middle where they're real query words.
_EDGE = {"on", "in", "to", "for", "with", "at", "by", "of", "and", "the", "a", "an", "from"}


def _last_user(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _clean_query(text, drop=(), extra=()):
    """Extract the search subject: drop command/filler words (anywhere, since commands like
    'search youtube AND search X' interleave) + any caller-named host tokens. Keeps real
    query words verbatim so the search engine does the final matching (fixes typos too).
    `extra` adds context-specific fillers -- for a PLAY command, "the first song" is an
    instruction about WHICH result to open, not part of the artist being searched for."""
    dropset = {d.lower() for d in drop} | {e.lower() for e in extra}
    out = []
    for w in re.split(r"\s+", text.strip()):
        core = re.sub(r"^[^\w]+|[^\w.'-]+$", "", w)   # strip surrounding punctuation
        lw = core.lower()
        if not lw or lw in _STOP or lw in dropset:
            continue
        out.append(core)
    while out and out[0].lower() in _EDGE:      # trim dangling connectors off both ends
        out.pop(0)
    while out and out[-1].lower() in _EDGE:
        out.pop()
    return " ".join(out).strip(" .,!?")


def _resolve_intent(text, allow):
    """Return (tool, args, honest_confirmation) for a clear single-step browser command, or
    None. Every intent here needs open_url, so bail if it's not allowed."""
    t = (text or "").strip()
    if not t or (allow is not None and "open_url" not in allow):
        return None
    low = t.lower()

    # GATE 1: only an IMPERATIVE drives a real action. GATE 2: a complaint/question wins even
    # over a leading verb. Without these, a sentence that merely MENTIONED youtube got dumped
    # verbatim into a search query, and a "stop, start over" was itself executed as a command.
    if not _CMD_LEAD.match(low) or _CHATTER.search(low):
        return None

    want_play = bool(_PLAY.search(low))

    # 1) YouTube, or any "play X" -> PLAY the real top result; else the results page.
    if _YT.search(low) or want_play:
        q = _clean_query(t, drop=("youtube", "you", "tube", "youtube.com", "youtu.be"),
                         extra=(_PLAY_FILLER if want_play else ()))
        if want_play and q:
            if allow is not None and "youtube_play" not in allow:
                return None
            # Confirmation is None ON PURPOSE: youtube_play reports what it ACTUALLY did
            # (played the real top hit, or couldn't parse one and opened the results page).
            # A canned "it's playing now" is exactly the lie this whole gate exists to kill.
            return ("youtube_play", {"query": q}, None)
        if q:
            return ("open_url", {"url": _YT_RESULTS + urllib.parse.quote(q)},
                    'Opened YouTube and searched "%s" -- the results are on screen.' % q)
        if _YT.search(low):
            return ("open_url", {"url": "https://www.youtube.com"}, "Opened YouTube.")
        return None

    # 2) Explicit web search -> a real Google results page he can SEE.
    if _SEARCH_LEAD.match(low):
        q = _clean_query(t)
        if q:
            return ("open_url", {"url": _GOOGLE + urllib.parse.quote(q)},
                    'Opened Google and searched "%s" -- the results are on screen.' % q)

    # 3) Open a specific site/domain (open espn.com / go to github.com).
    if _OPEN_LEAD.match(low):
        m = _DOMAIN.search(t)
        if m:
            return ("open_url", {"url": m.group(1)}, "Opened %s." % m.group(1))
    return None


# A model's plain-text answer that ASSERTS it performed a machine action. Used only to catch
# a hallucinated completion (claim with zero tools fired) -- never to block a real recap.
_CLAIM = re.compile(
    r"\b(i(?:'ve| have| just)?\s*(?:opened|searched|launched|typed|ran|run|executed|did|done|"
    r"completed|pulled up|brought up)|opening|searching|launching|here (?:you|u) go|"
    r"(?:it'?s|task is?)\s*done|successfully)\b", re.I)


def _complete(messages, model, mode, fast, log):
    chain = router.route_chain(model, mode=mode, fast=fast, messages=messages)
    text, label = router.complete(chain, messages, {"temperature": config.ENGINE_DEFAULT_TEMP}, log=log)
    return text, label


def run(messages, model=None, mode="auto", fast=False, max_steps=None, allow=None, log=None):
    """Drive the think->act->observe loop. `messages` is OpenAI-shaped; the first system
    message is augmented with the tool protocol (or one is inserted)."""
    allow = allow if allow is not None else config.TOOLS_ALLOW
    max_steps = int(max_steps if max_steps is not None else config.TOOLS_MAX_STEPS)

    # Deterministic shortcut: a clear browser/search/open command fires for REAL here,
    # verified by the tool's own result -- it never depends on the model composing JSON,
    # and it can never be narrated-but-not-done. This is the Codex-grade reliability.
    intent = _resolve_intent(_last_user(messages), allow)
    if intent:
        iname, iargs, human = intent
        result = tools.dispatch(iname, iargs, log=log)
        ok = not re.match(r"\s*(?:error|refused)\b", result, re.I) and "failed" not in result.lower()
        # `human is None` = the tool itself is the honest narrator (see youtube_play).
        return {"text": (human or result) if ok else ("I tried but it failed: " + result),
                "actions": [{"tool": iname, "args": iargs, "result": result}],
                "steps": 1, "route": "intent"}

    msgs = [dict(m) for m in (messages or [])]
    protocol = _tool_prompt(allow)
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = (msgs[0].get("content") or "") + "\n\n" + protocol
    else:
        msgs.insert(0, {"role": "system", "content": protocol})

    actions = []
    last_sig = None
    label = "local"
    forced = False
    for step in range(max_steps):
        text, label = _complete(msgs, model, mode, fast, log)
        name, args = parse_tool_call(text)
        if not name:
            answer = text.strip()
            # Anti-hallucination: if NO tool has fired this whole run yet and the "answer"
            # asserts it DID something on the machine, it's a fake completion (the exact
            # "I've opened YouTube and searched" bug). Force a real tool call ONCE; if it
            # still only narrates, be honest rather than lie on the model's behalf.
            if not actions and _CLAIM.search(answer):
                if not forced:
                    forced = True
                    msgs.append({"role": "assistant", "content": text})
                    msgs.append({"role": "user", "content":
                        "STOP. You called NO tool, so nothing actually happened on the "
                        "computer -- you only described it. Never claim an action you did "
                        "not take. If this needs doing something on the machine, emit the "
                        "tool call NOW as a single ```tool block. If it genuinely needs no "
                        "action, answer plainly without claiming you did anything."})
                    continue
                return {"text": "I didn't actually do that -- I had no working tool call for "
                                "it, so I won't pretend I did. Tell me the exact site or "
                                "command and I'll really run it.",
                        "actions": actions, "steps": step, "route": label}
            return {"text": answer, "actions": actions, "steps": step, "route": label}

        sig = name + json.dumps(args or {}, sort_keys=True)
        if sig == last_sig:                      # same call twice = stuck; stop acting
            note = ("(stopped: repeated the same tool call. Result was already: %s)"
                    % (actions[-1]["result"] if actions else ""))
            return {"text": (text.split("```")[0].strip() or note), "actions": actions,
                    "steps": step, "route": label}
        last_sig = sig

        result = tools.dispatch(name, args, log=log)
        actions.append({"tool": name, "args": args, "result": result})
        # Feed the model its own action + the real result, so it reasons from truth.
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "user",
                     "content": "TOOL RESULT (%s):\n%s\n\nContinue, or give your final answer in plain text."
                                % (name, result)})

    # Ran out of steps -> ask for a wrap-up from what it has learned.
    msgs.append({"role": "user", "content": "Step limit reached. Give your final answer now in plain text."})
    try:
        text, label = _complete(msgs, model, mode, fast, log)
    except Exception:
        text = actions[-1]["result"] if actions else "I couldn't finish that."
    return {"text": text.strip(), "actions": actions, "steps": max_steps, "route": label}


def actions_summary(actions):
    """A one-line-per-action human recap (for a reply footer / audit)."""
    if not actions:
        return ""
    return "\n".join("- %s(%s)" % (a["tool"], ", ".join("%s=%s" % (k, v)
                     for k, v in (a["args"] or {}).items())) for a in actions)
