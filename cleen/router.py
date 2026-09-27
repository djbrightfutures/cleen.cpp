"""cleen.cpp smart router -- the 'smarter' half of "a better, smarter llama.cpp".

Given a request + a mode, decide LOCAL (free, private, any hardware) or CLOUD
(fast, when you need speed). The pitch: "how fast do you actually need tokens?"

  mode 'local'  -> always local  (free, offline, private)
  mode 'cloud'  -> always cloud  (fast; falls back to local only if EVERY provider fails)
  mode 'auto'   -> local by default; go cloud when it clearly wins:
        * caller asked for speed (fast=True)
        * the local runtime is down
        * SMART routing: the prompt looks hard (code / long context / multi-step
          reasoning / math) and a cloud provider is available

The model IDENTITY is the same either way: the catalog resolves one logical name to
a local id and a cloud id, so your prompt hits "the same model", just at a different
speed. That seam is the product.

Two things a raw inference engine never does, added here as generators the server
drives:
  * route_chain()          -> an ORDERED list of backends to try (primary + failovers)
  * complete() / stream_with_failover()
                           -> run that chain so a request is NEVER dark while any brain
                              is reachable (provider -> next provider -> local).
"""
import re

from . import config
from .engine import LocalBackend, CloudBackend, BackendError


def resolve_model(name):
    """Logical catalog name -> (name, spec{local,cloud,desc}). A raw id passes through."""
    name = name or config.DEFAULT_MODEL
    if name in config.MODELS:
        return name, config.MODELS[name]
    return name, {"local": name, "cloud": name, "desc": "raw model id (passed through)"}


# --------------------------------------------------------------------------------
# Smart difficulty classifier -- cheap signals only (regex + membership), no model
# call, so it costs microseconds. Easy/chit-chat -> local (free, private); hard
# (code, long context, multi-step reasoning, math) -> fast cloud.
# --------------------------------------------------------------------------------
_CODE_RE = re.compile(
    r"```|\b(def|class|function|import|const|let|var|public|private|async|await|"
    r"return|select|insert|update|delete|#include|std::|println|printf|console\.log)\b|"
    r"=>|[{};]\s*$", re.I | re.M)
_CODE_WORDS = ("refactor", "debug", "stack trace", "traceback", "exception", "compile",
               "implement", "algorithm", "regex", "endpoint", "null pointer", "segfault",
               "syntax error", "unit test", "code review", "pull request", "typescript",
               "python", "javascript", "rust", "golang", "sql query")
_MATH_RE = re.compile(
    r"\b(solve|integral|derivative|equation|prove|theorem|matrix|probability|"
    r"factorial|calculate|logarithm|polynomial)\b|\d+\s*[\^*/×÷]\s*\d+", re.I)
_STRONG_REASON = ("step by step", "step-by-step", "think step", "chain of thought",
                  "reason through", "in great detail", "comprehensive analysis",
                  "walk me through")
_REASON_WORDS = ("explain why", "compare", "trade-off", "tradeoff", "pros and cons",
                 "architecture", "design a", "strategy", "analyze", "evaluate",
                 "optimi", "in detail", "root cause")
_TRIVIAL_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|ok|okay|cool|nice|lol|"
    r"good morning|good night|how are you|what'?s up|whats up)\b", re.I)


def classify_prompt(messages):
    """Return (target, reason): target in {'cloud','local'}, reason a short tag string."""
    text = "\n".join((m.get("content") or "") for m in (messages or [])
                     if isinstance(m, dict)).strip()
    n = len(text)
    if n == 0:
        return "local", "empty"
    low = text.lower()
    score, reasons = 0, []

    if n > config.AUTO_SMART_LONG_CHARS:
        score += 2; reasons.append("long")
    elif n > 400:
        score += 1; reasons.append("medium")

    if _CODE_RE.search(text) or any(w in low for w in _CODE_WORDS):
        score += 2; reasons.append("code")
    if _MATH_RE.search(text):
        score += 1; reasons.append("math")
    if any(w in low for w in _STRONG_REASON):
        score += 2; reasons.append("reasoning")
    elif any(w in low for w in _REASON_WORDS):
        score += 1; reasons.append("reasoning")
    if text.count("?") >= 3:
        score += 1; reasons.append("multi-Q")

    # trivial chit-chat / tiny asks stay local no matter what
    if n <= 120 and _TRIVIAL_RE.search(text):
        return "local", "short/simple"
    if n <= 40 and score == 0:
        return "local", "short"
    if score >= 2:
        return "cloud", "+".join(reasons)
    return "local", ("+".join(reasons) if reasons else "simple")


def _auto_decision(local, cloud, fast, messages):
    """The shared 'auto' brain -> (go_cloud, reason). One definition so route() and
    route_chain() can never disagree about where 'auto' sends a request."""
    if fast:
        return True, "speed"
    if not local.available():
        return True, "local-down"
    if config.AUTO_SMART and messages and cloud.available():
        target, why = classify_prompt(messages)
        return (target == "cloud"), "smart:" + why
    return False, ""


def route(logical_model=None, mode="auto", fast=False, messages=None):
    """Return (backend, concrete_model_id, decision_str). Single-backend choice
    (used by the CLI); the server uses route_chain() for failover."""
    name, spec = resolve_model(logical_model)
    local = LocalBackend()
    cloud = CloudBackend()

    if mode == "local":
        return local, spec["local"], f"local ({spec['local']})"

    if mode == "cloud":
        if not cloud.available():
            raise RuntimeError("cloud mode requested but no provider API key is set")
        cid = config.cloud_id(spec, cloud.provider)
        return cloud, cid, f"cloud:{cloud.provider} ({cid})"

    # auto
    go_cloud, reason = _auto_decision(local, cloud, fast, messages)
    if go_cloud and cloud.available():
        cid = config.cloud_id(spec, cloud.provider)
        return cloud, cid, f"cloud:{cloud.provider} ({cid}) [auto:{reason}]"
    if go_cloud and not cloud.available():
        return local, spec["local"], f"local ({spec['local']}) [auto: wanted cloud, no key -> local]"
    return local, spec["local"], f"local ({spec['local']})" + (f" [auto:{reason}]" if reason else " [auto]")


def route_chain(logical_model=None, mode="auto", fast=False, messages=None):
    """The ordered list of (backend, concrete_id, label) to try for this request:
    the routed primary FIRST, then remaining cloud providers, then local as a last
    resort -- so a single dead brain never fails the whole request (never-dark).

    'local' mode stays local ONLY (privacy is the point of asking for local)."""
    name, spec = resolve_model(logical_model)
    lb = LocalBackend()
    local_id = spec["local"]

    def cloud_entries(tag):
        out = []
        for pname, base, key in config.cloud_providers():
            cid = config.cloud_id(spec, pname)
            lbl = f"cloud:{pname} ({cid})" + (f" [{tag}]" if tag else "")
            out.append((CloudBackend(pname, base, key), cid, lbl))
        return out

    if mode == "local":
        return [(lb, local_id, f"local ({local_id})")]
    if mode == "cloud":
        return cloud_entries("cloud") + [(lb, local_id, f"local ({local_id}) [never-dark]")]

    # auto
    go_cloud, reason = _auto_decision(lb, CloudBackend(), fast, messages)
    tag = "auto" + (":" + reason if reason else "")
    if go_cloud:
        return cloud_entries(tag) + [(lb, local_id, f"local ({local_id}) [{tag} -> local fallback]")]
    return [(lb, local_id, f"local ({local_id}) [{tag}]")] + cloud_entries("auto:fallback")


def complete(chain, messages, opts, log=None):
    """NON-stream with full failover: try each backend in order, buffering the whole
    answer, falling over on ANY error or an empty result. Returns (text, label).
    Nothing is on the wire yet, so we can retry freely across the whole chain."""
    errs = []
    for backend, concrete, label in chain:
        try:
            text = "".join(backend.chat(messages, concrete, stream=False, **opts))
        except BackendError as e:
            errs.append(f"{label}: {e}")
            if log:
                log(f"failover: {label} FAILED ({e})")
            continue
        if not text.strip():
            errs.append(f"{label}: empty")
            if log:
                log(f"failover: {label} returned empty, trying next")
            continue
        if log:
            log(f"served by {label}")
        return text, label
    raise BackendError("all backends failed -> " + " | ".join(errs))


def stream_with_failover(chain, messages, opts, log=None):
    """STREAM with failover BEFORE the first token: try each backend; the first to emit
    a token wins and we commit to it (bytes already streamed can't be un-sent). Yields
    ('meta', label) once, then ('chunk', text)...; raises BackendError if all fail."""
    errs = []
    for backend, concrete, label in chain:
        try:
            it = iter(backend.chat(messages, concrete, stream=True, **opts))
            first = next(it)              # triggers the HTTP call; may raise BackendError
        except BackendError as e:
            errs.append(f"{label}: {e}")
            if log:
                log(f"failover: {label} FAILED ({e})")
            continue
        except StopIteration:
            errs.append(f"{label}: empty")
            if log:
                log(f"failover: {label} returned empty, trying next")
            continue
        if log:
            log(f"committed to {label}")
        yield ("meta", label)
        if first:
            yield ("chunk", first)
        for chunk in it:                 # a mid-stream error here propagates (committed)
            yield ("chunk", chunk)
        return
    raise BackendError("all backends failed -> " + " | ".join(errs))
