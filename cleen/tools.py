"""cleen.cpp computer-control tools -- the HANDS + EYES any model gets through cleen.

A raw model (local or cloud) is pure text: it can describe opening a browser but it
cannot DO it. cleen is a router, so we add the doing here, ONCE, and every model that
routes through cleen inherits it -- even a small 8B. That is the point the owner's own
model named ("I can't open a browser... I don't have access to your OS"): the access
lives in the tool layer, not in the weights.

Each tool wraps the SAME mechanism the rest of the machine already uses (os.startfile
for launching, a subprocess for the shell, a keyless web lookup), so this is not a new
control path -- it is the proven one, exposed to the model behind a clean name.

Portable by design: the CORE tools below are stdlib-only (Windows-first) so they ship
with the sellable product untouched. A machine can register EXTRA tools (creep, typing
into a specific window, vision) from a plugin dir via CLEEN_TOOLS_EXTRA_DIR without the
product ever depending on that personal code.

A tool is a function (args: dict) -> str (a short human/for-model-readable result).
Register more at runtime with tools.register(name, fn, description, schema).
"""
import os
import re
import json
import subprocess
import urllib.request
import urllib.parse

from . import config


# --------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------
_REGISTRY = {}   # name -> {"fn":..., "description":..., "schema":...}


def register(name, fn, description, schema=None):
    """Add/replace a tool. schema is a short {arg: "what it is"} map for the prompt."""
    _REGISTRY[name] = {"fn": fn, "description": description, "schema": schema or {}}


def names():
    return list(_REGISTRY)


def spec_for_prompt(allow=None):
    """A compact, model-readable description of the available tools (filtered by `allow`)."""
    lines = []
    for name, t in _REGISTRY.items():
        if allow is not None and name not in allow:
            continue
        args = ", ".join('"%s": <%s>' % (k, v) for k, v in t["schema"].items()) or ""
        lines.append('- %s {%s}\n    %s' % (name, args, t["description"]))
    return "\n".join(lines)


def dispatch(name, args, log=None):
    """Run a tool by name. NEVER raises: a failed tool returns an error string the model
    can read and recover from (a thrown exception would kill the whole agent turn)."""
    t = _REGISTRY.get(name)
    if not t:
        return "ERROR: no such tool '%s'. Available: %s" % (name, ", ".join(_REGISTRY) or "(none)")
    try:
        out = t["fn"](args or {})
        res = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    except Exception as e:
        res = "ERROR running %s: %s: %s" % (name, type(e).__name__, e)
    if log:
        log("tool %s(%s) -> %s" % (name, _short(args), _short(res, 180)))
    _audit(name, args, res)
    return res[: config.TOOLS_MAX_RESULT]


def _short(x, n=120):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def _audit(name, args, result):
    """Append every tool action to an audit log so the owner can always see what his
    models did on his machine -- transparency IS the safety floor for 'whole computer'."""
    path = config.TOOLS_LOG
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": int(__import__("time").time()), "tool": name,
                                "args": args, "result": _short(result, 400)},
                               ensure_ascii=False) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------------
# CORE tools (stdlib only, Windows-first) -- the sellable baseline.
# --------------------------------------------------------------------------------
def _open_url(args):
    url = (args.get("url") or args.get("u") or "").strip()
    if not url:
        return "ERROR: open_url needs a 'url'."
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url            # bare 'youtube.com' -> https://youtube.com
    os.startfile(url)                      # the proven Windows launch (same as desktop_mcp)
    return "Opened %s in the default browser." % url


def _open_app(args):
    name = (args.get("name") or args.get("app") or args.get("path") or "").strip()
    if not name:
        return "ERROR: open_app needs a 'name' (e.g. 'notepad', 'calc', 'msedge', or a full path)."
    os.startfile(name)
    return "Launched %s." % name


_CMD_DENY = [re.compile(p, re.I) for p in config.TOOLS_CMD_DENY]


def _run_command(args):
    cmd = (args.get("command") or args.get("cmd") or "").strip()
    if not cmd:
        return "ERROR: run_command needs a 'command'."
    for pat in _CMD_DENY:
        if pat.search(cmd):
            return ("REFUSED: that command matches a catastrophic/irreversible pattern "
                    "(disk wipe class) and is blocked. Everything else is allowed.")
    shell = args.get("shell") or "cmd"     # cmd | powershell
    try:
        timeout = int(args.get("timeout") or config.TOOLS_CMD_TIMEOUT)
    except (TypeError, ValueError):
        timeout = config.TOOLS_CMD_TIMEOUT
    if shell == "powershell":
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
    else:
        argv = ["cmd", "/c", cmd]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return "Command timed out after %ss (still may be running detached)." % timeout
    out = (p.stdout or "") + (("\n[stderr] " + p.stderr) if p.stderr.strip() else "")
    out = out.strip() or "(no output)"
    return "exit %s\n%s" % (p.returncode, out)


def _web_search(args):
    """Keyless web lookup so a model can 'look something up' itself. DuckDuckGo Instant
    Answer (JSON) first -> HTML results scrape as fallback. Returns top snippets as text."""
    q = (args.get("query") or args.get("q") or "").strip()
    if not q:
        return "ERROR: web_search needs a 'query'."
    # 1) Instant Answer API (stable, keyless, JSON)
    try:
        u = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": q, "format": "json", "no_html": 1, "no_redirect": 1})
        req = urllib.request.Request(u, headers={"User-Agent": "cleen.cpp/agent"})
        d = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "replace"))
        bits = []
        if d.get("AbstractText"):
            bits.append(d["AbstractText"] + (" (%s)" % d["AbstractURL"] if d.get("AbstractURL") else ""))
        for t in (d.get("RelatedTopics") or [])[:6]:
            if isinstance(t, dict) and t.get("Text"):
                bits.append("- " + t["Text"] + (" <%s>" % t["FirstURL"] if t.get("FirstURL") else ""))
        if bits:
            return ("Results for %r:\n" % q) + "\n".join(bits)
    except Exception:
        pass
    # 2) HTML results fallback
    try:
        u = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 cleen.cpp"})
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "replace")
        hits = re.findall(r'result__a[^>]*>(.*?)</a>.*?result__snippet[^>]*>(.*?)</a>', html, re.S)
        out = []
        for title, snip in hits[:6]:
            out.append("- %s: %s" % (_strip_html(title), _strip_html(snip)))
        if out:
            return ("Results for %r:\n" % q) + "\n".join(out)
    except Exception:
        pass
    # Direct fetch blocked (often a VPN/proxy resetting the connection). Don't dead-end:
    # tell the model it can still SEE the results by opening the search page in the browser.
    return ("web_search couldn't fetch results directly (network/VPN may be blocking it). "
            "Use open_url with this to see them in the browser instead: "
            "https://duckduckgo.com/?q=" + urllib.parse.quote(q))


def _strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


_YT_RESULTS_URL = "https://www.youtube.com/results?search_query="
_YT_ID = re.compile(r'"videoId"\s*:\s*"([\w-]{11})"')


def _youtube_play(args):
    """PLAY the real top YouTube hit for a query.

    Asked to "play the first song", a model will INVENT an 11-character video id, open a dead
    video, and report success -- the id looks perfectly well-formed, so nothing catches it. So
    this reads the actual results page and plays the FIRST REAL videoId on it. Keyless (no Data
    API key). If the page can't be fetched or parsed it opens the results page and SAYS nothing
    is playing, instead of claiming a win it can't verify."""
    q = (args.get("query") or args.get("q") or "").strip()
    if not q:
        return "ERROR: youtube_play needs a 'query'."
    results = _YT_RESULTS_URL + urllib.parse.quote(q)
    try:
        req = urllib.request.Request(results, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9"})
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
    except Exception as e:
        os.startfile(results)
        return ("Couldn't read YouTube to pick the top video (%s), so NOTHING is playing. "
                "Opened the search results for %r -- click the first one." % (type(e).__name__, q))
    m = _YT_ID.search(html)
    if not m:
        os.startfile(results)
        return ("Opened the YouTube results for %r, but couldn't identify the top video, so "
                "NOTHING is playing yet -- click the first one." % q)
    vid = m.group(1)
    tm = re.search(r'"videoId"\s*:\s*"%s".{0,400}?"text"\s*:\s*"([^"]{2,120})"' % re.escape(vid),
                   html, re.S)
    title = tm.group(1) if tm else ""
    os.startfile("https://www.youtube.com/watch?v=" + vid)
    return ("Started the top YouTube result for %r%s (watch?v=%s)."
            % (q, (': "%s"' % title) if title else "", vid))


_SENDKEYS_ESC = {"{": "{{}", "}": "{}}", "+": "{+}", "^": "{^}", "%": "{%}",
                 "~": "{~}", "(": "{(}", ")": "{)}", "[": "{[}", "]": "{]}"}


def _type_text(args):
    """Type into the CURRENTLY FOCUSED window via Windows SendKeys (self-contained, no
    extra deps). The model usually open_app's/clicks first, then types here."""
    text = args.get("text")
    if not text:
        return "ERROR: type_text needs 'text'."
    text = str(text)
    enter = bool(args.get("enter"))
    esc = "".join(_SENDKEYS_ESC.get(c, c) for c in text)
    ps = ("Add-Type -AssemblyName System.Windows.Forms; Start-Sleep -Milliseconds 250; "
          "[System.Windows.Forms.SendKeys]::SendWait(%s)" % _ps_quote(esc))
    if enter:
        ps += "; [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')"
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, text=True, timeout=30)
    return "Typed %d chars into the focused window%s." % (len(text), " + Enter" if enter else "")


def _ps_quote(s):
    return "'" + s.replace("'", "''") + "'"


def _screenshot(args):
    """Capture the screen to a PNG (self-contained .NET). Returns the saved path. A model
    with no vision can't read it alone; a vision extension (registered on this machine)
    turns it into text -- but the capture itself is core + portable."""
    # A model will happily invent a path under some OTHER user's profile ("C:/Users/<name>/
    # Pictures/..."). Writing there fails and the capture silently doesn't exist, so the model
    # then "verifies" the screen from a screenshot that was never taken. Only honour a caller
    # path whose folder actually exists; otherwise fall back to TEMP, which always works.
    out = args.get("path")
    if not out or not os.path.isdir(os.path.dirname(os.path.abspath(out))):
        out = os.path.join(os.environ.get("TEMP", "."),
                          "cleen_shot_%d.png" % int(__import__("time").time()))
    ps = ("Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
          "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen; "
          "$bmp=New-Object System.Drawing.Bitmap($b.Width,$b.Height); "
          "$g=[System.Drawing.Graphics]::FromImage($bmp); "
          "$g.CopyFromScreen($b.Left,$b.Top,0,0,$bmp.Size); "
          "$bmp.Save(%s);" % _ps_quote(out))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, text=True, timeout=30)
    ok = os.path.isfile(out)
    return ("Saved a full-screen screenshot to %s" % out) if ok else "screenshot failed."


def _list_dir(args):
    path = args.get("path") or "."
    try:
        entries = sorted(os.listdir(path))
        return "%s (%d):\n%s" % (path, len(entries), "\n".join(entries[:200]))
    except Exception as e:
        return "list_dir failed: %s" % e


def _read_file(args):
    path = args.get("path")
    if not path:
        return "ERROR: read_file needs a 'path'."
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = f.read(int(config.TOOLS_MAX_RESULT))
        return data or "(empty file)"
    except Exception as e:
        return "read_file failed: %s" % e


def _register_core():
    register("open_url", _open_url,
             "Open a website/URL in the owner's default browser. Use this to open YouTube, a site, a search page, anything with a URL.",
             {"url": "full or bare url, e.g. https://youtube.com or youtube.com"})
    register("open_app", _open_app,
             "Launch a Windows app by name or full path (notepad, calc, msedge, explorer, or C:/path/app.exe).",
             {"name": "app name or path"})
    register("run_command", _run_command,
             "Run a shell command on the owner's computer and get its output. Full control (cmd or powershell). Catastrophic disk-wipe commands are blocked.",
             {"command": "the command line", "shell": "cmd (default) or powershell", "timeout": "seconds (optional)"})
    register("youtube_play", _youtube_play,
             "PLAY a song/video on YouTube: finds the REAL top result for the query and opens it playing. Use this for 'play X' / 'play the first song' -- never guess a watch?v= id yourself.",
             {"query": "what to play, e.g. an artist or song name"})
    register("web_search", _web_search,
             "Search the web and get text results back, so you can look something up yourself before answering.",
             {"query": "what to search for"})
    register("type_text", _type_text,
             "Type text into the currently focused window (open/click the target first). Set enter:true to press Enter after.",
             {"text": "what to type", "enter": "true to press Enter after (optional)"})
    register("screenshot", _screenshot,
             "Capture the whole screen to a PNG file (see what is on screen). Returns the file path.",
             {"path": "optional output .png path"})
    register("list_dir", _list_dir, "List the files in a folder.", {"path": "folder path"})
    register("read_file", _read_file, "Read a text file's contents.", {"path": "file path"})


def _load_extensions():
    """Load extra tools a machine registers privately (creep, window-typing, vision...) from
    CLEEN_TOOLS_EXTRA_DIR. Each .py there may define register_cleen_tools(register) and we
    call it. This keeps personal tools OFF the sellable product while still arming this box."""
    d = config.TOOLS_EXTRA_DIR
    if not d or not os.path.isdir(d):
        return
    import importlib.util
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".py"):
            continue
        try:
            spec = importlib.util.spec_from_file_location("cleen_ext_" + fn[:-3], os.path.join(d, fn))
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "register_cleen_tools"):
                mod.register_cleen_tools(register)
        except Exception:
            continue


_register_core()
_load_extensions()
