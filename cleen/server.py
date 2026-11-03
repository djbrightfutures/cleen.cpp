"""cleen.cpp server -- an OpenAI-compatible API in front of the smart router.

This is "our own API server services": any OpenAI SDK / app / editor points at it,
and cleen decides local vs cloud per request. Point a tool at http://host:port/v1
with any api_key and it just works.

  GET  /health                    (now also reports response-cache stats)
  GET  /v1/models
  POST /v1/chat/completions       (stream + non-stream)

Plus a local-runtime-compatible shim (/api/generate, /api/chat, /api/tags, /api/ps).

Every completion path runs through the smart driver the way a raw engine never does:
  * RESPONSE CACHE   -- identical request -> stored answer in ms (X-Cleen-Cache: hit)
  * SMART ROUTING    -- 'auto' sends easy prompts local, hard ones to fast cloud
  * PROVIDER FAILOVER-- one dead brain falls through to the next, then to local

Route control per request (both optional):
  - header  X-Cleen-Mode: local | cloud | auto | fast   (default from CLEEN_MODE)
  - body    "cleen_fast": true                            (auto, prefer cloud for speed)
  - body    "no_cache": true                              (skip the response cache)
Pure stdlib http.server so it runs anywhere.
"""
import json
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, router, cache, compute, agent, __version__
from .engine import BackendError


def _now():
    return int(time.time())


class Handler(BaseHTTPRequestHandler):
    server_version = "cleen.cpp/" + __version__

    def log_message(self, *a):
        pass  # quiet (we log our own route/failover/cache decisions via _log)

    def _log(self, msg):
        """Surface a routing / failover / cache decision on stderr (-> serve log)."""
        try:
            sys.stderr.write("[cleen] " + msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    # ---- helpers ----
    def _json(self, code, obj, extra_headers=None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("content-length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _sse_open(self, extra_headers=None):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("access-control-allow-origin", "*")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _sse(self, obj):
        self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode("utf-8"))
        self.wfile.flush()

    # ---- routes ----
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "*")
        self.send_header("access-control-allow-methods", "*")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            return self._json(200, {"ok": True, "service": "cleen.cpp", "version": __version__,
                                    "cache": cache.stats(), "compute": compute.stats()})
        if self.path.rstrip("/") == "/v1/models":
            data = [{"id": n, "object": "model", "owned_by": "cleen.cpp",
                     "cleen": {"local": s["local"], "cloud": s["cloud"]}}
                    for n, s in config.MODELS.items()]
            return self._json(200, {"object": "list", "data": data})
        if self.path.rstrip("/") in ("/api/tags", "/api/ps", "/api/version"):
            return self._local_proxy_get(self.path.rstrip("/"))
        # THE .rea LIBRARY, searchable from here (the owner, 2026-10-15:
        # "in our cleen.cpp we can just search this library and click to use
        # it"). IMPORTED, never re-implemented: bridge28 and bridge128 read the
        # same rea_lib, so there is ONE index and ONE GPU check rather than
        # three copies free to drift.
        #   GET /rea            -> every entry (compact rows)
        #   GET /rea?q=ghidra   -> search
        #   GET /rea/get?name=X -> one full entry
        #   GET /rea/gpu        -> measured VRAM + what is resident
        if self.path.split("?")[0].rstrip("/") in ("/rea", "/rea/get", "/rea/gpu"):
            return self._rea_get()
        # MT8 CHART - local only, for testing his own expert against his own bars.
        if self.path.split("?")[0].rstrip("/") in ("/mt8", "/mt8/bars"):
            return self._mt8_get()
        return self._json(404, {"error": {"message": "not found"}})

    def _rea_ui(self, L, q):
        """Render the .rea library as a page you can read and click."""
        import html as _h
        rows = L.search(q, limit=100)
        g = L.gpu_state()
        css = ("body{margin:0;background:#07090f;color:#e8ecf3;font:14.5px/1.55 "
               "system-ui,Segoe UI,sans-serif}.w{max-width:900px;margin:0 auto;"
               "padding:28px 18px 70px}h1{font-size:25px;margin:0 0 4px}"
               ".sub{opacity:.66;margin:0 0 20px;font-size:13.5px}"
               "form{margin:0 0 18px}input{background:#141b26;border:1px solid "
               "#2a3444;color:inherit;border-radius:10px;padding:10px 13px;"
               "width:min(380px,100%);font:inherit}"
               ".e{border:1px solid #2a3444;background:#141b26;border-radius:13px;"
               "padding:14px 16px;margin:0 0 11px}"
               ".e b{font-size:15.5px}.ok{color:#6ee7a8}.bad{color:#ffb4a2}"
               ".m{opacity:.7;font-size:13px;margin-top:5px;word-break:break-all}"
               "a{color:#5bc8ff}.g{border:1px solid #2a3444;border-radius:11px;"
               "padding:11px 14px;margin:0 0 20px;font-size:13px;opacity:.85;"
               "background:rgba(255,255,255,.03)}.n{opacity:.6;font-size:12.5px}")
        bits = ["<!doctype html><meta charset=utf-8>",
                "<meta name=viewport content='width=device-width,initial-scale=1'>",
                "<title>Library</title><style>%s</style><div class=w>" % css,
                "<h1>Analysis library</h1>",
                "<p class=sub>%d entr%s. A refusal is a valid entry &mdash; a library "
                "that hides its failures reports a cleaner world than exists.</p>"
                % (len(rows), "y" if len(rows) == 1 else "ies")]
        if g.get("measured"):
            bits.append("<div class=g>GPU %s / %s MB used &middot; <b>%s MB free</b>%s</div>"
                        % (g["used_mb"], g["total_mb"], g["free_mb"],
                           "".join(" &middot; resident: %s (%s MB)"
                                   % (_h.escape(str(r["name"])), r["vram_mb"])
                                   for r in g.get("resident", []))))
        else:
            bits.append("<div class=g>GPU: UNMEASURABLE (%s) &mdash; not read as empty.</div>"
                        % _h.escape(str(g.get("why", ""))))
        bits.append("<form method=get action=/rea><input name=q placeholder='search the library'"
                    " value='%s'><input type=hidden name=ui value=1></form>" % _h.escape(q or ""))
        if not rows:
            bits.append("<div class=e>Nothing matches. <a href='/rea?ui=1'>show all</a></div>")
        for r in rows:
            ok = r.get("status") == "ok"
            bits.append("<div class=e><b>%s</b> <span class=%s>%s</span>%s"
                        "<div class=m>%s</div><div class=n>%s &middot; %ss &middot; "
                        "<a href='/rea/get?name=%s'>open the evidence</a></div></div>"
                        % (_h.escape(str(r.get("name"))),
                           "ok" if ok else "bad", _h.escape(str(r.get("status"))),
                           (" &middot; %s" % _h.escape(str(r.get("provider"))))
                           if r.get("provider") else "",
                           _h.escape(str(r.get("target") or "")),
                           _h.escape(str(r.get("reason") or
                                          ("%s nodes / %s unknowns" % (r.get("nodes"),
                                           r.get("unknowns")) if r.get("nodes")
                                           else "%s limitation(s) declared"
                                           % r.get("limitations")))),
                           r.get("elapsed_s"), _h.escape(str(r.get("name")))))
        bits.append("</div>")
        body = "".join(bits).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mt8_get(self):
        """MT8 Chart. /mt8 renders the page, /mt8/bars serves the series."""
        import os as _os
        import sys as _sys
        import urllib.parse as _up
        LIB = r"A:\TRADING\MT8"
        parts = _up.urlsplit(self.path)
        route = parts.path.rstrip("/") or "/mt8"
        qs = _up.parse_qs(parts.query or "")
        if not _os.path.isdir(LIB):
            return self._json(503, {"error": {"message": "no MT8 at %s" % LIB}})
        if LIB not in _sys.path:
            _sys.path.insert(0, LIB)
        try:
            import mt8_chart as C
        except Exception as e:
            return self._json(503, {"error": {"message": "mt8_chart unavailable: %s: %s"
                                              % (type(e).__name__, e)}})
        if route == "/mt8/bars":
            sym = (qs.get("sym") or ["XAUUSD"])[0]
            n = int((qs.get("n") or ["700"])[0])
            def _i(k, d):
                try:
                    return int((qs.get(k) or [str(d)])[0])
                except ValueError:
                    return d
            return self._json(200, C.series(sym, n=n, fast=_i("fast", 5),
                                            mid=_i("mid", 20), slow=_i("slow", 200),
                                            atr_n=_i("atr", 14)))
        syms = C.symbols()
        opts = "".join("<option%s>%s</option>"
                       % (" selected" if s == "XAUUSD" else "", s) for s in syms)
        page = MT8_PAGE.replace("__OPTS__", opts)
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _rea_get(self):
        """Serve the .rea library. Lazy import so cleen never pays for it, and
        a MISSING library answers 503 with the reason rather than an empty
        list -- 'no entries' and 'no library' are different facts."""
        import os
        import sys
        import urllib.parse as _up
        LIB = r"C:\Users\user\REA_LIBRARY"
        parts = _up.urlsplit(self.path)
        route = parts.path.rstrip("/") or "/rea"
        qs = _up.parse_qs(parts.query or "")
        if not os.path.isdir(LIB):
            return self._json(503, {"error": {"message": "no .rea library at %s" % LIB}})
        if LIB not in sys.path:
            sys.path.insert(0, LIB)
        try:
            import rea_lib as L
        except Exception as e:
            return self._json(503, {"error": {"message": "rea_lib unavailable: %s: %s"
                                              % (type(e).__name__, e)}})
        if route == "/rea/gpu":
            return self._json(200, L.gpu_state())
        # A HUMAN-VIEWABLE LIBRARY. the owner asked to "search this library and
        # click to use it" and the first cut shipped JSON only -- which is why
        # he could not SEE it. Served when the client asks for HTML, so the
        # JSON API is unchanged for every programmatic caller.
        wants_html = "text/html" in (self.headers.get("Accept") or "")
        if route == "/rea" and (wants_html or "ui" in qs):
            return self._rea_ui(L, (qs.get("q") or [""])[0])
        if route == "/rea/get":
            name = (qs.get("name") or [""])[0]
            ent = L.get(name) if name else None
            if ent is None:
                return self._json(404, {"error": {"message": "no entry %r" % name}})
            return self._json(200, ent)
        rows = L.search((qs.get("q") or [""])[0],
                        limit=int((qs.get("limit") or ["25"])[0]))
        return self._json(200, {"object": "list", "library": LIB,
                                "count": len(rows), "data": rows})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            n = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})
        if path == "/api/generate":
            return self._local_handle(body, is_chat=False)
        if path == "/api/chat":
            return self._local_handle(body, is_chat=True)
        if path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})

        messages = body.get("messages") or []
        model = body.get("model") or config.DEFAULT_MODEL
        stream = bool(body.get("stream"))
        raw_mode = self.headers.get("X-Cleen-Mode") or body.get("cleen_mode") or config.DEFAULT_MODE
        mode, mode_fast = config.split_mode(raw_mode)
        fast = mode_fast or bool(body.get("cleen_fast"))
        temperature = body.get("temperature", config.ENGINE_DEFAULT_TEMP)
        max_tokens = body.get("max_tokens")
        no_cache = bool(body.get("no_cache") or body.get("cleen_no_cache"))

        cid = "chatcmpl-cleen-%d" % _now()
        opts = {"temperature": temperature}
        if max_tokens:
            opts["max_tokens"] = max_tokens

        # AGENT layer: give THIS request real computer control (open browser/apps, run
        # commands, web search, type, screenshot). Opt-in per request so the plain API
        # never regresses: body cleen_tools:true / header X-Cleen-Tools / mode 'agent'
        # (or the install-wide CLEEN_TOOLS default).
        want_tools = (config.TOOLS_DEFAULT or bool(body.get("cleen_tools"))
                      or str(self.headers.get("X-Cleen-Tools") or "").strip().lower()
                      in ("1", "true", "on", "yes")
                      or raw_mode.strip().lower() in ("agent", "tools"))
        if want_tools:
            return self._serve_agent(cid, model, messages, mode, fast, stream)

        # --- COMPUTE lane: exact work (arithmetic / numeric comparison) answered by code
        # BEFORE any model -- instant, 0 tokens, always correct. Returns None for anything
        # fuzzy, so a normal prompt is never hijacked (see compute.try_compute). ---
        if config.COMPUTE_ON and not body.get("cleen_no_compute"):
            computed = compute.try_compute(messages)
            if computed is not None:
                self._log(f"COMPUTE /v1 model={model} -> {computed!r}")
                if stream:
                    self._sse_open(extra_headers={"x-cleen-route": "compute"})
                    self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                               "model": model, "cleen": {"route": "compute", "cache": "n/a"},
                               "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                    for piece in cache.replay_chunks(computed):
                        self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                                   "model": model,
                                   "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                    self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                               "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                return self._json(200, {
                    "id": cid, "object": "chat.completion", "created": _now(), "model": model,
                    "cleen": {"route": "compute", "cache": "n/a"},
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": computed},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                }, extra_headers={"x-cleen-route": "compute"})

        # --- response cache: exact-match key, computed with ZERO network so a hit is instant ---
        key = cache.make_key(model, messages, opts, mode, fast) \
            if cache.cacheable(temperature, no_cache) else None
        hit = cache.get(key) if key else None
        if hit is not None:
            self._log(f"cache HIT /v1 model={model} [{mode}{' fast' if fast else ''}]")
            if stream:
                self._sse_open(extra_headers={"x-cleen-cache": "hit"})
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                           "model": model, "cleen": {"route": "cache", "cache": "hit"},
                           "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                for piece in cache.replay_chunks(hit):
                    self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                               "model": model,
                               "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                           "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": _now(), "model": model,
                "cleen": {"route": "cache", "cache": "hit"},
                "choices": [{"index": 0, "message": {"role": "assistant", "content": hit},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            }, extra_headers={"x-cleen-cache": "hit"})

        # --- miss: build the failover chain (smart primary + never-dark fallbacks) ---
        try:
            chain = router.route_chain(model, mode=mode, fast=fast, messages=messages)
        except Exception as e:
            return self._json(400, {"error": {"message": str(e)}})

        if stream:
            self._sse_open(extra_headers={"x-cleen-cache": "miss"})
            acc = []
            try:
                for kind, val in router.stream_with_failover(chain, messages, opts, log=self._log):
                    if kind == "meta":
                        self._log(f"route /v1 model={model} -> {val}")
                        self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                                   "model": model, "cleen": {"route": val, "cache": "miss"},
                                   "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                "finish_reason": None}]})
                    else:
                        acc.append(val)
                        self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                                   "model": model,
                                   "choices": [{"index": 0, "delta": {"content": val},
                                                "finish_reason": None}]})
            except BackendError as e:
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                           "model": model,
                           "choices": [{"index": 0, "delta": {"content": f"\n[cleen backend error: {e}]"},
                                        "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                       "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            if key and acc:
                cache.put(key, "".join(acc))
            return

        # non-stream: collect with full failover
        try:
            text, label = router.complete(chain, messages, opts, log=self._log)
        except BackendError as e:
            return self._json(502, {"error": {"message": str(e)}})
        if key:
            cache.put(key, text)
        return self._json(200, {
            "id": cid, "object": "chat.completion", "created": _now(), "model": model,
            "cleen": {"route": label, "cache": "miss"},
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }, extra_headers={"x-cleen-cache": "miss"})

    # ---- agent layer (computer control) --------------------------------------------------
    def _serve_agent(self, cid, model, messages, mode, fast, stream):
        """Run the think->act->observe loop for this request and return it as a normal
        completion. The model can open the browser, run commands, search, type, see the
        screen -- so it ACTS like a full assistant instead of saying it can't."""
        self._log(f"AGENT model={model} tools=on [{mode}{' fast' if fast else ''}]")
        try:
            res = agent.run(messages, model=model, mode=mode, fast=fast, log=self._log)
        except Exception as e:
            return self._json(502, {"error": {"message": f"agent error: {e}"}})
        text = res.get("text") or ""
        meta = {"route": f"agent({res['steps']} steps)->{res.get('route', '')}",
                "tools": res.get("actions", [])}
        if stream:
            self._sse_open(extra_headers={"x-cleen-cache": "agent"})
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                       "model": model, "cleen": meta,
                       "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            for piece in cache.replay_chunks(text):
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                           "model": model,
                           "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": _now(),
                       "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        return self._json(200, {
            "id": cid, "object": "chat.completion", "created": _now(), "model": model,
            "cleen": meta,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }, extra_headers={"x-cleen-cache": "agent"})

    # ---- local-runtime-compatible shim ----------------------------------------------------------
    # Many local tools call the runtime's /api/generate + /api/chat at :11434. This shim speaks
    # that exact API in front of the smart router, so any such tool can point at cleen.cpp
    # with NO code change and transparently burst to a fast cloud provider when local is slow
    # -- with a LOCAL fallback so the brain is never dark. /api/tags + /api/ps proxy the real
    # local runtime so model listings stay truthful.
    def _iso(self):
        return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    def _ndjson_open(self, extra_headers=None):
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("access-control-allow-origin", "*")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _ndjson(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _local_proxy_get(self, path):
        try:
            with urllib.request.urlopen(config.LOCAL_BASE.rstrip("/") + path, timeout=8) as r:
                raw = r.read()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("access-control-allow-origin", "*")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except Exception:
            # local runtime down: still answer so a caller doesn't crash on model discovery
            if path == "/api/tags":
                return self._json(200, {"models": [{"name": n, "model": n} for n in config.MODELS]})
            if path == "/api/ps":
                return self._json(200, {"models": []})
            return self._json(200, {"version": "cleen.cpp-" + __version__})

    def _shim_opts(self, body):
        o = body.get("options") or {}
        opts = {"temperature": o.get("temperature", body.get("temperature", config.ENGINE_DEFAULT_TEMP))}
        npred = o.get("num_predict") or body.get("max_tokens")
        try:
            if npred and int(npred) > 0:
                opts["max_tokens"] = int(npred)
        except (TypeError, ValueError):
            pass
        return opts

    def _shim_chain(self, model, body, messages):
        """Build the failover chain for the local shim -> (chain, mode, fast).
        An unknown raw model routes via 'cleen-chat' but keeps its EXACT id on any LOCAL
        hop (only a cloud burst needs a catalog substitution)."""
        raw_mode = self.headers.get("X-Cleen-Mode") or body.get("cleen_mode") or config.DEFAULT_MODE
        mode, mode_fast = config.split_mode(raw_mode)
        fast = mode_fast or bool(body.get("cleen_fast"))
        logical = model if model in config.MODELS else "cleen-chat"
        chain = router.route_chain(logical, mode=mode, fast=fast, messages=messages)
        if model not in config.MODELS:
            chain = [
                (b, (model if b.kind == "local" else c),
                 d + (f" [shim:local {model}]" if b.kind == "local" else ""))
                for (b, c, d) in chain
            ]
        return chain, mode, fast

    def _local_handle(self, body, is_chat):
        """Unified /api/generate + /api/chat handler with cache + failover."""
        model = body.get("model") or config.DEFAULT_MODEL
        if is_chat:
            messages = body.get("messages") or []
        else:
            system = body.get("system")
            messages = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": body.get("prompt") or ""}]
        opts = self._shim_opts(body)
        stream = bool(body.get("stream", False))
        raw_mode = self.headers.get("X-Cleen-Mode") or body.get("cleen_mode") or config.DEFAULT_MODE
        mode, mode_fast = config.split_mode(raw_mode)
        fast = mode_fast or bool(body.get("cleen_fast"))
        no_cache = bool(body.get("no_cache") or body.get("cleen_no_cache"))

        if config.COMPUTE_ON and not body.get("cleen_no_compute"):
            computed = compute.try_compute(messages)
            if computed is not None:
                self._log(f"COMPUTE /api/{'chat' if is_chat else 'generate'} model={model} -> {computed!r}")
                return self._local_serve_cached(model, computed, is_chat, stream)

        key = cache.make_key(model, messages, opts, mode, fast) \
            if cache.cacheable(opts.get("temperature"), no_cache) else None
        hit = cache.get(key) if key else None
        if hit is not None:
            self._log(f"cache HIT /api/{'chat' if is_chat else 'generate'} model={model}")
            return self._local_serve_cached(model, hit, is_chat, stream)

        try:
            chain, _, _ = self._shim_chain(model, body, messages)
        except Exception as e:
            return self._json(400, {"error": {"message": str(e)}})

        if stream:
            self._ndjson_open(extra_headers={"x-cleen-cache": "miss"})
            acc = []
            try:
                for kind, val in router.stream_with_failover(chain, messages, opts, log=self._log):
                    if kind == "meta":
                        self._log(f"route /api model={model} -> {val}")
                        continue  # the local protocol has no meta frame; route is logged
                    acc.append(val)
                    self._ndjson(self._local_frame(model, val, is_chat, done=False))
            except BackendError as e:
                self._ndjson(self._local_frame(model, f"[cleen: {e}]", is_chat, done=False))
                acc = []  # never cache an errored stream
            self._ndjson(self._local_frame(model, "", is_chat, done=True))
            if key and acc:
                cache.put(key, "".join(acc))
            return

        try:
            text, label = router.complete(chain, messages, opts, log=self._log)
        except BackendError as e:
            return self._json(502, {"error": {"message": str(e)}})
        if key:
            cache.put(key, text)
        resp = {"model": model, "created_at": self._iso(), "done": True, "done_reason": "stop",
                "cleen": {"route": label, "cache": "miss"}}
        if is_chat:
            resp["message"] = {"role": "assistant", "content": text}
        else:
            resp["response"] = text
        return self._json(200, resp, extra_headers={"x-cleen-cache": "miss"})

    def _local_frame(self, model, content, is_chat, done):
        """One local-runtime ndjson frame, chat- or generate-shaped."""
        f = {"model": model, "created_at": self._iso(), "done": done}
        if done:
            f["done_reason"] = "stop"
        if is_chat:
            f["message"] = {"role": "assistant", "content": content}
        else:
            f["response"] = content
        return f

    def _local_serve_cached(self, model, text, is_chat, stream):
        if not stream:
            resp = {"model": model, "created_at": self._iso(), "done": True, "done_reason": "stop",
                    "cleen": {"route": "cache", "cache": "hit"}}
            if is_chat:
                resp["message"] = {"role": "assistant", "content": text}
            else:
                resp["response"] = text
            return self._json(200, resp, extra_headers={"x-cleen-cache": "hit"})
        self._ndjson_open(extra_headers={"x-cleen-cache": "hit"})
        for piece in cache.replay_chunks(text):
            self._ndjson(self._local_frame(model, piece, is_chat, done=False))
        self._ndjson(self._local_frame(model, "", is_chat, done=True))


def run(host="127.0.0.1", port=8088):
    srv = ThreadingHTTPServer((host, port), Handler)
    prov, _, _ = config.active_provider()
    print(f"cleen.cpp {__version__} serving OpenAI-compatible API on http://{host}:{port}/v1")
    print(f"  local: {config.LOCAL_KIND} @ {config.LOCAL_BASE}   cloud: {config.PROVIDERS[prov]['label'] if prov else 'none'}")
    print(f"  cache: {'on' if config.CACHE_ON else 'off'} (ttl {config.CACHE_TTL}s, max {config.CACHE_MAX})"
          f"   smart-auto: {'on' if config.AUTO_SMART else 'off'}")
    print(f"  try:  curl http://{host}:{port}/health")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\ncleen.cpp server stopped")
        srv.shutdown()


# MT8 CHART PAGE. Canvas only -- no CDN, no webfont, nothing external: a tool
# you open to check an expert must not wait on a network it may not have.
MT8_PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>MT8 Chart</title><style>
*{box-sizing:border-box}body{margin:0;background:#07090f;color:#e8ecf3;
font:14px/1.5 system-ui,Segoe UI,sans-serif}
.w{max-width:1180px;margin:0 auto;padding:16px 14px 60px}
h1{font-size:19px;margin:0 0 3px}.sub{opacity:.6;font-size:12.5px;margin:0 0 14px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}
select,input{background:#141b26;border:1px solid #2a3444;color:inherit;
border-radius:9px;padding:7px 10px;font:inherit}input{width:66px}
button{background:#5bc8ff;color:#06121c;border:0;border-radius:9px;
padding:8px 15px;font-weight:650;cursor:pointer}
button[disabled]{opacity:.5;cursor:default}
.wrap{border:1px solid #2a3444;border-radius:13px;background:#0b1118;padding:10px}
canvas{width:100%;display:block;border-radius:8px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;opacity:.85;
margin:9px 2px 0}.k{display:inline-block;width:10px;height:10px;border-radius:2px;
margin-right:5px;vertical-align:-1px}
.note{opacity:.58;font-size:12px;margin:13px 2px 0;line-height:1.5}
.stat{opacity:.8;font-size:12.5px;margin:8px 2px 0}
</style><div class=w>
<h1>MT8 Chart</h1>
<p class=sub>Your cached bars with your expert's own moving averages and ATR on
top, and a marker wherever its entry rule fires. Local only.</p>
<div class=bar>
  <select id=sym>__OPTS__</select>
  <label>bars <input id=n value=700></label>
  <label>fast <input id=fast value=5></label>
  <label>mid <input id=mid value=20></label>
  <label>slow <input id=slow value=200></label>
  <label>atr <input id=atr value=14></label>
  <button id=go>Draw</button>
</div>
<div class=wrap><canvas id=cv height=430></canvas></div>
<div class=legend>
  <span><i class=k style=background:#44506a></i>high-low range</span>
  <span><i class=k style=background:#e8ecf3></i>close</span>
  <span><i class=k style=background:#5bc8ff></i>fast MA</span>
  <span><i class=k style=background:#ffd166></i>mid MA</span>
  <span><i class=k style=background:#b48cff></i>slow MA (trend filter)</span>
  <span><i class=k style=background:#6ee7a8></i>buy signal</span>
  <span><i class=k style=background:#ff7a6b></i>sell signal</span>
</div>
<div class=stat id=stat></div>
<p class=note id=note></p>
</div>
<script>
(function(){
 var cv=document.getElementById('cv'),cx=cv.getContext('2d'),go=document.getElementById('go');
 function v(id){return document.getElementById(id).value;}
 function fit(){var r=cv.getBoundingClientRect();
   var dpr=window.devicePixelRatio||1;cv.width=r.width*dpr;cv.height=430*dpr;
   cx.setTransform(dpr,0,0,dpr,0,0);return {w:r.width,h:430};}
 function line(pts,col,lw){cx.beginPath();cx.strokeStyle=col;cx.lineWidth=lw||1.2;
   var started=false;
   for(var i=0;i<pts.length;i++){var p=pts[i];if(!p){started=false;continue;}
     if(!started){cx.moveTo(p[0],p[1]);started=true;}else{cx.lineTo(p[0],p[1]);}}
   cx.stroke();}
 function draw(d){
   var g=fit(),W=g.w,H=g.h,PL=54,PR=10,PT=12,PB=26;
   cx.clearRect(0,0,W,H);
   var lo=Infinity,hi=-Infinity,i;
   for(i=0;i<d.c.length;i++){if(d.l[i]<lo)lo=d.l[i];if(d.h[i]>hi)hi=d.h[i];}
   var pad=(hi-lo)*0.04||1;lo-=pad;hi+=pad;
   var X=function(k){return PL+(W-PL-PR)*(k/(d.c.length-1||1));};
   var Y=function(p){return PT+(H-PT-PB)*(1-(p-lo)/(hi-lo||1));};
   // grid + price axis
   cx.strokeStyle='#19212e';cx.lineWidth=1;cx.fillStyle='#6b7687';
   cx.font='11px system-ui';cx.textAlign='right';
   for(i=0;i<=4;i++){var p=lo+(hi-lo)*i/4,y=Y(p);
     cx.beginPath();cx.moveTo(PL,y);cx.lineTo(W-PR,y);cx.stroke();
     cx.fillText(p.toFixed(p<10?4:2),PL-7,y+3.5);}
   // high-low range (NOT a candle - the cache has no open)
   cx.strokeStyle='#44506a';cx.lineWidth=Math.max(0.6,(W-PL-PR)/d.c.length*0.55);
   for(i=0;i<d.c.length;i++){var x=X(i);
     cx.beginPath();cx.moveTo(x,Y(d.h[i]));cx.lineTo(x,Y(d.l[i]));cx.stroke();}
   function mk(arr){var o=[];for(var j=0;j<arr.length;j++){
     o.push(arr[j]==null?null:[X(j),Y(arr[j])]);}return o;}
   line(mk(d.c),'#e8ecf3',1.1);
   line(mk(d.ma_fast),'#5bc8ff',1.3);
   line(mk(d.ma_mid),'#ffd166',1.3);
   line(mk(d.ma_slow),'#b48cff',1.5);
   // signals
   for(i=0;i<d.signals.length;i++){var s=d.signals[i],sx=X(s.i),sy=Y(s.px);
     cx.fillStyle=s.side==='buy'?'#6ee7a8':'#ff7a6b';
     cx.beginPath();cx.arc(sx,sy,3.6,0,6.3);cx.fill();
     cx.strokeStyle=cx.fillStyle;cx.lineWidth=1;
     cx.beginPath();cx.moveTo(sx,sy+(s.side==='buy'?9:-9));cx.lineTo(sx,sy);cx.stroke();}
   var b=d.signals.filter(function(s){return s.side==='buy';}).length;
   document.getElementById('stat').textContent =
     d.symbol+' '+d.tf+' - '+d.bars+' bars - '+d.signals.length+
     ' signals ('+b+' buy / '+(d.signals.length-b)+' sell) - last close '+
     d.c[d.c.length-1];
   document.getElementById('note').textContent = d.note+' '+d.signal_rule;
 }
 function load(){
   go.disabled=true;go.textContent='...';
   var u='/mt8/bars?sym='+encodeURIComponent(v('sym'))+'&n='+v('n')+
         '&fast='+v('fast')+'&mid='+v('mid')+'&slow='+v('slow')+'&atr='+v('atr');
   fetch(u).then(function(r){return r.json();}).then(function(d){
     go.disabled=false;go.textContent='Draw';
     if(d.error){document.getElementById('stat').textContent='ERROR: '+d.error;return;}
     draw(d);window.__d=d;
   }).catch(function(e){go.disabled=false;go.textContent='Draw';
     document.getElementById('stat').textContent='could not load: '+e;});
 }
 go.addEventListener('click',load);
 window.addEventListener('resize',function(){if(window.__d)draw(window.__d);});
 load();
})();
</script>"""
