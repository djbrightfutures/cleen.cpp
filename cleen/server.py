"""cleen.cpp server -- an OpenAI-compatible API in front of the smart router.

This is "our own API server services": any OpenAI SDK / app / editor points at it,
and cleen decides local vs cloud per request. Point a tool at http://host:port/v1
with any api_key and it just works.

  GET  /health                    (now also reports response-cache stats)
  GET  /v1/models
  POST /v1/chat/completions       (stream + non-stream)

Plus an Ollama-compatible shim (/api/generate, /api/chat, /api/tags, /api/ps).

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

from . import config, router, cache, __version__
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
                                    "cache": cache.stats()})
        if self.path.rstrip("/") == "/v1/models":
            data = [{"id": n, "object": "model", "owned_by": "cleen.cpp",
                     "cleen": {"local": s["local"], "cloud": s["cloud"]}}
                    for n, s in config.MODELS.items()]
            return self._json(200, {"object": "list", "data": data})
        if self.path.rstrip("/") in ("/api/tags", "/api/ps", "/api/version"):
            return self._ollama_proxy_get(self.path.rstrip("/"))
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            n = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})
        if path == "/api/generate":
            return self._ollama_handle(body, is_chat=False)
        if path == "/api/chat":
            return self._ollama_handle(body, is_chat=True)
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

    # ---- Ollama-compatible shim ----------------------------------------------------------
    # Many local tools call Ollama's /api/generate + /api/chat at :11434. This shim speaks
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

    def _ollama_proxy_get(self, path):
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
        """Build the failover chain for the Ollama shim -> (chain, mode, fast).
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

    def _ollama_handle(self, body, is_chat):
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

        key = cache.make_key(model, messages, opts, mode, fast) \
            if cache.cacheable(opts.get("temperature"), no_cache) else None
        hit = cache.get(key) if key else None
        if hit is not None:
            self._log(f"cache HIT /api/{'chat' if is_chat else 'generate'} model={model}")
            return self._ollama_serve_cached(model, hit, is_chat, stream)

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
                        continue  # the Ollama protocol has no meta frame; route is logged
                    acc.append(val)
                    self._ndjson(self._ollama_frame(model, val, is_chat, done=False))
            except BackendError as e:
                self._ndjson(self._ollama_frame(model, f"[cleen: {e}]", is_chat, done=False))
                acc = []  # never cache an errored stream
            self._ndjson(self._ollama_frame(model, "", is_chat, done=True))
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

    def _ollama_frame(self, model, content, is_chat, done):
        """One Ollama ndjson frame, chat- or generate-shaped."""
        f = {"model": model, "created_at": self._iso(), "done": done}
        if done:
            f["done_reason"] = "stop"
        if is_chat:
            f["message"] = {"role": "assistant", "content": content}
        else:
            f["response"] = content
        return f

    def _ollama_serve_cached(self, model, text, is_chat, stream):
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
            self._ndjson(self._ollama_frame(model, piece, is_chat, done=False))
        self._ndjson(self._ollama_frame(model, "", is_chat, done=True))


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
