"""cleen.cpp engine: two backends behind ONE interface.

LocalBackend  -> the free llama.cpp-family runtime on your machine (Ollama API by
                 default; a llama.cpp `llama-server` works too -- same OpenAI shape).
CloudBackend  -> a fast OpenAI-compatible provider (Cerebras, NVIDIA NIM, ...), the
                 tier we resell.

Both expose .chat(messages, model, stream) yielding text chunks, so the router and
the server never care which one ran. Pure stdlib (urllib) so cleen runs on anything.
"""
import json
import urllib.request
import urllib.error

from . import config


class BackendError(Exception):
    pass


def _post(url, payload, headers, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _wants_no_think(model):
    """qwen3 / deepseek-r1 emit 'thinking'. For a coder tool we want the answer,
    fast -- so disable it locally. (Ollama honors a top-level `think: false`.)"""
    m = (model or "").lower()
    return any(t in m for t in ("qwen3", "deepseek-r1", "-r1", "reason"))


def _openai_stream(url, key, model, messages, opts, timeout=None):
    """Stream an OpenAI-compatible /chat/completions endpoint (cloud, or llama-server).

    `timeout` bounds each socket read so a hung provider fails fast enough to fail over
    (defaults to the cloud timeout; a dead host is refused instantly regardless).
    """
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = "Bearer " + key
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": opts.get("temperature", config.ENGINE_DEFAULT_TEMP),
    }
    if opts.get("max_tokens"):
        payload["max_tokens"] = opts["max_tokens"]
    try:
        resp = _post(url, payload, headers, timeout=timeout or config.CLOUD_TIMEOUT)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise BackendError(f"HTTP {e.code} from {url}: {body}")
    except Exception as e:
        raise BackendError(f"{type(e).__name__}: {e}")
    for raw in resp:
        raw = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            o = json.loads(data)
            delta = o["choices"][0]["delta"].get("content")
            if delta:
                yield delta
        except Exception:
            continue


class LocalBackend:
    kind = "local"

    def __init__(self, base=None, local_kind=None):
        self.base = (base or config.LOCAL_BASE).rstrip("/")
        self.local_kind = local_kind or config.LOCAL_KIND
        self.label = "local (%s)" % self.local_kind

    def available(self):
        try:
            probe = "/api/tags" if self.local_kind == "ollama" else "/v1/models"
            urllib.request.urlopen(self.base + probe, timeout=4).read()
            return True
        except Exception:
            return False

    def list_models(self):
        try:
            if self.local_kind == "ollama":
                data = json.loads(urllib.request.urlopen(self.base + "/api/tags", timeout=6).read())
                return [m["name"] for m in data.get("models", [])]
            data = json.loads(urllib.request.urlopen(self.base + "/v1/models", timeout=6).read())
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def chat(self, messages, model, stream=True, **opts):
        timeout = opts.get("timeout") or config.LOCAL_TIMEOUT
        if self.local_kind != "ollama":
            yield from _openai_stream(self.base + "/v1/chat/completions", None, model,
                                      messages, opts, timeout=timeout)
            return
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": opts.get("temperature", config.ENGINE_DEFAULT_TEMP)},
        }
        if _wants_no_think(model):
            payload["think"] = False
        try:
            resp = _post(self.base + "/api/chat", payload, {"content-type": "application/json"},
                         timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            raise BackendError(f"HTTP {e.code} from local runtime: {body}")
        except Exception as e:
            raise BackendError(f"local runtime unreachable: {e}")
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            msg = o.get("message") or {}
            if msg.get("content"):
                yield msg["content"]
            if o.get("done"):
                break


class CloudBackend:
    kind = "cloud"

    def __init__(self, provider=None, base=None, key=None):
        if base and key:
            self.provider, self.base, self.key = (provider or "custom"), base.rstrip("/"), key
        else:
            self.provider, b, self.key = config.active_provider()
            self.base = (b or "").rstrip("/")
        self.label = "cloud:%s" % (self.provider or "none")

    def available(self):
        return bool(self.base and self.key)

    def chat(self, messages, model, stream=True, **opts):
        if not self.available():
            raise BackendError("no cloud provider configured (set a provider API key)")
        timeout = opts.get("timeout") or config.CLOUD_TIMEOUT
        yield from _openai_stream(self.base + "/chat/completions", self.key, model,
                                  messages, opts, timeout=timeout)
