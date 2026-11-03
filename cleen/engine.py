"""cleen.cpp engine: two backends behind ONE interface.

LocalBackend  -> the free llama.cpp-family runtime on your machine (its HTTP API by
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
from . import budget


# The default local runtime's HTTP API. "local" is the CANONICAL token;
# "ollama" is still ACCEPTED so an existing CLEEN_LOCAL_KIND keeps working --
# the vendor name leaves the code without breaking anyone's configuration.
_DEFAULT_KINDS = ("local", "ollama")


class BackendError(Exception):
    pass


def _post(url, payload, headers, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _wants_no_think(model):
    """qwen3 / deepseek-r1 emit 'thinking'. For a coder tool we want the answer,
    fast -- so disable it locally. (the runtime honors a top-level `think: false`.)"""
    m = (model or "").lower()
    return any(t in m for t in ("qwen3", "deepseek-r1", "-r1", "reason"))


def _openai_stream(url, key, model, messages, opts, timeout=None, no_think=False):
    """Stream an OpenAI-compatible /chat/completions endpoint (cloud, or llama-server).

    `timeout` bounds each socket read so a hung provider fails fast enough to fail over
    (defaults to the cloud timeout; a dead host is refused instantly regardless).
    `no_think` (llama.cpp only) sends chat_template_kwargs.enable_thinking=false so a
    qwen3 reasoning model answers directly instead of spending the whole budget inside
    <think> and returning empty content.
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
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
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
            probe = "/api/tags" if self.local_kind in _DEFAULT_KINDS else "/v1/models"
            urllib.request.urlopen(self.base + probe, timeout=4).read()
            return True
        except Exception:
            return False

    def list_models(self):
        try:
            if self.local_kind in _DEFAULT_KINDS:
                data = json.loads(urllib.request.urlopen(self.base + "/api/tags", timeout=6).read())
                return [m["name"] for m in data.get("models", [])]
            data = json.loads(urllib.request.urlopen(self.base + "/v1/models", timeout=6).read())
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def chat(self, messages, model, stream=True, **opts):
        timeout = opts.get("timeout") or config.LOCAL_TIMEOUT
        # Per-model base override: one model can run on a different local engine than the
        # default (built for the uncensored MTP GGUF -> llama-swap; all else -> the default runtime).
        base, kind = self.base, self.local_kind
        for sub, ov in config.LOCAL_MODEL_BASES.items():
            if sub and sub in (model or ""):
                base = (ov.get("base") or base).rstrip("/")
                kind = ov.get("kind") or kind
                break
        if kind not in _DEFAULT_KINDS:
            yield from _openai_stream(base + "/v1/chat/completions", None, model,
                                      messages, opts, timeout=timeout,
                                      no_think=_wants_no_think(model))
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
            resp = _post(base + "/api/chat", payload, {"content-type": "application/json"},
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

        # THE SPEND CEILING LIVES HERE AND NOWHERE ELSE. This is the only place in cleen
        # where a request goes out carrying a KEY, so it is the only place that can cost
        # money - LocalBackend.chat() never consults the budget and is never blocked.
        # Token counts are approximated from characters (~4 chars/token): an estimate
        # that is always present beats an exact number that arrives on a statement.
        approx_in = max(1, sum(len(str(m.get("content") or "")) for m in messages) // 4)
        try:
            budget.check(model, approx_in)
        except budget.BudgetExceeded as e:
            raise BackendError("refused by the spend ceiling: %s" % e)

        out_chars = 0
        try:
            for chunk in _openai_stream(self.base + "/chat/completions", self.key, model,
                                        messages, opts, timeout=timeout):
                out_chars += len(chunk)
                yield chunk
        finally:
            # record in a finally, so a stream the caller walks away from is still
            # charged. An abandoned expensive call costs the same as a finished one.
            budget.record(model, approx_in, max(1, out_chars // 4), provider=self.provider)
