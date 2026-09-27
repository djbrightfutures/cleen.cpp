"""cleen.cpp configuration: the model catalog + the provider registry.

cleen.cpp is the BRAND (a friendlier, hybrid llama.cpp). The local runtime is the
llama.cpp family, served the way Ollama packages llama.cpp -- or a llama.cpp
`llama-server` directly. The cloud tier is any OpenAI-compatible provider we resell.

Keys are read from the environment (and an optional local .env) at RUNTIME and are
NEVER hardcoded here -- a shipped key is a published key.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# --------------------------------------------------------------------------------
# Minimal .env loader (no dependency). Loads cleen's OWN .env if present, without
# clobbering a variable already set in the real environment.
# --------------------------------------------------------------------------------
def _load_env_file(path):
    """Load KEY=VALUE lines from one file, never clobbering an already-set var."""
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def _load_env():
    # 1) cleen's own .env (this install's local config; gitignored, never shipped).
    _load_env_file(os.environ.get("CLEEN_ENV_FILE", os.path.join(ROOT, ".env")))
    # 2) any extra env files it points at (e.g. an EXISTING key file elsewhere).
    #    This keeps real secrets OUT of cleen's tree: the path lives only in the
    #    local .env, so the shipped source has neither the key nor a hint of it.
    for p in [x.strip() for x in os.environ.get("CLEEN_EXTRA_ENV", "").split(",") if x.strip()]:
        _load_env_file(p)


_load_env()


# --------------------------------------------------------------------------------
# Local runtime (free, runs on your machine).
# Default speaks the Ollama HTTP API (which runs on llama.cpp under the hood). A
# standalone download can point CLEEN_LOCAL_BASE at a llama.cpp `llama-server`
# (OpenAI-compatible) instead -- same interface, no code change.
# --------------------------------------------------------------------------------
LOCAL_KIND = os.environ.get("CLEEN_LOCAL_KIND", "ollama")          # ollama | llamacpp
LOCAL_BASE = os.environ.get("CLEEN_LOCAL_BASE", "http://127.0.0.1:11434")


# --------------------------------------------------------------------------------
# Cloud tier (fast, when you need speed -- "our own API we resell").
# Every provider is OpenAI-compatible, so ONE cloud backend serves all of them.
# --------------------------------------------------------------------------------
PROVIDERS = {
    "gemini": {     # Google Gemini -- FREE tier, and OpenAI-compatible. Preferred first
                    # so a free key beats a metered one; inert until GEMINI_API_KEY is set.
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "label": "Google Gemini (free)",
    },
    "cerebras": {   # the pitch's ~2000 tok/s provider
        "base": "https://api.cerebras.ai/v1",
        "key_env": ["CEREBRAS_API_KEY"],
        "label": "Cerebras",
    },
    "nim": {        # NVIDIA NIM -- our working default (has credit)
        "base": "https://integrate.api.nvidia.com/v1",
        "key_env": ["NVIDIA_KEY", "NIM_API_KEY", "NVIDIA_API_KEY"],
        "label": "NVIDIA NIM",
    },
    "openrouter": {
        "base": "https://openrouter.ai/api/v1",
        "key_env": ["OPENROUTER_API_KEY", "OPENROUTER_KEY"],
        "label": "OpenRouter",
    },
    "openai": {
        "base": "https://api.openai.com/v1",
        "key_env": ["OPENAI_API_KEY"],
        "label": "OpenAI",
    },
}
# Auto-pick order: first provider WITH A KEY wins. Cerebras (fastest) first, then
# NIM (our default with credit), then the rest.
PROVIDER_ORDER = ["gemini", "cerebras", "nim", "openrouter", "openai"]


# --------------------------------------------------------------------------------
# The model catalog: ONE logical name -> a local resolution AND a cloud one, so
# `cleen run` uses the SAME model whether it runs local or bursts to the cloud.
# --------------------------------------------------------------------------------
# cloud ids are PROVIDER-AWARE: the same logical model resolves to real Qwen on the
# providers that host it (Cerebras = the pitch's ~2000 tok/s tier, OpenRouter), and to
# that provider's best equivalent otherwise (NIM has no Qwen, so its strongest coder) --
# so `cleen run --cloud` never 404s, whatever key you hold. "default" = the ideal id.
# (Cerebras/OpenRouter ids are best-known; cleen verifies them the first time you add a key.)
MODELS = {
    "cleen-coder": {        # default: Qwen coder -- the "amazing free model" from the pitch
        "local": "qwen2.5-coder:7b",
        "cloud": {
            "cerebras":   "qwen-3-coder-480b",
            "openrouter": "qwen/qwen3-coder",
            "nim":        "nvidia/nemotron-3-super-120b-a12b",
            "openai":     "gpt-4o-mini",
            "gemini":     "gemini-2.5-flash",
            "default":    "qwen/qwen3-coder",
        },
        "desc": "Qwen coder - free & fast locally (fits a 6GB GPU), Qwen3-Coder in the cloud.",
    },
    "cleen-coder-max": {    # the MoE "runs on anything, even from disk" model
        "local": "qwen3-coder:30b",
        "cloud": {
            "cerebras":   "qwen-3-coder-480b",
            "openrouter": "qwen/qwen3-coder",
            "nim":        "nvidia/nemotron-3-super-120b-a12b",
            "gemini":     "gemini-2.5-pro",
            "default":    "qwen/qwen3-coder-480b-a35b-instruct",
        },
        "desc": "Qwen3-Coder MoE - runs on modest hardware locally (few active params), 480B in the cloud.",
    },
    "cleen-chat": {         # general chat/reasoning
        "local": "qwen3:8b",
        "cloud": {
            "cerebras":   "qwen-3-235b-a22b-instruct",
            "gemini":     "gemini-2.5-flash",
            "openrouter": "qwen/qwen3-235b-a22b",
            "nim":        "nvidia/nemotron-3-super-120b-a12b",
            "default":    "qwen/qwen3-235b-a22b",
        },
        "desc": "General chat & reasoning.",
    },
}


def cloud_id(spec, provider):
    """Resolve a catalog entry's cloud id for a given provider (dict or raw string)."""
    c = spec.get("cloud")
    if isinstance(c, dict):
        return c.get(provider) or c.get("default") or next(iter(c.values()))
    return c
DEFAULT_MODEL = os.environ.get("CLEEN_MODEL", "cleen-coder")

# The default routing policy for THIS install: auto | local | cloud | fast.
# Unset => "auto" (local-first, free, no surprise bills) -- the safe product default.
# This machine sets CLEEN_MODE=fast in .env => API performance by default, with a
# silent fall back to local when there's no key / no internet.
DEFAULT_MODE = os.environ.get("CLEEN_MODE", "auto").strip().lower()


def split_mode(m):
    """Map a mode string to the router's (mode, fast) pair -- the ONE place this
    meaning lives, so the CLI and the server always agree.

    'fast' = auto routing that prefers the cloud for speed.
    """
    m = (m or "auto").strip().lower()
    if m == "fast":
        return "auto", True
    if m in ("local", "cloud", "auto"):
        return m, False
    return "auto", False


# --------------------------------------------------------------------------------
# Smart-driver tuning knobs (all env-overridable, all with safe defaults). These are
# the "smarter than a raw engine" layer: a response cache, difficulty-aware auto
# routing, and provider failover. Every default keeps existing behavior intact.
# --------------------------------------------------------------------------------
def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_num(name, default, cast=float):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


ENGINE_DEFAULT_TEMP = 0.3     # the temperature the engine uses when a caller omits it

# 1) Response cache
CACHE_ON = _env_bool("CLEEN_CACHE", True)                 # master on/off
CACHE_TTL = _env_num("CLEEN_CACHE_TTL", 3600, int)        # seconds an answer stays fresh
CACHE_MAX = _env_num("CLEEN_CACHE_MAX", 512, int)         # max entries (LRU eviction)
CACHE_MAXTEMP = _env_num("CLEEN_CACHE_MAXTEMP", 0.3)      # only cache temperature <= this
CACHE_REPLAY_CHUNK = _env_num("CLEEN_CACHE_REPLAY_CHUNK", 180, int)  # stream-replay chunk size

# 2) Smart auto-routing: let 'auto' send easy prompts local, hard ones to fast cloud.
AUTO_SMART = _env_bool("CLEEN_AUTO_SMART", True)
AUTO_SMART_LONG_CHARS = _env_num("CLEEN_AUTO_LONG_CHARS", 1600, int)  # long context -> cloud

# 3) Failover timeouts (short, so one dead provider doesn't hang the request).
CLOUD_TIMEOUT = _env_num("CLEEN_CLOUD_TIMEOUT", 60, int)
LOCAL_TIMEOUT = _env_num("CLEEN_LOCAL_TIMEOUT", 120, int)


def _provider_base(name):
    """A provider's base URL, with an optional per-provider override so a user can point
    cleen at a gateway/proxy/self-host (e.g. CLEEN_BASE_GEMINI=...). Defaults to the
    built-in base, so this is inert unless explicitly set."""
    override = os.environ.get("CLEEN_BASE_" + name.upper())
    return (override or PROVIDERS[name]["base"]).rstrip("/")


def _provider_key(name):
    for env in PROVIDERS[name]["key_env"]:
        v = os.environ.get(env)
        if v:
            return v
    return None


def cloud_providers():
    """The FULL failover chain: every configured provider that has a key, in order ->
    [(name, base, key), ...].  CLEEN_PROVIDER (if keyed) is tried first, then the rest
    of PROVIDER_ORDER. active_provider() is simply this list's first element; having the
    whole ordered list is what lets one dead provider fall through to the next."""
    forced = os.environ.get("CLEEN_PROVIDER", "").strip().lower()
    order = ([forced] if forced in PROVIDERS else []) + \
            [p for p in PROVIDER_ORDER if p != forced]
    out = []
    for name in order:
        if name not in PROVIDERS:
            continue
        key = _provider_key(name)
        if key:
            out.append((name, _provider_base(name), key))
    return out


def active_provider():
    """Pick the cloud provider -> (name, base, key). Else (None, None, None).

    CLEEN_PROVIDER forces one (if it has a key); otherwise the first provider in
    PROVIDER_ORDER that has a key set wins. (= first element of cloud_providers().)
    """
    chain = cloud_providers()
    return chain[0] if chain else (None, None, None)
