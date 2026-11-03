"""cleen.cpp configuration: the model catalog + the provider registry.

cleen.cpp is the BRAND (a friendlier, hybrid llama.cpp). The local runtime is the
llama.cpp family, served the way the default runtime packages llama.cpp -- or a llama.cpp
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
# Default speaks the local runtime's HTTP API (which runs on llama.cpp under the hood). A
# standalone download can point CLEEN_LOCAL_BASE at a llama.cpp `llama-server`
# (OpenAI-compatible) instead -- same interface, no code change.
# --------------------------------------------------------------------------------
LOCAL_KIND = os.environ.get("CLEEN_LOCAL_KIND", "local")           # local | llamacpp
LOCAL_BASE = os.environ.get("CLEEN_LOCAL_BASE", "http://127.0.0.1:11434")

# Per-model LOCAL base override (the owner, 2026-09-30): a JSON map
#   {"<substring of the local model id>": {"base": "...", "kind": "local|llamacpp"}}
# so ONE model can run on a DIFFERENT local engine than the default. Built for the
# uncensored MTP GGUF that the default loader rejects but our llamacpp-cuda (via
# llama-swap) loads -- that one entry points at llama-swap while every other local
# model stays on the default runtime. Default "{}" = no-op (never regresses existing routing).
try:
    LOCAL_MODEL_BASES = __import__("json").loads(
        os.environ.get("CLEEN_LOCAL_MODEL_BASES", "") or "{}")
except Exception:
    LOCAL_MODEL_BASES = {}


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
    "anthropic": {  # Claude. Anthropic serves an OpenAI-shaped /chat/completions at this base,
                    # so it drops into the same client as every other provider here.
                    # Inert until ANTHROPIC_API_KEY is set (exactly like gemini above), and it
                    # NEVER preempts local: mode decides local-vs-cloud, this list only decides
                    # WHICH cloud is used once a cloud call was already chosen.
        "base": "https://api.anthropic.com/v1",
        "key_env": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "label": "Anthropic Claude",
    },
    "deepseek": {   # DeepSeek -- OpenAI-compatible, cheap + strong (V3.2 chat / R1 reasoner).
                    # Inert until DEEPSEEK_API_KEY is set (exactly like gemini/anthropic above);
                    # NEVER preempts local. base includes /v1 so the client appends
                    # /chat/completions -> https://api.deepseek.com/v1/chat/completions.
        "base": "https://api.deepseek.com/v1",
        "key_env": ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"],
        "label": "DeepSeek",
    },
}
# Auto-pick order: first provider WITH A KEY wins. the owner 2026-10-23: "make it work with
# both claude's api and local" -> anthropic leads the CLOUD list, while LOCAL stays the
# default engine (CLEEN_MODE=local in .env). Cloud is only reached on an explicit --cloud/
# --fast, or when a local engine is genuinely unavailable.
PROVIDER_ORDER = ["anthropic", "gemini", "deepseek", "cerebras", "nim", "openrouter", "openai"]


# --------------------------------------------------------------------------------
# The model catalog: ONE logical name -> a local resolution AND a cloud one, so
# `cleen run` uses the SAME model whether it runs local or bursts to the cloud.
# --------------------------------------------------------------------------------
# cloud ids are PROVIDER-AWARE: the same logical model resolves to real Qwen on the
# providers that host it (Cerebras = the pitch's ~2000 tok/s tier, OpenRouter), and to
# that provider's best equivalent otherwise (NIM has no Qwen, so its strongest coder) --
# so `cleen run --cloud` never 404s, whatever key you hold. "default" = the ideal id.
# VERIFIED, NOT "BEST-KNOWN" (2026-10-25). This comment used to say the ids were
# best-known and would be verified "the first time you add a key" - deferring the check
# to the worst possible moment, a 404 mid-task. `python -m cleen.model_check` now audits
# every OpenRouter-shaped id against the public catalogue with NO KEY, and found TWO
# dead ones sitting here: qwen/qwen3-235b-a22b (retired in favour of the dated
# ...-a22b-2507) and qwen/qwen3-coder-480b-a35b-instruct (never existed under that id).
# Run it after touching this table; it exits non-zero on a name nobody serves.
MODELS = {
    "cleen-coder": {        # default: Qwen coder -- the "amazing free model" from the pitch
        "local": "qwen2.5-coder:7b",
        "cloud": {
            "cerebras":   "qwen-3-coder-480b",
            "openrouter": "qwen/qwen3-coder",
            "nim":        "nvidia/nemotron-3-super-120b-a12b",
            "openai":     "gpt-4o-mini",
            "gemini":     "gemini-2.5-flash",
            "anthropic":  "claude-sonnet-5-5",
            "deepseek":   "deepseek-chat",
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
            "anthropic":  "claude-opus-5-5",
            "deepseek":   "deepseek-chat",
            "default":    "qwen/qwen3-coder",
        },
        "desc": "Qwen3-Coder MoE - runs on modest hardware locally (few active params), 480B in the cloud.",
    },
    "cleen-chat": {         # general chat/reasoning
        "local": "qwen3:8b",
        "cloud": {
            "cerebras":   "qwen-3-235b-a22b-instruct",
            "gemini":     "gemini-2.5-flash",
            "openrouter": "qwen/qwen3-235b-a22b-2507",
            "nim":        "nvidia/nemotron-3-super-120b-a12b",
            "anthropic":  "claude-sonnet-5-5",
            "deepseek":   "deepseek-chat",
            "default":    "qwen/qwen3-235b-a22b-2507",
        },
        "desc": "General chat & reasoning.",
    },
    "cleen-reason": {       # DeepSeek R1 reasoning -- strongest LOCAL reasoner that fits a 6GB
                            # GPU: deepseek-r1:7b (already pulled, 4.7GB). DeepSeek reasoner in
                            # the cloud. For HARD multi-step questions (emits <think> chains,
                            # slower per-token by design -- send easy/exact work elsewhere).
        "local": "deepseek-r1:7b",
        "cloud": {
            "deepseek":   "deepseek-reasoner",
            "nim":        "nvidia/nemotron-3-super-120b-a12b",
            "gemini":     "gemini-2.5-pro",
            "anthropic":  "claude-opus-5-5",
            "default":    "deepseek-reasoner",
        },
        "desc": "DeepSeek R1 reasoning - deepseek-r1:7b locally (fits 6GB, already installed), DeepSeek reasoner in the cloud. Best for hard multi-step reasoning.",
    },
    "cleen-reason-fast": {  # R1 reasoning that FITS a 6GB card alongside a warm model:
                            # deepseek-r1:1.5b (~2GB). Weaker than r1:7b/cloud, but it RUNS
                            # locally when the heavy reasoners can't get a GPU slot.
        "local": "deepseek-r1:1.5b",
        "cloud": {
            "deepseek":   "deepseek-reasoner",
            "default":    "deepseek-reasoner",
        },
        "desc": "DeepSeek R1 1.5B - the reasoner that fits 6GB beside a warm model (local, free). Smaller/weaker than cleen-reason; use cleen-reason or the cloud reasoner for strong reasoning.",
    },
    "cleen-uncensored": {   # local-only: uncensored Qwen3.8-27B (IQ2_M). No cloud twin.
        "local": "hf.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:IQ2_M",
        # No provider hosts this exact uncensored build, so 'cloud' points back at the
        # local id (honest passthrough): a forced --cloud call has no censored substitute
        # silently swapped in, and auto/fast failover lands on local (never-dark). RUN --local.
        "cloud": "hf.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:IQ2_M",
        "desc": "Uncensored Qwen3.8-27B (IQ2_M, ~11GB). LOCAL-ONLY -- run with --local / mode=local.",
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

# 1b) COMPUTE lane: answer EXACT work (arithmetic, numeric comparison) with CODE,
# before any model -- instant, 0 tokens, always correct. OFF => every request goes
# to a model (old behavior). Opt a single request out with body "cleen_no_compute":true.
COMPUTE_ON = _env_bool("CLEEN_COMPUTE", True)

# 2) Smart auto-routing: let 'auto' send easy prompts local, hard ones to fast cloud.
AUTO_SMART = _env_bool("CLEEN_AUTO_SMART", True)
AUTO_SMART_LONG_CHARS = _env_num("CLEEN_AUTO_LONG_CHARS", 1600, int)  # long context -> cloud

# 3) Failover timeouts (short, so one dead provider doesn't hang the request).
CLOUD_TIMEOUT = _env_num("CLEEN_CLOUD_TIMEOUT", 60, int)
LOCAL_TIMEOUT = _env_num("CLEEN_LOCAL_TIMEOUT", 120, int)


# --------------------------------------------------------------------------------
# COMPUTER-CONTROL TOOLS (the owner 2026-09-30: "whenever we change to any model it
# has control of my whole entire computer... even small models"). The agent layer in
# agent.py gives every model routed through cleen real hands+eyes (open browser/apps,
# run commands, search the web, type, screenshot). All OFF by default for the plain
# OpenAI API (an editor hitting /v1 must get a normal answer, never a tool loop); a
# caller opts in per request (body cleen_tools:true / header X-Cleen-Tools / mode
# 'agent'), and ghost.py flips it on so the owner's models act from his phone.
# --------------------------------------------------------------------------------
TOOLS_DEFAULT = _env_bool("CLEEN_TOOLS", False)          # global default for the agent layer
TOOLS_MAX_STEPS = _env_num("CLEEN_TOOLS_MAX_STEPS", 8, int)   # tool calls per answer (anti-loop)
TOOLS_MAX_RESULT = _env_num("CLEEN_TOOLS_MAX_RESULT", 6000, int)  # cap a tool result fed back
TOOLS_CMD_TIMEOUT = _env_num("CLEEN_TOOLS_CMD_TIMEOUT", 60, int)  # run_command timeout (s)
TOOLS_LOG = os.environ.get("CLEEN_TOOLS_LOG", os.path.join(ROOT, "agent_actions.jsonl"))
TOOLS_EXTRA_DIR = os.environ.get("CLEEN_TOOLS_EXTRA_DIR", "")  # private per-machine tool plugins

# Which tools a caller may use (None = all registered). Lets a deploy ship a narrower
# surface than this box runs. Empty env => all.
_allow = [x.strip() for x in os.environ.get("CLEEN_TOOLS_ALLOW", "").split(",") if x.strip()]
TOOLS_ALLOW = _allow or None

# run_command safety: a SMALL denylist of catastrophic, irreversible, non-task commands
# (disk-wipe class) so a fumbling small model can't brick the machine -- everything else
# is allowed (full control is the point). Tunable via CLEEN_TOOLS_CMD_DENY (|-separated
# regexes, REPLACES the defaults). These are matched case-insensitively against the command.
_DEFAULT_CMD_DENY = [
    r"\bformat\b\s+[a-z]:",                 # format C:
    r"\bdiskpart\b",                        # scriptable disk wipe
    r"\bcipher\s+/w",                       # secure-wipe free space
    r"\b(rmdir|rd)\b.*\s/s.*\s[a-z]:\\?\s*$",   # rd /s on a drive root
    r"\bdel\b.*\s/[sq].*\\windows\b",       # del /s|/q under C:\Windows
    r"\brm\s+-rf?\s+(/|[a-z]:[\\/]?)(\s|$)",    # rm -rf / or rm -rf C:\
    r"\bmkfs\b", r"\bdd\b\s+if=.*of=/dev/",
]
_deny_env = os.environ.get("CLEEN_TOOLS_CMD_DENY", "").strip()
TOOLS_CMD_DENY = [p for p in _deny_env.split("|") if p.strip()] if _deny_env else _DEFAULT_CMD_DENY


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
