# cleen.cpp

**Run the best free AI on any computer. Or go fast in the cloud. Same model. You choose.**

> Part of **Luxor Net Technologies** — free, open-source, privacy-first tools. MIT licensed.

cleen.cpp is a friendlier, hybrid **llama.cpp**: local-first and private by default,
with a one-flag jump to a fast cloud tier when you actually need the speed. The best
open models (Qwen3-Coder and friends) are free and run on almost any hardware — only a
few experts activate per token, so a big model can even stream from disk. It's just
*slower* on weak hardware. So the only real question is:

> **How fast do you actually need tokens?**

Most of the time: run it free and local. When you need 2,000 tokens/sec: flip to the
cloud. Same model identity, same API, your choice per request.

---

## Quick start

```bash
# 1. check your machine
cleen doctor

# 2. run a prompt (auto-routes: local by default)
cleen run "write a python function that reverses a string"

# 3. force a tier
cleen run --local "keep it private, on my machine"
cleen run --fast  "I want speed — use the cloud"

# 4. serve an OpenAI-compatible API (point any tool at it)
cleen serve --port 8088
```

Point Cursor / Continue / any OpenAI SDK at `http://127.0.0.1:8088/v1` with any API
key — you get **private local** or **fast cloud** with zero code change.

---

## How it works

```
        your prompt
             │
      ┌──────▼───────┐   same model identity, same API either way
      │ SMART ROUTER │   (the "smarter" in "a better, smarter llama.cpp")
      └──┬────────┬──┘
   local │        │ cloud
 (free,  │        │ (fast, when you
 private)│        │  need speed)
   ┌─────▼──┐  ┌──▼───────────────┐
   │llama.cpp│  │ OpenAI-compatible │
   │ family  │  │ provider we resell│
   │(Ollama/ │  │ (Cerebras, NIM…)  │
   │llama-   │  └───────────────────┘
   │server)  │
   └─────────┘
```

- **Local** — the free llama.cpp-family runtime on your machine. Private: nothing
  leaves your computer unless *you* flip to cloud.
- **Cloud** — any OpenAI-compatible provider. cleen resells fast compute under one
  brand, so you're never locked to one vendor's model names.
- **Router** — `auto` runs local by default and only bursts to cloud when it clearly
  wins (you asked for speed, or local is down). Force it with `--local` / `--cloud`.

## The model catalog

One logical name resolves to a **local** id *and* a **cloud** id, so "the same model"
means the same thing whether it runs on your laptop or in the cloud. Cloud ids are
**provider-aware**: real Qwen3-Coder on the providers that host it (Cerebras / OpenRouter),
and that provider's best equivalent otherwise — so `--cloud` never 404s, whatever key
you hold.

| name              | local                 | cloud (ideal)            |
|-------------------|-----------------------|--------------------------|
| `cleen-coder` *   | `qwen2.5-coder:7b`    | Qwen3-Coder              |
| `cleen-coder-max` | `qwen3-coder:30b` MoE | Qwen3-Coder-480B         |
| `cleen-chat`      | `qwen3:8b`            | Qwen3-235B               |

`*` default. Override with `-m NAME` or `CLEEN_MODEL`.

## Configuration (environment / `.env`)

Keys are read from the environment (or a local `.env`) at runtime and are **never**
shipped in the code.

| variable            | what                                                        |
|---------------------|-------------------------------------------------------------|
| `CLEEN_MODE`        | default routing: `auto` (local-first) / `fast` (cloud API first) / `local` / `cloud` |
| `CLEEN_MODEL`       | default catalog model (default `cleen-coder`)               |
| `CLEEN_LOCAL_BASE`  | local runtime URL (default `http://127.0.0.1:11434`, Ollama)|
| `CLEEN_LOCAL_KIND`  | `ollama` (default) or `llamacpp` (a llama.cpp `llama-server`)|
| `CLEEN_PROVIDER`    | force a cloud provider (`cerebras`/`nim`/`openrouter`/`openai`)|
| `CLEEN_EXTRA_ENV`   | comma-separated extra `.env` files to source keys from (keeps secrets out of the tree)|
| `CEREBRAS_API_KEY`  | cloud tier — the ~2,000 tok/s provider                       |
| `NVIDIA_KEY`        | cloud tier — NVIDIA NIM                                      |
| `OPENROUTER_API_KEY`| cloud tier — OpenRouter (hosts real Qwen)                    |
| `OPENAI_API_KEY`    | cloud tier — OpenAI                                          |

If no cloud key is set, cleen is happily **local-only** — free, private, runs on anything.

**Performance profile.** `CLEEN_MODE` sets the default so you never pass a flag: leave it
`auto` for local-first (the safe default — no surprise cloud bills), or set `fast` to prefer
the cloud API for speed and fall back to local only when there's no key or no internet. You
can still override per call with `--local` / `--cloud` / `--fast` (CLI) or `X-Cleen-Mode` (API).

## Make it yours (bring your own brain)

cleen ships with a default brain -- **Qwen** (`qwen2.5-coder:7b` local, Qwen3-Coder in the
cloud) behind the smart local-first router. You can swap in your own, two ways -- copy
`.env.example` to `.env` and edit:

- **Your own local model** — set `CLEEN_MODEL` to any id your runtime has (e.g.
  `llama3.1:8b`), or point `CLEEN_LOCAL_BASE` / `CLEEN_LOCAL_KIND` at your own
  Ollama or llama.cpp `llama-server`. Non-catalog ids pass straight through.
- **Your own cloud provider** — set any one provider key (`GEMINI_API_KEY`,
  `CEREBRAS_API_KEY`, `NVIDIA_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`); optionally
  force one with `CLEEN_PROVIDER`. No key set = stays 100% local & free.

Every knob, with inline notes and safe empty defaults, lives in **`.env.example`**.

## Ship a clean copy

`python package.py` writes a buyer-ready `dist/cleen.cpp/` — only the code, launchers,
installers, README, `.env.example`, and web page — then a **verify gate** refuses the build
if any secret, real name, or machine path slipped in. Your real `.env`, logs, and local ops
scripts never travel. `python package.py --verify` re-sweeps an existing build.

## The server (OpenAI-compatible)

```
GET  /health
GET  /v1/models
POST /v1/chat/completions        # stream + non-stream
```

Per-request routing (both optional):
- header `X-Cleen-Mode: local | cloud | auto | fast`  (default = `CLEEN_MODE`, else `auto`)
- body `"cleen_fast": true`  (auto, prefer cloud for speed)

## Requirements

- Python 3.8+ (pure stdlib — no pip installs for the core).
- A local runtime for local mode: [Ollama](https://ollama.com) (default) or a
  llama.cpp `llama-server`. Cloud mode needs only a provider key.

## License

The cleen.cpp brand and router are a Luxucleen project. Local inference is powered by
the llama.cpp family (MIT). Cloud inference is powered by OpenAI-compatible providers.
