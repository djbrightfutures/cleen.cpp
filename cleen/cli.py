"""cleen -- run the best free AI locally, or burst to fast cloud. Same model.

  cleen run [--local|--cloud|--fast] [-m MODEL] [--system S] "your prompt"
  cleen chat  [--local|--cloud|--fast] [-m MODEL]      # interactive REPL
  cleen models                                         # catalog + what's available
  cleen doctor                                         # check local + cloud backends
  cleen serve [--host H] [--port P]                    # OpenAI-compatible API server
"""
import sys
import time
import argparse

from . import config, router, __version__
from .engine import LocalBackend, CloudBackend, BackendError


def _emit(s):
    sys.stderr.write(s + "\n")
    sys.stderr.flush()


def _stream(backend, model, messages, decision):
    _emit(f"[cleen] {decision}")
    t0 = time.time()
    chars = 0
    first = None
    try:
        for chunk in backend.chat(messages, model, stream=True):
            if first is None:
                first = time.time()
            chars += len(chunk)
            sys.stdout.write(chunk)
            sys.stdout.flush()
    except BackendError as e:
        _emit(f"\n[cleen] backend error: {e}")
        return 1
    dt = time.time() - t0
    ttf = (first - t0) if first else dt
    approx_tok = max(1, chars // 4)
    sys.stdout.write("\n")
    _emit(f"[cleen] ~{approx_tok} tok in {dt:.1f}s  (~{approx_tok/dt:.0f} tok/s, first token {ttf:.2f}s)")
    return 0


def _mode_from_args(a):
    if a.local:
        return "local", False
    if a.cloud:
        return "cloud", False
    if a.fast:
        return "auto", True
    # no explicit flag -> this install's default policy (CLEEN_MODE; default "auto")
    return config.split_mode(config.DEFAULT_MODE)


def cmd_run(a):
    prompt = " ".join(a.prompt).strip()
    if not prompt:
        _emit("nothing to run. usage: cleen run \"your prompt\"")
        return 2
    mode, fast = _mode_from_args(a)
    backend, model, decision = router.route(a.model, mode=mode, fast=fast)
    messages = []
    if a.system:
        messages.append({"role": "system", "content": a.system})
    messages.append({"role": "user", "content": prompt})
    return _stream(backend, model, messages, decision)


def cmd_chat(a):
    mode, fast = _mode_from_args(a)
    _emit(f"cleen chat ({mode}{' fast' if fast else ''}) - Ctrl-C to quit")
    history = []
    if a.system:
        history.append({"role": "system", "content": a.system})
    while True:
        try:
            user = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            _emit("\nbye")
            return 0
        if not user:
            continue
        if user in ("/quit", "/exit"):
            return 0
        history.append({"role": "user", "content": user})
        backend, model, decision = router.route(a.model, mode=mode, fast=fast)
        sys.stdout.write("cleen > ")
        # capture the reply into history
        parts = []
        _emit(f"[{decision}]")
        try:
            for chunk in backend.chat(history, model, stream=True):
                parts.append(chunk)
                sys.stdout.write(chunk)
                sys.stdout.flush()
        except BackendError as e:
            _emit(f"\n[error: {e}]")
            history.pop()
            continue
        sys.stdout.write("\n")
        history.append({"role": "assistant", "content": "".join(parts)})


def cmd_models(a):
    local = LocalBackend()
    have_local = local.available()
    local_ids = set(local.list_models()) if have_local else set()
    prov, base, key = config.active_provider()
    print(f"cleen.cpp {__version__}  -  model catalog")
    print(f"  local runtime : {'UP' if have_local else 'DOWN'}  ({config.LOCAL_KIND} @ {config.LOCAL_BASE})")
    print(f"  cloud tier    : {config.PROVIDERS[prov]['label'] if prov else 'none (set a provider key)'}")
    print()
    for name, spec in config.MODELS.items():
        star = " *" if name == config.DEFAULT_MODEL else "  "
        ready = "ready" if spec["local"] in local_ids else "pull needed"
        cdisp = config.cloud_id(spec, prov or "default")
        print(f"{star}{name:<18} {spec['desc']}")
        print(f"     local: {spec['local']:<28} ({ready})   cloud: {cdisp}")
    print("\n  (* = default. Override with -m NAME or CLEEN_MODEL.)")
    return 0


def cmd_doctor(a):
    print("cleen.cpp doctor")
    local = LocalBackend()
    up = local.available()
    print(f"  [local ] {config.LOCAL_KIND} @ {config.LOCAL_BASE} : {'UP' if up else 'DOWN'}")
    if up:
        ids = local.list_models()
        qwen = [m for m in ids if 'qwen' in m.lower()]
        print(f"           {len(ids)} models installed, {len(qwen)} qwen "
              f"(default local: {config.MODELS[config.DEFAULT_MODEL]['local']} "
              f"{'OK' if config.MODELS[config.DEFAULT_MODEL]['local'] in ids else 'MISSING'})")
    prov, base, key = config.active_provider()
    if prov:
        print(f"  [cloud ] {config.PROVIDERS[prov]['label']} @ {base} : KEY SET")
    else:
        print(f"  [cloud ] none - set CEREBRAS_API_KEY / NVIDIA_KEY / OPENROUTER_API_KEY to enable fast mode")
    verdict = "READY" if (up or prov) else "NO BACKENDS"
    print(f"  => {verdict} (local for free/private, cloud for speed)")
    return 0 if (up or prov) else 1


def cmd_serve(a):
    from . import server
    server.run(host=a.host, port=a.port)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="cleen", description="run the best free AI locally, or burst to fast cloud")
    p.add_argument("--version", action="version", version=f"cleen.cpp {__version__}")
    sub = p.add_subparsers(dest="cmd")

    def tier_flags(sp):
        sp.add_argument("--local", action="store_true", help="force local (free, private)")
        sp.add_argument("--cloud", action="store_true", help="force cloud (fast)")
        sp.add_argument("--fast", action="store_true", help="auto, but prefer cloud for speed")
        sp.add_argument("-m", "--model", default=None, help="catalog name or raw model id")
        sp.add_argument("--system", default=None, help="system prompt")

    r = sub.add_parser("run", help="one-shot prompt")
    tier_flags(r)
    r.add_argument("prompt", nargs="+")

    c = sub.add_parser("chat", help="interactive REPL")
    tier_flags(c)

    sub.add_parser("models", help="list the model catalog")
    sub.add_parser("doctor", help="check backends")

    s = sub.add_parser("serve", help="OpenAI-compatible API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8088)
    return p


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    a = parser.parse_args(argv)
    if not a.cmd:
        parser.print_help()
        return 0
    return {
        "run": cmd_run, "chat": cmd_chat, "models": cmd_models,
        "doctor": cmd_doctor, "serve": cmd_serve,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
