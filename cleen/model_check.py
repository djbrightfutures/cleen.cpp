# -*- coding: utf-8 -*-
r"""IS EVERY MODEL ID WE ROUTE TO A REAL MODEL, AND WHAT DOES IT COST?

    python -m cleen.model_check            # audit config.py against the live catalogue
    python -m cleen.model_check --write     # also write model_prices.json for budget.py

WHY (2026-10-25). A reel told the owner to route cheap work to "GLM 5.3 Flash". There is
no such model. OpenRouter's catalogue has z-ai/glm-5 and z-ai/glm-4.7-flash, and nothing
called 5.3. A model id that does not exist does not fail when you write it - it fails
the first time something routes to it, which is the worst possible moment, and the error
comes back as a 404 from a provider rather than as "you typed a name nobody has".

So this asks the catalogue instead of trusting a name. It needs NO KEY: OpenRouter
serves /api/v1/models publicly, with prices, and that is also where the prices for the
spend ceiling come from.

IT ALSO ANSWERS THE QUESTION THE REEL GOT WRONG. The reel's framing is "cheap means the
Chinese models". Measured on the same catalogue on the same day, that is not true: an
OpenAI model sat below GLM-5 on both input and output price. Cheap is a NUMBER, and the
number is published.
"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "model_prices.json")
CATALOGUE = "https://openrouter.ai/api/v1/models"


def catalogue(url=CATALOGUE, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "cleen-model-check/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = {}
    for m in data.get("data", []):
        p = m.get("pricing") or {}
        try:
            out[m["id"]] = {"in": float(p.get("prompt") or 0) * 1e6,
                            "out": float(p.get("completion") or 0) * 1e6,
                            "ctx": m.get("context_length")}
        except (TypeError, ValueError):
            continue
    return out


def routed_ids():
    """Every OpenRouter-shaped id config.py can hand to the cloud lane.

    THAT IS TWO KEYS, NOT ONE. "openrouter" is the id used when that provider is
    selected; "default" is the id used when a provider has no entry of its own, and it
    is written OpenRouter-shaped too. The first version of this checker read only
    "openrouter" and PASSED a dead "default" sitting two lines below a dead one it had
    just caught - the same scope gap as a census that covers nine of sixteen files.

    The other keys (cerebras, nim, openai, gemini, anthropic) are deliberately NOT
    audited here: they are that vendor's own naming and would read as missing from
    OpenRouter's catalogue when they are perfectly valid.
    """
    from cleen import config
    out = {}
    for name, spec in (getattr(config, "MODELS", {}) or {}).items():
        cloud = (spec or {}).get("cloud") or {}
        if not isinstance(cloud, dict):
            continue                    # local-only entries carry no cloud mapping
        for key in ("openrouter", "default"):
            if cloud.get(key):
                out.setdefault(cloud[key], []).append("%s.%s" % (name, key))
    return out


def main(argv):
    cat = catalogue()
    print("catalogue: %d models, no key required" % len(cat))

    bad = []
    routed = routed_ids()
    print("\nwhat this machine routes to (%d ids over %d call sites):"
          % (len(routed), sum(len(v) for v in routed.values())))
    for mid in sorted(routed):
        where = ", ".join(routed[mid])
        hit = cat.get(mid)
        if hit:
            print("   OK      %-40s in $%-6.3f out $%-6.3f ctx %-9s [%s]"
                  % (mid, hit["in"], hit["out"], hit["ctx"], where))
        else:
            print("   MISSING %-40s <- no such id in the catalogue  [%s]" % (mid, where))
            bad.append("%s (%s)" % (mid, where))

    cheap = sorted((v["in"] + v["out"], k) for k, v in cat.items() if v["in"] > 0)[:8]
    print("\ncheapest eight that actually exist (in+out per million):")
    for tot, k in cheap:
        print("   $%-8.3f %-40s ctx %s" % (tot, k, cat[k]["ctx"]))

    if "--write" in argv:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"source": CATALOGUE, "models": cat}, f, indent=1, sort_keys=True)
        print("\nwrote %s (%d priced models) - the spend ceiling reads this"
              % (os.path.normpath(OUT), len(cat)))

    if bad:
        print("\nFAIL - %d routed model id(s) do not exist: %s" % (len(bad), ", ".join(bad)))
        return 1
    print("\nPASS - every routed model id exists and is priced.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
