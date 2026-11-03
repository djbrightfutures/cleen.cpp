# -*- coding: utf-8 -*-
r"""A HARD CEILING ON WHAT THE CLOUD LANE MAY SPEND.

the owner, 2026-10-25, after a reel on multi-model routing: "u can update all with this
data." The reel's whole pitch - one OpenRouter key, every model, split-test them - was
ALREADY BUILT here (config.PROVIDER_ORDER has had openrouter in it for weeks). The one
idea in it that was genuinely missing is this one: PUT A DAILY LIMIT ON THE ACCOUNT SO A
RUNAWAY AGENT CANNOT BURN YOUR MONEY.

That matters more here than it does for the person in the reel, because the agents on
this machine are not typed at one request at a time. They retry, they keep, they loop,
and they run while he is asleep. A loop that is harmless on a local model is a bill on a
paid one.

WHAT THIS IS, AND WHAT IT IS NOT
  * It is a LOCAL ceiling, enforced before the request leaves. It is not a billing
    integration and it cannot know what a provider actually charged.
  * Cost is ESTIMATED from the token counts the provider returns, priced from
    model_prices.json (written by model_check.py off OpenRouter's public catalogue).
    An estimate that is roughly right and always present beats an exact number that
    arrives on a statement next month.
  * A model we have no price for is charged at UNKNOWN_PER_M, deliberately HIGH. An
    unpriced model must never be the cheap one by accident.
  * LOCAL CALLS ARE NOT METERED AND NEVER BLOCKED. The guard is only consulted when a
    request carries a key, because that is the only kind that can cost anything.

THE LEDGER is plain JSON next to the config, one entry per UTC day, so he can read it,
and so a crash cannot lose more than the current process's last call.
"""
import datetime
import json
import os
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.environ.get("CLEEN_BUDGET_FILE", os.path.join(HERE, "..", "spend.json"))
PRICES = os.path.join(HERE, "..", "model_prices.json")

# $ per day across every paid provider. 0 disables the cloud lane entirely; a negative
# number means "no ceiling", which has to be typed on purpose.
DAILY_USD = float(os.environ.get("CLEEN_DAILY_USD", "2.00"))
# refuse a single call expected to cost more than this, before it is sent
PER_CALL_USD = float(os.environ.get("CLEEN_PER_CALL_USD", "0.50"))
# what an unpriced model is assumed to cost, per million tokens. Deliberately above the
# most expensive thing in the catalogue.
UNKNOWN_PER_M = float(os.environ.get("CLEEN_UNKNOWN_PER_M", "30.0"))

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    """Raised INSTEAD of sending a request. Never raised for a local call."""


def _today():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def prices():
    """{model_id: {"in": $/M, "out": $/M}} - written by model_check.py, keyless."""
    return _load(PRICES, {}).get("models", {})


def price_of(model):
    """The catalogue entry for a model id, or None.

    Some providers are configured with a BARE vendor id - config.py sends OpenAI
    "gpt-4o-mini", not "openai/gpt-4o-mini". The catalogue is keyed the prefixed way, so
    a bare id would miss, be treated as unpriced, and get charged at the punitive
    unknown rate - which would REFUSE a perfectly ordinary call. Measured: a 20k-token
    prompt on a bare id estimated $0.60 and would have been blocked. So a miss falls
    back to the unique catalogue entry whose name after the "/" matches.
    """
    table = prices()
    if model in table:
        return table[model]
    if model and "/" not in model:
        hits = [v for k, v in table.items() if k.split("/", 1)[-1] == model]
        if len(hits) == 1:                 # only when it is UNAMBIGUOUS
            return hits[0]
    return None


def estimate(model, in_tokens, out_tokens):
    """Dollars for a call of this shape. An unknown model is priced HIGH on purpose."""
    p = price_of(model or "")
    pin = float(p["in"]) if p else UNKNOWN_PER_M
    pout = float(p["out"]) if p else UNKNOWN_PER_M
    return (in_tokens / 1e6) * pin + (out_tokens / 1e6) * pout


def spent_today():
    led = _load(LEDGER, {})
    return float(led.get(_today(), {}).get("usd", 0.0))


def check(model, in_tokens, expect_out=1200):
    """Ask BEFORE sending. Returns the estimate, or raises BudgetExceeded.

    expect_out is a guess at the reply length, because the ceiling has to be applied to
    a cost that has not happened yet. Guessing zero would make every call look free and
    the guard would never fire - which is the failure mode this exists to avoid.
    """
    if DAILY_USD < 0:
        return 0.0                                   # ceiling off, typed on purpose
    est = estimate(model, in_tokens, expect_out)
    if est > PER_CALL_USD:
        raise BudgetExceeded(
            "one call is estimated at $%.4f, over the per-call ceiling of $%.2f "
            "(model %s). Raise CLEEN_PER_CALL_USD to allow it." % (est, PER_CALL_USD, model))
    already = spent_today()
    if already + est > DAILY_USD:
        raise BudgetExceeded(
            "today's cloud spend is $%.4f and this call would add $%.4f, over the daily "
            "ceiling of $%.2f. Raise CLEEN_DAILY_USD or wait for UTC midnight."
            % (already, est, DAILY_USD))
    return est


def record(model, in_tokens, out_tokens, provider=""):
    """Write what a finished call cost. Called after the stream closes."""
    usd = estimate(model, in_tokens, out_tokens)
    with _lock:
        led = _load(LEDGER, {})
        day = led.setdefault(_today(), {"usd": 0.0, "calls": 0, "by_model": {}})
        day["usd"] = round(float(day.get("usd", 0.0)) + usd, 6)
        day["calls"] = int(day.get("calls", 0)) + 1
        m = day.setdefault("by_model", {}).setdefault(
            model or "?", {"usd": 0.0, "calls": 0, "in": 0, "out": 0})
        m["usd"] = round(float(m["usd"]) + usd, 6)
        m["calls"] += 1
        m["in"] += int(in_tokens)
        m["out"] += int(out_tokens)
        if provider:
            day.setdefault("providers", {})
            day["providers"][provider] = round(
                float(day["providers"].get(provider, 0.0)) + usd, 6)
        # keep a month, no more
        for k in sorted(led)[:-31]:
            led.pop(k, None)
        tmp = LEDGER + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(led, f, indent=2, sort_keys=True)
        os.replace(tmp, LEDGER)
    return usd


def status():
    led = _load(LEDGER, {})
    day = led.get(_today(), {})
    return {"day": _today(), "usd": round(float(day.get("usd", 0.0)), 4),
            "calls": int(day.get("calls", 0)), "daily_ceiling": DAILY_USD,
            "per_call_ceiling": PER_CALL_USD,
            "priced_models": len(prices())}


if __name__ == "__main__":
    s = status()
    print("cloud spend %s: $%.4f of $%.2f across %d calls  (per-call ceiling $%.2f, "
          "%d models priced)" % (s["day"], s["usd"], s["daily_ceiling"], s["calls"],
                                 s["per_call_ceiling"], s["priced_models"]))
