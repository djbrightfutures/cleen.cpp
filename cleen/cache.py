"""cleen.cpp response cache -- the biggest "faster than llama" win.

A raw inference engine (llama.cpp, a bare llama-server, Ollama) has NO response
cache: ask the same question twice and it re-runs the full forward pass twice. cleen
is a *router*, so it can remember. This is an exact-match cache: identical request in
-> stored answer out, in milliseconds, no GPU, no tokens billed.

Key  = sha256 of (logical model + effective mode/fast + normalized messages + the
       params that change the OUTPUT: temperature, max_tokens).
Policy = only near-deterministic requests are cached (temperature <= a threshold, or
       unset), so we never memoize a deliberately-random draw. A caller can always
       opt a single request out with  "no_cache": true.
Bounds = every entry has a TTL, and the store is an LRU capped at a max size, so it
       can never grow without limit or serve stale answers.

Pure stdlib + one lock, because the server is threaded (ThreadingHTTPServer).
"""
import hashlib
import json
import threading
import time
from collections import OrderedDict

from . import config


class ResponseCache:
    """Thread-safe TTL + LRU store of completion text, keyed by request signature."""

    def __init__(self, ttl, maxlen):
        self.ttl = ttl
        self.maxlen = max(1, int(maxlen))
        self._d = OrderedDict()          # key -> (expires_at, text)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """Return fresh cached text (and mark it most-recently-used), else None."""
        if not key:
            return None
        now = time.time()
        with self._lock:
            item = self._d.get(key)
            if item is None:
                self.misses += 1
                return None
            expires, text = item
            if expires < now:
                # expired -> drop it, count as a miss
                self._d.pop(key, None)
                self.misses += 1
                return None
            self._d.move_to_end(key)     # LRU touch
            self.hits += 1
            return text

    def put(self, key, text):
        """Store a non-empty completion; evict the oldest entries past the cap."""
        if not key or not text or not text.strip():
            return  # never cache an empty/failed answer
        with self._lock:
            self._d[key] = (time.time() + self.ttl, text)
            self._d.move_to_end(key)
            while len(self._d) > self.maxlen:
                self._d.popitem(last=False)   # evict least-recently-used

    def clear(self):
        with self._lock:
            self._d.clear()

    def stats(self):
        with self._lock:
            total = self.hits + self.misses
            return {
                "enabled": bool(config.CACHE_ON),
                "entries": len(self._d),
                "max": self.maxlen,
                "ttl_s": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
            }


# One process-wide cache instance (the server is a single process, many threads).
_CACHE = ResponseCache(ttl=config.CACHE_TTL, maxlen=config.CACHE_MAX)


def cacheable(temperature, no_cache):
    """Should THIS request be cached? Only near-deterministic, opt-in-able requests.

    temperature unset -> treated as the engine default (0.3), which is cacheable.
    A high temperature is a deliberate random draw -> never memoized.
    """
    if not config.CACHE_ON or no_cache:
        return False
    try:
        t = config.ENGINE_DEFAULT_TEMP if temperature is None else float(temperature)
    except (TypeError, ValueError):
        return False
    return t <= config.CACHE_MAXTEMP


def make_key(model, messages, opts, mode, fast):
    """Stable signature of everything that determines the answer. Computed with ZERO
    network so a hit can be served before we ever probe a backend."""
    payload = {
        "model": model,
        "mode": mode,
        "fast": bool(fast),
        "temperature": opts.get("temperature"),
        "max_tokens": opts.get("max_tokens"),
        "messages": messages,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "cc-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key):
    return _CACHE.get(key)


def put(key, text):
    _CACHE.put(key, text)


def stats():
    return _CACHE.stats()


def clear():
    _CACHE.clear()


def replay_chunks(text, size=None):
    """Yield a cached answer back as a few chunks, so a STREAMING caller gets the same
    incremental contract it expects (not one giant delta)."""
    size = size or config.CACHE_REPLAY_CHUNK
    if size <= 0 or len(text) <= size:
        yield text
        return
    for i in range(0, len(text), size):
        yield text[i:i + size]
