"""cleen.cpp COMPUTE lane -- answer EXACT work with code, never a model.

A language model PREDICTS the next token. For anything EXACT -- arithmetic, a
numeric comparison -- prediction is the wrong instrument: it is slow (a whole
forward pass locally, or a cloud round-trip) and it can be WRONG (ask a 7B for
8472*391 and it will happily invent digits). cleen is a *router*, so it can do
what a raw inference engine never does: recognise an exact request and COMPUTE
the answer deterministically -- sub-millisecond, zero tokens, always correct --
BEFORE any model is touched.

This is the "compute, don't predict" principle (proven in Bridge128's
smart_brain and gated by test_hands_100): route exact tasks to code, fuzzy
tasks to the model.

SAFETY / NO-HIJACK (the whole point):
  try_compute() returns a string ONLY when the LAST user message is an
  UNAMBIGUOUS exact request -- arithmetic that reduces to a pure number/operator
  expression, or an explicit "which is bigger, A or B" numeric comparison. ANY
  prose, ANY letters left after stripping the lead-in, ANY ambiguity -> it
  returns None and the request falls straight through to the normal cache->model
  path. It can NEVER answer a fuzzy question ("what is the capital of France",
  "what is love", "write a function to add two numbers"): those keep letters and
  fail the strict fullmatch. Pure stdlib (ast + re); no eval(), whitelisted ops
  only; div-by-zero and runaway exponents are refused (fall through).
"""
import ast
import re
import threading

from . import config

# ---- stats (so /health can show the lane is earning its keep) ----
_lock = threading.Lock()
_hits = 0


def _hit():
    global _hits
    with _lock:
        _hits += 1


def stats():
    with _lock:
        return {"enabled": bool(getattr(config, "COMPUTE_ON", True)), "hits": _hits}


def _fmt_num(value) -> str:
    """Render a number the way a person would: no trailing .0, no float noise."""
    if isinstance(value, bool):                      # bool is an int subclass -- guard
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _eval_arith(expr: str):
    """Evaluate a PURE arithmetic expression via a whitelisted AST walk. NEVER eval().
    Supports + - * / % ** and parentheses over int/float; refuses div-by-zero and a
    runaway exponent. Anything outside this grammar raises -> caller returns None."""
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                raise ZeroDivisionError
            if isinstance(node.op, ast.Pow) and abs(right) > 32:
                raise ValueError("exponent too large")
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            return left ** right
        raise ValueError("unsupported expression")

    return ev(tree)


# words people use in place of operators (applied only after the strict gate below)
_WORD_OPS = (
    (r"\bdivided\s+by\b", "/"),
    (r"\bover\b", "/"),
    (r"\bto\s+the\s+power\s+of\b", "**"),
    (r"\bmultiplied\s+by\b", "*"),
    (r"\btimes\b", "*"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\bmod(?:ulo)?\b", "%"),
    (r"(?<=\d)\s*x\s*(?=\d)", "*"),        # "12 x 7" -> 12*7 (only BETWEEN digits)
)

# optional, strippable lead-in phrasing ("what is", "calculate", ...)
_ARITH_LEAD = re.compile(
    r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:tell\s+me\s+)?"
    r"(?:what(?:'s|\s+is|\s+are)?|whats|calculate|compute|evaluate|work\s+out|how\s+much\s+is)\s+",
    re.I)

# after word-op substitution the remainder MUST be ONLY an arithmetic expression
_ARITH_OK = re.compile(r"^[0-9\.\+\-\*/%\(\)\s]+$")
_HAS_OP = re.compile(r"[\+\-\*/%]")

# explicit numeric comparison -- a classic model failure (9.11 vs 9.9)
_CMP = re.compile(
    r"^(?:which\s+(?:one\s+)?is\s+)?(?:the\s+)?"
    r"(bigger|larger|greater|more|highest|higher|smaller|less|lower|lowest)(?:\s+number)?[:,]?\s+"
    r"(-?\d+(?:\.\d+)?)\s*(?:or|vs\.?|versus|,|than)\s*(-?\d+(?:\.\d+)?)\s*\??$",
    re.I)


def _last_user(messages):
    for m in reversed(messages or []):
        if isinstance(m, dict) and (m.get("role") or "user") == "user":
            return (m.get("content") or "").strip()
    return ""


def _try_compare(raw: str):
    m = _CMP.match(raw)
    if not m:
        return None
    word = m.group(1).lower()
    a, b = float(m.group(2)), float(m.group(3))
    if a == b:
        return f"They are equal: {_fmt_num(a)}."
    low = word in ("smaller", "less", "lower", "lowest")
    return _fmt_num(min(a, b) if low else max(a, b))


def _try_arith(raw: str):
    expr = _ARITH_LEAD.sub("", raw.lower()).strip()
    expr = re.sub(r"\s*=\s*\??\s*$", "", expr)       # trailing "= ?" / "="
    expr = expr.rstrip(" ?.!")
    expr = expr.replace("^", "**")                   # caret -> power (ast ^ is XOR)
    expr = expr.replace(",", "")                     # 1,000 -> 1000
    for pat, op in _WORD_OPS:
        expr = re.sub(pat, op, expr)
    expr = expr.strip()
    if not expr or not _ARITH_OK.match(expr):        # any letter left -> not ours
        return None
    if not re.search(r"\d", expr):                   # must contain a number
        return None
    if not _HAS_OP.search(expr.replace("**", "*")):  # must be an operation, not a lone number
        return None
    try:
        return _fmt_num(_eval_arith(expr))
    except Exception:                                # div0, bad grammar, overflow -> fall through
        return None


def try_compute(messages):
    """The lane. Return an exact answer string, or None to fall through to a model.
    Fires ONLY on an unambiguous exact request in the LAST user message."""
    if not getattr(config, "COMPUTE_ON", True):
        return None
    raw = _last_user(messages)
    if not raw or len(raw) > 200:                    # exact asks are short; long text is prose
        return None
    for fn in (_try_compare, _try_arith):
        ans = fn(raw)
        if ans is not None:
            _hit()
            return ans
    return None
