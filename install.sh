#!/usr/bin/env bash
# cleen.cpp installer (macOS/Linux). Free. Local-first.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
echo "cleen.cpp installer"
echo "==================="
echo "By installing, you agree to the Agreement & Data Terms:"
echo "  https://luxucleen.com/cleen/agreement.html"
echo "  Free local use; anonymized usage data helps cleen improve over time."
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "! Python 3 is required. Install it, then re-run."
  exit 1
fi
echo "[ok] python3: $(python3 --version 2>&1)"

if curl -fsS -m 4 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[ok] local runtime reachable (Ollama @ 11434)"
  if command -v ollama >/dev/null 2>&1; then
    echo "[..] pulling the default free model (qwen2.5-coder:7b) - one time"
    ollama pull qwen2.5-coder:7b || echo "[warn] pull skipped - pull it later with: ollama pull qwen2.5-coder:7b"
  fi
else
  echo "[!!] no local runtime found. For local mode, install Ollama: https://ollama.com"
  echo "     (cloud mode still works with a provider API key - see README.)"
fi

chmod +x "$HERE/bin/cleen" 2>/dev/null || true
echo
echo "[ok] cleen is ready. Add it to your PATH:"
echo "       export PATH=\"$HERE/bin:\$PATH\""
echo
echo "Then try:"
echo "       cleen doctor"
echo "       cleen run \"write a python function that reverses a string\""
echo
PYTHONPATH="$HERE:$PYTHONPATH" python3 -m cleen doctor || true
