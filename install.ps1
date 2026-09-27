# cleen.cpp installer (Windows PowerShell). Free. Local-first.
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "cleen.cpp installer"
Write-Host "==================="
Write-Host "By installing, you agree to the Agreement & Data Terms:"
Write-Host "  https://luxucleen.com/cleen/agreement.html"
Write-Host "  Free local use; anonymized usage data helps cleen improve over time."
Write-Host ""

# python
try {
    $pv = (python --version) 2>&1
    Write-Host "[ok] $pv"
} catch {
    Write-Host "! Python 3 is required. Install it from https://python.org, then re-run."
    exit 1
}

# local runtime
$localUp = $false
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 4 | Out-Null
    $localUp = $true
} catch {}

if ($localUp) {
    Write-Host "[ok] local runtime reachable (Ollama @ 11434)"
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Write-Host "[..] pulling the default free model (qwen2.5-coder:7b) - one time"
        ollama pull qwen2.5-coder:7b
    }
} else {
    Write-Host "[!!] no local runtime found. For local mode, install Ollama: https://ollama.com"
    Write-Host "     (cloud mode still works with a provider API key - see README.)"
}

$env:PYTHONPATH = "$Here;$env:PYTHONPATH"
Write-Host ""
Write-Host "[ok] cleen is ready. Add it to your PATH:"
Write-Host "       `$env:Path = `"$Here\bin;`$env:Path`""
Write-Host ""
Write-Host "Then try:"
Write-Host "       cleen doctor"
Write-Host "       cleen run `"write a python function that reverses a string`""
Write-Host ""
python -m cleen doctor
