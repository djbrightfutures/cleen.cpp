@echo off
rem cleen.cpp launcher (Windows) - runs from anywhere, keeps your working dir.
setlocal
set "CLEEN_HOME=%~dp0"
set "PYTHONPATH=%CLEEN_HOME%;%PYTHONPATH%"
python -m cleen %*
