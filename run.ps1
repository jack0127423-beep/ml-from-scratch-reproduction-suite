$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.matplotlib'
$env:LOKY_MAX_CPU_COUNT = '4'

if (-not (Test-Path '.venv')) {
    py -3.12 -m venv .venv
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' run_all.py
