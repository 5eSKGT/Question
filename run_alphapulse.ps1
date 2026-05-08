# =============================================================================
#  AlphaPulse — PowerShell launcher
#
#  Usage:
#     .\run_alphapulse.ps1                # just launch
#     .\run_alphapulse.ps1 -Update        # git pull, then launch
#     .\run_alphapulse.ps1 -Install       # install/update Python deps
#     .\run_alphapulse.ps1 -Update -Debug # update, then launch with console
# =============================================================================
param(
    [switch]$Update,
    [switch]$Install,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Pick interpreter — prefer venv if present.
$venvPy  = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$venvPyw = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
if (Test-Path $venvPy) {
    $py  = $venvPy
    $pyw = $venvPyw
} else {
    $py  = "python"
    $pyw = "pythonw"
}

if ($Update) {
    Write-Host "→ git pull" -ForegroundColor Cyan
    git pull
    if ($LASTEXITCODE -ne 0) { throw "git pull failed" }
}

if ($Install) {
    Write-Host "→ pip install -r requirements.txt" -ForegroundColor Cyan
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

Write-Host "→ launching AlphaPulse" -ForegroundColor Green
if ($Debug) {
    & $py -m crypto_trend
    Read-Host "Press Enter to exit"
} else {
    Start-Process -FilePath $pyw -ArgumentList "-m", "crypto_trend" `
                  -WorkingDirectory $PSScriptRoot
}
