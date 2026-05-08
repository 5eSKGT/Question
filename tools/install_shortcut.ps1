# =============================================================================
#  Create a desktop shortcut named "AlphaPulse" with the project icon.
#
#  Run once:
#     powershell -ExecutionPolicy Bypass -File tools\install_shortcut.ps1
#
#  Result:
#     - %USERPROFILE%\Desktop\AlphaPulse.lnk
#     - Target: run_alphapulse.bat (silent launcher → pythonw -m crypto_trend)
#     - Icon  : crypto_trend\desktop\assets\icon.ico
#     - Working dir: project root, so git pull updates take effect
#
#  After this you double-click the desktop icon like a normal app, and to
#  update the code you just `git pull` — no rebuild needed.
# =============================================================================
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bat      = Join-Path $repoRoot "run_alphapulse.bat"
$icon     = Join-Path $repoRoot "crypto_trend\desktop\assets\icon.ico"

if (-not (Test-Path $bat))  { throw "missing $bat — run from inside the cloned repo" }
if (-not (Test-Path $icon)) { throw "missing $icon — run python tools\generate_assets.py first" }

$desktop = [Environment]::GetFolderPath('Desktop')
$lnk     = Join-Path $desktop "AlphaPulse.lnk"

$ws = New-Object -ComObject WScript.Shell
$s  = $ws.CreateShortcut($lnk)
$s.TargetPath       = $bat
$s.WorkingDirectory = $repoRoot
$s.IconLocation     = "$icon, 0"
$s.WindowStyle      = 7        # 7 = minimized; the bat exits fast anyway
$s.Description      = "AlphaPulse · Crypto Trend Following — Bitget"
$s.Save()

Write-Host ""
Write-Host "✓ shortcut created" -ForegroundColor Green
Write-Host "  $lnk"
Write-Host ""
Write-Host "  더블클릭으로 GUI 실행. 업데이트는 그냥 `git pull` 만 하면 됩니다."
