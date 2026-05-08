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
#
#  Notes:
#   * If you previously created a shortcut and Windows still shows the old
#     icon, this script flushes the per-user icon cache so the new icon
#     takes effect on the next login (or after restarting Explorer).
#   * The taskbar/alt-tab icon is set by the application itself via
#     SetCurrentProcessExplicitAppUserModelID — see crypto_trend/desktop/app.py.
# =============================================================================
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bat      = Join-Path $repoRoot "run_alphapulse.bat"
$icon     = Join-Path $repoRoot "crypto_trend\desktop\assets\icon.ico"

if (-not (Test-Path $bat))  { throw "missing $bat — run from inside the cloned repo" }
if (-not (Test-Path $icon)) {
    Write-Host "icon not found — generating it now…" -ForegroundColor Yellow
    & python (Join-Path $repoRoot "tools\generate_assets.py")
}

$desktop = [Environment]::GetFolderPath('Desktop')
$lnk     = Join-Path $desktop "AlphaPulse.lnk"

# If a stale shortcut exists, delete it first so Windows reads the icon fresh.
if (Test-Path $lnk) { Remove-Item $lnk -Force }

$ws = New-Object -ComObject WScript.Shell
$s  = $ws.CreateShortcut($lnk)
$s.TargetPath       = $bat
$s.WorkingDirectory = $repoRoot
$s.IconLocation     = "$icon, 0"
$s.WindowStyle      = 7        # 7 = minimized; the bat exits fast anyway
$s.Description      = "AlphaPulse · Crypto Trend Following — Bitget"
$s.Save()

# Flush the per-user icon cache so Explorer picks up the new icon.
$cacheRoot = Join-Path $env:LOCALAPPDATA "Microsoft\Windows\Explorer"
if (Test-Path $cacheRoot) {
    Get-ChildItem $cacheRoot -Filter "iconcache_*.db" -ErrorAction SilentlyContinue |
        ForEach-Object {
            try { Remove-Item $_.FullName -Force -ErrorAction Stop } catch { }
        }
}

Write-Host ""
Write-Host "✓ shortcut created" -ForegroundColor Green
Write-Host "  $lnk"
Write-Host ""
Write-Host "  더블클릭으로 GUI 실행. 업데이트는 그냥 ``git pull`` 만 하면 됩니다."
Write-Host ""
Write-Host "  아이콘이 여전히 기본 아이콘으로 보이면 다음 중 하나를 시도하세요:" -ForegroundColor Yellow
Write-Host "    1) 작업표시줄에서 Explorer 재시작 (Ctrl+Shift+Esc → Windows 탐색기 → 다시 시작)"
Write-Host "    2) 또는 한번 로그아웃 후 다시 로그인"
