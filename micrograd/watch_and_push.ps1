# ============================================================
#  watch_and_push.ps1
#  Watches micrograd/*.md for changes and auto-pushes to GitHub
#  Repo: https://github.com/satyabhan007/AI-ML
#  Strategy: polls LastWriteTime every 2s with datetime debounce
# ============================================================

$WatchDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir    = "C:\Users\satya\.gemini\antigravity-ide\brain\06154d7f-e65f-424e-9e45-8433ee08ae92\scratch\ai-ml-repo"
$RemoteUrl  = "https://github.com/satyabhan007/AI-ML.git"
$DebounceS  = 3   # seconds stable before pushing

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI-ML Auto-Push Watcher" -ForegroundColor Cyan
Write-Host "  Watching: $WatchDir\*.md" -ForegroundColor Cyan
Write-Host "  Repo:     $RemoteUrl" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ensure clone exists
if (-not (Test-Path "$RepoDir\.git")) {
    Write-Host "[INIT] Cloning repo..." -ForegroundColor Yellow
    git clone $RemoteUrl $RepoDir
}

function Push-File([string]$FilePath) {
    $FileName = Split-Path -Leaf $FilePath
    if (-not (Test-Path $FilePath)) { return }

    Write-Host ""
    Write-Host "[PUSH] $FileName  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Magenta
    Copy-Item -Path $FilePath -Destination "$RepoDir\$FileName" -Force

    Push-Location $RepoDir
    try {
        git pull --rebase --quiet origin main 2>&1 | Out-Null
        git add $FileName 2>&1 | Out-Null
        $st = git status --porcelain $FileName
        if ($st) {
            $ts  = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $msg = "auto: update $FileName ($ts)"
            git commit -m $msg 2>&1 | Out-Null
            git push origin main 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  [OK] $msg" -ForegroundColor Green
            } else {
                Write-Host "  [WARN] push failed" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  [SKIP] no diff" -ForegroundColor DarkGray
        }
    } finally {
        Pop-Location
    }
}

# Initial snapshot: file path -> LastWriteTime
$snapshot = @{}
Get-ChildItem -Path $WatchDir -Filter "*.md" | ForEach-Object {
    $snapshot[$_.FullName] = $_.LastWriteTime
}

Write-Host "Polling every 2 s... Ctrl+C to stop" -ForegroundColor Green
Write-Host "Tracking $($snapshot.Count) .md file(s)" -ForegroundColor DarkGray
Write-Host ""

# pending: file path -> [datetime] when first dirtied
$pending = @{}

while ($true) {
    Start-Sleep -Seconds 2

    # Detect changes
    Get-ChildItem -Path $WatchDir -Filter "*.md" | ForEach-Object {
        $path = $_.FullName
        $lwt  = $_.LastWriteTime
        $prev = $snapshot[$path]

        if (($null -eq $prev) -or ($lwt -ne $prev)) {
            $snapshot[$path] = $lwt
            if (-not $pending.ContainsKey($path)) {
                $pending[$path] = [datetime]::Now
                Write-Host "[DIRTY] $(Split-Path -Leaf $path)  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
            }
        }
    }

    # Push anything stable for >= $DebounceS seconds
    $now    = [datetime]::Now
    $toPush = @($pending.Keys | Where-Object {
        ($now - $pending[$_]).TotalSeconds -ge $DebounceS
    })

    foreach ($path in $toPush) {
        $pending.Remove($path)
        Push-File $path
    }
}
