# ============================================================
#  watch_and_push.ps1 — Persistent AI-ML Auto-Sync Watcher
#  Watches: micrograd, tokenizer, lab, assets, .github, root files
#  Target:  https://github.com/satyabhan007/AI-ML
#  Repo:    D:\test\account rotate\AI-ML
# ============================================================

$WorkspaceRoot = "D:\test\account rotate\account_rotator"
$RepoDir       = "D:\test\account rotate\AI-ML"
$RemoteUrl     = "https://github.com/satyabhan007/AI-ML.git"
$LogFile       = "$RepoDir\auto_sync.log"
$DebounceSec   = 3

# Subdirectories to watch relative to workspace
$WatchFolders  = @("micrograd", "tokenizer", "lab", "assets", ".github")
$WatchedExts   = @(".md", ".py", ".html", ".css", ".js", ".json", ".yml")

# Root-level files to watch (directly in workspace root)
$WatchRootFiles = @("index.html", "404.html", ".gitignore")

function Log-Message([string]$msg, [string]$color = "White") {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue
}

Log-Message "========================================" "Cyan"
Log-Message "  AI-ML Persistent Auto-Push Watcher" "Cyan"
Log-Message "  Workspace: $WorkspaceRoot" "Cyan"
Log-Message "  Target:    $RepoDir" "Cyan"
Log-Message "========================================" "Cyan"

# Ensure local clone exists
if (-not (Test-Path "$RepoDir\.git")) {
    Log-Message "[INIT] Cloning repo from $RemoteUrl..." "Yellow"
    git clone $RemoteUrl $RepoDir
}

function Sync-And-Push([string]$SourcePath) {
    if (-not (Test-Path $SourcePath)) { return }
    $relPath = $SourcePath.Substring($WorkspaceRoot.Length).TrimStart("\/")
    $parts = $relPath -split "[\\/]"
        $fileName = Split-Path -Leaf $SourcePath

    Log-Message "[SYNC] Detected change in $relPath" "Magenta"

        # Determine the relative directory path (everything except the filename)
    if ($parts.Count -gt 1) {
        $relDir = ($parts[0..($parts.Count - 2)]) -join "\"
        $destFolder = "$RepoDir\$relDir"
    } else {
        $destFolder = $RepoDir
    }
    # File is in a subdirectory (e.g., micrograd/engine.py, assets/css/style.css)
    if (-not (Test-Path $destFolder)) {
        New-Item -ItemType Directory -Path $destFolder -Force | Out-Null
    }
    Copy-Item -Path $SourcePath -Destination "$destFolder\$fileName" -Force

    # If it's a markdown explainer from micrograd, also mirror to repo root
        if ($parts[0] -eq "micrograd" -and $fileName.EndsWith(".md")) {
        Copy-Item -Path $SourcePath -Destination "$RepoDir\$fileName" -Force
    }

    Push-Location $RepoDir
    try {
        git pull --rebase --quiet origin main 2>&1 | Out-Null
        git add -A 2>&1 | Out-Null
        $st = git status --porcelain
        if ($st) {
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $msg = "auto: update $relPath ($ts)"
            git commit -m $msg 2>&1 | Out-Null
            git push origin main 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Log-Message "  [OK] Successfully pushed: $msg" "Green"
            } else {
                Log-Message "  [WARN] Git push failed (will retry next save)" "Yellow"
            }
        } else {
            Log-Message "  [SKIP] No git diff detected" "DarkGray"
        }
    } catch {
        Log-Message "  [ERROR] Sync failed: $_" "Red"
    } finally {
        Pop-Location
    }
}

# Collect all files to track
function Get-TrackedFiles {
    $files = @()
    foreach ($f in $WatchFolders) {
        $folderPath = "$WorkspaceRoot\$f"
        if (Test-Path $folderPath) {
            Get-ChildItem -Path $folderPath -Recurse -File | Where-Object {
                $ext = [System.IO.Path]::GetExtension($_.FullName).ToLower()
                $WatchedExts -contains $ext
            } | ForEach-Object {
                $files += $_
            }
        }
    }
    # Also track root-level files
    foreach ($f in $WatchRootFiles) {
        $filePath = "$WorkspaceRoot\$f"
        if (Test-Path $filePath) {
            $files += Get-Item $filePath
        }
    }
    return $files
}

# Snapshot: FullName -> LastWriteTime
$snapshot = @{}
Get-TrackedFiles | ForEach-Object {
    $snapshot[$_.FullName] = $_.LastWriteTime
}

Log-Message "Watching $($snapshot.Count) files across: $($WatchFolders -join ', ')" "Green"
Log-Message "Polling interval: 2 seconds. Ready." "Green"

# Pending: FullName -> [datetime] when first changed
$pending = @{}

while ($true) {
    Start-Sleep -Seconds 2

    # Check for new or modified files
    $currentFiles = Get-TrackedFiles
    foreach ($file in $currentFiles) {
        $path = $file.FullName
        $lwt  = $file.LastWriteTime
        $prev = $snapshot[$path]

        if (($null -eq $prev) -or ($lwt -ne $prev)) {
            $snapshot[$path] = $lwt
            if (-not $pending.ContainsKey($path)) {
                $pending[$path] = [datetime]::Now
                $rel = $path.Substring($WorkspaceRoot.Length).TrimStart("\/")
                Log-Message "[DIRTY] $rel" "Yellow"
            }
        }
    }

    # Process files that have been stable for >= $DebounceSec seconds
    $now = [datetime]::Now
    $ready = @($pending.Keys | Where-Object {
        ($now - $pending[$_]).TotalSeconds -ge $DebounceSec
    })

    foreach ($path in $ready) {
        $pending.Remove($path)
        Sync-And-Push $path
    }
}
