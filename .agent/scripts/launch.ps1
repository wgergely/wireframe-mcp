# .agent/scripts/launch.ps1
# Antigravity launch script with interactive worktree selection.

Param(
    [string]$LogLevel = "warn"
)

# --- Branch Selection ---
$SelectedWorktree = & (Join-Path $PSScriptRoot "_branch-selector.ps1")
if (-not $SelectedWorktree) { exit 1 }

# Derive paths
$ProjectRoot = $SelectedWorktree
$WorktreesContainer = (Get-Item "$SelectedWorktree\..").FullName
# .env is ALWAYS loaded from the main branch
$MainBranch = Join-Path $WorktreesContainer "main"
$EnvFile = Join-Path $MainBranch ".env"
$VenvPath = Join-Path $WorktreesContainer ".venv"

# --- Environment Setup ---
. (Join-Path $PSScriptRoot "_env-setup.ps1") -EnvFile $EnvFile -VenvPath $VenvPath

$ProfilePath = Join-Path $ProjectRoot ".agent\profile"
$AntigravityExecutable = "antigravity"

Write-Host "Launching Antigravity for $ProjectRoot..." -ForegroundColor Green
& $AntigravityExecutable --log $LogLevel --sync off --user-data-dir "$ProfilePath" "$ProjectRoot"

# Give the process a moment to hand off
Start-Sleep -Seconds 1

$LogsDir = Join-Path $ProfilePath "logs"
if (Test-Path $LogsDir) {
    # Find the main.log that was modified MOST recently across ALL subdirectories
    $ActiveLog = Get-ChildItem -Path $LogsDir -Filter "main.log" -Recurse | 
                 Sort-Object LastWriteTime -Descending | 
                 Select-Object -First 1
    
    if ($ActiveLog) {
        Write-Host "`n--- Found Active Session: $($ActiveLog.Directory.Name) ---" -ForegroundColor Cyan
        Write-Host "Tailing logs from: $($ActiveLog.FullName)" -ForegroundColor Gray
        Write-Host "(Terminal will stay open to show live updates. Press Ctrl+C to stop tailing)`n" -ForegroundColor Gray
        
        # Stream the logs.
        Get-Content -Path $ActiveLog.FullName -Tail 50 -Wait
    } else {
        Write-Warning "No active main.log found in $LogsDir. Ensure Antigravity is actually running."
    }
} else {
    Write-Warning "Logs directory not found at $LogsDir"
}

# Fallback/Persistence
Write-Host "`nLog stream disconnected." -ForegroundColor Yellow
Read-Host "Press Enter to exit the terminal"
