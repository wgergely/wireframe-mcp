# .agent/scripts/claude-launch.ps1
# Claude Code launch script with interactive worktree selection.

# --- Branch Selection ---
$SelectedWorktree = & (Join-Path $PSScriptRoot "_branch-selector.ps1")
if (-not $SelectedWorktree) { exit 1 }

# Derive paths
$ProjectRoot = $SelectedWorktree
$WorktreesContainer = (Get-Item "$SelectedWorktree\..").FullName
# .env is ALWAYS loaded from the main branch
$MainBranch = Join-Path $WorktreesContainer "main"
$EnvFile = Join-Path $MainBranch ".env"
$VenvPath = Join-Path $WorktreesContainer "venv-win"

# --- Environment Setup ---
. (Join-Path $PSScriptRoot "_env-setup.ps1") -EnvFile $EnvFile -VenvPath $VenvPath

# Change to the selected worktree
Set-Location $ProjectRoot

Write-Host "Launching Claude Code for $ProjectRoot..." -ForegroundColor Green
& "claude"
