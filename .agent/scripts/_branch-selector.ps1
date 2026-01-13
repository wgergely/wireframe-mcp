# .agent/scripts/_branch-selector.ps1
# Interactive worktree selector using fzf.
# Returns the selected worktree path.

# From .agent/scripts, go up 3 levels: scripts -> .agent -> main -> wireframe-mcp-worktrees
$WorktreesContainer = (Get-Item "$PSScriptRoot\..\..\..").FullName

# Scan for worktree directories, excluding venv-* and hidden directories
$Worktrees = Get-ChildItem -Path $WorktreesContainer -Directory |
             Where-Object { $_.Name -notmatch "^venv-" -and $_.Name -notmatch "^\." } |
             Select-Object -ExpandProperty Name

if (-not $Worktrees) {
    Write-Error "No worktrees found in $WorktreesContainer"
    exit 1
}

# Use fzf for selection
$Selected = $Worktrees | fzf --prompt="Select worktree: " --height=10 --reverse

if (-not $Selected) {
    Write-Warning "No selection made. Exiting."
    exit 1
}

$SelectedPath = Join-Path $WorktreesContainer $Selected
Write-Host "Selected worktree: $SelectedPath" -ForegroundColor Green

# Return the path for the caller to use
return $SelectedPath
