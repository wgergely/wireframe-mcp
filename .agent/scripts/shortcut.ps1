# .agent/scripts/shortcut.ps1
# Creates or updates Windows Terminal profiles for the current repository.

# From .agent/scripts, go up 2 levels: scripts -> .agent -> main
$ProjectRoot = (Get-Item "$PSScriptRoot\..\..").FullName
# Worktrees container is one level above the repo
$ParentDir = (Get-Item "$ProjectRoot\..").FullName
$RepoName = (Split-Path $ParentDir -Leaf).Replace("-worktrees", "")

$LaunchScript = Join-Path $ProjectRoot ".agent\scripts\launch.ps1"
$ProfileName = "Antigravity -- $RepoName"

# Find Windows Terminal settings.json
$SettingsPath = Get-ChildItem -Path "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal*\LocalState\settings.json" -ErrorAction SilentlyContinue | 
                 Select-Object -First 1 -ExpandProperty FullName

if (-not $SettingsPath) {
    Write-Error "Windows Terminal settings.json not found."
    exit 1
}

Write-Host "Found settings at: $SettingsPath" -ForegroundColor Gray

# Create a backup
$BackupPath = "$SettingsPath.bak"
if (-not (Test-Path $BackupPath)) {
    Copy-Item $SettingsPath $BackupPath -ErrorAction SilentlyContinue
    Write-Host "Backup created at: $BackupPath" -ForegroundColor Cyan
}

try {
    # Load settings as JSON
    $Content = Get-Content $SettingsPath -Raw
    $Settings = $Content | ConvertFrom-Json

    # Dynamically find existing Antigravity info from ANY profile
    $ExistingAntigravityProfile = $Settings.profiles.list | Where-Object { $_.name -like "*Antigravity*" -and $_.icon } | Select-Object -First 1
    
    $IconPath = "C:\Users\hello\AppData\Local\Programs\Antigravity\antigravity.exe" # Fallback
    $ColorScheme = "GitHub Dark" # Fallback

    if ($ExistingAntigravityProfile) {
        $IconPath = $ExistingAntigravityProfile.icon
        if ($ExistingAntigravityProfile.colorScheme) { $ColorScheme = $ExistingAntigravityProfile.colorScheme }
        Write-Host "Discovered existing Antigravity config: Icon=$IconPath, Scheme=$ColorScheme" -ForegroundColor Cyan
    }

    # Function to create or update a profile
    function Add-Or-Update-Profile {
        param(
            [Parameter(Mandatory=$true)] $Settings,
            [Parameter(Mandatory=$true)] $Name,
            [Parameter(Mandatory=$true)] $LaunchScript,
            [Parameter(Mandatory=$true)] $Icon,
            [Parameter(Mandatory=$true)] $ColorScheme
        )

        $ProfileProps = [PSCustomObject]@{
            name = $Name
            commandline = "pwsh.exe -NoExit -ExecutionPolicy Bypass -File `"$LaunchScript`""
            startingDirectory = $ProjectRoot
            icon = $Icon
            hidden = $false
            colorScheme = $ColorScheme
        }

        # Search for existing profile and remove it if it exists
        $ExistingIndices = 0..($Settings.profiles.list.Count - 1) | Where-Object { $Settings.profiles.list[$_].name -eq $Name }
        
        if ($ExistingIndices) {
            Write-Host "Updating existing profile: $Name" -ForegroundColor Cyan
            $NewList = New-Object System.Collections.Generic.List[PSCustomObject]
            for ($i=0; $i -lt $Settings.profiles.list.Count; $i++) {
                if ($i -notin $ExistingIndices) {
                    $NewList.Add($Settings.profiles.list[$i])
                }
            }
            $Settings.profiles.list = $NewList
        } else {
            Write-Host "Creating new profile: $Name" -ForegroundColor Green
        }

        # Add the new/updated profile
        $Settings.profiles.list += $ProfileProps
    }

    # Add Antigravity Profile
    Add-Or-Update-Profile -Settings $Settings `
                          -Name $ProfileName `
                          -LaunchScript $LaunchScript `
                          -Icon $IconPath `
                          -ColorScheme $ColorScheme

    # Add Claude Profile
    $ClaudeLaunchScript = Join-Path $ProjectRoot ".agent\scripts\claude-launch.ps1"
    $ClaudeProfileName = "Claude -- $RepoName"
    $ClaudeIconPath = "C:\Users\hello\.local\bin\claude.exe"
    Add-Or-Update-Profile -Settings $Settings `
                          -Name $ClaudeProfileName `
                          -LaunchScript $ClaudeLaunchScript `
                          -Icon $ClaudeIconPath `
                          -ColorScheme $ColorScheme

    # Convert back to JSON and save
    $UpdatedJson = $Settings | ConvertTo-Json -Depth 10
    $UpdatedJson | Set-Content $SettingsPath -Encoding utf8

    Write-Host "Successfully updated Windows Terminal profiles." -ForegroundColor Green
    Write-Host "Restart Windows Terminal or check the dropdown menu to see your new profile." -ForegroundColor Green

} catch {
    Write-Error "Failed to update settings.json: $_"
    exit 1
}
