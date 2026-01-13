# .agent/launch.ps1
# Load .env file from project root
$ProjectRoot = (Get-Item "$PSScriptRoot\..").FullName
$EnvFile = Join-Path $ProjectRoot ".env"

if (Test-Path $EnvFile) {
    Write-Host "Loading environment from $EnvFile" -ForegroundColor Cyan
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line -match "=") {
            $key, $value = $line -split "=", 2
            [System.Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), "Process")
        }
    }
} else {
    Write-Warning ".env file not found at $EnvFile"
}

$ProfilePath = Join-Path $ProjectRoot ".agent\profile"
$AntigravityExecutable = "antigravity"

& $AntigravityExecutable --log warn --sync off --user-data-dir "$ProfilePath" "$ProjectRoot"

# Give the process a moment to hand off
Start-Sleep -Seconds 1

$LogsDir = Join-Path $ProfilePath "logs"
if (Test-Path $LogsDir) {
    # Find the main.log that was modified MOST recently across ALL subdirectories
    # This correctly finds the running instance even if the current command created a new empty log folder
    $ActiveLog = Get-ChildItem -Path $LogsDir -Filter "main.log" -Recurse | 
                 Sort-Object LastWriteTime -Descending | 
                 Select-Object -First 1
    
    if ($ActiveLog) {
        Write-Host "`n--- Found Active Session: $($ActiveLog.Directory.Name) ---" -ForegroundColor Cyan
        Write-Host "Tailing logs from: $($ActiveLog.FullName)" -ForegroundColor Gray
        Write-Host "(Terminal will stay open to show live updates. Press Ctrl+C to stop tailing)`n" -ForegroundColor Gray
        
        # Stream the logs. Use -Tail to see recent history.
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
