# .agent/scripts/_env-setup.ps1
# Shared utility to load .env and activate Python virtual environment.
# Supports both standard venv and conda environments.

Param(
    [Parameter(Mandatory=$true)][string]$EnvFile,
    [Parameter(Mandatory=$true)][string]$VenvPath
)

function Activate-Venv {
    param([string]$Path)

    # Check for conda environment (has conda-meta directory)
    $CondaMeta = Join-Path $Path "conda-meta"
    if (Test-Path $CondaMeta) {
        Write-Host "Activating conda environment from $Path" -ForegroundColor Cyan
        # Prepend to PATH and set conda env vars (minimal activation)
        $env:PATH = "$Path;$Path\Library\bin;$Path\Scripts;$env:PATH"
        $env:CONDA_PREFIX = $Path
        $env:CONDA_DEFAULT_ENV = (Split-Path $Path -Leaf)
        return $true
    }

    # Standard venv activation
    $ActivateScript = Join-Path $Path "Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        Write-Host "Activating virtual environment from $Path" -ForegroundColor Cyan
        . $ActivateScript
        return $true
    }
    return $false
}

function Load-EnvFile {
    param([string]$Path)
    if (Test-Path $Path) {
        Write-Host "Loading environment from $Path" -ForegroundColor Cyan
        Get-Content $Path | ForEach-Object {
            $line = $_.Trim()
            if ($line -and -not $line.StartsWith("#") -and $line -match "=") {
                $key, $value = $line -split "=", 2
                [System.Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), "Process")
            }
        }
        return $true
    }
    return $false
}

# Execution Flow
if (-not (Activate-Venv -Path $VenvPath)) {
    Write-Warning "Virtual environment not found or failed to activate at $VenvPath"
}

if (-not (Load-EnvFile -Path $EnvFile)) {
    Write-Warning ".env file not found at $EnvFile"
}
