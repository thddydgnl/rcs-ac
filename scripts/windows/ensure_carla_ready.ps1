param(
    [int]$CarlaPort = 2000,
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$StartScript = "C:\Users\thddy\Documents\code\RCS-AC-first\scripts\windows\start_carla_interactive.ps1",
    [int]$TimeoutSeconds = 180,
    [int]$StableSeconds = 8,
    [switch]$ForceRestart,
    [switch]$SkipPythonProbe
)

$ErrorActionPreference = "Stop"

function Test-CarlaPort {
    param([int]$Port)
    return (Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
}

function Get-CarlaProcesses {
    return @(Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue)
}

function Test-CarlaPythonProbe {
    param([int]$Port)

    if ($SkipPythonProbe) {
        return $true
    }

    $probeCode = "import carla; client = carla.Client('127.0.0.1', $Port); client.set_timeout(10.0); print(client.get_server_version()); print(client.get_world().get_map().name)"

    Push-Location $ProjectRoot
    try {
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $output = & py -3.12 -B -c $probeCode 2>&1
        $ErrorActionPreference = $previousErrorActionPreference
        if ($LASTEXITCODE -ne 0) {
            Write-Host "PYTHON_PROBE_FAILED"
            Write-Host $output
            return $false
        }
        Write-Host "PYTHON_PROBE_OK"
        Write-Host $output
        return $true
    }
    finally {
        Pop-Location
    }
}

function Test-CarlaHealthy {
    param([int]$Port)

    $processes = Get-CarlaProcesses
    if ($processes.Count -eq 0) {
        Write-Host "CARLA_HEALTH=NO_PROCESS"
        return $false
    }

    if (-not (Test-CarlaPort -Port $Port)) {
        Write-Host "CARLA_HEALTH=PORT_CLOSED"
        return $false
    }

    if (-not (Test-CarlaPythonProbe -Port $Port)) {
        Write-Host "CARLA_HEALTH=PYTHON_PROBE_FAILED"
        return $false
    }

    Write-Host "CARLA_HEALTH=OK"
    $processes | Select-Object Id,ProcessName,CPU,StartTime,Responding | Format-Table -AutoSize
    return $true
}

if ($ForceRestart -or -not (Test-CarlaHealthy -Port $CarlaPort)) {
    Write-Host "CARLA_RESTART_REQUIRED=True"
    & powershell -ExecutionPolicy Bypass -File $StartScript -CarlaPort $CarlaPort
}
else {
    Write-Host "CARLA_RESTART_REQUIRED=False"
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-CarlaHealthy -Port $CarlaPort) {
        Write-Host "CARLA_STABLE_WAIT_SECONDS=$StableSeconds"
        Start-Sleep -Seconds $StableSeconds
        if (Test-CarlaHealthy -Port $CarlaPort) {
            Write-Host "CARLA_READY=True"
            exit 0
        }
    }
    Start-Sleep -Seconds 2
}

Write-Host "CARLA_READY=False"
exit 1
