param(
    [string]$TaskName = "RCSCarlaStart",
    [string]$CarlaExe = "C:\Users\thddy\Documents\CARLA_0.9.16\CarlaUE4.exe",
    [int]$CarlaPort = 2000,
    [string]$QualityLevel = "Low",
    [int]$ResX = 800,
    [int]$ResY = 600,
    [int]$StartupTimeoutSeconds = 180,
    [int]$PostPortSettleSeconds = 10,
    [string]$InteractiveUser = "song\song"
)

$ErrorActionPreference = "Stop"

Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

if (-not (Test-Path $CarlaExe)) {
    throw "CARLA executable not found: $CarlaExe"
}

$arguments = "-carla-rpc-port=$CarlaPort -quality-level=$QualityLevel -nosound -windowed -ResX=$ResX -ResY=$ResY"
$workingDirectory = Split-Path -Parent $CarlaExe

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction -Execute $CarlaExe -Argument $arguments -WorkingDirectory $workingDirectory
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$principal = New-ScheduledTaskPrincipal -UserId $InteractiveUser -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Host "started scheduled task $TaskName for $InteractiveUser"

for ($index = 0; $index -lt $StartupTimeoutSeconds; $index++) {
    $portOpen = (Test-NetConnection -ComputerName 127.0.0.1 -Port $CarlaPort -WarningAction SilentlyContinue).TcpTestSucceeded
    if ($portOpen) {
        Write-Host "PORT_OPEN_AT_SECOND=$index"
        break
    }
    if (($index % 15) -eq 0) {
        Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue |
            Select-Object Id,ProcessName,CPU,Responding |
            Format-Table -AutoSize
    }
    Start-Sleep -Seconds 1
}

$finalPortOpen = (Test-NetConnection -ComputerName 127.0.0.1 -Port $CarlaPort -WarningAction SilentlyContinue).TcpTestSucceeded
Write-Host "FINAL_PORT_OPEN=$finalPortOpen"
Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue |
    Select-Object Id,ProcessName,CPU,StartTime,Responding |
    Format-Table -AutoSize

if (-not $finalPortOpen) {
    throw "CARLA did not open port $CarlaPort within $StartupTimeoutSeconds seconds."
}

if ($PostPortSettleSeconds -gt 0) {
    Write-Host "POST_PORT_SETTLE_SECONDS=$PostPortSettleSeconds"
    Start-Sleep -Seconds $PostPortSettleSeconds
}
