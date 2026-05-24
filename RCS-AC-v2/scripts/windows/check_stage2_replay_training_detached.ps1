param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay",
    [string]$LogPath = "",
    [int]$ProcessId = 0,
    [string]$TaskName = "",
    [int]$Tail = 20
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
if ($LogPath -eq "") {
    $LogPath = Join-Path $OutputDir "train_stage2_privileged_from_replay.log"
}

$ManifestPath = Join-Path $OutputDir "stage2_privileged_replay_checkpoint_manifest.json"
$LogExists = Test-Path $LogPath
$ManifestExists = Test-Path $ManifestPath
$Running = $false
if ($ProcessId -gt 0) {
    $Running = $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}
$TaskState = "UNSET"
if ($TaskName -ne "") {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $Task) {
        $TaskState = [string]$Task.State
        $Running = $Running -or ($TaskState -eq "Running")
    }
    else {
        $TaskState = "MISSING"
    }
}

$ExitCode = "UNSET"
if ($LogExists) {
    $ExitLine = Get-Content $LogPath | Where-Object { $_ -like "TRAIN_EXIT_CODE=*" } | Select-Object -Last 1
    if ($null -ne $ExitLine) {
        $ExitCode = ($ExitLine -replace "^TRAIN_EXIT_CODE=", "").Trim()
    }
}

$ReadyForProbe = ($ManifestExists -and $ExitCode -eq "0")
Write-Host "TRAIN_OUTPUT_DIR=$OutputDir"
Write-Host "TRAIN_LOG_PATH=$LogPath"
Write-Host "TRAIN_LOG_EXISTS=$LogExists"
Write-Host "TRAIN_MANIFEST_PATH=$ManifestPath"
Write-Host "TRAIN_MANIFEST_EXISTS=$ManifestExists"
Write-Host "TRAIN_PROCESS_ID=$ProcessId"
Write-Host "TRAIN_TASK_NAME=$TaskName"
Write-Host "TRAIN_TASK_STATE=$TaskState"
Write-Host "TRAIN_RUNNING=$Running"
Write-Host "TRAIN_EXIT_CODE=$ExitCode"
Write-Host "TRAIN_READY_FOR_PROBE=$ReadyForProbe"
if ($LogExists -and $Tail -gt 0) {
    Write-Host "TRAIN_LOG_TAIL_BEGIN"
    Get-Content $LogPath -Tail $Tail
    Write-Host "TRAIN_LOG_TAIL_END"
}

if ($ReadyForProbe) {
    exit 0
}
if ($Running) {
    exit 3
}
exit 2
