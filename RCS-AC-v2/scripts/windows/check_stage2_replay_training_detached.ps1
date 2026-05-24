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
$LaunchManifestPath = Join-Path $OutputDir "train_stage2_privileged_from_replay.launch.json"
if (Test-Path $LaunchManifestPath) {
    try {
        $LaunchManifest = Get-Content $LaunchManifestPath -Raw | ConvertFrom-Json
        if ($LogPath -eq "" -and $null -ne $LaunchManifest.log_path) {
            $LogPath = [string]$LaunchManifest.log_path
        }
        if ($TaskName -eq "" -and $null -ne $LaunchManifest.task_name) {
            $TaskName = [string]$LaunchManifest.task_name
        }
    }
    catch {
        Write-Host "TRAIN_LAUNCH_MANIFEST_PARSE_ERROR=$($_.Exception.Message)"
    }
}
if ($LogPath -eq "") {
    $LogPath = Join-Path $OutputDir "train_stage2_privileged_from_replay.log"
}

$ManifestPath = Join-Path $OutputDir "stage2_privileged_replay_checkpoint_manifest.json"
$LogExists = Test-Path $LogPath
$ManifestExists = Test-Path $ManifestPath
$ManifestValid = $false
$CheckpointFilesValid = $false
$ManifestValidationError = "UNSET"
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
    $LogText = [System.IO.File]::ReadAllText((Resolve-Path $LogPath).Path) -replace "`0", ""
    $ExitLine = $LogText -split "\r?\n" | Where-Object { $_ -like "TRAIN_EXIT_CODE=*" } | Select-Object -Last 1
    if ($null -ne $ExitLine) {
        $ExitCode = ($ExitLine -replace "^TRAIN_EXIT_CODE=", "").Trim()
    }
}

function Resolve-ManifestFile {
    param([string]$RawPath)
    if ([System.IO.Path]::IsPathRooted($RawPath)) {
        return $RawPath
    }
    return Join-Path $ProjectRoot $RawPath
}

if ($ManifestExists) {
    try {
        $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
        $WorldModelPath = Resolve-ManifestFile ([string]$Manifest.world_model.path)
        $PolicyPath = Resolve-ManifestFile ([string]$Manifest.policy.path)
        if (-not (Test-Path $WorldModelPath)) {
            throw "world_model checkpoint missing: $WorldModelPath"
        }
        if (-not (Test-Path $PolicyPath)) {
            throw "policy checkpoint missing: $PolicyPath"
        }
        $WorldModelSha = (Get-FileHash -Algorithm SHA256 $WorldModelPath).Hash.ToLowerInvariant()
        $PolicySha = (Get-FileHash -Algorithm SHA256 $PolicyPath).Hash.ToLowerInvariant()
        if ($WorldModelSha -ne ([string]$Manifest.world_model.sha256).ToLowerInvariant()) {
            throw "world_model sha256 mismatch"
        }
        if ($PolicySha -ne ([string]$Manifest.policy.sha256).ToLowerInvariant()) {
            throw "policy sha256 mismatch"
        }
        $ManifestValid = $true
        $CheckpointFilesValid = $true
        $ManifestValidationError = "none"
    }
    catch {
        $ManifestValidationError = $_.Exception.Message
    }
}

$ReadyForProbe = ($ManifestExists -and $ManifestValid -and $CheckpointFilesValid -and $ExitCode -eq "0")
Write-Host "TRAIN_OUTPUT_DIR=$OutputDir"
Write-Host "TRAIN_LAUNCH_MANIFEST_PATH=$LaunchManifestPath"
Write-Host "TRAIN_LAUNCH_MANIFEST_EXISTS=$(Test-Path $LaunchManifestPath)"
Write-Host "TRAIN_LOG_PATH=$LogPath"
Write-Host "TRAIN_LOG_EXISTS=$LogExists"
Write-Host "TRAIN_MANIFEST_PATH=$ManifestPath"
Write-Host "TRAIN_MANIFEST_EXISTS=$ManifestExists"
Write-Host "TRAIN_MANIFEST_VALID=$ManifestValid"
Write-Host "TRAIN_CHECKPOINT_FILES_VALID=$CheckpointFilesValid"
Write-Host "TRAIN_MANIFEST_VALIDATION_ERROR=$ManifestValidationError"
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
