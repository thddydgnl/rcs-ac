param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ReplayPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged",
    [string]$ReplayManifestPath = "",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay",
    [int]$SequenceLength = 16,
    [int]$BatchSize = 8,
    [int]$WorldModelSteps = 200,
    [int]$PolicySteps = 200,
    [double]$LearningRate = 0.0003,
    [double]$ImitationWeight = 1.0,
    [int]$HiddenDim = 128,
    [string]$Device = "auto",
    [string]$LogPath = "",
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"

function Quote-PowerShellString {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

Set-Location $ProjectRoot
if ($ReplayManifestPath -eq "") {
    $ReplayManifestPath = Join-Path $ReplayPath "stage2_privileged_replay_manifest.json"
}
if ($LogPath -eq "") {
    $LogPath = Join-Path $OutputDir "train_stage2_privileged_from_replay.log"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

$TrainingScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\train_stage2_privileged_from_replay.ps1"
$QuotedProjectRoot = Quote-PowerShellString $ProjectRoot
$QuotedTrainingScript = Quote-PowerShellString $TrainingScript
$QuotedReplayPath = Quote-PowerShellString $ReplayPath
$QuotedReplayManifestPath = Quote-PowerShellString $ReplayManifestPath
$QuotedOutputDir = Quote-PowerShellString $OutputDir
$QuotedDevice = Quote-PowerShellString $Device
$QuotedLogPath = Quote-PowerShellString $LogPath

$InnerCommand = @"
`$ErrorActionPreference = 'Stop'
Set-Location $QuotedProjectRoot
powershell -ExecutionPolicy Bypass -File $QuotedTrainingScript -ProjectRoot $QuotedProjectRoot -ReplayPath $QuotedReplayPath -ReplayManifestPath $QuotedReplayManifestPath -OutputDir $QuotedOutputDir -SequenceLength $SequenceLength -BatchSize $BatchSize -WorldModelSteps $WorldModelSteps -PolicySteps $PolicySteps -LearningRate $LearningRate -ImitationWeight $ImitationWeight -HiddenDim $HiddenDim -Device $QuotedDevice *>&1 | Tee-Object -FilePath $QuotedLogPath
`$Code = `$LASTEXITCODE
"TRAIN_EXIT_CODE=`$Code" | Tee-Object -FilePath $QuotedLogPath -Append
exit `$Code
"@

$EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($InnerCommand))
$Arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $EncodedCommand)

Write-Host "STAGE2_DETACHED_REPLAY_TRAIN_READY=True"
Write-Host "REPLAY_PATH=$ReplayPath"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "OUTPUT_DIR=$OutputDir"
Write-Host "TRAIN_LOG_PATH=$LogPath"
if ($NoStart) {
    Write-Host "TRAIN_DETACHED_DRY_RUN=True"
    exit 0
}

$Process = Start-Process -FilePath "powershell" -ArgumentList $Arguments -WindowStyle Hidden -PassThru
Write-Host "TRAIN_PROCESS_ID=$($Process.Id)"
