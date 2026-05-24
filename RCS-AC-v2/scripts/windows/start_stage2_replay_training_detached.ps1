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
    [double]$BehaviorWeight = 1.0,
    [switch]$ImitationClassBalance,
    [int]$HiddenDim = 128,
    [string]$Device = "auto",
    [string]$LogPath = "",
    [string]$TaskName = "RCSV2Stage2ReplayTrain",
    [string]$InteractiveUser = "song\song",
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
$ResolvedOutputDir = (Resolve-Path $OutputDir).Path
$ResolvedLogPath = (Resolve-Path (Split-Path -Parent $LogPath)).Path
$ResolvedLogPath = Join-Path $ResolvedLogPath (Split-Path -Leaf $LogPath)

$TrainingScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\train_stage2_privileged_from_replay.ps1"
$QuotedProjectRoot = Quote-PowerShellString $ProjectRoot
$QuotedTrainingScript = Quote-PowerShellString $TrainingScript
$QuotedReplayPath = Quote-PowerShellString $ReplayPath
$QuotedReplayManifestPath = Quote-PowerShellString $ReplayManifestPath
$QuotedOutputDir = Quote-PowerShellString $ResolvedOutputDir
$QuotedDevice = Quote-PowerShellString $Device
$QuotedLogPath = Quote-PowerShellString $ResolvedLogPath
$InnerScriptPath = Join-Path $ResolvedOutputDir "train_stage2_privileged_from_replay.detached.ps1"
$QuotedInnerScriptPath = Quote-PowerShellString $InnerScriptPath
$ImitationClassBalanceArg = ""
if ($ImitationClassBalance) {
    $ImitationClassBalanceArg = " -ImitationClassBalance"
}

$InnerCommand = @"
`$ErrorActionPreference = 'Stop'
`$LogPath = $QuotedLogPath
"DETACHED_TRAIN_STARTED=`$(Get-Date -Format o)" | Set-Content -Path `$LogPath
try {
    Set-Location $QuotedProjectRoot
    powershell -ExecutionPolicy Bypass -File $QuotedTrainingScript -ProjectRoot $QuotedProjectRoot -ReplayPath $QuotedReplayPath -ReplayManifestPath $QuotedReplayManifestPath -OutputDir $QuotedOutputDir -SequenceLength $SequenceLength -BatchSize $BatchSize -WorldModelSteps $WorldModelSteps -PolicySteps $PolicySteps -LearningRate $LearningRate -ImitationWeight $ImitationWeight -BehaviorWeight $BehaviorWeight$ImitationClassBalanceArg -HiddenDim $HiddenDim -Device $QuotedDevice *>&1 | Tee-Object -FilePath `$LogPath -Append
    `$Code = `$LASTEXITCODE
}
catch {
    `$Code = 99
    "TRAIN_EXCEPTION=`$(`$_.Exception.GetType().FullName): `$(`$_.Exception.Message)" | Tee-Object -FilePath `$LogPath -Append
}
"TRAIN_EXIT_CODE=`$Code" | Tee-Object -FilePath `$LogPath -Append
exit `$Code
"@

$InnerCommand | Set-Content -Path $InnerScriptPath
$TaskArgument = "-NoProfile -ExecutionPolicy Bypass -File `"$InnerScriptPath`""

Write-Host "STAGE2_DETACHED_REPLAY_TRAIN_READY=True"
Write-Host "REPLAY_PATH=$ReplayPath"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "OUTPUT_DIR=$ResolvedOutputDir"
Write-Host "TRAIN_LOG_PATH=$ResolvedLogPath"
Write-Host "TRAIN_SCRIPT_PATH=$InnerScriptPath"
Write-Host "TRAIN_TASK_NAME=$TaskName"
if ($NoStart) {
    Write-Host "TRAIN_DETACHED_DRY_RUN=True"
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $TaskArgument -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$Principal = New-ScheduledTaskPrincipal -UserId $InteractiveUser -LogonType Interactive -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "TRAIN_TASK_STARTED=True"
