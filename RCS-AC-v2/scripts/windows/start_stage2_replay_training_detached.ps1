param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ReplayPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged",
    [string]$ReplayManifestPath = "",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay",
    [int]$Seed = 20260524,
    [int]$SequenceLength = 16,
    [int]$BatchSize = 8,
    [int]$WorldModelSteps = 200,
    [int]$PolicySteps = 200,
    [double]$LearningRate = 0.0003,
    [double]$ImitationWeight = 1.0,
    [double]$BehaviorWeight = 1.0,
    [switch]$ImitationClassBalance,
    [double]$ImitationBalancePower = 1.0,
    [double]$SteerSequenceFraction = 0.0,
    [double]$MovingSteerFalsePositiveWeight = 0.0,
    [double]$MovingSteerBinaryWeight = 0.0,
    [double]$WorldModelActionAuxWeight = 0.0,
    [int]$HiddenDim = 128,
    [string]$Device = "cuda",
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
$CudaAssertScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\assert_stage2_cuda_available.ps1"
powershell -ExecutionPolicy Bypass -File $CudaAssertScript -ProjectRoot $ProjectRoot -Device $Device
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
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
$LaunchManifestPath = Join-Path $ResolvedOutputDir "train_stage2_privileged_from_replay.launch.json"

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
    powershell -ExecutionPolicy Bypass -File $QuotedTrainingScript -ProjectRoot $QuotedProjectRoot -ReplayPath $QuotedReplayPath -ReplayManifestPath $QuotedReplayManifestPath -OutputDir $QuotedOutputDir -Seed $Seed -SequenceLength $SequenceLength -BatchSize $BatchSize -WorldModelSteps $WorldModelSteps -PolicySteps $PolicySteps -LearningRate $LearningRate -ImitationWeight $ImitationWeight -BehaviorWeight $BehaviorWeight -ImitationBalancePower $ImitationBalancePower -SteerSequenceFraction $SteerSequenceFraction -MovingSteerFalsePositiveWeight $MovingSteerFalsePositiveWeight -MovingSteerBinaryWeight $MovingSteerBinaryWeight -WorldModelActionAuxWeight $WorldModelActionAuxWeight$ImitationClassBalanceArg -HiddenDim $HiddenDim -Device $QuotedDevice *>&1 | Tee-Object -FilePath `$LogPath -Append
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
$GitCommit = (git rev-parse HEAD).Trim()
$GitBranch = (git rev-parse --abbrev-ref HEAD).Trim()
$PythonVersion = (& py -3.12 --version 2>&1 | Out-String).Trim()
$HostName = [System.Net.Dns]::GetHostName()

$LaunchManifest = [ordered]@{
    project_root = $ProjectRoot
    git_commit = $GitCommit
    git_branch = $GitBranch
    python_version = $PythonVersion
    host = $HostName
    started_by = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    replay_path = $ReplayPath
    replay_manifest_path = $ReplayManifestPath
    output_dir = $ResolvedOutputDir
    log_path = $ResolvedLogPath
    inner_script_path = $InnerScriptPath
    task_name = $TaskName
    seed = $Seed
    sequence_length = $SequenceLength
    batch_size = $BatchSize
    world_model_steps = $WorldModelSteps
    policy_steps = $PolicySteps
    learning_rate = $LearningRate
    imitation_weight = $ImitationWeight
    behavior_weight = $BehaviorWeight
    imitation_class_balance = [bool]$ImitationClassBalance
    imitation_balance_power = $ImitationBalancePower
    steer_sequence_fraction = $SteerSequenceFraction
    moving_steer_false_positive_weight = $MovingSteerFalsePositiveWeight
    moving_steer_binary_weight = $MovingSteerBinaryWeight
    world_model_action_aux_weight = $WorldModelActionAuxWeight
    hidden_dim = $HiddenDim
    device = $Device
    created_at = (Get-Date -Format o)
}
$LaunchManifest | ConvertTo-Json -Depth 4 | Set-Content -Path $LaunchManifestPath

Write-Host "STAGE2_DETACHED_REPLAY_TRAIN_READY=True"
Write-Host "REPLAY_PATH=$ReplayPath"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "OUTPUT_DIR=$ResolvedOutputDir"
Write-Host "TRAIN_LOG_PATH=$ResolvedLogPath"
Write-Host "TRAIN_LAUNCH_MANIFEST_PATH=$LaunchManifestPath"
Write-Host "TRAIN_SCRIPT_PATH=$InnerScriptPath"
Write-Host "TRAIN_TASK_NAME=$TaskName"
if ($NoStart) {
    Write-Host "TRAIN_DETACHED_DRY_RUN=True"
    exit 0
}

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $ExistingTask -and [string]$ExistingTask.State -eq "Running") {
    Write-Host "TRAIN_TASK_ALREADY_RUNNING=True"
    Write-Host "TRAIN_TASK_NAME=$TaskName"
    Write-Host "TRAIN_LAUNCH_MANIFEST_PATH=$LaunchManifestPath"
    exit 3
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $TaskArgument -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$Principal = New-ScheduledTaskPrincipal -UserId $InteractiveUser -LogonType Interactive -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "TRAIN_TASK_STARTED=True"
