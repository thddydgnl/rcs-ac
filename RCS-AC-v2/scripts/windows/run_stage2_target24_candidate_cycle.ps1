param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_steer2_seq010_aux1_balpow05_iw5\clean_512_stage2_privileged_7f87656_steer2_seq010_aux1_balpow05_iw5_with_checkpoints.yaml",
    [string]$ReplayPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_t24_failure_window",
    [string]$ReplayManifestPath = "",
    [string]$SecondaryReplayManifestPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_6e74d9e_long512_t25_steer2_probe\stage2_privileged_replay_manifest.json",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_t24_aux1_balpow05_iw5",
    [string]$CrossProbeOutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_t24_aux1_balpow05_iw5_cross_probe",
    [string]$TaskName = "RCSV2Stage2T24Aux1BalPow05",
    [string]$CandidateName = "target24",
    [string]$PrimaryReplayName = "",
    [switch]$SkipGitPull,
    [switch]$StartIfMissing,
    [switch]$ProbeWhenReady,
    [int]$Seed = 20260524,
    [int]$SequenceLength = 16,
    [int]$BatchSize = 8,
    [int]$WorldModelSteps = 1200,
    [int]$PolicySteps = 1200,
    [double]$LearningRate = 0.0003,
    [double]$ImitationWeight = 5.0,
    [double]$BehaviorWeight = 0.0,
    [double]$ImitationBalancePower = 0.5,
    [double]$SteerSequenceFraction = 0.10,
    [double]$MovingSteerFalsePositiveWeight = 0.0,
    [double]$WorldModelActionAuxWeight = 1.0,
    [int]$HiddenDim = 128,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
if ($ReplayManifestPath -eq "") {
    $ReplayManifestPath = Join-Path $ReplayPath "stage2_privileged_replay_manifest.json"
}
if ($PrimaryReplayName -eq "") {
    $PrimaryReplayName = $CandidateName
}
if (-not $SkipGitPull) {
    git pull --ff-only
}

$RequestedOutputDir = $OutputDir
$LatestPointerPath = "${RequestedOutputDir}.latest.json"
if (Test-Path $LatestPointerPath) {
    try {
        $LatestPointer = Get-Content $LatestPointerPath -Raw | ConvertFrom-Json
        if ($null -ne $LatestPointer.active_output_dir -and (Test-Path ([string]$LatestPointer.active_output_dir))) {
            $OutputDir = [string]$LatestPointer.active_output_dir
            Write-Host "CANDIDATE_LATEST_POINTER_PATH=$LatestPointerPath"
            Write-Host "CANDIDATE_USING_LATEST_OUTPUT_DIR=$OutputDir"
        }
    }
    catch {
        Write-Host "CANDIDATE_LATEST_POINTER_PARSE_ERROR=$($_.Exception.Message)"
    }
}

$CheckScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\check_stage2_replay_training_detached.ps1"
$StartScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\start_stage2_replay_training_detached.ps1"
$CrossProbeScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\probe_stage2_candidate_cross_replays.ps1"
$CheckpointManifestPath = Join-Path $OutputDir "stage2_privileged_replay_checkpoint_manifest.json"

Write-Host "STAGE2_REPLAY_CANDIDATE_CYCLE_READY=True"
Write-Host "CANDIDATE_NAME=$CandidateName"
Write-Host "PROJECT_ROOT=$ProjectRoot"
Write-Host "CONFIG_PATH=$ConfigPath"
Write-Host "REPLAY_PATH=$ReplayPath"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "REQUESTED_OUTPUT_DIR=$RequestedOutputDir"
Write-Host "OUTPUT_DIR=$OutputDir"
Write-Host "CANDIDATE_LATEST_POINTER_PATH=$LatestPointerPath"
Write-Host "CHECKPOINT_MANIFEST_PATH=$CheckpointManifestPath"
Write-Host "TASK_NAME=$TaskName"

& powershell -ExecutionPolicy Bypass -File $CheckScript `
    -ProjectRoot $ProjectRoot `
    -OutputDir $OutputDir `
    -TaskName $TaskName
$CheckExitCode = $LASTEXITCODE
Write-Host "CANDIDATE_CHECK_EXIT_CODE=$CheckExitCode"

if ($CheckExitCode -eq 0) {
    Write-Host "CANDIDATE_TRAIN_READY_FOR_PROBE=True"
    if ($ProbeWhenReady) {
        powershell -ExecutionPolicy Bypass -File $CrossProbeScript `
            -ProjectRoot $ProjectRoot `
            -ConfigPath $ConfigPath `
            -CheckpointManifestPath $CheckpointManifestPath `
            -OutputDir $CrossProbeOutputDir `
            -PrimaryReplayName $PrimaryReplayName `
            -PrimaryReplayManifestPath $ReplayManifestPath `
            -SecondaryReplayName "steer2" `
            -SecondaryReplayManifestPath $SecondaryReplayManifestPath `
            -Device $Device
        exit $LASTEXITCODE
    }
    exit 0
}

if ($CheckExitCode -eq 3) {
    Write-Host "CANDIDATE_TRAIN_RUNNING=True"
    exit 3
}

if (-not $StartIfMissing) {
    Write-Host "CANDIDATE_TRAIN_READY_FOR_PROBE=False"
    Write-Host "CANDIDATE_NEXT_ACTION=rerun with -StartIfMissing to launch detached training"
    exit 2
}

$RerunOutputDir = $OutputDir
if (Test-Path $OutputDir) {
    $RerunOutputDir = "${RequestedOutputDir}_rerun_$(Get-Date -Format yyyyMMddTHHmmss)"
    Write-Host "CANDIDATE_EXISTING_OUTPUT_DIR_NOT_READY=$OutputDir"
    Write-Host "CANDIDATE_RERUN_OUTPUT_DIR=$RerunOutputDir"
}
$PointerParent = Split-Path -Parent $LatestPointerPath
if ($PointerParent -ne "") {
    New-Item -ItemType Directory -Force -Path $PointerParent | Out-Null
}
$LatestPointer = [ordered]@{
    candidate_name = $CandidateName
    primary_replay_name = $PrimaryReplayName
    requested_output_dir = $RequestedOutputDir
    active_output_dir = $RerunOutputDir
    replay_manifest_path = $ReplayManifestPath
    task_name = $TaskName
    updated_at = (Get-Date -Format o)
}
$LatestPointer | ConvertTo-Json -Depth 4 | Set-Content -Path $LatestPointerPath
Write-Host "CANDIDATE_LATEST_POINTER_UPDATED=True"
Write-Host "CANDIDATE_LATEST_POINTER_ACTIVE_OUTPUT_DIR=$RerunOutputDir"

powershell -ExecutionPolicy Bypass -File $StartScript `
    -ProjectRoot $ProjectRoot `
    -ReplayPath $ReplayPath `
    -ReplayManifestPath $ReplayManifestPath `
    -OutputDir $RerunOutputDir `
    -Seed $Seed `
    -SequenceLength $SequenceLength `
    -BatchSize $BatchSize `
    -WorldModelSteps $WorldModelSteps `
    -PolicySteps $PolicySteps `
    -LearningRate $LearningRate `
    -ImitationWeight $ImitationWeight `
    -BehaviorWeight $BehaviorWeight `
    -ImitationClassBalance `
    -ImitationBalancePower $ImitationBalancePower `
    -SteerSequenceFraction $SteerSequenceFraction `
    -MovingSteerFalsePositiveWeight $MovingSteerFalsePositiveWeight `
    -WorldModelActionAuxWeight $WorldModelActionAuxWeight `
    -HiddenDim $HiddenDim `
    -Device $Device `
    -TaskName $TaskName
exit $LASTEXITCODE
