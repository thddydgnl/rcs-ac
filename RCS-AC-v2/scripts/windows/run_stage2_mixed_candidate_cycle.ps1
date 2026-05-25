param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_steer2_seq010_aux1_balpow05_iw5\clean_512_stage2_privileged_7f87656_steer2_seq010_aux1_balpow05_iw5_with_checkpoints.yaml",
    [string]$PrimaryReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_target22_failure_window",
    [string]$PrimaryReplayName = "target22",
    [string]$SecondaryReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_6e74d9e_long512_t25_steer2_probe",
    [string]$SecondaryReplayName = "steer2",
    [string]$MixedReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_mixed_target22_steer2",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015",
    [string]$CrossProbeOutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015_cross_probe",
    [string]$TaskName = "RCSV2Stage2MixedT22Steer2Ssf015",
    [switch]$SkipGitPull,
    [switch]$MakeReplayIfMissing,
    [switch]$ForceMixedReplay,
    [switch]$StartIfMissing,
    [switch]$ProbeWhenReady,
    [int]$Seed = 20260524,
    [int]$SequenceLength = 16,
    [int]$BatchSize = 8,
    [int]$WorldModelSteps = 1500,
    [int]$PolicySteps = 1500,
    [double]$LearningRate = 0.0003,
    [double]$ImitationWeight = 5.0,
    [double]$BehaviorWeight = 0.0,
    [double]$ImitationBalancePower = 0.5,
    [double]$SteerSequenceFraction = 0.15,
    [double]$MovingSteerFalsePositiveWeight = 0.0,
    [double]$MovingSteerBinaryWeight = 0.0,
    [double]$WorldModelActionAuxWeight = 1.0,
    [int]$HiddenDim = 128,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
if (-not $SkipGitPull) {
    git pull --ff-only
}
$CudaAssertScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\assert_stage2_cuda_available.ps1"
powershell -ExecutionPolicy Bypass -File $CudaAssertScript -ProjectRoot $ProjectRoot -Device $Device
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$MixedManifestPath = Join-Path $MixedReplayDir "stage2_privileged_replay_manifest.json"
$PrimaryReplayManifestPath = Join-Path $PrimaryReplayDir "stage2_privileged_replay_manifest.json"
$SecondaryReplayManifestPath = Join-Path $SecondaryReplayDir "stage2_privileged_replay_manifest.json"
$MakeScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\make_stage2_mixed_replay_manifest.ps1"
$CheckScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\check_stage2_replay_training_detached.ps1"
$StartScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\start_stage2_replay_training_detached.ps1"
$CrossProbeScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\probe_stage2_candidate_cross_replays.ps1"

Write-Host "STAGE2_MIXED_CANDIDATE_CYCLE_READY=True"
Write-Host "MIXED_REPLAY_DIR=$MixedReplayDir"
Write-Host "MIXED_REPLAY_MANIFEST_PATH=$MixedManifestPath"
Write-Host "PRIMARY_REPLAY_MANIFEST_PATH=$PrimaryReplayManifestPath"
Write-Host "SECONDARY_REPLAY_MANIFEST_PATH=$SecondaryReplayManifestPath"
Write-Host "OUTPUT_DIR=$OutputDir"
Write-Host "CROSS_PROBE_OUTPUT_DIR=$CrossProbeOutputDir"
Write-Host "TASK_NAME=$TaskName"

if ((-not (Test-Path $MixedManifestPath)) -or $ForceMixedReplay) {
    if (-not $MakeReplayIfMissing -and -not $ForceMixedReplay) {
        Write-Host "MIXED_REPLAY_MANIFEST_EXISTS=False"
        Write-Host "MIXED_REPLAY_NEXT_ACTION=rerun with -MakeReplayIfMissing"
        exit 2
    }
    $MakeArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $MakeScript,
        "-ProjectRoot", $ProjectRoot,
        "-PrimaryReplayDir", $PrimaryReplayDir,
        "-PrimaryName", $PrimaryReplayName,
        "-SecondaryReplayDir", $SecondaryReplayDir,
        "-SecondaryName", $SecondaryReplayName,
        "-OutputDir", $MixedReplayDir,
        "-ConfigPath", $ConfigPath
    )
    if ($ForceMixedReplay) {
        $MakeArgs += "-Force"
    }
    powershell @MakeArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

foreach ($RequiredManifest in @($MixedManifestPath, $PrimaryReplayManifestPath, $SecondaryReplayManifestPath)) {
    if (-not (Test-Path $RequiredManifest)) {
        Write-Host "REQUIRED_REPLAY_MANIFEST_MISSING=$RequiredManifest"
        exit 2
    }
}
Write-Host "MIXED_REPLAY_MANIFEST_EXISTS=True"

$RequestedOutputDir = $OutputDir
$LatestPointerPath = "${RequestedOutputDir}.latest.json"
if (Test-Path $LatestPointerPath) {
    try {
        $LatestPointer = Get-Content $LatestPointerPath -Raw | ConvertFrom-Json
        if ($null -ne $LatestPointer.active_output_dir -and (Test-Path ([string]$LatestPointer.active_output_dir))) {
            $OutputDir = [string]$LatestPointer.active_output_dir
            Write-Host "MIXED_CANDIDATE_LATEST_POINTER_PATH=$LatestPointerPath"
            Write-Host "MIXED_CANDIDATE_USING_LATEST_OUTPUT_DIR=$OutputDir"
        }
    }
    catch {
        Write-Host "MIXED_CANDIDATE_LATEST_POINTER_PARSE_ERROR=$($_.Exception.Message)"
    }
}

$CheckpointManifestPath = Join-Path $OutputDir "stage2_privileged_replay_checkpoint_manifest.json"
Write-Host "CHECKPOINT_MANIFEST_PATH=$CheckpointManifestPath"

& powershell -ExecutionPolicy Bypass -File $CheckScript `
    -ProjectRoot $ProjectRoot `
    -OutputDir $OutputDir `
    -TaskName $TaskName
$CheckExitCode = $LASTEXITCODE
Write-Host "MIXED_CANDIDATE_CHECK_EXIT_CODE=$CheckExitCode"

if ($CheckExitCode -eq 0) {
    Write-Host "MIXED_CANDIDATE_TRAIN_READY_FOR_PROBE=True"
    if ($ProbeWhenReady) {
        powershell -ExecutionPolicy Bypass -File $CrossProbeScript `
            -ProjectRoot $ProjectRoot `
            -ConfigPath $ConfigPath `
            -CheckpointManifestPath $CheckpointManifestPath `
            -OutputDir $CrossProbeOutputDir `
            -PrimaryReplayName $PrimaryReplayName `
            -PrimaryReplayManifestPath $PrimaryReplayManifestPath `
            -SecondaryReplayName $SecondaryReplayName `
            -SecondaryReplayManifestPath $SecondaryReplayManifestPath `
            -Device $Device
        exit $LASTEXITCODE
    }
    exit 0
}

if ($CheckExitCode -eq 3) {
    Write-Host "MIXED_CANDIDATE_TRAIN_RUNNING=True"
    exit 3
}

if (-not $StartIfMissing) {
    Write-Host "MIXED_CANDIDATE_TRAIN_READY_FOR_PROBE=False"
    Write-Host "MIXED_CANDIDATE_NEXT_ACTION=rerun with -StartIfMissing to launch detached training"
    exit 2
}

$RerunOutputDir = $OutputDir
if (Test-Path $OutputDir) {
    $RerunOutputDir = "${RequestedOutputDir}_rerun_$(Get-Date -Format yyyyMMddTHHmmss)"
    Write-Host "MIXED_CANDIDATE_EXISTING_OUTPUT_DIR_NOT_READY=$OutputDir"
    Write-Host "MIXED_CANDIDATE_RERUN_OUTPUT_DIR=$RerunOutputDir"
}
$PointerParent = Split-Path -Parent $LatestPointerPath
if ($PointerParent -ne "") {
    New-Item -ItemType Directory -Force -Path $PointerParent | Out-Null
}
$LatestPointer = [ordered]@{
    candidate_name = "mixed_${PrimaryReplayName}_${SecondaryReplayName}"
    primary_replay_name = $PrimaryReplayName
    secondary_replay_name = $SecondaryReplayName
    requested_output_dir = $RequestedOutputDir
    active_output_dir = $RerunOutputDir
    mixed_replay_manifest_path = $MixedManifestPath
    primary_replay_manifest_path = $PrimaryReplayManifestPath
    secondary_replay_manifest_path = $SecondaryReplayManifestPath
    task_name = $TaskName
    seed = $Seed
    steer_sequence_fraction = $SteerSequenceFraction
    moving_steer_false_positive_weight = $MovingSteerFalsePositiveWeight
    moving_steer_binary_weight = $MovingSteerBinaryWeight
    updated_at = (Get-Date -Format o)
}
$LatestPointer | ConvertTo-Json -Depth 4 | Set-Content -Path $LatestPointerPath
Write-Host "MIXED_CANDIDATE_LATEST_POINTER_UPDATED=True"
Write-Host "MIXED_CANDIDATE_LATEST_POINTER_ACTIVE_OUTPUT_DIR=$RerunOutputDir"

powershell -ExecutionPolicy Bypass -File $StartScript `
    -ProjectRoot $ProjectRoot `
    -ReplayPath $MixedReplayDir `
    -ReplayManifestPath $MixedManifestPath `
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
    -MovingSteerBinaryWeight $MovingSteerBinaryWeight `
    -WorldModelActionAuxWeight $WorldModelActionAuxWeight `
    -HiddenDim $HiddenDim `
    -Device $Device `
    -TaskName $TaskName
exit $LASTEXITCODE
