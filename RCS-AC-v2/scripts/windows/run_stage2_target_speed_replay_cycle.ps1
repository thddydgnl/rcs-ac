param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_steer2_seq010_aux1_balpow05_iw5\clean_512_stage2_privileged_7f87656_steer2_seq010_aux1_balpow05_iw5_with_checkpoints.yaml",
    [string]$CandidateName = "target22",
    [double]$TargetSpeedKmh = 22.0,
    [double]$SteerSign = -1.0,
    [int]$Episodes = 2,
    [int]$MaxSteps = 512,
    [string]$ReplayOutputDir = "",
    [string]$CheckpointOutputDir = "",
    [string]$CrossProbeOutputDir = "",
    [string]$SecondaryReplayManifestPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_6e74d9e_long512_t25_steer2_probe\stage2_privileged_replay_manifest.json",
    [string]$TaskName = "",
    [switch]$SkipGitPull,
    [switch]$SkipCarlaReady,
    [switch]$CollectIfMissing,
    [switch]$StartIfMissing,
    [switch]$ProbeWhenReady
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
if (-not $SkipGitPull) {
    git pull --ff-only
}

if ($ReplayOutputDir -eq "") {
    $ReplayOutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_${CandidateName}_failure_window"
}
if ($CheckpointOutputDir -eq "") {
    $CheckpointOutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_${CandidateName}_aux1_balpow05_iw5"
}
if ($CrossProbeOutputDir -eq "") {
    $CrossProbeOutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_${CandidateName}_aux1_balpow05_iw5_cross_probe"
}
if ($TaskName -eq "") {
    $TaskName = "RCSV2Stage2$($CandidateName)Aux1BalPow05"
}

$ReplayManifestPath = Join-Path $ReplayOutputDir "stage2_privileged_replay_manifest.json"
$CollectScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\collect_stage2_privileged_replay.ps1"
$CycleScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\run_stage2_target24_candidate_cycle.ps1"

Write-Host "STAGE2_TARGET_SPEED_REPLAY_CYCLE_READY=True"
Write-Host "CANDIDATE_NAME=$CandidateName"
Write-Host "TARGET_SPEED_KMH=$TargetSpeedKmh"
Write-Host "REPLAY_OUTPUT_DIR=$ReplayOutputDir"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "CHECKPOINT_OUTPUT_DIR=$CheckpointOutputDir"
Write-Host "CROSS_PROBE_OUTPUT_DIR=$CrossProbeOutputDir"
Write-Host "TASK_NAME=$TaskName"

if (-not (Test-Path $ReplayManifestPath)) {
    if (-not $CollectIfMissing) {
        Write-Host "TARGET_SPEED_REPLAY_MANIFEST_EXISTS=False"
        Write-Host "TARGET_SPEED_NEXT_ACTION=rerun with -CollectIfMissing"
        exit 2
    }
    $CollectArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $CollectScript,
        "-ProjectRoot", $ProjectRoot,
        "-ConfigPath", $ConfigPath,
        "-OutputDir", $ReplayOutputDir,
        "-Episodes", $Episodes,
        "-MaxSteps", $MaxSteps,
        "-TargetSpeedKmh", $TargetSpeedKmh,
        "-SteerSign", $SteerSign,
        "-SkipGitPull"
    )
    if ($SkipCarlaReady) {
        $CollectArgs += "-SkipCarlaReady"
    }
    powershell @CollectArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path $ReplayManifestPath)) {
    Write-Host "TARGET_SPEED_REPLAY_MANIFEST_EXISTS=False_AFTER_COLLECT"
    exit 2
}
Write-Host "TARGET_SPEED_REPLAY_MANIFEST_EXISTS=True"

$CycleArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $CycleScript,
    "-ProjectRoot", $ProjectRoot,
    "-ConfigPath", $ConfigPath,
    "-ReplayPath", $ReplayOutputDir,
    "-ReplayManifestPath", $ReplayManifestPath,
    "-SecondaryReplayManifestPath", $SecondaryReplayManifestPath,
    "-OutputDir", $CheckpointOutputDir,
    "-CrossProbeOutputDir", $CrossProbeOutputDir,
    "-TaskName", $TaskName,
    "-CandidateName", $CandidateName,
    "-PrimaryReplayName", $CandidateName,
    "-SkipGitPull"
)
if ($StartIfMissing) {
    $CycleArgs += "-StartIfMissing"
}
if ($ProbeWhenReady) {
    $CycleArgs += "-ProbeWhenReady"
}
powershell @CycleArgs
exit $LASTEXITCODE
