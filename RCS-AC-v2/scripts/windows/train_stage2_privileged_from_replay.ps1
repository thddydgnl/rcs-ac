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
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$env:PYTHONPATH"
if ($ReplayManifestPath -eq "") {
    $ReplayManifestPath = Join-Path $ReplayPath "stage2_privileged_replay_manifest.json"
}

Write-Host "STAGE2_PRIVILEGED_REPLAY_TRAIN_READY=True"
Write-Host "REPLAY_PATH=$ReplayPath"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "OUTPUT_DIR=$OutputDir"
$TrainArgs = @(
    "-3.12", "-B", "-m", "rcs_ac_v2.training.train_stage2_privileged_from_replay",
    "--replay", $ReplayPath,
    "--replay-manifest", $ReplayManifestPath,
    "--output-dir", $OutputDir,
    "--seed", $Seed,
    "--sequence-length", $SequenceLength,
    "--batch-size", $BatchSize,
    "--world-model-steps", $WorldModelSteps,
    "--policy-steps", $PolicySteps,
    "--learning-rate", $LearningRate,
    "--imitation-weight", $ImitationWeight,
    "--behavior-weight", $BehaviorWeight,
    "--imitation-balance-power", $ImitationBalancePower,
    "--steer-sequence-fraction", $SteerSequenceFraction,
    "--moving-steer-false-positive-weight", $MovingSteerFalsePositiveWeight,
    "--moving-steer-binary-weight", $MovingSteerBinaryWeight,
    "--world-model-action-aux-weight", $WorldModelActionAuxWeight,
    "--hidden-dim", $HiddenDim,
    "--device", $Device
)
if ($ImitationClassBalance) {
    $TrainArgs += "--imitation-class-balance"
}
& py @TrainArgs
exit $LASTEXITCODE
