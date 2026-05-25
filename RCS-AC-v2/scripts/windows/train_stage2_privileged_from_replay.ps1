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
    [switch]$PdfStage1Strict,
    [double]$ImitationWeight = 0.0,
    [double]$BehaviorWeight = 1.0,
    [double]$BehaviorGamma = 0.99,
    [double]$BehaviorLambda = 1.0,
    [double]$BehaviorActorWeight = 0.1,
    [double]$BehaviorCriticWeight = 1.0,
    [double]$BehaviorEntropyWeight = 0.001,
    [switch]$ImitationClassBalance,
    [double]$ImitationBalancePower = 1.0,
    [double]$SteerSequenceFraction = 0.0,
    [double]$MovingSteerFalsePositiveWeight = 0.0,
    [double]$MovingSteerBinaryWeight = 0.0,
    [double]$MovingSteerNonmovingMarginWeight = 0.0,
    [double]$MovingSteerNonmovingMargin = 0.25,
    [double]$WorldModelActionAuxWeight = 0.0,
    [string]$ComponentSampleWeights = "",
    [double]$ActionMassCapWeight = 0.0,
    [string]$ActionMassCaps = "",
    [double]$RouteSpeedActionMassCapWeight = 0.0,
    [string]$RouteSpeedActionIds = "",
    [double]$RouteSpeedActionMassCap = 1.0,
    [double]$RouteSpeedMinSpeedNorm = 0.0,
    [double]$RouteSpeedMinRouteProgress = 0.0,
    [double]$RouteSpeedMaxRouteProgress = 1.0,
    [double]$RouteActionMassCapWeight = 0.0,
    [string]$RouteActionMassCapIds = "",
    [double]$RouteActionMassCap = 1.0,
    [double]$RouteActionMassCapMinSpeedNorm = 0.0,
    [double]$RouteActionMassCapMinRouteProgress = 0.0,
    [double]$RouteActionMassCapMaxRouteProgress = 1.0,
    [double]$RouteThrottleCeilingWeight = 0.0,
    [double]$RouteThrottleCeiling = 1.0,
    [double]$RouteThrottleMinSpeedNorm = 0.0,
    [double]$RouteThrottleMinRouteProgress = 0.0,
    [double]$RouteThrottleMaxRouteProgress = 1.0,
    [double]$RouteLateThrottleCeilingWeight = 0.0,
    [double]$RouteLateThrottleCeiling = 1.0,
    [double]$RouteLateThrottleMinSpeedNorm = 0.0,
    [double]$RouteLateThrottleMinRouteProgress = 0.0,
    [double]$RouteLateThrottleMaxRouteProgress = 1.0,
    [double]$RouteActionOverrideWeight = 0.0,
    [int]$RouteActionOverrideId = -1,
    [double]$RouteActionOverrideMinSpeedNorm = 0.0,
    [double]$RouteActionOverrideMinRouteProgress = 0.0,
    [double]$RouteActionOverrideMaxRouteProgress = 1.0,
    [int]$HiddenDim = 128,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$env:PYTHONPATH"
$CudaAssertScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\assert_stage2_cuda_available.ps1"
powershell -ExecutionPolicy Bypass -File $CudaAssertScript -ProjectRoot $ProjectRoot -Device $Device
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
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
    "--behavior-gamma", $BehaviorGamma,
    "--behavior-lambda", $BehaviorLambda,
    "--behavior-actor-weight", $BehaviorActorWeight,
    "--behavior-critic-weight", $BehaviorCriticWeight,
    "--behavior-entropy-weight", $BehaviorEntropyWeight,
    "--imitation-balance-power", $ImitationBalancePower,
    "--steer-sequence-fraction", $SteerSequenceFraction,
    "--moving-steer-false-positive-weight", $MovingSteerFalsePositiveWeight,
    "--moving-steer-binary-weight", $MovingSteerBinaryWeight,
    "--moving-steer-nonmoving-margin-weight", $MovingSteerNonmovingMarginWeight,
    "--moving-steer-nonmoving-margin", $MovingSteerNonmovingMargin,
    "--world-model-action-aux-weight", $WorldModelActionAuxWeight,
    "--action-mass-cap-weight", $ActionMassCapWeight,
    "--route-speed-action-mass-cap-weight", $RouteSpeedActionMassCapWeight,
    "--route-speed-action-mass-cap", $RouteSpeedActionMassCap,
    "--route-speed-min-speed-norm", $RouteSpeedMinSpeedNorm,
    "--route-speed-min-route-progress", $RouteSpeedMinRouteProgress,
    "--route-speed-max-route-progress", $RouteSpeedMaxRouteProgress,
    "--route-action-mass-cap-weight", $RouteActionMassCapWeight,
    "--route-action-mass-cap", $RouteActionMassCap,
    "--route-action-mass-cap-min-speed-norm", $RouteActionMassCapMinSpeedNorm,
    "--route-action-mass-cap-min-route-progress", $RouteActionMassCapMinRouteProgress,
    "--route-action-mass-cap-max-route-progress", $RouteActionMassCapMaxRouteProgress,
    "--route-throttle-ceiling-weight", $RouteThrottleCeilingWeight,
    "--route-throttle-ceiling", $RouteThrottleCeiling,
    "--route-throttle-min-speed-norm", $RouteThrottleMinSpeedNorm,
    "--route-throttle-min-route-progress", $RouteThrottleMinRouteProgress,
    "--route-throttle-max-route-progress", $RouteThrottleMaxRouteProgress,
    "--route-late-throttle-ceiling-weight", $RouteLateThrottleCeilingWeight,
    "--route-late-throttle-ceiling", $RouteLateThrottleCeiling,
    "--route-late-throttle-min-speed-norm", $RouteLateThrottleMinSpeedNorm,
    "--route-late-throttle-min-route-progress", $RouteLateThrottleMinRouteProgress,
    "--route-late-throttle-max-route-progress", $RouteLateThrottleMaxRouteProgress,
    "--route-action-override-weight", $RouteActionOverrideWeight,
    "--route-action-override-id", $RouteActionOverrideId,
    "--route-action-override-min-speed-norm", $RouteActionOverrideMinSpeedNorm,
    "--route-action-override-min-route-progress", $RouteActionOverrideMinRouteProgress,
    "--route-action-override-max-route-progress", $RouteActionOverrideMaxRouteProgress,
    "--hidden-dim", $HiddenDim,
    "--device", $Device
)
if ($ComponentSampleWeights -ne "") {
    $TrainArgs += @("--component-sample-weight", $ComponentSampleWeights)
}
if ($ActionMassCaps -ne "") {
    $TrainArgs += @("--action-mass-cap", $ActionMassCaps)
}
if ($RouteSpeedActionIds -ne "") {
    $TrainArgs += @("--route-speed-action-ids", $RouteSpeedActionIds)
}
if ($RouteActionMassCapIds -ne "") {
    $TrainArgs += @("--route-action-mass-cap-ids", $RouteActionMassCapIds)
}
if ($ImitationClassBalance) {
    $TrainArgs += "--imitation-class-balance"
}
if ($PdfStage1Strict) {
    $TrainArgs += "--pdf-stage1-strict"
}
& py @TrainArgs
exit $LASTEXITCODE
