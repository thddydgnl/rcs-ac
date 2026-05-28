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
    [double]$RoutePrioritySampleFraction = 0.0,
    [string]$RoutePriorityActionIds = "",
    [string]$RoutePriorityComponents = "",
    [double]$RoutePriorityMinRouteProgress = 0.0,
    [double]$RoutePriorityMaxRouteProgress = 1.0,
    [double]$MovingSteerFalsePositiveWeight = 0.0,
    [double]$MovingSteerBinaryWeight = 0.0,
    [double]$MovingSteerNonmovingMarginWeight = 0.0,
    [double]$MovingSteerNonmovingMargin = 0.25,
    [double]$RouteLateMovingSteerMarginWeight = 0.0,
    [double]$RouteLateMovingSteerMargin = 0.25,
    [double]$RouteLateMovingSteerMinSpeedNorm = 0.0,
    [double]$RouteLateMovingSteerMinRouteProgress = 0.0,
    [double]$RouteLateMovingSteerMaxRouteProgress = 1.0,
    [double]$RouteLateRightSteerMarginWeight = 0.0,
    [double]$RouteLateRightSteerMargin = 0.25,
    [double]$RouteLateRightSteerMinSpeedNorm = 0.0,
    [double]$RouteLateRightSteerMinRouteProgress = 0.0,
    [double]$RouteLateRightSteerMaxRouteProgress = 1.0,
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
    [double]$RouteActionLogitCapWeight = 0.0,
    [string]$RouteActionLogitCapIds = "",
    [double]$RouteActionLogitCapMargin = 0.0,
    [double]$RouteActionLogitCapMinSpeedNorm = 0.0,
    [double]$RouteActionLogitCapMinRouteProgress = 0.0,
    [double]$RouteActionLogitCapMaxRouteProgress = 1.0,
    [double]$RouteActionDistributionWeight = 0.0,
    [double]$RouteActionDistributionMinSpeedNorm = 0.0,
    [double]$RouteActionDistributionMinRouteProgress = 0.0,
    [double]$RouteActionDistributionMaxRouteProgress = 1.0,
    [double]$RouteLateActionLogitCapWeight = 0.0,
    [string]$RouteLateActionLogitCapIds = "",
    [double]$RouteLateActionLogitCapMargin = 0.0,
    [double]$RouteLateActionLogitCapMinSpeedNorm = 0.0,
    [double]$RouteLateActionLogitCapMinRouteProgress = 0.0,
    [double]$RouteLateActionLogitCapMaxRouteProgress = 1.0,
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
    "--route-priority-sample-fraction", $RoutePrioritySampleFraction,
    "--route-priority-min-route-progress", $RoutePriorityMinRouteProgress,
    "--route-priority-max-route-progress", $RoutePriorityMaxRouteProgress,
    "--moving-steer-false-positive-weight", $MovingSteerFalsePositiveWeight,
    "--moving-steer-binary-weight", $MovingSteerBinaryWeight,
    "--moving-steer-nonmoving-margin-weight", $MovingSteerNonmovingMarginWeight,
    "--moving-steer-nonmoving-margin", $MovingSteerNonmovingMargin,
    "--route-late-moving-steer-margin-weight", $RouteLateMovingSteerMarginWeight,
    "--route-late-moving-steer-margin", $RouteLateMovingSteerMargin,
    "--route-late-moving-steer-min-speed-norm", $RouteLateMovingSteerMinSpeedNorm,
    "--route-late-moving-steer-min-route-progress", $RouteLateMovingSteerMinRouteProgress,
    "--route-late-moving-steer-max-route-progress", $RouteLateMovingSteerMaxRouteProgress,
    "--route-late-right-steer-margin-weight", $RouteLateRightSteerMarginWeight,
    "--route-late-right-steer-margin", $RouteLateRightSteerMargin,
    "--route-late-right-steer-min-speed-norm", $RouteLateRightSteerMinSpeedNorm,
    "--route-late-right-steer-min-route-progress", $RouteLateRightSteerMinRouteProgress,
    "--route-late-right-steer-max-route-progress", $RouteLateRightSteerMaxRouteProgress,
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
    "--route-action-logit-cap-weight", $RouteActionLogitCapWeight,
    "--route-action-logit-cap-margin", $RouteActionLogitCapMargin,
    "--route-action-logit-cap-min-speed-norm", $RouteActionLogitCapMinSpeedNorm,
    "--route-action-logit-cap-min-route-progress", $RouteActionLogitCapMinRouteProgress,
    "--route-action-logit-cap-max-route-progress", $RouteActionLogitCapMaxRouteProgress,
    "--route-action-distribution-weight", $RouteActionDistributionWeight,
    "--route-action-distribution-min-speed-norm", $RouteActionDistributionMinSpeedNorm,
    "--route-action-distribution-min-route-progress", $RouteActionDistributionMinRouteProgress,
    "--route-action-distribution-max-route-progress", $RouteActionDistributionMaxRouteProgress,
    "--route-late-action-logit-cap-weight", $RouteLateActionLogitCapWeight,
    "--route-late-action-logit-cap-margin", $RouteLateActionLogitCapMargin,
    "--route-late-action-logit-cap-min-speed-norm", $RouteLateActionLogitCapMinSpeedNorm,
    "--route-late-action-logit-cap-min-route-progress", $RouteLateActionLogitCapMinRouteProgress,
    "--route-late-action-logit-cap-max-route-progress", $RouteLateActionLogitCapMaxRouteProgress,
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
if ($RoutePriorityActionIds -ne "") {
    $TrainArgs += @("--route-priority-action-ids", $RoutePriorityActionIds)
}
if ($RoutePriorityComponents -ne "") {
    $TrainArgs += @("--route-priority-components", $RoutePriorityComponents)
}
if ($RouteActionMassCapIds -ne "") {
    $TrainArgs += @("--route-action-mass-cap-ids", $RouteActionMassCapIds)
}
if ($RouteActionLogitCapIds -ne "") {
    $TrainArgs += @("--route-action-logit-cap-ids", $RouteActionLogitCapIds)
}
if ($RouteLateActionLogitCapIds -ne "") {
    $TrainArgs += @("--route-late-action-logit-cap-ids", $RouteLateActionLogitCapIds)
}
if ($ImitationClassBalance) {
    $TrainArgs += "--imitation-class-balance"
}
if ($PdfStage1Strict) {
    $TrainArgs += "--pdf-stage1-strict"
}
& py @TrainArgs
exit $LASTEXITCODE
