param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_steer2_seq010_aux1_balpow05_iw5\clean_512_stage2_privileged_7f87656_steer2_seq010_aux1_balpow05_iw5_with_checkpoints.yaml",
    [string]$PrimaryReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_target22_failure_window",
    [string]$PrimaryReplayName = "target22",
    [string]$SecondaryReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_6e74d9e_long512_t25_steer2_probe",
    [string]$SecondaryReplayName = "steer2",
    [string]$TertiaryReplayDir = "",
    [string]$TertiaryReplayName = "late",
    [string]$QuaternaryReplayDir = "",
    [string]$QuaternaryReplayName = "late2",
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
    [double]$RoutePrioritySampleFraction = 0.0,
    [string]$RoutePriorityActionIds = "",
    [string]$RoutePriorityComponents = "",
    [double]$RoutePriorityMinRouteProgress = 0.0,
    [double]$RoutePriorityMaxRouteProgress = 1.0,
    [double]$MovingSteerFalsePositiveWeight = 0.0,
    [double]$MovingSteerBinaryWeight = 0.0,
    [double]$MovingSteerNonmovingMarginWeight = 0.0,
    [double]$MovingSteerNonmovingMargin = 0.25,
    [double]$RouteLateRightSteerMarginWeight = 0.0,
    [double]$RouteLateRightSteerMargin = 0.25,
    [double]$RouteLateRightSteerMinSpeedNorm = 0.0,
    [double]$RouteLateRightSteerMinRouteProgress = 0.0,
    [double]$RouteLateRightSteerMaxRouteProgress = 1.0,
    [double]$WorldModelActionAuxWeight = 1.0,
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
    [string]$ForbiddenActionIds = "",
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
$TertiaryReplayManifestPath = if ($TertiaryReplayDir -ne "") { Join-Path $TertiaryReplayDir "stage2_privileged_replay_manifest.json" } else { "" }
$QuaternaryReplayManifestPath = if ($QuaternaryReplayDir -ne "") { Join-Path $QuaternaryReplayDir "stage2_privileged_replay_manifest.json" } else { "" }
$MakeScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\make_stage2_mixed_replay_manifest.ps1"
$CheckScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\check_stage2_replay_training_detached.ps1"
$StartScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\start_stage2_replay_training_detached.ps1"
$CrossProbeScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\probe_stage2_candidate_cross_replays.ps1"

Write-Host "STAGE2_MIXED_CANDIDATE_CYCLE_READY=True"
Write-Host "MIXED_REPLAY_DIR=$MixedReplayDir"
Write-Host "MIXED_REPLAY_MANIFEST_PATH=$MixedManifestPath"
Write-Host "PRIMARY_REPLAY_MANIFEST_PATH=$PrimaryReplayManifestPath"
Write-Host "SECONDARY_REPLAY_MANIFEST_PATH=$SecondaryReplayManifestPath"
if ($TertiaryReplayManifestPath -ne "") {
    Write-Host "TERTIARY_REPLAY_MANIFEST_PATH=$TertiaryReplayManifestPath"
}
if ($QuaternaryReplayManifestPath -ne "") {
    Write-Host "QUATERNARY_REPLAY_MANIFEST_PATH=$QuaternaryReplayManifestPath"
}
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
    if ($TertiaryReplayDir -ne "") {
        $MakeArgs += @(
            "-TertiaryReplayDir", $TertiaryReplayDir,
            "-TertiaryName", $TertiaryReplayName
        )
    }
    if ($QuaternaryReplayDir -ne "") {
        $MakeArgs += @(
            "-QuaternaryReplayDir", $QuaternaryReplayDir,
            "-QuaternaryName", $QuaternaryReplayName
        )
    }
    if ($ForceMixedReplay) {
        $MakeArgs += "-Force"
    }
    powershell @MakeArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$RequiredManifests = @($MixedManifestPath, $PrimaryReplayManifestPath, $SecondaryReplayManifestPath)
if ($TertiaryReplayManifestPath -ne "") {
    $RequiredManifests += $TertiaryReplayManifestPath
}
if ($QuaternaryReplayManifestPath -ne "") {
    $RequiredManifests += $QuaternaryReplayManifestPath
}
foreach ($RequiredManifest in $RequiredManifests) {
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
        $CrossProbeArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", $CrossProbeScript,
            "-ProjectRoot", $ProjectRoot,
            "-ConfigPath", $ConfigPath,
            "-CheckpointManifestPath", $CheckpointManifestPath,
            "-OutputDir", $CrossProbeOutputDir,
            "-PrimaryReplayName", $PrimaryReplayName,
            "-PrimaryReplayManifestPath", $PrimaryReplayManifestPath,
            "-SecondaryReplayName", $SecondaryReplayName,
            "-SecondaryReplayManifestPath", $SecondaryReplayManifestPath,
            "-Device", $Device
        )
        if ($ForbiddenActionIds -ne "") {
            $CrossProbeArgs += @("-ForbiddenActionIds", $ForbiddenActionIds)
        }
        powershell @CrossProbeArgs
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
    tertiary_replay_name = $TertiaryReplayName
    quaternary_replay_name = $QuaternaryReplayName
    requested_output_dir = $RequestedOutputDir
    active_output_dir = $RerunOutputDir
    mixed_replay_manifest_path = $MixedManifestPath
    primary_replay_manifest_path = $PrimaryReplayManifestPath
    secondary_replay_manifest_path = $SecondaryReplayManifestPath
    tertiary_replay_manifest_path = $TertiaryReplayManifestPath
    quaternary_replay_manifest_path = $QuaternaryReplayManifestPath
    task_name = $TaskName
    seed = $Seed
    steer_sequence_fraction = $SteerSequenceFraction
    route_priority_sample_fraction = $RoutePrioritySampleFraction
    route_priority_action_ids = $RoutePriorityActionIds
    route_priority_components = $RoutePriorityComponents
    route_priority_min_route_progress = $RoutePriorityMinRouteProgress
    route_priority_max_route_progress = $RoutePriorityMaxRouteProgress
    moving_steer_false_positive_weight = $MovingSteerFalsePositiveWeight
    moving_steer_binary_weight = $MovingSteerBinaryWeight
    moving_steer_nonmoving_margin_weight = $MovingSteerNonmovingMarginWeight
    moving_steer_nonmoving_margin = $MovingSteerNonmovingMargin
    route_late_right_steer_margin_weight = $RouteLateRightSteerMarginWeight
    route_late_right_steer_margin = $RouteLateRightSteerMargin
    route_late_right_steer_min_speed_norm = $RouteLateRightSteerMinSpeedNorm
    route_late_right_steer_min_route_progress = $RouteLateRightSteerMinRouteProgress
    route_late_right_steer_max_route_progress = $RouteLateRightSteerMaxRouteProgress
    component_sample_weights = $ComponentSampleWeights
    action_mass_cap_weight = $ActionMassCapWeight
    action_mass_caps = $ActionMassCaps
    route_speed_action_mass_cap_weight = $RouteSpeedActionMassCapWeight
    route_speed_action_ids = $RouteSpeedActionIds
    route_speed_action_mass_cap = $RouteSpeedActionMassCap
    route_speed_min_speed_norm = $RouteSpeedMinSpeedNorm
    route_speed_min_route_progress = $RouteSpeedMinRouteProgress
    route_speed_max_route_progress = $RouteSpeedMaxRouteProgress
    route_action_mass_cap_weight = $RouteActionMassCapWeight
    route_action_mass_cap_ids = $RouteActionMassCapIds
    route_action_mass_cap = $RouteActionMassCap
    route_action_mass_cap_min_speed_norm = $RouteActionMassCapMinSpeedNorm
    route_action_mass_cap_min_route_progress = $RouteActionMassCapMinRouteProgress
    route_action_mass_cap_max_route_progress = $RouteActionMassCapMaxRouteProgress
    route_action_logit_cap_weight = $RouteActionLogitCapWeight
    route_action_logit_cap_ids = $RouteActionLogitCapIds
    route_action_logit_cap_margin = $RouteActionLogitCapMargin
    route_action_logit_cap_min_speed_norm = $RouteActionLogitCapMinSpeedNorm
    route_action_logit_cap_min_route_progress = $RouteActionLogitCapMinRouteProgress
    route_action_logit_cap_max_route_progress = $RouteActionLogitCapMaxRouteProgress
    route_action_distribution_weight = $RouteActionDistributionWeight
    route_action_distribution_min_speed_norm = $RouteActionDistributionMinSpeedNorm
    route_action_distribution_min_route_progress = $RouteActionDistributionMinRouteProgress
    route_action_distribution_max_route_progress = $RouteActionDistributionMaxRouteProgress
    route_late_action_logit_cap_weight = $RouteLateActionLogitCapWeight
    route_late_action_logit_cap_ids = $RouteLateActionLogitCapIds
    route_late_action_logit_cap_margin = $RouteLateActionLogitCapMargin
    route_late_action_logit_cap_min_speed_norm = $RouteLateActionLogitCapMinSpeedNorm
    route_late_action_logit_cap_min_route_progress = $RouteLateActionLogitCapMinRouteProgress
    route_late_action_logit_cap_max_route_progress = $RouteLateActionLogitCapMaxRouteProgress
    route_throttle_ceiling_weight = $RouteThrottleCeilingWeight
    route_throttle_ceiling = $RouteThrottleCeiling
    route_throttle_min_speed_norm = $RouteThrottleMinSpeedNorm
    route_throttle_min_route_progress = $RouteThrottleMinRouteProgress
    route_throttle_max_route_progress = $RouteThrottleMaxRouteProgress
    route_late_throttle_ceiling_weight = $RouteLateThrottleCeilingWeight
    route_late_throttle_ceiling = $RouteLateThrottleCeiling
    route_late_throttle_min_speed_norm = $RouteLateThrottleMinSpeedNorm
    route_late_throttle_min_route_progress = $RouteLateThrottleMinRouteProgress
    route_late_throttle_max_route_progress = $RouteLateThrottleMaxRouteProgress
    route_action_override_weight = $RouteActionOverrideWeight
    route_action_override_id = $RouteActionOverrideId
    route_action_override_min_speed_norm = $RouteActionOverrideMinSpeedNorm
    route_action_override_min_route_progress = $RouteActionOverrideMinRouteProgress
    route_action_override_max_route_progress = $RouteActionOverrideMaxRouteProgress
    updated_at = (Get-Date -Format o)
}
$LatestPointer | ConvertTo-Json -Depth 4 | Set-Content -Path $LatestPointerPath
Write-Host "MIXED_CANDIDATE_LATEST_POINTER_UPDATED=True"
Write-Host "MIXED_CANDIDATE_LATEST_POINTER_ACTIVE_OUTPUT_DIR=$RerunOutputDir"

$StartArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $StartScript,
    "-ProjectRoot", $ProjectRoot,
    "-ReplayPath", $MixedReplayDir,
    "-ReplayManifestPath", $MixedManifestPath,
    "-OutputDir", $RerunOutputDir,
    "-Seed", $Seed,
    "-SequenceLength", $SequenceLength,
    "-BatchSize", $BatchSize,
    "-WorldModelSteps", $WorldModelSteps,
    "-PolicySteps", $PolicySteps,
    "-LearningRate", $LearningRate,
    "-ImitationWeight", $ImitationWeight,
    "-BehaviorWeight", $BehaviorWeight,
    "-ImitationClassBalance",
    "-ImitationBalancePower", $ImitationBalancePower,
    "-SteerSequenceFraction", $SteerSequenceFraction,
    "-RoutePrioritySampleFraction", $RoutePrioritySampleFraction,
    "-RoutePriorityMinRouteProgress", $RoutePriorityMinRouteProgress,
    "-RoutePriorityMaxRouteProgress", $RoutePriorityMaxRouteProgress,
    "-MovingSteerFalsePositiveWeight", $MovingSteerFalsePositiveWeight,
    "-MovingSteerBinaryWeight", $MovingSteerBinaryWeight,
    "-MovingSteerNonmovingMarginWeight", $MovingSteerNonmovingMarginWeight,
    "-MovingSteerNonmovingMargin", $MovingSteerNonmovingMargin,
    "-RouteLateRightSteerMarginWeight", $RouteLateRightSteerMarginWeight,
    "-RouteLateRightSteerMargin", $RouteLateRightSteerMargin,
    "-RouteLateRightSteerMinSpeedNorm", $RouteLateRightSteerMinSpeedNorm,
    "-RouteLateRightSteerMinRouteProgress", $RouteLateRightSteerMinRouteProgress,
    "-RouteLateRightSteerMaxRouteProgress", $RouteLateRightSteerMaxRouteProgress,
    "-WorldModelActionAuxWeight", $WorldModelActionAuxWeight,
    "-ActionMassCapWeight", $ActionMassCapWeight,
    "-RouteSpeedActionMassCapWeight", $RouteSpeedActionMassCapWeight,
    "-RouteSpeedActionMassCap", $RouteSpeedActionMassCap,
    "-RouteSpeedMinSpeedNorm", $RouteSpeedMinSpeedNorm,
    "-RouteSpeedMinRouteProgress", $RouteSpeedMinRouteProgress,
    "-RouteSpeedMaxRouteProgress", $RouteSpeedMaxRouteProgress,
    "-RouteActionMassCapWeight", $RouteActionMassCapWeight,
    "-RouteActionMassCap", $RouteActionMassCap,
    "-RouteActionMassCapMinSpeedNorm", $RouteActionMassCapMinSpeedNorm,
    "-RouteActionMassCapMinRouteProgress", $RouteActionMassCapMinRouteProgress,
    "-RouteActionMassCapMaxRouteProgress", $RouteActionMassCapMaxRouteProgress,
    "-RouteActionLogitCapWeight", $RouteActionLogitCapWeight,
    "-RouteActionLogitCapMargin", $RouteActionLogitCapMargin,
    "-RouteActionLogitCapMinSpeedNorm", $RouteActionLogitCapMinSpeedNorm,
    "-RouteActionLogitCapMinRouteProgress", $RouteActionLogitCapMinRouteProgress,
    "-RouteActionLogitCapMaxRouteProgress", $RouteActionLogitCapMaxRouteProgress,
    "-RouteActionDistributionWeight", $RouteActionDistributionWeight,
    "-RouteActionDistributionMinSpeedNorm", $RouteActionDistributionMinSpeedNorm,
    "-RouteActionDistributionMinRouteProgress", $RouteActionDistributionMinRouteProgress,
    "-RouteActionDistributionMaxRouteProgress", $RouteActionDistributionMaxRouteProgress,
    "-RouteLateActionLogitCapWeight", $RouteLateActionLogitCapWeight,
    "-RouteLateActionLogitCapMargin", $RouteLateActionLogitCapMargin,
    "-RouteLateActionLogitCapMinSpeedNorm", $RouteLateActionLogitCapMinSpeedNorm,
    "-RouteLateActionLogitCapMinRouteProgress", $RouteLateActionLogitCapMinRouteProgress,
    "-RouteLateActionLogitCapMaxRouteProgress", $RouteLateActionLogitCapMaxRouteProgress,
    "-RouteThrottleCeilingWeight", $RouteThrottleCeilingWeight,
    "-RouteThrottleCeiling", $RouteThrottleCeiling,
    "-RouteThrottleMinSpeedNorm", $RouteThrottleMinSpeedNorm,
    "-RouteThrottleMinRouteProgress", $RouteThrottleMinRouteProgress,
    "-RouteThrottleMaxRouteProgress", $RouteThrottleMaxRouteProgress,
    "-RouteLateThrottleCeilingWeight", $RouteLateThrottleCeilingWeight,
    "-RouteLateThrottleCeiling", $RouteLateThrottleCeiling,
    "-RouteLateThrottleMinSpeedNorm", $RouteLateThrottleMinSpeedNorm,
    "-RouteLateThrottleMinRouteProgress", $RouteLateThrottleMinRouteProgress,
    "-RouteLateThrottleMaxRouteProgress", $RouteLateThrottleMaxRouteProgress,
    "-RouteActionOverrideWeight", $RouteActionOverrideWeight,
    "-RouteActionOverrideId", $RouteActionOverrideId,
    "-RouteActionOverrideMinSpeedNorm", $RouteActionOverrideMinSpeedNorm,
    "-RouteActionOverrideMinRouteProgress", $RouteActionOverrideMinRouteProgress,
    "-RouteActionOverrideMaxRouteProgress", $RouteActionOverrideMaxRouteProgress,
    "-HiddenDim", $HiddenDim,
    "-Device", $Device,
    "-TaskName", $TaskName
)
if ($ComponentSampleWeights -ne "") {
    $StartArgs += @("-ComponentSampleWeights", $ComponentSampleWeights)
}
if ($ActionMassCaps -ne "") {
    $StartArgs += @("-ActionMassCaps", $ActionMassCaps)
}
if ($RouteSpeedActionIds -ne "") {
    $StartArgs += @("-RouteSpeedActionIds", $RouteSpeedActionIds)
}
if ($RoutePriorityActionIds -ne "") {
    $StartArgs += @("-RoutePriorityActionIds", $RoutePriorityActionIds)
}
if ($RoutePriorityComponents -ne "") {
    $StartArgs += @("-RoutePriorityComponents", $RoutePriorityComponents)
}
if ($RouteActionMassCapIds -ne "") {
    $StartArgs += @("-RouteActionMassCapIds", $RouteActionMassCapIds)
}
if ($RouteActionLogitCapIds -ne "") {
    $StartArgs += @("-RouteActionLogitCapIds", $RouteActionLogitCapIds)
}
if ($RouteLateActionLogitCapIds -ne "") {
    $StartArgs += @("-RouteLateActionLogitCapIds", $RouteLateActionLogitCapIds)
}
powershell @StartArgs
exit $LASTEXITCODE
