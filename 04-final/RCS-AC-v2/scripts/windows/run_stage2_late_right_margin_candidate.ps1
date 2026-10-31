param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656",
    [string]$ConfigPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_steer2_seq010_aux1_balpow05_iw5\clean_512_stage2_privileged_7f87656_steer2_seq010_aux1_balpow05_iw5_with_checkpoints.yaml",
    [string]$PrimaryReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_target22_failure_window",
    [string]$SecondaryReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_6e74d9e_long512_t25_steer2_probe",
    [string]$MixedReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_mixed_target22_steer2",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\f7d903a2_mixed_t22_steer2_aux1_iw5_ssf005_rightmargin_w1_cap131619_pre0666",
    [string]$CrossProbeOutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\f7d903a2_mixed_t22_steer2_aux1_iw5_ssf005_rightmargin_w1_cap131619_pre0666_cross_probe",
    [string]$TaskName = "RCSV2Stage2F7RightMarginCap131619",
    [switch]$SkipGitPull,
    [switch]$MakeReplayIfMissing,
    [switch]$ForceMixedReplay,
    [switch]$StartIfMissing,
    [switch]$ProbeWhenReady,
    [switch]$PlanOnly,
    [int]$Seed = 20260524,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$Runner = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\run_stage2_mixed_candidate_cycle.ps1"

Write-Host "STAGE2_LATE_RIGHT_MARGIN_CANDIDATE_READY=True"
Write-Host "PROJECT_ROOT=$ProjectRoot"
Write-Host "OUTPUT_DIR=$OutputDir"
Write-Host "CROSS_PROBE_OUTPUT_DIR=$CrossProbeOutputDir"
Write-Host "TASK_NAME=$TaskName"
Write-Host "DEVICE=$Device"
Write-Host "OFFICIAL_512_CLEAN_EVAL_BLOCKED_UNTIL_READY_FOR_CLEAN_EVAL=True"
Write-Host "OFFICIAL_512_CLEAN_EVAL_BLOCKER=Run summarize_stage2_candidate_gate.ps1 on stage2_candidate_cross_replay_gate.json and require ready_for_clean_eval=true before any official 512 clean eval."
Write-Host "GATE_SUMMARY_COMMAND=powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\summarize_stage2_candidate_gate.ps1 -ProjectRoot $ProjectRoot -GatePath $CrossProbeOutputDir\stage2_candidate_cross_replay_gate.json -CandidateName late_right_margin -FallbackCandidateName target22"

$ArgsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $Runner,
    "-ProjectRoot", $ProjectRoot,
    "-ConfigPath", $ConfigPath,
    "-PrimaryReplayDir", $PrimaryReplayDir,
    "-PrimaryReplayName", "target22",
    "-SecondaryReplayDir", $SecondaryReplayDir,
    "-SecondaryReplayName", "steer2",
    "-MixedReplayDir", $MixedReplayDir,
    "-OutputDir", $OutputDir,
    "-CrossProbeOutputDir", $CrossProbeOutputDir,
    "-TaskName", $TaskName,
    "-Seed", $Seed,
    "-WorldModelSteps", 2000,
    "-PolicySteps", 1000,
    "-ImitationWeight", 5.0,
    "-BehaviorWeight", 0.0,
    "-ImitationBalancePower", 0.5,
    "-SteerSequenceFraction", 0.05,
    "-WorldModelActionAuxWeight", 1.0,
    "-RouteLateRightSteerMarginWeight", 1.0,
    "-RouteLateRightSteerMargin", 0.25,
    "-RouteLateRightSteerMinSpeedNorm", 0.0,
    "-RouteLateRightSteerMinRouteProgress", 0.666,
    "-RouteLateRightSteerMaxRouteProgress", 0.70,
    "-RouteActionMassCapWeight", 4.0,
    "-RouteActionMassCapIds", "13,16,19",
    "-RouteActionMassCap", 0.08,
    "-RouteActionMassCapMinSpeedNorm", 0.0,
    "-RouteActionMassCapMinRouteProgress", 0.55,
    "-RouteActionMassCapMaxRouteProgress", 0.66,
    "-Device", $Device
)

if ($SkipGitPull) {
    $ArgsList += "-SkipGitPull"
}
if ($MakeReplayIfMissing) {
    $ArgsList += "-MakeReplayIfMissing"
}
if ($ForceMixedReplay) {
    $ArgsList += "-ForceMixedReplay"
}
if ($StartIfMissing) {
    $ArgsList += "-StartIfMissing"
}
if ($ProbeWhenReady) {
    $ArgsList += "-ProbeWhenReady"
}

if ($PlanOnly) {
    Write-Host "PLAN_ONLY=True"
    Write-Host "PLAN_ONLY_NO_GIT_PULL=True"
    Write-Host "PLAN_ONLY_NO_CUDA_GUARD=True"
    Write-Host "PLAN_ONLY_NO_SCHEDULED_TASK=True"
    Write-Host "RESOLVED_RUNNER=$Runner"
    Write-Host "RESOLVED_ARGS=$($ArgsList -join ' ')"
    exit 0
}

powershell @ArgsList
exit $LASTEXITCODE
