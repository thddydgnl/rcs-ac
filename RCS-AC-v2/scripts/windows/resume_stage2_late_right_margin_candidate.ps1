param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656",
    [string]$GatePath = "",
    [switch]$SkipGitPull,
    [switch]$SkipDependencyCheck,
    [switch]$RequireCuda,
    [switch]$PlanOnly,
    [switch]$MakeReplayIfMissing,
    [switch]$ForceMixedReplay,
    [switch]$StartIfMissing,
    [switch]$ProbeWhenReady,
    [switch]$SummarizeOnly,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$LateRightScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\run_stage2_late_right_margin_candidate.ps1"
$DependencyScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\check_stage2_windows_dependencies.ps1"
$SummaryScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\summarize_stage2_candidate_gate.ps1"
$DefaultGatePath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\f7d903a2_mixed_t22_steer2_aux1_iw5_ssf005_rightmargin_w1_cap131619_pre0666_cross_probe\stage2_candidate_cross_replay_gate.json"
if ($GatePath -eq "") {
    $GatePath = $DefaultGatePath
}

Write-Host "STAGE2_LATE_RIGHT_RESUME_READY=True"
Write-Host "PROJECT_ROOT=$ProjectRoot"
Write-Host "DEVICE=$Device"
Write-Host "GATE_PATH=$GatePath"
Write-Host "OFFICIAL_512_CLEAN_EVAL_BLOCKED_UNTIL_READY_FOR_CLEAN_EVAL=True"
Write-Host "OFFICIAL_512_CLEAN_EVAL_NOT_CALLED_BY_THIS_WRAPPER=True"

Set-Location $ProjectRoot

if (-not $SkipGitPull) {
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host "STAGE2_LATE_RIGHT_RESUME_GIT_PULL_OK=False"
        exit 3
    }
    Write-Host "STAGE2_LATE_RIGHT_RESUME_GIT_PULL_OK=True"
}
else {
    Write-Host "STAGE2_LATE_RIGHT_RESUME_GIT_PULL_SKIPPED=True"
}

if (-not $SkipDependencyCheck) {
    $DependencyArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $DependencyScript,
        "-ProjectRoot", $ProjectRoot
    )
    if ($RequireCuda -or $Device -eq "cuda") {
        $DependencyArgs += "-RequireCuda"
    }
    powershell @DependencyArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "STAGE2_LATE_RIGHT_RESUME_DEPENDENCY_OK=False"
        if ($LASTEXITCODE -eq 4) {
            exit 10
        }
        exit 2
    }
    Write-Host "STAGE2_LATE_RIGHT_RESUME_DEPENDENCY_OK=True"
}
else {
    Write-Host "STAGE2_LATE_RIGHT_RESUME_DEPENDENCY_SKIPPED=True"
}

$CandidateArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $LateRightScript,
    "-ProjectRoot", $ProjectRoot,
    "-Device", $Device
)
if ($MakeReplayIfMissing) {
    $CandidateArgs += "-MakeReplayIfMissing"
}
if ($ForceMixedReplay) {
    $CandidateArgs += "-ForceMixedReplay"
}
if ($StartIfMissing) {
    $CandidateArgs += "-StartIfMissing"
}
if ($ProbeWhenReady) {
    $CandidateArgs += "-ProbeWhenReady"
}

$PlanArgs = $CandidateArgs + "-PlanOnly"
Write-Host "STAGE2_LATE_RIGHT_RESUME_PLANONLY_BEGIN=True"
powershell @PlanArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "STAGE2_LATE_RIGHT_RESUME_PLANONLY_OK=False"
    exit 4
}
Write-Host "STAGE2_LATE_RIGHT_RESUME_PLANONLY_OK=True"

if (-not $PlanOnly -and -not $SummarizeOnly) {
    Write-Host "STAGE2_LATE_RIGHT_RESUME_CANDIDATE_BEGIN=True"
    powershell @CandidateArgs
    $CandidateExitCode = $LASTEXITCODE
    if ($CandidateExitCode -eq 3) {
        Write-Host "STAGE2_LATE_RIGHT_RESUME_CANDIDATE_RUNNING=True"
        Write-Host "STAGE2_LATE_RIGHT_RESUME_NEXT_ACTION=Rerun this wrapper with -ProbeWhenReady after detached training finishes."
        exit 3
    }
    if ($CandidateExitCode -ne 0) {
        Write-Host "STAGE2_LATE_RIGHT_RESUME_CANDIDATE_OK=False"
        if ($ProbeWhenReady -and (Test-Path $GatePath)) {
            Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_EXISTS=True"
            powershell -ExecutionPolicy Bypass -File $SummaryScript `
                -ProjectRoot $ProjectRoot `
                -GatePath $GatePath `
                -CandidateName late_right_margin `
                -FallbackCandidateName target22
            $SummaryExitCode = $LASTEXITCODE
            if ($SummaryExitCode -eq 4) {
                Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_SUMMARY_REJECTED=True"
            }
            elseif ($SummaryExitCode -ne 0) {
                Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_SUMMARY_OK=False"
            }
            else {
                Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_SUMMARY_READY_FOR_CLEAN_EVAL=True"
            }
        }
        exit 4
    }
    Write-Host "STAGE2_LATE_RIGHT_RESUME_CANDIDATE_OK=True"
}
elseif ($PlanOnly) {
    Write-Host "STAGE2_LATE_RIGHT_RESUME_PLAN_ONLY_STOP=True"
}

if ($ProbeWhenReady -or $SummarizeOnly) {
    if (-not (Test-Path $GatePath)) {
        Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_EXISTS=False"
        Write-Host "STAGE2_LATE_RIGHT_RESUME_NEXT_ACTION=Finish candidate training and cross replay probes before official 512 clean eval."
        exit 5
    }
    Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_EXISTS=True"
    powershell -ExecutionPolicy Bypass -File $SummaryScript `
        -ProjectRoot $ProjectRoot `
        -GatePath $GatePath `
        -CandidateName late_right_margin `
        -FallbackCandidateName target22
    $SummaryExitCode = $LASTEXITCODE
    if ($SummaryExitCode -ne 0 -and $SummaryExitCode -ne 4) {
        Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_SUMMARY_OK=False"
        exit 6
    }
    if ($SummaryExitCode -eq 4) {
        Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_SUMMARY_REJECTED=True"
    }
    else {
        Write-Host "STAGE2_LATE_RIGHT_RESUME_GATE_SUMMARY_READY_FOR_CLEAN_EVAL=True"
    }
}

Write-Host "STAGE2_LATE_RIGHT_RESUME_DONE=True"
exit 0
