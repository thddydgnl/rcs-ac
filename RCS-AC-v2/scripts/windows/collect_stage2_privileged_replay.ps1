param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged",
    [int]$Episodes = 4,
    [int]$MaxSteps = 256,
    [double]$TargetSpeedKmh = 18.0,
    [double]$SteerSign = -1.0,
    [string]$EnsureCarlaScript = "",
    [string]$StartCarlaScript = "",
    [switch]$SkipGitPull,
    [switch]$SkipCarlaReady
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$SupportCodePath = "$ProjectRoot\RCS-AC-v1.1\goal_achieved_round_0002_execution\support_code"
$V11CodePath = "$ProjectRoot\RCS-AC-v1.1\code"
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$SupportCodePath;$V11CodePath;$WorkspaceRoot;$env:PYTHONPATH"

if ($EnsureCarlaScript -eq "") {
    $EnsureCarlaScript = "$ProjectRoot\RCS-AC-v2\scripts\windows\ensure_carla_ready.ps1"
}
if ($StartCarlaScript -eq "") {
    $StartCarlaScript = "$ProjectRoot\RCS-AC-v2\scripts\windows\start_carla_interactive.ps1"
}

if (-not $SkipGitPull) {
    git pull --ff-only
}

$DependencyCheckScript = "$ProjectRoot\RCS-AC-v2\scripts\windows\check_stage2_windows_dependencies.ps1"
& powershell -ExecutionPolicy Bypass -File $DependencyCheckScript `
    -ProjectRoot $ProjectRoot `
    -ConfigPath $ConfigPath `
    -EnsureCarlaScript $EnsureCarlaScript `
    -StartCarlaScript $StartCarlaScript `
    -SupportCodePath $SupportCodePath `
    -V11CodePath $V11CodePath `
    -SharedCorePath "$WorkspaceRoot\shared_core"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $SkipCarlaReady) {
    & powershell -ExecutionPolicy Bypass -File $EnsureCarlaScript `
        -ProjectRoot $ProjectRoot `
        -StartScript $StartCarlaScript
}

Write-Host "STAGE2_PRIVILEGED_REPLAY_COLLECT_READY=True"
Write-Host "CONFIG_PATH=$ConfigPath"
Write-Host "OUTPUT_DIR=$OutputDir"
& py -3.12 -B -m rcs_ac_v2.training.collect_stage2_privileged_replay `
    --config $ConfigPath `
    --output-dir $OutputDir `
    --episodes $Episodes `
    --max-steps $MaxSteps `
    --target-speed-kmh $TargetSpeedKmh `
    --steer-sign $SteerSign
exit $LASTEXITCODE
