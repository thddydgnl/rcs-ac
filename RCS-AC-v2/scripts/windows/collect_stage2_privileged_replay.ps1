param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged",
    [int]$Episodes = 4,
    [int]$MaxSteps = 256,
    [double]$TargetSpeedKmh = 18.0,
    [double]$SteerSign = -1.0,
    [double]$RecordAfterRouteProgress = 0.0,
    [int]$MinRecordedSteps = 1,
    [string]$EnsureCarlaScript = "",
    [string]$StartCarlaScript = "",
    [string]$EnvConfigPathOverride = "",
    [string]$SupportCodePath = "",
    [string]$V11CodePath = "",
    [string]$SharedCorePath = "",
    [switch]$SkipGitPull,
    [switch]$SkipCarlaReady
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
if ($SupportCodePath -eq "") {
    $DefaultSupportCodePath = "$ProjectRoot\RCS-AC-v1.1\goal_achieved_round_0002_execution\support_code"
    if (Test-Path $DefaultSupportCodePath) {
        $SupportCodePath = $DefaultSupportCodePath
    }
}
if ($V11CodePath -eq "") {
    $DefaultV11CodePath = "$ProjectRoot\RCS-AC-v1.1\code"
    if (Test-Path $DefaultV11CodePath) {
        $V11CodePath = $DefaultV11CodePath
    }
}
if ($SharedCorePath -eq "") {
    $SharedCorePath = "$WorkspaceRoot\shared_core"
}
if ($EnvConfigPathOverride -ne "") {
    $env:RCS_AC_V2_CARLA_ENV_CONFIG_PATH = $EnvConfigPathOverride
}
$pythonPathParts = @("$ProjectRoot\RCS-AC-v2\code")
if ($SupportCodePath -ne "") {
    $pythonPathParts += $SupportCodePath
}
if ($V11CodePath -ne "") {
    $pythonPathParts += $V11CodePath
}
$pythonPathParts += $WorkspaceRoot
$env:PYTHONPATH = (($pythonPathParts + @($env:PYTHONPATH)) -join ";")

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
$DependencyArgs = @{
    ProjectRoot = $ProjectRoot
    ConfigPath = $ConfigPath
    EnsureCarlaScript = $EnsureCarlaScript
    StartCarlaScript = $StartCarlaScript
}
if ($EnvConfigPathOverride -ne "") {
    $DependencyArgs.EnvConfigPathOverride = $EnvConfigPathOverride
}
if ($SupportCodePath -ne "") {
    $DependencyArgs.SupportCodePath = $SupportCodePath
}
if ($V11CodePath -ne "") {
    $DependencyArgs.V11CodePath = $V11CodePath
}
if ($SharedCorePath -ne "") {
    $DependencyArgs.SharedCorePath = $SharedCorePath
}
& powershell -ExecutionPolicy Bypass -File $DependencyCheckScript @DependencyArgs
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
    --steer-sign $SteerSign `
    --record-after-route-progress $RecordAfterRouteProgress `
    --min-recorded-steps $MinRecordedSteps
exit $LASTEXITCODE
