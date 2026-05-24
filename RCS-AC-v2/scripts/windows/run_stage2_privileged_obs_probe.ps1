param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$EnsureCarlaScript = "",
    [string]$StartCarlaScript = "",
    [string]$EnvConfigPathOverride = "",
    [string]$SupportCodePath = "",
    [string]$V11CodePath = "",
    [string]$SharedCorePath = "",
    [switch]$StepNeutral,
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
& powershell -ExecutionPolicy Bypass -File $DependencyCheckScript `
    -ProjectRoot $ProjectRoot `
    -ConfigPath $ConfigPath `
    -EnsureCarlaScript $EnsureCarlaScript `
    -StartCarlaScript $StartCarlaScript `
    -EnvConfigPathOverride $EnvConfigPathOverride `
    -SupportCodePath $SupportCodePath `
    -V11CodePath $V11CodePath `
    -SharedCorePath $SharedCorePath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $SkipCarlaReady) {
    & powershell -ExecutionPolicy Bypass -File $EnsureCarlaScript `
        -ProjectRoot $ProjectRoot `
        -StartScript $StartCarlaScript
}

$OutputPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_128\privileged_obs_probe.json"
$ProbeArgs = @(
    "-3.12", "-B", "-m", "rcs_ac_v2.evaluation.probe_stage2_windows_env",
    "--config", $ConfigPath,
    "--output", $OutputPath
)
if ($StepNeutral) {
    $ProbeArgs += @("--step-neutral")
}

Write-Host "STAGE2_PRIVILEGED_OBS_PROBE_READY=True"
Write-Host "CONFIG_PATH=$ConfigPath"
Write-Host "OUTPUT_PATH=$OutputPath"
& py @ProbeArgs
exit $LASTEXITCODE
