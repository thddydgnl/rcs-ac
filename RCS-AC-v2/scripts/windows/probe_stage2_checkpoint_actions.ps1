param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$CheckpointManifestPath = "",
    [string]$ReplayManifestPath = "",
    [string]$OutputConfigPath = "",
    [string]$OutputPath = "",
    [int]$MaxStepsPerFile = 32,
    [double]$MinNonzeroThrottleFraction = 0.5,
    [double]$MinNonzeroSteerFraction = 0.02,
    [int]$MinNonzeroSteerCount = 12,
    [double]$MaxNonzeroSteerFraction = 0.5,
    [double]$MaxPreviousStuckAction8Fraction = 0.05,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$env:PYTHONPATH"

if ($CheckpointManifestPath -eq "") {
    throw "CheckpointManifestPath is required"
}
if ($OutputPath -eq "") {
    $OutputPath = Join-Path (Split-Path -Parent $CheckpointManifestPath) "stage2_checkpoint_action_probe.json"
}
if ($OutputConfigPath -eq "") {
    $OutputConfigPath = Join-Path (Split-Path -Parent $CheckpointManifestPath) "clean_128_stage2_privileged_with_checkpoints.yaml"
}

Write-Host "STAGE2_CHECKPOINT_ACTION_PROBE_READY=True"
Write-Host "CONFIG_PATH=$ConfigPath"
Write-Host "CHECKPOINT_MANIFEST_PATH=$CheckpointManifestPath"
Write-Host "OUTPUT_CONFIG_PATH=$OutputConfigPath"
Write-Host "OUTPUT_PATH=$OutputPath"

$ProbeArgs = @(
    "-3.12", "-B", "-m", "rcs_ac_v2.evaluation.checkpoint_action_probe",
    "--config", $ConfigPath,
    "--checkpoint-manifest", $CheckpointManifestPath,
    "--project-root", $ProjectRoot,
    "--output-config", $OutputConfigPath,
    "--output", $OutputPath,
    "--max-steps-per-file", $MaxStepsPerFile,
    "--min-nonzero-throttle-fraction", $MinNonzeroThrottleFraction,
    "--min-nonzero-steer-fraction", $MinNonzeroSteerFraction,
    "--min-nonzero-steer-count", $MinNonzeroSteerCount,
    "--max-nonzero-steer-fraction", $MaxNonzeroSteerFraction,
    "--max-previous-stuck-action8-fraction", $MaxPreviousStuckAction8Fraction,
    "--device", $Device
)
if ($ReplayManifestPath -ne "") {
    $ProbeArgs += @("--replay-manifest", $ReplayManifestPath)
}

& py @ProbeArgs
exit $LASTEXITCODE
