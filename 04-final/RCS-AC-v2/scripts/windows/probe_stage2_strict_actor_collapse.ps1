param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$CheckpointManifestPath = "",
    [string]$ReplayManifestPath = "",
    [string]$OutputPath = "",
    [int]$MaxStepsPerFile = 128,
    [int]$RolloutHorizon = 16,
    [double]$Gamma = 0.99,
    [double]$Lambda = 1.0,
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
if ($CheckpointManifestPath -eq "") {
    throw "CheckpointManifestPath is required"
}
if ($ReplayManifestPath -eq "") {
    throw "ReplayManifestPath is required"
}
if ($OutputPath -eq "") {
    $OutputPath = Join-Path (Split-Path -Parent $CheckpointManifestPath) "stage2_strict_actor_collapse_probe.json"
}

Write-Host "STAGE2_STRICT_ACTOR_COLLAPSE_PROBE_READY=True"
Write-Host "CONFIG_PATH=$ConfigPath"
Write-Host "CHECKPOINT_MANIFEST_PATH=$CheckpointManifestPath"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "OUTPUT_PATH=$OutputPath"

& py -3.12 -B -m rcs_ac_v2.evaluation.strict_actor_collapse_probe `
    --config $ConfigPath `
    --checkpoint-manifest $CheckpointManifestPath `
    --replay-manifest $ReplayManifestPath `
    --project-root $ProjectRoot `
    --max-steps-per-file $MaxStepsPerFile `
    --rollout-horizon $RolloutHorizon `
    --gamma $Gamma `
    --lambda $Lambda `
    --device $Device `
    --output $OutputPath
exit $LASTEXITCODE
