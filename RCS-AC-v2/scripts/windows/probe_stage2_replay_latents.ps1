param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$CheckpointManifestPath = "",
    [string]$ReplayManifestPath = "",
    [string]$OutputPath = "",
    [int]$TrainSteps = 300,
    [double]$LearningRate = 0.01,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$env:PYTHONPATH"

if ($CheckpointManifestPath -eq "") {
    throw "CheckpointManifestPath is required"
}
if ($ReplayManifestPath -eq "") {
    throw "ReplayManifestPath is required"
}
if ($OutputPath -eq "") {
    $OutputPath = Join-Path (Split-Path -Parent $CheckpointManifestPath) "stage2_replay_latent_probe.json"
}

Write-Host "STAGE2_REPLAY_LATENT_PROBE_READY=True"
Write-Host "CONFIG_PATH=$ConfigPath"
Write-Host "CHECKPOINT_MANIFEST_PATH=$CheckpointManifestPath"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "OUTPUT_PATH=$OutputPath"

& py -3.12 -B -m rcs_ac_v2.evaluation.replay_latent_probe `
    --config $ConfigPath `
    --checkpoint-manifest $CheckpointManifestPath `
    --replay-manifest $ReplayManifestPath `
    --project-root $ProjectRoot `
    --train-steps $TrainSteps `
    --learning-rate $LearningRate `
    --device $Device `
    --output $OutputPath
exit $LASTEXITCODE
