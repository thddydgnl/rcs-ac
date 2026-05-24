param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ReplayPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged",
    [string]$ReplayManifestPath = "",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay",
    [int]$SequenceLength = 16,
    [int]$BatchSize = 8,
    [int]$WorldModelSteps = 200,
    [int]$PolicySteps = 200,
    [double]$LearningRate = 0.0003,
    [int]$HiddenDim = 128,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$env:PYTHONPATH"
if ($ReplayManifestPath -eq "") {
    $ReplayManifestPath = Join-Path $ReplayPath "stage2_privileged_replay_manifest.json"
}

Write-Host "STAGE2_PRIVILEGED_REPLAY_TRAIN_READY=True"
Write-Host "REPLAY_PATH=$ReplayPath"
Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
Write-Host "OUTPUT_DIR=$OutputDir"
& py -3.12 -B -m rcs_ac_v2.training.train_stage2_privileged_from_replay `
    --replay $ReplayPath `
    --replay-manifest $ReplayManifestPath `
    --output-dir $OutputDir `
    --sequence-length $SequenceLength `
    --batch-size $BatchSize `
    --world-model-steps $WorldModelSteps `
    --policy-steps $PolicySteps `
    --learning-rate $LearningRate `
    --hidden-dim $HiddenDim `
    --device $Device
exit $LASTEXITCODE
