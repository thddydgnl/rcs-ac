param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$CheckpointManifestPath = "",
    [string]$OutputDir = "",
    [string]$PrimaryReplayName = "target24",
    [string]$PrimaryReplayManifestPath = "",
    [string]$SecondaryReplayName = "steer2",
    [string]$SecondaryReplayManifestPath = "",
    [int]$MaxStepsPerFile = 400,
    [double]$MinNonzeroThrottleFraction = 0.5,
    [double]$MinNonzeroSteerFraction = 0.02,
    [int]$MinNonzeroSteerCount = 12,
    [double]$MaxNonzeroSteerFraction = 0.20,
    [double]$MaxDominantActionFraction = 0.85,
    [double]$MaxPreviousStuckAction8Fraction = 0.05,
    [int]$LatentProbeTrainSteps = 400,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
if ($CheckpointManifestPath -eq "") {
    throw "CheckpointManifestPath is required"
}
if ($PrimaryReplayManifestPath -eq "") {
    throw "PrimaryReplayManifestPath is required"
}
if ($OutputDir -eq "") {
    $OutputDir = Join-Path (Split-Path -Parent $CheckpointManifestPath) "cross_replay_probes"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$ActionProbeScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\probe_stage2_checkpoint_actions.ps1"
$LatentProbeScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\probe_stage2_replay_latents.ps1"

function Invoke-ReplayProbePair {
    param(
        [Parameter(Mandatory = $true)][string]$ReplayName,
        [Parameter(Mandatory = $true)][string]$ReplayManifestPath,
        [Parameter(Mandatory = $true)][bool]$WritePromotedConfig
    )

    $ReplayOutputDir = Join-Path $OutputDir $ReplayName
    New-Item -ItemType Directory -Force -Path $ReplayOutputDir | Out-Null
    $OutputConfigPath = Join-Path $OutputDir "clean_stage2_candidate_with_checkpoints.yaml"
    if (-not $WritePromotedConfig -and (Test-Path $OutputConfigPath)) {
        $OutputConfigPath = Join-Path $ReplayOutputDir "clean_stage2_candidate_with_checkpoints.shadow.yaml"
    }
    $ActionProbePath = Join-Path $ReplayOutputDir "stage2_checkpoint_action_probe.json"
    $LatentProbePath = Join-Path $ReplayOutputDir "stage2_replay_latent_probe.json"

    Write-Host "STAGE2_CROSS_REPLAY_PROBE_BEGIN=$ReplayName"
    Write-Host "REPLAY_MANIFEST_PATH=$ReplayManifestPath"
    powershell -ExecutionPolicy Bypass -File $ActionProbeScript `
        -ProjectRoot $ProjectRoot `
        -ConfigPath $ConfigPath `
        -CheckpointManifestPath $CheckpointManifestPath `
        -ReplayManifestPath $ReplayManifestPath `
        -OutputConfigPath $OutputConfigPath `
        -OutputPath $ActionProbePath `
        -MaxStepsPerFile $MaxStepsPerFile `
        -MinNonzeroThrottleFraction $MinNonzeroThrottleFraction `
        -MinNonzeroSteerFraction $MinNonzeroSteerFraction `
        -MinNonzeroSteerCount $MinNonzeroSteerCount `
        -MaxNonzeroSteerFraction $MaxNonzeroSteerFraction `
        -MaxDominantActionFraction $MaxDominantActionFraction `
        -MaxPreviousStuckAction8Fraction $MaxPreviousStuckAction8Fraction `
        -Device $Device
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    powershell -ExecutionPolicy Bypass -File $LatentProbeScript `
        -ProjectRoot $ProjectRoot `
        -ConfigPath $OutputConfigPath `
        -CheckpointManifestPath $CheckpointManifestPath `
        -ReplayManifestPath $ReplayManifestPath `
        -OutputPath $LatentProbePath `
        -TrainSteps $LatentProbeTrainSteps `
        -Device $Device
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    Write-Host "STAGE2_CROSS_REPLAY_PROBE_DONE=$ReplayName"
    Write-Host "ACTION_PROBE_PATH=$ActionProbePath"
    Write-Host "LATENT_PROBE_PATH=$LatentProbePath"
}

Write-Host "STAGE2_CANDIDATE_CROSS_REPLAY_PROBES_READY=True"
Write-Host "CONFIG_PATH=$ConfigPath"
Write-Host "CHECKPOINT_MANIFEST_PATH=$CheckpointManifestPath"
Write-Host "OUTPUT_DIR=$OutputDir"
Invoke-ReplayProbePair `
    -ReplayName $PrimaryReplayName `
    -ReplayManifestPath $PrimaryReplayManifestPath `
    -WritePromotedConfig $true

if ($SecondaryReplayManifestPath -ne "") {
    Invoke-ReplayProbePair `
        -ReplayName $SecondaryReplayName `
        -ReplayManifestPath $SecondaryReplayManifestPath `
        -WritePromotedConfig $false
}

Write-Host "STAGE2_CANDIDATE_CROSS_REPLAY_PROBES_COMPLETE=True"
