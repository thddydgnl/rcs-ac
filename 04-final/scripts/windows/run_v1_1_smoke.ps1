param(
    [Parameter(Mandatory = $true)]
    [string]$EnvConfig,

    [string]$RepoRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$V11Config = "configs\v1_1\full_bev_smoke.yaml",
    [string]$Dataset = "data\v1_1\bev_expert_tiny_carla_smoke.npz",
    [string]$CheckpointDir = "checkpoints\v1_1_bev_bc_tiny_carla_smoke",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 2000,
    [int]$DatasetSteps = 128,
    [int]$BatchSize = 8,
    [string]$Device = "cuda",
    [ValidateSet("all", "health", "collect", "train", "eval64", "eval128")]
    [string]$Stage = "all",
    [switch]$SkipGitPull
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Body
    )
    Write-Host ""
    Write-Host "=== $Name ==="
    & $Body
}

Set-Location $RepoRoot

if (-not $SkipGitPull) {
    Invoke-Step "git pull" {
        git pull --ff-only
    }
}

if ($Stage -eq "all" -or $Stage -eq "health") {
    Invoke-Step "CARLA health" {
        .\scripts\windows\ensure_carla_ready.ps1
    }
}

if ($Stage -eq "all" -or $Stage -eq "collect") {
    Invoke-Step "collect tiny v1.1 BEV dataset" {
        New-Item -ItemType Directory -Force -Path (Split-Path $Dataset) | Out-Null
        py -3.12 -B -m rcs_ac_v1_1.scripts.collect_bev_expert_dataset `
            --config $V11Config `
            --env-config $EnvConfig `
            --output $Dataset `
            --num-steps $DatasetSteps `
            --host $HostName `
            --port $Port
    }
}

if ($Stage -eq "all" -or $Stage -eq "train") {
    Invoke-Step "train v1.1 BC tiny smoke" {
        py -3.12 -B -m rcs_ac_v1_1.scripts.train_bev_bc `
            --config $V11Config `
            --dataset $Dataset `
            --output-dir $CheckpointDir `
            --epochs 1 `
            --batch-size $BatchSize `
            --device $Device
    }
}

$Checkpoint = Join-Path $CheckpointDir "last.pt"

if ($Stage -eq "all" -or $Stage -eq "eval64") {
    Invoke-Step "v1.1 CARLA 64 smoke" {
        py -3.12 -B -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit `
            --config $V11Config `
            --env-config $EnvConfig `
            --checkpoint $Checkpoint `
            --run-name v1_1_carla_eval_64_smoke `
            --output-dir logs\v1_1_carla_eval_64_smoke `
            --num-steps 64 `
            --host $HostName `
            --port $Port `
            --device $Device `
            --mode carla
    }
}

if ($Stage -eq "all" -or $Stage -eq "eval128") {
    Invoke-Step "v1.1 CARLA 128 smoke" {
        py -3.12 -B -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit `
            --config $V11Config `
            --env-config $EnvConfig `
            --checkpoint $Checkpoint `
            --run-name v1_1_carla_eval_128_smoke `
            --output-dir logs\v1_1_carla_eval_128_smoke `
            --num-steps 128 `
            --host $HostName `
            --port $Port `
            --device $Device `
            --mode carla
    }
}

Write-Host ""
Write-Host "RCS-AC v1.1 smoke runner finished stage=$Stage"
