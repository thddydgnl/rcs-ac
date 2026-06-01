param(
    [int[]]$SpawnIndices = @(22, 160, 161, 162, 176, 192, 193, 232, 234),
    [int]$Steps = 512,
    [int]$Seed = 7,
    [string]$RunPrefix = "ppo_matched_r7j_imitation_teacher_primary",
    [string]$Checkpoint = "PPO\results\ppo_matched_r7j_seed7_imitation_teacher_train\last.pt",
    [string]$Device = "cuda",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 2000,
    [switch]$EnsureReadyBetweenRuns,
    [int]$InterRunSleepSeconds = 5
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Repo

$env:PYTHONPATH = "$Repo\RCS-AC-v1.1\code;$Repo;$env:PYTHONPATH"

$RuntimeConfig = "RCS-AC-v1.1\configs\v1_1\full_bev_smoke_shared_agile_midbalanced_route_geometry_cc4_r7j_target_y_lowpass_light.yaml"
$BaseEnvConfig = "RCS-AC-v1.1\portable_snapshots\r7j_generalized_20260507\configs\v1_1\bc_clean_v1_1_2048_eval.yaml"
$GeneratedConfigDir = "PPO\configs\generated"
$LogRoot = "PPO\results\multistart\${RunPrefix}_${Steps}_seed${Seed}"
$AggregateDir = Join-Path $LogRoot "_aggregate"

New-Item -ItemType Directory -Force $GeneratedConfigDir | Out-Null
New-Item -ItemType Directory -Force $LogRoot | Out-Null

$labels = @{
    1 = "east_straight_ref"
    9 = "west_straight_offset"
    20 = "spawn_20"
    21 = "spawn_21"
    22 = "north_sharp_lane_switch"
    151 = "north_straight_offset"
    160 = "spawn_160"
    161 = "spawn_161"
    162 = "east_sharp_transition"
    176 = "west_sharp_transition"
    192 = "east_medium_connector"
    193 = "spawn_193"
    232 = "south_sharp_transition"
    234 = "spawn_234"
    365 = "south_gentle_diagonal"
    368 = "north_gentle_diagonal"
}

$results = @()

foreach ($spawnIndex in $SpawnIndices) {
    if ($EnsureReadyBetweenRuns) {
        Write-Host "=== CARLA_READY_BEFORE spawn=$spawnIndex ==="
        & ".\scripts\windows\ensure_carla_ready.ps1"
        if ($InterRunSleepSeconds -gt 0) {
            Start-Sleep -Seconds $InterRunSleepSeconds
        }
    }

    $label = $labels[$spawnIndex]
    if (-not $label) { $label = "spawn_$spawnIndex" }
    $runName = "${RunPrefix}${Steps}_${label}_idx${spawnIndex}_seed${Seed}"
    $envConfig = Join-Path $GeneratedConfigDir "ppo_matched_r7j_multistart_${spawnIndex}_${Steps}_seed${Seed}_eval.yaml"
    $outputDir = Join-Path $LogRoot $runName

    $text = Get-Content $BaseEnvConfig -Raw
    $text = $text -replace "(?m)^seed:\s*\d+", "seed: $Seed"
    $text = $text -replace "(?m)^run_name:\s*.*$", "run_name: $runName"
    $text = $text -replace "(?m)^training_stage:\s*.*$", "training_stage: ppo_matched_r7j_multistart_eval"
    $text = $text -replace "(?m)^max_episode_steps:\s*\d+", "max_episode_steps: $Steps"
    if ($text -match "(?m)^env:\s*$") {
        $text = $text -replace "(?m)^env:\s*$", "env:`n  spawn_point_index: $spawnIndex"
    } else {
        $text += "`nenv:`n  spawn_point_index: $spawnIndex`n"
    }
    Set-Content -Path $envConfig -Value $text -Encoding UTF8

    Write-Host "=== PPO_MATCHED_R7J_RUN spawn=$spawnIndex label=$label steps=$Steps run=$runName ==="
    try {
        py -3.12 -B -m PPO.scripts.evaluate_matched_r7j `
            --mode carla `
            --config $RuntimeConfig `
            --env-config $envConfig `
            --checkpoint $Checkpoint `
            --run-name $runName `
            --output-dir $outputDir `
            --num-steps $Steps `
            --host $HostAddress `
            --port $Port `
            --device $Device
        $exitCode = $LASTEXITCODE
    } catch {
        $exitCode = 1
        Write-Host "RUN_EXCEPTION spawn=$spawnIndex $($_.Exception.Message)"
    }

    $summaryPath = Join-Path $outputDir "summary.json"
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $row = [PSCustomObject]@{
            spawn_index = $spawnIndex
            label = $label
            seed = $Seed
            steps_requested = $Steps
            exit_code = $exitCode
            steps_completed = $summary.num_steps_completed
            done_reason = $summary.first_done_reason
            done_step = $summary.first_done_step
            collision_count = $summary.collision_count
            offroad_count = $summary.offroad_count
            stuck_count = $summary.stuck_count
            max_route_progress = $summary.max_route_progress
            final_route_progress = $summary.final_route_progress
            mean_abs_steer = $summary.mean_abs_steer
            waypoint_valid_rate = $summary.waypoint_valid_rate
            path_valid_rate = $summary.path_valid_rate
            used_synthetic_fallback_rate = $summary.used_synthetic_fallback_rate
            used_route_fallback_rate = $summary.used_route_fallback_rate
            output_dir = $outputDir
        }
    } else {
        $row = [PSCustomObject]@{
            spawn_index = $spawnIndex
            label = $label
            seed = $Seed
            steps_requested = $Steps
            exit_code = $exitCode
            steps_completed = $null
            done_reason = "missing_summary"
            done_step = $null
            output_dir = $outputDir
        }
    }
    $results += $row
    $row | ConvertTo-Json -Compress
}

New-Item -ItemType Directory -Force $AggregateDir | Out-Null
$results | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $AggregateDir "summary_by_spawn.csv")
$results | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $AggregateDir "summary_by_spawn.json")

py -3.12 -B -m rcs_ac_v1_1.scripts.aggregate_multistart_results `
    --log-root $LogRoot `
    --output-dir $AggregateDir `
    --steps $Steps `
    --primary-starts "22,160,161,162,176,192,193,232,234"

Write-Host "=== PPO_MATCHED_R7J_MULTISTART_DONE $AggregateDir ==="
$results | Format-Table -AutoSize
