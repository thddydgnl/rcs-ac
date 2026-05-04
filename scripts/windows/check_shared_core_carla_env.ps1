param(
    [string]$SharedCoreEnvPath = "C:\Users\thddy\Documents\code\shared_core\env\carla_env.py"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SharedCoreEnvPath)) {
    Write-Error "shared_core carla_env.py not found: $SharedCoreEnvPath"
    exit 1
}

$content = Get-Content -Raw -Encoding UTF8 $SharedCoreEnvPath
$requiredPatterns = @(
    @{ Name = "reuse current requested town"; Pattern = "current_town == self.town" },
    @{ Name = "stage4 action contract"; Pattern = "action_contract_version" },
    @{ Name = "target-speed action wrapper"; Pattern = "target_speed_kmh" },
    @{ Name = "maneuver route hints"; Pattern = "maneuver_command" },
    @{ Name = "advanced observation inputs"; Pattern = "PolicyObservationInputs" },
    @{ Name = "full env step log file"; Pattern = "env_step_log.csv" },
    @{ Name = "full env step log schema"; Pattern = "ENV_STEP_LOG_FIELDS" },
    @{ Name = "route target x logging"; Pattern = "target_x_local" },
    @{ Name = "route metadata logging"; Pattern = "route_waypoint_count" },
    @{ Name = "route turn delta logging"; Pattern = "lookahead_turn_delta_rad" }
)

$missing = @()
foreach ($item in $requiredPatterns) {
    if ($content -notmatch [regex]::Escape($item.Pattern)) {
        $missing += "$($item.Name): $($item.Pattern)"
    }
}

if ($missing.Count -gt 0) {
    Write-Output "SHARED_CORE_CARLA_ENV_CHECK=FAIL"
    foreach ($entry in $missing) {
        Write-Output "MISSING=$entry"
    }
    exit 1
}

Write-Output "SHARED_CORE_CARLA_ENV_CHECK=PASS"
Write-Output "SHARED_CORE_CARLA_ENV_PATH=$SharedCoreEnvPath"
