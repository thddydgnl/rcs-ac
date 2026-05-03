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
    @{ Name = "advanced observation inputs"; Pattern = "PolicyObservationInputs" }
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
