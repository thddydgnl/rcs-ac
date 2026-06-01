param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$EnsureCarlaScript = "",
    [string]$StartCarlaScript = "",
    [string]$EnvConfigPathOverride = "",
    [string]$SupportCodePath = "",
    [string]$V11CodePath = "",
    [string]$SharedCorePath = "",
    [switch]$SkipPythonImportProbe,
    [switch]$RequireCuda
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

if ($EnsureCarlaScript -eq "") {
    $EnsureCarlaScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\ensure_carla_ready.ps1"
}
if ($StartCarlaScript -eq "") {
    $StartCarlaScript = Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\start_carla_interactive.ps1"
}
if ($SupportCodePath -eq "") {
    $DefaultSupportCodePath = Join-Path $ProjectRoot "RCS-AC-v1.1\goal_achieved_round_0002_execution\support_code"
    if (Test-Path $DefaultSupportCodePath) {
        $SupportCodePath = $DefaultSupportCodePath
    }
}
if ($V11CodePath -eq "") {
    $DefaultV11CodePath = Join-Path $ProjectRoot "RCS-AC-v1.1\code"
    if (Test-Path $DefaultV11CodePath) {
        $V11CodePath = $DefaultV11CodePath
    }
}
if ($SharedCorePath -eq "") {
    $SharedCorePath = Join-Path (Split-Path -Parent $ProjectRoot) "shared_core"
}

$V2CodePath = Join-Path $ProjectRoot "RCS-AC-v2\code"
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$envConfigMatch = Select-String -Path $ConfigPath -Pattern '^\s*env_config_path:\s*(.+)$' | Select-Object -First 1
if ($null -eq $envConfigMatch) {
    Write-Host "STAGE2_DEPENDENCY_ENV_CONFIG=MISSING:carla.env_config_path"
    Write-Host "STAGE2_WINDOWS_DEPENDENCIES_OK=False"
    exit 2
}
$EnvConfigPath = $envConfigMatch.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
if ($EnvConfigPathOverride -ne "") {
    $EnvConfigPath = $EnvConfigPathOverride
}
if (-not [System.IO.Path]::IsPathRooted($EnvConfigPath)) {
    $EnvConfigPath = Join-Path $ProjectRoot $EnvConfigPath
}
$requiredPaths = @(
    @{ Name = "PROJECT_ROOT"; Path = $ProjectRoot },
    @{ Name = "CONFIG"; Path = $ConfigPath },
    @{ Name = "ENV_CONFIG"; Path = $EnvConfigPath },
    @{ Name = "V2_CODE"; Path = $V2CodePath },
    @{ Name = "STAGE2_RUNNER"; Path = (Join-Path $ProjectRoot "RCS-AC-v2\scripts\windows\run_stage2_clean_eval.ps1") },
    @{ Name = "CARLA_READY_SCRIPT"; Path = $EnsureCarlaScript },
    @{ Name = "CARLA_START_SCRIPT"; Path = $StartCarlaScript },
    @{ Name = "WORKSPACE_ROOT"; Path = $WorkspaceRoot },
    @{ Name = "SHARED_CORE"; Path = $SharedCorePath }
)
if ($SupportCodePath -ne "") {
    $requiredPaths += @{ Name = "V11_SUPPORT_CODE"; Path = $SupportCodePath }
}
else {
    Write-Host "STAGE2_DEPENDENCY_V11_SUPPORT_CODE=SKIPPED:not configured"
}
if ($V11CodePath -ne "") {
    $requiredPaths += @{ Name = "V11_CODE"; Path = $V11CodePath }
}
else {
    Write-Host "STAGE2_DEPENDENCY_V11_CODE=SKIPPED:not configured"
}

$missing = @()
foreach ($entry in $requiredPaths) {
    if (Test-Path $entry.Path) {
        Write-Host "STAGE2_DEPENDENCY_$($entry.Name)=OK:$($entry.Path)"
    }
    else {
        Write-Host "STAGE2_DEPENDENCY_$($entry.Name)=MISSING:$($entry.Path)"
        $missing += $entry
    }
}

if ($missing.Count -gt 0) {
    Write-Host "STAGE2_WINDOWS_DEPENDENCIES_OK=False"
    exit 2
}

$pythonPathParts = @($V2CodePath)
if ($SupportCodePath -ne "") {
    $pythonPathParts += $SupportCodePath
}
if ($V11CodePath -ne "") {
    $pythonPathParts += $V11CodePath
}
$pythonPathParts += $WorkspaceRoot
$env:PYTHONPATH = (($pythonPathParts + @($env:PYTHONPATH)) -join ";")
Write-Host "STAGE2_PYTHONPATH=$env:PYTHONPATH"

if (-not $SkipPythonImportProbe) {
    $probeCode = @"
import importlib
modules = [
    'rcs_ac_v2.evaluation.evaluate_clean',
    'rcs_ac_v2.evaluation.carla_adapter',
    'shared_core.env.carla_env',
]
for module in modules:
    importlib.import_module(module)
    print('PY_IMPORT_OK=' + module)
"@

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & py -3.12 -B -c $probeCode 2>&1
    $ErrorActionPreference = $previousErrorActionPreference
    Write-Host $output
    if ($LASTEXITCODE -ne 0) {
        Write-Host "STAGE2_WINDOWS_DEPENDENCIES_OK=False"
        exit 3
    }
}

if ($RequireCuda) {
    $cudaProbeCode = @"
import torch
available = torch.cuda.is_available()
print('STAGE2_TORCH_CUDA_AVAILABLE=' + str(available))
print('STAGE2_TORCH_CUDA_DEVICE=' + (torch.cuda.get_device_name(0) if available else 'NO_CUDA'))
if not available:
    raise SystemExit(4)
"@

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $cudaOutput = & py -3.12 -B -c $cudaProbeCode 2>&1
    $ErrorActionPreference = $previousErrorActionPreference
    Write-Host $cudaOutput
    if ($LASTEXITCODE -ne 0) {
        Write-Host "STAGE2_WINDOWS_CUDA_OK=False"
        Write-Host "STAGE2_WINDOWS_DEPENDENCIES_OK=False"
        exit 4
    }
    Write-Host "STAGE2_WINDOWS_CUDA_OK=True"
}

Write-Host "STAGE2_WINDOWS_DEPENDENCIES_OK=True"
exit 0
