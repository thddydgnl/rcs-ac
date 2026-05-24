param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$EnsureCarlaScript = "",
    [string]$StartCarlaScript = "",
    [string]$EnvConfigPathOverride = "",
    [string]$SupportCodePath = "",
    [string]$V11CodePath = "",
    [string]$SharedCorePath = "",
    [string]$AdapterFactory = "",
    [string]$RunOutputDir = "",
    [string]$CarlaVersion = "0.9.16",
    [switch]$DryRun,
    [switch]$SkipGitPull,
    [switch]$CheckDependenciesOnly
)

$ErrorActionPreference = "Stop"

function Copy-Stage2RunArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    $ResolvedSource = (Resolve-Path $SourcePath).Path
    $SourceFull = [System.IO.Path]::GetFullPath($ResolvedSource)
    $DestinationFull = [System.IO.Path]::GetFullPath($DestinationPath)
    if ($SourceFull -ieq $DestinationFull) {
        Write-Host "ARTIFACT_COPY_SKIPPED_SAME_PATH=$DestinationFull"
        return
    }
    Copy-Item $SourceFull $DestinationFull -Force
    Write-Host "ARTIFACT_COPIED=$DestinationFull"
}

Set-Location $ProjectRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
if ($SupportCodePath -eq "") {
    $DefaultSupportCodePath = "$ProjectRoot\RCS-AC-v1.1\goal_achieved_round_0002_execution\support_code"
    if (Test-Path $DefaultSupportCodePath) {
        $SupportCodePath = $DefaultSupportCodePath
    }
}
if ($V11CodePath -eq "") {
    $DefaultV11CodePath = "$ProjectRoot\RCS-AC-v1.1\code"
    if (Test-Path $DefaultV11CodePath) {
        $V11CodePath = $DefaultV11CodePath
    }
}
if ($SharedCorePath -eq "") {
    $SharedCorePath = "$WorkspaceRoot\shared_core"
}
if ($EnvConfigPathOverride -ne "") {
    $env:RCS_AC_V2_CARLA_ENV_CONFIG_PATH = $EnvConfigPathOverride
}
$pythonPathParts = @("$ProjectRoot\RCS-AC-v2\code")
if ($SupportCodePath -ne "") {
    $pythonPathParts += $SupportCodePath
}
if ($V11CodePath -ne "") {
    $pythonPathParts += $V11CodePath
}
$pythonPathParts += $WorkspaceRoot
$env:PYTHONPATH = (($pythonPathParts + @($env:PYTHONPATH)) -join ";")

if ($EnsureCarlaScript -eq "") {
    $EnsureCarlaScript = "$ProjectRoot\RCS-AC-v2\scripts\windows\ensure_carla_ready.ps1"
}
if ($StartCarlaScript -eq "") {
    $StartCarlaScript = "$ProjectRoot\RCS-AC-v2\scripts\windows\start_carla_interactive.ps1"
}

if ($DryRun) {
    py -3.12 -B -m rcs_ac_v2.evaluation.evaluate_clean `
        --config $ConfigPath `
        --dry-run
    exit $LASTEXITCODE
}

if (-not $SkipGitPull) {
    git pull --ff-only
}

$DependencyCheckScript = "$ProjectRoot\RCS-AC-v2\scripts\windows\check_stage2_windows_dependencies.ps1"
$DependencyArgs = @{
    ProjectRoot = $ProjectRoot
    ConfigPath = $ConfigPath
    EnsureCarlaScript = $EnsureCarlaScript
    StartCarlaScript = $StartCarlaScript
}
if ($EnvConfigPathOverride -ne "") {
    $DependencyArgs.EnvConfigPathOverride = $EnvConfigPathOverride
}
if ($SupportCodePath -ne "") {
    $DependencyArgs.SupportCodePath = $SupportCodePath
}
if ($V11CodePath -ne "") {
    $DependencyArgs.V11CodePath = $V11CodePath
}
if ($SharedCorePath -ne "") {
    $DependencyArgs.SharedCorePath = $SharedCorePath
}
& powershell -ExecutionPolicy Bypass -File $DependencyCheckScript @DependencyArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
if ($CheckDependenciesOnly) {
    exit 0
}

& powershell -ExecutionPolicy Bypass -File $EnsureCarlaScript `
    -ProjectRoot $ProjectRoot `
    -StartScript $StartCarlaScript

$Commit = (git rev-parse HEAD).Trim()
$Branch = (git rev-parse --abbrev-ref HEAD).Trim()
$WindowsCommand = "powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_clean_eval.ps1 -ConfigPath $ConfigPath"
if ($AdapterFactory -ne "") {
    $WindowsCommand = "$WindowsCommand -AdapterFactory $AdapterFactory"
}
if ($RunOutputDir -ne "") {
    $WindowsCommand = "$WindowsCommand -RunOutputDir $RunOutputDir"
}
if ($SkipGitPull) {
    $WindowsCommand = "$WindowsCommand -SkipGitPull"
}
$SummaryPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_128\summary.json"
$MetricsPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_128\metrics.csv"
$EvalArgs = @(
    "-3.12", "-B", "-m", "rcs_ac_v2.evaluation.evaluate_clean",
    "--config", $ConfigPath,
    "--run-carla",
    "--commit", $Commit,
    "--branch", $Branch,
    "--windows-command", $WindowsCommand,
    "--carla-version", $CarlaVersion
)

if ($AdapterFactory -ne "") {
    $EvalArgs += @("--adapter-factory", $AdapterFactory)
}

Write-Host "STAGE2_CLEAN_EVAL_READY=True"
Write-Host "CONFIG_PATH=$ConfigPath"
Write-Host "SUMMARY_PATH=$SummaryPath"
Write-Host "METRICS_PATH=$MetricsPath"
Remove-Item $SummaryPath, $MetricsPath -ErrorAction SilentlyContinue
& py @EvalArgs
$EvalExitCode = $LASTEXITCODE

if (Test-Path $SummaryPath) {
    py -3.12 -B -m rcs_ac_v2.evaluation.evaluate_clean --validate-summary $SummaryPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    if ($RunOutputDir -ne "") {
        New-Item -ItemType Directory -Force -Path $RunOutputDir | Out-Null
        $ResolvedRunOutputDir = (Resolve-Path $RunOutputDir).Path
        Copy-Stage2RunArtifact `
            -SourcePath $SummaryPath `
            -DestinationPath (Join-Path $ResolvedRunOutputDir "summary.json")
        Copy-Stage2RunArtifact `
            -SourcePath $MetricsPath `
            -DestinationPath (Join-Path $ResolvedRunOutputDir "metrics.csv")
        Copy-Stage2RunArtifact `
            -SourcePath $ConfigPath `
            -DestinationPath (Join-Path $ResolvedRunOutputDir (Split-Path -Leaf $ConfigPath))
        Write-Host "STAGE2_RUN_OUTPUT_DIR=$ResolvedRunOutputDir"
    }
}
else {
    Write-Host "SUMMARY_MISSING=$SummaryPath"
    exit 2
}

exit $EvalExitCode
