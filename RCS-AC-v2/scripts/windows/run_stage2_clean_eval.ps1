param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$ConfigPath = "RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml",
    [string]$EnsureCarlaScript = "C:\Users\thddy\Documents\code\RCS-AC-first\shared\scripts\windows\ensure_carla_ready.ps1",
    [string]$StartCarlaScript = "C:\Users\thddy\Documents\code\RCS-AC-first\shared\scripts\windows\start_carla_interactive.ps1",
    [string]$AdapterFactory = "",
    [string]$CarlaVersion = "0.9.16",
    [switch]$DryRun,
    [switch]$SkipGitPull
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$SupportCodePath = "$ProjectRoot\RCS-AC-v1.1\goal_achieved_round_0002_execution\support_code"
$V11CodePath = "$ProjectRoot\RCS-AC-v1.1\code"
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$SupportCodePath;$V11CodePath;$env:PYTHONPATH"

if ($DryRun) {
    py -3.12 -B -m rcs_ac_v2.evaluation.evaluate_clean `
        --config $ConfigPath `
        --dry-run
    exit $LASTEXITCODE
}

if (-not $SkipGitPull) {
    git pull --ff-only
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
& py @EvalArgs
$EvalExitCode = $LASTEXITCODE

if (Test-Path $SummaryPath) {
    py -3.12 -B -m rcs_ac_v2.evaluation.evaluate_clean --validate-summary $SummaryPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
else {
    Write-Host "SUMMARY_MISSING=$SummaryPath"
    exit 2
}

exit $EvalExitCode
