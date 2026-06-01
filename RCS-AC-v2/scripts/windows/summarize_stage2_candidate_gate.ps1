param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$GatePath = "",
    [string]$CandidateName = "target24",
    [string]$FallbackCandidateName = "target22",
    [string]$CleanConfigPath = "",
    [string]$CheckpointManifestPath = "",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$env:PYTHONPATH"

if ($GatePath -eq "") {
    throw "GatePath is required"
}

Write-Host "STAGE2_CANDIDATE_GATE_SUMMARY_READY=True"
Write-Host "GATE_PATH=$GatePath"
Write-Host "CANDIDATE_NAME=$CandidateName"
Write-Host "FALLBACK_CANDIDATE_NAME=$FallbackCandidateName"

$ArgsList = @(
    "-3.12", "-B", "-m", "rcs_ac_v2.evaluation.candidate_decision_summary",
    "--gate-path", $GatePath,
    "--candidate-name", $CandidateName,
    "--fallback-candidate-name", $FallbackCandidateName
)
if ($CleanConfigPath -ne "") {
    $ArgsList += @("--clean-config-path", $CleanConfigPath)
}
if ($CheckpointManifestPath -ne "") {
    $ArgsList += @("--checkpoint-manifest-path", $CheckpointManifestPath)
}
if ($OutputPath -ne "") {
    $ArgsList += @("--output", $OutputPath)
}
& py @ArgsList
exit $LASTEXITCODE
