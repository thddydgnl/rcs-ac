param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$PrimaryReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_target22_failure_window",
    [string]$PrimaryName = "target22",
    [string]$SecondaryReplayDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_6e74d9e_long512_t25_steer2_probe",
    [string]$SecondaryName = "steer2",
    [string]$TertiaryReplayDir = "",
    [string]$TertiaryName = "late",
    [string]$QuaternaryReplayDir = "",
    [string]$QuaternaryName = "late2",
    [string]$OutputDir = "RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_mixed_target22_steer2",
    [string]$ConfigPath = "RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_steer2_seq010_aux1_balpow05_iw5\clean_512_stage2_privileged_7f87656_steer2_seq010_aux1_balpow05_iw5_with_checkpoints.yaml",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\RCS-AC-v2\code;$env:PYTHONPATH"

$ReplayComponents = @(
    @{ Name = $PrimaryName; Dir = $PrimaryReplayDir },
    @{ Name = $SecondaryName; Dir = $SecondaryReplayDir }
)
if ($TertiaryReplayDir -ne "") {
    $ReplayComponents += @{ Name = $TertiaryName; Dir = $TertiaryReplayDir }
}
if ($QuaternaryReplayDir -ne "") {
    $ReplayComponents += @{ Name = $QuaternaryName; Dir = $QuaternaryReplayDir }
}

foreach ($ReplayComponent in $ReplayComponents) {
    $ReplayDir = [string]$ReplayComponent.Dir
    if (-not (Test-Path $ReplayDir)) {
        throw "Replay directory missing: $ReplayDir"
    }
    $ReplayFiles = @(Get-ChildItem $ReplayDir -Filter "*.npz" | Sort-Object Name)
    if ($ReplayFiles.Count -eq 0) {
        throw "Replay directory has no .npz files: $ReplayDir"
    }
}
if (-not (Test-Path $ConfigPath)) {
    throw "ConfigPath missing: $ConfigPath"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ExistingFiles = @(Get-ChildItem $OutputDir -Filter "*.npz" -ErrorAction SilentlyContinue)
if ($ExistingFiles.Count -gt 0 -and -not $Force) {
    throw "OutputDir already has .npz files; rerun with -Force to rebuild: $OutputDir"
}
if ($Force) {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $OutputDir "*.npz")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $OutputDir "stage2_privileged_replay_manifest.json")
}

function Copy-ReplayEpisodes {
    param(
        [string]$SourceDir,
        [string]$Prefix
    )
    $Index = 0
    foreach ($Source in Get-ChildItem $SourceDir -Filter "*.npz" | Sort-Object Name) {
        $Target = Join-Path $OutputDir ("episode_{0}_{1:D4}.npz" -f $Prefix, $Index)
        Copy-Item $Source.FullName -Destination $Target
        $Index += 1
    }
}

foreach ($ReplayComponent in $ReplayComponents) {
    Copy-ReplayEpisodes -SourceDir ([string]$ReplayComponent.Dir) -Prefix ([string]$ReplayComponent.Name)
}

$PythonSource = @'
from pathlib import Path
import subprocess

from rcs_ac_v2.training.privileged_replay import (
    validate_replay_manifest,
    write_privileged_replay_manifest,
)

root = Path(r"{PROJECT_ROOT}")
output_dir = root / r"{OUTPUT_DIR}"
config_path = root / r"{CONFIG_PATH}"
manifest_path = output_dir / "stage2_privileged_replay_manifest.json"
episode_paths = sorted(output_dir.glob("*.npz"))
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

write_privileged_replay_manifest(
    manifest_path,
    config_path=config_path,
    episode_paths=episode_paths,
    extra={
        "git_commit": git_commit,
        "run_kind": "mixed_replay_manifest_from_existing_windows_replays",
        "components": [
            {"name": r"{PRIMARY_NAME}", "source_dir": r"{PRIMARY_REPLAY_DIR}"},
            {"name": r"{SECONDARY_NAME}", "source_dir": r"{SECONDARY_REPLAY_DIR}"}{TERTIARY_COMPONENT}{QUATERNARY_COMPONENT}
        ],
    },
)
payload = validate_replay_manifest(manifest_path, replay_paths=[output_dir])
print("MIXED_REPLAY_MANIFEST=" + str(manifest_path))
print("MIXED_REPLAY_EPISODES=" + str(payload["episodes"]))
print("MIXED_REPLAY_TOTAL_STEPS=" + str(payload["total_steps"]))
print("MIXED_REPLAY_GIT_COMMIT=" + str(payload["git_commit"]))
print("MIXED_REPLAY_FILES=" + str(len(payload["files"])))
'@
$PythonSource = $PythonSource.Replace("{PROJECT_ROOT}", $ProjectRoot)
$PythonSource = $PythonSource.Replace("{OUTPUT_DIR}", $OutputDir)
$PythonSource = $PythonSource.Replace("{CONFIG_PATH}", $ConfigPath)
$PythonSource = $PythonSource.Replace("{PRIMARY_NAME}", $PrimaryName)
$PythonSource = $PythonSource.Replace("{PRIMARY_REPLAY_DIR}", $PrimaryReplayDir)
$PythonSource = $PythonSource.Replace("{SECONDARY_NAME}", $SecondaryName)
$PythonSource = $PythonSource.Replace("{SECONDARY_REPLAY_DIR}", $SecondaryReplayDir)
if ($TertiaryReplayDir -ne "") {
    $PythonSource = $PythonSource.Replace("{TERTIARY_COMPONENT}", ("," + [Environment]::NewLine + "            {""name"": r""" + $TertiaryName + """, ""source_dir"": r""" + $TertiaryReplayDir + """}"))
}
else {
    $PythonSource = $PythonSource.Replace("{TERTIARY_COMPONENT}", "")
}
if ($QuaternaryReplayDir -ne "") {
    $PythonSource = $PythonSource.Replace("{QUATERNARY_COMPONENT}", ("," + [Environment]::NewLine + "            {""name"": r""" + $QuaternaryName + """, ""source_dir"": r""" + $QuaternaryReplayDir + """}"))
}
else {
    $PythonSource = $PythonSource.Replace("{QUATERNARY_COMPONENT}", "")
}

$TempPython = New-TemporaryFile
try {
    Set-Content -Path $TempPython -Value $PythonSource -Encoding UTF8
    & py -3.12 -B $TempPython
    exit $LASTEXITCODE
}
finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $TempPython
}
