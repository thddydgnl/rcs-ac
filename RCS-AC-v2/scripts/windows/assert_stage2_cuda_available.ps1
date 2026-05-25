param(
    [string]$ProjectRoot = "C:\Users\thddy\Documents\code\RCS-AC-first",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$ResolvedDevice = $Device.ToLowerInvariant()
Write-Host "STAGE2_CUDA_GUARD_REQUESTED_DEVICE=$Device"
if ($ResolvedDevice -eq "cpu") {
    Write-Host "STAGE2_CUDA_GUARD_CPU_DIAGNOSTIC_OVERRIDE=True"
    exit 0
}
if ($ResolvedDevice -ne "cuda" -and $ResolvedDevice -ne "auto") {
    throw "Unsupported Stage 2 device for GPU-first workflow: $Device"
}

Set-Location $ProjectRoot
$CudaProbeCode = @'
import sys
import torch
available = torch.cuda.is_available()
print("STAGE2_CUDA_GUARD_TORCH_VERSION=" + str(torch.__version__))
print("STAGE2_CUDA_GUARD_TORCH_CUDA_VERSION=" + str(torch.version.cuda))
print("STAGE2_CUDA_GUARD_AVAILABLE=" + str(available))
if available:
    device_index = torch.cuda.current_device()
    print("STAGE2_CUDA_GUARD_DEVICE_INDEX=" + str(device_index))
    print("STAGE2_CUDA_GUARD_DEVICE_NAME=" + torch.cuda.get_device_name(device_index))
    tensor = torch.ones((1,), device="cuda")
    print("STAGE2_CUDA_GUARD_TENSOR_DEVICE=" + str(tensor.device))
else:
    print("STAGE2_CUDA_GUARD_ERROR=CUDA is required for this Stage 2 run")
    sys.exit(10)
'@

$ProbePath = Join-Path ([System.IO.Path]::GetTempPath()) "rcs_ac_v2_stage2_cuda_guard.py"
$CudaProbeCode | Set-Content -Path $ProbePath -Encoding UTF8
try {
    & py -3.12 -B $ProbePath
    exit $LASTEXITCODE
}
finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $ProbePath
}
