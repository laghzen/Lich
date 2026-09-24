$ErrorActionPreference = "Stop"

Write-Host "=== iPC x TileLang Windows setup ==="

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher ('py') is not installed. Install Python 3.11/3.12 and re-run."
}

$python = "py"
$venv = Join-Path $PSScriptRoot "..\.venv"

if (-not (Test-Path $venv)) {
    & $python -3.12 -m venv $venv
}

$venvPython = Join-Path $venv "Scripts\python.exe"

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "..\requirements.txt")
& $venvPython -m pip install -e (Join-Path $PSScriptRoot "..")

Write-Host ""
Write-Host "Installed. Verifying TileLang/PyTorch/CUDA..."
& $venvPython -c "import torch, tilelang; print('torch:', torch.__version__); print('tilelang:', tilelang.__version__); print('cuda:', torch.cuda.is_available()); print('capability:', torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"

Write-Host ""
Write-Host "If CUDA reports False, fix the NVIDIA driver/PyTorch installation before running kernels."


Write-Host ""
Write-Host "=== Host compiler setup ==="
Write-Host "TileLang/NVCC must use MSVC cl.exe, not clang-cl.exe."
Write-Host "For CUDA 12.9, NVIDIA documents MSVC 193x; the recommended side-by-side VS 2022 toolset is v14.39 (17.9)."
Write-Host "In Visual Studio Installer add: MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.39-17.9)."
Write-Host "After installation, run: . .\\scripts\\use_msvc.ps1"
