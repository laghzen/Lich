$ErrorActionPreference = "Stop"

$vswhereCandidates = @(
    "$env:ProgramFiles(x86)\Microsoft Visual Studio\Installer\vswhere.exe",
    "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
)
$vswhere = $vswhereCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vswhere) { throw "vswhere.exe was not found." }

$vs = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $vs -or -not (Test-Path $vs)) { throw "Visual Studio installation was not found." }

$vcRoot = Join-Path $vs "VC\Tools\MSVC"
$toolsets = Get-ChildItem $vcRoot -Directory | Where-Object {
    Test-Path (Join-Path $_.FullName "bin\Hostx64\x64\cl.exe")
} | Sort-Object Name -Descending

$tool = $toolsets | Where-Object { $_.Name -like "14.39.*" -or $_.Name -like "14.38.*" } | Select-Object -First 1
if (-not $tool) { $tool = $toolsets | Select-Object -First 1 }
if (-not $tool) { throw "No MSVC x64 toolset found." }

$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$baseVersion = ($tool.Name -split '\.')[0..1] -join '.'
$cmd = "call `"$vcvars`" -vcvars_ver=$baseVersion >nul && set"
$envLines = cmd.exe /d /c $cmd
foreach ($line in $envLines) {
    if ($line -match '^(?<name>[^=]+)=(?<value>.*)$') {
        Set-Item -Path ("Env:" + $Matches.name) -Value $Matches.value
    }
}

$env:CC = "cl.exe"
$env:CXX = "cl.exe"
$env:NVCC_CCBIN = "cl.exe"
$env:IPC_MSVC_TOOLSET = $tool.Name
$env:IPC_MSVC_ROOT = $tool.FullName

Write-Host "[windows-toolchain] MSVC: $($tool.Name)"
Write-Host "[windows-toolchain] cl.exe: $($tool.FullName)\bin\Hostx64\x64\cl.exe"
Write-Host "[windows-toolchain] CC/CXX/NVCC_CCBIN = cl.exe"
if ($tool.Name -notlike "14.39.*" -and $tool.Name -notlike "14.38.*") {
    Write-Warning "No MSVC 14.39/14.38 toolset found. CUDA 12.9 documents MSVC 193x; current 14.44 is outside that documented family."
}
