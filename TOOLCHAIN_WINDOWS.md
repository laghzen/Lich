# Windows CUDA/NVCC toolchain

The failure seen on the RTX 3060 came from NVCC invoking `clang-cl.exe`, not from the iPC kernels.

The generated command contained:

```text
-ccbin=D:\Setup\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clang-cl.exe
```

For CUDA 12.9, NVIDIA's Windows documentation lists MSVC 193x / Visual Studio 2022 17.x as the supported host compiler family. Your installed MSVC 14.44 is the 194x family. The project therefore prefers a side-by-side MSVC 14.39 or 14.38 toolset when available.

## Recommended setup

In Visual Studio Installer add:

`MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.39-17.9)`

Then from the project root:

```powershell
. .\scripts\use_msvc.ps1
python .\scripts\toolchain_probe.py
```

The probe must report `cl.exe` as the NVCC host compiler and finish the minimal `sm_86` cubin compilation.

The normal project CLI also initializes the MSVC environment automatically before TileLang JIT compilation.

## Cache cleanup after changing toolchain

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.tilelang\cache" -ErrorAction SilentlyContinue
```

Then retry the smoke test.
