# iPC TileLang Stage 1.5

This stage fixes the Windows MSVC environment bootstrap for Python + `cmd.exe`.

The previous implementation passed a quoted `vcvars64.bat` path as an argv item to
`cmd.exe /c`. On Windows, Python's command-line quoting can preserve the embedded
quotes as literal characters, producing errors of the form `\"D:\\...vcvars64.bat\" is not recognized`.

Stage 1.5 invokes a temporary `.cmd` file instead, captures the environment, and
puts the selected MSVC `cl.exe` directory first in `PATH` so that `clang-cl.exe`
cannot hijack NVCC through inherited compiler variables.

It first tries the selected 14.39/14.38 toolset and falls back to the default VS
x64 environment if that exact `-vcvars_ver` is not accepted by the installation.
The selected `cl.exe` remains explicit through `NVCC_CCBIN=cl.exe`.
