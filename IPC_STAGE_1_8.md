# iPC TileLang Stage 1.8

## Root-cause fix

Stage 1.7 passed the NVCC host compiler through `compile_flags`, but `_jit_compile_flags()` was evaluated when the eager JIT decorator was constructed. The `smoke` process did not configure Windows MSVC before importing the kernel builder, so `IPC_MSVC_CL` was absent and no `-ccbin` flag was emitted.

Stage 1.8 fixes this at two levels:

1. `scripts/run_cli.py` initializes MSVC before running any command script.
2. `tilelang_kernels._require_tilelang()` lazily initializes the Windows toolchain before the JIT decorator is constructed, so direct Python imports are also safe.

The expected TileLang NVCC command must now contain:

`-ccbin=D:\\Setup\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.39.33519\\bin\\Hostx64\\x64\\cl.exe`

and must not select `clang-cl.exe`.
