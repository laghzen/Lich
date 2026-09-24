# iPC TileLang Stage 1.7

## Fix
TileLang 0.1.13 may select `clang-cl.exe` for NVCC even when the process environment contains `NVCC_CCBIN=cl.exe`. This stage makes the host compiler authoritative by passing `-ccbin=<selected cl.exe>` through TileLang JIT `compile_flags`.

CUDA 12.9 documents MSVC 193x as the supported native Windows x86_64 host compiler family. The project therefore selects the installed 14.39 toolset and passes its absolute `cl.exe` path to NVCC.
