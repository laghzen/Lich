# iPC TileLang Stage 1.4 — Windows toolchain + separated-source layout fix

## What changed

1. Fixed the Windows `vcvars64.bat` invocation. The previous `cmd.exe /s /c` call could reinterpret nested quotes and turn the executable path into a literal quoted token. Stage 1.4 uses `cmd.exe /d /c` with the command string so `call "...\\vcvars64.bat"` is interpreted correctly.
2. `use_msvc.ps1` uses the same corrected invocation.
3. Added `scripts/_bootstrap.py` to locate `src/ipc_tilelang` or a direct `ipc_tilelang` package when `scripts` is stored in a sibling/outer directory. For unusual layouts set `IPC_TILELANG_ROOT` explicitly.
4. Added `scripts/run_cli.py`, the recommended Windows launcher when the source and script folders are separated. It bootstraps the source path before running the selected command.
5. `ipc_tilelang.cli` now supports `IPC_TILELANG_SCRIPTS_DIR` and no longer assumes scripts are exactly two parents above the package.
6. Every script that imports `ipc_tilelang` bootstraps the source path first.

## Recommended command

From the folder containing `scripts`:

```powershell
python .\scripts\run_cli.py toolchain
```

Then:

```powershell
python .\scripts\run_cli.py smoke --M 64 --K 64 --B 128
python .\scripts\run_cli.py smoke --M 10 --K 64 --B 128
```

The toolchain probe first selects an installed 14.39/14.38 MSVC toolset when present, imports the full `vcvars64` environment, forces `NVCC_CCBIN=cl.exe`, and performs a minimal `sm_86` CUBIN compile.

## If auto-discovery still cannot find the package

Use:

```powershell
$env:IPC_TILELANG_ROOT = 'D:\path\to\the\folder\containing\src\ipc_tilelang'
python .\scripts\run_cli.py toolchain
```

The value should be the parent containing `src\\ipc_tilelang`, not the `ipc_tilelang` package directory itself.
