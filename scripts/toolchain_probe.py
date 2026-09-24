from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.windows_toolchain import configure_windows_msvc


def main() -> None:
    if os.name != "nt":
        raise SystemExit("toolchain_probe.py is intended for Windows")

    configure_windows_msvc(quiet=False)
    print("\n=== versions ===")
    print("--- where cl ---")
    subprocess.run(["where", "cl"], check=False)
    print("--- cl version ---")
    subprocess.run(["cl"], check=False)
    print(f"IPC_MSVC_TOOLSET={os.environ.get('IPC_MSVC_TOOLSET', '')}")
    print(f"IPC_MSVC_ROOT={os.environ.get('IPC_MSVC_ROOT', '')}")
    subprocess.run(["where", "nvcc"], check=False)
    subprocess.run(["nvcc", "--version"], check=False)

    print("\n=== minimal NVCC sm_86 compile ===")
    src = "extern \"C\" __global__ void probe(float* x) { x[threadIdx.x] += 1.0f; }\n"
    with tempfile.TemporaryDirectory() as td:
        cu = Path(td) / "probe.cu"
        cubin = Path(td) / "probe.cubin"
        cu.write_text(src, encoding="utf-8")
        cmd = ["nvcc", "-ccbin=cl.exe", "--cubin", "-O3", "-arch=sm_86", str(cu), "-o", str(cubin)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"OK: {cubin}")


if __name__ == "__main__":
    main()
