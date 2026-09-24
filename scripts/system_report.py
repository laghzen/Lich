from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


def nvidia_smi(args: list[str]) -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not found on PATH"
    p = subprocess.run([exe, *args], capture_output=True, text=True, check=False)
    return (p.stdout or p.stderr).strip()


def main() -> None:
    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi": nvidia_smi([
            "--query-gpu=name,compute_cap,pci.bus_id,memory.total,power.limit,temperature.gpu",
            "--format=csv,noheader",
        ]),
    }
    try:
        import torch
        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            report["gpu"] = torch.cuda.get_device_name(i)
            report["capability"] = torch.cuda.get_device_capability(i)
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["torch_error"] = repr(exc)
    try:
        import tilelang
        report["tilelang"] = tilelang.__version__
    except Exception as exc:
        report["tilelang_error"] = repr(exc)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
