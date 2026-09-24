"""Windows-friendly launcher that discovers the ipc_tilelang source tree."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

# Configure native MSVC before any ipc_tilelang module is imported.
# This is process-local: each `python run_cli.py ...` command is a fresh process.
if sys.platform == "win32":
    from ipc_tilelang.windows_toolchain import configure_windows_msvc
    configure_windows_msvc(quiet=True)
SCRIPTS = Path(__file__).resolve().parent

COMMANDS = {
    "smoke": "smoke.py",
    "reference": "validate_reference.py",
    "autotune": "autotune.py",
    "train-mnist": "train_mnist.py",
    "efficiency": "train_paper_efficiency.py",
    "thermal-bench": "thermal_bench.py",
    "toolchain": "toolchain_probe.py",
}


def main() -> None:
    p = argparse.ArgumentParser(prog="python scripts/run_cli.py")
    p.add_argument("command", choices=sorted(COMMANDS))
    ns, extra = p.parse_known_args()
    script = SCRIPTS / COMMANDS[ns.command]
    sys.argv = [str(script), *extra]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
