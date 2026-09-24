from __future__ import annotations

import argparse
import runpy
import sys

if sys.platform == "win32":
    try:
        from .windows_toolchain import configure_windows_msvc
        configure_windows_msvc(quiet=False)
    except Exception as exc:
        raise SystemExit(f"[windows-toolchain] {exc}") from exc
from pathlib import Path
import os


def _find_scripts_dir() -> Path:
    explicit = os.environ.get("IPC_TILELANG_SCRIPTS_DIR")
    if explicit and (Path(explicit) / "smoke.py").exists():
        return Path(explicit).resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "scripts",
        here.parents[1] / "scripts",
        here.parent.parent / "scripts",
    ]
    for p in candidates:
        if (p / "smoke.py").exists():
            return p
    for ancestor in here.parents:
        try:
            for child in ancestor.iterdir():
                if child.is_dir() and (child / "smoke.py").exists():
                    return child
        except OSError:
            pass
    raise SystemExit(
        "Could not find scripts directory. Set IPC_TILELANG_SCRIPTS_DIR or use "
        "scripts\\run_cli.py from the source checkout."
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ipc_tilelang.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, script in (
        ("smoke", "smoke.py"),
        ("reference", "validate_reference.py"),
        ("autotune", "autotune.py"),
        ("train-mnist", "train_mnist.py"),
        ("efficiency", "train_paper_efficiency.py"),
        ("thermal-bench", "thermal_bench.py"),
        ("toolchain", "../scripts/toolchain_probe.py"),
    ):
        sub.add_parser(name).set_defaults(_script=script)
    ns, extra = p.parse_known_args()
    scripts_dir = _find_scripts_dir()
    script = scripts_dir / ns._script
    sys.argv = [str(script), *extra]
    runpy.run_path(str(script), run_name="__main__")

if __name__ == "__main__": main()
