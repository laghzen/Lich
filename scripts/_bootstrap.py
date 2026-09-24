"""Find the local ipc_tilelang source tree when scripts are stored elsewhere.

Supported layouts include:
  project/src/ipc_tilelang + project/scripts
  project/ipc_tilelang + project/scripts
  scripts/ next to a separate sibling project containing src/ipc_tilelang

For unusual layouts set IPC_TILELANG_ROOT to the directory containing either
src/ipc_tilelang or ipc_tilelang.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_roots(here: Path):
    seen: set[Path] = set()

    env_root = os.environ.get("IPC_TILELANG_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p not in seen:
            seen.add(p)
            yield p

    # The current script directory and its ancestors.
    for p in (here, *here.parents):
        if p not in seen:
            seen.add(p)
            yield p

        # Also inspect siblings. This covers cases where `scripts/` lives in
        # one folder and the source tree is in a sibling folder.
        try:
            for child in p.iterdir():
                if not child.is_dir() or child in seen:
                    continue
                name = child.name.lower()
                if name in {".git", ".venv", "venv", "node_modules", "__pycache__"}:
                    continue
                seen.add(child)
                yield child
        except OSError:
            continue


def bootstrap() -> Path:
    here = Path(__file__).resolve().parent
    for root in _candidate_roots(here):
        src_pkg = root / "src" / "ipc_tilelang" / "__init__.py"
        direct_pkg = root / "ipc_tilelang" / "__init__.py"
        if src_pkg.exists():
            src = str(root / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            os.environ.setdefault("IPC_TILELANG_ROOT", str(root))
            return root
        if direct_pkg.exists():
            value = str(root)
            if value not in sys.path:
                sys.path.insert(0, value)
            os.environ.setdefault("IPC_TILELANG_ROOT", str(root))
            return root

    raise RuntimeError(
        "Could not locate ipc_tilelang source tree. Set IPC_TILELANG_ROOT to the "
        "folder containing either src\\ipc_tilelang or ipc_tilelang."
    )
