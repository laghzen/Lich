from __future__ import annotations

from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECK = [
    ROOT / "src" / "ipc_tilelang" / "weight_update_specialization.py",
    ROOT / "scripts" / "benchmark_small_b_weight_update.py",
    ROOT / "scripts" / "benchmark_weight_update_modes.py",
]

# Build forbidden strings without putting the exact tokens into this source file.
FORBIDDEN = ["torch." + "compile", "tri" + "ton"]
forbidden = {p: [token for token in FORBIDDEN if token in p.read_text(encoding="utf-8").lower()] for p in CHECK}
errors = {str(p): toks for p, toks in forbidden.items() if toks}
if errors:
    raise SystemExit(f"Forbidden compiler dependency found: {errors}")

for p in CHECK:
    ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

print("Stage 2.5 no-compiler static verification: PASS")
