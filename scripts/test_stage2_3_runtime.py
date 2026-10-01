from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tests" / "test_stage2_3_runtime_plan.py"), run_name="__main__")
