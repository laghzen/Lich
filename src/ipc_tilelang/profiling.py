from __future__ import annotations

import json
from pathlib import Path
import statistics
import time
from dataclasses import asdict

from .thermal import ThermalGuard, read_telemetry


def timed(fn, *, warmup: int = 10, rep: int = 50, guard: ThermalGuard | None = None) -> dict[str, object]:
    for _ in range(warmup):
        if guard: guard.wait_until_safe()
        fn()
    samples = []
    telemetry = []
    for _ in range(rep):
        if guard: telemetry.append(asdict(guard.wait_until_safe()))
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": samples[min(len(samples) - 1, int(len(samples) * 0.90))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "telemetry": telemetry,
    }


def save_json(data: object, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
