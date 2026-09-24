from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics
import time

import torch

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.tilelang_kernels import KernelConfig, build_prediction_error, build_inference_update, build_weight_update
from ipc_tilelang.thermal import ThermalGuard


def configs() -> list[KernelConfig]:
    out = []
    for bm, bn, bk, threads, stages, sw in itertools.product(
        (64, 128, 256), (64, 128, 256), (16, 32, 64), (64, 128, 256), (1, 2, 3), (False, True)
    ):
        # First-pass architecture filters; keep the search finite and avoid obvious resource traps.
        if threads == 64 and (bm >= 256 or bn >= 256):
            continue
        if threads == 256 and bm == 256 and bn == 256:
            continue
        if stages == 3 and bk == 64 and bm * bk + bn * bk > 32768:
            continue
        out.append(KernelConfig(bm, bn, bk, threads, stages, sw, 8, False))
    return out


def _time(fn, guard: ThermalGuard, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        guard.wait_until_safe()
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rep):
        guard.wait_until_safe()
        ev0.record(); fn(); ev1.record(); ev1.synchronize()
        samples.append(ev0.elapsed_time(ev1))
    return statistics.median(samples)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["prediction", "inference", "weight"], default="prediction")
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--activation", default="relu")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--max-configs", type=int, default=80)
    p.add_argument("--out", default="results/autotune.json")
    p.add_argument("--temperature-max", type=float, default=78.0)
    a = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("Expected an SM86 CUDA device")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    x_up = torch.randn((a.K, a.B), device=dev, dtype=torch.float16)
    w = torch.randn((a.M, a.K), device=dev, dtype=torch.float16) * 0.05
    x_lo = torch.randn((a.M, a.B), device=dev, dtype=torch.float16)
    e = torch.randn_like(x_lo)
    guard = ThermalGuard(max_temp_c=a.temperature_max)
    rows = []
    cs = configs()[: a.max_configs]
    for idx, cfg in enumerate(cs, 1):
        try:
            if a.kind == "prediction":
                k = build_prediction_error(a.M, a.K, a.B, a.activation, "float16", cfg)
                fn = lambda: k(x_up, w, x_lo, e)
            elif a.kind == "inference":
                k = build_inference_update(a.M, a.K, a.B, a.activation, "float16", cfg, 0.5)
                fn = lambda: k(x_lo, e, w, e)
            else:
                k = build_weight_update(a.M, a.K, a.B, a.activation, "float16", cfg, 1e-4, True)
                fn = lambda: k(w, e, x_up)
            ms = _time(fn, guard, a.warmup, a.rep)
            row = {"index": idx, "ms": ms, "config": cfg.__dict__}
            rows.append(row)
            print(f"{a.kind:10s} {idx:03d}/{len(cs):03d} {ms:8.4f} ms {cfg}")
        except Exception as exc:
            print(f"SKIP {idx}: {cfg}: {type(exc).__name__}: {exc}")
    rows.sort(key=lambda r: r["ms"])
    result = {
        "kind": a.kind,
        "shape": {"M": a.M, "K": a.K, "B": a.B},
        "device": torch.cuda.get_device_name(dev),
        "capability": torch.cuda.get_device_capability(dev),
        "best": rows[:10],
        "tested": len(rows),
    }
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["best"][:3], indent=2))

if __name__ == "__main__": main()
