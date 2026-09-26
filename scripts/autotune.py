from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re
import statistics

import torch

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.tilelang_kernels import (
    KernelConfig,
    build_inference_update,
    build_prediction_error,
    build_weight_update,
)
from ipc_tilelang.thermal import ThermalGuard


BASELINE = KernelConfig(64, 128, 16, 128, 2, True, 8, False)


def configs() -> list[KernelConfig]:
    """SM86-first search space with explicit small-tile candidates.

    Stage 1.11 intentionally favors CTA-level parallelism for the small
    fully-connected iPC shapes used by the MNIST/64-wide profile. The old
    64/128/256-only prefix search is retained only as the baseline entry,
    not as the search-space definition.
    """
    out: list[KernelConfig] = []
    for bm, bn, bk, threads, stages, sw in itertools.product(
        (16, 32, 64, 128, 256),
        (32, 64, 128, 256),
        (16, 32, 64),
        (64, 128, 256),
        (1, 2, 3),
        (False, True),
    ):
        # Conservative compile/resource filters. The actual GPU result is
        # still the authority; these only remove obviously pathological cases.
        pred_smem = (2 * bk * bn + bm * bk) * 2 * stages
        inf_smem = (bk * bn + bk * bm) * 2 * stages
        weight_smem = (bm * bn + 2 * bk * bn) * 2 * stages
        if max(pred_smem, inf_smem, weight_smem) > 64 * 1024:
            continue
        if threads == 64 and (bm >= 256 or bn >= 256):
            continue
        if threads == 256 and bm * bn > 8192:
            continue
        if stages == 3 and bk == 64 and bm * bk + bn * bk > 49152:
            continue
        if bm == 16 and threads == 256:
            continue
        out.append(KernelConfig(bm, bn, bk, threads, stages, sw, 8, False))

    # Keep the known Stage-1.10 baseline first for direct A/B comparison.
    out = [BASELINE] + [c for c in out if c != BASELINE]
    return out


def select_configs(all_configs: list[KernelConfig], limit: int) -> list[KernelConfig]:
    if limit <= 0 or limit >= len(all_configs):
        return list(all_configs)
    if limit == 1:
        return [all_configs[0]]
    selected: list[KernelConfig] = [all_configs[0]]
    remaining = limit - 1
    for i in range(1, remaining + 1):
        pos = round(i * (len(all_configs) - 1) / remaining)
        candidate = all_configs[pos]
        if candidate not in selected:
            selected.append(candidate)
    # Back-fill in case duplicate positions occurred after rounding.
    if len(selected) < limit:
        for cfg in all_configs:
            if cfg not in selected:
                selected.append(cfg)
                if len(selected) == limit:
                    break
    return selected


def _time(fn, guard: ThermalGuard, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        guard.wait_until_safe()
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rep):
        guard.wait_until_safe()
        ev0.record()
        fn()
        ev1.record()
        ev1.synchronize()
        samples.append(ev0.elapsed_time(ev1))
    return statistics.median(samples)


def _inspect_kernel(kernel) -> dict[str, object]:
    """Best-effort source inspection.

    TileLang 0.1.13 does not expose a stable cross-version register-count API.
    When kernel source is available, parse PTX-style register declarations and
    obvious local-memory instructions. Missing fields remain null rather than
    being guessed.
    """
    info: dict[str, object] = {
        "registers_estimate": None,
        "spill_suspected": None,
        "source_available": False,
    }
    getter = getattr(kernel, "get_kernel_source", None)
    if getter is None:
        return info
    try:
        source = str(getter())
    except Exception:
        return info
    info["source_available"] = True
    # PTX declarations look like: .reg .b32 %r<123>; / .reg .b64 %rd<17>;
    total = 0
    found_reg_decl = False
    for bits_s, count_s in re.findall(r"\.reg\s+\.b(16|32|64|128)\s+%[A-Za-z_]+<([0-9]+)>", source):
        found_reg_decl = True
        bits = int(bits_s)
        count = int(count_s)
        total += count * max(1, math.ceil(bits / 32))
    if found_reg_decl:
        info["registers_estimate"] = total

    # This is deliberately named 'suspected': source-level local storage can
    # also arise from compiler-generated temporaries and is not proof of a spill.
    info["spill_suspected"] = bool(
        re.search(r"\b(?:ld|st)\.local\b", source) or ".local" in source
    )
    return info


def _make_case(kind: str, M: int, K: int, B: int, device: torch.device, activation: str, cfg: KernelConfig):
    if kind == "prediction":
        x_up = torch.randn((K, B), device=device, dtype=torch.float16)
        w = torch.randn((M, K), device=device, dtype=torch.float16) * 0.05
        x_lo = torch.randn((M, B), device=device, dtype=torch.float16)
        e = torch.empty_like(x_lo)
        kernel = build_prediction_error(M, K, B, activation, "float16", cfg)
        return kernel, lambda: kernel(x_up, w, x_lo, e)

    if kind == "inference":
        x = torch.randn((M, B), device=device, dtype=torch.float16)
        e = torch.randn((M, B), device=device, dtype=torch.float16)
        # IMPORTANT: Stage 1.11 keeps the real kernel ABI W_lower[K,M].
        w_lower = torch.randn((K, M), device=device, dtype=torch.float16) * 0.05
        e_lower = torch.randn((K, B), device=device, dtype=torch.float16)
        kernel = build_inference_update(M, K, B, activation, "float16", cfg, 0.5, False)
        return kernel, lambda: kernel(x, e, w_lower, e_lower)

    if kind == "weight":
        w = torch.randn((M, K), device=device, dtype=torch.float16) * 0.05
        e = torch.randn((M, B), device=device, dtype=torch.float16)
        x_up = torch.randn((K, B), device=device, dtype=torch.float16)
        kernel = build_weight_update(M, K, B, activation, "float16", cfg, 1e-4, True)
        return kernel, lambda: kernel(w, e, x_up)

    raise ValueError(kind)


def profile_shapes(profile: str, B: int, hidden: int = 64) -> dict[str, list[tuple[int, int, int]]]:
    if profile == "mnist64":
        return {
            "prediction": [(10, hidden, B), (hidden, hidden, B), (hidden, 784, B)],
            "inference": [(hidden, 10, B), (hidden, hidden, B)],
            "weight": [(10, hidden, B), (hidden, hidden, B), (hidden, 784, B)],
        }
    raise ValueError(f"Unsupported profile: {profile}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1.11 SM86 shape-aware iPC autotuner")
    p.add_argument("--kind", choices=["prediction", "inference", "weight", "all"], default="prediction")
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--activation", default="relu")
    p.add_argument("--profile", choices=["custom", "mnist64"], default="custom")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--max-configs", type=int, default=80)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", default="results/autotune_stage_1_11.json")
    p.add_argument("--temperature-max", type=float, default=78.0)
    p.add_argument("--reject-spill", action="store_true")
    p.add_argument("--max-registers", type=int, default=160)
    a = p.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("Expected an SM86 CUDA device")
    device = torch.device("cuda")
    guard = ThermalGuard(max_temp_c=a.temperature_max)

    if a.profile == "mnist64":
        shape_map = profile_shapes("mnist64", a.B, a.hidden)
        kinds = [a.kind] if a.kind != "all" else ["prediction", "inference", "weight"]
        shapes = [(k, s) for k in kinds for s in shape_map[k]]
    else:
        shapes = [(a.kind, (a.M, a.K, a.B))]
        if a.kind == "all":
            shapes = [
                ("prediction", (a.M, a.K, a.B)),
                ("inference", (a.M, a.K, a.B)),
                ("weight", (a.M, a.K, a.B)),
            ]

    candidate_pool = configs()
    selected = select_configs(candidate_pool, a.max_configs)
    print(f"Stage 1.11 candidate pool={len(candidate_pool)} selected={len(selected)}")

    output_rows: list[dict[str, object]] = []
    for shape_index, (kind, (M, K, B)) in enumerate(shapes, 1):
        rows: list[dict[str, object]] = []
        print(f"\\n[{shape_index}/{len(shapes)}] {kind} M={M} K={K} B={B}")
        for idx, cfg in enumerate(selected, 1):
            try:
                kernel, fn = _make_case(kind, M, K, B, device, a.activation, cfg)
                inspection = _inspect_kernel(kernel)
                if (
                    a.reject_spill
                    and inspection.get("spill_suspected") is True
                ):
                    print(f"SKIP {idx:03d}: suspected local-memory use: {cfg}")
                    continue
                regs = inspection.get("registers_estimate")
                if a.reject_spill and isinstance(regs, int) and regs > a.max_registers:
                    print(f"SKIP {idx:03d}: registers_estimate={regs} > {a.max_registers}: {cfg}")
                    continue
                ms = _time(fn, guard, a.warmup, a.rep)
                row = {
                    "index": idx,
                    "kind": kind,
                    "shape": {"M": M, "K": K, "B": B},
                    "activation": a.activation,
                    "dtype": "float16",
                    "median_ms": ms,
                    "config": cfg.__dict__,
                    **inspection,
                }
                rows.append(row)
                print(f"  {idx:03d}/{len(selected):03d} {ms:8.4f} ms {cfg}")
            except Exception as exc:
                print(f"SKIP {idx:03d}: {cfg}: {type(exc).__name__}: {exc}")

        rows.sort(key=lambda r: float(r["median_ms"]))
        if not rows:
            raise RuntimeError(f"No valid configurations for {kind} M={M} K={K} B={B}")
        output_rows.append(
            {
                "kind": kind,
                "shape": {"M": M, "K": K, "B": B},
                "activation": a.activation,
                "dtype": "float16",
                "best": rows[: a.topk],
                "tested": len(rows),
            }
        )
        best = rows[0]
        print(f"BEST {kind} M={M} K={K} B={B}: {float(best['median_ms']):.4f} ms {best['config']}")

    # One best entry per exact (kind, M, K, B, activation, dtype) key is what
    # the Stage 1.11 trainer consumes at runtime.
    entries = []
    for result in output_rows:
        best = result["best"][0]
        entries.append(
            {
                "kind": result["kind"],
                "shape": result["shape"],
                "activation": result["activation"],
                "dtype": result["dtype"],
                "median_ms": best["median_ms"],
                "registers_estimate": best.get("registers_estimate"),
                "spill_suspected": best.get("spill_suspected"),
                "config": best["config"],
            }
        )

    payload = {
        "schema_version": 1,
        "stage": "1.11",
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "search": {
            "candidate_pool": len(candidate_pool),
            "selected_per_shape": len(selected),
            "warmup": a.warmup,
            "rep": a.rep,
            "reject_spill": a.reject_spill,
            "max_registers": a.max_registers,
        },
        "results": output_rows,
        "entries": entries,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\\nWrote {out}")


if __name__ == "__main__":
    main()
