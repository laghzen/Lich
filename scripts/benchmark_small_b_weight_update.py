from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc_tilelang.trainer import IPCConfig, TileLangIPC
from ipc_tilelang.weight_update_specialization import SmallBWeightUpdater


def measure_fixed(fn, *, warmup: int, repeats: int) -> list[float]:
    """Measure only the GPU update with CUDA events; no graph compiler involved."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        starter.record()
        fn()
        ender.record()
        ender.synchronize()
        samples.append(float(starter.elapsed_time(ender)))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark TileLang GEMM vs eager CUDA small-B outer-product on RTX 3060"
    )
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--out", default="results/weight_update_policy.json")
    ap.add_argument("--seed", type=int, default=9876)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(device)
    if cap != (8, 6):
        raise RuntimeError(f"Target is locked to SM86; got {cap}")
    if args.hidden <= 0:
        raise ValueError("--hidden must be positive")
    batches = sorted(set(int(x) for x in args.batch_sizes if int(x) > 0))
    if not batches:
        raise ValueError("at least one positive batch size is required")

    dims = (784, int(args.hidden), int(args.hidden), 10)
    rows: list[dict] = []
    updater = SmallBWeightUpdater(alpha=IPCConfig(dims=dims).alpha)

    print("Stage 2.5 small-B weight-update benchmark (no graph compiler)")
    print(f"device={torch.cuda.get_device_name(device)} capability={cap} dims={dims}")

    for B in batches:
        cfg = IPCConfig(
            dims=dims,
            activation="relu",
            dtype=torch.float16,
            recompute_activation=True,
            hierarchical_policy_auto=False,
            use_grid_z=False,
            grid_z_auto=False,
            weight_update_mode="gemm",
        )
        model = TileLangIPC(cfg, device=device, seed=args.seed + B)
        x_in = torch.randn((dims[-1], B), device=device, dtype=torch.float16)
        y = torch.zeros((dims[0], B), device=device, dtype=torch.float16)
        labels = torch.randint(0, dims[0], (B,), device=device)
        y.scatter_(0, labels.view(1, -1), 1.0)
        model.initialize_batch(x_in, y)

        for layer, (M, K) in enumerate(zip(dims[:-1], dims[1:])):
            gemm = model._weight_singleton_kernels[layer]
            if gemm is None:
                raise RuntimeError(f"missing GEMM weight kernel at layer={layer} B={B}")
            outer = updater.make_singleton(M=M, K=K, B=B, activation="relu", dtype=torch.float16)
            if outer is None:
                raise RuntimeError("small-B outer updater unexpectedly unavailable")

            g = torch.Generator(device=device)
            g.manual_seed(args.seed + 10000 * B + 101 * layer)
            w_base = torch.empty((M, K), device=device, dtype=torch.float16)
            e = torch.empty((M, B), device=device, dtype=torch.float16)
            x = torch.empty((K, B), device=device, dtype=torch.float16)
            torch.nn.init.uniform_(w_base, -0.05, 0.05, generator=g)
            torch.nn.init.uniform_(e, -0.05, 0.05, generator=g)
            torch.nn.init.uniform_(x, -0.05, 0.05, generator=g)

            w_g = w_base.clone()
            w_o = w_base.clone()
            ref = w_base.float() - cfg.alpha * (e.float() @ torch.relu(x.float()).transpose(0, 1))
            gemm(w_g, e, x)
            torch.cuda.synchronize()
            gemm_err = float((w_g.float() - ref).abs().max().item())
            outer(w_o, e, x)
            torch.cuda.synchronize()
            outer_err = float((w_o.float() - ref).abs().max().item())

            w_g = w_base.clone()
            gemm_samples = measure_fixed(lambda: gemm(w_g, e, x), warmup=args.warmup, repeats=args.repeats)
            gemm_med = statistics.median(gemm_samples)

            w_o = w_base.clone()
            outer_samples = measure_fixed(lambda: outer(w_o, e, x), warmup=args.warmup, repeats=args.repeats)
            outer_med = statistics.median(outer_samples)
            speedup = gemm_med / outer_med if outer_med > 0 else 0.0
            selected = (
                "outer"
                if outer_med > 0.0
                and outer_med <= gemm_med * (1.0 - float(args.margin))
                and outer_err <= 2.5e-2
                else "gemm"
            )

            row = {
                "M": M, "K": K, "B": B, "count": 1,
                "activation": "relu", "dtype": "float16",
                "backend": selected,
                "gemm_ms": gemm_med,
                "outer_ms": outer_med,
                "outer_speedup": speedup,
                "gemm_max_abs_error": gemm_err,
                "outer_max_abs_error": outer_err,
                "outer_impl": "eager_cuda_bmm_no_graph_compiler",
            }
            rows.append(row)
            print(
                f"shape={M}x{K} B={B}: gemm={gemm_med:.5f} ms "
                f"outer={outer_med:.5f} ms speedup={speedup:.3f}x selected={selected} "
                f"outer_err={outer_err:.3e}"
            )

    result = {
        "version": 4,
        "stage": "2.5-small-b",
        "device": torch.cuda.get_device_name(device),
        "capability": list(cap),
        "dims": list(dims),
        "alpha": IPCConfig(dims=dims).alpha,
        "selection_margin": float(args.margin),
        "outer_impl": "eager_cuda_bmm_no_graph_compiler",
        "entries": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
