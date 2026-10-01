from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from ipc_tilelang.trainer import IPCConfig, TileLangIPC


def timed(fn, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2.4 steady-state direct vs CUDA-Graph benchmark")
    ap.add_argument("--dims", nargs="+", type=int, default=[784, 64, 64, 10])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", default="results/stage2_4_steady_state.json")
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(device)
    if cap != (8, 6):
        raise RuntimeError(f"Target is locked to SM86; got {cap}")

    torch.manual_seed(args.seed)
    x = torch.randn((args.dims[-1], args.batch), device=device, dtype=torch.float16)
    y = torch.zeros((args.dims[0], args.batch), device=device, dtype=torch.float16)
    labels = torch.randint(0, args.dims[0], (args.batch,), device=device)
    y.zero_()
    y.scatter_(0, labels.view(1, -1), 1.0)

    cfg = IPCConfig(
        dims=tuple(args.dims),
        activation="relu",
        dtype=torch.float16,
        hierarchical_policy_auto=not args.no_policy,
        steady_state_graph_auto=True,
        steady_state_graph_default=False,
        weight_update_mode="gemm",
    )
    model = TileLangIPC(cfg, device=device, seed=args.seed)
    graph_active = model.prepare_steady_state(x, y, steps=args.steps, use_graph=None)

    # Direct path uses the same prepared kernels and buffers as graph path.
    direct = timed(
        lambda: model.run_prepared_steady_state(
            steps=args.steps, repeats=1, use_graph=False
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # Re-capture after direct benchmarking so both modes begin from a valid graph state.
    graph_active = model.prepare_steady_state(x, y, steps=args.steps, use_graph=True)
    graph = timed(
        lambda: model.run_prepared_steady_state(
            steps=args.steps, repeats=1, use_graph=True
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # Values are the elapsed time for `steps` iPC updates; report per-step too.
    direct_med = statistics.median(direct)
    graph_med = statistics.median(graph)
    direct_step = direct_med / args.steps
    graph_step = graph_med / args.steps
    speedup = direct_step / graph_step if graph_step > 0 else float("inf")

    result = {
        "stage": "2.4",
        "device": torch.cuda.get_device_name(device),
        "capability": list(cap),
        "dims": list(args.dims),
        "batch": args.batch,
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "graph_active": bool(graph_active),
        "direct_ms_block_median": direct_med,
        "graph_ms_block_median": graph_med,
        "direct_ms_step": direct_step,
        "graph_ms_step": graph_step,
        "graph_speedup": speedup,
        "direct_samples": direct,
        "graph_samples": graph,
    }

    print("Stage 2.4 steady-state benchmark")
    print(f"device={result['device']} capability={tuple(cap)} dims={tuple(args.dims)} batch={args.batch}")
    print(f"direct={direct_step:.4f} ms/step")
    print(f"graph ={graph_step:.4f} ms/step")
    print(f"speedup={speedup:.3f}x")
    print(f"graph_active={graph_active}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
