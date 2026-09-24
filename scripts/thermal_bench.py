from __future__ import annotations

import argparse
import time
import torch

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.trainer import IPCConfig, TileLangIPC, prepare_batch
from ipc_tilelang.tilelang_kernels import KernelConfig
from ipc_tilelang.thermal import ThermalGuard


def main() -> None:
    p = argparse.ArgumentParser(description="Sustained SM86 iPC thermal/performance test")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--temperature-max", type=float, default=76.0)
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--save-activation", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("This benchmark is pinned to SM86 / RTX 3060")

    device = torch.device("cuda")
    dims = (10,) + (args.hidden,) * args.depth + (784,)
    model = TileLangIPC(
        IPCConfig(
            dims=dims,
            alpha=1e-4,
            gamma=0.5,
            activation="relu",
            dtype=torch.float16,
            kernel=KernelConfig(64, 128, 16, 128, 2, True, 8, False),
            recompute_activation=not args.save_activation,
        ),
        device=device,
    )

    labels = torch.arange(args.batch, device="cpu", dtype=torch.long).remainder(10)
    images = torch.rand((args.batch, 1, 28, 28), device="cpu")
    x, y = prepare_batch(images, labels, device=device, dtype=torch.float16, num_classes=10)
    model.initialize_batch(x, y)
    graph = model.capture_graph(args.steps) if args.graph else None

    guard = ThermalGuard(max_temp_c=args.temperature_max, poll_s=args.poll, telemetry_interval_s=1.0)
    guard.wait_until_safe(verbose=True)

    start = time.perf_counter()
    last_report = start
    iterations = 0
    samples = 0
    while True:
        now = time.perf_counter()
        if now - start >= args.seconds:
            break
        guard.wait_until_safe(verbose=True)
        if graph is not None:
            graph.replay()
        else:
            for _ in range(args.steps):
                model.step_initialized(collect_metrics=False)
        iterations += 1
        samples += args.batch
        now = time.perf_counter()
        if now - last_report >= 1.0:
            torch.cuda.synchronize()
            elapsed = now - start
            t = guard.check(force=True)
            print(f"t={elapsed:7.1f}s iterations={iterations:7d} samples/s={samples/max(elapsed,1e-9):9.1f} {guard.format_status(t)}")
            last_report = now

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    t = guard.check(force=True)
    print(f"FINAL elapsed={elapsed:.3f}s iterations={iterations} samples/s={samples/max(elapsed,1e-9):.1f} {guard.format_status(t)}")


if __name__ == "__main__":
    main()
