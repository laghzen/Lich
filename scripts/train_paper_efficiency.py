from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.trainer import IPCConfig, TileLangIPC, prepare_batch
from ipc_tilelang.tilelang_kernels import KernelConfig
from ipc_tilelang.thermal import ThermalGuard


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-style full-width iPC efficiency profile")
    p.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, choices=[3, 4, 6], default=None)
    p.add_argument("--depths", type=int, nargs="+", choices=[3, 4, 6], default=None)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--subset", type=int, default=250)
    p.add_argument("--out", default="results/paper_efficiency.jsonl")
    p.add_argument("--temperature-max", type=float, default=76.0)
    p.add_argument("--thermal-poll", type=float, default=0.5)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--save-activation", action="store_true")
    a = p.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("Expected SM86 CUDA device")
    depths = a.depths if a.depths is not None else ([a.depth] if a.depth is not None else [4])

    dev = torch.device("cuda")
    tfm = transforms.Compose([transforms.ToTensor()])
    ds_cls = datasets.MNIST if a.dataset == "MNIST" else datasets.FashionMNIST
    ds = ds_cls("./data", train=True, download=True, transform=tfm)
    ds = Subset(ds, range(min(a.subset, len(ds))))
    dl = DataLoader(
        ds, batch_size=a.batch, shuffle=False, drop_last=True, pin_memory=True,
        num_workers=2, persistent_workers=True,
    )

    guard = ThermalGuard(max_temp_c=a.temperature_max, poll_s=a.thermal_poll)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open("a", encoding="utf-8") as f:
        for depth in depths:
            dims = (10,) + (a.hidden,) * depth + (784,)
            model = TileLangIPC(
                IPCConfig(
                    dims=dims,
                    alpha=1e-4,
                    gamma=0.5,
                    activation="relu",
                    dtype=torch.float16,
                    kernel=KernelConfig(64, 128, 16, 128, 2, True, 8, False),
                    recompute_activation=not a.save_activation,
                ),
                device=dev,
            )
            graph_runner = None
            for epoch in range(a.epochs):
                t0 = time.perf_counter()
                examples = 0
                for images, labels in dl:
                    guard.wait_until_safe(verbose=True)
                    x, y = prepare_batch(images, labels, device=dev, dtype=torch.float16, num_classes=10)
                    model.initialize_batch(x, y)
                    if a.graph:
                        if graph_runner is None:
                            graph_runner = model.capture_graph(a.steps)
                        else:
                            graph_runner.replay()
                    else:
                        for _ in range(a.steps):
                            model.step_initialized(collect_metrics=False)
                    examples += labels.numel()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                t = guard.check(force=True)
                row = {
                    "dataset": a.dataset, "depth": depth, "hidden": a.hidden,
                    "batch": a.batch, "epoch": epoch + 1, "steps": a.steps,
                    "seconds": dt, "samples_per_s": examples / max(dt, 1e-12),
                    "temperature_c": t.temperature_c, "power_w": t.power_w,
                    "sm_clock_mhz": t.clock_sm_mhz, "throttle_reasons": t.throttle_reasons,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row)


if __name__ == "__main__":
    main()
