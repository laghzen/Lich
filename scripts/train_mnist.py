from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=2, help="number of hidden layers")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--activation", default="relu")
    p.add_argument("--subset", type=int, default=0, help="0=full train set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature-max", type=float, default=76.0)
    p.add_argument("--thermal-poll", type=float, default=0.5)
    p.add_argument("--graph", action="store_true", help="capture the fixed iPC step loop in a CUDA Graph")
    p.add_argument("--save-activation", action="store_true", help="materialize f(x) for weight updates instead of recomputing it")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (8, 6):
        raise RuntimeError("This project is intentionally specialized for SM86 / RTX 3060.")
    tfm = transforms.Compose([transforms.ToTensor()])
    cls = datasets.MNIST if a.dataset == "MNIST" else datasets.FashionMNIST
    ds = cls(a.data, train=True, download=True, transform=tfm)
    if a.subset:
        ds = Subset(ds, range(min(a.subset, len(ds))))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)

    dims = (10,) + (a.hidden,) * a.depth + (784,)
    model = TileLangIPC(IPCConfig(
        dims=dims, alpha=a.alpha, gamma=a.gamma, activation=a.activation,
        dtype=torch.float16,
        kernel=KernelConfig(block_m=64, block_n=128, block_k=16, threads=128, num_stages=2, swizzle=True),
        recompute_activation=not a.save_activation,
    ), device=device, seed=a.seed)
    guard = ThermalGuard(max_temp_c=a.temperature_max, poll_s=a.thermal_poll)

    graph_runner = None
    for epoch in range(a.epochs):
        t0 = time.perf_counter(); total = 0; correct = 0
        for images, labels in dl:
            guard.wait_until_safe(verbose=True)
            x, y = prepare_batch(images, labels, device=device, dtype=torch.float16, num_classes=10)
            model.initialize_batch(x, y)
            if a.graph:
                # Capture once per fixed batch shape; initialize_batch reuses the
                # same state/error allocations for all subsequent replays.
                if graph_runner is None:
                    graph_runner = model.capture_graph(a.steps)
                else:
                    graph_runner.replay()
            else:
                model.step_initialized(collect_metrics=False)
                for _ in range(a.steps - 1):
                    model.step_initialized(collect_metrics=False)
            with torch.no_grad():
                logits = model.predict(x).T.float()[:, :10]
                pred = logits.argmax(dim=1)
                correct += int((pred == labels.to(device)).sum().item())
            total += labels.numel()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"epoch={epoch+1:03d} time={dt:.3f}s train_acc={correct/max(total,1):.4f} samples/s={total/dt:.1f}")


if __name__ == "__main__":
    main()
